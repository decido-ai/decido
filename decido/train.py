"""Verdict LoRA training on the verbalizer readout (soft-target CE + Brier + RPS), multi-GPU via torchrun (FSDP).

    torchrun --nproc_per_node 2 -m verdict.train --model google/gemma-4-26B-A4B-it --train data/train.jsonl \
        --val data/val.jsonl --out runs/train/r1 --rank 32 --lr 1e-4 --epochs 1

Each training example is one (state, question) pair rendered exactly like inference (verdict.template):
tok(prefix) + tok(suffix). The loss reads the logits of the label tokens at the readout position only
(label score = logsumexp over the bare/space variants), so the trained model is used unchanged by the
vLLM, HF and llama.cpp readouts. Choice questions with K > 26 options use two-symbol labels: the first
step is trained on the per-letter aggregated target, the second step (digit) on the conditional target
of the highest-mass letter, read from the same sequence with that letter appended.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from decido.template import DIGITS, ChatFormat, LabelTrie, question_block

HF_CACHE = os.environ.get("HF_HUB_CACHE")  # None = default HF cache


# ----------------------------------------------------------------------------- data

def read_jsonl(p):
    with open(p) as fh:
        return [json.loads(x) for x in fh if x.strip()]


class Encoder:
    def __init__(self, tok, cf: ChatFormat, max_len: int):
        self.tok, self.cf, self.max_len = tok, cf, max_len
        self.tries = {}

    def trie(self, labels):
        k = tuple(labels)
        if k not in self.tries:
            self.tries[k] = LabelTrie(self.tok, labels)
        return self.tries[k]

    def example(self, state, q, target, rng: random.Random, perm_p: float):
        target = np.asarray(target, dtype=np.float64)
        target = target / target.sum()
        order = None
        if q["type"] == "choice" and rng.random() < perm_p:
            order = list(range(len(q["criteria"])))
            rng.shuffle(order)
        text, labels, order = question_block(q, order)
        tdisp = target[np.array(order)]                     # target in display order
        suffix = self.cf.encode(self.cf.suffix_text(text))
        prefix = self.cf.encode(self.cf.prefix_text(state))
        tr = self.trie(labels)
        ext = None
        if tr.two_step:
            letters = tr.first_syms
            agg = np.array([tdisp[[i for i, l in enumerate(labels) if l[0] == s]].sum() for s in letters])
            best = letters[int(np.argmax(agg))]
            members = [i for i, l in enumerate(labels) if l[0] == best]
            cond = tdisp[members] / max(tdisp[members].sum(), 1e-9)
            ext = {"letter_tok": tr.first_canon[best],
                   "digit_variants": [tr.second_variants[labels[i][1]] for i in members],
                   "target2": cond.tolist()}
            step1_variants = [tr.first_variants[s] for s in letters]
            step1_target = agg.tolist()
        else:
            step1_variants = [tr.first_variants[l] for l in labels]
            step1_target = tdisp.tolist()
        budget = self.max_len - len(suffix) - (1 if ext else 0)
        if len(prefix) > budget:
            closing = self.cf.encode("\n</document>\n\n")
            prefix = prefix[:max(budget - len(closing), 16)] + closing
        ids = prefix + suffix + ([ext["letter_tok"]] if ext else [])
        return {"ids": ids, "pos1": len(prefix) + len(suffix) - 1, "v1": step1_variants, "t1": step1_target,
                "ext": ext, "kind": q["type"], "score_target": tdisp.tolist() if q["type"] == "score" else None}


def expand_rows(rows):
    out = []
    for r in rows:
        for qid, q in r["questions"].items():
            if qid in r.get("targets", {}):
                out.append((r["state"], q, r["targets"][qid], r.get("meta", {}).get("weight", 1.0)))
    return out


def token_batches(lengths, max_tokens, rng):
    """Length-bucketed batches with <= max_tokens padded tokens each (shuffled)."""
    idx = list(range(len(lengths)))
    rng.shuffle(idx)
    chunks = [sorted(idx[i:i + 4096], key=lambda j: lengths[j]) for i in range(0, len(idx), 4096)]
    batches = []
    for ch in chunks:
        cur, mx = [], 0
        for j in ch:
            m2 = max(mx, lengths[j])
            if cur and m2 * (len(cur) + 1) > max_tokens:
                batches.append(cur)
                cur, m2 = [], lengths[j]
            cur.append(j)
            mx = m2
        if cur:
            batches.append(cur)
    rng.shuffle(batches)
    return batches


# ----------------------------------------------------------------------------- loss

def label_scores(hidden_at, lm_w, variants, softcap):
    """hidden_at: [H]; variants: list (per label) of token-id lists -> [K] label scores."""
    flat = sorted({t for v in variants for t in v})
    pos = {t: i for i, t in enumerate(flat)}
    lg = hidden_at.float() @ lm_w[flat].float().T
    if softcap:
        lg = torch.tanh(lg / softcap) * softcap
    return torch.stack([torch.logsumexp(lg[[pos[t] for t in v]], 0) for v in variants])


def dist_loss(scores, target, kind, w_brier, w_rps):
    t = torch.tensor(target, device=scores.device, dtype=torch.float32)
    logp = torch.log_softmax(scores, -1)
    p = logp.exp()
    ce = -(t * logp).sum()
    brier = ((p - t) ** 2).sum() / 2
    loss = ce + w_brier * brier
    rps = torch.zeros((), device=scores.device)
    if kind == "score" and len(t) > 1:
        rps = ((torch.cumsum(p, 0) - torch.cumsum(t, 0)) ** 2)[:-1].sum() / (len(t) - 1)
        loss = loss + w_rps * rps
    return loss, ce.detach(), brier.detach(), rps.detach(), float(int(p.argmax()) == int(t.argmax()))


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--targets", default="attn", help="attn | attn+experts | attn+mlp | comma list of module names")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--max-len", type=int, default=3072)
    ap.add_argument("--max-tokens", type=int, default=16384, help="padded tokens per micro-batch per GPU")
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--perm-p", type=float, default=0.3)
    ap.add_argument("--w-brier", type=float, default=0.5)
    ap.add_argument("--w-rps", type=float, default=1.0)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--val-n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--experts-impl", default="grouped_mm")
    ap.add_argument("--fsdp", type=int, default=0, help="1 = FSDP2 sharding (untested); 0 = replica per GPU + grad all-reduce")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--init-adapter", default=None, help="warm start: load LoRA weights from a saved adapter dir")
    args = ap.parse_args()

    ddp = "RANK" in os.environ
    if ddp:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    else:
        rank, world = 0, 1
    dev = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(args.seed + rank)
    os.makedirs(args.out, exist_ok=True)
    log = open(os.path.join(args.out, "log.jsonl"), "a") if rank == 0 else None

    def P(*a):
        if rank == 0:
            print(*a, flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, cache_dir=HF_CACHE)
    cf = ChatFormat(tok)
    enc = Encoder(tok, cf, args.max_len)

    rows = read_jsonl(args.train)
    exs = expand_rows(rows)
    if args.limit:
        exs = exs[:args.limit]
    rng = random.Random(args.seed)
    P(f"train questions: {len(exs)}")
    val = expand_rows(read_jsonl(args.val))[:args.val_n] if args.val else []
    # pre-tokenize lengths (one pass; permutation does not change length materially)
    t0 = time.time()
    lengths = [len(enc.example(s, q, t, rng, 0.0)["ids"]) for s, q, t, _ in exs]
    P(f"tokenized lengths in {time.time() - t0:.0f}s; p50={int(np.median(lengths))} p95={int(np.percentile(lengths, 95))} "
      f"tokens={sum(lengths)}")

    kw = dict(cache_dir=HF_CACHE, dtype=torch.bfloat16, attn_implementation="sdpa")
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, experts_implementation=args.experts_impl, **kw)
    except (TypeError, ValueError):
        model = AutoModelForCausalLM.from_pretrained(args.model, **kw)
    model.config.use_cache = False
    softcap = getattr(model.config, "final_logit_softcapping", None) or \
        getattr(getattr(model.config, "text_config", None), "final_logit_softcapping", None)
    from peft import LoraConfig, get_peft_model
    names = {n.split(".")[-1] for n, _ in model.named_modules()}
    attn = [n for n in ("q_proj", "k_proj", "v_proj", "o_proj") if n in names]
    tm, tp = list(attn), []
    if "mlp" in args.targets:
        tm += [n for n in ("gate_proj", "up_proj", "down_proj") if n in names]
    if "experts" in args.targets:
        tp = sorted({n.split(".", n.count(".") - 1)[-1] for n, _ in model.named_parameters()
                     if n.endswith(("experts.gate_up_proj", "experts.down_proj"))})
    if "router" in args.targets:   # exact paths: a bare "proj" suffix would also match vision/audio modules
        tm += [n for n, _ in model.named_modules() if n.endswith("router.proj") and "vision" not in n and "audio" not in n]
    if not set(args.targets.split("+")) <= {"attn", "experts", "mlp", "router"}:
        tm = args.targets.split(",")
    lcfg = LoraConfig(r=args.rank, lora_alpha=args.alpha or 2 * args.rank, lora_dropout=0.0, target_modules=tm,
                      target_parameters=tp or None, bias="none",
                      exclude_modules=r".*(vision|audio|visual|multi_modal).*")
    if args.init_adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
        P(f"warm start from {args.init_adapter}")
    else:
        model = get_peft_model(model, lcfg)
    if rank == 0:
        model.print_trainable_parameters()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    lm_head = model.get_output_embeddings()
    base = model.get_base_model()
    inner = base.model if hasattr(base, "model") else base

    if ddp and args.fsdp:
        from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
        mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
        layers = None
        for n, m in model.named_modules():
            if n.endswith("layers") and isinstance(m, torch.nn.ModuleList) and "vision" not in n and "audio" not in n:
                layers = m
                break
        for blk in layers:
            fully_shard(blk, mp_policy=mp)
        fully_shard(model, mp_policy=mp)
    model.to(dev)
    params = [p for p in model.parameters() if p.requires_grad]
    if rank == 0:
        P(f"trainable tensors {len(params)}; dtype {params[0].dtype}; cuda mem after load {torch.cuda.memory_allocated() / 1e9:.1f} GB")
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.99))

    batches = token_batches(lengths, args.max_tokens, random.Random(args.seed))
    n_ep = args.epochs
    sched_batches = []
    e = 0
    while e < n_ep:
        bs = token_batches(lengths, args.max_tokens, random.Random(args.seed + int(e * 1000)))
        take = bs if n_ep - e >= 1 else bs[:int(len(bs) * (n_ep - e))]
        sched_batches += take
        e += 1
    sched_batches = sched_batches[:len(sched_batches) - len(sched_batches) % world]
    my_batches = sched_batches[rank::world]
    total_steps = len(my_batches) // args.grad_accum
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    warm = max(1, int(total_steps * args.warmup))
    P(f"micro-batches/rank={len(my_batches)} optimizer steps={total_steps}")

    def lr_at(s):
        if s < warm:
            return args.lr * (s + 1) / warm
        return args.lr * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total_steps - warm)))

    def forward_batch(items):
        L = max(len(x["ids"]) for x in items)
        pad = tok.pad_token_id if tok.pad_token_id is not None else 0
        ids = torch.full((len(items), L), pad, dtype=torch.long)
        att = torch.zeros((len(items), L), dtype=torch.long)
        for i, x in enumerate(items):
            ids[i, :len(x["ids"])] = torch.tensor(x["ids"])
            att[i, :len(x["ids"])] = 1
        out = inner(input_ids=ids.to(dev), attention_mask=att.to(dev), use_cache=False)
        h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        W = lm_head.weight
        if hasattr(W, "full_tensor"):
            W = W.full_tensor()
        losses, stats = [], []
        for i, x in enumerate(items):
            s1 = label_scores(h[i, x["pos1"]], W, x["v1"], softcap)
            l1, ce, br, rps, acc = dist_loss(s1, x["t1"], x["kind"], args.w_brier, args.w_rps)
            loss = l1
            if x["ext"] is not None:
                s2 = label_scores(h[i, x["pos1"] + 1], W, x["ext"]["digit_variants"], softcap)
                l2, *_ = dist_loss(s2, x["ext"]["target2"], "choice", args.w_brier, 0.0)
                loss = loss + l2
            losses.append(loss)
            stats.append((float(ce), float(br), float(rps), acc))
        return torch.stack(losses).mean(), stats

    def evaluate():
        model.eval()
        tot = {"ce": 0.0, "brier": 0.0, "acc": 0.0, "n": 0}
        vr = random.Random(123)
        mine = val[rank::world]
        with torch.no_grad():
            for i in range(0, len(mine), 8):
                items = [enc.example(s, q, t, vr, 0.0) for s, q, t, _ in mine[i:i + 8]]
                _, st = forward_batch(items)
                for ce, br, _, acc in st:
                    tot["ce"] += ce
                    tot["brier"] += br
                    tot["acc"] += acc
                    tot["n"] += 1
        if ddp:
            t = torch.tensor([tot["ce"], tot["brier"], tot["acc"], tot["n"]], device=dev, dtype=torch.float64)
            dist.all_reduce(t)
            tot = dict(zip(["ce", "brier", "acc", "n"], t.tolist()))
        model.train()
        n = max(tot["n"], 1)
        return {"val_ce": tot["ce"] / n, "val_brier": tot["brier"] / n, "val_acc": tot["acc"] / n, "val_n": int(n)}

    def save(tag):
        d = os.path.join(args.out, tag)
        if not args.fsdp:
            if rank == 0:
                model.save_pretrained(d)
                json.dump(vars(args), open(os.path.join(d, "train_args.json"), "w"), indent=1)
            if ddp:
                dist.barrier()
            return
        from torch.distributed.checkpoint.state_dict import get_model_state_dict, StateDictOptions
        sd = get_model_state_dict(model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
        if rank == 0:
            os.makedirs(d, exist_ok=True)
            from peft import get_peft_model_state_dict
            from safetensors.torch import save_file
            lora_sd = get_peft_model_state_dict(model, state_dict=sd)
            save_file({k: v.contiguous() for k, v in lora_sd.items()}, os.path.join(d, "adapter_model.safetensors"))
            model.peft_config["default"].save_pretrained(d)
            json.dump(vars(args), open(os.path.join(d, "train_args.json"), "w"), indent=1)
        if ddp:
            dist.barrier()

    model.train()
    step, t_start = 0, time.time()
    run = {"loss": 0.0, "ce": 0.0, "acc": 0.0, "n": 0}
    if val:
        ev = evaluate()
        P(json.dumps({"step": 0, **ev}))
        if log:
            log.write(json.dumps({"step": 0, **ev}) + "\n")
            log.flush()
    for bi in range(total_steps * args.grad_accum):
        b = my_batches[bi]
        items = [enc.example(exs[j][0], exs[j][1], exs[j][2], rng, args.perm_p) for j in b]
        loss, st = forward_batch(items)
        (loss / args.grad_accum).backward()
        run["loss"] += float(loss.detach())
        run["ce"] += sum(s[0] for s in st) / len(st)
        run["acc"] += sum(s[3] for s in st) / len(st)
        run["n"] += 1
        if (bi + 1) % args.grad_accum:
            continue
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        if ddp and not args.fsdp:   # data parallel over replicas: average LoRA grads across ranks
            flat = [p.grad for p in params if p.grad is not None]
            for g in flat:
                dist.all_reduce(g, op=dist.ReduceOp.AVG)
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        if step % 20 == 0:
            rec = {"step": step, "max_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1), "loss": run["loss"] / run["n"], "ce": run["ce"] / run["n"], "acc": run["acc"] / run["n"],
                   "lr": lr_at(step), "gnorm": float(gn), "elapsed_s": round(time.time() - t_start),
                   "eta_s": round((time.time() - t_start) / step * (total_steps - step))}
            P(json.dumps(rec))
            if log:
                log.write(json.dumps(rec) + "\n")
                log.flush()
            run = {"loss": 0.0, "ce": 0.0, "acc": 0.0, "n": 0}
        if val and step % args.eval_every == 0:
            ev = evaluate()
            P(json.dumps({"step": step, **ev}))
            if log:
                log.write(json.dumps({"step": step, **ev}) + "\n")
                log.flush()
        if step % args.save_every == 0:
            save(f"step{step}")
    save("final")
    if val:
        ev = evaluate()
        P(json.dumps({"step": step, "final": True, **ev}))
        if log:
            log.write(json.dumps({"step": step, "final": True, **ev}) + "\n")
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
