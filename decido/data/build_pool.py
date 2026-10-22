"""Build the raw Verdict training pool (unified schema, research/data.md §3.8) from TRAIN splits only.

    python -m decido.data.build_pool [--only src1 src2 ...] [--seed 0]

Writes data/pool/<source>.jsonl. Every row:
  {"request_id", "state", "questions": {qid: {type, instructions, criteria}}, "targets": {qid: [...]},
   "meta": {source, task_id, family, label_type, license, license_use, split, group_id, generator, ...}}
Targets are aligned with eval.parity.common.option_keys(q): noul -> [p_false, p_true]; score -> per level;
choice -> criteria key order. Hard labels are one-hot; human vote shares / exact posteriors are kept soft.

Never reads an eval split: every loader below opens a train file explicitly. The leakage gate
(python -m eval.parity.leakage check) runs afterwards on everything written here.
"""

from __future__ import annotations

import argparse
import collections
import glob
import gzip
import hashlib
import json
import math
import os
import random
import re

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from eval.parity.common import NLI_CRITERIA, NLI_INSTRUCTIONS

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "data", "src")
POOL = os.path.join(ROOT, "data", "pool")
NLI_KEYS = ["entailment", "neutral", "contradiction"]


# ----------------------------------------------------------------------------- helpers

def rid(*parts) -> str:
    return hashlib.sha1("/".join(map(str, parts)).encode()).hexdigest()[:20]


def onehot(n, i):
    v = [0.0] * n
    v[i] = 1.0
    return v


def read_parquets(pattern, columns=None):
    fs = sorted(glob.glob(os.path.join(SRC, pattern)))
    if not fs:
        raise FileNotFoundError(pattern)
    return pd.concat([pq.read_table(f, columns=columns).to_pandas() for f in fs], ignore_index=True)


def read_jsonl(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def meta(source, task_id, family, label_type, license, license_use, group_id, generator="human/source", **kw):
    m = {"source": source, "task_id": task_id, "family": family, "label_type": label_type, "license": license,
         "license_use": license_use, "split": "train", "group_id": group_id, "generator": generator}
    m.update(kw)
    return m


def choice_q(instr, options, descs=None):
    crit = {}
    for i, o in enumerate(options):
        crit[o] = None if descs is None else descs[i]
    return {"type": "choice", "instructions": instr, "criteria": crit}


def nli_row(source, gid, premise, hypothesis, label_or_dist, lic, use, family="nli", task_id=None, **kw):
    if isinstance(label_or_dist, str):
        tgt = onehot(3, NLI_KEYS.index(label_or_dist))
        lt = "hard"
    else:
        tgt = list(map(float, label_or_dist))
        lt = "soft_human"
    return {"request_id": rid(source, gid, hypothesis), "state": {"premise": premise, "hypothesis": hypothesis},
            "questions": {"relation": {"type": "choice", "instructions": NLI_INSTRUCTIONS, "criteria": dict(NLI_CRITERIA)}},
            "targets": {"relation": tgt},
            "meta": meta(source, task_id or f"{source}/nli", family, lt, lic, use, gid, **kw)}


def write(name, rows):
    os.makedirs(POOL, exist_ok=True)
    p = os.path.join(POOL, f"{name}.jsonl")
    n_q = collections.Counter()
    with open(p, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            for q in r["questions"].values():
                n_q[q["type"]] += 1
    print(f"[{name}] rows={len(rows)} questions={dict(n_q)}", flush=True)
    return len(rows)


def valid_row(r):
    for qid, q in r["questions"].items():
        t = r["targets"].get(qid)
        if t is None:
            return False
        n = 2 if q["type"] == "noul" else len(q["criteria"])
        if len(t) != n or n < 2 or not np.isfinite(t).all() or abs(sum(t) - 1) > 1e-3 or min(t) < 0:
            return False
        if q["type"] == "score" and n > 10:
            return False
        if q["type"] == "choice" and n > 260:
            return False
    return bool(r["questions"])


# ----------------------------------------------------------------------------- tasksource jev-typed

TS_EXCLUDE_PREFIX = ("procedural-typed-decisions/", "multilingual/")


def _arr(x):
    return list(x) if x is not None else []


def tasksource_typed(rng, budget=130000, cap=2000, score_budget=40000):
    cols = ["state", "kind", "id", "options", "target", "question", "source", "variant", "split", "group_id",
            "question_id", "license", "license_use", "source_index"]
    t = read_parquets("tasksource__tasksource-jev-typed-decisions/data/train-*", cols)
    t = t[(t.license_use == "commercial") & (t.split == "train")]
    t = t[~t.source.str.startswith(TS_EXCLUDE_PREFIX)]
    t = t[~t.source_index.astype(str).str.startswith(("test", "valid", "dev"))]
    keep_idx = []
    for sub, bud in ((t[t.kind != "score"], budget), (t[t.kind == "score"], score_budget)):
        counts = sub.source.value_counts()
        w = counts ** 0.5
        alloc = (w / w.sum() * bud).clip(upper=cap)
        alloc = np.minimum(alloc.round().astype(int), counts)
        for src, n in alloc.items():
            idx = sub.index[sub.source == src].tolist()
            rng.shuffle(idx)
            keep_idx.extend(idx[:n])
    t = t.loc[keep_idx]
    groups = collections.OrderedDict()
    for r in t.itertuples(index=False):
        groups.setdefault((r.group_id, r.state), []).append(r)
    rows = []
    for (gid, state), rs in groups.items():
        qs, tg = {}, {}
        for j, r in enumerate(rs):
            opts = [str(o) for o in _arr(r.options)]
            tgt = [float(x) for x in _arr(r.target)]
            qid = f"q{j}" if len(rs) > 1 else "decision"
            if r.kind == "noul":
                if len(tgt) == 1:
                    tgt = [1 - tgt[0], tgt[0]]
                elif len(tgt) != 2:
                    continue
                qs[qid] = {"type": "noul", "instructions": r.question}
            elif r.kind == "choice":
                if len(set(opts)) != len(opts) or len(opts) != len(tgt):
                    continue
                qs[qid] = choice_q(r.question, opts)
            elif r.kind == "score":
                if len(opts) != len(tgt) or len(opts) > 10:
                    continue
                qs[qid] = {"type": "score", "instructions": r.question, "criteria": opts}
            else:
                continue
            s = sum(tgt)
            if s <= 0:
                del qs[qid]
                continue
            tg[qid] = [x / s for x in tgt]
        if not qs:
            continue
        r0 = rs[0]
        soft = any(0 < x < 1 for v in tg.values() for x in v)
        row = {"request_id": rid("ts", gid, state[:200]), "state": state, "questions": qs, "targets": tg,
               "meta": meta("tasksource-jev-typed", f"ts/{r0.source}", "tasksource", "soft_human" if soft else "hard",
                            r0.license, "commercial", gid, variant=r0.variant)}
        if valid_row(row):
            rows.append(row)
    return rows


# ----------------------------------------------------------------------------- procedural-typed (tasksource)

def procedural_typed(rng, per_config=2000):
    rows = []
    for d in sorted(glob.glob(os.path.join(SRC, "tasksource__procedural-typed-decisions", "*"))):
        cfg = os.path.basename(d)
        if cfg == "all" or not os.path.isdir(d):
            continue
        t = read_parquets(f"tasksource__procedural-typed-decisions/{cfg}/train-*")
        idx = list(range(len(t)))
        rng.shuffle(idx)
        for i in idx[:per_config]:
            r = t.iloc[i]
            qs = json.loads(r["questions"])
            ans = json.loads(r["answers"])
            tg, qq = {}, {}
            for qid, q in qs.items():
                a = ans.get(qid)
                if a is None:
                    continue
                if q["type"] == "noul":
                    p = float(a["noul"])
                    tg[qid] = [1 - p, p]
                elif q["type"] == "choice":
                    keys = list(q["criteria"].keys())
                    tg[qid] = [float(a["probabilities"][k]) for k in keys]
                elif q["type"] == "score":
                    tg[qid] = [float(a["probabilities"][str(k)]) for k in range(len(q["criteria"]))]
                else:
                    continue
                qq[qid] = q
            soft = any(0 < x < 1 for v in tg.values() for x in v)
            row = {"request_id": rid("proc", r["id"]), "state": r["state"], "questions": qq, "targets": tg,
                   "meta": meta("procedural-typed", f"proc/{cfg}", "procedural", "exact", "apache-2.0", "commercial",
                                r["id"], generator="rule", level=str(r["level"]), soft=soft)}
            if valid_row(row):
                rows.append(row)
    return rows


# ----------------------------------------------------------------------------- Open-Jev (CC0) train splits

def _split_opt(o):
    if ": " in o:
        k, d = o.split(": ", 1)
        return k.strip(), d.strip()
    return o.strip(), None


def open_jev(rng, cap_per_source=2500):
    rows = []
    specs = [("ZefanCai__Open-Jev/data/*/train-*", "open-jev", "cc0-1.0"),
             ("ZefanCai__Open-Jev-v1.1/data/*/train-*", "open-jev-v1.1", "cc0-1.0 / cc-by-4.0 (WANLI)")]
    seen = set()
    for pat, name, lic in specs:
        t = read_parquets(pat)
        t = t[t.split == "train"]
        t = t[~t.source.astype(str).str.contains("customer-control|wanli|WANLI")]  # licence caveat; WANLI loaded directly
        by_src = collections.defaultdict(list)
        for i, r in enumerate(t.itertuples(index=False)):
            by_src[r.source].append(r)
        for src, rs in by_src.items():
            groups = collections.OrderedDict()
            for r in rs:
                groups.setdefault((r.group_id, r.state_json), []).append(r)
            keys = list(groups)
            rng.shuffle(keys)
            n_rows = 0
            for gk in keys:
                if n_rows >= cap_per_source:
                    break
                qs, tg = {}, {}
                for j, r in enumerate(groups[gk]):
                    if r.id in seen:
                        continue
                    seen.add(r.id)
                    opts = [str(o) for o in _arr(r.options)]
                    tgt = [float(x) for x in _arr(r.target)]
                    qid = f"q{j}"
                    if r.kind == "noul":
                        if len(tgt) == 1:
                            tgt = [1 - tgt[0], tgt[0]]
                        qs[qid] = {"type": "noul", "instructions": r.question}
                    elif r.kind == "choice":
                        kd = [_split_opt(o) for o in opts]
                        ks = [k for k, _ in kd]
                        if len(set(ks)) != len(ks):
                            continue
                        qs[qid] = choice_q(r.question, ks, [d for _, d in kd])
                    elif r.kind == "score":
                        qs[qid] = {"type": "score", "instructions": r.question, "criteria": opts}
                    else:
                        continue
                    if len(tgt) != (2 if r.kind == "noul" else len(opts)):
                        del qs[qid]
                        continue
                    tg[qid] = tgt
                if not qs:
                    continue
                st = json.loads(gk[1])
                row = {"request_id": rid(name, gk[0]), "state": st, "questions": qs, "targets": tg,
                       "meta": meta(name, f"{name}/{src}", "synthetic_control", "exact", lic, "commercial", gk[0],
                                    generator="rule (Open-Jev)")}
                if valid_row(row):
                    rows.append(row)
                    n_rows += 1
    return rows


# ----------------------------------------------------------------------------- knowledge MC (train splits)

DEFAULT_HYP = "The correct answer is: {}"
FEWSHOT_HYP = "Answer: {}"
MC_INSTR = ["Pick the answer that is right.", "Choose the right response to this question.",
            "Identify the true answer among the candidates.", "What is the right answer here?",
            "Select the candidate that is factually right."]


def _mc_items(rng):
    """(dataset, id, question, options, gold, hyp_fn, license, license_use) from TRAIN splits."""
    items = []
    for cfg in ("ARC-Easy", "ARC-Challenge"):
        t = read_parquets(f"allenai__ai2_arc/{cfg}/train-*")
        for r in t.itertuples(index=False):
            labs = list(r.choices["label"])
            if r.answerKey in labs:
                items.append(("arc", f"{cfg}/{r.id}", r.question.strip(), [x.strip() for x in r.choices["text"]],
                              labs.index(r.answerKey), DEFAULT_HYP.format, "cc-by-sa-4.0", "commercial-sa"))
    t = read_parquets("cais__mmlu/all/auxiliary_train-*")
    idx = list(range(len(t)))
    rng.shuffle(idx)
    for i in idx[:30000]:
        r = t.iloc[i]
        tr = r["train"] if "train" in t.columns else r
        q, ch, a = tr["question"], list(tr["choices"]), int(tr["answer"])
        if len(ch) >= 2 and 0 <= a < len(ch):
            items.append(("mmlu_aux", f"aux/{i}", q.strip(), [c.strip() for c in ch], a, DEFAULT_HYP.format, "mit",
                          "commercial"))
    t = read_parquets("allenai__openbookqa/main/train-*")
    for r in t.itertuples(index=False):
        labs = list(r.choices["label"])
        if r.answerKey in labs:
            items.append(("openbookqa", r.id, r.question_stem.strip(), [x.strip() for x in r.choices["text"]],
                          labs.index(r.answerKey), DEFAULT_HYP.format, "unknown (card)", "unspecified"))
    t = read_parquets("tau__commonsense_qa/data/train-*")
    for r in t.itertuples(index=False):
        labs = list(r.choices["label"])
        if r.answerKey in labs:
            items.append(("commonsense_qa", r.id, r.question.strip(), [x.strip() for x in r.choices["text"]],
                          labs.index(r.answerKey), DEFAULT_HYP.format, "mit", "commercial"))
    t = read_parquets("allenai__winogrande/winogrande_xl/train-*")
    idx = list(range(len(t)))
    rng.shuffle(idx)
    for i in idx[:12000]:
        r = t.iloc[i]
        s = r["sentence"]
        if r["answer"] not in ("1", "2"):
            continue
        items.append(("winogrande", f"xl/{i}", s, [r["option1"], r["option2"]], int(r["answer"]) - 1,
                      (lambda o, s=s: s.replace("_", o)), "cc-by-4.0", "commercial"))
    t = read_parquets("Rowan__hellaswag/data/train-*")
    idx = list(range(len(t)))
    rng.shuffle(idx)
    for i in idx[:12000]:
        r = t.iloc[i]
        ctx = (r["ctx_a"] + " " + r["ctx_b"].capitalize()).strip() if r["ctx_b"] else r["ctx_a"].strip()
        if str(r["label"]) == "":
            continue
        items.append(("hellaswag", f"train/{i}", f"{r['activity_label']}: {ctx}", [e.strip() for e in r["endings"]],
                      int(r["label"]), (lambda o: o), "mit", "commercial"))
    items.extend(_gsm8k_items(rng))
    return items


def _num_str(v):
    return str(int(v)) if float(v) == int(v) else f"{v:.2f}"


def _gsm8k_items(rng):
    t = read_parquets("openai__gsm8k/main/train-*")
    out = []
    for i, r in enumerate(t.itertuples(index=False)):
        m = re.search(r"####\s*(-?[\d,]*\.?\d+)", r.answer)
        if not m:
            continue
        gv = float(m.group(1).replace(",", ""))
        g = _num_str(gv)
        k = rng.choice([4, 4, 10])
        cands = set()
        tries = 0
        while len(cands) < k - 1 and tries < 300:
            tries += 1
            op = rng.randrange(7)
            d = rng.choice([1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 50, 100])
            v = [gv + d, gv - d, gv * 2, gv / 2, gv * rng.choice([3, 4, 5]), gv + gv * rng.choice([0.1, 0.25, 0.5]),
                 abs(gv) + rng.randint(1, 60)][op]
            if v < 0:
                continue
            vs = _num_str(v)
            if vs != g:
                cands.add(vs)
        opts = [g] + list(cands)
        rng.shuffle(opts)
        out.append(("gsm8k", f"train/{i}", r.question.strip(), opts, opts.index(g), "The answer is {}.".format, "mit",
                    "commercial"))
    return out


def chess_items(rng, n=3000):
    import chess
    out = []
    while len(out) < n:
        b = chess.Board()
        for _ in range(rng.randint(8, 50)):
            mv = list(b.legal_moves)
            if not mv or b.is_game_over():
                break
            b.push(rng.choice(mv))
        legal = list(b.legal_moves)
        if len(legal) < 2 or b.is_game_over():
            continue
        legal_san = {b.san(m) for m in legal}
        good = b.san(rng.choice(legal))
        bad = set()
        for _ in range(400):
            if len(bad) >= 3:
                break
            own = [s for s in chess.SQUARES if b.piece_at(s) and b.piece_at(s).color == b.turn]
            sq = rng.choice(own)
            pc = b.piece_at(sq)
            to = rng.choice(chess.SQUARES)
            if to == sq or (b.piece_at(to) and b.piece_at(to).color == b.turn):
                continue
            cap = b.piece_at(to) is not None
            san = ((chess.square_name(sq)[0] + "x" if cap else "") + chess.square_name(to)) if pc.piece_type == chess.PAWN \
                else chess.piece_symbol(pc.piece_type).upper() + ("x" if cap else "") + chess.square_name(to)
            if san not in legal_san and san != good:
                bad.add(san)
        if len(bad) < 3:
            continue
        opts = [good] + sorted(bad)
        rng.shuffle(opts)
        pgn = chess.Board().variation_san(b.move_stack)
        side = "White" if b.turn else "Black"
        q = rng.choice([f"Game so far: {pgn}\nPosition (FEN): {b.fen()}\nIt is {side}'s turn. Which listed move is legal here?",
                        f"FEN: {b.fen()}\nMoves played: {pgn}\n{side} to play. Exactly one of these moves is legal; which?"])
        out.append(("chess_gen", f"seed{len(out)}", q, opts, opts.index(good), DEFAULT_HYP.format, "generated (MIT code)",
                    "commercial"))
    return out


def knowledge_rows(rng, n_nli_items=14000, items=None):
    """items: optional list of (dataset, id, question, options, gold, hyp_fn, license, license_use) tuples
    (default: the round-1 knowledge sources); used unchanged by verdict/data/knowledge2.py."""
    items = list(items) if items is not None else _mc_items(rng) + chess_items(rng)
    rng.shuffle(items)
    choice_rows, noul_rows, nli_rows = [], [], []
    by_ds = collections.Counter()
    for ds, iid, q, opts, gold, hyp, lic, use in items:
        if len(set(opts)) != len(opts) or len(opts) < 2:
            continue
        by_ds[ds] += 1
        gid = f"{ds}/{iid}"
        fam = "knowledge_mc"
        # 1) choice form (options as keys, letters are assigned by the template)
        state = q if rng.random() < 0.7 else {"question": q}
        if ds == "winogrande":
            instr = rng.choice(["Which candidate should replace the underscore so the sentence makes sense?",
                                "Put one of the candidates in place of _ ; which one fits?"])
        elif ds == "hellaswag":
            instr = rng.choice(["What is the likeliest next part of this scene?", "Choose the most natural follow-on sentence."])
        else:
            instr = rng.choice(MC_INSTR)
        choice_rows.append({"request_id": rid("mc", gid), "state": state,
                            "questions": {"answer": choice_q(instr, opts)},
                            "targets": {"answer": onehot(len(opts), gold)},
                            "meta": meta(ds, f"{ds}/choice", fam, "hard", lic, use, gid, eval_task_mirror=ds in ("gsm8k", "chess_gen"))})
        # 2) noul verification of one option (coherence pair with the choice form)
        if rng.random() < 0.35:
            j = gold if rng.random() < 0.5 else rng.choice([x for x in range(len(opts)) if x != gold])
            noul_rows.append({"request_id": rid("mcv", gid, j), "state": state,
                              "questions": {"is_correct": {"type": "noul", "instructions": f"Is \"{opts[j]}\" the correct answer?"}},
                              "targets": {"is_correct": onehot(2, int(j == gold))},
                              "meta": meta(ds, f"{ds}/verify", fam, "hard", lic, use, gid, coherence_pair=gid)})
    # 3) NLI rerank form, mirroring the OpenJev request shape (premise=question, hypothesis=template(option))
    rng.shuffle(items)
    for ds, iid, q, opts, gold, hyp, lic, use in items[:n_nli_items]:
        if len(set(opts)) != len(opts):
            continue
        gid = f"{ds}/{iid}"
        prem, h = q, hyp
        if ds not in ("winogrande", "hellaswag", "gsm8k") and rng.random() < 0.25:
            demos = [it for it in rng.sample(items, 8) if it[0] == ds and it[1] != iid][:5]
            if demos:
                prem = "\n\n".join(f"{d[2]}\nAnswer: {d[3][d[4]]}" for d in demos) + "\n\n" + q
                h = FEWSHOT_HYP.format
        for j, o in enumerate(opts):
            nli_rows.append(nli_row(ds, gid, prem, h(o), "entailment" if j == gold else "contradiction", lic, use,
                                    family="knowledge_mc_nli", task_id=f"{ds}/nli_rerank", option_index=j,
                                    n_options=len(opts), eval_task_mirror=ds in ("gsm8k", "chess_gen")))
    print("knowledge items per dataset:", dict(by_ds), flush=True)
    return choice_rows, noul_rows, nli_rows


# ----------------------------------------------------------------------------- NLI train splits

def nli_train(rng):
    rows = []
    t = read_parquets("nyu-mll__multi_nli/data/train-*", ["premise", "hypothesis", "label", "genre"])
    t = t[t.label.isin([0, 1, 2])].sample(n=20000, random_state=1)
    for i, r in enumerate(t.itertuples(index=False)):
        rows.append(nli_row("mnli", f"mnli/{i}", r.premise, r.hypothesis, NLI_KEYS[r.label], "cc-by-3.0/cc-by-sa-3.0/mit",
                            "commercial-sa"))
    for rnd in (1, 2, 3):
        t = read_parquets(f"facebook__anli/plain_text/train_r{rnd}-*")
        t = t[t.label.isin([0, 1, 2])].sample(n=min(4000, len(t)), random_state=rnd)
        for r in t.itertuples(index=False):
            rows.append(nli_row("anli", f"anli/{r.uid}", r.premise, r.hypothesis, NLI_KEYS[r.label], "cc-by-nc-4.0",
                                "noncommercial-internal", task_id="anli/nli"))
    w = read_jsonl(os.path.join(SRC, "alisawuffles__WANLI", "train.jsonl"))
    rng.shuffle(w)
    for r in w[:15000]:
        if r.get("gold") in NLI_KEYS:
            rows.append(nli_row("wanli", f"wanli/{r.get('id', r.get('pairID'))}", r["premise"], r["hypothesis"], r["gold"],
                                "cc-by-4.0", "commercial"))
    t = read_parquets("stanfordnlp__snli/plain_text/train-*")
    t = t[t.label.isin([0, 1, 2])].sample(n=8000, random_state=2)
    for i, r in enumerate(t.itertuples(index=False)):
        rows.append(nli_row("snli", f"snli/{i}", r.premise, r.hypothesis, NLI_KEYS[r.label], "cc-by-sa-4.0", "commercial-sa"))
    # SciTail rows are built here (keeps the RNG stream identical to the original run) but are REMOVED from the final
    # training data (meta.source == "scitail"): its hypotheses restate answers to ARC test items (see docs/REPRODUCE.md).
    t = read_parquets("allenai__scitail/snli_format/train-*")
    t = t.sample(n=min(6000, len(t)), random_state=3)
    for i, r in enumerate(t.itertuples(index=False)):
        lab = {"entailment": "entailment", "neutral": "neutral"}.get(r.gold_label)
        if lab:
            rows.append(nli_row("scitail", f"scitail/{i}", r.sentence1, r.sentence2, lab, "apache-2.0 (AllenAI)", "commercial"))
    # noul recasts of NLI ("is the hypothesis entailed?")
    nr = []
    for r in rng.sample(rows, 8000):
        p = r["targets"]["relation"]
        prem, hyp = r["state"]["premise"], r["state"]["hypothesis"]
        nr.append({"request_id": rid("nlinoul", r["request_id"]), "state": {"text": prem, "claim": hyp},
                   "questions": {"follows": {"type": "noul", "instructions": rng.choice([
                       "Does `claim` follow from `text`?", "Is `claim` necessarily true if `text` is true?",
                       "Based only on `text`, must `claim` be true?"])}},
                   "targets": {"follows": [1 - p[0], p[0]]},
                   "meta": {**r["meta"], "task_id": r["meta"]["source"] + "/noul", "coherence_pair": r["meta"]["group_id"]}})
    return rows + nr


# ----------------------------------------------------------------------------- intent / routing / fact verification

MASSIVE_SCEN_DESC = {
    "alarm": "Alarms: setting, checking or cancelling them.", "audio": "Device sound and volume control.",
    "calendar": "Calendar entries, meetings and reminders.", "cooking": "Recipes and how to cook things.",
    "datetime": "Questions about dates, times and time zones.", "email": "Email and contacts.",
    "general": "Chit-chat, jokes and generic assistant commands.", "iot": "Smart-home devices: lights, plugs, vacuum, coffee machine.",
    "lists": "To-do and shopping lists.", "music": "Music playback and preferences.", "news": "News updates.",
    "play": "Playing media: radio, podcasts, audiobooks, games.", "qa": "General knowledge questions, definitions, currency, stock.",
    "recommendation": "Recommendations for places, events or movies.", "social": "Social-media posting and complaints.",
    "takeaway": "Ordering food for delivery or takeaway.", "transport": "Taxis, trains, tickets and traffic.",
    "weather": "Weather questions.",
}


def massive_train(rng):
    path = glob.glob(os.path.join(SRC, "massive", "*", "data", "en-US.jsonl"))[0]
    recs = [r for r in read_jsonl(path) if r.get("partition") == "train"]
    intents = sorted({r["intent"] for r in recs})
    scen = sorted({r["scenario"] for r in recs})
    rows = []
    for r in recs:
        qs, tg = {}, {}
        sc_keys = scen[:]
        rng.shuffle(sc_keys)
        if rng.random() < 0.5:
            qs["domain"] = {"type": "choice", "instructions": rng.choice([
                "Which skill area of a voice assistant does this request belong to?",
                "Route this voice command to the right assistant domain."]),
                "criteria": {k: MASSIVE_SCEN_DESC.get(k) for k in sc_keys}}
        else:
            qs["domain"] = choice_q("Which category best fits this voice-assistant request?", sc_keys)
        tg["domain"] = onehot(len(sc_keys), sc_keys.index(r["scenario"]))
        k = rng.choice([4, 8, 12, 20, 30, len(intents)])
        neg = [x for x in intents if x != r["intent"]]
        opts = rng.sample(neg, min(k - 1, len(neg))) + [r["intent"]]
        rng.shuffle(opts)
        qs["intent"] = choice_q("What does the user want the assistant to do?", opts)
        tg["intent"] = onehot(len(opts), opts.index(r["intent"]))
        state = r["utt"] if rng.random() < 0.5 else {"utterance": r["utt"]}
        rows.append({"request_id": rid("massive", r["id"]), "state": state, "questions": qs, "targets": tg,
                     "meta": meta("massive-train", "massive/intent+scenario", "intent", "hard", "cc-by-4.0", "commercial",
                                  f"massive/{r['id']}")})
    return rows


def vitaminc_train(rng, n=20000):
    recs = read_jsonl(os.path.join(SRC, "tals__vitaminc", "train.jsonl"))
    rng.shuffle(recs)
    labels = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
    variants = [
        ("Using only the evidence passage, how does it relate to the claim?",
         {"SUPPORTS": "The passage backs the claim up.", "REFUTES": "The passage shows the claim is false.",
          "NOT ENOUGH INFO": "The passage does not settle whether the claim is true."}),
        ("Does the evidence support or refute the claim, or is it insufficient?", None),
    ]
    rows = []
    for r in recs[:n]:
        if r.get("label") not in labels:
            continue
        instr, desc = rng.choice(variants)
        keys = labels[:]
        if rng.random() < 0.5:
            rng.shuffle(keys)
        q = {"type": "choice", "instructions": instr, "criteria": {k: (desc[k] if desc else None) for k in keys}}
        rows.append({"request_id": rid("vitc", r.get("unique_id", r["claim"])), "state": {"evidence": r["evidence"], "claim": r["claim"]},
                     "questions": {"verdict": q}, "targets": {"verdict": onehot(3, keys.index(r["label"]))},
                     "meta": meta("vitaminc-train", "vitaminc/fact", "fact_verification", "hard", "cc-by-sa-3.0", "commercial-sa",
                                  f"vitc/{r.get('case_id', r.get('unique_id'))}")})
    return rows


def _readable(lbl):
    return lbl.replace("_", " ")


def intent_sets(rng):
    rows = []
    t = read_parquets("legacy-datasets__banking77/data/train-*")
    names = ["activate_my_card", "age_limit", "apple_pay_or_google_pay", "atm_support", "automatic_top_up",
             "balance_not_updated_after_bank_transfer", "balance_not_updated_after_cheque_or_cash_deposit",
             "beneficiary_not_allowed", "cancel_transfer", "card_about_to_expire", "card_acceptance", "card_arrival",
             "card_delivery_estimate", "card_linking", "card_not_working", "card_payment_fee_charged",
             "card_payment_not_recognised", "card_payment_wrong_exchange_rate", "card_swallowed", "cash_withdrawal_charge",
             "cash_withdrawal_not_recognised", "change_pin", "compromised_card", "contactless_not_working",
             "country_support", "declined_card_payment", "declined_cash_withdrawal", "declined_transfer",
             "direct_debit_payment_not_recognised", "disposable_card_limits", "edit_personal_details", "exchange_charge",
             "exchange_rate", "exchange_via_app", "extra_charge_on_statement", "failed_transfer", "fiat_currency_support",
             "get_disposable_virtual_card", "get_physical_card", "getting_spare_card", "getting_virtual_card",
             "lost_or_stolen_card", "lost_or_stolen_phone", "order_physical_card", "passcode_forgotten",
             "pending_card_payment", "pending_cash_withdrawal", "pending_top_up", "pending_transfer", "pin_blocked",
             "receiving_money", "Refund_not_showing_up", "request_refund", "reverted_card_payment",
             "supported_cards_and_currencies", "terminate_account", "top_up_by_bank_transfer_charge", "top_up_by_card_charge",
             "top_up_by_cash_or_cheque", "top_up_failed", "top_up_limits", "top_up_reverted", "topping_up_by_card",
             "transaction_charged_twice", "transfer_fee_charged", "transfer_into_account", "transfer_not_received_by_recipient",
             "transfer_timing", "unable_to_verify_identity", "verify_my_identity", "verify_source_of_funds", "verify_top_up",
             "virtual_card_not_working", "visa_or_mastercard", "why_verify_identity", "wrong_amount_of_cash_received",
             "wrong_exchange_rate_for_cash_withdrawal"]
    t = t.sample(n=min(7000, len(t)), random_state=4)
    for i, r in enumerate(t.itertuples(index=False)):
        gold = names[int(r.label)]
        k = rng.choice([5, 10, 20, 40, 77, 77])
        opts = rng.sample([x for x in names if x != gold], min(k - 1, 76)) + [gold]
        rng.shuffle(opts)
        rows.append({"request_id": rid("b77", i, r.text), "state": r.text,
                     "questions": {"intent": choice_q(rng.choice(["Which banking support topic is this message about?",
                                                                  "Classify the customer's banking request."]), opts)},
                     "targets": {"intent": onehot(len(opts), opts.index(gold))},
                     "meta": meta("banking77-train", "banking77/intent", "intent", "hard", "cc-by-4.0", "commercial", f"b77/{i}")})
    t = read_parquets("clinc__clinc_oos/plus/train-*")
    lab_col = "intent"
    nm = None
    try:
        nm = pq.read_schema(glob.glob(os.path.join(SRC, "clinc__clinc_oos/plus/train-*"))[0]).metadata
    except Exception:
        pass
    names_c = None
    if nm and b"huggingface" in nm:
        info = json.loads(nm[b"huggingface"])
        names_c = info["info"]["features"][lab_col].get("names")
    t = t.sample(n=min(8000, len(t)), random_state=5)
    all_l = names_c or sorted({str(x) for x in t[lab_col]})
    for i, r in enumerate(t.itertuples(index=False)):
        gold = all_l[int(getattr(r, lab_col))] if names_c else str(getattr(r, lab_col))
        k = rng.choice([4, 8, 15, 30, 50])
        opts = rng.sample([x for x in all_l if x != gold], min(k - 1, len(all_l) - 1)) + [gold]
        rng.shuffle(opts)
        desc = None
        if gold == "oos" or "oos" in opts:
            desc = ["None of the other categories applies." if o == "oos" else None for o in opts]
        rows.append({"request_id": rid("clinc", i, r.text), "state": r.text,
                     "questions": {"intent": choice_q("What is the user asking the assistant for?", opts, desc)},
                     "targets": {"intent": onehot(len(opts), opts.index(gold))},
                     "meta": meta("clinc-train", "clinc/intent", "intent", "hard", "cc-by-3.0", "commercial", f"clinc/{i}")})
    t = read_parquets("google__boolq/data/train-*").sample(n=6000, random_state=6)
    for i, r in enumerate(t.itertuples(index=False)):
        rows.append({"request_id": rid("boolq", i, r.question), "state": {"passage": r.passage},
                     "questions": {"answer": {"type": "noul", "instructions": r.question.strip().capitalize() + "?"}},
                     "targets": {"answer": onehot(2, int(bool(r.answer)))},
                     "meta": meta("boolq-train", "boolq/yesno", "yes_no_qa", "hard", "cc-by-sa-3.0", "commercial-sa", f"boolq/{i}")})
    return rows


# ----------------------------------------------------------------------------- human soft labels

def soft_sets(rng):
    rows = []
    # civil_comments: toxicity = fraction of raters -> soft noul; severity score from the fraction
    t = read_parquets("google__civil_comments/data/train-00000-*")
    t = pd.concat([t[t.toxicity >= 0.3].sample(n=4000, random_state=7), t[t.toxicity < 0.3].sample(n=4000, random_state=8)])
    for i, r in enumerate(t.itertuples(index=False)):
        p = float(r.toxicity)
        qs = {"toxic": {"type": "noul", "instructions": rng.choice(["Would a typical reader find this comment toxic?",
                                                                    "Is this comment rude, disrespectful or hateful?"])}}
        tg = {"toxic": [1 - p, p]}
        if rng.random() < 0.4:
            qs["insult"] = {"type": "noul", "instructions": "Does the comment insult someone?"}
            tg["insult"] = [1 - float(r.insult), float(r.insult)]
        rows.append({"request_id": rid("civil", i, r.text[:100]), "state": r.text, "questions": qs, "targets": tg,
                     "meta": meta("civil_comments", "civil/toxicity", "toxicity", "soft_human", "cc0-1.0", "commercial", f"civil/{i}")})
    # go_emotions raw: per-rater labels -> soft distribution over a sampled label subset
    t = read_parquets("google-research-datasets__go_emotions/raw/train-*")
    emo = [c for c in t.columns if c not in ("text", "id", "author", "subreddit", "link_id", "parent_id", "created_utc",
                                             "rater_id", "example_very_unclear")]
    g = t.groupby("id")
    ids = list(g.groups)
    rng.shuffle(ids)
    n = 0
    for eid in ids:
        if n >= 6000:
            break
        grp = g.get_group(eid)
        votes = grp[emo].sum(axis=0).astype(float)
        if votes.sum() == 0 or len(grp) < 2:
            continue
        top = votes.sort_values(ascending=False)
        pos = [k for k, v in top.items() if v > 0][:4]
        others = [k for k in emo if k not in pos]
        opts = pos + rng.sample(others, rng.randint(2, 6))
        rng.shuffle(opts)
        v = np.array([votes[o] for o in opts]) + 1e-9
        rows.append({"request_id": rid("goemo", eid), "state": grp.text.iloc[0],
                     "questions": {"emotion": choice_q(rng.choice(["Pick the feeling that best matches this message.", "What mood comes through most strongly here?"]), opts)},
                     "targets": {"emotion": (v / v.sum()).tolist()},
                     "meta": meta("go_emotions", "goemotions/emotion", "emotion", "soft_human", "apache-2.0", "commercial", f"goemo/{eid}")})
        n += 1
    # measuring-hate-speech: per-annotator ordinal items -> soft score distribution
    t = read_parquets("ucberkeley-dlab__measuring-hate-speech/measuring-hate-speech.parquet")
    g = t.groupby("comment_id")
    ids = list(g.groups)
    rng.shuffle(ids)
    n = 0
    for cid in ids:
        if n >= 12000:
            break
        grp = g.get_group(cid)
        if len(grp) < 2:
            continue
        qs, tg = {}, {}
        for col, instr, levels in [
            ("hatespeech", "Is this comment hate speech?", ["Not hate speech", "Unclear", "Hate speech"]),
            ("respect", "How respectful is this comment toward the people it targets?",
             ["Strongly respectful", "Respectful", "Neutral", "Disrespectful", "Strongly disrespectful"]),
            ("insult", "How insulting is the comment?", ["Not at all", "Slightly", "Moderately", "Very", "Extremely"]),
        ]:
            if col not in grp.columns:
                continue
            vals = grp[col].dropna().astype(int).clip(0, len(levels) - 1)
            h = np.bincount(vals, minlength=len(levels)).astype(float)
            if h.sum() == 0:
                continue
            qs[col] = {"type": "score", "instructions": instr, "criteria": levels}
            tg[col] = (h / h.sum()).tolist()
        if qs:
            rows.append({"request_id": rid("mhs", cid), "state": grp.text.iloc[0], "questions": qs, "targets": tg,
                         "meta": meta("measuring-hate-speech", "mhs/ordinal", "toxicity", "soft_human", "cc-by-4.0", "commercial", f"mhs/{cid}")})
            n += 1
    # oasst1: per-message label {value, count} -> soft 5-level scores
    t = read_parquets("OpenAssistant__oasst1/data/train-*")
    t = t[(t.lang == "en") & (t.role == "assistant")]
    msgs = read_parquets("OpenAssistant__oasst1/data/train-*", ["message_id", "text"]).set_index("message_id")["text"]
    n = 0
    for r in t.sample(frac=1.0, random_state=9).itertuples(index=False):
        if n >= 9000:
            break
        lab = r.labels
        if lab is None or not len(lab.get("name", [])):
            continue
        d = dict(zip(lab["name"], zip(lab["value"], lab["count"])))
        if "quality" not in d or d["quality"][1] < 2:
            continue
        prompt = msgs.get(r.parent_id)
        if not isinstance(prompt, str):
            continue
        qs, tg = {}, {}
        for nm, instr, levels in [
            ("quality", "Rate the overall quality of the assistant's reply.", ["Very poor", "Poor", "Acceptable", "Good", "Excellent"]),
            ("helpfulness", "How helpful is the reply to the user?", ["Not helpful", "Slightly", "Moderately", "Very", "Extremely helpful"]),
            ("creativity", "How creative is the reply?", ["Not creative", "Slightly", "Moderately", "Very", "Extremely creative"]),
        ]:
            if nm in d and d[nm][1] >= 2:
                m = float(d[nm][0]) * 4
                lo = int(math.floor(m))
                hi = min(lo + 1, 4)
                v = [0.0] * 5
                v[lo] += 1 - (m - lo)
                v[hi] += m - lo
                qs[nm] = {"type": "score", "instructions": instr, "criteria": levels}
                tg[nm] = v
        if qs:
            rows.append({"request_id": rid("oasst", r.message_id), "state": {"user": prompt, "assistant_reply": r.text},
                         "questions": qs, "targets": tg,
                         "meta": meta("oasst1", "oasst/quality", "ordinal_quality", "soft_human", "apache-2.0", "commercial",
                                      f"oasst/{r.message_tree_id}")})
            n += 1
    # HelpSteer2: human 0-4 Likert x 5 attributes -> multi-question score
    hs = read_jsonl(os.path.join(SRC, "nvidia__HelpSteer2", "train.jsonl.gz"))
    rng.shuffle(hs)
    attrs = {"helpfulness": "How helpful is the response overall?", "correctness": "How factually correct and complete is the response?",
             "coherence": "How clear and coherent is the response?", "complexity": "How much expertise does the response require to write?",
             "verbosity": "How verbose is the response relative to what was asked?"}
    for i, r in enumerate(hs[:12000]):
        ks = rng.sample(list(attrs), rng.randint(2, 5))
        qs = {k: {"type": "score", "instructions": attrs[k], "criteria": ["0 (lowest)", "1", "2", "3", "4 (highest)"]} for k in ks}
        tg = {k: onehot(5, int(r[k])) for k in ks}
        rows.append({"request_id": rid("hs2", i), "state": {"prompt": r["prompt"], "response": r["response"]}, "questions": qs,
                     "targets": tg, "meta": meta("helpsteer2", "helpsteer2/attributes", "ordinal_quality", "hard", "cc-by-4.0",
                                                 "commercial", f"hs2/{hashlib.md5(r['prompt'].encode()).hexdigest()[:12]}")})
    return [r for r in rows if valid_row(r)]


# ----------------------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    jobs = {
        "tasksource_typed": lambda r: write("tasksource_typed", tasksource_typed(r)),
        "procedural_typed": lambda r: write("procedural_typed", procedural_typed(r)),
        "open_jev": lambda r: write("open_jev", open_jev(r)),
        "knowledge": None,
        "nli_train": lambda r: write("nli_train", nli_train(r)),
        "massive_train": lambda r: write("massive_train", massive_train(r)),
        "vitaminc_train": lambda r: write("vitaminc_train", vitaminc_train(r)),
        "intent_sets": lambda r: write("intent_sets", intent_sets(r)),
        "soft_sets": lambda r: write("soft_sets", soft_sets(r)),
        # (the original run also had an own procedural generator here; it was removed after the leakage audit and is not
        #  part of the final data or of this repository)
    }
    for name in jobs:
        if a.only and name not in a.only:
            continue
        rng = random.Random(f"{a.seed}/{name}")
        if name == "knowledge":
            c, n, nl = knowledge_rows(rng)
            write("knowledge_choice", c)
            write("knowledge_verify", n)
            write("knowledge_nli", nl)
        else:
            jobs[name](rng)


if __name__ == "__main__":
    main()
