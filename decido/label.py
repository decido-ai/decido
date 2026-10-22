"""Run a (teacher or student) model over unified-schema rows: store predicted distributions, score vs targets.

    python -m decido.label --model Qwen/Qwen3.8-27B --in data/val.jsonl --out runs/label/val.qwen38.jsonl \
        [--perm 2] [--tp 1] [--limit N] [--shard i/n]

Output rows: {"request_id", "pred": {qid: [p...]}} (aligned with option_keys / the row's targets).
If rows have targets, prints accuracy (argmax vs target argmax), NLL, Brier per kind and per source.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--perm", type=int, default=1)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--chunk", type=int, default=4096)
    ap.add_argument("--quantization", default=None)
    ap.add_argument("--lora", default=None)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--max-num-seqs", type=int, default=256)
    a = ap.parse_args()

    rows = [json.loads(x) for x in open(a.inp) if x.strip()]
    si, sn = map(int, a.shard.split("/"))
    rows = rows[si::sn]
    if a.limit:
        rows = rows[:a.limit]
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(x)["request_id"] for x in open(a.out) if x.strip()}
    todo = [r for r in rows if r["request_id"] not in done]
    print(f"{len(rows)} rows, {len(done)} already labelled, {len(todo)} to go", flush=True)

    from decido.predictor_hf import VerdictVLLM
    pred = VerdictVLLM(model=a.model, tokenizer=a.tokenizer, tp=a.tp, gpu_mem=a.gpu_mem, perm=a.perm,
                       max_model_len=a.max_model_len, quantization=a.quantization, lora=a.lora,
                       max_num_seqs=a.max_num_seqs)
    pred.readout.temps = {}
    t0 = time.time()
    with open(a.out, "a") as fh:
        for s in range(0, len(todo), a.chunk):
            chunk = todo[s:s + a.chunk]
            reqs = [{"state": r["state"], "questions": r["questions"]} for r in chunk]
            outs = pred.predict_batch(reqs)
            for r, o in zip(chunk, outs):
                d = {}
                for qid, q in r["questions"].items():
                    ans = o["answers"][qid]
                    if q["type"] == "noul":
                        d[qid] = [1 - ans["noul"], ans["noul"]]
                    elif q["type"] == "choice":
                        d[qid] = [ans["probabilities"][k] for k in q["criteria"]]
                    else:
                        d[qid] = [ans["probabilities"][str(i)] for i in range(len(q["criteria"]))]
                fh.write(json.dumps({"request_id": r["request_id"], "pred": d}) + "\n")
            fh.flush()
            n = s + len(chunk)
            el = time.time() - t0
            print(f"  {n}/{len(todo)} rows  {n / el:.1f} rows/s  eta {(len(todo) - n) / max(n / el, 1e-9) / 60:.0f} min", flush=True)
    report(a.inp if not a.limit else None, a.out, rows)


def report(_inp, out, rows):
    preds = {}
    for x in open(out):
        if x.strip():
            j = json.loads(x)
            preds[j["request_id"]] = j["pred"]
    agg = collections.defaultdict(lambda: [0, 0.0, 0.0, 0.0])   # n, acc, nll, brier
    for r in rows:
        if "targets" not in r or r["request_id"] not in preds:
            continue
        for qid, t in r["targets"].items():
            p = np.asarray(preds[r["request_id"]][qid], dtype=float)
            t = np.asarray(t, dtype=float)
            t = t / t.sum()
            acc = float(int(p.argmax()) == int(t.argmax()))
            nll = float(-(t * np.log(np.clip(p, 1e-12, 1))).sum())
            br = float(((p - t) ** 2).sum())
            kind = r["questions"][qid]["type"]
            src = r.get("meta", {}).get("source", "?")
            for key in ("ALL", f"kind:{kind}", f"src:{src}"):
                a = agg[key]
                a[0] += 1
                a[1] += acc
                a[2] += nll
                a[3] += br
    res = {k: {"n": v[0], "acc": v[1] / v[0], "nll": v[2] / v[0], "brier": v[3] / v[0]} for k, v in agg.items() if v[0]}
    for k in sorted(res, key=lambda k: (not k.startswith("ALL"), not k.startswith("kind"), k)):
        v = res[k]
        print(f"{k:40s} n={v['n']:6d} acc={v['acc']:.4f} nll={v['nll']:.4f} brier={v['brier']:.4f}")
    json.dump(res, open(out + ".metrics.json", "w"), indent=1)


if __name__ == "__main__":
    main()
