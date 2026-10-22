"""GGUF (CPU llama.cpp) vs HF bf16 agreement on fixed non-test prompts (Phase E.3), plus per-backend accuracy.

    # reference (GPU, transformers bf16):
    python -m decido.agree run --backend hf --model models/merged/r1 --rows data/val.jsonl --n 300 --out runs/agree/hf.jsonl
    # each quant (CPU only):
    python -m decido.agree run --backend gguf --gguf models/gguf/r1-q4_0.gguf --bundle out/bundle --rows data/val.jsonl \
        --n 300 --out runs/agree/q4_0.jsonl --threads 40
    python -m decido.agree compare --ref runs/agree/hf.jsonl --other runs/agree/q4_0.jsonl --rows data/val.jsonl

Prompts: the first N rows of a fixed seeded shuffle of `rows` (all their questions). Distributions are raw
(no temperature) so the comparison isolates backend numerics.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time

import numpy as np


def pick(rows_path, n, seed=300):
    rows = [json.loads(x) for x in open(rows_path) if x.strip()]
    random.Random(seed).shuffle(rows)
    return rows[:n] if n else rows


def dist_of(q, ans):
    if q["type"] == "noul":
        return [1 - ans["noul"], ans["noul"]]
    if q["type"] == "choice":
        return [ans["probabilities"][k] for k in q["criteria"]]
    return [ans["probabilities"][str(i)] for i in range(len(q["criteria"]))]


def cmd_run(a):
    rows = pick(a.rows, a.n)
    if a.backend == "hf":
        from decido.predictor_hf import VerdictHF
        pred = VerdictHF(model=a.model, perm=a.perm)
    else:
        if not a.gpu_layers:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        from decido.predictor_llamacpp import VerdictLlamaCpp
        pred = VerdictLlamaCpp(gguf=a.gguf, bundle=a.bundle, threads=a.threads, perm=a.perm, workers=a.workers,
                               gpu_layers=a.gpu_layers)
    pred.readout.temps = {}
    t0 = time.time()
    reqs = [{"state": r["state"], "questions": r["questions"]} for r in rows]
    outs = pred.predict_batch(reqs) if hasattr(pred, "predict_batch") else [pred(q) for q in reqs]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as fh:
        for r, o in zip(rows, outs):
            fh.write(json.dumps({"request_id": r["request_id"],
                                 "pred": {qid: dist_of(q, o["answers"][qid]) for qid, q in r["questions"].items()}}) + "\n")
    print(f"{len(rows)} rows in {time.time() - t0:.0f}s -> {a.out}")


def load(p):
    return {json.loads(x)["request_id"]: json.loads(x)["pred"] for x in open(p) if x.strip()}


def cmd_compare(a):
    ref, oth = load(a.ref), load(a.other)
    rows = {json.loads(x)["request_id"]: json.loads(x) for x in open(a.rows) if x.strip()}
    agree = n = 0
    mae, maxd = [], 0.0
    acc_ref = acc_oth = nt = 0
    for rid, pr in ref.items():
        if rid not in oth:
            continue
        for qid, p in pr.items():
            p, q = np.asarray(p), np.asarray(oth[rid][qid])
            n += 1
            agree += int(p.argmax() == q.argmax())
            mae.append(float(np.abs(p - q).mean()))
            maxd = max(maxd, float(np.abs(p - q).max()))
            t = rows.get(rid, {}).get("targets", {}).get(qid)
            if t is not None:
                t = np.asarray(t)
                nt += 1
                acc_ref += int(p.argmax() == t.argmax())
                acc_oth += int(q.argmax() == t.argmax())
    res = {"questions": n, "argmax_agreement": agree / n, "prob_mae": float(np.mean(mae)), "prob_max_abs_diff": maxd,
           "acc_ref": acc_ref / max(nt, 1), "acc_other": acc_oth / max(nt, 1), "acc_delta_points": 100 * (acc_oth - acc_ref) / max(nt, 1)}
    print(json.dumps(res, indent=1))
    if a.save:
        json.dump(res, open(a.save, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--backend", choices=["hf", "gguf"], required=True)
    r.add_argument("--model")
    r.add_argument("--gguf")
    r.add_argument("--bundle")
    r.add_argument("--rows", required=True)
    r.add_argument("--n", type=int, default=300)
    r.add_argument("--out", required=True)
    r.add_argument("--threads", type=int, default=40)
    r.add_argument("--workers", type=int, default=1)
    r.add_argument("--perm", type=int, default=1)
    r.add_argument("--gpu-layers", type=int, default=0, help="llama.cpp CUDA offload (quant screening only)")
    c = sub.add_parser("compare")
    c.add_argument("--ref", required=True)
    c.add_argument("--other", required=True)
    c.add_argument("--rows", required=True)
    c.add_argument("--save", default=None)
    a = ap.parse_args()
    {"run": cmd_run, "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    main()
