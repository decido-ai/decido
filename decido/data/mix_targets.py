"""Combine gold/human/exact targets with teacher soft labels into training targets (Phase B -> C).

    python -m decido.data.mix_targets --pool data/train_pool.jsonl --teacher runs/label/pool.<t>.jsonl \
        --calib data/calib.jsonl --calib-teacher runs/label/calib.<t>.jsonl \
        [--synth data/synth/rows.jsonl --synth-teacher runs/label/synth.<t>.jsonl] --out data/train.jsonl

Rules (targets aligned with option_keys):
  exact       -> exact target (rule-computed; teacher not used)
  hard        -> w_gold * onehot(gold) + (1 - w_gold) * teacher_T
  soft_human  -> w_human * human + (1 - w_human) * teacher_T
  synthetic   -> teacher_T (teacher-written states have no gold)
teacher_T = teacher distribution re-tempered with a per-kind temperature fitted on calib.jsonl (gold NLL) only.
Rows without a teacher label keep their original target. Teacher targets are stored in meta for re-mixing.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from scipy.optimize import minimize_scalar


def load_preds(path):
    out = {}
    if path:
        for x in open(path):
            if x.strip():
                j = json.loads(x)
                out[j["request_id"]] = j["pred"]
    return out


def retemper(p, T):
    lp = np.log(np.clip(np.asarray(p, dtype=float), 1e-12, 1)) / T
    lp -= lp.max()
    q = np.exp(lp)
    return q / q.sum()


def fit_temps(rows, preds):
    by = {"noul": [], "choice": [], "score": []}
    for r in rows:
        pr = preds.get(r["request_id"])
        if not pr:
            continue
        for qid, t in r["targets"].items():
            if qid in pr:
                by[r["questions"][qid]["type"]].append((np.asarray(pr[qid]), np.asarray(t) / np.sum(t)))
    temps = {}
    for k, pairs in by.items():
        if len(pairs) < 50:
            temps[k] = 1.0
            continue

        def nll(T):
            return float(np.mean([-(t * np.log(np.clip(retemper(p, T), 1e-12, 1))).sum() for p, t in pairs]))
        res = minimize_scalar(nll, bounds=(0.3, 5.0), method="bounded")
        temps[k] = float(res.x)
        print(f"teacher temp {k}: T={res.x:.3f} nll {nll(1.0):.4f} -> {res.fun:.4f} (n={len(pairs)})")
    return temps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--calib-teacher", required=True)
    ap.add_argument("--synth", default=None)
    ap.add_argument("--synth-teacher", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--w-gold", type=float, default=0.7)
    ap.add_argument("--w-human", type=float, default=0.7)
    ap.add_argument("--teacher-name", default="teacher")
    a = ap.parse_args()

    calib = [json.loads(x) for x in open(a.calib)]
    temps = fit_temps(calib, load_preds(a.calib_teacher))
    tp = load_preds(a.teacher)
    stats = {"rows": 0, "questions": 0, "teacher_mixed": 0, "kept_original": 0, "synth": 0}
    with open(a.out, "w") as fh:
        def emit(r):
            stats["rows"] += 1
            stats["questions"] += len(r["targets"])
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

        for x in open(a.pool):
            r = json.loads(x)
            lt = r["meta"].get("label_type")
            pr = tp.get(r["request_id"])
            if lt == "exact" or not pr:
                stats["kept_original"] += len(r["targets"])
                emit(r)
                continue
            w = a.w_gold if lt == "hard" else a.w_human
            newt, tmeta = {}, {}
            for qid, t in r["targets"].items():
                t = np.asarray(t, dtype=float)
                t = t / t.sum()
                if qid not in pr or len(pr[qid]) != len(t):
                    newt[qid] = t.tolist()
                    continue
                q = retemper(pr[qid], temps[r["questions"][qid]["type"]])
                newt[qid] = (w * t + (1 - w) * q).tolist()
                tmeta[qid] = [round(float(v), 6) for v in q]
                stats["teacher_mixed"] += 1
            r["meta"]["gold_targets"] = r["targets"]
            r["meta"]["teacher_targets"] = tmeta
            r["meta"]["teacher"] = a.teacher_name
            r["targets"] = newt
            emit(r)
        if a.synth:
            sp = load_preds(a.synth_teacher)
            for x in open(a.synth):
                r = json.loads(x)
                pr = sp.get(r["request_id"])
                if not pr:
                    continue
                r["targets"] = {qid: retemper(p, temps[r["questions"][qid]["type"]]).tolist() for qid, p in pr.items()}
                r["meta"]["teacher"] = a.teacher_name
                stats["synth"] += len(r["targets"])
                emit(r)
    json.dump({"temps": temps, **stats}, open(a.out + ".stats.json", "w"), indent=1)
    print(stats)


if __name__ == "__main__":
    main()
