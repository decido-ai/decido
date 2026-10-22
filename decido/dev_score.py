"""Pre-registered dev score (RESULTS.md 'Round-selection rule'): accuracy + NLL on val_heldout_b + val_synth_b.
    python -m decido.dev_score PRED.jsonl [PRED2.jsonl ...]   (raw predictions from decido.label)"""
import json
import re
import sys

import numpy as np

W = re.compile(r"\w+")


def _norm(t):
    return " ".join(W.findall(t.lower()))


def _leaves(o):
    if isinstance(o, str):
        yield o
    elif isinstance(o, dict):
        for v in o.values():
            yield from _leaves(v)
    elif isinstance(o, list):
        for v in o:
            yield from _leaves(v)


def _keys(state):
    if isinstance(state, dict) and "premise" in state:
        state = state["premise"].split("\n\n")[-1]
    return {x for x in (_norm(t) for t in _leaves(state)) if len(x.split()) >= 6}

DEV = ["data/val_heldout_b.jsonl", "data/val_synth_b.jsonl"]
args = sys.argv[1:]
exclude_keys = set()
if "--exclude-overlap" in args:   # drop dev rows whose state shares a >=6-word leaf with this training file
    i = args.index("--exclude-overlap")
    for x in open(args[i + 1]):
        exclude_keys |= _keys(json.loads(x)["state"])
    args = args[:i] + args[i + 2:]
preds = {}
for p in args:
    for x in open(p):
        if x.strip():
            j = json.loads(x)
            preds[j["request_id"]] = j["pred"]
acc, nll, miss = [], [], 0
for f in DEV:
    for x in open(f):
        r = json.loads(x)
        if exclude_keys and _keys(r["state"]) & exclude_keys:
            excluded = globals().setdefault("excluded", [0])
            excluded[0] += len(r["targets"])
            continue
        if r["request_id"] not in preds:
            miss += len(r["targets"])
            continue
        for q, t in r["targets"].items():
            t = np.asarray(t, float) / sum(t)
            p = np.asarray(preds[r["request_id"]][q], float)
            acc.append(float(p.argmax() == t.argmax()))
            nll.append(float(-(t * np.log(np.clip(p, 1e-12, 1))).sum()))
print(json.dumps({"excluded_overlap_questions": globals().get("excluded", [0])[0], "dev_questions": len(acc), "missing": miss, "dev_acc": round(100 * np.mean(acc), 3), "dev_nll": round(float(np.mean(nll)), 4)}))
