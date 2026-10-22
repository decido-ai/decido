"""Assemble data/train.jsonl from the mixed pool (+ compound + synthetic) with per-source question caps.

    python -m decido.data.make_train --mixed data/train_mixed.jsonl --extra data/compound_train.jsonl \
        --out data/train.jsonl --cap 30000 --max-questions 330000
Synthetic workflows: 10% held out (whole workflows) into data/val_synth.jsonl.
"""
import argparse
import collections
import hashlib
import json
import random

ap = argparse.ArgumentParser()
ap.add_argument("--mixed", required=True)
ap.add_argument("--extra", nargs="*", default=[])
ap.add_argument("--out", required=True)
ap.add_argument("--val-synth", default="data/val_synth.jsonl")
ap.add_argument("--cap", type=int, default=30000, help="max questions per source")
ap.add_argument("--caps", default="", help="src=N,src=N overrides")
ap.add_argument("--max-questions", type=int, default=330000)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
caps = collections.defaultdict(lambda: a.cap)
for kv in filter(None, a.caps.split(",")):
    k, v = kv.split("=")
    caps[k] = int(v)
rows = []
for f in [a.mixed] + a.extra:
    rows += [json.loads(x) for x in open(f) if x.strip()]
rng = random.Random(a.seed)
rng.shuffle(rows)
val_synth, kept, used = [], [], collections.Counter()
for r in rows:
    m = r["meta"]
    if m.get("source") == "verdict-teacher-synth":
        h = int(hashlib.sha1(m["workflow_id"].encode()).hexdigest(), 16) % 10
        if h == 0:
            val_synth.append(r)
            continue
    nq = len(r.get("targets", {}))
    if nq == 0 or used[m["source"]] + nq > caps[m["source"]]:
        continue
    used[m["source"]] += nq
    kept.append(r)
total = sum(len(r["targets"]) for r in kept)
if total > a.max_questions:
    rng.shuffle(kept)
    acc, out = 0, []
    for r in kept:
        if acc >= a.max_questions:
            break
        out.append(r)
        acc += len(r["targets"])
    kept = out
with open(a.out, "w") as fh:
    for r in kept:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
with open(a.val_synth, "w") as fh:
    for r in val_synth:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
kinds = collections.Counter(q["type"] for r in kept for q in r["questions"].values() if q and True)
print("rows", len(kept), "questions", sum(len(r["targets"]) for r in kept), dict(kinds))
for s, n in sorted(used.items(), key=lambda x: -x[1]):
    print(f"  {s:32s} {n}")
print("val_synth rows", len(val_synth))
