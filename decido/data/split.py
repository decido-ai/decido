"""Task-held-out splits from the leakage-gated pool.

    python -m decido.data.split [--train-questions 400000] [--seed 0]

  data/val.jsonl          whole held-out TASKS (never in train/calib), ~3k questions
  data/calib.jsonl        ~3k questions (>= ~1k per kind) from seen tasks, disjoint groups from train
  data/train_pool.jsonl   seen tasks, ~400k questions, kind mix steered towards 45/25/30 choice/noul/score
  data/split_stats.json   counts, kind mix, per-source, option-count histogram, state-token p50/p95

Group integrity: every row sharing a meta.group_id lands in the same split (coherence pairs, NLI/choice/verify
recasts of one MC item, multi-question packs).
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import random

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CLEAN = os.path.join(ROOT, "data", "pool_clean")
OUT = os.path.join(ROOT, "data")

# Whole tasks held out for val (zero-shot transfer within seen families + a few unseen sub-families).
HELD_OUT_TASKS_EXACT = {
    "openbookqa/choice", "openbookqa/verify", "openbookqa/nli_rerank",      # knowledge MC: unseen dataset
    "clinc/intent",                                                       # routing with a new label set
    "goemotions/emotion", "mhs/ordinal",                                                 # soft-label emotion
    "proc/graph_navigation", "proc/goal_directed_planning",               # procedural configs (data.md T2)
    "vproc/posterior/dice", "vproc/extraction",                           # own generators: unseen sub-families
}
HELD_OUT_TS_FRACTION = 0.03      # random tasksource sources held out entirely
HELD_OUT_OPENJEV = 2             # Open-Jev sources held out entirely
WEIGHTS = {"knowledge": 2.0, "verdict-procedural": 3.0, "procedural-typed": 2.0, "soft": 2.0, "nli": 1.5}
KIND_TARGET = {"choice": 0.45, "noul": 0.25, "score": 0.30}


def _w(row):
    m = row["meta"]
    s = m["source"]
    if m["family"].startswith("knowledge"):
        return WEIGHTS["knowledge"]
    if s in ("verdict-procedural", "procedural-typed"):
        return WEIGHTS[s]
    if m["label_type"] == "soft_human":
        return WEIGHTS["soft"]
    if m["family"] == "nli":
        return WEIGHTS["nli"]
    return 1.0


def kinds(row):
    return collections.Counter(q["type"] for q in row["questions"].values())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-questions", type=int, default=400000)
    ap.add_argument("--val-questions", type=int, default=3000)
    ap.add_argument("--calib-per-kind", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    rng = random.Random(a.seed)
    rows = []
    for f in sorted(glob.glob(os.path.join(CLEAN, "*.jsonl"))):
        with open(f) as fh:
            rows.extend(json.loads(line) for line in fh)
    tasks = collections.Counter(r["meta"]["task_id"] for r in rows)
    ts_tasks = sorted(t for t in tasks if t.startswith("ts/"))
    oj_tasks = sorted(t for t in tasks if t.startswith(("open-jev/", "open-jev-v1.1/")))
    held = {t for t in HELD_OUT_TASKS_EXACT if t in tasks}
    held |= set(rng.sample(ts_tasks, max(1, int(len(ts_tasks) * HELD_OUT_TS_FRACTION))))
    held |= set(rng.sample(oj_tasks, min(HELD_OUT_OPENJEV, len(oj_tasks))))
    # groups that straddle a held-out and a seen task go entirely to held-out
    groups = collections.defaultdict(list)
    for r in rows:
        groups[r["meta"]["group_id"]].append(r)
    held_groups, seen_groups = [], []
    for gid, rs in groups.items():
        (held_groups if any(r["meta"]["task_id"] in held for r in rs) else seen_groups).append(gid)

    # ---- val: sample held-out groups, per-task cap
    rng.shuffle(held_groups)
    per_task_cap = max(60, a.val_questions // max(1, len(held)) * 2)
    val, vq, vtask = [], 0, collections.Counter()
    for gid in held_groups:
        rs = groups[gid]
        t = rs[0]["meta"]["task_id"]
        nq = sum(len(r["questions"]) for r in rs)
        if vtask[t] + nq > per_task_cap:
            continue
        val.extend(rs)
        vtask[t] += nq
        vq += nq
        if vq >= a.val_questions:
            break

    # ---- calib: seen groups until each kind has >= calib_per_kind questions
    rng.shuffle(seen_groups)
    calib, cq = [], collections.Counter()
    calib_ids = set()
    for gid in seen_groups:
        if all(cq[k] >= a.calib_per_kind for k in KIND_TARGET):
            break
        rs = groups[gid]
        kc = sum((kinds(r) for r in rs), collections.Counter())
        if not any(cq[k] < a.calib_per_kind for k in kc):
            continue
        calib.extend(rs)
        cq.update(kc)
        calib_ids.add(gid)

    # ---- train: weighted sampling without replacement (Efraimidis-Spirakis) under kind caps
    caps = {k: int(a.train_questions * v * 1.02) for k, v in KIND_TARGET.items()}
    cand = []
    for gid in seen_groups:
        if gid in calib_ids:
            continue
        rs = groups[gid]
        w = max(_w(r) for r in rs)
        cand.append((rng.random() ** (1.0 / w), gid))
    cand.sort(reverse=True)
    train, tq = [], collections.Counter()
    for _, gid in cand:
        rs = groups[gid]
        kc = sum((kinds(r) for r in rs), collections.Counter())
        if any(tq[k] + n > caps[k] for k, n in kc.items()):
            continue
        train.extend(rs)
        tq.update(kc)
        if sum(tq.values()) >= a.train_questions:
            break
    rng.shuffle(train)

    for name, rs in (("train_pool", train), ("val", val), ("calib", calib)):
        split = {"train_pool": "train", "val": "val_heldout_task", "calib": "calib"}[name]
        with open(os.path.join(OUT, f"{name}.jsonl"), "w") as fh:
            for r in rs:
                r["meta"]["split"] = split
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---- stats
    def stats(rs):
        kc = collections.Counter()
        src = collections.Counter()
        opt = collections.Counter()
        for r in rs:
            for q in r["questions"].values():
                kc[q["type"]] += 1
                src[r["meta"]["source"]] += 1
                n = 2 if q["type"] == "noul" else len(q["criteria"])
                opt["2" if n == 2 else "3-5" if n <= 5 else "6-10" if n <= 10 else "11-26" if n <= 26 else "27+"] += 1
        tot = sum(kc.values())
        return {"rows": len(rs), "questions": tot, "kinds": dict(kc),
                "kind_share": {k: round(v / tot, 3) for k, v in kc.items()} if tot else {},
                "per_source_questions": dict(src.most_common()), "option_count_hist": dict(opt),
                "tasks": len({r["meta"]["task_id"] for r in rs})}
    st = {"held_out_tasks": sorted(held), "train_pool": stats(train), "val": stats(val), "calib": stats(calib)}
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-30B-A3B-Instruct-2507")
        smp = rng.sample(train, min(20000, len(train)))
        L = sorted(len(tok.encode(r["state"] if isinstance(r["state"], str) else json.dumps(r["state"], ensure_ascii=False, indent=1),
                                  add_special_tokens=False)) for r in smp)
        st["train_state_tokens_qwen3"] = {"p50": L[len(L) // 2], "p95": L[int(len(L) * 0.95)], "max": L[-1], "n": len(L)}
    except Exception as e:  # tokenizer unavailable
        st["train_state_tokens_qwen3"] = f"unavailable: {e}"
    with open(os.path.join(OUT, "split_stats.json"), "w") as fh:
        json.dump(st, fh, indent=1)
    print(json.dumps({k: (v if k == "train_state_tokens_qwen3" else {kk: v[kk] for kk in ("rows", "questions", "kind_share", "tasks", "option_count_hist")})
                      for k, v in st.items() if k != "held_out_tasks"}, indent=1))
    print("held-out tasks:", len(held))


if __name__ == "__main__":
    main()
