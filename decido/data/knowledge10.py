"""Round-3 data: knowledge multiple choice expanded to 10 options (train splits only, exact gold labels).

    python -m decido.data.knowledge10 --out data/knowledge10_raw.jsonl
Takes single-answer `choice` knowledge questions already in our training data (data/train.jsonl knowledge sources +
data/knowledge2.jsonl choice rows), and adds distractors drawn from the answers of OTHER questions of the same source
(and same subject when available) until there are 10 options; options are shuffled. Targets: one-hot on the gold.
"""
import argparse
import collections
import hashlib
import json
import random

KNOW = {"mmlu_aux", "arc", "commonsense_qa", "medmcqa", "aquarat", "qasc", "quartz", "cosmosqa", "hellaswag", "winogrande"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=31)
    a = ap.parse_args()
    r = random.Random(a.seed)
    items = []
    for f in ("data/train.jsonl", "data/knowledge2.jsonl"):
        for x in open(f):
            row = json.loads(x)
            src = row["meta"]["source"]
            if not any(src.startswith(k) for k in KNOW):
                continue
            for qid, q in row["questions"].items():
                if q["type"] != "choice" or qid not in row["targets"]:
                    continue
                t = row["targets"][qid]
                keys = list(q["criteria"].keys())
                if max(t) < 0.99 or not (3 <= len(keys) <= 8):
                    continue
                gold = keys[t.index(max(t))]
                texts = {k: (q["criteria"][k] if q["criteria"][k] is not None else k) for k in keys}
                if any(not isinstance(v, str) for v in texts.values()):
                    continue
                subj = row["meta"].get("subject") or row["meta"].get("task_id", "").split("/")[-1]
                items.append({"src": src, "subj": subj, "state": row["state"], "instr": q["instructions"],
                              "gold": texts[gold], "others": [texts[k] for k in keys if k != gold]})
    by_src = collections.defaultdict(list)
    by_subj = collections.defaultdict(list)
    for it in items:
        by_src[it["src"]].append(it["gold"])
        by_subj[(it["src"], it["subj"])].append(it["gold"])
    r.shuffle(items)
    n = 0
    with open(a.out, "w") as fh:
        for it in items:
            if n >= a.n:
                break
            opts = [it["gold"]] + it["others"]
            pool = by_subj[(it["src"], it["subj"])] if len(by_subj[(it["src"], it["subj"])]) > 50 else by_src[it["src"]]
            tries = 0
            while len(opts) < 10 and tries < 100:
                c = r.choice(pool)
                tries += 1
                if c not in opts and len(c) < 400:
                    opts.append(c)
            if len(opts) < 10:
                continue
            r.shuffle(opts)
            letters = [f"option_{i + 1}" for i in range(10)]
            crit = {letters[i]: opts[i] for i in range(10)}
            tgt = [1.0 if o == it["gold"] else 0.0 for o in opts]
            qs = {"answer": {"type": "choice", "instructions": it["instr"], "criteria": crit}}
            rid = hashlib.sha1(json.dumps([it["state"], qs], sort_keys=True, default=str).encode()).hexdigest()[:20]
            fh.write(json.dumps({"request_id": "k10-" + rid, "state": it["state"], "questions": qs, "targets": {"answer": tgt},
                                 "meta": {"source": "knowledge10-" + it["src"], "task_id": f"knowledge10/{it['src']}", "family": "knowledge_mc",
                                          "label_type": "hard", "license": "as source", "license_use": "commercial", "split": "train",
                                          "generator": "distractor-augmented", "group_id": rid}}, ensure_ascii=False) + "\n")
            n += 1
    print("candidates", len(items), "written", n)


if __name__ == "__main__":
    main()
