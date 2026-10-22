"""Round-2 knowledge expansion (TRAIN splits only, permissive licences) recast with build_pool.knowledge_rows.

    python -m decido.data.knowledge2 --out data/knowledge2_raw.jsonl
Sources: medmcqa (apache-2.0), AQuA-RAT (apache-2.0), QASC (cc-by-4.0), QuaRTz (cc-by-4.0), CosmosQA (cc-by-4.0).
Formats (identical to round-1 knowledge rows): choice question, noul "is X the correct answer" checks, and NLI-rerank
requests in the exact eval NLI request shape (premise = question, hypothesis = templated option).
"""
import argparse
import collections
import json
import random
import re

from decido.data.build_pool import DEFAULT_HYP, knowledge_rows, read_parquets, valid_row

BAD_OPT = re.compile(r"\b(all of the above|none of the above|all the above|none of these|both (a|b|of)|all of these)\b", re.I)


def _ok(opts):
    return len(opts) >= 2 and len(set(o.lower() for o in opts)) == len(opts) and all(o.strip() for o in opts) \
        and not any(BAD_OPT.search(o) for o in opts)


def items(rng, n_med=14000, n_aqua=8000, n_cosmos=7000):
    out = []
    t = read_parquets("openlifescienceai__medmcqa/data/train-*")
    t = t[(t.choice_type == "single") & t.cop.isin([0, 1, 2, 3])]
    by_subj = collections.defaultdict(list)
    for r in t.itertuples(index=False):
        opts = [str(r.opa).strip(), str(r.opb).strip(), str(r.opc).strip(), str(r.opd).strip()]
        q = str(r.question).strip()
        if len(q) < 15 or not _ok(opts):
            continue
        by_subj[r.subject_name].append(("medmcqa", r.id, q, opts, int(r.cop), DEFAULT_HYP.format, "apache-2.0", "commercial"))
    per = max(1, n_med // max(1, len(by_subj)))
    pool = []
    for subj, its in by_subj.items():   # subject-balanced (cap per subject, then fill)
        rng.shuffle(its)
        out.extend(its[:per])
        pool.extend(its[per:])
    rng.shuffle(pool)
    out.extend(pool[:max(0, n_med - sum(min(len(v), per) for v in by_subj.values()))])
    t = read_parquets("deepmind__aqua_rat/raw/train-*")
    aq = []
    for i, r in enumerate(t.itertuples(index=False)):
        opts = [re.sub(r"^[A-E]\)\s*", "", str(o)).strip() for o in r.options]
        letters = [str(o)[0] for o in r.options]
        if r.correct not in letters or not _ok(opts):
            continue
        aq.append(("aqua_rat", f"train/{i}", str(r.question).strip(), opts, letters.index(r.correct), DEFAULT_HYP.format,
                   "apache-2.0", "commercial"))
    rng.shuffle(aq)
    out.extend(aq[:n_aqua])
    t = read_parquets("allenai__qasc/data/train-*")
    for r in t.itertuples(index=False):
        labs, opts = list(r.choices["label"]), [str(x).strip() for x in r.choices["text"]]
        if r.answerKey in labs and _ok(opts):
            out.append(("qasc", r.id, str(r.question).strip(), opts, labs.index(r.answerKey), DEFAULT_HYP.format,
                        "cc-by-4.0", "commercial"))
    t = read_parquets("allenai__quartz/data/train-*")
    for r in t.itertuples(index=False):
        labs, opts = list(r.choices["label"]), [str(x).strip() for x in r.choices["text"]]
        if r.answerKey in labs and _ok(opts):
            q = str(r.question).strip()
            out.append(("quartz", r.id, f"{str(r.para).strip()}\n\n{q}", opts, labs.index(r.answerKey),
                        (lambda o, q=q: f"{q} {o}"), "cc-by-4.0", "commercial"))
    t = read_parquets("allenai__cosmos_qa/default/train/*")
    cq = []
    for r in t.itertuples(index=False):
        opts = [str(r.answer0).strip(), str(r.answer1).strip(), str(r.answer2).strip(), str(r.answer3).strip()]
        if int(r.label) in (0, 1, 2, 3) and _ok(opts):
            cq.append(("cosmos_qa", r.id, f"{str(r.context).strip()}\n\nQuestion: {str(r.question).strip()}", opts,
                       int(r.label), DEFAULT_HYP.format, "cc-by-4.0", "commercial"))
    rng.shuffle(cq)
    out.extend(cq[:n_cosmos])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--n-nli-items", type=int, default=7000)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    its = items(rng)
    print("items per source:", dict(collections.Counter(i[0] for i in its)))
    ch, nu, nl = knowledge_rows(rng, n_nli_items=a.n_nli_items, items=its)
    rows = [r for r in ch + nu + nl if valid_row(r)]
    for r in rows:
        r["meta"]["generator"] = "recast"
        r["meta"]["round"] = "knowledge2"
    ids = collections.Counter(r["request_id"] for r in rows)
    dup = [k for k, v in ids.items() if v > 1]
    seen = set()
    with open(a.out, "w") as fh:
        for r in rows:
            if r["request_id"] in seen:
                continue
            seen.add(r["request_id"])
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"rows {len(seen)} (dup ids dropped {sum(ids[k] - 1 for k in dup)}); choice {len(ch)} noul {len(nu)} nli {len(nl)}")


if __name__ == "__main__":
    main()
