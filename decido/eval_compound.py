"""Compound-question eval: accuracy asking the compound question directly vs answering its atoms separately.

    python -m decido.eval_compound --predictor decido.predictor_hf:VerdictVLLM --predictor-kw model=... [--allow-gpu]
Writes eval/compound/results/<name>.json. (Held-out schemas; Jev reference from the teardown: phishing 62.6%
as one question vs 95.0% split into five; jev-kit compound 46.7% - different items, indicative only.)
"""
import argparse
import importlib
import inspect
import json
import os
import time

OPS = {"all": all, "any": any, "exactly_one": lambda v: sum(v) == 1, "none": lambda v: not any(v),
       "but_not": lambda v: v[0] and not v[1]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictor", required=True)
    ap.add_argument("--predictor-kw", action="append", default=[])
    ap.add_argument("--data", default="eval/compound/compound_eval.jsonl")
    ap.add_argument("--name", required=True)
    ap.add_argument("--allow-gpu", action="store_true")
    a = ap.parse_args()
    if not a.allow_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    kw = {}
    for kv in a.predictor_kw:
        k, _, v = kv.partition("=")
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            pass
        kw[k] = v
    mod, _, attr = a.predictor.partition(":")
    obj = getattr(importlib.import_module(mod), attr)
    pred = obj(**kw) if inspect.isclass(obj) else obj
    items = [json.loads(x) for x in open(a.data)]
    reqs = []
    for it in items:
        qs = {"compound": {"type": "noul", "instructions": it["compound"]}}
        reqs.append({"state": it["state"], "questions": qs})
        reqs.append({"state": it["state"], "questions": {f"a{i}": {"type": "noul", "instructions": at["q"]}
                                                        for i, at in enumerate(it["atoms"])}})
    t0 = time.time()
    outs = pred.predict_batch(reqs) if hasattr(pred, "predict_batch") else [pred(r) for r in reqs]
    res = {"n": len(items), "compound_acc": 0, "split_acc": 0, "atom_acc": 0, "atoms": 0, "by_op": {}}
    for i, it in enumerate(items):
        c = outs[2 * i]["answers"]["compound"]["noul"] >= 0.5
        av = [outs[2 * i + 1]["answers"][f"a{j}"]["noul"] >= 0.5 for j in range(len(it["atoms"]))]
        s = OPS[it["op"]](av)
        res["compound_acc"] += c == it["label"]
        res["split_acc"] += s == it["label"]
        res["atom_acc"] += sum(x == at["label"] for x, at in zip(av, it["atoms"]))
        res["atoms"] += len(av)
        o = res["by_op"].setdefault(it["op"], {"n": 0, "compound": 0, "split": 0})
        o["n"] += 1
        o["compound"] += c == it["label"]
        o["split"] += s == it["label"]
    res["compound_acc"] /= len(items)
    res["split_acc"] /= len(items)
    res["atom_acc"] /= res["atoms"]
    for o in res["by_op"].values():
        o["compound"] /= o["n"]
        o["split"] /= o["n"]
    res["seconds"] = time.time() - t0
    res["predictor"], res["kw"] = a.predictor, kw
    os.makedirs("eval/compound/results", exist_ok=True)
    json.dump(res, open(f"eval/compound/results/{a.name}.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
