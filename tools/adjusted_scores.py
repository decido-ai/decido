"""OpenJev / six-group mean with eval items that have near-duplicate training paraphrases EXCLUDED (Phase G #1b).
    python tools/adjusted_scores.py RUN [RUN...]
Flagged eval items (research/verify-leakage-r23.md): MMLU test 9589, 7287, 675, 1303, 1161 (both mmlu and mmlu_fewshot
variants) and ARC-E 265, 430, 594, 1813, 1842, 2039 / ARC-C 127, 684, 805.
"""
import json
import sys
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[1] / "eval" / "parity" / "results"
FLAG = {("mmlu", i) for i in (9589, 7287, 675, 1303, 1161)} | {("mmlu_fewshot", i) for i in (9589, 7287, 675, 1303, 1161)} | \
       {("arc_easy", i) for i in (265, 430, 594, 1813, 1842, 2039)} | {("arc_challenge", i) for i in (127, 684, 805)}


def item_index(uid):
    try:
        return int(uid.rsplit("/", 1)[-1])
    except ValueError:
        return None


for name in sys.argv[1:]:
    s = json.loads((R / name / "summary.json").read_text())
    units = [json.loads(x) for x in open(R / name / "units.jsonl") if '"openjev"' in x]
    per = dict(s["suites"]["openjev"]["per_task"])
    dropped = 0
    for t in ("mmlu", "mmlu_fewshot", "arc_easy", "arc_challenge"):
        us = [u for u in units if u["task"] == t and not u.get("missing")]
        keep = [u for u in us if (t, item_index(u["uid"])) not in FLAG]
        dropped += len(us) - len(keep)
        per[t] = 100 * float(np.mean([bool(u.get("correct")) for u in keep]))
    oj_adj = float(np.mean(list(per.values())))
    six = {k: v["score"] for k, v in s["suites"].items() if k in ("jevbench", "kev", "openjev", "nimble", "vitaminc", "massive")}
    out = {"run": name, "openjev_official": round(six["openjev"], 3), "openjev_adjusted": round(oj_adj, 3), "units_dropped": dropped}
    if len(six) == 6:
        six["openjev"] = oj_adj
        out["six_mean_adjusted"] = round(float(np.mean(list(six.values()))), 3)
        out["six_mean_official"] = round(s["parity"]["six_group_mean"], 3)
    print(json.dumps(out))
