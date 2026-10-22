"""Render parity summary.json files as a markdown table.  python tools/results_table.py RUN [RUN...]"""
import json
import sys
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "eval" / "parity" / "results"
SIX = ["jevbench", "kev", "openjev", "nimble", "vitaminc", "massive"]
JEV = {"jevbench": 87.18, "kev": 85.52, "openjev": 72.96, "nimble": 91.84, "vitaminc": 78.46, "massive": 87.14}
rows = ["| run | device | " + " | ".join(SIX) + " | six-mean | gap | 95% CI | Brier ok | ECE ok | flip | det (changed / argmax) |",
        "|---|---|" + "---:|" * (len(SIX) + 7)]
rows.append("| **Jev 1.13** | API | " + " | ".join(f"{JEV[s]:.2f}" for s in SIX) + " | 83.85 | | | | | 14.6% (K=64) | 4.3% |")
for name in sys.argv[1:]:
    s = json.loads((R / name / "summary.json").read_text())
    p = s.get("parity", {})
    dev = "GPU" if "GPU" in str(s["device_audit"].get("device", "")) else "CPU"
    sc = [f"{s['suites'][x]['score']:.2f}" if x in s["suites"] else "-" for x in SIX]
    ci = p.get("bootstrap95", {}).get("six_group_mean")
    bok = sum(c["ok"] for c in p.get("brier_checks", {}).values())
    eok = sum(c["ok"] for c in p.get("ece_checks", {}).values())
    fl = s.get("permutation", {}).get("pooled_flip_rate")
    d = s.get("determinism", {})
    m = p.get("six_group_mean")
    rows.append(f"| {name} | {dev} | " + " | ".join(sc) + f" | {m:.2f} | {p['gap']:+.2f} | "
                f"{ci[0]:.2f}-{ci[1]:.2f} | {bok}/{len(p.get('brier_checks', {}))} | {eok}/{len(p.get('ece_checks', {}))} | "
                f"{fl:.1%} | {d.get('share_changed_gt_1e-6', 0):.0%} / {d.get('share_argmax_changed', 0):.0%} |" if m is not None else f"| {name} | incomplete |")
print("\n".join(rows))
