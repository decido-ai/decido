"""Fit per-kind temperatures for the student on calib.jsonl ONLY (never on eval suites).

    python -m decido.fit_temps --calib data/calib.jsonl --pred runs/label/calib.<student>.jsonl --out temps.json
`pred` comes from `python -m decido.label --model <merged student> --in data/calib.jsonl` (raw, T=1).
"""
import argparse
import json

from decido.data.mix_targets import fit_temps, load_preds

ap = argparse.ArgumentParser()
ap.add_argument("--calib", required=True)
ap.add_argument("--pred", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
rows = [json.loads(x) for x in open(a.calib)]
temps = fit_temps(rows, load_preds(a.pred))
json.dump(temps, open(a.out, "w"), indent=1)
print(temps)
