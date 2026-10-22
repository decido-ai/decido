# Parity run: eval.parity.predictors:oracle_predictor

- run: sanity-oracle  (2026-10-15T22:33:49)
- limit_per_task: None (parity claims require full runs: limit_per_task = None)
- device audit: cpu_only=True []

| group | this predictor | Jev 1.13 | delta | ECE10 | units | missing | invalid |
|---|---:|---:|---:|---:|---:|---:|---:|
| jevbench | 100.00 | 87.18 | +12.82 | 0.0000 | 231 | 0 | 0 |
| kev | 100.00 | 85.52 | +14.48 | 0.0000 | 6964 | 0 | 0 |
| openjev | 100.00 | 72.96 | +27.04 | 0.0000 | 77253 | 0 | 0 |
| nimble | 100.00 | 91.84 | +8.16 | 0.0000 | 324 | 0 | 0 |
| vitaminc | 100.00 | 78.46 | +21.54 | 0.0000 | 599 | 0 | 0 |
| massive | 100.00 | 87.14 | +12.86 | 0.0000 | 350 | 0 | 0 |

**Six-group mean: 100.00 vs Jev 1.13 83.85 (gap +16.15); accuracy parity: True; ECE parity: True; Brier parity: True**

| ECE check | this | Jev | ok |
|---|---:|---:|---|
| jevbench | 0.0000 | 0.0318 | True |
| kev:transfer-v4/development | 0.0000 | 0.0492 | True |
| kev:transfer-v9/development | 0.0000 | 0.0335 | True |
| vitaminc | 0.0000 | 0.1040 | True |
| massive | 0.0000 | 0.0750 | True |

| BRIER check | this | Jev | ok |
|---|---:|---:|---|
| jevbench | 0.0000 | 0.1811 | True |
| kev:transfer-v4/development | 0.0000 | 0.2110 | True |
| kev:transfer-v9/development | 0.0000 | 0.2121 | True |
| vitaminc | 0.0000 | 0.3130 | True |
| massive | 0.0000 | 0.2040 | True |

Bootstrap 95% CI of the six-group mean (stratified, B=500): 100.00-100.00; P(mean >= 83.85) = 1.000. Jev's own run-to-run variation is not included (Jev is nondeterministic; independent runs differ by up to 1.6 points on VitaminC).

Six-group mean if OpenJev had its 19th task (GSM8K rerank, 89.5-95.0): 99.91-99.96

## Option-permutation flip rate

```
{
 "per_suite": {
  "jevbench": {
   "targets": 139,
   "permutations_per_target": 3,
   "trials": 417,
   "flip_rate": 0.0,
   "share_targets_any_flip": 0.0,
   "mean_tv": 0.0,
   "errors": 0
  },
  "kev": {
   "targets": 200,
   "permutations_per_target": 3,
   "trials": 600,
   "flip_rate": 0.0,
   "share_targets_any_flip": 0.0,
   "mean_tv": 0.0,
   "errors": 0
  },
  "openjev": {
   "targets": 200,
   "permutations_per_target": 3,
   "trials": 600,
   "flip_rate": 0.0,
   "share_targets_any_flip": 0.0,
   "mean_tv": 0.0,
   "errors": 0
  },
  "nimble": {
   "targets": 146,
   "permutations_per_target": 3,
   "trials": 438,
   "flip_rate": 0.0,
   "share_targets_any_flip": 0.0,
   "mean_tv": 0.0,
   "errors": 0
  },
  "vitaminc": {
   "targets": 200,
   "permutations_per_target": 3,
   "trials": 600,
   "flip_rate": 0.0,
   "share_targets_any_flip": 0.0,
   "mean_tv": 0.0,
   "errors": 0
  },
  "massive": {
   "targets": 200,
   "permutations_per_target": 3,
   "trials": 600,
   "flip_rate": 0.0,
   "share_targets_any_flip": 0.0,
   "mean_tv": 0.0,
   "errors": 0
  }
 },
 "pooled_flip_rate": 0.0,
 "pooled_trials": 3255
}
```

## Repeat-call determinism

```
{
 "requests": 100,
 "share_changed_gt_1e-6": 0.0,
 "share_argmax_changed": 0.0,
 "max_abs_dp": 0.0
}
```
