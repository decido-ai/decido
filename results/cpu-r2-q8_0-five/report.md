# Parity run: verdict.predictor_llamacpp:VerdictLlamaCpp

- run: cpu-r2-q8_0-five  (2026-10-19T10:36:55)
- limit_per_task: None (parity claims require full runs: limit_per_task = None)
- device: **CPU-only**; device audit: cpu_only=True []

| group | this predictor | Jev 1.13 | delta | ECE10 | units | missing | invalid |
|---|---:|---:|---:|---:|---:|---:|---:|
| jevbench | 87.61 | 87.18 | +0.43 | 0.0381 | 231 | 0 | 0 |
| kev | 84.35 | 85.52 | -1.17 | 0.0119 | 6964 | 0 | 0 |
| nimble | 87.23 | 91.84 | -4.61 | 0.0503 | 324 | 0 | 0 |
| vitaminc | 86.14 | 78.46 | +7.68 | 0.0411 | 599 | 0 | 0 |
| massive | 88.57 | 87.14 | +1.43 | 0.0252 | 350 | 0 | 0 |

| ECE check | this | Jev | ok |
|---|---:|---:|---|
| jevbench | 0.0381 | 0.0318 | False |
| kev:transfer-v4/development | 0.0286 | 0.0492 | True |
| kev:transfer-v9/development | 0.0198 | 0.0335 | True |
| vitaminc | 0.0411 | 0.1040 | True |
| massive | 0.0252 | 0.0750 | True |

| BRIER check | this | Jev | ok |
|---|---:|---:|---|
| jevbench | 0.1689 | 0.1811 | True |
| kev:transfer-v4/development | 0.2102 | 0.2110 | True |
| kev:transfer-v9/development | 0.2670 | 0.2121 | False |
| vitaminc | 0.2021 | 0.3130 | True |
| massive | 0.1642 | 0.2040 | True |

## Option-permutation flip rate

```
{
 "per_suite": {
  "jevbench": {
   "targets": 100,
   "permutations_per_target": 3,
   "trials": 300,
   "flip_rate": 0.07,
   "share_targets_any_flip": 0.1,
   "mean_tv": 0.04276089927359547,
   "errors": 0
  },
  "kev": {
   "targets": 100,
   "permutations_per_target": 3,
   "trials": 300,
   "flip_rate": 0.1,
   "share_targets_any_flip": 0.13,
   "mean_tv": 0.06769633150879424,
   "errors": 0
  },
  "nimble": {
   "targets": 100,
   "permutations_per_target": 3,
   "trials": 300,
   "flip_rate": 0.023333333333333334,
   "share_targets_any_flip": 0.04,
   "mean_tv": 0.044877842445484206,
   "errors": 0
  },
  "vitaminc": {
   "targets": 100,
   "permutations_per_target": 3,
   "trials": 300,
   "flip_rate": 0.013333333333333334,
   "share_targets_any_flip": 0.02,
   "mean_tv": 0.025408904855114708,
   "errors": 0
  },
  "massive": {
   "targets": 100,
   "permutations_per_target": 3,
   "trials": 300,
   "flip_rate": 0.03333333333333333,
   "share_targets_any_flip": 0.07,
   "mean_tv": 0.04622709021532021,
   "errors": 0
  }
 },
 "pooled_flip_rate": 0.048,
 "pooled_trials": 1500
}
```

## Repeat-call determinism

```
{
 "requests": 50,
 "share_changed_gt_1e-6": 0.0,
 "share_argmax_changed": 0.0,
 "max_abs_dp": 0.0
}
```
