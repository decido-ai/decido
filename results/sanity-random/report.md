# Parity run: eval.parity.predictors:random_predictor

- run: sanity-random  (2026-10-15T22:33:29)
- limit_per_task: None (parity claims require full runs: limit_per_task = None)
- device audit: cpu_only=True []

| group | this predictor | Jev 1.13 | delta | ECE10 | units | missing | invalid |
|---|---:|---:|---:|---:|---:|---:|---:|
| jevbench | 39.50 | 87.18 | -47.68 | 0.2097 | 231 | 0 | 0 |
| kev | 35.47 | 85.52 | -50.05 | 0.2431 | 6964 | 0 | 0 |
| openjev | 30.22 | 72.96 | -42.74 | 0.2726 | 77253 | 0 | 0 |
| nimble | 35.46 | 91.84 | -56.38 | 0.2656 | 324 | 0 | 0 |
| vitaminc | 36.56 | 78.46 | -41.90 | 0.2458 | 599 | 0 | 0 |
| massive | 7.43 | 87.14 | -79.71 | 0.1168 | 350 | 0 | 0 |

**Six-group mean: 30.77 vs Jev 1.13 83.85 (gap -53.08); accuracy parity: False; ECE parity: False; Brier parity: False**

| ECE check | this | Jev | ok |
|---|---:|---:|---|
| jevbench | 0.2097 | 0.0318 | False |
| kev:transfer-v4/development | 0.2600 | 0.0492 | False |
| kev:transfer-v9/development | 0.2555 | 0.0335 | False |
| vitaminc | 0.2458 | 0.1040 | False |
| massive | 0.1168 | 0.0750 | False |

| BRIER check | this | Jev | ok |
|---|---:|---:|---|
| jevbench | 0.7882 | 0.1811 | False |
| kev:transfer-v4/development | 0.7693 | 0.2110 | False |
| kev:transfer-v9/development | 0.8151 | 0.2121 | False |
| vitaminc | 0.8044 | 0.3130 | False |
| massive | 0.9809 | 0.2040 | False |

Bootstrap 95% CI of the six-group mean (stratified, B=500): 29.12-32.57; P(mean >= 83.85) = 0.000. Jev's own run-to-run variation is not included (Jev is nondeterministic; independent runs differ by up to 1.6 points on VitaminC).

Six-group mean if OpenJev had its 19th task (GSM8K rerank, 89.5-95.0): 31.29-31.34

## Option-permutation flip rate

```
{
 "per_suite": {
  "jevbench": {
   "targets": 139,
   "permutations_per_target": 3,
   "trials": 417,
   "flip_rate": 0.8201438848920863,
   "share_targets_any_flip": 0.9928057553956835,
   "mean_tv": 0.392326022787437,
   "errors": 0
  },
  "kev": {
   "targets": 200,
   "permutations_per_target": 3,
   "trials": 600,
   "flip_rate": 0.8616666666666667,
   "share_targets_any_flip": 1.0,
   "mean_tv": 0.4302005491684368,
   "errors": 0
  },
  "openjev": {
   "targets": 200,
   "permutations_per_target": 3,
   "trials": 600,
   "flip_rate": 0.8183333333333334,
   "share_targets_any_flip": 0.995,
   "mean_tv": 0.41263099614894355,
   "errors": 0
  },
  "nimble": {
   "targets": 146,
   "permutations_per_target": 3,
   "trials": 438,
   "flip_rate": 0.7785388127853882,
   "share_targets_any_flip": 1.0,
   "mean_tv": 0.4110903009661065,
   "errors": 0
  },
  "vitaminc": {
   "targets": 200,
   "permutations_per_target": 3,
   "trials": 600,
   "flip_rate": 0.8,
   "share_targets_any_flip": 0.985,
   "mean_tv": 0.4097042338308039,
   "errors": 0
  },
  "massive": {
   "targets": 200,
   "permutations_per_target": 3,
   "trials": 600,
   "flip_rate": 0.945,
   "share_targets_any_flip": 1.0,
   "mean_tv": 0.470397445440477,
   "errors": 0
  }
 },
 "pooled_flip_rate": 0.8411674347158218,
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
