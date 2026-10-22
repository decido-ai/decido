# Parity run: verdict.predictor_hf:VerdictVLLM

- run: zs-gemma4-26b-a4b  (2026-10-18T01:30:18)
- limit_per_task: None (parity claims require full runs: limit_per_task = None)
- device: **GPU (intermediate run, --allow-gpu; NOT a CPU correctness number)**; device audit: cpu_only=False []

| group | this predictor | Jev 1.13 | delta | ECE10 | units | missing | invalid |
|---|---:|---:|---:|---:|---:|---:|---:|
| jevbench | 83.83 | 87.18 | -3.35 | 0.1342 | 231 | 0 | 0 |
| kev | 78.83 | 85.52 | -6.69 | 0.1841 | 6964 | 0 | 0 |
| openjev | 56.88 | 72.96 | -16.08 | 0.2056 | 77253 | 0 | 0 |
| nimble | 82.27 | 91.84 | -9.57 | 0.1707 | 324 | 0 | 0 |
| vitaminc | 73.62 | 78.46 | -4.84 | 0.2590 | 599 | 0 | 0 |
| massive | 86.86 | 87.14 | -0.28 | 0.1277 | 350 | 0 | 0 |

**Six-group mean: 77.05 vs Jev 1.13 83.85 (gap -6.80); accuracy parity: False; ECE parity: False; Brier parity: False**

| ECE check | this | Jev | ok |
|---|---:|---:|---|
| jevbench | 0.1342 | 0.0318 | False |
| kev:transfer-v4/development | 0.1392 | 0.0492 | False |
| kev:transfer-v9/development | 0.2233 | 0.0335 | False |
| vitaminc | 0.2590 | 0.1040 | False |
| massive | 0.1277 | 0.0750 | False |

| BRIER check | this | Jev | ok |
|---|---:|---:|---|
| jevbench | 0.2778 | 0.1811 | False |
| kev:transfer-v4/development | 0.3073 | 0.2110 | False |
| kev:transfer-v9/development | 0.4908 | 0.2121 | False |
| vitaminc | 0.5184 | 0.3130 | False |
| massive | 0.2546 | 0.2040 | False |

Bootstrap 95% CI of the six-group mean (stratified, B=500): 75.77-78.31; P(mean >= 83.85) = 0.000. Jev's own run-to-run variation is not included (Jev is nondeterministic; independent runs differ by up to 1.6 points on VitaminC).

Six-group mean if OpenJev had its 19th task (GSM8K rerank, 89.5-95.0): 77.34-77.38

## Option-permutation flip rate

```
{
 "per_suite": {
  "jevbench": {
   "targets": 139,
   "permutations_per_target": 3,
   "trials": 417,
   "flip_rate": 0.11031175059952038,
   "share_targets_any_flip": 0.1510791366906475,
   "mean_tv": 0.10101280749987425,
   "errors": 0
  },
  "kev": {
   "targets": 300,
   "permutations_per_target": 3,
   "trials": 900,
   "flip_rate": 0.17555555555555555,
   "share_targets_any_flip": 0.22333333333333333,
   "mean_tv": 0.16885148906374084,
   "errors": 0
  },
  "openjev": {
   "targets": 300,
   "permutations_per_target": 3,
   "trials": 900,
   "flip_rate": 0.04666666666666667,
   "share_targets_any_flip": 0.07333333333333333,
   "mean_tv": 0.044010734464746265,
   "errors": 0
  },
  "nimble": {
   "targets": 146,
   "permutations_per_target": 3,
   "trials": 438,
   "flip_rate": 0.0776255707762557,
   "share_targets_any_flip": 0.1506849315068493,
   "mean_tv": 0.0801413097303678,
   "errors": 0
  },
  "vitaminc": {
   "targets": 300,
   "permutations_per_target": 3,
   "trials": 900,
   "flip_rate": 0.05,
   "share_targets_any_flip": 0.09333333333333334,
   "mean_tv": 0.04737919558955703,
   "errors": 0
  },
  "massive": {
   "targets": 300,
   "permutations_per_target": 3,
   "trials": 900,
   "flip_rate": 0.04,
   "share_targets_any_flip": 0.07333333333333333,
   "mean_tv": 0.03804391949585396,
   "errors": 0
  }
 },
 "pooled_flip_rate": 0.08103254769921436,
 "pooled_trials": 4455
}
```

## Repeat-call determinism

```
{
 "requests": 100,
 "share_changed_gt_1e-6": 0.45,
 "share_argmax_changed": 0.01,
 "max_abs_dp": 0.19511144453520174
}
```
