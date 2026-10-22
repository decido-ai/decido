# Parity run: verdict.predictor_hf:VerdictVLLM

- run: zs-qwen3-30b-a3b  (2026-10-18T01:25:22)
- limit_per_task: None (parity claims require full runs: limit_per_task = None)
- device: **GPU (intermediate run, --allow-gpu; NOT a CPU correctness number)**; device audit: cpu_only=False []

| group | this predictor | Jev 1.13 | delta | ECE10 | units | missing | invalid |
|---|---:|---:|---:|---:|---:|---:|---:|
| jevbench | 77.88 | 87.18 | -9.30 | 0.1918 | 231 | 0 | 0 |
| kev | 75.86 | 85.52 | -9.66 | 0.2180 | 6964 | 0 | 0 |
| openjev | 64.40 | 72.96 | -8.56 | 0.1832 | 77253 | 0 | 0 |
| nimble | 76.60 | 91.84 | -15.24 | 0.2199 | 324 | 0 | 0 |
| vitaminc | 70.28 | 78.46 | -8.18 | 0.2655 | 599 | 0 | 0 |
| massive | 85.71 | 87.14 | -1.43 | 0.1312 | 350 | 0 | 0 |

**Six-group mean: 75.12 vs Jev 1.13 83.85 (gap -8.73); accuracy parity: False; ECE parity: False; Brier parity: False**

| ECE check | this | Jev | ok |
|---|---:|---:|---|
| jevbench | 0.1918 | 0.0318 | False |
| kev:transfer-v4/development | 0.2165 | 0.0492 | False |
| kev:transfer-v9/development | 0.2445 | 0.0335 | False |
| vitaminc | 0.2655 | 0.1040 | False |
| massive | 0.1312 | 0.0750 | False |

| BRIER check | this | Jev | ok |
|---|---:|---:|---|
| jevbench | 0.3829 | 0.1811 | False |
| kev:transfer-v4/development | 0.4409 | 0.2110 | False |
| kev:transfer-v9/development | 0.5040 | 0.2121 | False |
| vitaminc | 0.5498 | 0.3130 | False |
| massive | 0.2658 | 0.2040 | False |

Bootstrap 95% CI of the six-group mean (stratified, B=500): 73.73-76.55; P(mean >= 83.85) = 0.000. Jev's own run-to-run variation is not included (Jev is nondeterministic; independent runs differ by up to 1.6 points on VitaminC).

Six-group mean if OpenJev had its 19th task (GSM8K rerank, 89.5-95.0): 75.34-75.39

## Option-permutation flip rate

```
{
 "per_suite": {
  "jevbench": {
   "targets": 139,
   "permutations_per_target": 3,
   "trials": 417,
   "flip_rate": 0.11270983213429256,
   "share_targets_any_flip": 0.18705035971223022,
   "mean_tv": 0.10503991975053273,
   "errors": 0
  },
  "kev": {
   "targets": 300,
   "permutations_per_target": 3,
   "trials": 900,
   "flip_rate": 0.11888888888888889,
   "share_targets_any_flip": 0.17666666666666667,
   "mean_tv": 0.12407097400662852,
   "errors": 0
  },
  "openjev": {
   "targets": 300,
   "permutations_per_target": 3,
   "trials": 900,
   "flip_rate": 0.043333333333333335,
   "share_targets_any_flip": 0.08,
   "mean_tv": 0.047930919018589736,
   "errors": 0
  },
  "nimble": {
   "targets": 146,
   "permutations_per_target": 3,
   "trials": 438,
   "flip_rate": 0.1780821917808219,
   "share_targets_any_flip": 0.2945205479452055,
   "mean_tv": 0.1682368486364367,
   "errors": 0
  },
  "vitaminc": {
   "targets": 300,
   "permutations_per_target": 3,
   "trials": 900,
   "flip_rate": 0.06111111111111111,
   "share_targets_any_flip": 0.09,
   "mean_tv": 0.05965589951995214,
   "errors": 0
  },
  "massive": {
   "targets": 300,
   "permutations_per_target": 3,
   "trials": 900,
   "flip_rate": 0.07111111111111111,
   "share_targets_any_flip": 0.11,
   "mean_tv": 0.0666636793219622,
   "errors": 0
  }
 },
 "pooled_flip_rate": 0.08754208754208755,
 "pooled_trials": 4455
}
```

## Repeat-call determinism

```
{
 "requests": 100,
 "share_changed_gt_1e-6": 0.48,
 "share_argmax_changed": 0.0,
 "max_abs_dp": 0.06028367405265683
}
```
