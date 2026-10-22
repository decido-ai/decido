# Parity run: verdict.predictor_hf:VerdictVLLM

- run: gpu-r2  (2026-10-19T10:31:06)
- limit_per_task: None (parity claims require full runs: limit_per_task = None)
- device: **GPU (intermediate run, --allow-gpu; NOT a CPU correctness number)**; device audit: cpu_only=False []

| group | this predictor | Jev 1.13 | delta | ECE10 | units | missing | invalid |
|---|---:|---:|---:|---:|---:|---:|---:|
| jevbench | 88.17 | 87.18 | +0.99 | 0.0440 | 231 | 0 | 0 |
| kev | 84.46 | 85.52 | -1.06 | 0.0117 | 6964 | 0 | 0 |
| openjev | 78.39 | 72.96 | +5.43 | 0.0432 | 77253 | 0 | 0 |
| nimble | 88.65 | 91.84 | -3.19 | 0.0588 | 324 | 0 | 0 |
| vitaminc | 87.15 | 78.46 | +8.69 | 0.0450 | 599 | 0 | 0 |
| massive | 88.57 | 87.14 | +1.43 | 0.0190 | 350 | 0 | 0 |

**Six-group mean: 85.90 vs Jev 1.13 83.85 (gap +2.05); accuracy parity: True; ECE parity: False; Brier parity: False**

| ECE check | this | Jev | ok |
|---|---:|---:|---|
| jevbench | 0.0440 | 0.0318 | False |
| kev:transfer-v4/development | 0.0230 | 0.0492 | True |
| kev:transfer-v9/development | 0.0229 | 0.0335 | True |
| vitaminc | 0.0450 | 0.1040 | True |
| massive | 0.0190 | 0.0750 | True |

| BRIER check | this | Jev | ok |
|---|---:|---:|---|
| jevbench | 0.1663 | 0.1811 | True |
| kev:transfer-v4/development | 0.2113 | 0.2110 | False |
| kev:transfer-v9/development | 0.2668 | 0.2121 | False |
| vitaminc | 0.2046 | 0.3130 | True |
| massive | 0.1657 | 0.2040 | True |

Bootstrap 95% CI of the six-group mean (stratified, B=500): 84.70-87.13; P(mean >= 83.85) = 1.000. Jev's own run-to-run variation is not included (Jev is nondeterministic; independent runs differ by up to 1.6 points on VitaminC).

OpenJev tie audit: official 78.39, tie-free (random tie-break) 78.39; rerank tie rate gpqa 0.000, gpqa_fewshot 0.000, max over tasks 0.000

Six-group mean if OpenJev had its 19th task (GSM8K rerank, 89.5-95.0): 86.00-86.04

## Option-permutation flip rate

```
{
 "per_suite": {
  "jevbench": {
   "targets": 139,
   "permutations_per_target": 3,
   "trials": 417,
   "flip_rate": 0.050359712230215826,
   "share_targets_any_flip": 0.08633093525179857,
   "mean_tv": 0.03642805518215242,
   "errors": 0
  },
  "kev": {
   "targets": 300,
   "permutations_per_target": 3,
   "trials": 900,
   "flip_rate": 0.07,
   "share_targets_any_flip": 0.11666666666666667,
   "mean_tv": 0.06512547430623913,
   "errors": 0
  },
  "openjev": {
   "targets": 300,
   "permutations_per_target": 3,
   "trials": 900,
   "flip_rate": 0.022222222222222223,
   "share_targets_any_flip": 0.043333333333333335,
   "mean_tv": 0.0332558272318542,
   "errors": 0
  },
  "nimble": {
   "targets": 146,
   "permutations_per_target": 3,
   "trials": 438,
   "flip_rate": 0.0319634703196347,
   "share_targets_any_flip": 0.0547945205479452,
   "mean_tv": 0.04781048634651356,
   "errors": 0
  },
  "vitaminc": {
   "targets": 300,
   "permutations_per_target": 3,
   "trials": 900,
   "flip_rate": 0.016666666666666666,
   "share_targets_any_flip": 0.03,
   "mean_tv": 0.028533572895468425,
   "errors": 0
  },
  "massive": {
   "targets": 300,
   "permutations_per_target": 3,
   "trials": 900,
   "flip_rate": 0.01888888888888889,
   "share_targets_any_flip": 0.03666666666666667,
   "mean_tv": 0.04364254986980056,
   "errors": 0
  }
 },
 "pooled_flip_rate": 0.03367003367003367,
 "pooled_trials": 4455
}
```

## Repeat-call determinism

```
{
 "requests": 100,
 "share_changed_gt_1e-6": 0.54,
 "share_argmax_changed": 0.0,
 "max_abs_dp": 0.052065951373539865
}
```
