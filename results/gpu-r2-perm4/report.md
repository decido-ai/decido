# Parity run: verdict.predictor_hf:VerdictVLLM

- run: gpu-r2-perm4  (2026-10-20T06:37:16)
- limit_per_task: None (parity claims require full runs: limit_per_task = None)
- device: **GPU (intermediate run, --allow-gpu; NOT a CPU correctness number)**; device audit: cpu_only=False []

| group | this predictor | Jev 1.13 | delta | ECE10 | units | missing | invalid |
|---|---:|---:|---:|---:|---:|---:|---:|
| jevbench | 88.23 | 87.18 | +1.05 | 0.0423 | 231 | 0 | 0 |
| kev | 84.71 | 85.52 | -0.81 | 0.0199 | 6964 | 0 | 0 |
| openjev | 78.57 | 72.96 | +5.61 | 0.0436 | 77253 | 0 | 0 |
| nimble | 87.59 | 91.84 | -4.25 | 0.0651 | 324 | 0 | 0 |
| vitaminc | 86.48 | 78.46 | +8.02 | 0.0320 | 599 | 0 | 0 |
| massive | 88.29 | 87.14 | +1.15 | 0.0186 | 350 | 0 | 0 |

**Six-group mean: 85.64 vs Jev 1.13 83.85 (gap +1.79); accuracy parity: True; ECE parity: False; Brier parity: False**

| ECE check | this | Jev | ok |
|---|---:|---:|---|
| jevbench | 0.0423 | 0.0318 | False |
| kev:transfer-v4/development | 0.0253 | 0.0492 | True |
| kev:transfer-v9/development | 0.0282 | 0.0335 | True |
| vitaminc | 0.0320 | 0.1040 | True |
| massive | 0.0186 | 0.0750 | True |

| BRIER check | this | Jev | ok |
|---|---:|---:|---|
| jevbench | 0.1701 | 0.1811 | True |
| kev:transfer-v4/development | 0.2123 | 0.2110 | False |
| kev:transfer-v9/development | 0.2700 | 0.2121 | False |
| vitaminc | 0.2011 | 0.3130 | True |
| massive | 0.1644 | 0.2040 | True |

Bootstrap 95% CI of the six-group mean (stratified, B=500): 84.38-86.90; P(mean >= 83.85) = 0.998. Jev's own run-to-run variation is not included (Jev is nondeterministic; independent runs differ by up to 1.6 points on VitaminC).

OpenJev tie audit: official 78.57, tie-free (random tie-break) 78.57; rerank tie rate gpqa 0.000, gpqa_fewshot 0.000, max over tasks 0.000

Six-group mean if OpenJev had its 19th task (GSM8K rerank, 89.5-95.0): 85.74-85.79

## Option-permutation flip rate

```
{
 "per_suite": {
  "jevbench": {
   "targets": 139,
   "permutations_per_target": 3,
   "trials": 417,
   "flip_rate": 0.011990407673860911,
   "share_targets_any_flip": 0.02158273381294964,
   "mean_tv": 0.013383611806503812,
   "errors": 0
  },
  "kev": {
   "targets": 300,
   "permutations_per_target": 3,
   "trials": 900,
   "flip_rate": 0.028888888888888888,
   "share_targets_any_flip": 0.05,
   "mean_tv": 0.02770441769481872,
   "errors": 0
  },
  "openjev": {
   "targets": 300,
   "permutations_per_target": 3,
   "trials": 900,
   "flip_rate": 0.006666666666666667,
   "share_targets_any_flip": 0.013333333333333334,
   "mean_tv": 0.008980073801817756,
   "errors": 0
  },
  "nimble": {
   "targets": 146,
   "permutations_per_target": 3,
   "trials": 438,
   "flip_rate": 0.01141552511415525,
   "share_targets_any_flip": 0.02054794520547945,
   "mean_tv": 0.014563499240676011,
   "errors": 0
  },
  "vitaminc": {
   "targets": 300,
   "permutations_per_target": 3,
   "trials": 900,
   "flip_rate": 0.006666666666666667,
   "share_targets_any_flip": 0.01,
   "mean_tv": 0.008048473609791523,
   "errors": 0
  },
  "massive": {
   "targets": 300,
   "permutations_per_target": 3,
   "trials": 900,
   "flip_rate": 0.014444444444444444,
   "share_targets_any_flip": 0.02666666666666667,
   "mean_tv": 0.025899170552516063,
   "errors": 0
  }
 },
 "pooled_flip_rate": 0.013692480359147026,
 "pooled_trials": 4455
}
```

## Repeat-call determinism

```
{
 "requests": 50,
 "share_changed_gt_1e-6": 0.7,
 "share_argmax_changed": 0.0,
 "max_abs_dp": 0.024854674504099616
}
```
