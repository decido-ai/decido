# Parity run: verdict.predictor_hf:VerdictVLLM

- run: gpu-ablate_indomain  (2026-10-20T05:54:07)
- limit_per_task: None (parity claims require full runs: limit_per_task = None)
- device: **GPU (intermediate run, --allow-gpu; NOT a CPU correctness number)**; device audit: cpu_only=False []

| group | this predictor | Jev 1.13 | delta | ECE10 | units | missing | invalid |
|---|---:|---:|---:|---:|---:|---:|---:|
| jevbench | 89.30 | 87.18 | +2.12 | 0.0530 | 231 | 0 | 0 |
| kev | 82.66 | 85.52 | -2.86 | 0.0185 | 6964 | 0 | 0 |
| openjev | 67.77 | 72.96 | -5.19 | 0.0670 | 77253 | 0 | 0 |
| nimble | 86.17 | 91.84 | -5.67 | 0.0417 | 324 | 0 | 0 |
| vitaminc | 79.47 | 78.46 | +1.01 | 0.0638 | 599 | 0 | 0 |
| massive | 86.86 | 87.14 | -0.28 | 0.0407 | 350 | 0 | 0 |

**Six-group mean: 82.04 vs Jev 1.13 83.85 (gap -1.81); accuracy parity: False; ECE parity: False; Brier parity: False**

| ECE check | this | Jev | ok |
|---|---:|---:|---|
| jevbench | 0.0530 | 0.0318 | False |
| kev:transfer-v4/development | 0.0407 | 0.0492 | True |
| kev:transfer-v9/development | 0.0323 | 0.0335 | True |
| vitaminc | 0.0638 | 0.1040 | True |
| massive | 0.0407 | 0.0750 | True |

| BRIER check | this | Jev | ok |
|---|---:|---:|---|
| jevbench | 0.1806 | 0.1811 | True |
| kev:transfer-v4/development | 0.2187 | 0.2110 | False |
| kev:transfer-v9/development | 0.2880 | 0.2121 | False |
| vitaminc | 0.3090 | 0.3130 | True |
| massive | 0.2083 | 0.2040 | False |

Bootstrap 95% CI of the six-group mean (stratified, B=500): 80.96-83.18; P(mean >= 83.85) = 0.006. Jev's own run-to-run variation is not included (Jev is nondeterministic; independent runs differ by up to 1.6 points on VitaminC).

OpenJev tie audit: official 67.77, tie-free (random tie-break) 67.77; rerank tie rate gpqa 0.000, gpqa_fewshot 0.000, max over tasks 0.000

Six-group mean if OpenJev had its 19th task (GSM8K rerank, 89.5-95.0): 82.23-82.28

## Option-permutation flip rate

```
{
 "per_suite": {
  "jevbench": {
   "targets": 139,
   "permutations_per_target": 3,
   "trials": 417,
   "flip_rate": 0.0407673860911271,
   "share_targets_any_flip": 0.07194244604316546,
   "mean_tv": 0.04162478671823324,
   "errors": 0
  },
  "kev": {
   "targets": 300,
   "permutations_per_target": 3,
   "trials": 900,
   "flip_rate": 0.08,
   "share_targets_any_flip": 0.12,
   "mean_tv": 0.07238899426132371,
   "errors": 0
  },
  "openjev": {
   "targets": 300,
   "permutations_per_target": 3,
   "trials": 900,
   "flip_rate": 0.04888888888888889,
   "share_targets_any_flip": 0.09333333333333334,
   "mean_tv": 0.049098907852585236,
   "errors": 0
  },
  "nimble": {
   "targets": 146,
   "permutations_per_target": 3,
   "trials": 438,
   "flip_rate": 0.0319634703196347,
   "share_targets_any_flip": 0.04794520547945205,
   "mean_tv": 0.04598836680540168,
   "errors": 0
  },
  "vitaminc": {
   "targets": 300,
   "permutations_per_target": 3,
   "trials": 900,
   "flip_rate": 0.03666666666666667,
   "share_targets_any_flip": 0.06,
   "mean_tv": 0.039246135072781685,
   "errors": 0
  },
  "massive": {
   "targets": 300,
   "permutations_per_target": 3,
   "trials": 900,
   "flip_rate": 0.03,
   "share_targets_any_flip": 0.05,
   "mean_tv": 0.06125429957977098,
   "errors": 0
  }
 },
 "pooled_flip_rate": 0.046464646464646465,
 "pooled_trials": 4455
}
```

## Repeat-call determinism

```
{
 "requests": 50,
 "share_changed_gt_1e-6": 0.6,
 "share_argmax_changed": 0.0,
 "max_abs_dp": 0.025909840119961836
}
```
