# Parity run: verdict.predictor_hf:VerdictVLLM

- run: gpu-r1  (2026-10-19T01:38:46)
- limit_per_task: None (parity claims require full runs: limit_per_task = None)
- device: **GPU (intermediate run, --allow-gpu; NOT a CPU correctness number)**; device audit: cpu_only=False []

| group | this predictor | Jev 1.13 | delta | ECE10 | units | missing | invalid |
|---|---:|---:|---:|---:|---:|---:|---:|
| jevbench | 88.35 | 87.18 | +1.17 | 0.0257 | 231 | 0 | 0 |
| kev | 83.84 | 85.52 | -1.68 | 0.0297 | 6964 | 0 | 0 |
| openjev | 77.56 | 72.96 | +4.60 | 0.0047 | 77253 | 0 | 0 |
| nimble | 86.52 | 91.84 | -5.32 | 0.0741 | 324 | 0 | 0 |
| vitaminc | 85.31 | 78.46 | +6.85 | 0.0340 | 599 | 0 | 0 |
| massive | 88.57 | 87.14 | +1.43 | 0.0678 | 350 | 0 | 0 |

**Six-group mean: 85.03 vs Jev 1.13 83.85 (gap +1.18); accuracy parity: True; ECE parity: False; Brier parity: False**

| ECE check | this | Jev | ok |
|---|---:|---:|---|
| jevbench | 0.0257 | 0.0318 | True |
| kev:transfer-v4/development | 0.0410 | 0.0492 | True |
| kev:transfer-v9/development | 0.0451 | 0.0335 | False |
| vitaminc | 0.0340 | 0.1040 | True |
| massive | 0.0678 | 0.0750 | True |

| BRIER check | this | Jev | ok |
|---|---:|---:|---|
| jevbench | 0.1590 | 0.1811 | True |
| kev:transfer-v4/development | 0.2168 | 0.2110 | False |
| kev:transfer-v9/development | 0.2763 | 0.2121 | False |
| vitaminc | 0.2242 | 0.3130 | True |
| massive | 0.1862 | 0.2040 | True |

Bootstrap 95% CI of the six-group mean (stratified, B=500): 83.86-86.33; P(mean >= 83.85) = 0.974. Jev's own run-to-run variation is not included (Jev is nondeterministic; independent runs differ by up to 1.6 points on VitaminC).

OpenJev tie audit: official 77.56, tie-free (random tie-break) 77.56; rerank tie rate gpqa 0.000, gpqa_fewshot 0.000, max over tasks 0.000

Six-group mean if OpenJev had its 19th task (GSM8K rerank, 89.5-95.0): 85.13-85.18

## Option-permutation flip rate

```
{
 "per_suite": {
  "jevbench": {
   "targets": 139,
   "permutations_per_target": 3,
   "trials": 417,
   "flip_rate": 0.045563549160671464,
   "share_targets_any_flip": 0.06474820143884892,
   "mean_tv": 0.03242248142954057,
   "errors": 0
  },
  "kev": {
   "targets": 300,
   "permutations_per_target": 3,
   "trials": 900,
   "flip_rate": 0.06333333333333334,
   "share_targets_any_flip": 0.10666666666666667,
   "mean_tv": 0.061512293571077965,
   "errors": 0
  },
  "openjev": {
   "targets": 300,
   "permutations_per_target": 3,
   "trials": 900,
   "flip_rate": 0.017777777777777778,
   "share_targets_any_flip": 0.02666666666666667,
   "mean_tv": 0.029520454582438706,
   "errors": 0
  },
  "nimble": {
   "targets": 146,
   "permutations_per_target": 3,
   "trials": 438,
   "flip_rate": 0.0410958904109589,
   "share_targets_any_flip": 0.0684931506849315,
   "mean_tv": 0.05039638583869252,
   "errors": 0
  },
  "vitaminc": {
   "targets": 300,
   "permutations_per_target": 3,
   "trials": 900,
   "flip_rate": 0.005555555555555556,
   "share_targets_any_flip": 0.013333333333333334,
   "mean_tv": 0.02943952739443594,
   "errors": 0
  },
  "massive": {
   "targets": 300,
   "permutations_per_target": 3,
   "trials": 900,
   "flip_rate": 0.021111111111111112,
   "share_targets_any_flip": 0.04666666666666667,
   "mean_tv": 0.03318749811156843,
   "errors": 0
  }
 },
 "pooled_flip_rate": 0.030078563411896745,
 "pooled_trials": 4455
}
```

## Repeat-call determinism

```
{
 "requests": 100,
 "share_changed_gt_1e-6": 0.61,
 "share_argmax_changed": 0.0,
 "max_abs_dp": 0.035306371701201156
}
```
