# Parity run: verdict.predictor_hf:VerdictVLLM

- run: zs-gpt-oss-20b  (2026-10-18T02:02:39)
- limit_per_task: None (parity claims require full runs: limit_per_task = None)
- device: **GPU (intermediate run, --allow-gpu; NOT a CPU correctness number)**; device audit: cpu_only=False []

| group | this predictor | Jev 1.13 | delta | ECE10 | units | missing | invalid |
|---|---:|---:|---:|---:|---:|---:|---:|
| jevbench | 65.63 | 87.18 | -21.55 | 0.2513 | 231 | 0 | 0 |
| kev | 67.70 | 85.52 | -17.82 | 0.1756 | 6964 | 0 | 0 |
| openjev | 45.49 | 72.96 | -27.47 | 0.5335 | 77253 | 0 | 0 |
| nimble | 61.35 | 91.84 | -30.49 | 0.2298 | 324 | 0 | 0 |
| vitaminc | 11.85 | 78.46 | -66.61 | 0.8622 | 599 | 0 | 0 |
| massive | 84.57 | 87.14 | -2.57 | 0.1060 | 350 | 0 | 0 |

**Six-group mean: 56.10 vs Jev 1.13 83.85 (gap -27.75); accuracy parity: False; ECE parity: False; Brier parity: False**

| ECE check | this | Jev | ok |
|---|---:|---:|---|
| jevbench | 0.2513 | 0.0318 | False |
| kev:transfer-v4/development | 0.1993 | 0.0492 | False |
| kev:transfer-v9/development | 0.1901 | 0.0335 | False |
| vitaminc | 0.8622 | 0.1040 | False |
| massive | 0.1060 | 0.0750 | False |

| BRIER check | this | Jev | ok |
|---|---:|---:|---|
| jevbench | 0.5743 | 0.1811 | False |
| kev:transfer-v4/development | 0.4909 | 0.2110 | False |
| kev:transfer-v9/development | 0.5424 | 0.2121 | False |
| vitaminc | 1.7038 | 0.3130 | False |
| massive | 0.2643 | 0.2040 | False |

Bootstrap 95% CI of the six-group mean (stratified, B=500): 54.53-57.46; P(mean >= 83.85) = 0.000. Jev's own run-to-run variation is not included (Jev is nondeterministic; independent runs differ by up to 1.6 points on VitaminC).

Six-group mean if OpenJev had its 19th task (GSM8K rerank, 89.5-95.0): 56.48-56.53

## Option-permutation flip rate

```
{
 "per_suite": {
  "jevbench": {
   "targets": 139,
   "permutations_per_target": 3,
   "trials": 417,
   "flip_rate": 0.21342925659472423,
   "share_targets_any_flip": 0.31654676258992803,
   "mean_tv": 0.18805137266670555,
   "errors": 0
  },
  "kev": {
   "targets": 300,
   "permutations_per_target": 3,
   "trials": 900,
   "flip_rate": 0.17222222222222222,
   "share_targets_any_flip": 0.26666666666666666,
   "mean_tv": 0.18546434225073383,
   "errors": 0
  },
  "openjev": {
   "targets": 300,
   "permutations_per_target": 3,
   "trials": 900,
   "flip_rate": 0.10777777777777778,
   "share_targets_any_flip": 0.2,
   "mean_tv": 0.12451956837098631,
   "errors": 0
  },
  "nimble": {
   "targets": 146,
   "permutations_per_target": 3,
   "trials": 438,
   "flip_rate": 0.2100456621004566,
   "share_targets_any_flip": 0.3835616438356164,
   "mean_tv": 0.18939370726602092,
   "errors": 0
  },
  "vitaminc": {
   "targets": 300,
   "permutations_per_target": 3,
   "trials": 900,
   "flip_rate": 0.017777777777777778,
   "share_targets_any_flip": 0.03,
   "mean_tv": 0.03084711328391497,
   "errors": 0
  },
  "massive": {
   "targets": 300,
   "permutations_per_target": 3,
   "trials": 900,
   "flip_rate": 0.058888888888888886,
   "share_targets_any_flip": 0.11,
   "mean_tv": 0.07064401061797264,
   "errors": 0
  }
 },
 "pooled_flip_rate": 0.11268237934904601,
 "pooled_trials": 4455
}
```

## Repeat-call determinism

```
{
 "requests": 100,
 "share_changed_gt_1e-6": 0.63,
 "share_argmax_changed": 0.0,
 "max_abs_dp": 0.06712818849553404
}
```
