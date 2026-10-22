# Parity run: verdict.predictor_hf:VerdictVLLM

- run: zs-qwen3.6-35b-a3b  (2026-10-18T02:48:25)
- limit_per_task: None (parity claims require full runs: limit_per_task = None)
- device: **GPU (intermediate run, --allow-gpu; NOT a CPU correctness number)**; device audit: cpu_only=False []

| group | this predictor | Jev 1.13 | delta | ECE10 | units | missing | invalid |
|---|---:|---:|---:|---:|---:|---:|---:|
| jevbench | 79.01 | 87.18 | -8.17 | 0.0939 | 231 | 0 | 0 |
| kev | 78.06 | 85.52 | -7.46 | 0.0990 | 6964 | 0 | 0 |
| openjev | 59.56 | 72.96 | -13.40 | 0.1097 | 77253 | 0 | 0 |
| nimble | 82.27 | 91.84 | -9.57 | 0.0924 | 324 | 0 | 0 |
| vitaminc | 73.29 | 78.46 | -5.17 | 0.1671 | 599 | 0 | 0 |
| massive | 84.00 | 87.14 | -3.14 | 0.1117 | 350 | 0 | 0 |

**Six-group mean: 76.03 vs Jev 1.13 83.85 (gap -7.82); accuracy parity: False; ECE parity: False; Brier parity: False**

| ECE check | this | Jev | ok |
|---|---:|---:|---|
| jevbench | 0.0939 | 0.0318 | False |
| kev:transfer-v4/development | 0.0946 | 0.0492 | False |
| kev:transfer-v9/development | 0.1150 | 0.0335 | False |
| vitaminc | 0.1671 | 0.1040 | False |
| massive | 0.1117 | 0.0750 | False |

| BRIER check | this | Jev | ok |
|---|---:|---:|---|
| jevbench | 0.2838 | 0.1811 | False |
| kev:transfer-v4/development | 0.3168 | 0.2110 | False |
| kev:transfer-v9/development | 0.3828 | 0.2121 | False |
| vitaminc | 0.4297 | 0.3130 | False |
| massive | 0.2655 | 0.2040 | False |

Bootstrap 95% CI of the six-group mean (stratified, B=500): 74.59-77.35; P(mean >= 83.85) = 0.000. Jev's own run-to-run variation is not included (Jev is nondeterministic; independent runs differ by up to 1.6 points on VitaminC).

Six-group mean if OpenJev had its 19th task (GSM8K rerank, 89.5-95.0): 76.29-76.34

## Option-permutation flip rate

```
{
 "per_suite": {
  "jevbench": {
   "targets": 139,
   "permutations_per_target": 3,
   "trials": 417,
   "flip_rate": 0.17266187050359713,
   "share_targets_any_flip": 0.22302158273381295,
   "mean_tv": 0.1467507915290274,
   "errors": 0
  },
  "kev": {
   "targets": 300,
   "permutations_per_target": 3,
   "trials": 900,
   "flip_rate": 0.20777777777777778,
   "share_targets_any_flip": 0.29,
   "mean_tv": 0.17342998745888216,
   "errors": 0
  },
  "openjev": {
   "targets": 300,
   "permutations_per_target": 3,
   "trials": 900,
   "flip_rate": 0.08333333333333333,
   "share_targets_any_flip": 0.12,
   "mean_tv": 0.08389001964210585,
   "errors": 0
  },
  "nimble": {
   "targets": 146,
   "permutations_per_target": 3,
   "trials": 438,
   "flip_rate": 0.15753424657534246,
   "share_targets_any_flip": 0.2945205479452055,
   "mean_tv": 0.16603468898652146,
   "errors": 0
  },
  "vitaminc": {
   "targets": 300,
   "permutations_per_target": 3,
   "trials": 900,
   "flip_rate": 0.07,
   "share_targets_any_flip": 0.12333333333333334,
   "mean_tv": 0.10635933155874505,
   "errors": 0
  },
  "massive": {
   "targets": 300,
   "permutations_per_target": 3,
   "trials": 900,
   "flip_rate": 0.044444444444444446,
   "share_targets_any_flip": 0.08333333333333333,
   "mean_tv": 0.06243032540894526,
   "errors": 0
  }
 },
 "pooled_flip_rate": 0.11358024691358025,
 "pooled_trials": 4455
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
