# Parity run: verdict.predictor_hf:VerdictVLLM

- run: gpu-r1-heldoutT  (2026-10-19T02:24:58)
- limit_per_task: None (parity claims require full runs: limit_per_task = None)
- device: **GPU (intermediate run, --allow-gpu; NOT a CPU correctness number)**; device audit: cpu_only=False []

| group | this predictor | Jev 1.13 | delta | ECE10 | units | missing | invalid |
|---|---:|---:|---:|---:|---:|---:|---:|
| jevbench | 88.76 | 87.18 | +1.58 | 0.0536 | 231 | 0 | 0 |
| kev | 83.65 | 85.52 | -1.87 | 0.0112 | 6964 | 0 | 0 |
| openjev | 77.18 | 72.96 | +4.22 | 0.0491 | 77253 | 0 | 0 |
| nimble | 87.23 | 91.84 | -4.61 | 0.0802 | 324 | 0 | 0 |
| vitaminc | 85.64 | 78.46 | +7.18 | 0.0501 | 599 | 0 | 0 |
| massive | 88.57 | 87.14 | +1.43 | 0.0367 | 350 | 0 | 0 |

**Six-group mean: 85.17 vs Jev 1.13 83.85 (gap +1.32); accuracy parity: True; ECE parity: False; Brier parity: False**

| ECE check | this | Jev | ok |
|---|---:|---:|---|
| jevbench | 0.0536 | 0.0318 | False |
| kev:transfer-v4/development | 0.0294 | 0.0492 | True |
| kev:transfer-v9/development | 0.0196 | 0.0335 | True |
| vitaminc | 0.0501 | 0.1040 | True |
| massive | 0.0367 | 0.0750 | True |

| BRIER check | this | Jev | ok |
|---|---:|---:|---|
| jevbench | 0.1595 | 0.1811 | True |
| kev:transfer-v4/development | 0.2122 | 0.2110 | False |
| kev:transfer-v9/development | 0.2727 | 0.2121 | False |
| vitaminc | 0.2262 | 0.3130 | True |
| massive | 0.1778 | 0.2040 | True |

Bootstrap 95% CI of the six-group mean (stratified, B=500): 83.97-86.45; P(mean >= 83.85) = 0.982. Jev's own run-to-run variation is not included (Jev is nondeterministic; independent runs differ by up to 1.6 points on VitaminC).

OpenJev tie audit: official 77.18, tie-free (random tie-break) 77.18; rerank tie rate gpqa 0.000, gpqa_fewshot 0.000, max over tasks 0.000

Six-group mean if OpenJev had its 19th task (GSM8K rerank, 89.5-95.0): 85.28-85.33

## Option-permutation flip rate

```
{
 "per_suite": {
  "jevbench": {
   "targets": 139,
   "permutations_per_target": 3,
   "trials": 417,
   "flip_rate": 0.06474820143884892,
   "share_targets_any_flip": 0.09352517985611511,
   "mean_tv": 0.03472219971143873,
   "errors": 0
  },
  "kev": {
   "targets": 300,
   "permutations_per_target": 3,
   "trials": 900,
   "flip_rate": 0.07333333333333333,
   "share_targets_any_flip": 0.11,
   "mean_tv": 0.06398091925559977,
   "errors": 0
  },
  "openjev": {
   "targets": 300,
   "permutations_per_target": 3,
   "trials": 900,
   "flip_rate": 0.021111111111111112,
   "share_targets_any_flip": 0.03666666666666667,
   "mean_tv": 0.02981828830560479,
   "errors": 0
  },
  "nimble": {
   "targets": 146,
   "permutations_per_target": 3,
   "trials": 438,
   "flip_rate": 0.06164383561643835,
   "share_targets_any_flip": 0.0821917808219178,
   "mean_tv": 0.052988825768404454,
   "errors": 0
  },
  "vitaminc": {
   "targets": 300,
   "permutations_per_target": 3,
   "trials": 900,
   "flip_rate": 0.012222222222222223,
   "share_targets_any_flip": 0.02,
   "mean_tv": 0.03217847894565533,
   "errors": 0
  },
  "massive": {
   "targets": 300,
   "permutations_per_target": 3,
   "trials": 900,
   "flip_rate": 0.016666666666666666,
   "share_targets_any_flip": 0.03333333333333333,
   "mean_tv": 0.04209437086841588,
   "errors": 0
  }
 },
 "pooled_flip_rate": 0.037037037037037035,
 "pooled_trials": 4455
}
```

## Repeat-call determinism

```
{
 "requests": 100,
 "share_changed_gt_1e-6": 0.57,
 "share_argmax_changed": 0.0,
 "max_abs_dp": 0.03269654606433886
}
```
