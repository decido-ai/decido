# Parity run: out.predictor_llamacpp:VerdictLlamaCpp

- run: mac-cpu-1h-five  (2026-10-21T21:25:58)
- limit_per_task: 50 (parity claims require full runs: limit_per_task = None)
- device: **CPU-only**; device audit: cpu_only=True []

| group | this predictor | Jev 1.13 | delta | ECE10 | units | missing | invalid |
|---|---:|---:|---:|---:|---:|---:|---:|
| jevbench | 89.10 | 87.18 | +1.92 | 0.0342 | 231 | 0 | 0 |
| kev | 86.99 | 85.52 | +1.47 | 0.0435 | 329 | 0 | 0 |
| nimble | 89.32 | 91.84 | -2.52 | 0.0296 | 240 | 0 | 0 |
| vitaminc | 84.00 | 78.46 | +5.54 | 0.0599 | 100 | 0 | 0 |
| massive | 92.00 | 87.14 | +4.86 | 0.0568 | 50 | 0 | 0 |

| ECE check | this | Jev | ok |
|---|---:|---:|---|
| jevbench | 0.0342 | 0.0318 | False |
| kev:transfer-v4/development | 0.0338 | 0.0492 | True |
| kev:transfer-v9/development | 0.0905 | 0.0335 | False |
| vitaminc | 0.0599 | 0.1040 | True |
| massive | 0.0568 | 0.0750 | True |

| BRIER check | this | Jev | ok |
|---|---:|---:|---|
| jevbench | 0.1682 | 0.1811 | True |
| kev:transfer-v4/development | 0.1257 | 0.2110 | True |
| kev:transfer-v9/development | 0.2295 | 0.2121 | False |
| vitaminc | 0.2091 | 0.3130 | True |
| massive | 0.1307 | 0.2040 | True |

## Repeat-call determinism

```
{
 "requests": 20,
 "share_changed_gt_1e-6": 0.0,
 "share_argmax_changed": 0.0,
 "max_abs_dp": 0.0
}
```
