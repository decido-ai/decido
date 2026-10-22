# Parity run: out.predictor_llamacpp:VerdictLlamaCpp

- run: mac-latency-quick  (2026-10-21T19:54:10)
- limit_per_task: 1 (parity claims require full runs: limit_per_task = None)
- device: **CPU-only**; device audit: cpu_only=True []

| group | this predictor | Jev 1.13 | delta | ECE10 | units | missing | invalid |
|---|---:|---:|---:|---:|---:|---:|---:|
| massive | 100.00 | 87.14 | +12.86 | 0.0391 | 1 | 0 | 0 |

| ECE check | this | Jev | ok |
|---|---:|---:|---|
| massive | 0.0391 | 0.0750 | True |

| BRIER check | this | Jev | ok |
|---|---:|---:|---|
| massive | 0.0020 | 0.2040 | True |

## CPU latency (1k-token state, 5 questions, batch 1)

```
{
 "threads=10": {
  "fixture": "support-ticket JSON state (~1k Qwen3 tokens) + 5 questions (2 noul, 2 choice[4,6], 1 score[5])",
  "state_tokens_qwen3": 999,
  "n": 10,
  "warmup": 5,
  "p50_ms": 4139.120958498097,
  "p95_ms": 4258.100287194247,
  "mean_ms": 4128.337162401294,
  "min_ms": 3971.093500003917,
  "torch_threads": 10,
  "load_seconds": 96.83765037500416,
  "device_audit": {
   "CUDA_VISIBLE_DEVICES": "",
   "torch": {
    "version": "2.14.1",
    "threads": 10,
    "mps_allocated_bytes": 0,
    "mps_peak_sampled_bytes": 0
   },
   "mlx": {
    "default_device": "Device(cpu, 0)"
   },
   "violations": [],
   "device": "CPU-only",
   "cpu_only": true
  }
 }
}
```
