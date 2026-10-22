# Parity run: verdict.predictor_llamacpp:VerdictLlamaCpp

- run: cpu-r2-q8_0-openjev300  (2026-10-19T20:23:06)
- limit_per_task: 300 (parity claims require full runs: limit_per_task = None)
- device: **CPU-only**; device audit: cpu_only=True []

| group | this predictor | Jev 1.13 | delta | ECE10 | units | missing | invalid |
|---|---:|---:|---:|---:|---:|---:|---:|
| openjev | 78.93 | 72.96 | +5.97 | 0.0150 | 5196 | 0 | 0 |

OpenJev tie audit: official 78.93, tie-free (random tie-break) 78.93; rerank tie rate gpqa 0.000, gpqa_fewshot 0.000, max over tasks 0.003
