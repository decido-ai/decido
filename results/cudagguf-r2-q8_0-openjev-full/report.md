# Parity run: verdict.predictor_llamacpp:VerdictLlamaCpp

- run: cudagguf-r2-q8_0-openjev-full  (2026-10-19T13:54:41)
- limit_per_task: None (parity claims require full runs: limit_per_task = None)
- device: **GPU (intermediate run, --allow-gpu; NOT a CPU correctness number)**; device audit: cpu_only=False []

| group | this predictor | Jev 1.13 | delta | ECE10 | units | missing | invalid |
|---|---:|---:|---:|---:|---:|---:|---:|
| openjev | 78.37 | 72.96 | +5.41 | 0.0403 | 77253 | 0 | 0 |

OpenJev tie audit: official 78.37, tie-free (random tie-break) 78.37; rerank tie rate gpqa 0.000, gpqa_fewshot 0.000, max over tasks 0.000
