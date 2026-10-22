# Parity run: eval.parity.predictors:jev_replay

- run: jev-replay-arxiv  (2026-10-15T22:16:35)
- limit_per_task: None (parity claims require full runs: limit_per_task = None)
- device audit: cpu_only=True []


## arxiv: 89.39 (mean task accuracy (replay validation))

| task | acc | ECE15 (arXiv def.) |
|---|---:|---:|
| ag_news | 88.51 | 0.0769 |
| anli | 73.94 | 0.1017 |
| arc | 98.76 | 0.0054 |
| banking77 | 79.71 | 0.0870 |
| boolq | 91.31 | 0.0427 |
| hellaswag | 95.49 | 0.0185 |
| paws | 84.99 | 0.0397 |
| rotten_tomatoes | 93.34 | 0.0380 |
| sst2 | 96.44 | 0.0148 |
| winogrande | 91.40 | 0.0250 |
