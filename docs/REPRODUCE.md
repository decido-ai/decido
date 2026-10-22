# Reproducing Decido

All commands run from the repository root. Python 3.12; training/labelling need CUDA GPUs (the reported model used
2× A100 80 GB, ~13 h for round 1 and ~7 h for round 2); evaluation of the GGUF runs on CPU.

## 1. Evaluation suites (`eval/parity/data/`)

Place the sources under `data/raw/parity/` (read as data only; no downloaded code is executed except the MIT
jev-benchmarking task classes used by the `arxiv` replay suite), then `PYTHONHASHSEED=0 python -m eval.parity.build`.

| suite | source (commit) | path expected under `data/raw/parity/` |
|---|---|---|
| JevBench (231) | github.com/fstandhartinger/jevbench @ 609a664e | `jevbench-repo/datasets/public/{original,easy,hard}.jsonl`, `jevbench-repo/COMMIT` |
| Kev (6 splits) | github.com/jaredpalmer/kev @ 1f04cfa2 | `kev-repo/evals/{v7/decision-v7,v4/transfer-v4,v9/transfer-v9}/{development,test}.jsonl`, `kev-repo/COMMIT` |
| Nimble (324; 282 scored) | github.com/bespokelabsai/nimble @ 4edb0703 | `nimble-repo/data/eval.jsonl`, `nimble-repo/docs/assets/public-benchmarks/subsets/*-manifest.json`, `nimble-repo/COMMIT`; NeoHorse-Jev-4B `tokenizer.json` at `neohorse/tokenizer/` (used to reconstruct the 282 subset) |
| VitaminC (599) | talschuster.github.io/static/vitaminc.zip + Nimble manifest | `vitaminc/vitaminc/dev.jsonl` |
| MASSIVE-en (350) | amazon-massive-dataset-1.1.tar.gz + Nimble manifest | `massive/amazon-massive-dataset-1.1.tar.gz` |
| OpenJev text (18 of 19 tasks) | AlexWortega/openjev code/eval.py @ a2044801 (re-implemented); data from the HF Hub; GPQA diamond CSV | `gpqa/dataset/gpqa_diamond.csv`; HF datasets cached under `data/hf-cache/` |
| arXiv replay (validation only) | jev-benchmarking @ 87562114 | `jev-benchmarking-src/`, `jev-benchmarking.COMMIT` |

The GSM8K `nli_rerank@4` candidate pool of OpenJev was never published; that task is reported as missing (conservative).
Check your build against `eval/parity/data/MANIFEST.json` (sha256 per suite). `python -m eval.parity.leakage` writes the
eval text hashes used by the training-data gate; `python -m eval.parity.fixture` rebuilds the latency fixture.

## 2. Training data
```bash
python -m decido.data.download                       # train splits only -> data/src/
python -m decido.data.build_pool                     # -> data/pool/*.jsonl
python -m eval.parity.leakage check data/pool/*.jsonl --out data/pool_clean --report data/leakage_report_pool.json
python -m decido.data.split                          # task-held-out train_pool / val / calib
python -m decido.label --model Qwen/Qwen3.8-27B --in <pool> --out runs/label/pool.jsonl --perm 2   # teacher labels
python -m decido.data.synth_gen workflows|states|rows ...                                         # teacher-written scenarios
python -m decido.data.mix_targets --pool ... --teacher ... --calib ... --calib-teacher ... --out data/train_mixed.jsonl
python -m decido.data.compound --train-out data/compound_train.jsonl --eval-out eval/compound/compound_eval.jsonl
python -m decido.data.make_train --mixed data/train_mixed.jsonl --extra data/compound_train.jsonl --out data/train.jsonl
```
Final-data exclusions applied after the leakage audit: rows with `meta.source` in {`scitail`} and any `hh-rlhf` tasksource
rows are removed; the project's own procedural generator was removed entirely (it mirrored the structure of specific
JevBench items). Round 2 adds `decido.data.knowledge2` (medmcqa, AQuA-RAT, QASC, QuaRTz, CosmosQA) and
`decido.data.decision_tables` plus ~110k replayed round-1 questions; every mixture is gated with
`python -m eval.parity.leakage check` (0 drops on the final files). Known quirk: `knowledge10.py` (round 3, not in the
released model) names two sources `aquarat`/`cosmosqa`, which never match the data's `aqua_rat`/`cosmos_qa`.

## 3. Training, calibration, export
```bash
torchrun --nproc_per_node 2 -m decido.train --model google/gemma-4-26B-A4B-it --train data/train.jsonl --val data/val.jsonl \
  --out runs/train/r1 --rank 16 --targets attn+mlp+router+experts --lr 1e-4 --epochs 1 --max-len 2048 --max-tokens 8192 --grad-accum 4
torchrun --nproc_per_node 2 -m decido.train ... --train data/train_r2.jsonl --init-adapter runs/train/r1/final --lr 5e-5 --out runs/train/r2
python -m decido.merge --base google/gemma-4-26B-A4B-it --adapter runs/train/r2/final --out models/merged/r2
python -m decido.label --model models/merged/r2 --in data/calib.jsonl --out runs/label/calib.r2.jsonl
python -m decido.fit_temps --calib data/calib.jsonl --pred runs/label/calib.r2.jsonl --out runs/train/r2/temps.json
python -m decido.make_bundle --model models/merged/r2 --out bundle --temps runs/train/r2/temps.json
python llama.cpp/convert_hf_to_gguf.py models/merged/r2 --outtype bf16 --outfile r2-bf16.gguf
llama.cpp/build/bin/llama-quantize r2-bf16.gguf decido-26b-a4b-q8_0.gguf Q8_0
```
Round selection used a pre-registered dev rule (`python -m decido.dev_score`): higher dev accuracy wins; within 0.5 points,
lower NLL. Temperatures are fitted on held-out-task calibration data only.
