"""Download training-source datasets (TRAIN splits only) into data/src/<repo>. Re-runnable.

    python -m decido.data.download
"""
import os
import sys

from huggingface_hub import snapshot_download

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "data", "src")

SOURCES = {
    "tasksource/tasksource-jev-typed-decisions": ["data/train-*", "sources.yaml", "AGENTS.md", "README.md"],
    "tasksource/procedural-typed-decisions": ["*/train-*", "README.md"],
    "ZefanCai/Open-Jev": ["*"],
    "ZefanCai/Open-Jev-v1.1": ["README.md", "export-manifest.json", "provenance/*", "data/*/train-*"],
    "allenai/ai2_arc": ["*/train-*"],
    "cais/mmlu": ["all/auxiliary_train-*"],
    "allenai/openbookqa": ["main/train-*", "README.md"],
    "tau/commonsense_qa": ["data/train-*"],
    "allenai/winogrande": ["winogrande_xl/train-*"],
    "Rowan/hellaswag": ["data/train-*"],
    "openai/gsm8k": ["main/train-*"],
    "nyu-mll/multi_nli": ["data/train-*"],
    "facebook/anli": ["plain_text/train_r*"],
    "alisawuffles/WANLI": ["train.jsonl"],
    "stanfordnlp/snli": ["plain_text/train-*"],
    "allenai/scitail": ["snli_format/train-*"],
    "tals/vitaminc": ["train.jsonl"],
    "legacy-datasets/banking77": ["data/train-*"],
    "clinc/clinc_oos": ["plus/train-*"],
    "google/civil_comments": ["data/train-00000-*"],
    "google-research-datasets/go_emotions": ["raw/train-*"],
    "ucberkeley-dlab/measuring-hate-speech": ["measuring-hate-speech.parquet"],
    "OpenAssistant/oasst1": ["data/train-*"],
    "nvidia/HelpSteer2": ["train.jsonl.gz"],
    "google/boolq": ["data/train-*"],
}


def main(only=None):
    for repo, pats in SOURCES.items():
        if only and repo not in only:
            continue
        dst = os.path.join(SRC, repo.replace("/", "__"))
        try:
            snapshot_download(repo, repo_type="dataset", allow_patterns=pats, local_dir=dst, max_workers=8)
            print("OK", repo, flush=True)
        except Exception as e:
            print("FAIL", repo, type(e).__name__, str(e)[:200], flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or None)
