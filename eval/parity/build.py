"""Freeze the parity suites into eval/parity/data/<suite>.jsonl plus data/MANIFEST.json.

    PYTHONHASHSEED=0 HF_HOME=data/hf-cache \
        python -m eval.parity.build --suites jevbench kev nimble vitaminc massive openjev arxiv

Every downloaded source is treated as data. The only third-party code that is executed is the MIT
jev-benchmarking harness (task definitions) for the `arxiv` replay-validation suite, after review.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("HF_HOME", str(__import__("pathlib").Path(__file__).resolve().parents[2] / "data" / "hf-cache"))
os.environ.setdefault("HF_DATASETS_TRUST_REMOTE_CODE", "0")

from eval.parity.common import DATA, sha256_file, write_jsonl  # noqa: E402

SUITES = ["jevbench", "kev", "nimble", "vitaminc", "massive", "openjev", "arxiv"]


def build_one(name: str):
    if name in ("jevbench", "kev", "nimble", "vitaminc", "massive"):
        from eval.parity.suites import local
        return getattr(local, f"build_{name}")()
    if name == "openjev":
        from eval.parity.suites.openjev import build_openjev
        return build_openjev()
    if name == "arxiv":
        from eval.parity.suites.arxiv import build_arxiv
        return build_arxiv()
    raise SystemExit(f"unknown suite {name}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--suites", nargs="+", default=SUITES)
    args = ap.parse_args(argv)
    if os.environ.get("PYTHONHASHSEED") != "0":
        print("warning: set PYTHONHASHSEED=0 for a reproducible openjev gsm8k_mc build", file=sys.stderr)
    DATA.mkdir(parents=True, exist_ok=True)
    mpath = DATA / "MANIFEST.json"
    manifest = json.loads(mpath.read_text()) if mpath.exists() else {}
    for name in args.suites:
        t0 = time.time()
        items, info = build_one(name)
        sha = write_jsonl(DATA / f"{name}.jsonl", items)
        info.update(file=f"eval/parity/data/{name}.jsonl", sha256=sha, built_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
                    build_seconds=round(time.time() - t0, 1))
        manifest[name] = info
        mpath.write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
        print(f"{name}: {len(items)} items -> {sha[:12]} ({info['build_seconds']}s)", flush=True)
    # SHA-256 per eval item, for leakage checks by the data/training agents.
    with open(DATA / "items.sha256", "w") as fh:
        for name in SUITES:
            p = DATA / f"{name}.jsonl"
            if p.exists():
                fh.write(f"{sha256_file(p)}  {p.name}\n")


if __name__ == "__main__":
    main()
