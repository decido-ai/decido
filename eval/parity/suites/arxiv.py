"""arXiv 2609.37647 replay-validation suite.

Rebuilds a subset of the paper's eval requests with the paper's own MIT harness task classes
(github.com/AppliedMachineLearning-Lab/jev-benchmarking, reviewed before use), so that every request
is byte-identical to what was sent to jev-1.13.0 and can be joined with the Zenodo raw responses
(10.5281/zenodo.23039006, responses.db) by the harness's request hash.

Purpose: prove that this harness's scoring reproduces the paper's published Jev numbers. These
datasets overlap with (but are formatted differently from) the six-benchmark parity suite, so they
are a validation/side-by-side suite, not part of the parity mean.

Licence note: the Jev responses are under the "Jev Responses License 1.0" (research, evaluation,
benchmarking allowed; distillation / imitation training and developing a competing product are
prohibited). They are only ever read by predictors.jev_replay and never enter training data.
"""

from __future__ import annotations

import sys

from eval.parity.common import RAW

HARNESS = RAW / "jev-benchmarking-src"
REPLAY_TASKS = ["anli", "boolq", "paws", "winogrande", "arc", "hellaswag", "ag_news", "sst2", "rotten_tomatoes",
                "banking77"]


def _harness():
    if str(HARNESS) not in sys.path:
        sys.path.insert(0, str(HARNESS))
    from jev_benchmarking.tasks import TASKS  # noqa: PLC0415  (MIT harness, reviewed)
    return TASKS


def build_arxiv(tasks=None) -> tuple[list[dict], dict]:
    TASKS = _harness()
    tasks = tasks or REPLAY_TASKS
    items, counts = [], {}
    for name in tasks:
        task = TASKS[name]
        exs = task.examples(split="eval")
        counts[name] = len(exs)
        for e in exs:
            golds = {}
            for head, kind in task.heads.items():
                g = e.gold[head]
                if isinstance(g, (set, frozenset, list, tuple)):
                    g = sorted(g)[0] if len(g) == 1 else sorted(g)
                golds[f"0:{head}"] = g
            items.append({"uid": f"{name}/{e.uid}", "suite": "arxiv", "task": name, "group": e.group or None,
                          "rule": "per_question", "requests": [{"state": e.state, "questions": e.questions}],
                          "golds": golds, "meta": {"subset": e.subset, "heads": dict(task.heads)}})
    info = {"source": "jev-benchmarking (MIT) task classes, eval split, HF datasets",
            "commit": (RAW / "jev-benchmarking.COMMIT").read_text().strip(), "tasks": counts, "n_items": len(items),
            "paper": "arXiv 2609.37647", "responses": "zenodo 10.5281/zenodo.23039006 responses.db (Jev Responses License 1.0)"}
    return items, info
