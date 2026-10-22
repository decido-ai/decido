"""Shared paths, item format, response normalization and metrics for the parity harness.

Item format (one JSON object per line in eval/parity/data/<suite>.jsonl):

    {"uid": str, "suite": str, "task": str, "group": str|None,
     "rule": "per_question" | "nli_cls" | "nli_rerank",
     "requests": [{"state": ..., "questions": {qid: {type, instructions, criteria}}}],
     "golds": {"<request_index>:<qid>": gold} | {"rerank": int},
     "meta": {...}}

gold is an option key (choice), a bool (noul) or an int level (score).
A "unit" is one scored decision: one question (per_question / nli_cls) or one reranked
multiple-choice item (nli_rerank). Suites aggregate units into their published score.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]          # repository root
RAW = ROOT / "data" / "raw" / "parity"
HF_CACHE = ROOT / "data" / "hf-cache"
DATA = Path(__file__).resolve().parent / "data"
RESULTS = Path(__file__).resolve().parent / "results"

SUM_TOL_STRICT = 1e-3
SUM_TOL_RENORM = 2e-2   # JevBench rule: rounded distributions inside this band are renormalized

NLI_LABELS = ("entailment", "neutral", "contradiction")
# Same option wording as the arXiv 2609.37647 Jev NLI template (MIT harness, tasks/nli.py).
NLI_CRITERIA = {
    "entailment": "`hypothesis` is definitely true given `premise`.",
    "neutral": "`hypothesis` may or may not be true; `premise` does not settle it.",
    "contradiction": "`hypothesis` is definitely false given `premise`.",
}
NLI_INSTRUCTIONS = "What is the relationship between `premise` and `hypothesis`?"


def nli_request(premise: str, hypothesis: str) -> dict:
    return {"state": {"premise": premise, "hypothesis": hypothesis},
            "questions": {"relation": {"type": "choice", "instructions": NLI_INSTRUCTIONS,
                                       "criteria": dict(NLI_CRITERIA)}}}


# ----------------------------------------------------------------------------- IO

def read_jsonl(path) -> list[dict]:
    with open(path, encoding="utf-8", newline="\n") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path, rows) -> str:
    """Write rows; return sha256 of the written bytes."""
    payload = "".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in rows).encode()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_hash(*parts) -> int:
    return int(hashlib.sha256("/".join(map(str, parts)).encode()).hexdigest()[:15], 16)


# ----------------------------------------------------------------------------- questions

def option_keys(q: dict) -> list[str]:
    """Canonical option keys in request order."""
    t = q["type"]
    if t == "noul":
        return ["false", "true"]
    if t == "choice":
        return list(q["criteria"].keys())
    if t == "score":
        return [str(i) for i in range(len(q["criteria"]))]
    raise ValueError(f"unknown question type {t!r}")


def gold_index(q: dict, gold) -> int:
    t = q["type"]
    if t == "noul":
        if not isinstance(gold, bool):
            raise ValueError("noul gold must be bool")
        return int(gold)
    if t == "score":
        return int(gold)
    return option_keys(q).index(gold)


class InvalidAnswer(ValueError):
    pass


def normalize_answer(q: dict, ans) -> tuple[np.ndarray, int | None]:
    """Map one predictor answer onto a probability vector over option_keys(q).

    Accepts Jev wire answers ({"type", "noul"} / {"probabilities": {...}, "choice"?}),
    a bare float for noul, a {key: p} dict, or a list aligned with option_keys(q).
    Returns (p, declared_index) where declared_index is the predictor's own argmax
    ("choice" field) when it is consistent with p, else None.
    Raises InvalidAnswer outside the JevBench rounding tolerance.
    """
    keys = option_keys(q)
    declared = None
    if isinstance(ans, dict) and "noul" in ans and q["type"] == "noul":
        ans = float(ans["noul"])
    elif isinstance(ans, dict) and "probabilities" in ans:
        declared = ans.get("choice") if q["type"] == "choice" else None
        ans = ans["probabilities"]
    if isinstance(ans, (int, float)) and not isinstance(ans, bool) and q["type"] == "noul":
        p_true = float(ans)
        if not (0.0 <= p_true <= 1.0) or not math.isfinite(p_true):
            raise InvalidAnswer(f"noul out of range: {p_true}")
        p = np.array([1.0 - p_true, p_true])
    elif isinstance(ans, dict):
        got = {str(k): v for k, v in ans.items()}
        if set(got) != set(keys):
            raise InvalidAnswer(f"label keys mismatch: missing={sorted(set(keys) - set(got))} extra={sorted(set(got) - set(keys))}")
        p = np.array([float(got[k]) for k in keys])
    elif isinstance(ans, (list, tuple, np.ndarray)):
        p = np.asarray(ans, dtype=float)
        if p.shape != (len(keys),):
            raise InvalidAnswer(f"expected {len(keys)} probabilities, got {p.shape}")
    else:
        raise InvalidAnswer(f"unparseable answer {type(ans).__name__}")
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1 + 1e-9).any():
        raise InvalidAnswer("probabilities must be finite and in [0, 1]")
    s = p.sum()
    if abs(s - 1.0) > SUM_TOL_RENORM or s <= 0:
        raise InvalidAnswer(f"probabilities sum to {s:.4f}")
    p = p / s
    decl_idx = None
    if declared is not None and declared in keys:
        i = keys.index(declared)
        if p[i] >= p.max() - 1e-9:
            decl_idx = i
    return p, decl_idx


def argmax_index(p: np.ndarray, keys: list[str], tie: str = "first", declared: int | None = None) -> int:
    """Tie rules: 'first' (np.argmax; Kev, OpenJev, Nimble), 'lex' (lexicographically smallest
    label; JevBench), 'declared' (predictor's own choice field if consistent; arXiv harness uses
    Jev's `choice`), 'noul_true' (p_true >= 0.5 -> true; arXiv harness binary rule)."""
    if tie == "declared" and declared is not None:
        return declared
    m = p.max()
    ties = [i for i in range(len(p)) if p[i] >= m - 1e-12]
    if len(ties) == 1:
        return ties[0]
    if tie == "lex":
        return min(ties, key=lambda i: keys[i])
    if tie == "noul_true" and len(p) == 2:
        return 1
    return ties[0]


# ----------------------------------------------------------------------------- metrics

def ece_top_label(conf, correct, n_bins: int = 10) -> float:
    """Top-label ECE, equal-width bins, last bin closed (JevBench/Kev/Nimble convention)."""
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=float)
    if len(conf) == 0:
        return float("nan")
    idx = np.minimum((np.clip(conf, 0, 1) * n_bins).astype(int), n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            total += m.sum() * abs(conf[m].mean() - correct[m].mean())
    return float(total / len(conf))


def brier_multiclass(p: np.ndarray, gi: int) -> float:
    y = np.zeros_like(p)
    y[gi] = 1.0
    return float(((p - y) ** 2).sum())


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)
