"""Jev 1.13 wire answers + derived fields (formulas from research/jev-teardown.md §4), computed unrounded."""

from __future__ import annotations

import numpy as np


def choice_confidence(p) -> float:
    n = len(p)
    if n <= 1:
        return 1.0
    return float((n * max(p) - 1) / (n - 1))


def score_confidence(p) -> float:
    p = np.asarray(p, dtype=float)
    n = len(p)
    if n <= 1:
        return 1.0
    m = int(np.argmax(p))
    idx = np.arange(n)
    mad_unif = float(np.mean(np.abs(idx - (n - 1) / 2)))
    return float(max(0.0, 1.0 - float((p * np.abs(idx - m)).sum()) / mad_unif))


def wire_answer(q: dict, p, round_dp: int | None = None) -> dict:
    """p is aligned with eval.parity.common.option_keys(q)."""
    p = np.asarray(p, dtype=float)
    r = (lambda x: round(float(x), round_dp)) if round_dp is not None else float
    t = q["type"]
    if t == "noul":
        return {"type": "noul", "noul": r(p[1])}
    if t == "choice":
        keys = list(q["criteria"].keys())
        return {"type": "choice", "choice": keys[int(np.argmax(p))],
                "probabilities": {k: r(v) for k, v in zip(keys, p)}, "confidence": r(choice_confidence(p))}
    if t == "score":
        levels = list(q["criteria"])
        return {"type": "score", "score": r(float((np.arange(len(p)) * p).sum())),
                "legend": {str(i): levels[i] for i in range(len(levels))},
                "probabilities": {str(i): r(v) for i, v in enumerate(p)}, "confidence": r(score_confidence(p))}
    raise ValueError(t)
