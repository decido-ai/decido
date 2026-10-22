"""Backend-agnostic verbalizer readout: request -> token queries -> calibrated distributions.

A backend only has to answer two kinds of query on (prefix_ids + suffix_ids [+ extra_ids]):
  "log-probabilities of these token ids at the next position".
Everything else (labels, variants, permutation averaging, temperature, wire format) lives here so the
vLLM, HF and llama.cpp paths produce the same numbers up to backend numerics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from decido.derive import wire_answer
from decido.template import LabelTrie, question_block


def _lse(xs):
    m = max(xs)
    if m == -math.inf:
        return -math.inf
    return m + math.log(sum(math.exp(x - m) for x in xs))


def cyclic_orders(k: int, m: int) -> list[list[int]]:
    m = max(1, min(m, k))
    shifts = sorted({round(j * k / m) % k for j in range(m)})
    return [[(i + s) % k for i in range(k)] for s in shifts]


@dataclass
class Query:
    qid: str
    order: list[int]
    suffix_ids: list[int]
    trie: LabelTrie
    step1: dict = field(default_factory=dict)            # token id -> logprob
    step2: dict = field(default_factory=dict)            # first symbol -> {token id -> logprob}


@dataclass
class Plan:
    request: dict
    prefix_ids: list[int]
    queries: list[Query]
    truncated_tokens: int = 0


class Readout:
    def __init__(self, chat_format, perm: int = 1, temps: dict | None = None, max_prompt_tokens: int = 16000,
                 on_overflow: str = "error"):
        self.cf = chat_format
        self.tok = chat_format.tok
        self.perm = int(perm)
        self.temps = temps or {}
        self.max_prompt_tokens = max_prompt_tokens
        self.on_overflow = on_overflow       # "error" (Jev-like max_tokens_exceeded) or "truncate" (flagged)
        self._trie_cache = {}
        self._enc_cache = {}

    def trie(self, labels: list[str]) -> LabelTrie:
        key = tuple(labels)
        if key not in self._trie_cache:
            self._trie_cache[key] = LabelTrie(self.tok, labels)
        return self._trie_cache[key]

    def _enc(self, text: str) -> list[int]:
        ids = self._enc_cache.get(text)
        if ids is None:
            ids = self.cf.encode(text)
            if len(self._enc_cache) < 200000:
                self._enc_cache[text] = ids
        return ids

    @staticmethod
    def validate(request: dict):
        """Jev request rules (research/jev-teardown.md §2.2): choice 2-255 options, score 2-10 levels."""
        qs = request.get("questions")
        if not isinstance(qs, dict) or not qs:
            raise ValueError("invalid_request: questions must be a non-empty map")
        for qid, q in qs.items():
            t = q.get("type")
            if t not in ("noul", "choice", "score"):
                raise ValueError(f"invalid_request: question {qid!r} has unsupported type {t!r}")
            if q.get("instructions") in (None, ""):
                raise ValueError(f"invalid_request: question {qid!r} needs instructions")
            if t == "choice" and not (isinstance(q.get("criteria"), dict) and 2 <= len(q["criteria"]) <= 255):
                raise ValueError(f"invalid_request: choice {qid!r} needs 2-255 options")
            if t == "score" and not (isinstance(q.get("criteria"), list) and 2 <= len(q["criteria"]) <= 10):
                raise ValueError(f"invalid_request: score {qid!r} needs 2-10 levels")

    def plan(self, request: dict, perm: int | None = None) -> Plan:
        self.validate(request)
        perm = self.perm if perm is None else perm
        prefix_ids = self.cf.encode(self.cf.prefix_text(request["state"]))
        queries = []
        max_suffix = 0
        for qid, q in request["questions"].items():
            orders = [None]
            if q["type"] == "choice" and perm > 1:
                orders = cyclic_orders(len(q["criteria"]), perm)
            for order in orders:
                text, labels, order_used = question_block(q, order)
                sids = self._enc(self.cf.suffix_text(text))
                max_suffix = max(max_suffix, len(sids))
                queries.append(Query(qid, order_used, sids, self.trie(labels)))
        budget = self.max_prompt_tokens - max_suffix - 4
        truncated = 0
        if len(prefix_ids) > budget:
            if self.on_overflow != "truncate":
                raise ValueError(f"max_tokens_exceeded: state+question needs {len(prefix_ids) + max_suffix} tokens, "
                                 f"limit {self.max_prompt_tokens}")
            closing = self.cf.encode("\n</document>\n\n")  # keep the head of the state; flagged in the response
            truncated = len(prefix_ids) - max(budget - len(closing), 16) - len(closing)
            prefix_ids = prefix_ids[:max(budget - len(closing), 16)] + closing
        pl = Plan(request, prefix_ids, queries)
        pl.truncated_tokens = truncated
        return pl

    # ---------------------------------------------------------------- combination

    def label_logprobs(self, qu: Query) -> np.ndarray:
        tr = qu.trie
        if not tr.two_step:
            s = [_lse([qu.step1.get(t, -math.inf) for t in tr.first_variants[sym]]) for sym in tr.labels]
        else:
            s = []
            for lab in tr.labels:
                a = _lse([qu.step1.get(t, -math.inf) for t in tr.first_variants[lab[0]]])
                b = _lse([qu.step2.get(lab[0], {}).get(t, -math.inf) for t in tr.second_variants[lab[1]]])
                s.append(a + b)
        s = np.array(s, dtype=float)
        if not np.isfinite(s).any():
            s = np.zeros_like(s)
        s = np.where(np.isfinite(s), s, -1e4)
        return s - (s.max() + np.log(np.exp(s - s.max()).sum()))

    def combine(self, plan: Plan, round_dp: int | None = None, raw: bool = False) -> dict:
        req = plan.request
        by_q: dict[str, list[np.ndarray]] = {}
        for qu in plan.queries:
            lp_disp = self.label_logprobs(qu)                   # display positions
            lp = np.empty_like(lp_disp)
            lp[np.array(qu.order)] = lp_disp                     # -> option_keys order
            by_q.setdefault(qu.qid, []).append(lp)
        answers = {}
        for qid, q in req["questions"].items():
            z = np.mean(by_q[qid], axis=0)
            T = float(self.temps.get(q["type"], 1.0))
            z = z / T
            p = np.exp(z - z.max())
            p /= p.sum()
            answers[qid] = p.tolist() if raw else wire_answer(q, p, round_dp)
        out = {"answers": answers}
        if plan.truncated_tokens:
            out["truncated_state_tokens"] = plan.truncated_tokens
        return out
