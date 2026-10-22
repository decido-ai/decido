"""Reference predictors for harness sanity checks.

Predictor contract (what `--predictor module:callable` must resolve to):
  * a callable `f(request) -> response`, or a class (instantiated with --predictor-kw key=value)
    whose instances are callable or expose `.predict(request)`;
  * optional `.predict_batch(list[request]) -> list[response]` for throughput;
  * optional attribute `needs_gold = True` -> called as f(request, gold={qid: gold}) (oracle only).
  request  = {"state": str|dict|list, "questions": {qid: {"type", "instructions", "criteria"?}}}
  response = Jev wire format {"answers": {qid: {...}}} or the bare answers map, or None (= no answer;
             counted as missing coverage, used by jev_replay for requests Jev never saw).
  answer   = noul: {"noul": p} or float; choice: {"probabilities": {option: p}, "choice"?};
             score: {"probabilities": {"0": p, ...}}; or a list aligned with the option order.
"""

from __future__ import annotations

import json
import os
import sqlite3

import numpy as np

from eval.parity.common import RAW, option_keys, stable_hash


def _req_seed(request) -> int:
    return stable_hash(json.dumps(request, sort_keys=True, ensure_ascii=False))


class RandomPredictor:
    """Deterministic random Dirichlet(1) distribution per (request, question). Expected accuracy = chance."""

    def __call__(self, request):
        out = {}
        for qid, q in request["questions"].items():
            rng = np.random.default_rng(_req_seed(request) ^ stable_hash(qid))
            p = rng.dirichlet(np.ones(len(option_keys(q))))
            out[qid] = p.tolist()
        return {"answers": out}


class UniformPredictor:
    """Uniform distribution: exposes tie-breaking rules (ties -> first option / lexicographic)."""

    def __call__(self, request):
        return {"answers": {qid: (np.ones(len(option_keys(q))) / len(option_keys(q))).tolist()
                            for qid, q in request["questions"].items()}}


class FirstOptionPredictor:
    """All mass on the first listed option: measures how much a benchmark rewards position bias."""

    def __call__(self, request):
        out = {}
        for qid, q in request["questions"].items():
            p = np.zeros(len(option_keys(q)))
            p[0] = 1.0
            out[qid] = p.tolist()
        return {"answers": out}


class OraclePredictor:
    """Puts `confidence` on the gold option (rest spread evenly). Accuracy must be 100%, and top-label
    ECE must equal 1 - confidence. Proves golds/argmax/aggregation are wired correctly."""

    needs_gold = True

    def __init__(self, confidence: float = 1.0):
        self.confidence = float(confidence)

    def __call__(self, request, gold=None):
        out = {}
        for qid, q in request["questions"].items():
            keys = option_keys(q)
            n = len(keys)
            if gold is None or qid not in gold:
                out[qid] = (np.ones(n) / n).tolist()
                continue
            g = gold[qid]
            gi = int(g) if q["type"] == "noul" else (int(g) if q["type"] == "score" else keys.index(g))
            p = np.full(n, (1 - self.confidence) / max(1, n - 1))
            p[gi] = self.confidence
            out[qid] = p.tolist()
        return {"answers": out}


class JevReplay:
    """Replays jev-1.13.0 responses from the arXiv 2609.37647 Zenodo database, keyed exactly like the
    paper's harness: sha256(json.dumps({"model", "state", "questions"}, ensure_ascii=False)).
    Returns None for requests Jev was never sent (all six parity suites use different requests).
    Evaluation use only (Jev Responses License 1.0 forbids training/imitation use)."""

    def __init__(self, db: str | None = None, model: str = "jev-1.13.0"):
        path = os.path.abspath(db or str(RAW / "zenodo" / "responses.db"))
        self.db = sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True, check_same_thread=False)
        self.model = model

    def key(self, request):
        import hashlib
        payload = json.dumps({"model": self.model, "state": request["state"], "questions": request["questions"]},
                             ensure_ascii=False)
        return hashlib.sha256(payload.encode()).hexdigest()

    def __call__(self, request):
        row = self.db.execute("SELECT response FROM responses WHERE key = ?", (self.key(request),)).fetchone()
        if row is None:
            return None
        return json.loads(row[0])

    def predict_batch(self, requests):
        return [self(r) for r in requests]


random_predictor = RandomPredictor
uniform_predictor = UniformPredictor
first_option_predictor = FirstOptionPredictor
oracle_predictor = OraclePredictor
jev_replay = JevReplay


class JevBenchRowsReplay:
    """Replays a published per-row JevBench run (official jevbench runner output: task_id + probs over the
    exact labels, e.g. Kev's runs/jevbench-public/<model>/{original,easy,hard}.jsonl). Used to check that
    this harness's JevBench scoring reproduces the official summaries for the same rows."""

    def __init__(self, run_dir: str):
        from eval.parity.common import DATA, read_jsonl
        self.by_hash = {}
        items = {it["uid"]: it for it in read_jsonl(DATA / "jevbench.jsonl")}
        rows = {}
        for tier in ("original", "easy", "hard"):
            for r in read_jsonl(os.path.join(run_dir, f"{tier}.jsonl")):
                rows[r["task_id"]] = r
        for uid, it in items.items():
            r = rows.get(uid)
            if r is None or not r.get("ok") or not isinstance(r.get("probs"), dict):
                continue
            req = it["requests"][0]
            q = req["questions"]["decision"]
            probs = r["probs"]
            if q["type"] == "noul":
                ans = {"noul": float(probs["yes"])}
            else:
                ans = {"probabilities": {str(k): float(v) for k, v in probs.items()}}
            self.by_hash[json.dumps(req, sort_keys=True, ensure_ascii=False)] = {"answers": {"decision": ans}}

    def __call__(self, request):
        return self.by_hash.get(json.dumps(request, sort_keys=True, ensure_ascii=False))


jevbench_rows_replay = JevBenchRowsReplay
