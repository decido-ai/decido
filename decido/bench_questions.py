"""Cost vs number of questions on one state (state-once prefix cache), CPU llama.cpp.

    python -m decido.bench_questions --gguf ... --bundle ... --threads 10 --counts 1,5,20 --reps 3
Uses the frozen latency fixture state (eval/parity/data/latency_fixture.json, ~1k tokens) and cycles its 5 questions
(2 noul, 2 choice, 1 score) with distinct wording to reach N questions. Reports wall time, prefill time, per-question
suffix time, and the no-cache cost estimate (N x full prompt).
"""
import argparse
import json
import time

from decido.predictor_llamacpp import VerdictLlamaCpp

ap = argparse.ArgumentParser()
ap.add_argument("--gguf", required=True)
ap.add_argument("--bundle", required=True)
ap.add_argument("--threads", type=int, default=10)
ap.add_argument("--counts", default="1,5,20")
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--out", default=None)
a = ap.parse_args()
fx = json.load(open("eval/parity/data/latency_fixture.json"))
base_qs = list(fx["request"]["questions"].items())
p = VerdictLlamaCpp(gguf=a.gguf, bundle=a.bundle, threads=a.threads)
res = []
for n in map(int, a.counts.split(",")):
    qs = {}
    for i in range(n):
        k, q = base_qs[i % len(base_qs)]
        q = dict(q)
        if i >= len(base_qs):
            q["instructions"] = (q["instructions"] if isinstance(q["instructions"], str) else json.dumps(q["instructions"])) + f" (variant {i // len(base_qs)})"
        qs[f"{k}_{i}"] = q
    req = {"state": fx["request"]["state"], "questions": qs}
    for r in range(a.reps + 1):
        # force a fresh prefix (different first token) so every rep pays the state prefill
        t0 = time.perf_counter()
        out = p({"state": {"rep": r, "doc": req["state"]}, "questions": qs})
        dt = time.perf_counter() - t0
        if r == 0:
            continue  # warmup
        tm = p.last_timing
        res.append({"n_questions": n, "rep": r, "wall_ms": dt * 1000, "prefill_ms": tm["prefill_ms"], "queries_ms": tm["queries_ms"],
                    "n_prefix": tm["n_prefix"], "reused": tm["reused"]})
        print(json.dumps(res[-1]), flush=True)
summ = {}
for n in sorted({r["n_questions"] for r in res}):
    rr = [r for r in res if r["n_questions"] == n]
    summ[n] = {k: sum(r[k] for r in rr) / len(rr) for k in ("wall_ms", "prefill_ms", "queries_ms")}
print(json.dumps(summ, indent=1))
if a.out:
    json.dump({"runs": res, "summary": summ, "threads": a.threads, "gguf": a.gguf}, open(a.out, "w"), indent=1)
