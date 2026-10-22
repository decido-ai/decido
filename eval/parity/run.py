"""Parity benchmark runner (CPU-only).

    python -m eval.parity.run --predictor eval.parity.predictors:random_predictor
    python -m eval.parity.run --predictor mypkg.verdict:Predictor --predictor-kw ckpt=... --latency
    python -m eval.parity.run --predictor eval.parity.predictors:jev_replay --suites arxiv

Scores the six-benchmark group (JevBench, Kev, OpenJev text, Nimble, VitaminC, MASSIVE-en) exactly as
published, plus ECE, option-permutation flip rate, repeat-call determinism and CPU latency.
CUDA and MPS are disabled before the predictor is imported, and device use is audited afterwards.
"""

from __future__ import annotations

# --- force CPU before anything can import torch / mlx --------------------------------------------
import os
import sys


def _early_threads(argv):
    for i, a in enumerate(argv):
        if a == "--threads" and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith("--threads="):
            return a.split("=", 1)[1]
    return None


_T = _early_threads(sys.argv)
# GPU opt-in for INTERMEDIATE runs only (both the flag and the env var are required, so it cannot happen by
# accident). Results are labelled device=GPU in summary.json/report.md and are never the correctness number.
ALLOW_GPU = ("--allow-gpu" in sys.argv) and os.environ.get("VERDICT_ALLOW_GPU") == "1"
if "--allow-gpu" in sys.argv and not ALLOW_GPU:
    raise SystemExit("--allow-gpu also requires VERDICT_ALLOW_GPU=1 in the environment")
if not ALLOW_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["VERDICT_FORCE_CPU"] = "0" if ALLOW_GPU else "1"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
if _T and _T not in ("default", "all"):
    for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
               "NUMEXPR_NUM_THREADS", "RAYON_NUM_THREADS", "ORT_NUM_THREADS"):
        os.environ[_v] = _T


def _patch_torch(threads):
    try:
        import torch
    except ImportError:
        return None
    import functools

    def _false():
        return False
    # lru_cache wrappers keep `__wrapped__`/`cache_clear`, which transformers introspects
    torch.backends.mps.is_available = functools.lru_cache()(_false)   # type: ignore[assignment]
    torch.backends.mps.is_built = functools.lru_cache()(_false)       # type: ignore[assignment]
    if ALLOW_GPU:
        return torch
    torch.cuda.is_available = functools.lru_cache()(_false)           # type: ignore[assignment]
    try:
        torch.set_default_device("cpu")
    except Exception:
        pass
    if threads and threads not in ("default", "all"):
        torch.set_num_threads(int(threads))
    return torch


def _patch_mlx():
    try:
        import mlx.core as mx  # noqa: F401
    except ImportError:
        return None
    mx.set_default_device(mx.cpu)
    return mx


_TORCH = _patch_torch(_T)
_MLX = _patch_mlx()
# -------------------------------------------------------------------------------------------------

import argparse  # noqa: E402
import collections  # noqa: E402
import importlib  # noqa: E402
import inspect  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import random  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

from eval.parity.common import DATA, RESULTS, option_keys, read_jsonl, stable_hash  # noqa: E402
from eval.parity.score import aggregate, bootstrap_six, item_requests, parity_summary, request_golds, score_item  # noqa: E402

SIX = ["jevbench", "kev", "openjev", "nimble", "vitaminc", "massive"]


# ----------------------------------------------------------------------------- predictor loading

class Predictor:
    def __init__(self, spec: str, kwargs: dict):
        mod, _, attr = spec.partition(":")
        obj = getattr(importlib.import_module(mod), attr or "predict")
        if inspect.isclass(obj):
            obj = obj(**kwargs)
        elif kwargs:
            obj = obj(**kwargs)  # factory function
        self.obj = obj
        self.fn = obj.predict if hasattr(obj, "predict") and not inspect.isfunction(obj) else obj
        self.batch_fn = getattr(obj, "predict_batch", None)
        self.needs_gold = bool(getattr(obj, "needs_gold", False))
        self.spec = spec

    def one(self, request, gold=None):
        if self.needs_gold:
            return self.fn(request, gold=gold)
        return self.fn(request)

    def many(self, requests, golds):
        if self.batch_fn is not None and not self.needs_gold:
            out = self.batch_fn(requests)
            if len(out) != len(requests):
                raise RuntimeError("predict_batch returned the wrong number of responses")
            return out
        return [self.one(r, g) for r, g in zip(requests, golds)]


def _safe_many(pred, reqs, golds, errors):
    try:
        return pred.many(reqs, golds)
    except Exception as e:  # fall back to single calls so one bad request cannot sink a batch
        errors.append(f"batch: {type(e).__name__}: {str(e)[:200]}")
        out = []
        for r, g in zip(reqs, golds):
            try:
                out.append(pred.one(r, g))
            except Exception as e2:
                errors.append(f"{type(e2).__name__}: {str(e2)[:200]}")
                out.append({"__error__": str(e2)[:200]})
        return out


# ----------------------------------------------------------------------------- data

def load_suite(name, limit_per_task=None, only_scored=True):
    items = read_jsonl(DATA / f"{name}.jsonl")
    if name == "kev" and only_scored:
        pass  # keep all variants: permuted/none_* rows are cheap and Kev reports them separately
    if limit_per_task:
        by = collections.defaultdict(list)
        for it in items:
            by[it["task"]].append(it)
        items = []
        for t, its in by.items():
            its = sorted(its, key=lambda it: stable_hash(name, it["uid"]))
            items.extend(its[:limit_per_task])
    return items


# ----------------------------------------------------------------------------- core passes

def predict_items(pred, items, batch_size, log_every=2000, errors=None):
    flat, owners = [], []
    for ii, it in enumerate(items):
        reqs = item_requests(it)
        golds = request_golds(it)
        for r, g in zip(reqs, golds):
            flat.append((r, g))
            owners.append(ii)
    responses = [None] * len(flat)
    t0 = time.perf_counter()
    last_log = 0.0
    for s in range(0, len(flat), batch_size):
        chunk = flat[s:s + batch_size]
        out = _safe_many(pred, [c[0] for c in chunk], [c[1] for c in chunk], errors)
        responses[s:s + len(chunk)] = out
        _sample_mps()
        el = time.perf_counter() - t0
        if log_every and el - last_log > 30:
            last_log = el
            print(f"    {s + len(chunk)}/{len(flat)} requests, {(s + len(chunk)) / el:.1f} req/s", flush=True)
    elapsed = time.perf_counter() - t0
    per_item = collections.defaultdict(list)
    for o, r in zip(owners, responses):
        per_item[o].append(None if (isinstance(r, dict) and "__error__" in r) else r)
    units = []
    for ii, it in enumerate(items):
        units.extend(score_item(it, per_item[ii]))
    return units, {"requests": len(flat), "seconds": elapsed, "req_per_s": len(flat) / elapsed if elapsed else None}


def _choice_targets(items):
    """(item, request_index, qid) for choice questions with >= 2 options (incl. NLI choices)."""
    out = []
    for it in items:
        reqs = item_requests(it)
        if it["rule"] == "nli_rerank":
            continue
        for ri, req in enumerate(reqs):
            for qid, q in req["questions"].items():
                if q["type"] == "choice" and len(q["criteria"]) >= 2 and (it["rule"] != "per_question" or f"{ri}:{qid}" in it["golds"]):
                    out.append((it, ri, qid))
    return out


def permutation_test(pred, items, n, k, seed=0):
    """Option-order flip rate: permute a choice question's criteria order k times; flip = argmax option
    (ties -> first listed) differs from the canonical-order argmax."""
    targets = _choice_targets(items)
    rng = random.Random(seed)
    targets = sorted(targets, key=lambda t: stable_hash("perm", t[0]["uid"], t[1], t[2]))[:n]
    flips = trials = 0
    tvs, any_flip, errors = [], 0, 0
    for it, ri, qid in targets:
        req = item_requests(it)[ri]
        q = req["questions"][qid]
        keys = option_keys(q)
        gold = request_golds(it)[ri]
        base = pred.one(req, gold)
        try:
            from eval.parity.common import normalize_answer
            ans = base.get("answers", base) if isinstance(base, dict) else base
            p0, _ = normalize_answer(q, ans[qid])
        except Exception:
            errors += 1
            continue
        lab0 = keys[int(np.argmax(p0))]
        flipped_here = False
        for j in range(k):
            perm = keys[:]
            r2 = random.Random(stable_hash("permk", it["uid"], ri, qid, j, seed))
            for _ in range(10):
                r2.shuffle(perm)
                if perm != keys:
                    break
            q2 = dict(q)
            q2["criteria"] = {kk: q["criteria"][kk] for kk in perm}
            req2 = {"state": req["state"], "questions": {**req["questions"], qid: q2}}
            out = pred.one(req2, gold)
            try:
                ans2 = out.get("answers", out) if isinstance(out, dict) else out
                p2, _ = normalize_answer(q2, ans2[qid])
            except Exception:
                errors += 1
                continue
            lab2 = perm[int(np.argmax(p2))]
            p2c = np.array([p2[perm.index(kk)] for kk in keys])
            tvs.append(0.5 * float(np.abs(p2c - p0).sum()))
            trials += 1
            if lab2 != lab0:
                flips += 1
                flipped_here = True
        any_flip += int(flipped_here)
    del rng
    return {"targets": len(targets), "permutations_per_target": k, "trials": trials,
            "flip_rate": flips / trials if trials else None,
            "share_targets_any_flip": any_flip / len(targets) if targets else None,
            "mean_tv": float(np.mean(tvs)) if tvs else None, "errors": errors}


def determinism_test(pred, items, n):
    flat = []
    for it in items:
        for r, g in zip(item_requests(it), request_golds(it)):
            flat.append((it["uid"], r, g))
    flat = sorted(flat, key=lambda t: stable_hash("det", t[0], json.dumps(t[1], sort_keys=True)[:200]))[:n]
    changed = argchg = 0
    maxd = 0.0
    from eval.parity.common import normalize_answer
    for uid, r, g in flat:
        a, b = pred.one(r, g), pred.one(r, g)
        if a is None or b is None:
            continue
        aa, bb = a.get("answers", a), b.get("answers", b)
        d_here, arg_here = 0.0, False
        for qid, q in r["questions"].items():
            try:
                pa, _ = normalize_answer(q, aa[qid])
                pb, _ = normalize_answer(q, bb[qid])
            except Exception:
                continue
            d_here = max(d_here, float(np.abs(pa - pb).max()))
            arg_here |= int(np.argmax(pa)) != int(np.argmax(pb))
        maxd = max(maxd, d_here)
        changed += d_here > 1e-6
        argchg += arg_here
    return {"requests": len(flat), "share_changed_gt_1e-6": changed / len(flat) if flat else None,
            "share_argmax_changed": argchg / len(flat) if flat else None, "max_abs_dp": maxd}


# ----------------------------------------------------------------------------- latency

def latency_fixture():
    p = DATA / "latency_fixture.json"
    if not p.exists():
        raise SystemExit("missing eval/parity/data/latency_fixture.json (run python -m eval.parity.fixture)")
    return json.loads(p.read_text())


def latency_in_process(pred, n, warmup):
    fx = latency_fixture()
    req = fx["request"]
    for _ in range(warmup):
        pred.one(req, None)
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        pred.one(req, None)
        ts.append(time.perf_counter() - t0)
    ts = np.array(ts) * 1000
    return {"fixture": fx["description"], "state_tokens_qwen3": fx["state_tokens_qwen3"], "n": n, "warmup": warmup,
            "p50_ms": float(np.percentile(ts, 50)), "p95_ms": float(np.percentile(ts, 95)),
            "mean_ms": float(ts.mean()), "min_ms": float(ts.min()),
            "torch_threads": (_TORCH.get_num_threads() if _TORCH is not None else None)}


def latency_subprocess(args, threads):
    cmd = [sys.executable, "-m", "eval.parity.run", "--predictor", args.predictor, "--latency-only",
           "--threads", str(threads), "--latency-n", str(args.latency_n)]
    for kv in args.predictor_kw or []:
        cmd += ["--predictor-kw", kv]
    env = dict(os.environ)
    if threads not in ("default", "all"):
        for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                  "NUMEXPR_NUM_THREADS", "RAYON_NUM_THREADS", "ORT_NUM_THREADS"):
            env[v] = str(threads)
    else:
        for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            env.pop(v, None)
    cp = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(DATA.parents[2]))
    for line in cp.stdout.splitlines()[::-1]:
        if line.startswith("LATENCY_JSON "):
            return json.loads(line[len("LATENCY_JSON "):])
    return {"error": (cp.stderr or cp.stdout)[-1500:]}


# ----------------------------------------------------------------------------- device audit

_MPS_PEAK = [0]


def _sample_mps():
    """Sample MPS allocations during prediction (a predictor could free GPU memory before the final audit)."""
    if "torch" in sys.modules:
        try:
            _MPS_PEAK[0] = max(_MPS_PEAK[0], int(sys.modules["torch"].mps.current_allocated_memory()))
        except Exception:
            pass


def device_audit():
    out = {"CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"), "torch": None, "mlx": None, "violations": [],
           "device": "GPU (intermediate run, --allow-gpu; NOT a CPU correctness number)" if ALLOW_GPU else "CPU-only"}
    if ALLOW_GPU:
        out["cpu_only"] = False
        return out
    if "torch" in sys.modules:
        import torch
        out["torch"] = {"version": torch.__version__, "threads": torch.get_num_threads()}
        try:
            mps_mem = torch.mps.current_allocated_memory()
            out["torch"]["mps_allocated_bytes"] = int(mps_mem)
            if mps_mem > 0:
                out["violations"].append(f"MPS memory allocated: {mps_mem} bytes")
            out["torch"]["mps_peak_sampled_bytes"] = _MPS_PEAK[0]
            if _MPS_PEAK[0] > 0 and mps_mem == 0:
                out["violations"].append(f"MPS memory was allocated during prediction (peak {_MPS_PEAK[0]} bytes)")
        except Exception:
            out["torch"]["mps_allocated_bytes"] = None
    if "mlx.core" in sys.modules:
        import mlx.core as mx
        dev = str(mx.default_device())
        out["mlx"] = {"default_device": dev}
        if "gpu" in dev.lower():
            out["violations"].append(f"MLX default device {dev}")
    if "onnxruntime" in sys.modules:
        out["onnxruntime"] = "loaded (CPUExecutionProvider required; check predictor)"
    out["cpu_only"] = not out["violations"]
    return out


def env_info():
    info = {"python": sys.version.split()[0], "platform": platform.platform(), "machine": platform.machine(),
            "cpu_count": os.cpu_count()}
    try:
        info["cpu"] = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
        info["perf_cores"] = subprocess.run(["sysctl", "-n", "hw.perflevel0.physicalcpu"], capture_output=True, text=True).stdout.strip()
    except Exception:
        pass
    return info


# ----------------------------------------------------------------------------- report

def write_report(path, summary):
    lines = [f"# Parity run: {summary['predictor']}", "", f"- run: {summary['run_name']}  ({summary['started']})",
             f"- limit_per_task: {summary['limit_per_task']} (parity claims require full runs: limit_per_task = None)",
             f"- device: **{summary['device_audit'].get('device')}**; device audit: cpu_only={summary['device_audit']['cpu_only']} {summary['device_audit']['violations']}", ""]
    ps = summary.get("parity")
    if ps:
        lines += ["| group | this predictor | Jev 1.13 | delta | ECE10 | units | missing | invalid |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for s in SIX:
            if s in summary["suites"]:
                a = summary["suites"][s]
                g = ps["per_group"][s]
                cov = a["coverage"]
                d = "n/a" if g["delta"] is None else f"{g['delta']:+.2f}"
                lines.append(f"| {s} | {a['score']:.2f} | {g['jev_1.13']:.2f} | {d} | {a['ece10']:.4f} | "
                             f"{cov['units']} | {cov['missing']} | {cov['invalid']} |")
        if ps.get("six_group_mean") is not None:
            lines += ["", f"**Six-group mean: {ps['six_group_mean']:.2f} vs Jev 1.13 {ps['jev_six_group_mean']:.2f} "
                          f"(gap {ps['gap']:+.2f}); accuracy parity: {ps['accuracy_parity']}; ECE parity: {ps['ece_parity']}; "
                          f"Brier parity: {ps.get('brier_parity')}**"]
        for nm in ("ece_checks", "brier_checks"):
            if ps.get(nm):
                lines += ["", f"| {nm.split('_')[0].upper()} check | this | Jev | ok |", "|---|---:|---:|---|"]
                for k, c in ps[nm].items():
                    lines.append(f"| {k} | {c['verdict']:.4f} | {c['jev']:.4f} | {c['ok']} |")
        if ps.get("bootstrap95"):
            b = ps["bootstrap95"]
            lines += ["", f"Bootstrap 95% CI of the six-group mean (stratified, B={b['B']}): "
                          f"{b['six_group_mean'][0]:.2f}-{b['six_group_mean'][1]:.2f}; P(mean >= 83.85) = {b['p_mean_ge_jev']:.3f}. "
                          "Jev's own run-to-run variation is not included (Jev is nondeterministic; independent runs differ "
                          "by up to 1.6 points on VitaminC)."]
        oj = summary["suites"].get("openjev", {})
        if oj.get("score_tie_free") is not None:
            tr = oj.get("per_task_tie_rate", {})
            lines += ["", f"OpenJev tie audit: official {oj['score']:.2f}, tie-free (random tie-break) {oj['score_tie_free']:.2f}; "
                          f"rerank tie rate gpqa {tr.get('gpqa', float('nan')):.3f}, gpqa_fewshot {tr.get('gpqa_fewshot', float('nan')):.3f}, "
                          f"max over tasks {max(tr.values()) if tr else float('nan'):.3f}"]
        if ps.get("six_group_mean_if_openjev19_bounds"):
            b = ps["six_group_mean_if_openjev19_bounds"]
            lines += ["", f"Six-group mean if OpenJev had its 19th task (GSM8K rerank, 89.5-95.0): {b[0]:.2f}-{b[1]:.2f}"]
    for s, a in summary["suites"].items():
        if s in SIX:
            continue
        lines += ["", f"## {s}: {a['score']:.2f} ({a['metric']})", "", "| task | acc | ECE15 (arXiv def.) |", "|---|---:|---:|"]
        for t, v in a.get("per_task", {}).items():
            lines.append(f"| {t} | {v:.2f} | {a.get('per_task_ece15', {}).get(t, float('nan')):.4f} |")
    if summary.get("permutation"):
        lines += ["", "## Option-permutation flip rate", "", "```", json.dumps(summary["permutation"], indent=1), "```"]
    if summary.get("determinism"):
        lines += ["", "## Repeat-call determinism", "", "```", json.dumps(summary["determinism"], indent=1), "```"]
    if summary.get("latency"):
        lines += ["", "## CPU latency (1k-token state, 5 questions, batch 1)", "", "```", json.dumps(summary["latency"], indent=1), "```"]
    Path(path).write_text("\n".join(lines) + "\n")


from pathlib import Path  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictor", required=True, help="module:callable (see eval/parity/predictors.py)")
    ap.add_argument("--predictor-kw", action="append", help="key=value passed to the predictor class/factory")
    ap.add_argument("--suites", nargs="+", default=SIX)
    ap.add_argument("--limit-per-task", type=int, default=None, help="deterministic subsample per task (dev only)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--threads", default="default")
    ap.add_argument("--perm-n", type=int, default=300, help="choice questions per suite for the flip test (0=off)")
    ap.add_argument("--perm-k", type=int, default=3)
    ap.add_argument("--determinism-n", type=int, default=100)
    ap.add_argument("--latency", action="store_true", help="also run the CPU latency benchmark (subprocesses)")
    ap.add_argument("--latency-threads", default="default,4")
    ap.add_argument("--latency-n", type=int, default=50)
    ap.add_argument("--latency-only", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--bootstrap", type=int, default=500)
    ap.add_argument("--allow-gpu", action="store_true",
                    help="INTERMEDIATE runs only: let the predictor use CUDA (also needs VERDICT_ALLOW_GPU=1)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--name", default=None)
    args = ap.parse_args(argv)

    kwargs = {}
    for kv in args.predictor_kw or []:
        k, _, v = kv.partition("=")
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            pass
        kwargs[k] = v
    t_load = time.perf_counter()
    pred = Predictor(args.predictor, kwargs)
    load_s = time.perf_counter() - t_load

    if args.latency_only:
        res = latency_in_process(pred, args.latency_n, warmup=5)
        res["load_seconds"] = load_s
        res["device_audit"] = device_audit()
        print("LATENCY_JSON " + json.dumps(res))
        return

    name = args.name or (args.predictor.split(":")[-1] + time.strftime("-%Y%m%d-%H%M%S"))
    out_dir = Path(args.out) if args.out else RESULTS / name
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"predictor": args.predictor, "predictor_kw": kwargs, "run_name": name,
               "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "limit_per_task": args.limit_per_task,
               "threads": args.threads, "env": env_info(), "load_seconds": load_s, "suites": {}, "throughput": {},
               "errors": {}, "data_manifest": json.loads((DATA / "MANIFEST.json").read_text())}
    all_units = []
    perm = {}
    for s in args.suites:
        print(f"[{s}] loading", flush=True)
        items = load_suite(s, args.limit_per_task)
        errs = []
        units, thr = predict_items(pred, items, args.batch_size, errors=errs)
        summary["throughput"][s] = thr
        summary["errors"][s] = {"count": len(errs), "first": errs[:5]}
        agg = aggregate(s, units)
        summary["suites"][s] = agg
        all_units.extend(units)
        print(f"[{s}] score={agg['score']:.2f} ece10={agg['ece10']:.4f} coverage={agg['coverage']} "
              f"({thr['requests']} req, {thr['seconds']:.1f}s)", flush=True)
        if args.perm_n and s != "arxiv":
            perm[s] = permutation_test(pred, items, args.perm_n, args.perm_k)
            print(f"[{s}] permutation: {perm[s]}", flush=True)
    if perm:
        tot = sum(p["trials"] for p in perm.values())
        fl = sum((p["flip_rate"] or 0) * p["trials"] for p in perm.values())
        summary["permutation"] = {"per_suite": perm, "pooled_flip_rate": fl / tot if tot else None, "pooled_trials": tot}
    if args.determinism_n:
        det_items = []
        for s in args.suites:
            det_items.extend(load_suite(s, 20))
        summary["determinism"] = determinism_test(pred, det_items, args.determinism_n)
    if any(s in SIX for s in args.suites):
        summary["parity"] = parity_summary(summary["suites"])
        if all(s in args.suites for s in SIX):
            ubs = collections.defaultdict(list)
            for u in all_units:
                ubs[u["suite"]].append(u)
            summary["parity"]["bootstrap95"] = bootstrap_six(ubs, B=args.bootstrap)
    if args.latency:
        summary["latency"] = {}
        for th in args.latency_threads.split(","):
            summary["latency"][f"threads={th}"] = latency_subprocess(args, th)
            print(f"[latency threads={th}] {summary['latency'][f'threads={th}']}", flush=True)
    summary["device_audit"] = device_audit()
    summary["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    with open(out_dir / "units.jsonl", "w") as fh:
        for u in all_units:
            fh.write(json.dumps(u, default=float) + "\n")
    write_report(out_dir / "report.md", summary)
    print(f"wrote {out_dir}/summary.json, report.md, units.jsonl")
    if summary.get("parity", {}).get("six_group_mean") is not None:
        p = summary["parity"]
        print(f"SIX-GROUP MEAN {p['six_group_mean']:.2f} vs Jev {p['jev_six_group_mean']:.2f} (gap {p['gap']:+.2f})")
    if ALLOW_GPU:
        print("NOTE: GPU intermediate run (--allow-gpu); not a CPU-only correctness number")
    elif not summary["device_audit"]["cpu_only"]:
        print("WARNING: CPU-only violated:", summary["device_audit"]["violations"])
        sys.exit(2)


if __name__ == "__main__":
    main()
