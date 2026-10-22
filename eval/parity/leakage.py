"""Write leakage-control artifacts for the training/data agents, and gate training data against them.

    python -m eval.parity.leakage                       # (re)write hashes + manifest (original behaviour)
    python -m eval.parity.leakage check IN.jsonl --out OUT.jsonl --report REPORT.json [--workers 40]

`check` drops every training row (unified schema: state/questions/targets/meta) for which any text unit
(state string leaves and whole state, instructions, criteria keys/descriptions) after _norm():
  (a) hash-matches eval_text_hashes.txt,
  (b) shares any word-level 13-gram with any eval text (six parity suites + arxiv suite), or
  (c) has estimated MinHash Jaccard >= 0.7 on word 5-gram shingles with an eval text (LSH, 128 perms,
      32 bands x 4 rows; candidates verified on the full signature).

Outputs
  eval/parity/data/eval_text_hashes.txt   sha256 of every normalized eval text unit (state strings, premises,
                                          hypotheses, option texts >= 20 chars), one per line, sorted, unique
  eval/parity/data/leakage_manifest.json  upstream datasets/splits that are eval-only, and rules
  data/leakage_log.md                     append-only log (project-wide)
"""

from __future__ import annotations

import hashlib
import json
import re
import time

from eval.parity.common import DATA, ROOT, read_jsonl

SUITES = ["jevbench", "kev", "openjev", "nimble", "vitaminc", "massive", "arxiv"]

EVAL_ONLY_SOURCES = {
    "jevbench": ["fstandhartinger/jevbench datasets/public/* (all 231)", "also: JevBench/jevbench HF (jevbench-240) - different set, treat as eval-only"],
    "kev": ["jaredpalmer/kev evals/v7/decision-v7/{development,test}", "evals/v4/transfer-v4/{development,test}",
            "evals/v9/transfer-v9/{development,test}",
            "underlying public test items: banking77 test, ag_news test, MNLI, TREC, DBpedia-14, BoolQ, SST-5, Yelp, Amazon reviews, IMDB, MMLU test, MMLU-Pro, emotion, SciQ, tweet_eval offensive, QNLI, PAWS (see _meta.repo/split per record)",
            "jaredpalmer/kev-suites (HF) decision/transfer development+test splits: eval-only; their train splits share templates with the eval suites - do not use without the 13-gram gate"],
    "openjev": ["MNLI validation_matched/mismatched", "ANLI test_r1-3", "WANLI test", "SciTail snli_format test",
                "ConTRoL-nli test", "ARC-Easy/Challenge test", "WinoGrande xl validation", "GSM8K test", "GPQA diamond",
                "HellaSwag validation", "MMLU test (+ MMLU dev used as few-shot demos)"],
    "nimble": ["bespokelabsai/nimble data/eval.jsonl (324)", "do not train on bespokelabsai/nimble data/train.jsonl either: same generator, Jev-style synthetic, license unclear"],
    "vitaminc": ["VitaminC dev (all 63,054 rows; the 599 eval ids are drawn from it) - VitaminC train is allowed only after the 13-gram gate (same Wikipedia revisions can recur)"],
    "massive": ["MASSIVE 1.1 test partition, all locales (ids are shared across locales)"],
    "arxiv": ["arXiv 2609.37647 eval splits of anli, boolq, paws, winogrande, arc, hellaswag, ag_news, sst2, rotten_tomatoes, banking77",
              "Zenodo 10.5281/zenodo.23039006 responses (Jev Responses License 1.0: never as training labels/targets)"],
}

RULES = [
    "Never train on any item whose normalized text hash is in eval_text_hashes.txt.",
    "Run a 13-gram overlap gate (and MinHash Jaccard >= 0.7 on 5-gram shingles) of every training shard against the eval texts; drop hits and log counts.",
    "Do not put eval label inventories, eval instructions or eval criteria wording verbatim into synthetic-data generator prompts.",
    "Never use any Jev output (Zenodo responses, jev-distill-corpus, Kev/JevBench Jev runs) as training labels (TypeSafe MCA 2.3(b); Jev Responses License 1.0).",
    "Tune templates/temperatures only on non-eval data (e.g. Kev calibration split is NOT eval but shares generators: gate it).",
]


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def _texts(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _texts(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _texts(v)


def main():
    hashes = set()
    counts = {}
    for s in SUITES:
        p = DATA / f"{s}.jsonl"
        if not p.exists():
            continue
        n0 = len(hashes)
        for it in read_jsonl(p):
            if "nli" in it:
                parts = [it["nli"]["premise"], *it["nli"]["hyps"]]
            else:
                parts = []
                for r in it["requests"]:
                    st = r["state"]
                    parts.append(st if isinstance(st, str) else json.dumps(st, ensure_ascii=False, sort_keys=True))
                    parts.extend(t for t in _texts(st))
            for t in parts:
                if isinstance(t, str) and len(t) >= 20:
                    hashes.add(hashlib.sha256(_norm(t).encode()).hexdigest())
        counts[s] = len(hashes) - n0
    (DATA / "eval_text_hashes.txt").write_text("\n".join(sorted(hashes)) + "\n")
    manifest = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "normalization": "lowercase, collapse whitespace, strip; sha256",
                "n_hashes": len(hashes), "new_hashes_per_suite": counts, "eval_only_sources": EVAL_ONLY_SOURCES, "rules": RULES}
    (DATA / "leakage_manifest.json").write_text(json.dumps(manifest, indent=1))
    log = ROOT / "data" / "leakage_log.md"
    new = not log.exists()
    with open(log, "a") as fh:
        if new:
            fh.write("# Verdict leakage log (append-only)\n\n")
        fh.write(f"- {manifest['created']} parity-bench: froze 6 parity suites + arxiv replay suite; "
                 f"{len(hashes)} eval text hashes in eval/parity/data/eval_text_hashes.txt; eval-only sources and rules in "
                 f"eval/parity/data/leakage_manifest.json. No training performed by parity-bench. Jev responses (Zenodo) used "
                 f"only for harness replay validation.\n")
    print(f"{len(hashes)} hashes; {counts}")


# ----------------------------------------------------------------------------- training-data gate

import os  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402

_WORD = re.compile(r"\w+")
_RNG = np.random.default_rng(1234567)
_POS_MUL = (_RNG.integers(1, 2**63 - 1, size=13, dtype=np.int64).astype(np.uint64) * np.uint64(2) + np.uint64(1))
N_PERM, BANDS, ROWS = 128, 32, 4
_MH_A = (_RNG.integers(1, 2**63 - 1, size=N_PERM, dtype=np.int64).astype(np.uint64) * np.uint64(2) + np.uint64(1))
_MH_B = _RNG.integers(0, 2**63 - 1, size=N_PERM, dtype=np.int64).astype(np.uint64)
_BAND_MUL = (_RNG.integers(1, 2**63 - 1, size=ROWS, dtype=np.int64).astype(np.uint64) * np.uint64(2) + np.uint64(1))
MH_MIN_WORDS = 6
JACCARD = 0.7
# template strings shared by construction between eval requests and our own NLI-format training requests;
# they carry no item content, so they are not indexed as eval text
_TEMPLATE_TEXTS = None


def _templates():
    global _TEMPLATE_TEXTS
    if _TEMPLATE_TEXTS is None:
        from eval.parity.common import NLI_CRITERIA, NLI_INSTRUCTIONS
        _TEMPLATE_TEXTS = {_norm(t) for t in [*NLI_CRITERIA.values(), NLI_INSTRUCTIONS]}
    return _TEMPLATE_TEXTS


_WH: dict = {}


def _whash(w: str) -> int:
    """Deterministic 64-bit word hash (Python's hash() is salted per process, which made LSH runs differ)."""
    h = _WH.get(w)
    if h is None:
        h = int.from_bytes(hashlib.blake2b(w.encode(), digest_size=8).digest(), "little")
        if len(_WH) < 5_000_000:
            _WH[w] = h
    return h


def _word_ids(text: str) -> np.ndarray:
    ws = _WORD.findall(text.lower())
    if not ws:
        return np.zeros(0, dtype=np.uint64)
    return np.array([_whash(w) for w in ws], dtype=np.uint64)


def _ngrams(ids: np.ndarray, n: int) -> np.ndarray:
    m = len(ids) - n + 1
    if m <= 0:
        return np.zeros(0, dtype=np.uint64)
    h = np.zeros(m, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for j in range(n):
            h += ids[j:j + m] * _POS_MUL[j]
    return h


def _minhash(shingles: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        v = shingles[:, None] * _MH_A[None, :] + _MH_B[None, :]
    return (v >> np.uint64(32)).min(axis=0)


def _band_keys(sig: np.ndarray) -> np.ndarray:
    s = sig.reshape(BANDS, ROWS)
    with np.errstate(over="ignore"):
        k = (s * _BAND_MUL[None, :]).sum(axis=1)
        k = k * np.uint64(0x9E3779B97F4A7C15) + np.arange(BANDS, dtype=np.uint64)
    return k


def _eval_text_units():
    """Unique normalized eval texts (content only)."""
    tmpl = _templates()
    texts = set()
    for s in SUITES:
        p = DATA / f"{s}.jsonl"
        if not p.exists():
            continue
        for it in read_jsonl(p):
            if "nli" in it:
                parts = [it["nli"]["premise"], *it["nli"]["hyps"]]
            else:
                parts = []
                for r in it["requests"]:
                    st = r["state"]
                    parts.append(st if isinstance(st, str) else json.dumps(st, ensure_ascii=False, sort_keys=True))
                    parts.extend(_texts(st))
                    for q in r["questions"].values():
                        parts.extend(_texts(q.get("instructions")))
                        crit = q.get("criteria")
                        if isinstance(crit, dict):
                            parts.extend(_texts(list(crit.values())))
                        elif crit is not None:
                            parts.extend(_texts(crit))
            for t in parts:
                if isinstance(t, str):
                    n = _norm(t)
                    if len(n) >= 20 and n not in tmpl:
                        texts.add(n)
    return sorted(texts)


_IDX = {}


def build_index():
    t0 = time.time()
    hashes = set((DATA / "eval_text_hashes.txt").read_text().split())
    texts = _eval_text_units()
    grams, sigs = [], []
    for t in texts:
        ids = _word_ids(t)
        g = _ngrams(ids, 13)
        if len(g):
            grams.append(g)
        if len(ids) >= MH_MIN_WORDS:
            sigs.append(_minhash(_ngrams(ids, 5)))
    g13 = np.unique(np.concatenate(grams)) if grams else np.zeros(0, dtype=np.uint64)
    sig = np.stack(sigs) if sigs else np.zeros((0, N_PERM), dtype=np.uint64)
    keys = np.stack([_band_keys(x) for x in sig]) if len(sig) else np.zeros((0, BANDS), dtype=np.uint64)
    flat_k = keys.reshape(-1)
    flat_i = np.repeat(np.arange(len(sig), dtype=np.int64), BANDS)
    o = np.argsort(flat_k, kind="stable")
    _IDX.update(hashes=hashes, g13=g13, sig=sig, bk=flat_k[o], bi=flat_i[o])
    return {"eval_texts": len(texts), "eval_13grams": int(len(g13)), "eval_minhash_texts": int(len(sig)),
            "index_seconds": round(time.time() - t0, 1)}


def _row_texts(row):
    out = []
    st = row.get("state")
    out.append(st if isinstance(st, str) else json.dumps(st, ensure_ascii=False, sort_keys=True))
    out.extend(_texts(st))
    for q in (row.get("questions") or {}).values():
        out.extend(_texts(q.get("instructions")))
        crit = q.get("criteria")
        if isinstance(crit, dict):
            out.extend(k for k in crit.keys())
            out.extend(_texts(list(crit.values())))
        elif crit is not None:
            out.extend(_texts(crit))
    return out


def check_row(row) -> str | None:
    """Return the first leakage reason for a row, or None if it is clean."""
    tmpl = _templates()
    hashes, g13, sig, bk, bi = _IDX["hashes"], _IDX["g13"], _IDX["sig"], _IDX["bk"], _IDX["bi"]
    seen = set()
    for t in _row_texts(row):
        if not isinstance(t, str):
            continue
        n = _norm(t)
        if n in seen or n in tmpl:
            continue
        seen.add(n)
        if len(n) >= 20 and hashlib.sha256(n.encode()).hexdigest() in hashes:
            return "exact_hash"
        ids = _word_ids(n)
        g = _ngrams(ids, 13)
        if len(g) and len(g13):
            pos = np.searchsorted(g13, g)
            pos[pos >= len(g13)] = 0
            if (g13[pos] == g).any():
                return "ngram13"
        if len(ids) >= MH_MIN_WORDS and len(sig):
            s = _minhash(_ngrams(ids, 5))
            ks = _band_keys(s)
            lo = np.searchsorted(bk, ks, "left")
            hi = np.searchsorted(bk, ks, "right")
            cand = set()
            for a, b in zip(lo, hi):
                if b > a:
                    cand.update(bi[a:min(b, a + 2000)].tolist())
            if cand:
                c = np.fromiter(cand, dtype=np.int64)
                est = (sig[c] == s[None, :]).mean(axis=1)
                if (est >= JACCARD).any():
                    return "minhash"
    return None


def _check_chunk(lines):
    out = []
    for line in lines:
        row = json.loads(line)
        out.append((line, check_row(row), (row.get("meta") or {}).get("source", "?")))
    return out


def check_main(argv):
    import argparse
    import collections
    import multiprocessing as mp
    ap = argparse.ArgumentParser(prog="python -m eval.parity.leakage check")
    ap.add_argument("inp", nargs="+")
    ap.add_argument("--out", required=True, help="output jsonl (or directory if several inputs)")
    ap.add_argument("--report", required=True)
    ap.add_argument("--workers", type=int, default=min(40, os.cpu_count() or 4))
    ap.add_argument("--dropped", default=None, help="optional jsonl of dropped rows with reasons")
    a = ap.parse_args(argv)
    info = build_index()
    print("index:", info, flush=True)
    multi = len(a.inp) > 1 or os.path.isdir(a.out)
    rep = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "index": info, "rules": RULES[:3], "files": {}}
    tot_in = tot_drop = 0
    by_reason_all = collections.Counter()
    ctx = mp.get_context("fork")
    with ctx.Pool(a.workers) as pool:
        for inp in a.inp:
            outp = os.path.join(a.out, os.path.basename(inp)) if multi else a.out
            os.makedirs(os.path.dirname(os.path.abspath(outp)), exist_ok=True)
            with open(inp) as fh:
                lines = fh.readlines()
            chunks = [lines[i:i + 500] for i in range(0, len(lines), 500)]
            by_src = collections.defaultdict(collections.Counter)
            dfh = open(a.dropped, "a") if a.dropped else None
            n_in = n_drop = 0
            with open(outp, "w") as ofh:
                for res in pool.imap(_check_chunk, chunks, chunksize=1):
                    for line, reason, src in res:
                        n_in += 1
                        by_src[src]["rows"] += 1
                        if reason:
                            n_drop += 1
                            by_src[src][reason] += 1
                            by_reason_all[reason] += 1
                            if dfh:
                                dfh.write(json.dumps({"reason": reason, "row": json.loads(line)}) + "\n")
                        else:
                            ofh.write(line if line.endswith("\n") else line + "\n")
            if dfh:
                dfh.close()
            rep["files"][inp] = {"rows_in": n_in, "dropped": n_drop, "by_source": {k: dict(v) for k, v in by_src.items()}}
            tot_in += n_in
            tot_drop += n_drop
            print(f"{inp}: {n_in} rows, dropped {n_drop}", flush=True)
    rep.update(rows_in=tot_in, dropped=tot_drop, by_reason=dict(by_reason_all))
    with open(a.report, "w") as fh:
        json.dump(rep, fh, indent=1)
    log = ROOT / "data" / "leakage_log.md"
    with open(log, "a") as fh:
        fh.write(f"- {rep['created']} leakage check: {len(a.inp)} file(s), {tot_in} rows in, {tot_drop} dropped "
                 f"({dict(by_reason_all)}); gate = exact hash + 13-gram + MinHash J>={JACCARD} vs "
                 f"{info['eval_texts']} eval texts (six suites + arxiv); report {a.report}\n")
    print(json.dumps({k: rep[k] for k in ("rows_in", "dropped", "by_reason")}))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        check_main(sys.argv[2:])
    else:
        main()
