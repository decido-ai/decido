"""Unit scoring and suite aggregation, matching each benchmark's published scoring rule.

Published Jev 1.13 reference numbers (sources in research/parity-bench.md):
  six-group (AutoTrust run, JEV-27B card 2026-09-27): JevBench 87.18, Kev 85.52, OpenJev text 72.96,
  Nimble 91.84, VitaminC 78.46, MASSIVE-en 87.14, mean 83.85.
"""

from __future__ import annotations

import collections
import math

import numpy as np

from eval.parity.common import (NLI_LABELS, InvalidAnswer, argmax_index, brier_multiclass, ece_top_label, gold_index,
                                normalize_answer, option_keys, wilson)

JEV_SIX = {"jevbench": 87.18, "kev": 85.52, "openjev": 72.96, "nimble": 91.84, "vitaminc": 78.46, "massive": 87.14}
JEV_SIX_MEAN = 83.85
# Published Jev ECE (top-label, 10 equal-width bins unless noted) per benchmark.
JEV_ECE = {
    "jevbench": (0.0318, "Open-Jev hosted run of jev-1.13.0 on the same 231 items (reports/jevbench-public-20260921)"),
    "kev:transfer-v4/development": (0.0492, "Kev repo runs/jev-transfer-v4/report.json"),
    "kev:transfer-v9/development": (0.0335, "Kev repo runs/jev-transfer-v9/report.json"),
    "vitaminc": (0.104, "Bespoke Nimble PUBLIC_BENCHMARKS.md, jev-1.13.0 on the same 599 ids"),
    "massive": (0.075, "Bespoke Nimble PUBLIC_BENCHMARKS.md, jev-1.13.0 on the same 350 ids"),
}
# Published Jev Brier (multiclass sum over options; Noul as a 2-vector), same items. Brier is a proper scoring rule,
# so unlike ECE it cannot be gamed by a uniform predictor (uniform passes the JevBench and MASSIVE ECE checks).
JEV_BRIER = {
    "jevbench": (0.1811, "Open-Jev hosted run, brier_mean"),
    "kev:transfer-v4/development": (0.2110, "Kev runs/jev-transfer-v4"),
    "kev:transfer-v9/development": (0.2121, "Kev runs/jev-transfer-v9"),
    "vitaminc": (0.313, "Bespoke Nimble public suite"),
    "massive": (0.204, "Bespoke Nimble public suite"),
}
# Other published Jev accuracies on exactly these items (independent runs; Jev is nondeterministic).
JEV_ALT = {
    "jevbench": {"family_macro": 87.78, "per_example": 86.58, "src": "Open-Jev run (200/231)"},
    "kev:transfer-v4/development": {"acc": 85.67, "src": "Kev runs/jev-transfer-v4"},
    "kev:transfer-v9/development": {"acc": 85.37, "src": "Kev runs/jev-transfer-v9"},
    "vitaminc": {"acc": 80.1, "src": "Bespoke Nimble public suite"},
    "massive": {"acc": 87.4, "src": "Bespoke Nimble public suite"},
}

TIE = {"jevbench": "lex", "arxiv": "declared"}   # default 'first'


def request_golds(item) -> list[dict]:
    """Per-request gold maps (used for the oracle predictor and for scoring)."""
    n = len(item_requests(item))
    if item["rule"] == "nli_rerank":
        correct = set(item["meta"].get("correct_set", [item["golds"]["rerank"]]))
        return [{"relation": "entailment" if j in correct else "contradiction"} for j in range(n)]
    out = [dict() for _ in range(n)]
    for k, g in item["golds"].items():
        ri, qid = k.split(":", 1)
        out[int(ri)][qid] = g
    return out


def item_requests(item) -> list[dict]:
    if "requests" in item:
        return item["requests"]
    from eval.parity.common import nli_request
    return [nli_request(item["nli"]["premise"], h) for h in item["nli"]["hyps"]]


def _answers(resp):
    if resp is None:
        return None
    if isinstance(resp, dict) and "answers" in resp and isinstance(resp["answers"], dict):
        return resp["answers"]
    return resp


def score_item(item, responses: list) -> list[dict]:
    """Turn one item's responses into scored units."""
    suite = item["suite"]
    reqs = item_requests(item)
    base = {"uid": item["uid"], "suite": suite, "task": item["task"]}
    if item["rule"] == "nli_rerank":
        pent, missing, invalid = [], False, False
        for req, resp in zip(reqs, responses):
            ans = _answers(resp)
            if ans is None or "relation" not in ans:
                missing = True
                break
            try:
                p, _ = normalize_answer(req["questions"]["relation"], ans["relation"])
                pent.append(p[0])
            except InvalidAnswer:
                invalid = True
                pent.append(-1.0)
        if missing:
            return [{**base, "qid": "rerank", "missing": True}]
        pred = int(np.argmax(pent))  # ties -> first option (OpenJev rule)
        gold_set = set(item["meta"].get("correct_set", [item["golds"]["rerank"]]))
        correct = pred in gold_set
        # tie audit (Phase G): options tied with the max P(entailment); tie-free credit = share of tied options
        # that are gold, i.e. the expected accuracy under a random tie-break instead of first-option.
        mx = max(pent)
        tied = [i for i, v in enumerate(pent) if v >= mx - 1e-12]
        tie_free = sum(1 for i in tied if i in gold_set) / len(tied)
        return [{**base, "qid": "rerank", "qtype": "rerank", "valid": not invalid, "correct": bool(correct),
                 "pred": pred, "n_options": len(reqs), "pent": [float(v) for v in pent], "n_tied": len(tied),
                 "tie_free_credit": float(tie_free)}]

    units = []
    golds = request_golds(item)
    for ri, (req, resp) in enumerate(zip(reqs, responses)):
        ans = _answers(resp)
        for qid, g in golds[ri].items():
            q = req["questions"][qid]
            u = {**base, "qid": qid if len(reqs) == 1 else f"{ri}:{qid}", "qtype": q["type"]}
            if item["suite"] == "kev":
                u["scored"] = item["meta"]["scored"]
                u["variant"] = item["meta"]["variant"]
            if item["suite"] == "nimble":
                u["in_282"] = item["meta"]["in_282"]
            if ans is None or qid not in ans:
                u["missing"] = True
                units.append(u)
                continue
            keys = option_keys(q)
            if item["rule"] == "nli_cls":
                gi = list(NLI_LABELS).index(g) if q["criteria"] and list(q["criteria"]) == list(NLI_LABELS) else keys.index(g)
            else:
                gi = gold_index(q, g)
            try:
                p, decl = normalize_answer(q, ans[qid])
            except InvalidAnswer as e:
                u.update(valid=False, correct=False, error=str(e)[:200])
                units.append(u)
                continue
            tie = TIE.get(suite, "first")
            if suite == "arxiv" and q["type"] == "noul":
                tie = "noul_true"
            pi = argmax_index(p, keys, tie=tie, declared=decl)
            conf = float(p[pi])
            if item["rule"] == "nli_cls" and item["meta"].get("two_class"):
                # SciTail: entailment vs not-entailment
                ent = keys.index("entailment")
                p2 = np.array([1 - p[ent], p[ent]])
                pred_ent = pi == ent
                gold_ent = gi == ent
                u.update(valid=True, correct=bool(pred_ent == gold_ent), conf=float(p2.max()),
                         brier=float(((p2 - np.eye(2)[int(gold_ent)]) ** 2).sum()), pred=keys[pi])
            else:
                u.update(valid=True, correct=bool(pi == gi), conf=conf, brier=brier_multiclass(p, gi), pred=keys[pi],
                         p_gold=float(p[gi]))
                if suite == "arxiv":
                    # arXiv harness ECE: choice -> top prob vs correct; noul -> p_true vs label (15 bins)
                    u["arxiv_conf"] = float(p.max()) if q["type"] != "noul" else float(p[1])
                    u["arxiv_hit"] = float(pi == gi) if q["type"] != "noul" else float(gi == 1)
            units.append(u)
    return units


# ----------------------------------------------------------------------------- aggregation

def _acc(units):
    scored = [u for u in units if not u.get("missing")]
    if not scored:
        return float("nan"), 0, 0
    k = sum(1 for u in scored if u.get("correct"))
    n = len(scored)
    return (k / n if n else float("nan")), k, n


def _ece(units, bins=10):
    v = [u for u in units if u.get("valid") and "conf" in u]
    if not v:
        return float("nan"), 0
    return ece_top_label([u["conf"] for u in v], [u["correct"] for u in v], bins), len(v)


def _coverage(units):
    n = len(units)
    miss = sum(1 for u in units if u.get("missing"))
    inv = sum(1 for u in units if u.get("valid") is False)
    return {"units": n, "missing": miss, "invalid": inv}


def aggregate(suite: str, units: list[dict]) -> dict:
    by_task = collections.defaultdict(list)
    for u in units:
        by_task[u["task"]].append(u)
    res = {"coverage": _coverage(units)}
    if suite == "jevbench":
        fam = {t: _acc(us)[0] for t, us in sorted(by_task.items())}
        acc, k, n = _acc(units)
        fv = [v for v in fam.values() if not math.isnan(v)]
        res.update(score=100 * float(np.mean(fv)) if fv else float("nan"), metric="family-macro accuracy (18 families)",
                   per_example_accuracy=100 * acc, n=n, per_task={t: 100 * a for t, a in fam.items()})
    elif suite == "kev":
        scored = [u for u in units if u.get("scored")]
        per = {}
        for t, us in sorted(by_task.items()):
            a, k, n = _acc([u for u in us if u.get("scored")])
            per[t] = 100 * a
            e, ne = _ece([u for u in us if u.get("scored")])
            res.setdefault("per_task_ece", {})[t] = e
            res.setdefault("per_task_n", {})[t] = n
            bv = [u["brier"] for u in us if u.get("scored") and u.get("valid") and "brier" in u]
            res.setdefault("per_task_brier", {})[t] = float(np.mean(bv)) if bv else float("nan")
        pv = [v for v in per.values() if not math.isnan(v)]
        res.update(score=float(np.mean(pv)) if len(pv) == 6 else float("nan"), metric="mean of 6 clean accuracies",
                   per_task=per, n=len([u for u in scored if not u.get("missing")]))
        units = scored  # ECE on scored units only
    elif suite == "openjev":
        per = {}
        mnli = [u for u in by_task.get("mnli", [])]
        for t, us in sorted(by_task.items()):
            if t == "mnli":
                m = _acc([u for u in us if "validation_matched" in u["uid"]])[0]
                mm = _acc([u for u in us if "validation_mismatched" in u["uid"]])[0]
                per[t] = 100 * (m + mm) / 2
                res["mnli_matched"], res["mnli_mismatched"] = 100 * m, 100 * mm
            else:
                per[t] = 100 * _acc(us)[0]
        # tie-free variant (rerank tasks: random tie-break credit; other tasks unchanged)
        tf, tie_rate = {}, {}
        for t, us in sorted(by_task.items()):
            rr = [u for u in us if u.get("qtype") == "rerank" and not u.get("missing")]
            if rr:
                tf[t] = 100 * float(np.mean([u.get("tie_free_credit", float(u.get("correct", False))) for u in rr]))
                tie_rate[t] = float(np.mean([u.get("n_tied", 1) > 1 for u in rr]))
            else:
                tf[t] = per[t]
        res["per_task_tie_free"] = tf
        res["per_task_tie_rate"] = tie_rate
        res["score_tie_free"] = float(np.mean([tf[t] for t in tf if not math.isnan(tf[t])])) if tf else float("nan")
        have = [t for t in per if not math.isnan(per[t])]
        if have and "gsm8k_rerank4" not in have:
            m18 = float(np.mean([per[t] for t in have]))
            # Published 19-task mean includes GSM8K nli_rerank@4 on an unpublished candidate pool (greedy 93.0%,
            # maj 93.5%, oracle@4 97.0%); all four published models score 89.5-95.0 there. Bounds only, never used
            # as the measured score.
            res["score_19_bounds_if_gsm8k_rerank_in_[89.5,95.0]"] = [(len(have) * m18 + 89.5) / (len(have) + 1),
                                                                     (len(have) * m18 + 95.0) / (len(have) + 1)]
        res.update(score=float(np.mean([per[t] for t in have])) if have else float("nan"),
                   metric=f"mean of {len(have)} task accuracies (published group uses 19; MNLI m/mm averaged first)",
                   n_tasks=len(have), per_task=per, n=sum(_acc(us)[2] for us in by_task.values()))
        units = [u for u in units if u.get("qtype") != "rerank"]  # ECE over NLI classification units
    elif suite == "nimble":
        a282, k, n = _acc([u for u in units if u.get("in_282")])
        a324 = _acc(units)[0]
        res.update(score=100 * a282, metric="per-example exact accuracy, NeoHorse 282 subset (reconstructed)",
                   acc_324=100 * a324, n=n, per_task={t: 100 * _acc([u for u in us if u.get("in_282")])[0] for t, us in sorted(by_task.items())},
                   per_type={t: 100 * _acc([u for u in units if u.get("in_282") and u.get("qtype") == t])[0] for t in ("choice", "noul", "score")})
        units = [u for u in units if u.get("in_282")]
    elif suite in ("vitaminc", "massive"):
        a, k, n = _acc(units)
        res.update(score=100 * a, metric="accuracy", n=n, ci95=[100 * x for x in wilson(k, n)])
    elif suite == "arxiv":
        per, ece15 = {}, {}
        for t, us in sorted(by_task.items()):
            per[t] = 100 * _acc(us)[0]
            v = [u for u in us if u.get("valid") and "arxiv_conf" in u]
            ece15[t] = ece_top_label([u["arxiv_conf"] for u in v], [u["arxiv_hit"] for u in v], 15) if v else float("nan")
        pv = [v for v in per.values() if not math.isnan(v)]
        res.update(score=float(np.mean(pv)) if pv else float("nan"), metric="mean task accuracy (replay validation)",
                   per_task=per, per_task_ece15=ece15, n=sum(_acc(us)[2] for us in by_task.values()))
    e, ne = _ece(units)
    res["ece10"] = e
    res["ece_n"] = ne
    v = [u for u in units if u.get("valid") and "brier" in u]
    res["brier"] = float(np.mean([u["brier"] for u in v])) if v else float("nan")
    return res


def parity_summary(agg: dict) -> dict:
    six = ["jevbench", "kev", "openjev", "nimble", "vitaminc", "massive"]
    have = {s: agg[s]["score"] for s in six if s in agg and not math.isnan(agg[s]["score"])}
    out = {"per_group": {s: {"verdict": have.get(s), "jev_1.13": JEV_SIX[s],
                             "delta": (have[s] - JEV_SIX[s]) if s in have else None} for s in six}}
    if len(have) == 6:
        m = float(np.mean(list(have.values())))
        out.update(six_group_mean=m, jev_six_group_mean=JEV_SIX_MEAN, gap=m - JEV_SIX_MEAN,
                   accuracy_parity=m >= JEV_SIX_MEAN,
                   note_openjev="OpenJev scored on the 18 obtainable tasks (GSM8K rerank pool unpublished): conservative")
        b = agg.get("openjev", {}).get("score_19_bounds_if_gsm8k_rerank_in_[89.5,95.0]")
        if b:
            out["six_group_mean_if_openjev19_bounds"] = [m + (b[0] - have["openjev"]) / 6, m + (b[1] - have["openjev"]) / 6]
    else:
        out.update(six_group_mean=None, note=f"only {len(have)}/6 groups scored")
    ece_checks = {}
    for key, (ref, src) in JEV_ECE.items():
        if key.startswith("kev:"):
            t = key.split(":", 1)[1]
            val = agg.get("kev", {}).get("per_task_ece", {}).get(t)
        else:
            val = agg.get(key, {}).get("ece10")
        if val is not None and not (isinstance(val, float) and math.isnan(val)):
            ece_checks[key] = {"verdict": val, "jev": ref, "ok": val <= ref, "src": src}
    out["ece_checks"] = ece_checks
    out["ece_parity"] = bool(ece_checks) and all(c["ok"] for c in ece_checks.values())
    brier_checks = {}
    for key, (ref, src) in JEV_BRIER.items():
        if key.startswith("kev:"):
            val = agg.get("kev", {}).get("per_task_brier", {}).get(key.split(":", 1)[1])
        else:
            val = agg.get(key, {}).get("brier")
        if val is not None and not (isinstance(val, float) and math.isnan(val)):
            brier_checks[key] = {"verdict": val, "jev": ref, "ok": val <= ref, "src": src}
    out["brier_checks"] = brier_checks
    out["brier_parity"] = bool(brier_checks) and all(c["ok"] for c in brier_checks.values())
    return out


def bootstrap_six(units_by_suite: dict, B: int = 500, seed: int = 0) -> dict:
    """Stratified (within task) bootstrap of each group score and of the six-group mean. 95% percentile CIs."""
    rng = np.random.default_rng(seed)
    six = ["jevbench", "kev", "openjev", "nimble", "vitaminc", "massive"]
    strata = {}
    for s in six:
        us = units_by_suite.get(s)
        if not us:
            return {}
        if s == "kev":
            us = [u for u in us if u.get("scored")]
        if s == "nimble":
            us = [u for u in us if u.get("in_282")]
        us = [u for u in us if not u.get("missing")]
        by = collections.defaultdict(list)
        for u in us:
            key = u["task"]
            if s == "openjev" and u["task"] == "mnli":
                key = "mnli_m" if "validation_matched" in u["uid"] else "mnli_mm"
            by[key].append(1.0 if u.get("correct") else 0.0)
        strata[s] = {k: np.array(v) for k, v in by.items()}
    draws = {s: [] for s in six}
    for _ in range(B):
        for s in six:
            accs = {k: v[rng.integers(0, len(v), len(v))].mean() for k, v in strata[s].items()}
            if s in ("jevbench", "kev"):
                val = 100 * np.mean(list(accs.values()))
            elif s == "openjev":
                m = (accs.pop("mnli_m") + accs.pop("mnli_mm")) / 2
                val = 100 * np.mean(list(accs.values()) + [m])
            else:
                allv = np.concatenate(list(strata[s].values()))
                val = 100 * allv[rng.integers(0, len(allv), len(allv))].mean()
            draws[s].append(val)
    out = {s: [float(np.percentile(draws[s], 2.5)), float(np.percentile(draws[s], 97.5))] for s in six}
    means = np.mean(np.array([draws[s] for s in six]), axis=0)
    out["six_group_mean"] = [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]
    out["p_mean_ge_jev"] = float((means >= JEV_SIX_MEAN).mean())
    out["B"] = B
    return out
