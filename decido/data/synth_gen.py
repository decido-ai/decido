"""Teacher-generated synthetic decision tasks (Phase B.3d): workflows -> states (labels come from decido.label).

    python -m decido.data.synth_gen workflows --model Qwen/Qwen3.8-27B --out data/synth/workflows.jsonl
    python -m decido.data.synth_gen states    --model Qwen/Qwen3.8-27B --workflows data/synth/workflows.jsonl \
        --out data/synth/states.jsonl --per-workflow 30
    python -m decido.data.synth_gen rows --workflows ... --states ... --out data/synth/rows.jsonl

Prompts follow research/data.md §3 (12 generic families x ~25 domains). Per the leakage rules, no eval
label inventories, instructions or criteria wording appear in the generator prompts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re

FAMILIES = [
    ("agent action gating", "whether an AI agent should execute a proposed tool call or action, its risk level, which policy governs it"),
    ("payment and spend approval", "whether to pay or approve an invoice/expense/refund, anomaly type, risk"),
    ("routing with a runtime taxonomy", "which team/queue/category should handle an item, choosing among 6-30 described categories including a 'none of these' option"),
    ("rubric scoring", "scoring a written answer, essay, code review or support reply against an explicit rubric on an ordered scale"),
    ("LLM output quality", "whether a model response is grounded in a given context, hallucination severity, instruction following"),
    ("structured-record judgment", "judging a JSON record (order, CRM account, log event, cloud alert, shipment) e.g. likely churn, fraud, needs escalation"),
    ("moderation and trust-and-safety", "whether content violates a stated policy, which policy category, severity"),
    ("triage and urgency", "urgency level, whether a human is needed, which next step"),
    ("stance, intent and tone", "user intent, sentiment toward a target, whether the writer is frustrated or sarcastic"),
    ("entity matching and deduplication", "whether two records/profiles/listings refer to the same entity, which fields conflict"),
    ("retrieval relevance", "how relevant a document/passage/product is to a query on an ordered scale, whether it answers the query"),
    ("conversation and agent traces", "whether the user's goal was achieved, whether the agent made an error, best next action"),
]
DOMAINS = ["healthcare clinic operations", "retail banking", "e-commerce marketplace", "insurance claims", "logistics and shipping",
           "software engineering", "cloud infrastructure operations", "human resources and recruiting", "legal contract review",
           "real estate and property management", "education and tutoring", "travel and hospitality", "telecommunications support",
           "manufacturing quality control", "energy utilities", "public sector services", "media and publishing",
           "gaming community management", "pharmaceutical safety reporting", "restaurant and food delivery", "automotive service",
           "nonprofit grant management", "cybersecurity operations", "personal finance apps", "scientific research labs"]

SYS = "You are a meticulous dataset designer. Output only valid JSON, no prose, no markdown fences."

WF_PROMPT = """Design {n} distinct decision workflows for the family "{family}" ({desc}) in the domain "{domain}".
A workflow is a reusable decision application: an item type plus 3-6 typed questions that a fast decision model answers about each item.
Question types:
- "noul": yes/no. "criteria" is an object {{"true": "...", "false": "..."}} describing what yes and no mean (may be omitted).
- "choice": pick one unordered option. "criteria" maps 2-12 short snake_case option names to one-sentence descriptions (occasionally 15-30 options for routing). Options must be mutually exclusive and cover the space (add an "other"/"none" option when natural).
- "score": ordered levels. "criteria" is a list of 3-6 level descriptions from lowest to highest.
Rules: questions must be answerable from the item text alone (state any policy, rubric or threshold inside the instructions or criteria); mix all three types; vary wording; no trivial lookups; include at least one question that often needs judgment under uncertainty.
Also give "item_type" (what one item is) and "state_format": one of "prose", "json", "list" (array of strings, e.g. chat turns or two records).
Return JSON: {{"workflows": [{{"name": str, "item_type": str, "state_format": str, "questions": {{"<qid>": {{"type": ..., "instructions": str, "criteria": ...}}}}}}]}}"""

ST_PROMPT = """Workflow (JSON):
{wf}

Write {n} realistic, varied items for this workflow. Each item is the "state" that the questions will be asked about.
Format: {fmt_rule}
Length: vary between short (1-3 sentences or a few fields) and long (several paragraphs / many fields); items {long_ids} should be long (300-900 words).
Difficulty plan (do NOT reveal it in the items): items {clear_ids} should have clear answers; items {border_ids} should be borderline between two answers for at least one question; items {amb_ids} should be genuinely ambiguous or missing key information for at least one question.
Never state the answers or mention the questions. Use realistic names, numbers, dates and details; include irrelevant details sometimes.
Return JSON: {{"items": [{{"id": <int>, "state": <the item>}}]}}"""

FMT_RULES = {"prose": "each state is a plain-text string (an email, ticket, report, message, document excerpt...)",
             "json": "each state is a JSON object with descriptive keys (nested objects/arrays allowed)",
             "list": "each state is a JSON array of strings (e.g. conversation turns, or two records to compare)"}


def _chat(tok, user):
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": user}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def _parse(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except Exception:
        i, j = text.find("{"), text.rfind("}")
        if i >= 0 and j > i:
            try:
                return json.loads(text[i:j + 1])
            except Exception:
                return None
    return None


def valid_question(q):
    t = q.get("type")
    if not isinstance(q.get("instructions"), str) or len(q["instructions"]) < 8:
        return False
    if t == "noul":
        c = q.get("criteria")
        if c is None:
            return True
        return isinstance(c, dict) and set(c) <= {"true", "false"} and all(isinstance(v, str) for v in c.values())
    if t == "choice":
        c = q.get("criteria")
        return isinstance(c, dict) and 2 <= len(c) <= 40 and all(isinstance(k, str) and k for k in c)
    if t == "score":
        c = q.get("criteria")
        return isinstance(c, list) and 2 <= len(c) <= 10 and all(isinstance(v, (str, dict)) for v in c)
    return False


def llm(model, tp):
    from transformers import AutoTokenizer
    from vllm import LLM
    hc = os.environ.get("HF_HUB_CACHE")
    tok = AutoTokenizer.from_pretrained(model, cache_dir=hc)
    eng = LLM(model=model, tensor_parallel_size=tp, max_model_len=16384, gpu_memory_utilization=0.9,
              enable_prefix_caching=True, download_dir=hc, seed=0, max_num_seqs=256, limit_mm_per_prompt={"image": 0, "video": 0})
    return tok, eng


def cmd_workflows(a):
    from vllm import SamplingParams
    tok, eng = llm(a.model, a.tp)
    rng = random.Random(0)
    jobs = [(f, d, k) for f in FAMILIES for d in DOMAINS for k in range(a.reps)]
    prompts = [_chat(tok, WF_PROMPT.format(n=a.n, family=f[0], desc=f[1], domain=d)) for f, d, _ in jobs]
    sps = [SamplingParams(temperature=0.9, top_p=0.95, max_tokens=6000, seed=rng.randrange(1 << 30)) for _ in jobs]
    outs = eng.generate(prompts, sps)
    n_ok = 0
    with open(a.out, "w") as fh:
        for (f, d, k), o in zip(jobs, outs):
            j = _parse(o.outputs[0].text)
            if not j or not isinstance(j.get("workflows"), list):
                continue
            for wi, wf in enumerate(j["workflows"]):
                qs = {qid: q for qid, q in (wf.get("questions") or {}).items() if isinstance(q, dict) and valid_question(q)}
                if len(qs) < 2 or wf.get("state_format") not in FMT_RULES:
                    continue
                wid = hashlib.sha1(f"{f[0]}|{d}|{k}|{wi}|{wf.get('name')}".encode()).hexdigest()[:12]
                fh.write(json.dumps({"workflow_id": wid, "family": f[0], "domain": d, "name": wf.get("name"),
                                     "item_type": wf.get("item_type"), "state_format": wf["state_format"], "questions": qs}) + "\n")
                n_ok += 1
    print(f"workflows kept: {n_ok}")


def cmd_states(a):
    from vllm import SamplingParams
    tok, eng = llm(a.model, a.tp)
    wfs = [json.loads(x) for x in open(a.workflows)]
    rng = random.Random(1)
    jobs = []
    for wf in wfs:
        for b in range(max(1, a.per_workflow // a.batch)):
            ids = list(range(1, a.batch + 1))
            rng.shuffle(ids)
            nc, nb = int(a.batch * 0.5), int(a.batch * 0.3)
            clear, border, amb = sorted(ids[:nc]), sorted(ids[nc:nc + nb]), sorted(ids[nc + nb:])
            long_ids = sorted(rng.sample(range(1, a.batch + 1), k=max(1, a.batch // 5)))
            wf_view = {"item_type": wf["item_type"], "questions": wf["questions"]}
            p = ST_PROMPT.format(wf=json.dumps(wf_view, ensure_ascii=False), n=a.batch, fmt_rule=FMT_RULES[wf["state_format"]],
                                 long_ids=long_ids, clear_ids=clear, border_ids=border, amb_ids=amb)
            jobs.append((wf["workflow_id"], b, p))
    sps = [SamplingParams(temperature=0.9, top_p=0.95, max_tokens=12000, seed=rng.randrange(1 << 30)) for _ in jobs]
    outs = eng.generate([_chat(tok, p) for _, _, p in jobs], sps)
    n = 0
    with open(a.out, "w") as fh:
        for (wid, b, _), o in zip(jobs, outs):
            j = _parse(o.outputs[0].text)
            if not j or not isinstance(j.get("items"), list):
                continue
            for it in j["items"]:
                st = it.get("state") if isinstance(it, dict) else None
                if st is None or (isinstance(st, str) and len(st) < 20):
                    continue
                fh.write(json.dumps({"workflow_id": wid, "batch": b, "item": it.get("id"), "state": st}, ensure_ascii=False) + "\n")
                n += 1
    print(f"states kept: {n}")


def cmd_rows(a):
    wfs = {json.loads(x)["workflow_id"]: json.loads(x) for x in open(a.workflows)}
    n = 0
    with open(a.out, "w") as fh:
        for x in open(a.states):
            s = json.loads(x)
            wf = wfs.get(s["workflow_id"])
            if not wf:
                continue
            rid = hashlib.sha1(json.dumps(s, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:20]
            fh.write(json.dumps({"request_id": "synth-" + rid, "state": s["state"], "questions": wf["questions"],
                                 "meta": {"source": "verdict-teacher-synth", "task_id": "synth/" + wf["workflow_id"],
                                          "family": wf["family"], "domain": wf["domain"], "workflow_id": wf["workflow_id"],
                                          "label_type": "soft_llm", "license": "own (teacher Apache-2.0 outputs)",
                                          "license_use": "commercial", "generator": a.generator, "group_id": wf["workflow_id"]}},
                                ensure_ascii=False) + "\n")
            n += 1
    print(f"rows: {n}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("workflows")
    w.add_argument("--model", required=True)
    w.add_argument("--out", required=True)
    w.add_argument("--n", type=int, default=3)
    w.add_argument("--reps", type=int, default=1)
    w.add_argument("--tp", type=int, default=1)
    s = sub.add_parser("states")
    s.add_argument("--model", required=True)
    s.add_argument("--workflows", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--per-workflow", type=int, default=30)
    s.add_argument("--batch", type=int, default=10)
    s.add_argument("--tp", type=int, default=1)
    r = sub.add_parser("rows")
    r.add_argument("--workflows", required=True)
    r.add_argument("--states", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--generator", default="teacher")
    a = ap.parse_args()
    {"workflows": cmd_workflows, "states": cmd_states, "rows": cmd_rows}[a.cmd](a)


if __name__ == "__main__":
    main()
