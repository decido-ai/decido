"""Random decision-table data (exact labels): generic rule application over random records.

    python -m decido.data.decision_tables --out data/dtables_raw.jsonl --n 16000

Design (written without consulting any eval item): a random domain supplies attribute specs; a record is sampled;
a random rule list (each rule = 1-3 conditions: numeric threshold, category membership, boolean flag, negations;
missing attributes count as "not satisfied" when the policy says so) maps records to outcomes under one of three
random semantics: first-match-wins, highest-applicable-severity, or all-checks-must-pass. Questions per record:
choice (which outcome), noul (does rule k apply / is outcome X), score (severity level, when outcomes are ordered).
The policy is placed either inside the state (JSON) or inside the question instructions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random

DOMAINS = {
    "warehouse_order": [("units", "int", (1, 500)), ("unit_price_eur", "float", (0.5, 900)), ("priority", "cat", ["standard", "express", "overnight"]),
                        ("hazardous", "bool"), ("customer_tier", "cat", ["bronze", "silver", "gold", "platinum"]), ("backorder_days", "int", (0, 40)),
                        ("destination_zone", "cat", ["domestic", "regional", "international"]), ("signature_waived", "bool")],
    "it_access_request": [("requested_role", "cat", ["viewer", "editor", "admin", "owner"]), ("employment_months", "int", (0, 180)),
                          ("mfa_enrolled", "bool"), ("manager_approved", "bool"), ("contractor", "bool"), ("open_incidents", "int", (0, 6)),
                          ("data_classification", "cat", ["public", "internal", "confidential", "restricted"])],
    "grant_application": [("budget_usd", "int", (2000, 900000)), ("team_size", "int", (1, 40)), ("prior_awards", "int", (0, 8)),
                          ("ethics_review_done", "bool"), ("field", "cat", ["biology", "physics", "economics", "education", "engineering"]),
                          ("duration_months", "int", (3, 60)), ("matching_funds", "bool")],
    "vehicle_inspection": [("mileage_km", "int", (0, 400000)), ("age_years", "int", (0, 30)), ("brake_wear_pct", "int", (0, 100)),
                           ("tire_tread_mm", "float", (0.5, 9)), ("emissions_pass", "bool"), ("fuel", "cat", ["petrol", "diesel", "hybrid", "electric"]),
                           ("recall_open", "bool")],
    "clinical_trial_screening": [("age", "int", (16, 90)), ("bmi", "float", (15, 48)), ("systolic_bp", "int", (90, 200)), ("pregnant", "bool"),
                                 ("prior_chemo", "bool"), ("egfr", "int", (10, 120)), ("smoker", "cat", ["never", "former", "current"])],
    "content_listing": [("word_count", "int", (5, 3000)), ("has_price", "bool"), ("images", "int", (0, 15)), ("external_links", "int", (0, 12)),
                        ("category", "cat", ["electronics", "apparel", "home", "toys", "food", "services"]), ("seller_rating", "float", (1, 5)),
                        ("reported_count", "int", (0, 9))],
    "travel_expense": [("amount_usd", "float", (5, 6000)), ("receipt", "bool"), ("expense_type", "cat", ["meal", "lodging", "airfare", "taxi", "conference"]),
                       ("days_after_trip", "int", (0, 120)), ("preapproved", "bool"), ("attendees", "int", (1, 12)), ("international", "bool")],
    "network_alert": [("packets_per_s", "int", (10, 900000)), ("distinct_ports", "int", (1, 4000)), ("source_internal", "bool"),
                      ("geo", "cat", ["home_country", "allied", "other"]), ("failed_logins", "int", (0, 300)), ("off_hours", "bool"),
                      ("asset_criticality", "cat", ["low", "medium", "high"])],
    "tenancy_application": [("monthly_rent", "int", (400, 7000)), ("monthly_income", "int", (800, 25000)), ("credit_score", "int", (450, 850)),
                            ("pets", "int", (0, 3)), ("references", "int", (0, 4)), ("prior_eviction", "bool"), ("lease_months", "cat", ["6", "12", "24"])],
    "manufacturing_lot": [("defect_rate_ppm", "int", (0, 20000)), ("lot_size", "int", (50, 100000)), ("supplier_audit", "cat", ["passed", "conditional", "failed", "pending"]),
                          ("temperature_excursion", "bool"), ("days_to_ship", "int", (0, 30)), ("customer_critical", "bool"), ("rework_count", "int", (0, 5))],
    "loan_servicing": [("days_past_due", "int", (0, 200)), ("balance", "int", (500, 400000)), ("hardship_flag", "bool"), ("prior_modifications", "int", (0, 4)),
                       ("collateral", "cat", ["none", "vehicle", "property", "equipment"]), ("autopay", "bool"), ("contact_attempts", "int", (0, 15))],
    "course_enrollment": [("gpa", "float", (0, 4)), ("credits_completed", "int", (0, 160)), ("prerequisite_met", "bool"), ("waitlist_position", "int", (0, 60)),
                          ("program", "cat", ["undergraduate", "graduate", "exchange", "audit"]), ("tuition_paid", "bool"), ("holds", "int", (0, 3))],
}
OUTCOME_SETS = [
    ["approve", "approve_with_conditions", "manual_review", "reject"],
    ["release", "hold", "quarantine", "destroy"],
    ["no_action", "notify_owner", "open_ticket", "page_oncall"],
    ["standard_queue", "priority_queue", "specialist_team", "legal_team", "decline"],
    ["eligible", "conditionally_eligible", "ineligible"],
    ["publish", "publish_after_edit", "send_to_moderator", "remove"],
    ["accept", "request_more_info", "escalate", "deny"],
]
SEVERITY = ["none", "low", "moderate", "high", "critical"]


def sample_record(spec, r):
    rec = {}
    for a in spec:
        name, typ = a[0], a[1]
        if r.random() < 0.06:            # occasionally missing
            continue
        if typ == "int":
            rec[name] = r.randint(*a[2])
        elif typ == "float":
            rec[name] = round(r.uniform(*a[2]), 2)
        elif typ == "cat":
            rec[name] = r.choice(a[2])
        else:
            rec[name] = r.random() < 0.5
    return rec


def sample_cond(spec, r):
    a = r.choice(spec)
    name, typ = a[0], a[1]
    neg = r.random() < 0.2
    if typ in ("int", "float"):
        lo, hi = a[2]
        v = round(r.uniform(lo, hi), 0 if typ == "int" else 1)
        v = int(v) if typ == "int" else v
        op = r.choice([">", ">=", "<", "<="])
        text = f"{name} {op} {v}"
        fn = lambda rec, n=name, o=op, v=v: (n in rec) and {">": rec[n] > v, ">=": rec[n] >= v, "<": rec[n] < v, "<=": rec[n] <= v}[o]
    elif typ == "cat":
        k = r.randint(1, max(1, len(a[2]) - 1))
        vals = r.sample(a[2], k)
        text = f"{name} is {vals[0]}" if k == 1 else f"{name} is one of {', '.join(vals)}"
        fn = lambda rec, n=name, vs=tuple(vals): (n in rec) and rec[n] in vs
    else:
        text = f"{name} is true"
        fn = lambda rec, n=name: (n in rec) and bool(rec[n])
    if neg:
        return f"NOT ({text})", (lambda rec, f=fn, n=name: (n in rec) and not f(rec))
    return text, fn


def sample_rules(spec, outcomes, r, ordered):
    k = r.randint(2, min(5, len(outcomes) + 1))
    rules = []
    for i in range(k):
        conds = [sample_cond(spec, r) for _ in range(r.randint(1, 3))]
        out = r.choice(outcomes[1:]) if ordered else r.choice(outcomes[:-1] if len(outcomes) > 2 else outcomes)
        rules.append({"conds": conds, "outcome": out})
    return rules


def rule_text(rule, i):
    return f"Rule {i + 1}: if " + " AND ".join(c[0] for c in rule["conds"]) + f" then {rule['outcome']}"


def holds(rule, rec):
    return all(fn(rec) for _, fn in rule["conds"])


def make(r):
    dom = r.choice(list(DOMAINS))
    spec = r.sample(DOMAINS[dom], k=min(len(DOMAINS[dom]), r.randint(5, 8)))
    sem = r.choice(["first_match", "severity", "all_pass"])
    if sem == "severity":
        n = r.randint(3, 5)
        outcomes = SEVERITY[:n]
    else:
        outcomes = r.choice(OUTCOME_SETS)
    rec = sample_record(spec, r)
    if sem == "all_pass":
        checks = [[sample_cond(spec, r) for _ in range(r.randint(1, 2))] for _ in range(r.randint(2, 4))]
        ok = all(all(fn(rec) for _, fn in c) for c in checks)
        pos, negout = outcomes[0], outcomes[-1]
        policy = [f"Check {i + 1}: " + " AND ".join(t for t, _ in c) for i, c in enumerate(checks)]
        policy.append(f"If every check passes the outcome is {pos}; otherwise {negout}. A missing field fails its check.")
        label = pos if ok else negout
        cands = [pos, negout] + [o for o in outcomes if o not in (pos, negout)][:r.randint(0, 2)]
        atoms = [("check", i, all(fn(rec) for _, fn in c)) for i, c in enumerate(checks)]
    else:
        rules = sample_rules(spec, outcomes, r, ordered=(sem == "severity"))
        policy = [rule_text(ru, i) for i, ru in enumerate(rules)]
        if sem == "first_match":
            policy.append(f"Apply rules in order; the first rule whose conditions all hold decides. If none holds: {outcomes[0]}. "
                          "A condition on a missing field is not satisfied.")
            label = next((ru["outcome"] for ru in rules if holds(ru, rec)), outcomes[0])
        else:
            policy.append(f"The level is the most severe outcome among all rules whose conditions hold; if none holds: {outcomes[0]}. "
                          "A condition on a missing field is not satisfied.")
            hit = [SEVERITY.index(ru["outcome"]) for ru in rules if holds(ru, rec)]
            label = SEVERITY[max(hit)] if hit else outcomes[0]
        cands = list(dict.fromkeys(outcomes))
        atoms = [("rule", i, holds(ru, rec)) for i, ru in enumerate(rules)]
    return dom, sem, spec, rec, policy, label, cands, atoms


def build_row(r):
    dom, sem, spec, rec, policy, label, cands, atoms = make(r)
    in_state = r.random() < 0.5
    pol_text = "\n".join(policy)
    state = {"policy": policy, dom: rec} if in_state else ({dom: rec} if r.random() < 0.5 else
                                                             "; ".join(f"{k.replace('_', ' ')}: {v}" for k, v in rec.items()))
    lead = "Using the policy in the record" if in_state else f"Policy:\n{pol_text}\nUsing this policy"
    qs, tg = {}, {}
    if sem == "severity":
        levels = cands
        qs["level"] = {"type": "score", "instructions": f"{lead}, what level applies to this {dom.replace('_', ' ')}?",
                       "criteria": [f"{lv}" for lv in levels]}
        t = [0.0] * len(levels)
        t[levels.index(label)] = 1.0
        tg["level"] = t
    else:
        opts = cands[:]
        r.shuffle(opts)
        qs["outcome"] = {"type": "choice", "instructions": f"{lead}, which outcome applies to this {dom.replace('_', ' ')}?",
                         "criteria": {o: None for o in opts}}
        tg["outcome"] = [1.0 if o == label else 0.0 for o in opts]
    if r.random() < 0.6 and atoms:
        kind, i, v = r.choice(atoms)
        qs["atom"] = {"type": "noul", "instructions": f"{lead}: do all conditions of {'Check' if kind == 'check' else 'Rule'} {i + 1} hold for this record?"}
        tg["atom"] = [0.0, 1.0] if v else [1.0, 0.0]
    if r.random() < 0.3:
        o = label if r.random() < 0.5 else r.choice(cands)
        qs["is_outcome"] = {"type": "noul", "instructions": f"{lead}, is the resulting outcome {o}?"}
        tg["is_outcome"] = [0.0, 1.0] if o == label else [1.0, 0.0]
    rid = hashlib.sha1(json.dumps([state, qs], sort_keys=True, default=str).encode()).hexdigest()[:20]
    return {"request_id": "dt-" + rid, "state": state, "questions": qs, "targets": tg,
            "meta": {"source": "verdict-dtables", "task_id": f"dtables/{sem}", "family": "rule_application", "label_type": "exact",
                     "license": "own", "license_use": "commercial", "split": "train", "generator": "procedural-random", "group_id": rid}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=16000)
    ap.add_argument("--seed", type=int, default=20261019)
    a = ap.parse_args()
    r = random.Random(a.seed)
    cnt = {}
    n = 0
    with open(a.out, "w") as fh:
        while n < a.n:
            row = build_row(r)
            key = (row["meta"]["task_id"], json.dumps(next(iter(row["targets"].values()))))
            if cnt.get(key, 0) > a.n / 12:   # keep label balance per semantics
                continue
            cnt[key] = cnt.get(key, 0) + 1
            fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            n += 1
    print("rows", n)


if __name__ == "__main__":
    main()
