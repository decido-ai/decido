"""Compound-question data (exact labels): train rows + a held-out eval with the same items asked split and compound.

    python -m decido.data.compound --train-out data/compound_train.jsonl --eval-out eval/compound/compound_eval.jsonl

A record (prose or JSON) has typed attributes; atomic predicates are checkable facts about it. A compound question
combines 2-4 predicates with all-of / any-of / A-but-not-B / exactly-one. Train and eval use DISJOINT schemas, so the
eval measures transfer of compositional reading, not memorised predicates. The eval file holds, per item, the compound
question and its atomic sub-questions (asked separately) for a compound-vs-split comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random

FIRST = ["Ava", "Liam", "Noah", "Mia", "Zoe", "Omar", "Priya", "Kenji", "Lucia", "Tomas", "Fatima", "Ivan", "Grace", "Chen"]
LAST = ["Okafor", "Silva", "Nguyen", "Schmidt", "Haddad", "Kowalski", "Tanaka", "Murphy", "Rossi", "Patel", "Larsen"]


def _name(r):
    return f"{r.choice(FIRST)} {r.choice(LAST)}"


# Each schema: make(r) -> record dict; preds: list of (text, fn(record)->bool)
def schema_loan(r):
    rec = {"applicant": _name(r), "age": r.randint(18, 75), "annual_income_usd": r.randrange(18000, 240000, 500),
           "credit_score": r.randint(480, 840), "existing_debt_usd": r.randrange(0, 90000, 250),
           "employment_years": r.randint(0, 30), "requested_amount_usd": r.randrange(2000, 60000, 500),
           "has_cosigner": r.random() < 0.3, "prior_default": r.random() < 0.15}
    preds = [("the applicant is at least 25 years old", lambda x: x["age"] >= 25),
             ("the credit score is 700 or higher", lambda x: x["credit_score"] >= 700),
             ("the requested amount is under 20% of annual income", lambda x: x["requested_amount_usd"] < 0.2 * x["annual_income_usd"]),
             ("the applicant has a co-signer", lambda x: x["has_cosigner"]),
             ("the applicant has a prior default", lambda x: x["prior_default"]),
             ("existing debt exceeds half of annual income", lambda x: x["existing_debt_usd"] > 0.5 * x["annual_income_usd"]),
             ("the applicant has been employed for at least 2 years", lambda x: x["employment_years"] >= 2)]
    return rec, preds


def schema_shipment(r):
    rec = {"shipment_id": f"SH-{r.randint(10000, 99999)}", "weight_kg": round(r.uniform(0.2, 120), 1),
           "declared_value_eur": r.randrange(5, 9000, 5), "destination_country": r.choice(["DE", "US", "BR", "JP", "NG", "IN", "CA", "FR"]),
           "contains_lithium_battery": r.random() < 0.25, "insured": r.random() < 0.4,
           "days_in_transit": r.randint(0, 25), "signature_required": r.random() < 0.5, "fragile": r.random() < 0.3}
    preds = [("the parcel weighs more than 30 kg", lambda x: x["weight_kg"] > 30),
             ("the declared value is above 1000 EUR", lambda x: x["declared_value_eur"] > 1000),
             ("the destination is outside the EU", lambda x: x["destination_country"] not in ("DE", "FR")),
             ("it contains a lithium battery", lambda x: x["contains_lithium_battery"]),
             ("the parcel is insured", lambda x: x["insured"]),
             ("it has been in transit for more than 10 days", lambda x: x["days_in_transit"] > 10),
             ("a signature is required on delivery", lambda x: x["signature_required"]),
             ("it is marked fragile", lambda x: x["fragile"])]
    return rec, preds


def schema_server(r):
    rec = {"host": f"node-{r.randint(1, 400)}", "cpu_pct": r.randint(1, 100), "memory_pct": r.randint(5, 100),
           "disk_free_gb": r.randint(1, 900), "uptime_days": r.randint(0, 400), "pending_security_patches": r.randint(0, 12),
           "region": r.choice(["eu-west", "us-east", "ap-south"]), "in_maintenance_window": r.random() < 0.2,
           "error_rate_per_min": round(r.uniform(0, 30), 1)}
    preds = [("CPU utilisation is above 85%", lambda x: x["cpu_pct"] > 85),
             ("memory utilisation is above 90%", lambda x: x["memory_pct"] > 90),
             ("less than 20 GB of disk is free", lambda x: x["disk_free_gb"] < 20),
             ("there are pending security patches", lambda x: x["pending_security_patches"] > 0),
             ("the host is in a maintenance window", lambda x: x["in_maintenance_window"]),
             ("the error rate exceeds 5 per minute", lambda x: x["error_rate_per_min"] > 5),
             ("the host has been up for more than 180 days", lambda x: x["uptime_days"] > 180)]
    return rec, preds


def schema_recipe(r):
    rec = {"dish": r.choice(["curry", "risotto", "tacos", "salad", "stew", "pasta bake", "stir fry"]),
           "prep_minutes": r.randint(5, 150), "calories_per_serving": r.randint(120, 1400), "servings": r.randint(1, 10),
           "contains_nuts": r.random() < 0.25, "contains_gluten": r.random() < 0.5, "vegetarian": r.random() < 0.45,
           "spice_level_1_to_5": r.randint(1, 5), "cost_per_serving_usd": round(r.uniform(0.8, 14), 2)}
    preds = [("it takes under 30 minutes to prepare", lambda x: x["prep_minutes"] < 30),
             ("it has more than 700 calories per serving", lambda x: x["calories_per_serving"] > 700),
             ("it contains nuts", lambda x: x["contains_nuts"]),
             ("it is gluten-free", lambda x: not x["contains_gluten"]),
             ("it is vegetarian", lambda x: x["vegetarian"]),
             ("its spice level is 4 or higher", lambda x: x["spice_level_1_to_5"] >= 4),
             ("it serves at least 4 people", lambda x: x["servings"] >= 4)]
    return rec, preds


# held-out eval schemas
def schema_email(r):
    rec = {"from_domain": r.choice(["company.com", "c0mpany.com", "partner.org", "mail-service.net", "company.co"]),
           "sender_in_address_book": r.random() < 0.5, "links": r.randint(0, 6), "link_domains_match_sender": r.random() < 0.5,
           "has_attachment": r.random() < 0.4, "attachment_type": r.choice(["pdf", "docx", "zip", "exe", "xlsm"]),
           "urgent_language": r.random() < 0.4, "requests_credentials": r.random() < 0.25, "sent_hour_local": r.randint(0, 23)}
    preds = [("the sender domain is exactly company.com", lambda x: x["from_domain"] == "company.com"),
             ("the sender is in the address book", lambda x: x["sender_in_address_book"]),
             ("the email contains at least one link", lambda x: x["links"] > 0),
             ("it has an attachment whose type is zip, exe or xlsm", lambda x: x["has_attachment"] and x["attachment_type"] in ("zip", "exe", "xlsm")),
             ("it uses urgent language", lambda x: x["urgent_language"]),
             ("it asks the reader for credentials", lambda x: x["requests_credentials"]),
             ("it was sent between 22:00 and 05:59 local time", lambda x: x["sent_hour_local"] >= 22 or x["sent_hour_local"] <= 5)]
    return rec, preds


def schema_rental(r):
    rec = {"tenant": _name(r), "monthly_rent_usd": r.randrange(600, 5200, 25), "monthly_income_usd": r.randrange(1500, 16000, 50),
           "pets": r.choice([0, 0, 1, 2]), "lease_months": r.choice([6, 12, 18, 24]), "references": r.randint(0, 3),
           "eviction_history": r.random() < 0.1, "smoker": r.random() < 0.2, "move_in_days_from_now": r.randint(3, 120)}
    preds = [("rent is at most one third of monthly income", lambda x: x["monthly_rent_usd"] * 3 <= x["monthly_income_usd"]),
             ("the tenant has pets", lambda x: x["pets"] > 0),
             ("the lease is at least 12 months", lambda x: x["lease_months"] >= 12),
             ("the tenant provided two or more references", lambda x: x["references"] >= 2),
             ("the tenant has an eviction history", lambda x: x["eviction_history"]),
             ("the tenant smokes", lambda x: x["smoker"]),
             ("move-in is within 30 days", lambda x: x["move_in_days_from_now"] <= 30)]
    return rec, preds


TRAIN_SCHEMAS = [schema_loan, schema_shipment, schema_server, schema_recipe]
EVAL_SCHEMAS = [schema_email, schema_rental]

OPS = {
    "all": ("Are ALL of the following true: {L}?", lambda v: all(v)),
    "any": ("Is AT LEAST ONE of the following true: {L}?", lambda v: any(v)),
    "but_not": ("Is it the case that {a}, but NOT the case that {b}?", None),
    "exactly_one": ("Is EXACTLY ONE of the following true: {L}?", lambda v: sum(v) == 1),
    "none": ("Are NONE of the following true: {L}?", lambda v: not any(v)),
}


def prose(rec, r):
    parts = [f"{k.replace('_', ' ')}: {('yes' if v else 'no') if isinstance(v, bool) else v}" for k, v in rec.items()]
    r.shuffle(parts)
    return "Record details - " + "; ".join(parts) + "."


def make_item(schema, r):
    rec, preds = schema(r)
    k = r.choice([2, 2, 3, 3, 4])
    chosen = r.sample(preds, k=min(k, len(preds)))
    vals = [fn(rec) for _, fn in chosen]
    op = r.choice(list(OPS))
    if op == "but_not":
        chosen, vals = chosen[:2], vals[:2]
        text = OPS[op][0].format(a=chosen[0][0], b=chosen[1][0])
        label = vals[0] and not vals[1]
    else:
        lst = "; ".join(f"({i + 1}) {t}" for i, (t, _) in enumerate(chosen))
        text = OPS[op][0].format(L=lst)
        label = OPS[op][1](vals)
    state = rec if r.random() < 0.5 else prose(rec, r)
    return state, text, bool(label), [(t, bool(v)) for (t, _), v in zip(chosen, vals)], op


def row(state, questions, targets, meta):
    rid = hashlib.sha1(json.dumps([state, questions], sort_keys=True).encode()).hexdigest()[:20]
    return {"request_id": "cmp-" + rid, "state": state, "questions": questions, "targets": targets, "meta": meta}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-out", required=True)
    ap.add_argument("--eval-out", required=True)
    ap.add_argument("--n-train", type=int, default=12000)
    ap.add_argument("--n-eval", type=int, default=600)
    a = ap.parse_args()
    r = random.Random(20261018)
    # balance labels in train (<= 60% majority per op)
    with open(a.train_out, "w") as fh:
        cnt = {True: 0, False: 0}
        n = 0
        while n < a.n_train:
            st, text, lab, atoms, op = make_item(r.choice(TRAIN_SCHEMAS), r)
            if cnt[lab] > 0.55 * (n + 1):
                continue
            cnt[lab] += 1
            qs = {"compound": {"type": "noul", "instructions": text,
                               "criteria": {"true": "The combined condition holds for this record.",
                                            "false": "The combined condition does not hold for this record."}}}
            tg = {"compound": [0.0, 1.0] if lab else [1.0, 0.0]}
            # also teach the atoms on some rows (multi-question states)
            if r.random() < 0.3:
                for i, (t, v) in enumerate(atoms):
                    qs[f"atom{i}"] = {"type": "noul", "instructions": f"Is it true that {t}?"}
                    tg[f"atom{i}"] = [0.0, 1.0] if v else [1.0, 0.0]
            fh.write(json.dumps(row(st, qs, tg, {"source": "verdict-compound", "task_id": f"compound/{op}", "family": "compound",
                                                    "label_type": "exact", "license": "own", "license_use": "commercial",
                                                    "split": "train", "generator": "procedural"})) + "\n")
            n += 1
    import os
    os.makedirs(os.path.dirname(a.eval_out), exist_ok=True)
    with open(a.eval_out, "w") as fh:
        cnt = {True: 0, False: 0}
        n = 0
        while n < a.n_eval:
            st, text, lab, atoms, op = make_item(r.choice(EVAL_SCHEMAS), r)
            if cnt[lab] >= a.n_eval // 2:
                continue
            cnt[lab] += 1
            fh.write(json.dumps({"uid": f"compound-{n}", "op": op, "state": st, "compound": text, "label": lab,
                                 "atoms": [{"q": f"Is it true that {t}?", "label": v} for t, v in atoms]}) + "\n")
            n += 1
    print("ok")


if __name__ == "__main__":
    main()
