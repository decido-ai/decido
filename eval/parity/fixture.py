"""Build the frozen CPU-latency fixture: one request with a ~1,000-token state (Qwen3 tokenizer count of
json.dumps(state)) and 5 questions (2 noul, 2 choice, 1 score). Deterministic, no model calls.

    python -m eval.parity.fixture
"""

from __future__ import annotations

import json

from eval.parity.common import DATA, HF_CACHE

TOKENIZER = next((HF_CACHE / "hub" / "models--Qwen--Qwen3-0.6B" / "snapshots").glob("*/tokenizer.json"))

MESSAGES = [
    "Hi, I placed order 48213 on the 3rd and the tracking page has said 'label created' for six days now.",
    "Thanks for reaching out. I can see the parcel was handed to the regional carrier yesterday evening.",
    "That is not what the carrier site says. It still shows no scan at all. I needed this for a trip on Friday.",
    "I understand. The carrier sometimes delays the first scan by a day or two after pickup from our warehouse.",
    "It has been almost a week. I paid for two-day shipping, which was an extra 14.99, and I would like that back.",
    "I have asked our logistics team to open a trace. Expedited shipping refunds are handled once the trace closes.",
    "How long does a trace take? Nobody told me any of this when I ordered, and the product page promised two days.",
    "Traces usually close within three business days. If the parcel is lost we will reship or refund in full.",
    "I would rather just cancel at this point. Can you stop the shipment and refund the card I used?",
    "Once a parcel is with the carrier we cannot intercept it, but you can refuse delivery and we will refund on return.",
    "So I have to wait for something I no longer need and then deal with sending it back? That seems backwards.",
    "I am sorry for the trouble. I have added a note to your account and applied a 10 percent credit for next time.",
    "A credit is not a refund. Please escalate this to a supervisor. I have been a customer for four years.",
    "I have escalated the ticket. A supervisor will reply by email within one business day.",
]


def build_state(n_msgs_repeat: int, tail_words: int) -> dict:
    thread = []
    for r in range(n_msgs_repeat):
        for i, m in enumerate(MESSAGES):
            who = "customer" if i % 2 == 0 else "agent"
            thread.append({"from": who, "day": 1 + (r * len(MESSAGES) + i) // 4, "text": m})
    note = ("Internal note: the warehouse confirms pickup scan exists in our system but the carrier API returns no "
            "events. Similar complaints rose this week for the same regional carrier. ").split()
    return {"ticket": {"id": "T-90417", "channel": "email", "customer_tier": "gold", "order_id": "48213",
                       "order_total_usd": 129.0, "shipping": "two-day", "shipping_fee_usd": 14.99},
            "thread": thread, "notes": " ".join((note * 20)[:tail_words])}


QUESTIONS = {
    "wants_refund": {"type": "noul", "instructions": "Is the customer explicitly asking for money back?"},
    "needs_supervisor": {"type": "noul", "instructions": "Has the customer asked for escalation to a supervisor?",
                         "criteria": {"true": "An explicit request to escalate or speak to a manager/supervisor.",
                                      "false": "No explicit escalation request."}},
    "department": {"type": "choice", "instructions": "Which team should own this ticket next?",
                   "criteria": {"logistics": "Shipping, carriers, traces, lost parcels.",
                                "billing": "Refunds, charges, credits, invoices.",
                                "product": "Product defects or questions about features.",
                                "retention": "Loyal customers at risk of churning."}},
    "root_cause": {"type": "choice", "instructions": "What is the most likely root cause of the delay?",
                   "criteria": {"carrier_scan_gap": None, "warehouse_delay": None, "address_error": None,
                                "payment_hold": None, "customs": None, "customer_error": None}},
    "frustration": {"type": "score", "instructions": "How frustrated is the customer by the end of the thread?",
                    "criteria": ["Calm", "Mildly annoyed", "Frustrated", "Very frustrated", "Furious"]},
}


def main():
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(TOKENIZER))

    def n(state):
        return len(tok.encode(json.dumps(state, ensure_ascii=False), add_special_tokens=False).ids)

    best = None
    for reps in (1, 2, 3):
        for tail in range(0, 400, 2):
            st = build_state(reps, tail)
            c = n(st)
            if best is None or abs(c - 1000) < abs(best[0] - 1000):
                best = (c, reps, tail, st)
            if c > 1010:
                break
    c, reps, tail, st = best
    qtok = len(tok.encode(json.dumps(QUESTIONS, ensure_ascii=False), add_special_tokens=False).ids)
    fx = {"description": "support-ticket JSON state (~1k Qwen3 tokens) + 5 questions (2 noul, 2 choice[4,6], 1 score[5])",
          "state_tokens_qwen3": c, "question_tokens_qwen3": qtok, "tokenizer": str(TOKENIZER.relative_to(HF_CACHE)),
          "request": {"state": st, "questions": QUESTIONS}}
    (DATA / "latency_fixture.json").write_text(json.dumps(fx, indent=1, ensure_ascii=False))
    print(f"fixture: state {c} tokens, questions {qtok} tokens (reps={reps}, tail={tail})")


if __name__ == "__main__":
    main()
