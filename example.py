"""Minimal Decido example. Run from this folder after ./build_vserve.sh and `pip install -r requirements.txt`."""
import json

from decido.predictor_llamacpp import DecidoLlamaCpp

m = DecidoLlamaCpp(gguf="decido-26b-a4b-q8_0.gguf", bundle="bundle", threads=10, jev_wire=True)
out = m({
    "state": "Customer: I was charged twice for order 4821. Please refund one of the charges today.",
    "questions": {
        "refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"},
        "team": {"type": "choice", "instructions": "Which team should handle this ticket?",
                 "criteria": {"billing": "Charges, refunds, invoices", "shipping": "Delivery and tracking",
                              "tech": "App or website problems"}},
        "urgency": {"type": "score", "instructions": "How urgent is this ticket?",
                    "criteria": ["Not urgent", "Normal", "Urgent"]},
    },
})
print(json.dumps(out, indent=1))
m.close()
