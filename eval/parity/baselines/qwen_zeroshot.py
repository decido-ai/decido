"""CPU baseline: zero-shot Qwen3 (0.6B by default) letter-logit scorer, in the style of the arXiv 2609.37647
`openmodel` backend (options rendered as single-letter labels, probabilities read from one forward pass,
thinking disabled). Not a Verdict candidate; it exists to exercise the harness end to end on CPU
(device forcing, batching, latency) and to give a floor for a small open model.

    python -m eval.parity.run --predictor eval.parity.baselines.qwen_zeroshot:QwenZeroShot --limit-per-task 50
"""

from __future__ import annotations

import json
import string

import numpy as np

from eval.parity.common import HF_CACHE

LETTERS = list(string.ascii_uppercase)


def _desc(v):
    if v is None:
        return ""
    return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)


class QwenZeroShot:
    def __init__(self, model: str = "Qwen/Qwen3-0.6B", max_len: int = 4096):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model, cache_dir=str(HF_CACHE / "hub"))
        self.model = AutoModelForCausalLM.from_pretrained(model, cache_dir=str(HF_CACHE / "hub"),
                                                          dtype=torch.float32).to("cpu").eval()
        self.max_len = max_len
        self.letter_ids = [self.tok.encode(l, add_special_tokens=False)[0] for l in LETTERS]

    def _prompt(self, state, q):
        st = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
        t = q["type"]
        if t == "noul":
            crit = q.get("criteria") or {}
            opts = [("true", _desc(crit.get("true")) or "Yes"), ("false", _desc(crit.get("false")) or "No")]
        elif t == "choice":
            opts = [(k, _desc(v)) for k, v in q["criteria"].items()]
        else:
            opts = [(str(i), _desc(v)) for i, v in enumerate(q["criteria"])]
        lines = [f"{LETTERS[i]}) {k}" + (f": {d}" if d and d != k else "") for i, (k, d) in enumerate(opts)]
        instr = q["instructions"] if isinstance(q["instructions"], str) else json.dumps(q["instructions"], ensure_ascii=False)
        user = (f"<document>\n{st}\n</document>\n\nQuestion: {instr}\nOptions:\n" + "\n".join(lines) +
                "\nAnswer with the letter of the correct option only.")
        msgs = [{"role": "user", "content": user}]
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return text, [k for k, _ in opts]

    def __call__(self, request):
        torch = self.torch
        out = {}
        with torch.inference_mode():
            for qid, q in request["questions"].items():
                text, keys = self._prompt(request["state"], q)
                ids = self.tok(text, return_tensors="pt", truncation=True, max_length=self.max_len)
                logits = self.model(**ids).logits[0, -1]
                z = logits[self.letter_ids[:len(keys)]].float().numpy()
                p = np.exp(z - z.max())
                p /= p.sum()
                if q["type"] == "noul":
                    out[qid] = {"noul": float(p[0])}
                else:
                    out[qid] = {"probabilities": {k: float(v) for k, v in zip(keys, p)}}
        return {"answers": out}
