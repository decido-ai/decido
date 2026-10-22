"""Freeze the prompt template + tokenizer (+ per-kind temperatures) for the CPU predictor.

    python -m decido.make_bundle --model <hf id or dir> --out out/bundle-<name> [--temps temps.json]
"""
import argparse
import json
import os
import shutil

from transformers import AutoTokenizer

from decido.template import ChatFormat, FastTok, export_template_spec

HF_CACHE = os.environ.get("HF_HUB_CACHE")  # None = default HF cache

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--temps", default=None)
a = ap.parse_args()
tok = AutoTokenizer.from_pretrained(a.model, cache_dir=HF_CACHE)
cf = ChatFormat(tok)
os.makedirs(a.out, exist_ok=True)
tok.save_pretrained(os.path.join(a.out, "hf_tokenizer"))
shutil.copy(os.path.join(a.out, "hf_tokenizer", "tokenizer.json"), os.path.join(a.out, "tokenizer.json"))
spec = export_template_spec(cf, {"model": a.model})
json.dump(spec, open(os.path.join(a.out, "template.json"), "w"), indent=1, ensure_ascii=False)
if a.temps:
    shutil.copy(a.temps, os.path.join(a.out, "temps.json"))
# self-check: FastTok must reproduce the HF tokenizer ids on a template sample
ft = FastTok(os.path.join(a.out, "tokenizer.json"))
sample = cf.prefix_text({"a": "Ünïcode ✓ text", "n": [1, 2.5]}) + cf.suffix_text("Question: x?\nA. yes\nB. no")
assert ft.encode(sample) == tok.encode(sample, add_special_tokens=False), "FastTok != HF tokenizer"
print("bundle ok:", a.out)
