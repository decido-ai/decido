"""Merge a Verdict LoRA adapter into the base model and save a full HF checkpoint (bf16).

    python -m decido.merge --base google/gemma-4-26B-A4B-it --adapter runs/train/r1/final --out models/merged/r1
"""
import argparse
import os
import shutil

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

HF_CACHE = os.environ.get("HF_HUB_CACHE")  # None = default HF cache
ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--adapter", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
model = AutoModelForCausalLM.from_pretrained(a.base, cache_dir=HF_CACHE, dtype=torch.bfloat16, device_map="cpu")
model = PeftModel.from_pretrained(model, a.adapter).merge_and_unload()
model.save_pretrained(a.out, safe_serialization=True, max_shard_size="5GB")
AutoTokenizer.from_pretrained(a.base, cache_dir=HF_CACHE).save_pretrained(a.out)
try:
    AutoProcessor.from_pretrained(a.base, cache_dir=HF_CACHE).save_pretrained(a.out)
except Exception as e:  # text-only use; processor is optional
    print("processor not saved:", e)
# keep the base repo's generation/chat files so vLLM and convert_hf_to_gguf see the same model card
from huggingface_hub import snapshot_download
snap = snapshot_download(a.base, cache_dir=HF_CACHE, local_files_only=True)
for f in os.listdir(snap):
    if f.endswith((".json", ".jinja")) and not os.path.exists(os.path.join(a.out, f)):
        shutil.copy(os.path.join(snap, f), os.path.join(a.out, f))
print("merged ->", a.out)
