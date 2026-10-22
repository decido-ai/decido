"""GPU predictor for the parity harness (intermediate runs only; the correctness number is the CPU GGUF run).

    VERDICT_ALLOW_GPU=1 python -m eval.parity.run --allow-gpu --batch-size 4096 \
        --predictor decido.predictor_hf:VerdictVLLM --predictor-kw model=Qwen/Qwen3-30B-A3B-Instruct-2507

Backends:
  VerdictVLLM  vLLM with automatic prefix caching: the state prefix is computed once per request and every
               question/permutation reuses it; readout = raw log-probs of the label tokens (logprob_token_ids).
  VerdictHF    plain transformers forward (bf16), used for GGUF-vs-HF logit agreement checks.
"""

from __future__ import annotations

import json
import os

from decido.readout import Readout
from decido.template import ChatFormat

HF_CACHE = os.environ.get("HF_HUB_CACHE")  # None = default HF cache


def _load_temps(temps):
    if temps is None or temps == "":
        return None
    if isinstance(temps, dict):
        return temps
    with open(temps) as fh:
        return json.load(fh)


class VerdictVLLM:
    def __init__(self, model: str, tokenizer: str | None = None, tp: int = 1, max_model_len: int = 16384,
                 gpu_mem: float = 0.90, perm: int = 1, temps=None, family: str | None = None,
                 lora: str | None = None, max_lora_rank: int = 64, enforce_eager: bool = False,
                 max_num_seqs: int = 512, dtype: str = "bfloat16", quantization: str | None = None):
        from transformers import AutoTokenizer
        from vllm import LLM
        self.tok = AutoTokenizer.from_pretrained(tokenizer or model, cache_dir=HF_CACHE)
        self.cf = ChatFormat(self.tok, family)
        self.readout = Readout(self.cf, perm=perm, temps=_load_temps(temps), max_prompt_tokens=max_model_len - 8)
        kw = dict(model=model, tokenizer=tokenizer or model, tensor_parallel_size=int(tp),
                  max_model_len=int(max_model_len), gpu_memory_utilization=float(gpu_mem),
                  enable_prefix_caching=True, dtype=dtype, download_dir=HF_CACHE, seed=0,
                  max_num_seqs=int(max_num_seqs), enforce_eager=bool(enforce_eager),
                  limit_mm_per_prompt={"image": 0, "video": 0, "audio": 0})
        if quantization:
            kw["quantization"] = quantization
        if lora:
            kw.update(enable_lora=True, max_lora_rank=int(max_lora_rank), max_loras=1)
        try:
            self.llm = LLM(**kw)
        except TypeError:
            kw.pop("limit_mm_per_prompt")
            self.llm = LLM(**kw)
        self.lora = None
        if lora:
            from vllm.lora.request import LoRARequest
            self.lora = LoRARequest("verdict", 1, lora)
        self.model_name = model

    def _run(self, prompts: list[list[int]], token_sets: list[list[int]]) -> list[dict]:
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt
        sps = [SamplingParams(max_tokens=1, temperature=0.0, logprob_token_ids=ts, logprobs=None, seed=0)
               for ts in token_sets]
        outs = self.llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts], sps, use_tqdm=False,
                                 lora_request=self.lora)
        res = []
        for o, ts in zip(outs, token_sets):
            lp = o.outputs[0].logprobs[0]
            res.append({t: float(lp[t].logprob) for t in ts if t in lp})
        return res

    def predict_batch(self, requests):
        plans = [self.readout.plan(r) for r in requests]
        prompts, sets, owners = [], [], []
        for pl in plans:
            for qu in pl.queries:
                prompts.append(pl.prefix_ids + qu.suffix_ids)
                sets.append(qu.trie.first_token_ids())
                owners.append(qu)
        for qu, r in zip(owners, self._run(prompts, sets)):
            qu.step1 = r
        # second step for >26-option choices: append each first symbol, read the digit
        prompts, sets, owners = [], [], []
        for pl in plans:
            for qu in pl.queries:
                if qu.trie.two_step:
                    digits = sorted({i for v in qu.trie.second_variants.values() for i in v})
                    for sym in qu.trie.first_syms:
                        prompts.append(pl.prefix_ids + qu.suffix_ids + [qu.trie.first_canon[sym]])
                        sets.append(digits)
                        owners.append((qu, sym))
        if prompts:
            for (qu, sym), r in zip(owners, self._run(prompts, sets)):
                qu.step2[sym] = r
        return [self.readout.combine(pl) for pl in plans]

    def __call__(self, request):
        return self.predict_batch([request])[0]


class VerdictHF:
    """Reference transformers implementation (single request at a time, prefix KV reused across questions)."""

    def __init__(self, model: str, perm: int = 1, temps=None, family: str | None = None, device: str = "cuda",
                 dtype: str = "bfloat16", lora: str | None = None, max_model_len: int = 16384):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model, cache_dir=HF_CACHE)
        self.cf = ChatFormat(self.tok, family)
        self.readout = Readout(self.cf, perm=perm, temps=_load_temps(temps), max_prompt_tokens=max_model_len - 8)
        self.model = AutoModelForCausalLM.from_pretrained(model, cache_dir=HF_CACHE, dtype=getattr(torch, dtype),
                                                          device_map=device).eval()
        if lora:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, lora).merge_and_unload().eval()
        self.device = next(self.model.parameters()).device

    def next_logprobs(self, ids: list[int]):
        torch = self.torch
        with torch.inference_mode():
            x = torch.tensor([ids], device=self.device)
            logits = self.model(input_ids=x).logits[0, -1].float()
            return torch.log_softmax(logits, -1).cpu()

    def __call__(self, request):
        pl = self.readout.plan(request)
        for qu in pl.queries:
            lp = self.next_logprobs(pl.prefix_ids + qu.suffix_ids)
            qu.step1 = {t: float(lp[t]) for t in qu.trie.first_token_ids()}
            if qu.trie.two_step:
                digits = sorted({i for v in qu.trie.second_variants.values() for i in v})
                for sym in qu.trie.first_syms:
                    lp2 = self.next_logprobs(pl.prefix_ids + qu.suffix_ids + [qu.trie.first_canon[sym]])
                    qu.step2[sym] = {t: float(lp2[t]) for t in digits}
        return self.readout.combine(pl)
