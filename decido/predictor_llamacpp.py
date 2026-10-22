"""Decido CPU predictor (llama.cpp via the bundled vserve binary, state-once prefix cache).

    from decido.predictor_llamacpp import DecidoLlamaCpp
    m = DecidoLlamaCpp(gguf="decido-26b-a4b-q8_0.gguf", bundle="bundle", threads=10, jev_wire=True)
    m({"state": ..., "questions": {qid: {"type": "noul"|"choice"|"score", "instructions": ..., "criteria": ...}}})

`bundle` holds template.json (frozen chat head/tail + template version), tokenizer.json and temps.json.
The vserve binary (tools/vserve/vserve.cpp, linked against a CPU-only libllama) prefills the state once per
request and answers every question/permutation suffix from the cached prefix. n_gpu_layers defaults to 0
(CPU only, the deliverable); gpu_layers>0 exists only for fast GPU screening of quants on non-test data.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading

from decido.readout import Readout
from decido.template import FastTok, StaticChatFormat

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_BIN = os.path.join(os.path.dirname(_HERE), "tools", "vserve", "vserve")


class DecidoLlamaCpp:
    def __init__(self, gguf: str, bundle: str, threads: int | None = None, n_ctx: int = 16384, ubatch: int = 512,
                 perm: int = 1, vserve: str | None = None, temps: str | None = None, workers: int = 1,
                 gpu_layers: int = 0, gpu_ids: str | None = None, on_overflow: str = "error",
                 jev_wire: bool = False):
        spec = json.load(open(os.path.join(bundle, "template.json")))
        self.tok = FastTok(os.path.join(bundle, "tokenizer.json"))
        self.cf = StaticChatFormat(self.tok, spec)
        tp = temps or os.path.join(bundle, "temps.json")
        tmap = json.load(open(tp)) if tp and os.path.exists(tp) else None
        self.readout = Readout(self.cf, perm=perm, temps=tmap, max_prompt_tokens=n_ctx - 8, on_overflow=on_overflow)
        self.jev_wire = bool(jev_wire)   # True: Jev 1.13 wire format (2-dp rounding computed after derived fields,
                                         # `model` and `usage`); False (evaluation default): unrounded probabilities
        self.bin = vserve or os.environ.get("VERDICT_VSERVE", _DEFAULT_BIN)
        if threads is None:  # the harness latency mode sets OMP_NUM_THREADS per subprocess
            threads = int(os.environ.get("OMP_NUM_THREADS") or 10)
        self.gpu_layers = int(gpu_layers)  # 0 = CPU only (the deliverable); >0 only for GPU quant screening
        if self.gpu_layers and vserve is None:
            self.bin = os.environ.get("VERDICT_VSERVE_CUDA", self.bin + "-cuda")
        self.args = [self.bin, "-m", gguf, "-t", str(threads), "-c", str(n_ctx), "-ub", str(ubatch),
                     "--gpu-layers", str(self.gpu_layers)]
        self.gpu_ids = [g for g in str(gpu_ids).split(",") if g] if (gpu_ids is not None and self.gpu_layers) else []
        self.procs = [self._spawn(i) for i in range(int(workers))]
        self.info = {"gguf": gguf, "threads": threads, "workers": workers, "n_gpu_layers": self.gpu_layers,
                     "template_version": spec["template_version"], "temps": tmap}

    def _spawn(self, i: int = 0):
        if not os.path.exists(self.bin):
            raise FileNotFoundError(f"vserve binary not found at {self.bin}: build it first "
                                    "(out/build_vserve_mac.sh <llama.cpp dir> <build dir>) or set VERDICT_VSERVE")
        env = dict(os.environ) if self.gpu_layers else dict(os.environ, CUDA_VISIBLE_DEVICES="")
        if self.gpu_ids:
            env["CUDA_VISIBLE_DEVICES"] = self.gpu_ids[i % len(self.gpu_ids)]
        p = subprocess.Popen(self.args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1, env=env)
        ready = json.loads(p.stdout.readline())
        if not ready.get("ready") or ready.get("gpu_layers", 0) != self.gpu_layers:
            raise RuntimeError(f"vserve failed to start: {ready}")
        return p

    def _ask(self, proc, prefix, queries):
        proc.stdin.write(json.dumps({"prefix": prefix, "queries": queries}) + "\n")
        proc.stdin.flush()
        out = json.loads(proc.stdout.readline())
        if "error" in out:
            raise RuntimeError(out["error"])
        return out

    def _predict(self, proc, request):
        pl = self.readout.plan(request)
        qs = []
        for qu in pl.queries:
            q = {"suffix": qu.suffix_ids, "ids": qu.trie.first_token_ids()}
            if qu.trie.two_step:
                q["ext"] = [qu.trie.first_canon[s] for s in qu.trie.first_syms]
                q["ext_ids"] = sorted({i for v in qu.trie.second_variants.values() for i in v})
            qs.append(q)
        out = self._ask(proc, pl.prefix_ids, qs)
        self.last_timing = {k: out.get(k) for k in ("prefill_ms", "queries_ms", "n_prefix", "reused")}
        for qu, q, lps, ext in zip(pl.queries, qs, out["logprobs"], out["ext_logprobs"]):
            qu.step1 = dict(zip(q["ids"], lps))
            for sym, elp in zip(qu.trie.first_syms if qu.trie.two_step else [], ext):
                qu.step2[sym] = dict(zip(q["ext_ids"], elp))
        if not self.jev_wire:
            return self.readout.combine(pl)
        out = self.readout.combine(pl, round_dp=2)
        out.update(model="decido-26b-a4b-q8_0", usage={"input_tokens": len(pl.prefix_ids) + sum(
            len(q.suffix_ids) for q in pl.queries), "output_tokens": 0})
        return out

    def __call__(self, request):
        return self._predict(self.procs[0], request)

    def predict_batch(self, requests):
        if len(self.procs) == 1:
            return [self._predict(self.procs[0], r) for r in requests]
        res = [None] * len(requests)

        n = len(self.procs)
        bounds = [len(requests) * w // n for w in range(n + 1)]

        def work(w):  # contiguous chunks keep consecutive requests (shared premises) on one worker's KV
            for i in range(bounds[w], bounds[w + 1]):
                res[i] = self._predict(self.procs[w], requests[i])
        ts = [threading.Thread(target=work, args=(w,)) for w in range(len(self.procs))]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        return res

    def close(self):
        for p in self.procs:
            p.stdin.close()
            p.wait(timeout=30)


VerdictLlamaCpp = DecidoLlamaCpp  # original internal name
