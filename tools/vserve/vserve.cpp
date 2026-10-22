// vserve: CPU libllama verbalizer-readout server for Verdict (state-once prefix cache).
//
// Protocol (one JSON object per line on stdin, one per line on stdout):
//   in : {"prefix":[tok...], "queries":[{"suffix":[tok...], "ids":[tok...]}, ...]}
//   out: {"logprobs":[[lp(ids[0]), lp(ids[1]), ...], ...], "prefill_ms":x, "queries_ms":y, "n_prefix":n}
// lp = full-vocabulary log-softmax at the last suffix position (exact, fp32/fp64 accumulation).
// The prefix is decoded once; every query is evaluated against the cached prefix: KV truncation
// (llama_memory_seq_rm) when the memory supports it, otherwise a snapshot/restore of the sequence state
// (hybrid / recurrent models). Queries are evaluated one at a time in request order, so outputs do not
// depend on how many questions are batched: deterministic for a fixed thread count.
//
// usage: vserve -m model.gguf [-t threads] [-c n_ctx] [-ub ubatch] [--gpu-layers N (default 0 = CPU)]
#include "llama.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

static double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// --- minimal JSON parsing for the fixed protocol (arrays of ints) ---------------------------------
struct P {
    const std::string & s; size_t i = 0;
    explicit P(const std::string & s_) : s(s_) {}
    void ws() { while (i < s.size() && isspace((unsigned char)s[i])) i++; }
    bool eat(char c) { ws(); if (i < s.size() && s[i] == c) { i++; return true; } return false; }
    std::string key() { ws(); if (s[i] != '"') throw 1; size_t j = s.find('"', i + 1); std::string k = s.substr(i + 1, j - i - 1); i = j + 1; return k; }
    long num() { ws(); char * e; long v = strtol(s.c_str() + i, &e, 10); i = e - s.c_str(); return v; }
    std::vector<int> ints() { std::vector<int> v; if (!eat('[')) throw 2; if (eat(']')) return v; do { v.push_back((int)num()); } while (eat(',')); if (!eat(']')) throw 3; return v; }
    void skip() {  // skip any JSON value
        ws(); char c = s[i];
        if (c == '"') { key(); return; }
        if (c == '[' || c == '{') { int d = 0; bool str = false; for (; i < s.size(); i++) { char x = s[i]; if (str) { if (x == '\\') i++; else if (x == '"') str = false; continue; } if (x == '"') str = true; else if (x == '[' || x == '{') d++; else if (x == ']' || x == '}') { if (--d == 0) { i++; return; } } } return; }
        while (i < s.size() && s[i] != ',' && s[i] != '}' && s[i] != ']') i++;
    }
};

struct Query { std::vector<int> suffix, ids, ext, ext_ids; };

static bool parse(const std::string & line, std::vector<int> & prefix, std::vector<Query> & qs) {
    try {
        P p(line);
        if (!p.eat('{')) return false;
        do {
            std::string k = p.key(); p.eat(':');
            if (k == "prefix") prefix = p.ints();
            else if (k == "queries") {
                p.eat('[');
                if (!p.eat(']')) {
                    do {
                        Query q; p.eat('{');
                        do { std::string kk = p.key(); p.eat(':'); if (kk == "suffix") q.suffix = p.ints(); else if (kk == "ids") q.ids = p.ints(); else if (kk == "ext") q.ext = p.ints(); else if (kk == "ext_ids") q.ext_ids = p.ints(); else p.skip(); } while (p.eat(','));
                        p.eat('}'); qs.push_back(q);
                    } while (p.eat(','));
                    p.eat(']');
                }
            } else p.skip();
        } while (p.eat(','));
        return true;
    } catch (...) { return false; }
}

int main(int argc, char ** argv) {
    std::string model_path; int threads = 8, n_ctx = 16384, ubatch = 512, ngl = 0, chunk = 256;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto nx = [&]() { return std::string(argv[++i]); };
        if (a == "-m") model_path = nx(); else if (a == "-t") threads = atoi(nx().c_str());
        else if (a == "-c") n_ctx = atoi(nx().c_str()); else if (a == "-ub") ubatch = atoi(nx().c_str());
        else if (a == "--gpu-layers") ngl = atoi(nx().c_str());
        else if (a == "--chunk") chunk = atoi(nx().c_str());
    }
    llama_backend_init();
    // print warnings/errors, except known-harmless Gemma-4 vocab/cache notices emitted on every load
    llama_log_set([](ggml_log_level lvl, const char * txt, void *) {
        if (lvl < GGML_LOG_LEVEL_WARN) return;
        static const char * benign[] = {"control-looking token", "is a normal token here, removing it from EOG list",
                                        "using full-size SWA cache"};
        for (const char * b : benign) if (strstr(txt, b)) return;
        fputs(txt, stderr);
    }, nullptr);
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) { fprintf(stderr, "load failed\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    auto cp = llama_context_default_params();
    cp.n_ctx = n_ctx; cp.n_batch = n_ctx; cp.n_ubatch = std::max(ubatch, chunk); cp.n_seq_max = 1; cp.swa_full = true;
    cp.n_threads = threads; cp.n_threads_batch = threads; cp.no_perf = true;
    if (ngl == 0) { cp.offload_kqv = false; cp.op_offload = false; }
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "ctx failed\n"); return 1; }
    llama_memory_t mem = llama_get_memory(ctx);
    llama_batch batch = llama_batch_init(n_ctx, 0, 1);
    auto add = [&](int t, int pos, bool logits) {
        int i = batch.n_tokens++; batch.token[i] = t; batch.pos[i] = pos; batch.n_seq_id[i] = 1; batch.seq_id[i][0] = 0; batch.logits[i] = logits;
    };
    std::vector<uint8_t> snap, snap2;
    std::vector<int> cached;            // tokens currently in the KV for seq 0 (prefix of the last request)
    const bool can_trunc = !(llama_model_is_recurrent(model) || llama_model_is_hybrid(model));
    auto logprobs_at = [&](int idx, const std::vector<int> & ids, std::string & out) {
        const float * lg = llama_get_logits_ith(ctx, idx);
        float mx = -INFINITY; for (int v = 0; v < n_vocab; v++) mx = std::max(mx, lg[v]);
        double z = 0; for (int v = 0; v < n_vocab; v++) z += std::exp((double)lg[v] - mx);
        double lz = mx + std::log(z);
        char buf[64]; out += "[";
        for (size_t k = 0; k < ids.size(); k++) {
            int id = ids[k];
            double lp = (id >= 0 && id < n_vocab) ? (double)lg[id] - lz : -1e30;
            snprintf(buf, sizeof buf, "%s%.9g", k ? "," : "", lp); out += buf;
        }
        out += "]";
    };
    fprintf(stdout, "{\"ready\":true,\"n_vocab\":%d,\"threads\":%d,\"n_ctx\":%d,\"gpu_layers\":%d}\n", n_vocab, threads, n_ctx, ngl);
    fflush(stdout);
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        std::vector<int> prefix; std::vector<Query> qs;
        if (!parse(line, prefix, qs)) { printf("{\"error\":\"bad request\"}\n"); fflush(stdout); continue; }
        int L = (int)prefix.size();
        size_t maxs = 0; for (auto & q : qs) maxs = std::max(maxs, q.suffix.size());
        if (L + (int)maxs > n_ctx || L == 0) { printf("{\"error\":\"too long\"}\n"); fflush(stdout); continue; }
        double t0 = now_ms();
        // reuse the longest common prefix with the previous request (KV truncation models only)
        int c = 0;
        if (can_trunc) {
            while (c < (int)cached.size() && c < L - 1 && cached[c] == prefix[c]) c++;
            c = (c / chunk) * chunk;   // resume only at a canonical chunk boundary (history-independent numerics)
            if (c > 0) llama_memory_seq_rm(mem, 0, c, -1); else llama_memory_clear(mem, true);
        } else {
            llama_memory_clear(mem, true);
        }
        bool ok = true;
        // canonical decode: fixed absolute chunks [k*chunk, (k+1)*chunk), so every prefix token is always computed
        // in the same batch composition whether or not earlier chunks came from the cache -> bit-identical outputs
        for (int s0 = c; s0 < L && ok; s0 += chunk) {
            batch.n_tokens = 0;
            for (int i = s0; i < std::min(L, s0 + chunk); i++) add(prefix[i], i, false);
            ok = llama_decode(ctx, batch) == 0;
        }
        if (!ok) { cached.clear(); printf("{\"error\":\"prefix decode failed\"}\n"); fflush(stdout); continue; }
        llama_synchronize(ctx);
        cached = prefix;
        double t1 = now_ms();
        bool trunc = can_trunc;
        if (!trunc) {
            snap.resize(llama_state_seq_get_size(ctx, 0));
            llama_state_seq_get_data(ctx, snap.data(), snap.size(), 0);
        }
        std::string out = "{\"logprobs\":[", ext_out = ",\"ext_logprobs\":[";
        for (size_t qi = 0; qi < qs.size(); qi++) {
            const Query & q = qs[qi];
            if (qi > 0) {
                if (trunc) llama_memory_seq_rm(mem, 0, L, -1);
                else { llama_memory_clear(mem, true); llama_state_seq_set_data(ctx, snap.data(), snap.size(), 0); }
            }
            int S = (int)q.suffix.size();
            batch.n_tokens = 0;
            for (int i = 0; i < S; i++) add(q.suffix[i], L + i, i + 1 == S);
            if (llama_decode(ctx, batch) != 0) { ok = false; break; }
            if (qi) out += ",";
            logprobs_at(batch.n_tokens - 1, q.ids, out);
            // second-step tokens: each evaluated at position L+S on top of the cached suffix
            if (qi) ext_out += ",";
            ext_out += "[";
            if (!q.ext.empty() && !trunc) {
                snap2.resize(llama_state_seq_get_size(ctx, 0));
                llama_state_seq_get_data(ctx, snap2.data(), snap2.size(), 0);
            }
            for (size_t e = 0; e < q.ext.size() && ok; e++) {
                if (e > 0) {
                    if (trunc) llama_memory_seq_rm(mem, 0, L + S, -1);
                    else { llama_memory_clear(mem, true); llama_state_seq_set_data(ctx, snap2.data(), snap2.size(), 0); }
                }
                batch.n_tokens = 0;
                add(q.ext[e], L + S, true);
                if (llama_decode(ctx, batch) != 0) { ok = false; break; }
                if (e) ext_out += ",";
                logprobs_at(0, q.ext_ids, ext_out);
            }
            ext_out += "]";
        }
        if (trunc) llama_memory_seq_rm(mem, 0, L, -1);
        else cached.clear();
        if (!ok) { cached.clear(); printf("{\"error\":\"query decode failed\"}\n"); fflush(stdout); continue; }
        char tail[160];
        snprintf(tail, sizeof tail, "],\"prefill_ms\":%.2f,\"queries_ms\":%.2f,\"n_prefix\":%d,\"reused\":%d,\"trunc\":%s}\n", t1 - t0, now_ms() - t1, L, c, trunc ? "true" : "false");
        out += "]" + ext_out + tail;
        fputs(out.c_str(), stdout); fflush(stdout);
    }
    llama_batch_free(batch); llama_free(ctx); llama_model_free(model);
    return 0;
}
