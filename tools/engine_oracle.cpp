// dump_ref: the llama.cpp oracle for oxe (M1/M2).
//
// Links the installed-production static libraries (KleidiAI ON, OpenMP ON).
// Uses the Rust engine oracle record format, but preserves the production numeric domain.
// Runs a fixed token sequence through it, ONE TOKEN PER llama_decode
// call (the decode path: gemv kernels, fused autoregressive gated delta net), or as one batch
// with --batch. While doing so it can
//   - dump every named intermediate tensor (cb_eval) for selected positions  (--dump-pos)
//   - write the full logits of every evaluated position                      (--logits)
//   - continue greedily for N tokens                                          (--gen N)
// Output directory layout:
//   manifest.json            tokens, generated ids, options
//   pos<P>.tensors           records: u32 name_len, name, u32 type (0 f32, 1 i32), u32 n_dims,
//                            i64 ne[4], payload (contiguous, ne0 fastest)
//   logits.f32               n_positions x n_vocab f32 (if --logits)
//
// Usage: dump_ref --model M.gguf --out DIR (--fixture code|math|json | --tokens 1,2,3)
//                 [--threads 4] [--ctx 4096] [--gen 0] [--dump-pos 0-3] [--logits] [--batch]
//                 [--max-elems 600000]
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cinttypes>
#include <filesystem>
#include <sys/stat.h>
#include <stdexcept>
#include "nlohmann/json.hpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

struct Dumper {
    FILE * f = nullptr;
    int64_t max_elems = 65536;
    size_t n_written = 0;
    size_t bytes_written = 0;
    bool failed = false;
};

static bool want_tensor(const ggml_tensor * t, const Dumper * d) {
    if (t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_I32) {
        return false;
    }
    const char * n = t->name;
    if (n[0] == 0) {
        return false;
    }
    const int64_t ne = ggml_nelements(t);
    if (ne > d->max_elems && strcmp(n, "result_output") != 0) {
        return false;
    }
    return true;
}

static bool cb_eval(struct ggml_tensor * t, bool ask, void * user) {
    Dumper * d = (Dumper *) user;
    if (!d || !d->f) {
        return false;
    }
    if (ask) {
        return want_tensor(t, d);
    }
    if (!want_tensor(t, d)) {
        return true;
    }
    const size_t esz = ggml_type_size(t->type);
    std::vector<uint8_t> raw;
    const bool host = t->buffer == nullptr || ggml_backend_buffer_is_host(t->buffer);
    const uint8_t * base;
    if (host) {
        base = (const uint8_t *) t->data;
    } else {
        raw.resize(ggml_nbytes(t));
        ggml_backend_tensor_get(t, raw.data(), 0, raw.size());
        base = raw.data();
    }
    std::vector<uint8_t> out((size_t) ggml_nelements(t) * esz);
    size_t o = 0;
    for (int64_t i3 = 0; i3 < t->ne[3]; i3++)
        for (int64_t i2 = 0; i2 < t->ne[2]; i2++)
            for (int64_t i1 = 0; i1 < t->ne[1]; i1++)
                for (int64_t i0 = 0; i0 < t->ne[0]; i0++) {
                    const uint8_t * p = base + i0 * t->nb[0] + i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3];
                    memcpy(&out[o], p, esz);
                    o += esz;
                }
    const uint32_t nl = (uint32_t) strlen(t->name);
    const uint32_t ty = t->type == GGML_TYPE_F32 ? 0 : 1;
    const uint32_t nd = 4;
    const size_t bytes = 4 + nl + 8 + 32 + out.size();
    if (d->bytes_written + bytes > 512ULL * 1024 * 1024) { d->failed = true; return false; }
    d->bytes_written += bytes;
    fwrite(&nl, 4, 1, d->f);
    fwrite(t->name, 1, nl, d->f);
    fwrite(&ty, 4, 1, d->f);
    fwrite(&nd, 4, 1, d->f);
    fwrite(t->ne, 8, 4, d->f);
    fwrite(out.data(), 1, out.size(), d->f);
    if (ferror(d->f)) { d->failed = true; return false; }
    d->n_written++;
    return true;
}

static std::vector<int32_t> parse_ids(const std::string & s) {
    std::vector<int32_t> v;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) {
            v.push_back(atoi(item.c_str()));
        }
    }
    return v;
}

static void parse_range(const std::string & s, std::set<int> & out) {
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        auto dash = item.find('-');
        if (dash == std::string::npos) {
            out.insert(atoi(item.c_str()));
        } else {
            int a = atoi(item.substr(0, dash).c_str()), b = atoi(item.substr(dash + 1).c_str());
            for (int i = a; i <= b; i++) out.insert(i);
        }
    }
}

int main(int argc, char ** argv) {
    umask(0077);
    std::string model, out, tokens_s, prompt, fixture, dump_pos;
    int threads = 8, batch_threads = 12, n_ctx = 4096, n_gen = 0;
    bool logits = false, batch = false;
    int64_t max_elems = 65536;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", a.c_str()); exit(2); }
            return argv[++i];
        };
        if (a == "--model") model = next();
        else if (a == "--out") out = next();
        else if (a == "--tokens") tokens_s = next();
        else if (a == "--fixture") fixture = next();
        else if (a == "--threads") threads = atoi(next().c_str());
        else if (a == "--threads-batch") batch_threads = atoi(next().c_str());
        else if (a == "--ctx") n_ctx = atoi(next().c_str());
        else if (a == "--gen") n_gen = atoi(next().c_str());
        else if (a == "--dump-pos") dump_pos = next();
        else if (a == "--logits") logits = true;
        else if (a == "--batch") batch = true;
        else if (a == "--max-elems") max_elems = atoll(next().c_str());
        else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 2; }
    }
    if (model.empty() || out.empty()) {
        fprintf(stderr, "usage: dump_ref --model M --out DIR (--tokens ..|--prompt ..|--prompt-file ..) [opts]\n");
        return 2;
    }
    if (fixture == "code") prompt = "def add(a, b):\n    return a + b\n\n# Explain this function briefly.\n";
    else if (fixture == "math") prompt = "What is 17 times 23? Show a short calculation.\n";
    else if (fixture == "json") prompt = "Return valid JSON with keys x and y containing the integers 3 and 4.\n";
    else if (tokens_s.empty()) { fprintf(stderr, "Choose code/math/json fixture or explicit synthetic token IDs\n"); return 2; }
    if (threads < 1 || threads > 12 || batch_threads < 1 || batch_threads > 12 || n_gen < 0 || n_gen > 64 || n_ctx < 128 || n_ctx > 16384 || max_elems < 1 || max_elems > 1000000) return 2;
    namespace fs = std::filesystem;
    const fs::path dest(out);
    if (!dest.is_absolute() || fs::exists(dest) || fs::is_symlink(dest) || !fs::create_directory(dest)) {
        fprintf(stderr, "Fresh absolute output directory required\n"); return 2;
    }

    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    mp.load_mode = LLAMA_LOAD_MODE_MMAP;
    mp.no_host = true;    // as production (--no-host)
    mp.load_mtp = false;  // MTP off
    mp.use_extra_bufts = true;  // CPU repack (what --repack does)
    llama_model * m = llama_model_load_from_file(model.c_str(), mp);
    if (!m) { fprintf(stderr, "load failed\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(m);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    std::vector<int32_t> toks = parse_ids(tokens_s);
    if (toks.empty() && !prompt.empty()) {
        std::vector<llama_token> buf(prompt.size() + 16);
        int n = llama_tokenize(vocab, prompt.c_str(), (int) prompt.size(), buf.data(), (int) buf.size(), false, true);
        if (n < 0) { fprintf(stderr, "tokenize failed\n"); return 1; }
        toks.assign(buf.begin(), buf.begin() + n);
    }
    if (toks.empty() || toks.size() > 256 || toks.size() + n_gen >= (size_t)n_ctx) { fprintf(stderr, "Invalid bounded token count\n"); return 2; }
    for (auto token : toks) if (token < 0 || token >= n_vocab) return 2;

    Dumper d;
    d.max_elems = max_elems;
    std::set<int> dpos;
    if (!dump_pos.empty()) parse_range(dump_pos, dpos);

    auto cp = llama_context_default_params();
    cp.n_ctx = n_ctx;
    cp.n_batch = batch ? std::max<int>(toks.size(), 1) : 1;
    cp.n_ubatch = cp.n_batch;
    cp.n_seq_max = 1;
    cp.n_threads = threads;
    cp.n_threads_batch = batch_threads;
    cp.offload_kqv = false;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.type_k = GGML_TYPE_F16;
    cp.type_v = GGML_TYPE_F16;
    cp.no_perf = true;
    cp.op_offload = false;
    if (!dpos.empty()) {
        cp.cb_eval = cb_eval;
        cp.cb_eval_user_data = &d;
    }
    llama_context * ctx = llama_init_from_model(m, cp);
    if (!ctx) { fprintf(stderr, "ctx failed\n"); return 1; }
    auto tp_params = ggml_threadpool_params_default(threads);
    auto batch_params = ggml_threadpool_params_default(batch_threads);
    tp_params.strict_cpu = batch_params.strict_cpu = true;
    for (int i = 0; i < threads; ++i) tp_params.cpumask[i] = true;
    for (int i = 0; i < batch_threads; ++i) batch_params.cpumask[i] = true;
    tp_params.paused = true;
    auto * tp_batch = ggml_threadpool_new(&batch_params);
    auto * tp = ggml_threadpool_new(&tp_params);
    llama_attach_threadpool(ctx, tp, tp_batch);

    fprintf(stderr, "ORACLE_MODEL_READY\n");
    FILE * lf = logits ? fopen((out + "/logits.f32").c_str(), "wb") : nullptr;
    if (logits && !lf) return 1;
    std::vector<int32_t> gen;
    auto argmax = [&](const float * lg) {
        int best = 0;
        for (int i = 1; i < n_vocab; i++) if (lg[i] > lg[best]) best = i;
        return best;
    };
    int n_eval = 0;
    auto eval_one = [&](int32_t tok, int pos) -> const float * {
        if (dpos.count(pos)) {
            d.f = fopen((out + "/pos" + std::to_string(pos) + ".tensors").c_str(), "wb");
        }
        llama_batch b = llama_batch_init(1, 0, 1);
        b.n_tokens = 1;
        b.token[0] = tok;
        b.pos[0] = pos;
        b.n_seq_id[0] = 1;
        b.seq_id[0][0] = 0;
        b.logits[0] = 1;
        int rc = llama_decode(ctx, b);
        llama_batch_free(b);
        if (d.f) { fclose(d.f); d.f = nullptr; }
        if (rc != 0 || d.failed) { fprintf(stderr, "decode failed at %d\n", pos); exit(1); }
        n_eval++;
        const float * lg = llama_get_logits_ith(ctx, -1);
        if (lf && fwrite(lg, 4, n_vocab, lf) != (size_t)n_vocab) exit(1);
        return lg;
    };

    const float * last = nullptr;
    if (batch) {
        // whole prompt in one ubatch (prefill path); dumps go to pos<last>.tensors
        int P = (int) toks.size();
        if (!dpos.empty()) d.f = fopen((out + "/pos" + std::to_string(P - 1) + ".batch.tensors").c_str(), "wb");
        llama_batch b = llama_batch_init(P, 0, 1);
        b.n_tokens = P;
        for (int i = 0; i < P; i++) {
            b.token[i] = toks[i]; b.pos[i] = i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = 1;
        }
        if (llama_decode(ctx, b) != 0) { fprintf(stderr, "batch decode failed\n"); return 1; }
        llama_batch_free(b);
        if (d.f) { fclose(d.f); d.f = nullptr; }
        for (int i = 0; i < P; i++) {
            const float * lg = llama_get_logits_ith(ctx, i);
            if (lf && fwrite(lg, 4, n_vocab, lf) != (size_t)n_vocab) exit(1);
            last = lg;
        }
        n_eval = P;
    } else {
        for (size_t i = 0; i < toks.size(); i++) last = eval_one(toks[i], (int) i);
    }
    int pos = (int) toks.size();
    for (int g = 0; g < n_gen; g++) {
        int t = argmax(last);
        gen.push_back(t);
        if (g + 1 < n_gen) last = eval_one(t, pos++);
    }
    if (lf) fclose(lf);

    nlohmann::json manifest = {
        {"format", "engine-oracle-v1"}, {"reference_domain", "production-kleidiai"},
        {"model", model}, {"n_vocab", n_vocab}, {"threads", threads}, {"batch_threads", batch_threads},
        {"n_ctx", n_ctx}, {"batch", batch}, {"n_eval", n_eval}, {"tokens", toks},
        {"generated", gen}, {"dump_pos", dpos}, {"logits", logits}, {"fixture", fixture},
        {"tensor_records", d.n_written}, {"tensor_bytes", d.bytes_written}, {"complete", !d.failed},
        {"note", "Synthetic fixtures only. Serial logits contain one full-vocabulary F32 row per evaluated position. Capturing synchronizes the graph: never use dump timings as inference benchmarks."}
    };
    std::ofstream mf(out + "/manifest.json"); mf << manifest.dump(2) << "\n"; mf.close();
    if (!mf || d.failed) return 1;
    fprintf(stderr, "dump_ref: %d evals, %zu generated, %zu tensors dumped -> %s\n", n_eval, gen.size(), d.n_written, out.c_str());
    llama_detach_threadpool(ctx);
    ggml_threadpool_free(tp); ggml_threadpool_free(tp_batch);
    llama_free(ctx);
    llama_model_free(m);
    llama_backend_free();
    return 0;
}
