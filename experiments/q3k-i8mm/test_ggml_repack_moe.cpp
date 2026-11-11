// ggml-level correctness test for the Q3_K repack (patch 1) and the MUL_MAT_ID gemm path (patch 2).
//
// Links only libggml-cpu.a + libggml-base.a from a build of the patched tree. Graph threads: Q3K_TEST_THREADS (default 1).
//
// For every repacked type that this CPU selects (Q3_K, Q4_0, Q4_K, Q5_K, Q6_K, IQ4_NL, MXFP4, Q8_0, Q1_0):
//   1. kernel level: on real repacked weights, the gemm the MUL_MAT_ID path uses (4 interleaved rows) vs gemv
//      (each row), required bitwise. For Q4_K/Q5_K that is ggml_gemm_q{4,5}_K_8x8_q8_K_gemv_order; the MUL_MAT
//      gemm is printed for information only. Also the row interleave vs ggml_quantize_mat_* on the same floats.
//   2. graph level: MUL_MAT_ID with Ornith-shaped experts (256 experts, top-8, 2048 -> 512 "gate" and
//      512 -> 2048 "down"), uneven routing (counts 61/37/23/13/8/7/5/2 plus 1-2-row cold experts), weights in
//      CPU_REPACK vs the same weights in the plain CPU buffer (vec_dot), bitwise mismatches + an output hash.
//      Tokens 1, 3, 61 and 256 (a full ubatch). Run with GGML_CPU_MOE_GEMM=0, =1 and =exact at 1, 4 and 8 threads;
//      run_test_ggml_repack_moe.sh requires every output hash to be identical across all nine runs, i.e. the MoE
//      gemm must be bitwise identical to the gemv path for every type (a model-level KLD gate expects exactly 0).
//   3. MUL_MAT 2-D Q3_K (M = 1, 3, 7, 37): repacked (gemm + gemv) vs vec_dot.
//
// Build: see run_test_ggml_repack_moe.sh

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <strings.h>
#include <random>
#include <string>
#include <vector>

// Kernel entry points (C linkage, defined in libggml-cpu.a; declared in ggml-cpu/repack.h)
extern "C" {
#define KDECL(name) void name(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc);
KDECL(ggml_gemv_q3_K_8x8_q8_K) KDECL(ggml_gemm_q3_K_8x8_q8_K)
KDECL(ggml_gemv_q4_0_4x8_q8_0) KDECL(ggml_gemm_q4_0_4x8_q8_0)
KDECL(ggml_gemv_q4_K_8x8_q8_K) KDECL(ggml_gemm_q4_K_8x8_q8_K) KDECL(ggml_gemm_q4_K_8x8_q8_K_gemv_order)
KDECL(ggml_gemv_q5_K_8x8_q8_K) KDECL(ggml_gemm_q5_K_8x8_q8_K) KDECL(ggml_gemm_q5_K_8x8_q8_K_gemv_order)
KDECL(ggml_gemv_q6_K_8x8_q8_K) KDECL(ggml_gemm_q6_K_8x8_q8_K)
KDECL(ggml_gemv_iq4_nl_4x4_q8_0) KDECL(ggml_gemm_iq4_nl_4x4_q8_0)
KDECL(ggml_gemv_mxfp4_4x4_q8_0) KDECL(ggml_gemm_mxfp4_4x4_q8_0)
KDECL(ggml_gemv_q8_0_4x8_q8_0) KDECL(ggml_gemm_q8_0_4x8_q8_0)
KDECL(ggml_gemv_q1_0_4x8_q8_0) KDECL(ggml_gemm_q1_0_4x8_q8_0)
#undef KDECL
void ggml_quantize_mat_q8_0_4x4(const float * x, void * vy, int64_t k);
void ggml_quantize_mat_q8_0_4x8(const float * x, void * vy, int64_t k);
void ggml_quantize_mat_q8_K_4x8(const float * x, void * vy, int64_t k);
}

typedef void (*kernel_t)(int, float *, size_t, const void *, const void *, int, int);

struct type_info {
    ggml_type type;
    int       inter;  // activation interleave (blocklen) of the selected ARM trait
    kernel_t  gemv, gemm;
    kernel_t  gemm_mmid;  // gemm used by MUL_MAT_ID (nullptr: same as gemm)
};

// The traits ggml_repack_get_optimal_repack_type picks on NEON + dotprod + i8mm (Oryon).
static const type_info TYPES[] = {
    { GGML_TYPE_Q3_K,   8, ggml_gemv_q3_K_8x8_q8_K,   ggml_gemm_q3_K_8x8_q8_K },
    { GGML_TYPE_Q4_0,   8, ggml_gemv_q4_0_4x8_q8_0,   ggml_gemm_q4_0_4x8_q8_0 },
    { GGML_TYPE_Q4_K,   8, ggml_gemv_q4_K_8x8_q8_K,   ggml_gemm_q4_K_8x8_q8_K, ggml_gemm_q4_K_8x8_q8_K_gemv_order },
    { GGML_TYPE_Q5_K,   8, ggml_gemv_q5_K_8x8_q8_K,   ggml_gemm_q5_K_8x8_q8_K, ggml_gemm_q5_K_8x8_q8_K_gemv_order },
    { GGML_TYPE_Q6_K,   8, ggml_gemv_q6_K_8x8_q8_K,   ggml_gemm_q6_K_8x8_q8_K },
    { GGML_TYPE_IQ4_NL, 4, ggml_gemv_iq4_nl_4x4_q8_0, ggml_gemm_iq4_nl_4x4_q8_0 },
    { GGML_TYPE_MXFP4,  4, ggml_gemv_mxfp4_4x4_q8_0,  ggml_gemm_mxfp4_4x4_q8_0 },
    { GGML_TYPE_Q8_0,   8, ggml_gemv_q8_0_4x8_q8_0,   ggml_gemm_q8_0_4x8_q8_0 },
    { GGML_TYPE_Q1_0,   8, ggml_gemv_q1_0_4x8_q8_0,   ggml_gemm_q1_0_4x8_q8_0 },
};

static int g_fail = 0;

#define CHECK(cond, ...)                      \
    do {                                      \
        if (!(cond)) {                        \
            printf("  FAIL: " __VA_ARGS__);   \
            printf("\n");                     \
            g_fail++;                         \
        }                                     \
    } while (0)

static uint64_t fnv1a(const void * p, size_t n, uint64_t h = 1469598103934665603ULL) {
    const uint8_t * b = (const uint8_t *) p;
    for (size_t i = 0; i < n; i++) {
        h = (h ^ b[i]) * 1099511628211ULL;
    }
    return h;
}

struct cmp_result {
    size_t mismatches = 0;
    double max_rel    = 0.0;
};

static cmp_result compare(const float * a, const float * b, size_t n) {
    cmp_result r;
    double     amax = 0;
    for (size_t i = 0; i < n; i++) {
        amax = std::max(amax, (double) std::fabs(a[i]));
    }
    for (size_t i = 0; i < n; i++) {
        if (memcmp(&a[i], &b[i], sizeof(float)) != 0) {
            r.mismatches++;
            r.max_rel = std::max(r.max_rel, std::fabs((double) a[i] - b[i]) / (amax > 0 ? amax : 1));
        }
    }
    return r;
}

static ggml_backend_buffer_type_t g_repack_buft = nullptr;

static void find_repack_buft() {
    ggml_backend_reg_t reg = ggml_backend_cpu_reg();
    ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, 0);
    auto get_extra = (ggml_backend_dev_get_extra_bufts_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_get_extra_bufts");
    for (ggml_backend_buffer_type_t * p = get_extra(dev); p && *p; p++) {
        if (strcmp(ggml_backend_buft_name(*p), "CPU_REPACK") == 0) {
            g_repack_buft = *p;
        }
    }
}

// Weights for n_expert experts of shape [K, N]: a pool of distinct experts, replicated.
static std::vector<uint8_t> make_weights(ggml_type type, int K, int N, int n_expert, std::mt19937 & rng) {
    const int    pool      = std::min(n_expert, 16);
    const size_t row_size  = ggml_row_size(type, K);
    const size_t expert_sz = row_size * N;
    std::vector<float>   f((size_t) pool * N * K);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (auto & v : f) {
        v = nd(rng);
    }
    std::vector<uint8_t> q((size_t) n_expert * expert_sz);
    ggml_quantize_chunk(type, f.data(), q.data(), 0, (int64_t) pool * N, K, nullptr);
    for (int e = pool; e < n_expert; e++) {
        memcpy(q.data() + e * expert_sz, q.data() + (e % pool) * expert_sz, expert_sz);
    }
    return q;
}

// tensor in a buffer of the given type, data uploaded with ggml_backend_tensor_set (repacks for CPU_REPACK)
struct weight_tensor {
    ggml_context *        ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_tensor *         t   = nullptr;
    bool                  repacked = false;

    weight_tensor(ggml_type type, int K, int N, int n_expert, ggml_backend_buffer_type_t buft, const std::vector<uint8_t> & data) {
        ggml_init_params ip = { ggml_tensor_overhead() * 4, nullptr, true };
        ctx = ggml_init(ip);
        t = n_expert > 0 ? ggml_new_tensor_3d(ctx, type, K, N, n_expert) : ggml_new_tensor_2d(ctx, type, K, N);
        buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
        if (buft == g_repack_buft) {
            repacked = t->extra != nullptr;
            if (!repacked) {
                return;  // no trait (type not selected / disabled by env): leave unset
            }
        }
        ggml_backend_tensor_set(t, data.data(), 0, data.size());
    }
    ~weight_tensor() {
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }
};

static std::vector<float> run_mul_mat_id(ggml_backend_t be, ggml_tensor * as, const std::vector<float> & b, int ne10,
                                         int ne11, int n_tokens, const std::vector<int32_t> & ids, int n_used) {
    const size_t mem = 64 * 1024 * 1024 + b.size() * 4 + ids.size() * 4 + (size_t) as->ne[1] * n_used * n_tokens * 4 * 2;
    ggml_init_params ip = { mem, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * tb   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ne10, ne11, n_tokens);
    ggml_tensor * tids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, n_tokens);
    memcpy(tb->data, b.data(), b.size() * 4);
    memcpy(tids->data, ids.data(), ids.size() * 4);
    ggml_tensor * out = ggml_mul_mat_id(ctx, as, tb, tids);
    ggml_cgraph * gf  = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    GGML_ASSERT(ggml_backend_graph_compute(be, gf) == GGML_STATUS_SUCCESS);
    std::vector<float> r(ggml_nelements(out));
    memcpy(r.data(), out->data, r.size() * 4);
    ggml_free(ctx);
    return r;
}

static std::vector<float> run_mul_mat(ggml_backend_t be, ggml_tensor * a, const std::vector<float> & b, int K, int M) {
    ggml_init_params ip = { 64 * 1024 * 1024 + b.size() * 4 + (size_t) a->ne[1] * M * 4, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * tb = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
    memcpy(tb->data, b.data(), b.size() * 4);
    ggml_tensor * out = ggml_mul_mat(ctx, a, tb);
    ggml_cgraph * gf  = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    GGML_ASSERT(ggml_backend_graph_compute(be, gf) == GGML_STATUS_SUCCESS);
    std::vector<float> r(ggml_nelements(out));
    memcpy(r.data(), out->data, r.size() * 4);
    ggml_free(ctx);
    return r;
}

// Routing: hot experts with fixed uneven counts, the rest of each token's 8 slots from random cold experts.
static std::vector<int32_t> make_ids(int n_tokens, int n_used, int n_expert, std::mt19937 & rng, std::vector<int> & counts) {
    static const int hot[8]        = { 3, 17, 42, 99, 128, 200, 254, 255 };
    static const int hot_count[8]  = { 61, 37, 23, 13, 8, 7, 5, 2 };
    std::vector<int32_t> ids((size_t) n_tokens * n_used);
    counts.assign(n_expert, 0);
    std::uniform_int_distribution<int> ud(0, n_expert - 1);
    for (int t = 0; t < n_tokens; t++) {
        std::vector<int> sel;
        for (int j = 0; j < 8 && (int) sel.size() < n_used; j++) {
            if (n_expert == 256 && t < hot_count[j]) {
                sel.push_back(hot[j]);
            }
        }
        while ((int) sel.size() < n_used) {
            int e = ud(rng);
            bool dup = false;
            for (int s : sel) dup |= (s == e);
            for (int h : hot) dup |= (n_expert == 256 && h == e);  // keep hot counts exact
            if (!dup) sel.push_back(e);
        }
        std::shuffle(sel.begin(), sel.end(), rng);
        for (int j = 0; j < n_used; j++) {
            ids[(size_t) t * n_used + j] = sel[j];
            counts[sel[j]]++;
        }
    }
    return ids;
}

static void interleave_rows(ggml_type param, int bl, const void * const rows[4], void * out, int64_t K) {
    if (param == GGML_TYPE_Q8_0) {
        const int64_t nb = K / 32;
        // block_q8_0 = { ggml_half d; int8_t qs[32]; }, block_q8_0x4 = { ggml_half d[4]; int8_t qs[128]; }
        const size_t src_bs = 2 + 32, dst_bs = 8 + 128;
        for (int64_t b = 0; b < nb; b++) {
            uint8_t * o = (uint8_t *) out + b * dst_bs;
            for (int r = 0; r < 4; r++) {
                const uint8_t * s = (const uint8_t *) rows[r] + b * src_bs;
                memcpy(o + 2 * r, s, 2);
                for (int ch = 0; ch < 32 / bl; ch++) {
                    memcpy(o + 8 + ch * 4 * bl + r * bl, s + 2 + ch * bl, bl);
                }
            }
        }
    } else {
        // block_q8_K = { float d; int8_t qs[256]; int16_t bsums[16]; }, block_q8_Kx4 = { float d[4]; int8_t qs[1024]; int16_t bsums[64]; }
        const int64_t nb = K / 256;
        const size_t src_bs = 4 + 256 + 32, dst_bs = 16 + 1024 + 128;
        for (int64_t b = 0; b < nb; b++) {
            uint8_t * o = (uint8_t *) out + b * dst_bs;
            for (int r = 0; r < 4; r++) {
                const uint8_t * s = (const uint8_t *) rows[r] + b * src_bs;
                memcpy(o + 4 * r, s, 4);
                for (int ch = 0; ch < 256 / bl; ch++) {
                    memcpy(o + 16 + ch * 4 * bl + r * bl, s + 4 + ch * bl, bl);
                }
                for (int g = 0; g < 16; g++) {
                    memcpy(o + 16 + 1024 + 2 * ((g / 4) * 16 + r * 4 + (g % 4)), s + 4 + 256 + 2 * g, 2);
                }
            }
        }
    }
}

// Kernel level: gemm on 4 interleaved rows vs gemv per row, on repacked expert 0 (and an offset column range).
static void test_kernels(const type_info & ti, ggml_tensor * rep, int K, int N, std::mt19937 & rng) {
    const ggml_type param = ggml_get_type_traits_cpu(ti.type)->vec_dot_type;
    const auto *    pt    = ggml_get_type_traits_cpu(param);
    const size_t    rs    = ggml_row_size(param, K);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    std::vector<float> x(4 * (size_t) K);
    for (auto & v : x) v = nd(rng);
    std::vector<uint8_t> q(4 * rs), tile(4 * rs), tile_ref(4 * rs);
    const void * rows[4];
    for (int r = 0; r < 4; r++) {
        pt->from_float(x.data() + (size_t) r * K, q.data() + r * rs, K);
        rows[r] = q.data() + r * rs;
    }
    interleave_rows(param, ti.inter, rows, tile.data(), K);

    // interleave of from_float rows vs ggml_quantize_mat_* on the same floats (informational for Q8_0:
    // MUL_MAT uses quantize_mat, MUL_MAT_ID uses from_float + interleave)
    if (param == GGML_TYPE_Q8_K) {
        ggml_quantize_mat_q8_K_4x8(x.data(), tile_ref.data(), K);
        // the bitwise check covers d, qs and bsums; only 4x8 is used on ARM
        if (ti.inter == 8) {
            CHECK(memcmp(tile.data(), tile_ref.data(), tile.size()) == 0, "%s: interleave(from_float) != quantize_mat_q8_K_4x8", ggml_type_name(ti.type));
        }
    } else {
        if (ti.inter == 8) ggml_quantize_mat_q8_0_4x8(x.data(), tile_ref.data(), K);
        else               ggml_quantize_mat_q8_0_4x4(x.data(), tile_ref.data(), K);
        const bool same = memcmp(tile.data(), tile_ref.data(), tile.size()) == 0;
        printf("  interleave(from_float q8_0) %s quantize_mat_q8_0_4x%d\n", same ? "==" : "!=", ti.inter);
    }

    // full column range and an offset range (columns 8..N-8), like a thread's slice
    for (int pass = 0; pass < 2; pass++) {
        const int c0 = pass == 0 ? 0 : 8;
        const int nc = pass == 0 ? N : N - 16;
        const char * w = (const char *) rep->data + (size_t) c0 * rep->nb[1];
        std::vector<float> sg(4 * (size_t) nc), sv(4 * (size_t) nc);
        for (int r = 0; r < 4; r++) {
            ti.gemv(K, sv.data() + (size_t) r * nc, nc, w, rows[r], 1, nc);
        }
        if (ti.gemm_mmid) {
            ti.gemm(K, sg.data(), nc, w, tile.data(), 4, nc);
            cmp_result c = compare(sv.data(), sg.data(), sv.size());
            printf("  kernel MUL_MAT gemm vs gemv (cols %d..%d): %zu/%zu mismatches, max_rel %.3g (information only)\n", c0, c0 + nc,
                   c.mismatches, sv.size(), c.max_rel);
        }
        std::fill(sg.begin(), sg.end(), -1.0f);
        (ti.gemm_mmid ? ti.gemm_mmid : ti.gemm)(K, sg.data(), nc, w, tile.data(), 4, nc);
        cmp_result c = compare(sv.data(), sg.data(), sv.size());
        printf("  kernel MUL_MAT_ID gemm vs gemv (cols %d..%d): %zu/%zu mismatches, max_rel %.3g\n", c0, c0 + nc, c.mismatches, sv.size(), c.max_rel);
        CHECK(c.mismatches == 0, "%s MUL_MAT_ID gemm != gemv (cols %d..%d)", ggml_type_name(ti.type), c0, c0 + nc);
    }
}

int main(int argc, char ** argv) {
    const char * only = argc > 1 ? argv[1] : nullptr;
    printf("GGML_CPU_MOE_GEMM=%s GGML_CPU_REPACK_Q3K=%s\n", getenv("GGML_CPU_MOE_GEMM") ? getenv("GGML_CPU_MOE_GEMM") : "(unset)",
           getenv("GGML_CPU_REPACK_Q3K") ? getenv("GGML_CPU_REPACK_Q3K") : "(unset)");
    ggml_cpu_init();
    find_repack_buft();
    GGML_ASSERT(g_repack_buft);
    ggml_backend_t be = ggml_backend_cpu_init();
    const char * threads_env  = getenv("Q3K_TEST_THREADS");
    const int    test_threads = threads_env ? atoi(threads_env) : 1;
    GGML_ASSERT(test_threads >= 1 && test_threads <= 64);
    ggml_backend_cpu_set_n_threads(be, test_threads);
    printf("graph_threads=%d\n", test_threads);

    std::mt19937 rng(1234);

    for (const type_info & ti : TYPES) {
        if (only && strcasecmp(only, ggml_type_name(ti.type)) != 0) continue;
        printf("== %s\n", ggml_type_name(ti.type));
        const int n_expert = 256, n_used = 8;
        struct shape { const char * name; int K, N, ne11; };
        const shape shapes[] = { { "gate 2048->512", 2048, 512, 1 }, { "down 512->2048", 512, 2048, 8 } };
        for (const shape & sh : shapes) {
            if (ti.type == GGML_TYPE_Q1_0 && sh.K % 128 != 0) continue;
            std::vector<uint8_t> w = make_weights(ti.type, sh.K, sh.N, n_expert, rng);
            weight_tensor ref(ti.type, sh.K, sh.N, n_expert, ggml_backend_cpu_buffer_type(), w);
            weight_tensor rep(ti.type, sh.K, sh.N, n_expert, g_repack_buft, w);
            printf(" %s: repacked=%s\n", sh.name, rep.repacked ? "yes" : "NO (no repack trait selected)");
            if (!rep.repacked) continue;

            if (sh.ne11 == 1) {
                test_kernels(ti, rep.t, sh.K, sh.N, rng);
            }

            for (int n_tokens : { 1, 3, 61, 256 }) {
                std::vector<int> counts;
                std::vector<int32_t> ids = make_ids(n_tokens, n_used, n_expert, rng, counts);
                std::normal_distribution<float> nd(0.0f, 1.0f);
                std::vector<float> b((size_t) sh.K * sh.ne11 * n_tokens);
                for (auto & v : b) v = nd(rng);
                std::vector<float> o_ref = run_mul_mat_id(be, ref.t, b, sh.K, sh.ne11, n_tokens, ids, n_used);
                std::vector<float> o_rep = run_mul_mat_id(be, rep.t, b, sh.K, sh.ne11, n_tokens, ids, n_used);
                cmp_result c = compare(o_ref.data(), o_rep.data(), o_ref.size());
                int ge4 = 0, rem = 0;
                for (int e = 0; e < n_expert; e++) { ge4 += counts[e] >= 4; rem += counts[e] % 4 != 0; }
                printf("  mul_mat_id tokens=%2d (experts with >=4 rows: %d, with 1-3 row remainder: %d): repack vs vec_dot %zu/%zu mismatches, max_rel %.3g, hash %016" PRIx64 "\n",
                       n_tokens, ge4, rem, c.mismatches, o_ref.size(), c.max_rel, fnv1a(o_rep.data(), o_rep.size() * 4));
                if (ti.type == GGML_TYPE_Q3_K) {
                    CHECK(c.mismatches == 0, "Q3_K mul_mat_id repack != vec_dot (tokens=%d)", n_tokens);
                }
            }
        }

        if (ti.type == GGML_TYPE_Q3_K) {
            const int K = 2048, N = 512;
            std::vector<uint8_t> w = make_weights(ti.type, K, N, 1, rng);
            weight_tensor ref(ti.type, K, N, 0, ggml_backend_cpu_buffer_type(), w);
            weight_tensor rep(ti.type, K, N, 0, g_repack_buft, w);
            if (rep.repacked) {
                for (int M : { 1, 3, 7, 37 }) {
                    std::normal_distribution<float> nd(0.0f, 1.0f);
                    std::vector<float> b((size_t) K * M);
                    for (auto & v : b) v = nd(rng);
                    std::vector<float> o_ref = run_mul_mat(be, ref.t, b, K, M);
                    std::vector<float> o_rep = run_mul_mat(be, rep.t, b, K, M);
                    cmp_result c = compare(o_ref.data(), o_rep.data(), o_ref.size());
                    printf("  mul_mat 2-D M=%2d: repack vs vec_dot %zu/%zu mismatches, max_rel %.3g, hash %016" PRIx64 "\n", M, c.mismatches,
                           o_ref.size(), c.max_rel, fnv1a(o_rep.data(), o_rep.size() * 4));
                    CHECK(c.mismatches == 0, "Q3_K mul_mat repack != vec_dot (M=%d)", M);
                }
            }
        }
    }

    ggml_backend_free(be);
    printf("%s (%d failures)\n", g_fail ? "FAILED" : "ALL PASSED", g_fail);
    return g_fail ? 1 : 0;
}
