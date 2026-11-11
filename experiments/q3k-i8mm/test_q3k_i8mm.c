// Correctness test for the Q3_K 8x8 repack prototype. Single thread, runs in well under a second.
//   nice -n 19 gcc -O2 -march=armv8.6-a+i8mm+dotprod -o test_q3k_i8mm test_q3k_i8mm.c -lm && ./test_q3k_i8mm
#include "q3k_i8mm.h"

#include <stdio.h>
#include <stdlib.h>

static uint64_t rng_state = 0x9E3779B97F4A7C15ull;

static uint64_t rng(void) {
    uint64_t x = rng_state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    return rng_state = x;
}

static float frand(float lo, float hi) {
    return lo + (hi - lo) * (float) ((rng() >> 11) * (1.0 / 9007199254740992.0));
}

static void random_q3_K(block_q3_K * b) {
    for (int i = 0; i < 32; i++) b->hmask[i] = (uint8_t) rng();
    for (int i = 0; i < 64; i++) b->qs[i] = (uint8_t) rng();
    for (int i = 0; i < 12; i++) b->scales[i] = (uint8_t) rng();
    float d = frand(1e-4f, 2e-2f);
    if (rng() & 1) d = -d;
    b->d = fp32_to_fp16(d);
}

static int failures = 0;

#define CHECK(cond, ...)                         \
    do {                                         \
        if (!(cond)) {                           \
            failures++;                          \
            printf("  FAIL: " __VA_ARGS__);      \
            printf("\n");                        \
        }                                        \
    } while (0)

typedef struct {
    long   n;
    long   bitwise;
    double max_rel;  // |got - ref| / sum_k |w_k * a_k|
} cmp_t;

static void cmp_add(cmp_t * c, float got, float ref, double scale) {
    c->n++;
    if (memcmp(&got, &ref, sizeof(float)) == 0) c->bitwise++;
    double rel = fabs((double) got - (double) ref) / (scale > 0 ? scale : 1e-30);
    if (rel > c->max_rel) c->max_rel = rel;
}

static void cmp_print(const char * name, const cmp_t * c) {
    printf("  %-44s n=%-7ld bitwise=%-7ld (%.1f%%) max_rel=%.3g\n", name, c->n, c->bitwise,
           100.0 * c->bitwise / (c->n ? c->n : 1), c->max_rel);
}

// One shape: weight matrix N x K (N output rows, K inner), M activation rows (M % 4 == 0).
static void run_shape(int K, int N, int M, int extreme) {
    const int nb = K / QK_K;
    printf("shape K=%d N=%d M=%d%s\n", K, N, M, extreme ? " (extreme values)" : "");

    block_q3_K *   W   = malloc(sizeof(block_q3_K) * N * nb);
    block_q3_Kx8 * Wr  = malloc(sizeof(block_q3_Kx8) * N * nb / 8);
    float *        Xf  = malloc(sizeof(float) * M * K);
    block_q8_K *   Aq  = malloc(sizeof(block_q8_K) * M * nb);
    block_q8_Kx4 * Aq4 = malloc(sizeof(block_q8_Kx4) * (M / 4) * nb);
    block_q8_Kx4 * Am4 = malloc(sizeof(block_q8_Kx4) * (M / 4) * nb);
    float *        Wf  = malloc(sizeof(float) * N * K);

    for (int i = 0; i < N * nb; i++) {
        random_q3_K(&W[i]);
        if (extreme) {
            // all weights -4 (hmask 0, qs 0), all raw scales 0 (-> -32): maximal |v * s|
            memset(W[i].hmask, 0, 32);
            memset(W[i].qs, 0, 64);
            memset(W[i].scales, 0, 12);
            W[i].d = fp32_to_fp16(1.0f / 64);
        }
    }
    for (int i = 0; i < M * K; i++) {
        Xf[i] = extreme ? -1.0f : frand(-1.0f, 1.0f) * (rng() % 16 == 0 ? 8.0f : 1.0f);
    }
    for (int r = 0; r < M; r++) {
        quantize_row_q8_K_ref(Xf + (size_t) r * K, Aq + (size_t) r * nb, K);
    }

    // --- 1. repack is size-neutral and lossless ------------------------------------------------
    CHECK(repack_q3_K_to_q3_K_8_bl(Wr, W, N, K) == 0, "repack returned error");
    {
        long bad_v = 0, bad_s = 0, bad_d = 0, bad_deq = 0;
        for (int r = 0; r < N; r++) {
            dequantize_row_q3_K(W + (size_t) r * nb, Wf + (size_t) r * K, K);
        }
        for (int x = 0; x < N / 8; x++) {
            for (int b = 0; b < nb; b++) {
                const block_q3_Kx8 * w = &Wr[x * nb + b];
                for (int c = 0; c < 8; c++) {
                    const block_q3_K * o = &W[(size_t) (x * 8 + c) * nb + b];
                    uint8_t            raw[16];
                    q3k_raw_scales(o, raw);
                    if (w->d[c] != o->d) bad_d++;
                    for (int g = 0; g < 16; g++) {
                        if (q3x8_raw_scale(w, c, g) != raw[g]) bad_s++;
                    }
                    const float d_all = fp16_to_fp32(w->d[c]);
                    for (int k = 0; k < QK_K; k++) {
                        const int v = q3x8_value(w, c, k);
                        if (v != q3k_value(o, k)) bad_v++;
                        // same float expression as dequantize_row_q3_K
                        const float dl = d_all * (q3x8_raw_scale(w, c, k / 16) - 32);
                        const float f  = dl * v;
                        const float ref = Wf[(size_t) (x * 8 + c) * K + b * QK_K + k];
                        if (memcmp(&f, &ref, 4) != 0) bad_deq++;
                    }
                }
            }
        }
        printf("  repack lossless: value mismatches=%ld scale mismatches=%ld d mismatches=%ld dequant bit mismatches=%ld\n",
               bad_v, bad_s, bad_d, bad_deq);
        CHECK(bad_v == 0 && bad_s == 0 && bad_d == 0 && bad_deq == 0, "repack not lossless");
    }

    // --- 2. q8_K row interleave == ggml_quantize_mat_q8_K_4x8 -----------------------------------
    for (int y = 0; y < M / 4; y++) {
        const block_q8_K * rows[4] = { Aq + (size_t) (4 * y + 0) * nb, Aq + (size_t) (4 * y + 1) * nb,
                                       Aq + (size_t) (4 * y + 2) * nb, Aq + (size_t) (4 * y + 3) * nb };
        q8_K_rows_to_q8_Kx4(rows, Aq4 + (size_t) y * nb, nb);
        ggml_quantize_mat_q8_K_4x8_generic(Xf + (size_t) 4 * y * K, Am4 + (size_t) y * nb, K);
    }
    {
        const int same = memcmp(Aq4, Am4, sizeof(block_q8_Kx4) * (M / 4) * nb) == 0;
        printf("  q8_K x4 interleave of quantized rows == ggml_quantize_mat_q8_K_4x8: %s\n", same ? "bitwise" : "DIFFERENT");
        CHECK(same, "q8_K_rows_to_q8_Kx4 differs from quantize_mat");
    }

    // --- 3. kernels ------------------------------------------------------------------------------
    float * out_vd_neon = malloc(sizeof(float) * M * N);
    float * out_vd_gen  = malloc(sizeof(float) * M * N);
    float * out_gemv    = malloc(sizeof(float) * M * N);
    float * out_gemv_g  = malloc(sizeof(float) * M * N);
    float * out_gemm    = malloc(sizeof(float) * M * N);
    float * out_gemm_g  = malloc(sizeof(float) * M * N);
    double * ref        = malloc(sizeof(double) * M * N);
    double * mag        = malloc(sizeof(double) * M * N);

    for (int r = 0; r < M; r++) {
        for (int c = 0; c < N; c++) {
            ggml_vec_dot_q3_K_q8_K_neon(K, &out_vd_neon[r * N + c], 0, W + (size_t) c * nb, 0, Aq + (size_t) r * nb, 0, 1);
            ggml_vec_dot_q3_K_q8_K_generic(K, &out_vd_gen[r * N + c], 0, W + (size_t) c * nb, 0, Aq + (size_t) r * nb, 0, 1);
            // float64 reference on dequantized weights and dequantized activations
            double acc = 0, m = 0;
            for (int b = 0; b < nb; b++) {
                const block_q8_K * q = &Aq[(size_t) r * nb + b];
                for (int k = 0; k < QK_K; k++) {
                    const double p = (double) Wf[(size_t) c * K + b * QK_K + k] * ((double) q->d * q->qs[k]);
                    acc += p;
                    m += fabs(p);
                }
            }
            ref[r * N + c] = acc;
            mag[r * N + c] = m;
        }
        ggml_gemv_q3_K_8x8_q8_K(K, out_gemv + r * N, 0, Wr, Aq + (size_t) r * nb, 1, N);
        ggml_gemv_q3_K_8x8_q8_K_generic(K, out_gemv_g + r * N, 0, Wr, Aq + (size_t) r * nb, 1, N);
    }
    ggml_gemm_q3_K_8x8_q8_K(K, out_gemm, N, Wr, Aq4, M, N);
    ggml_gemm_q3_K_8x8_q8_K_generic(K, out_gemm_g, N, Wr, Aq4, M, N);

    cmp_t c_gemv_vdn = { 0 }, c_gemv_vdg = { 0 }, c_gemvg_vdn = { 0 }, c_gemm_gemv = { 0 }, c_gemmg_gemv = { 0 },
          c_gemm_vdn = { 0 }, c_vdn_vdg = { 0 };
    double max_ref_gemv = 0, max_ref_gemm = 0, max_ref_vdn = 0;
    for (int i = 0; i < M * N; i++) {
        cmp_add(&c_gemv_vdn, out_gemv[i], out_vd_neon[i], mag[i]);
        cmp_add(&c_gemv_vdg, out_gemv[i], out_vd_gen[i], mag[i]);
        cmp_add(&c_gemvg_vdn, out_gemv_g[i], out_vd_neon[i], mag[i]);
        cmp_add(&c_gemm_gemv, out_gemm[i], out_gemv[i], mag[i]);
        cmp_add(&c_gemmg_gemv, out_gemm_g[i], out_gemv[i], mag[i]);
        cmp_add(&c_gemm_vdn, out_gemm[i], out_vd_neon[i], mag[i]);
        cmp_add(&c_vdn_vdg, out_vd_neon[i], out_vd_gen[i], mag[i]);
        const double s = mag[i] > 0 ? mag[i] : 1e-30;
        max_ref_gemv   = fmax(max_ref_gemv, fabs(out_gemv[i] - ref[i]) / s);
        max_ref_gemm   = fmax(max_ref_gemm, fabs(out_gemm[i] - ref[i]) / s);
        max_ref_vdn    = fmax(max_ref_vdn, fabs(out_vd_neon[i] - ref[i]) / s);
    }
    cmp_print("new gemv (NEON) vs upstream vec_dot NEON", &c_gemv_vdn);
    cmp_print("new gemm (i8mm) vs upstream vec_dot NEON", &c_gemm_vdn);
    cmp_print("new gemm (i8mm) vs new gemv", &c_gemm_gemv);
    cmp_print("new gemv generic vs upstream vec_dot NEON", &c_gemvg_vdn);
    cmp_print("new gemm generic vs new gemv", &c_gemmg_gemv);
    cmp_print("new gemv (NEON) vs upstream vec_dot generic", &c_gemv_vdg);
    cmp_print("(baseline) upstream vec_dot NEON vs generic", &c_vdn_vdg);
    printf("  vs float64 dequant reference: gemv max_rel=%.3g gemm max_rel=%.3g upstream-NEON max_rel=%.3g\n",
           max_ref_gemv, max_ref_gemm, max_ref_vdn);

    const double tol = 1e-5;  // relative to sum |w*a|; fp32 accumulation over <= 8 super-blocks
    CHECK(c_gemv_vdn.max_rel <= tol, "gemv vs vec_dot NEON beyond tolerance");
    CHECK(c_gemm_vdn.max_rel <= tol, "gemm vs vec_dot NEON beyond tolerance");
    CHECK(c_gemm_gemv.max_rel <= tol, "gemm vs gemv beyond tolerance");
    CHECK(c_gemvg_vdn.max_rel <= tol, "gemv generic vs vec_dot NEON beyond tolerance");
    CHECK(c_gemmg_gemv.max_rel <= tol, "gemm generic vs gemv beyond tolerance");
    CHECK(c_gemv_vdg.max_rel <= tol, "gemv vs vec_dot generic beyond tolerance");
    CHECK(max_ref_gemv <= tol && max_ref_gemm <= tol, "kernels vs float64 reference beyond tolerance");
    if (extreme) {
        // every product is (-4 * -32) * -127 per element; exact integers must survive the int16 narrowing
        CHECK(c_gemv_vdn.bitwise == c_gemv_vdn.n, "extreme case: gemv not bitwise equal to vec_dot NEON");
        CHECK(c_gemm_vdn.bitwise == c_gemm_vdn.n, "extreme case: gemm not bitwise equal to vec_dot NEON");
    }

    free(W); free(Wr); free(Xf); free(Aq); free(Aq4); free(Am4); free(Wf);
    free(out_vd_neon); free(out_vd_gen); free(out_gemv); free(out_gemv_g); free(out_gemm); free(out_gemm_g);
    free(ref); free(mag);
}

int main(void) {
    printf("block_q3_K=%zu B, block_q3_Kx8=%zu B (8x=%zu), block_q8_K=%zu B, block_q8_Kx4=%zu B\n", sizeof(block_q3_K),
           sizeof(block_q3_Kx8), 8 * sizeof(block_q3_K), sizeof(block_q8_K), sizeof(block_q8_Kx4));

    run_shape(256, 8, 4, 0);
    run_shape(768, 24, 8, 0);
    run_shape(2048, 512, 4, 0);   // Ornith ffn_gate_exps / ffn_up_exps (one expert)
    run_shape(512, 2048, 4, 0);   // Ornith ffn_down_exps (one expert)
    run_shape(512, 64, 4, 1);     // int16 narrowing / overflow edge

    printf("%s (%d failure%s)\n", failures ? "FAILED" : "ALL PASSED", failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
