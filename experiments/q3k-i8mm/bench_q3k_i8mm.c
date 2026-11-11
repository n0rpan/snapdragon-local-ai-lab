// Micro-benchmark: upstream ggml_vec_dot_q3_K_q8_K (NEON) vs repacked Q3_Kx8 gemv/gemm on
// Ornith-1.5-35B-A3B expert shapes. FOR CODEX UNDER THE INFERENCE LEASE ONLY (timing-sensitive).
//
// Build:  gcc -O3 -march=armv8.6-a+i8mm+dotprod -pthread -o bench_q3k_i8mm bench_q3k_i8mm.c -lm
// Run:    see README.md "Benchmark steps for Codex".
//
// Model shapes (qwen35moe, benchmarks/results/ornith-moe-download.json): n_embd 2048, 256 experts,
// top-8, expert FFN 512.  ffn_gate_exps / ffn_up_exps: K=2048 -> N=512 per expert.
// ffn_down_exps: K=512 -> N=2048 per expert.
//
// Work split mirrors ggml MUL_MAT_ID (repack.cpp:4880-4893 / ggml-cpu.c mul_mat_id): per routed
// expert, output rows are split across threads in NB_COLS(=8)-aligned slices, one barrier per op.
//
// Modes
//   decode : T tokens one at a time, each routed to A experts (random, fixed seed). One barrier
//            per token, like one MUL_MAT_ID node per token.
//            old  = per output row ggml_vec_dot_q3_K_q8_K (nrows=1, current Q3_K path)
//            gemv = ggml_gemv_q3_K_8x8_q8_K over the thread's 8-row slices
//   prefill: all T tokens in one MUL_MAT_ID node, grouped by expert.
//            old  = vec_dot per (routed token, row)
//            gemv = gemv per routed token (what upstream repack forward_mul_mat_id does today)
//            gemm = routed tokens of an expert in groups of 4 -> q8_Kx4 interleave + gemm,
//                   remainder via gemv (proposed MUL_MAT_ID gemm extension, INTEGRATION.md step 6)
#define _GNU_SOURCE
#include "q3k_i8mm.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef enum { K_OLD = 0, K_GEMV = 1, K_GEMM = 2 } kernel_t;
static const char * kname[] = { "old_vec_dot", "new_gemv", "new_gemm" };

typedef struct {
    int K, N, E, A, T;
    int nb;
    int nth;
    int prefill;
    block_q3_K *   W;    // [E][N][nb]
    block_q3_Kx8 * Wr;   // [E][N/8][nb]
    block_q8_K *   act;  // [T][nb]
    int *          route;  // [T][A]
    // prefill grouping: per expert, list of token indices
    int * cnt;           // [E]
    int * rows;          // [E][T]
    float * out;         // [T][A][N]
    pthread_barrier_t bar;
    kernel_t          kern;
    int               reps, warmup;
    double *          times;  // [reps]
} bench_t;

typedef struct {
    bench_t * B;
    int       ith;
} worker_t;

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void slice_rows(int N, int ith, int nth, int * r0, int * r1) {
    int s = (int) (((long) ith * N) / nth), e = (int) (((long) (ith + 1) * N) / nth);
    s     = (s % 8) ? s + 8 - (s % 8) : s;
    e     = (e % 8) ? e + 8 - (e % 8) : e;
    if (e > N) e = N;
    *r0 = s;
    *r1 = e < s ? s : e;
}

static void old_rows(const bench_t * B, int e, const block_q8_K * a, float * y, int r0, int r1) {
    const block_q3_K * We = B->W + (size_t) e * B->N * B->nb;
    for (int r = r0; r < r1; r++) {
        ggml_vec_dot_q3_K_q8_K_neon(B->K, &y[r], 0, We + (size_t) r * B->nb, 0, a, 0, 1);
    }
}

static void gemv_rows(const bench_t * B, int e, const block_q8_K * a, float * y, int r0, int r1) {
    if (r1 <= r0) return;
    const block_q3_Kx8 * We = B->Wr + (size_t) e * (B->N / 8) * B->nb;
    ggml_gemv_q3_K_8x8_q8_K(B->K, y + r0, 0, We + (size_t) (r0 / 8) * B->nb, a, 1, r1 - r0);
}

static void run_once(bench_t * B, int ith, block_q8_Kx4 * tile) {
    int r0, r1;
    slice_rows(B->N, ith, B->nth, &r0, &r1);
    const int N = B->N, A = B->A, nb = B->nb;

    if (!B->prefill) {
        for (int t = 0; t < B->T; t++) {
            const block_q8_K * a = B->act + (size_t) t * nb;
            for (int s = 0; s < A; s++) {
                const int e = B->route[t * A + s];
                float *   y = B->out + ((size_t) t * A + s) * N;
                if (B->kern == K_OLD) {
                    old_rows(B, e, a, y, r0, r1);
                } else {
                    gemv_rows(B, e, a, y, r0, r1);
                }
            }
            pthread_barrier_wait(&B->bar);
        }
        return;
    }

    // prefill: one MUL_MAT_ID node, grouped by expert (row id = t*A + s)
    for (int e = 0; e < B->E; e++) {
        const int  c    = B->cnt[e];
        const int * ids = B->rows + (size_t) e * B->T * A;
        int        i    = 0;
        if (B->kern == K_GEMM && r1 > r0) {
            const block_q3_Kx8 * We = B->Wr + (size_t) e * (N / 8) * nb + (size_t) (r0 / 8) * nb;
            for (; i + 4 <= c; i += 4) {
                const block_q8_K * rows4[4];
                for (int j = 0; j < 4; j++) rows4[j] = B->act + (size_t) (ids[i + j] / A) * nb;
                q8_K_rows_to_q8_Kx4(rows4, tile, nb);
                // gemm writes rows of s with stride bs; outputs of the 4 routed rows are not
                // contiguous in ggml either, so use a scratch tile and scatter (cost included).
                float tmp[4 * 2048];
                ggml_gemm_q3_K_8x8_q8_K(B->K, tmp, (size_t) (r1 - r0), We, tile, 4, r1 - r0);
                for (int j = 0; j < 4; j++) {
                    memcpy(B->out + (size_t) ids[i + j] * N + r0, tmp + (size_t) j * (r1 - r0), sizeof(float) * (r1 - r0));
                }
            }
        }
        for (; i < c; i++) {
            const int          id = ids[i];
            const block_q8_K * a  = B->act + (size_t) (id / A) * nb;
            float *            y  = B->out + (size_t) id * N;
            if (B->kern == K_OLD) {
                old_rows(B, e, a, y, r0, r1);
            } else {
                gemv_rows(B, e, a, y, r0, r1);
            }
        }
    }
    pthread_barrier_wait(&B->bar);
}

static void * worker(void * p) {
    worker_t *     w    = p;
    bench_t *      B    = w->B;
    block_q8_Kx4 * tile = aligned_alloc(64, sizeof(block_q8_Kx4) * B->nb + 64);
    for (int it = 0; it < B->warmup + B->reps; it++) {
        pthread_barrier_wait(&B->bar);
        const double t0 = now_s();
        run_once(B, w->ith, tile);
        const double t1 = now_s();  // after the op's final barrier: identical on all threads
        if (w->ith == 0 && it >= B->warmup) B->times[it - B->warmup] = t1 - t0;
    }
    free(tile);
    return NULL;
}

static int cmp_d(const void * a, const void * b) {
    double x = *(const double *) a, y = *(const double *) b;
    return (x > y) - (x < y);
}

static double run_kernel(bench_t * B, kernel_t k, double * tmin) {
    B->kern = k;
    pthread_barrier_init(&B->bar, NULL, B->nth);
    pthread_t th[256];
    worker_t  wk[256];
    for (int i = 0; i < B->nth; i++) {
        wk[i] = (worker_t) { B, i };
        pthread_create(&th[i], NULL, worker, &wk[i]);
    }
    for (int i = 0; i < B->nth; i++) pthread_join(th[i], NULL);
    pthread_barrier_destroy(&B->bar);
    qsort(B->times, B->reps, sizeof(double), cmp_d);
    *tmin = B->times[0];
    return B->times[B->reps / 2];
}

static uint64_t rs = 0x2545F4914F6CDD1Dull;
static uint64_t rnd(void) {
    rs ^= rs << 13;
    rs ^= rs >> 7;
    rs ^= rs << 17;
    return rs;
}

static void usage(void) {
    fprintf(stderr,
            "usage: bench_q3k_i8mm [--shape gate|down] [--mode decode|prefill] [--threads N]\n"
            "                      [--experts E] [--active A] [--tokens T] [--reps R] [--warmup W]\n"
            "defaults: --shape gate --mode decode --threads 1 --experts 256 --active 8 --tokens 32 (decode) / 512 (prefill)\n"
            "          --reps 15 --warmup 3\n");
    exit(2);
}

int main(int argc, char ** argv) {
    bench_t B = { 0 };
    const char * shape = "gate";
    int reverse_order = 0;
    B.nth = 1; B.E = 256; B.A = 8; B.T = -1; B.reps = 15; B.warmup = 3;
    for (int i = 1; i < argc; i++) {
        if (i + 1 >= argc) usage();
        if (!strcmp(argv[i], "--shape")) shape = argv[++i];
        else if (!strcmp(argv[i], "--mode")) B.prefill = !strcmp(argv[++i], "prefill");
        else if (!strcmp(argv[i], "--threads")) B.nth = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--experts")) B.E = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--active")) B.A = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--tokens")) B.T = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--reps")) B.reps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--warmup")) B.warmup = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--reverse-order")) reverse_order = atoi(argv[++i]) != 0;
        else usage();
    }
    if (B.T < 0) B.T = B.prefill ? 512 : 32;
    if (!strcmp(shape, "gate")) { B.K = 2048; B.N = 512; }
    else if (!strcmp(shape, "down")) { B.K = 512; B.N = 2048; }
    else usage();
    if (B.nth < 1 || B.nth > 256 || B.A > B.E || B.reps < 1) usage();
    B.nb = B.K / QK_K;

    const size_t nW = (size_t) B.E * B.N * B.nb;
    B.W     = aligned_alloc(64, sizeof(block_q3_K) * nW + 64);
    B.Wr    = aligned_alloc(64, sizeof(block_q3_Kx8) * nW / 8 + 64);
    B.act   = malloc(sizeof(block_q8_K) * B.T * B.nb);
    B.route = malloc(sizeof(int) * B.T * B.A);
    B.cnt   = calloc(B.E, sizeof(int));
    B.rows  = malloc(sizeof(int) * (size_t) B.E * B.T * B.A);
    B.out   = malloc(sizeof(float) * (size_t) B.T * B.A * B.N);
    B.times = malloc(sizeof(double) * B.reps);
    if (!B.W || !B.Wr || !B.act || !B.route || !B.rows || !B.out) { fprintf(stderr, "alloc failed\n"); return 1; }

    for (size_t i = 0; i < nW; i++) {
        block_q3_K * b = &B.W[i];
        for (int j = 0; j < 32; j++) b->hmask[j] = (uint8_t) rnd();
        for (int j = 0; j < 64; j++) b->qs[j] = (uint8_t) rnd();
        for (int j = 0; j < 12; j++) b->scales[j] = (uint8_t) rnd();
        b->d = fp32_to_fp16((float) ((rnd() % 1000) + 1) * 1e-5f);
    }
    for (int e = 0; e < B.E; e++) {
        repack_q3_K_to_q3_K_8_bl(B.Wr + (size_t) e * (B.N / 8) * B.nb, B.W + (size_t) e * B.N * B.nb, B.N, B.K);
    }
    float * xf = malloc(sizeof(float) * B.K);
    for (int t = 0; t < B.T; t++) {
        for (int k = 0; k < B.K; k++) xf[k] = (float) ((int) (rnd() % 2001) - 1000) * 1e-3f;
        quantize_row_q8_K_ref(xf, B.act + (size_t) t * B.nb, B.K);
        for (int s = 0; s < B.A; s++) {
            int e, dup;
            do {
                e   = (int) (rnd() % B.E);
                dup = 0;
                for (int j = 0; j < s; j++) dup |= B.route[t * B.A + j] == e;
            } while (dup);
            B.route[t * B.A + s]                                   = e;
            B.rows[(size_t) e * B.T * B.A + B.cnt[e]++] = t * B.A + s;
        }
    }
    free(xf);

    // correctness gate: every kernel must agree with old_vec_dot on the same work (max rel 1e-5)
    float * ref = malloc(sizeof(float) * (size_t) B.T * B.A * B.N);
    int     nk  = B.prefill ? 3 : 2;
    int     reps = B.reps, warm = B.warmup;
    B.reps = 1; B.warmup = 0;
    double dummy;
    for (int k = 0; k < nk; k++) {
        memset(B.out, 0, sizeof(float) * (size_t) B.T * B.A * B.N);
        run_kernel(&B, (kernel_t) k, &dummy);
        if (k == 0) {
            memcpy(ref, B.out, sizeof(float) * (size_t) B.T * B.A * B.N);
            continue;
        }
        long   bitwise = 0;
        double maxd = 0, maxr = 0;
        for (size_t i = 0; i < (size_t) B.T * B.A * B.N; i++) {
            bitwise += memcmp(&B.out[i], &ref[i], 4) == 0;
            maxd = fmax(maxd, fabs(B.out[i] - ref[i]));
            maxr = fmax(maxr, fabs(ref[i]));
        }
        printf("check %-12s bitwise %ld/%zu max_abs_diff %.3g (max |y| %.3g)\n", kname[k], bitwise,
               (size_t) B.T * B.A * B.N, maxd, maxr);
        if (maxd > 1e-5 * (maxr > 0 ? maxr : 1)) { fprintf(stderr, "correctness gate failed\n"); return 1; }
    }
    free(ref);
    B.reps = reps; B.warmup = warm;

    const double wbytes_expert = (double) B.N * B.nb * sizeof(block_q3_K);
    const double routed        = (double) B.T * B.A;
    double       med[3] = { 0 }, mn[3] = { 0 };
    printf("shape=%s K=%d N=%d experts=%d active=%d tokens=%d mode=%s threads=%d reps=%d warmup=%d\n", shape, B.K, B.N,
           B.E, B.A, B.T, B.prefill ? "prefill" : "decode", B.nth, B.reps, B.warmup);
    for (int ki = 0; ki < nk; ki++) {
        const int k = reverse_order ? nk - 1 - ki : ki;
        med[k] = run_kernel(&B, (kernel_t) k, &mn[k]);
        // decode: every routed expert-matrix is streamed once per token.
        // prefill: each distinct used expert is streamed once per node (ideal), report both.
        printf("%-12s median %9.3f ms  min %9.3f ms  %8.2f us/routed-matvec  %7.2f GB/s(weights, per routed matvec)\n",
               kname[k], med[k] * 1e3, mn[k] * 1e3, med[k] * 1e6 / routed, wbytes_expert * routed / med[k] / 1e9);
    }
    printf("speedup new_gemv vs old: %.3fx (median)  %.3fx (min)\n", med[0] / med[1], mn[0] / mn[1]);
    if (nk == 3) printf("speedup new_gemm vs old: %.3fx (median)  %.3fx (min)\n", med[0] / med[2], mn[0] / mn[2]);
    printf("RESULT {\"shape\":\"%s\",\"mode\":\"%s\",\"threads\":%d,\"experts\":%d,\"active\":%d,\"tokens\":%d,"
           "\"old_ms\":%.4f,\"gemv_ms\":%.4f,\"gemm_ms\":%.4f,\"old_min_ms\":%.4f,\"gemv_min_ms\":%.4f,\"gemm_min_ms\":%.4f}\n",
           shape, B.prefill ? "prefill" : "decode", B.nth, B.E, B.A, B.T, med[0] * 1e3, med[1] * 1e3,
           nk == 3 ? med[2] * 1e3 : 0.0, mn[0] * 1e3, mn[1] * 1e3, nk == 3 ? mn[2] * 1e3 : 0.0);
    return 0;
}
