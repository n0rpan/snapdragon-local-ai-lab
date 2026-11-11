// Q3_K 8-row interleaved repack + NEON dotprod GEMV + NEON i8mm GEMM prototype.
//
// Standalone, header-only. Target: AArch64 with dotprod and i8mm (Snapdragon X Elite / Oryon).
// Compile: gcc -O2 -march=armv8.6-a+i8mm+dotprod
//
// Layout and kernel design: see README.md in this directory.
// Upstream reference copies (dequant, q8_K quantization, vec_dot) live in upstream_ref.h.

#pragma once

#if !defined(__aarch64__) || !defined(__ARM_NEON) || !defined(__ARM_FEATURE_DOTPROD) || !defined(__ARM_FEATURE_MATMUL_INT8)
#error "build with -march=armv8.6-a+i8mm+dotprod on AArch64"
#endif

#include <arm_neon.h>

// header-only: not every includer uses every helper
#pragma GCC diagnostic ignored "-Wunused-function"
#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

// ---------------------------------------------------------------------------------------------
// Minimal ggml shims (same definitions as ggml-common.h / ggml-cpu-impl.h at 954de24)
// ---------------------------------------------------------------------------------------------
#define QK_K          256
#define K_SCALE_SIZE  12
#define GGML_RESTRICT restrict
#define UNUSED(x)     (void) (x)
#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif
#ifndef MAX
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#endif

typedef uint16_t ggml_half;

typedef uint8x16x2_t ggml_uint8x16x2_t;
typedef uint8x16x4_t ggml_uint8x16x4_t;
typedef int8x16x4_t  ggml_int8x16x4_t;
#define ggml_vld1q_u8_x2       vld1q_u8_x2
#define ggml_vld1q_s8_x4       vld1q_s8_x4
#define ggml_vdotq_s32(a, b, c) vdotq_s32(a, b, c)

static inline float fp16_to_fp32(ggml_half h) {
    __fp16 f;
    memcpy(&f, &h, sizeof(f));
    return (float) f;
}

static inline ggml_half fp32_to_fp16(float x) {
    __fp16    f = (__fp16) x;
    ggml_half h;
    memcpy(&h, &f, sizeof(h));
    return h;
}

// ggml-common.h:315-321
typedef struct {
    uint8_t   hmask[QK_K / 8];  // quants - high bit
    uint8_t   qs[QK_K / 4];     // quants - low 2 bits
    uint8_t   scales[12];       // scales, quantized with 6 bits
    ggml_half d;                // super-block scale
} block_q3_K;
_Static_assert(sizeof(block_q3_K) == sizeof(ggml_half) + QK_K / 4 + QK_K / 8 + 12, "wrong q3_K block size/padding");

// ggml-common.h:371-375
typedef struct {
    float   d;               // delta
    int8_t  qs[QK_K];        // quants
    int16_t bsums[QK_K / 16];  // sum of quants in groups of 16
} block_q8_K;

// ggml-cpu/repack.h:100-104
typedef struct {
    float   d[4];
    int8_t  qs[QK_K * 4];
    int16_t bsums[QK_K / 4];
} block_q8_Kx4;

#include "upstream_ref.h"

// ---------------------------------------------------------------------------------------------
// New: block_q3_Kx8
// ---------------------------------------------------------------------------------------------
//
// Eight block_q3_K (8 consecutive weight rows, same k-range of 256) become one block_q3_Kx8.
// The size is exactly 8 * sizeof(block_q3_K) = 880 bytes, which the ggml repack buffer requires
// (it allocates ggml_nbytes(tensor) and repacks in place). It is a lossless re-encoding:
//
//   d[8]       fp16 super-block scales, row order 0..7 (unchanged values)
//   scales[96] the 128 raw 6-bit sub-block scales (16 per row), re-packed for vector decode:
//              idx i = g*8 + c  (g = sub-block 0..15, c = row 0..7), raw in 0..63
//              scales[i]      (i <  64) = (raw[i] & 15) | (raw[i+64] & 15) << 4
//              scales[64 + i] (i <  32) = sum_{q=0..3} (raw[i+32q] >> 4) << 2q
//   qs[768]    the 2048 signed 3-bit weights v = q2 - (hbit ? 0 : 4) in [-4, 3], as 16 "triples".
//              Triple (t, cp), t = k/64 (0..3), cp = column pair (0..3), at qs + ((t*4 + cp)*3)*16,
//              is three 16-byte vectors X, Y, Z. Lane L (0..15) belongs to row c = 2cp + L/8 and
//              k-offset kk = L%8. Field f_i = v(c, 64t + 8i + kk) & 7 for i = 0..7 is stored as
//                X = f0<<5 | f1<<2 | (f6 & 3)
//                Y = f2<<5 | f3<<2 | (f6>>2 & 1) | (f7 & 1)<<1
//                Z = f4<<5 | f5<<2 | (f7>>1 & 3)
//              so that 15 NEON ops produce eight int8x16 SDOT/SMMLA operands already in [-4, 3]
//              (no -4 bias or bsums correction needed). Operand i holds rows (2cp, 2cp+1) x 8
//              consecutive k, which is exactly the SMMLA 2x8 operand shape and the 8x8 SDOT
//              shape used by upstream q4_K/q6_K 8x8 kernels.
typedef struct {
    ggml_half d[8];
    uint8_t   scales[96];
    uint8_t   qs[768];
} block_q3_Kx8;
_Static_assert(sizeof(block_q3_Kx8) == 8 * sizeof(block_q3_K), "block_q3_Kx8 must be size-neutral");

// Raw 6-bit scales (0..63) of a block_q3_K, in sub-block order (same unpack as dequantize_row_q3_K).
static inline void q3k_raw_scales(const block_q3_K * b, uint8_t raw[16]) {
    const uint32_t kmask1 = 0x03030303;
    const uint32_t kmask2 = 0x0f0f0f0f;
    uint32_t       aux[4];
    memcpy(aux, b->scales, 12);
    uint32_t tmp = aux[2];
    aux[2]       = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3]       = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0]       = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    aux[1]       = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
    memcpy(raw, aux, 16);
}

// Signed 3-bit weight of element e (0..255) of a block_q3_K (same semantics as dequantize_row_q3_K).
static inline int q3k_value(const block_q3_K * b, int e) {
    const int n = e / 128, j = (e % 128) / 32, l = e % 32;
    const int q = (b->qs[n * 32 + l] >> (2 * j)) & 3;
    const int h = (b->hmask[l] >> (4 * n + j)) & 1;
    return q - (h ? 0 : 4);
}

// Decode all 256 weights of a block_q3_K as 3-bit two's complement fields (v & 7), in k order.
// Same loop shape as ggml_vec_dot_q3_K_q8_K_generic so the compiler vectorizes it.
static inline void q3k_fields(const block_q3_K * b, uint8_t f[QK_K]) {
    const uint8_t * q3 = b->qs;
    uint8_t *       a  = f;
    uint8_t         m  = 1;
    for (int n = 0; n < QK_K; n += 128) {
        for (int shift = 0; shift < 8; shift += 2) {
            for (int l = 0; l < 32; ++l) {
                a[l] = (uint8_t) ((((q3[l] >> shift) & 3) - ((b->hmask[l] & m) ? 0 : 4)) & 7);
            }
            a += 32;
            m <<= 1;
        }
        q3 += 32;
    }
}

static void make_block_q3_Kx8(const block_q3_K in[8], block_q3_Kx8 * out) {
    uint8_t raw[128];
    uint8_t fv[8][QK_K];
    for (int c = 0; c < 8; c++) {
        q3k_fields(&in[c], fv[c]);
    }
    for (int c = 0; c < 8; c++) {
        out->d[c] = in[c].d;
        uint8_t r[16];
        q3k_raw_scales(&in[c], r);
        for (int g = 0; g < 16; g++) {
            raw[g * 8 + c] = r[g];
        }
    }
    for (int i = 0; i < 64; i++) {
        out->scales[i] = (uint8_t) ((raw[i] & 15) | ((raw[i + 64] & 15) << 4));
    }
    for (int i = 0; i < 32; i++) {
        out->scales[64 + i] = (uint8_t) ((raw[i] >> 4) | ((raw[i + 32] >> 4) << 2) | ((raw[i + 64] >> 4) << 4) |
                                         ((raw[i + 96] >> 4) << 6));
    }
    for (int t = 0; t < 4; t++) {
        for (int cp = 0; cp < 4; cp++) {
            uint8_t * X = out->qs + ((t * 4 + cp) * 3) * 16;
            uint8_t * Y = X + 16;
            uint8_t * Z = X + 32;
            for (int L = 0; L < 16; L++) {
                const int c  = 2 * cp + L / 8;
                const int kk = L % 8;
                int       f[8];
                for (int i = 0; i < 8; i++) {
                    f[i] = fv[c][64 * t + 8 * i + kk];
                }
                X[L] = (uint8_t) (f[0] << 5 | f[1] << 2 | (f[6] & 3));
                Y[L] = (uint8_t) (f[2] << 5 | f[3] << 2 | ((f[6] >> 2) & 1) | (f[7] & 1) << 1);
                Z[L] = (uint8_t) (f[4] << 5 | f[5] << 2 | ((f[7] >> 1) & 3));
            }
        }
    }
}

// Repack a row-major Q3_K matrix (nrow x ncol, nrow % 8 == 0, ncol % 256 == 0).
// Mirrors repack_q6_K_to_q6_K_8_bl (ggml-cpu/repack.cpp:3761): x-block-major, then k-block.
static int repack_q3_K_to_q3_K_8_bl(block_q3_Kx8 * dst, const block_q3_K * src, int nrow, int ncol) {
    const int nrows_interleaved = 8;
    const int nblocks           = ncol / QK_K;
    if (nrow % nrows_interleaved != 0 || ncol % QK_K != 0) {
        return -1;
    }
    block_q3_K tmp[8];
    for (int b = 0; b < nrow; b += nrows_interleaved) {
        for (int x = 0; x < nblocks; x++) {
            for (int i = 0; i < nrows_interleaved; i++) {
                tmp[i] = src[x + i * nblocks];
            }
            make_block_q3_Kx8(tmp, dst++);
        }
        src += nrows_interleaved * nblocks;
    }
    return 0;
}

// Scalar accessors for the repacked layout (used by the generic kernels and the lossless test).
static inline int q3x8_raw_scale(const block_q3_Kx8 * b, int c, int g) {
    const int i  = g * 8 + c;
    const int lo = (i < 64) ? (b->scales[i] & 15) : (b->scales[i - 64] >> 4);
    const int hi = (b->scales[64 + (i % 32)] >> (2 * (i / 32))) & 3;
    return lo | hi << 4;
}

static inline int q3x8_sext3(int f) {
    return f - ((f & 4) << 1);
}

static inline int q3x8_value(const block_q3_Kx8 * b, int c, int k) {
    const int       t = k / 64, i = (k % 64) / 8, kk = k % 8, cp = c / 2;
    const int       L = (c % 2) * 8 + kk;
    const uint8_t * p = b->qs + ((t * 4 + cp) * 3) * 16;
    const int       X = p[L], Y = p[16 + L], Z = p[32 + L];
    int             f;
    switch (i) {
        case 0: f = X >> 5; break;
        case 1: f = (X >> 2) & 7; break;
        case 2: f = Y >> 5; break;
        case 3: f = (Y >> 2) & 7; break;
        case 4: f = Z >> 5; break;
        case 5: f = (Z >> 2) & 7; break;
        case 6: f = (X & 3) | (Y & 1) << 2; break;
        default: f = ((Y >> 1) & 1) | (Z & 3) << 1; break;
    }
    return q3x8_sext3(f);
}

// Interleave four already-quantized q8_K rows into block_q8_Kx4 (blocklen 8), bit-identical to
// ggml_quantize_mat_q8_K_4x8 on the same floats. Lets MUL_MAT_ID gather routed rows into gemm
// tiles without re-quantizing (see INTEGRATION.md, step 6).
static void q8_K_rows_to_q8_Kx4(const block_q8_K * const rows[4], block_q8_Kx4 * out, int nb) {
    for (int b = 0; b < nb; b++) {
        for (int r = 0; r < 4; r++) {
            out[b].d[r] = rows[r][b].d;
            for (int ch = 0; ch < QK_K / 8; ch++) {
                memcpy(out[b].qs + ch * 32 + r * 8, rows[r][b].qs + ch * 8, 8);
            }
            for (int g = 0; g < QK_K / 16; g++) {
                out[b].bsums[(g / 4) * 16 + r * 4 + (g % 4)] = rows[r][b].bsums[g];
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Generic (scalar) kernels on the repacked layout: ggml_*_q3_K_8x8_q8_K_generic candidates.
// Same signature and semantics as the upstream repack gemv/gemm:
//   gemv: s[0..nc) = W(nc x n) . a, a is one row of block_q8_K
//   gemm: s[r*bs + c] for r in [0, nr), nr % 4 == 0, activations as block_q8_Kx4
// Float rounding mirrors ggml_vec_dot_q3_K_q8_K (NEON): per super-block exact int32 isum,
// then sum += (a.d * w.d) * isum.
// ---------------------------------------------------------------------------------------------
static void ggml_gemv_q3_K_8x8_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx,
                                            const void * GGML_RESTRICT vy, int nr, int nc) {
    const int nb = n / QK_K;
    assert(n % QK_K == 0);
    assert(nc % 8 == 0);
    UNUSED(bs);
    UNUSED(nr);
    const block_q8_K * a = (const block_q8_K *) vy;
    for (int x = 0; x < nc / 8; x++) {
        const block_q3_Kx8 * w = (const block_q3_Kx8 *) vx + x * nb;
        float                sumf[8] = { 0 };
        for (int b = 0; b < nb; b++) {
            for (int c = 0; c < 8; c++) {
                int32_t isum = 0;
                for (int g = 0; g < 16; g++) {
                    int32_t dot = 0;
                    for (int k = g * 16; k < g * 16 + 16; k++) {
                        dot += q3x8_value(&w[b], c, k) * a[b].qs[k];
                    }
                    isum += dot * (q3x8_raw_scale(&w[b], c, g) - 32);
                }
                const float d = a[b].d * fp16_to_fp32(w[b].d[c]);
                sumf[c] += d * isum;
            }
        }
        for (int c = 0; c < 8; c++) {
            s[x * 8 + c] = sumf[c];
        }
    }
}

static void ggml_gemm_q3_K_8x8_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx,
                                            const void * GGML_RESTRICT vy, int nr, int nc) {
    const int nb = n / QK_K;
    assert(n % QK_K == 0);
    assert(nr % 4 == 0);
    assert(nc % 8 == 0);
    for (int y = 0; y < nr / 4; y++) {
        const block_q8_Kx4 * a = (const block_q8_Kx4 *) vy + y * nb;
        for (int x = 0; x < nc / 8; x++) {
            const block_q3_Kx8 * w = (const block_q3_Kx8 *) vx + x * nb;
            float                sumf[4][8] = { { 0 } };
            for (int b = 0; b < nb; b++) {
                for (int r = 0; r < 4; r++) {
                    for (int c = 0; c < 8; c++) {
                        int32_t isum = 0;
                        for (int g = 0; g < 16; g++) {
                            int32_t dot = 0;
                            for (int k = g * 16; k < g * 16 + 16; k++) {
                                dot += q3x8_value(&w[b], c, k) * a[b].qs[(k / 8) * 32 + r * 8 + (k % 8)];
                            }
                            isum += dot * (q3x8_raw_scale(&w[b], c, g) - 32);
                        }
                        const float d = a[b].d[r] * fp16_to_fp32(w[b].d[c]);
                        sumf[r][c] += d * isum;
                    }
                }
            }
            for (int r = 0; r < 4; r++) {
                for (int c = 0; c < 8; c++) {
                    s[(y * 4 + r) * bs + x * 8 + c] = sumf[r][c];
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// NEON helpers
// ---------------------------------------------------------------------------------------------

// Decode the 96-byte scale area into 8 x int8x16 of signed scales (raw - 32).
// S[v] lane m = scale of sub-block g = 2v + m/8, row c = m%8.  ~32 ops per super-block of 8 rows.
static inline void q3x8_decode_scales(const uint8_t * GGML_RESTRICT sc, int8x16_t S[8]) {
    const uint8x16_t m3  = vdupq_n_u8(3);
    const uint8x16_t o32 = vdupq_n_u8(32);
    const uint8x16_t L0 = vld1q_u8(sc), L1 = vld1q_u8(sc + 16), L2 = vld1q_u8(sc + 32), L3 = vld1q_u8(sc + 48);
    const uint8x16_t H0 = vld1q_u8(sc + 64), H1 = vld1q_u8(sc + 80);
    // vsli keeps the low 4 bits of the first operand, so the low nibble needs no mask.
    S[0] = vreinterpretq_s8_u8(vsubq_u8(vsliq_n_u8(L0, vandq_u8(H0, m3), 4), o32));
    S[1] = vreinterpretq_s8_u8(vsubq_u8(vsliq_n_u8(L1, vandq_u8(H1, m3), 4), o32));
    S[2] = vreinterpretq_s8_u8(vsubq_u8(vsliq_n_u8(L2, vandq_u8(vshrq_n_u8(H0, 2), m3), 4), o32));
    S[3] = vreinterpretq_s8_u8(vsubq_u8(vsliq_n_u8(L3, vandq_u8(vshrq_n_u8(H1, 2), m3), 4), o32));
    S[4] = vreinterpretq_s8_u8(vsubq_u8(vsliq_n_u8(vshrq_n_u8(L0, 4), vandq_u8(vshrq_n_u8(H0, 4), m3), 4), o32));
    S[5] = vreinterpretq_s8_u8(vsubq_u8(vsliq_n_u8(vshrq_n_u8(L1, 4), vandq_u8(vshrq_n_u8(H1, 4), m3), 4), o32));
    S[6] = vreinterpretq_s8_u8(vsubq_u8(vsliq_n_u8(vshrq_n_u8(L2, 4), vshrq_n_u8(H0, 6), 4), o32));
    S[7] = vreinterpretq_s8_u8(vsubq_u8(vsliq_n_u8(vshrq_n_u8(L3, 4), vshrq_n_u8(H1, 6), 4), o32));
}

// Decode one triple into eight signed int8x16 operands (15 ALU ops). R[i]: rows (2cp, 2cp+1) x k = 64t + 8i + 0..7.
#define Q3X8_DECODE_TRIPLE(p, R)                                                              \
    do {                                                                                      \
        const int8x16_t X_ = vld1q_s8((const int8_t *) (p));                                  \
        const int8x16_t Y_ = vld1q_s8((const int8_t *) (p) + 16);                             \
        const int8x16_t Z_ = vld1q_s8((const int8_t *) (p) + 32);                             \
        (R)[0]             = vshrq_n_s8(X_, 5);                                               \
        (R)[1]             = vshrq_n_s8(vshlq_n_s8(X_, 3), 5);                                \
        (R)[2]             = vshrq_n_s8(Y_, 5);                                               \
        (R)[3]             = vshrq_n_s8(vshlq_n_s8(Y_, 3), 5);                                \
        (R)[4]             = vshrq_n_s8(Z_, 5);                                               \
        (R)[5]             = vshrq_n_s8(vshlq_n_s8(Z_, 3), 5);                                \
        (R)[6]             = vshrq_n_s8(vsliq_n_s8(vshlq_n_s8(X_, 5), Y_, 7), 5);             \
        (R)[7]             = vshrq_n_s8(vsliq_n_s8(vshlq_n_s8(Y_, 4), Z_, 6), 5);             \
    } while (0)

// ---------------------------------------------------------------------------------------------
// GEMV (decode): 1 activation row, 8 weight rows per tile, SDOT.
// Per sub-block of 16 k and 8 rows: 8 SDOT, ~7.5 decode ops, 2 ADDP, 2 XTN, 2 SMLAL.
// ---------------------------------------------------------------------------------------------
static void ggml_gemv_q3_K_8x8_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx,
                                    const void * GGML_RESTRICT vy, int nr, int nc) {
    const int nb = n / QK_K;
    assert(n % QK_K == 0);
    assert(nc % 8 == 0);
    UNUSED(bs);
    UNUSED(nr);

    const block_q8_K * GGML_RESTRICT a = (const block_q8_K *) vy;

    for (int x = 0; x < nc / 8; x++) {
        const block_q3_Kx8 * GGML_RESTRICT w = (const block_q3_Kx8 *) vx + x * nb;

        float32x4_t accf0 = vdupq_n_f32(0.0f);  // rows 0..3
        float32x4_t accf1 = vdupq_n_f32(0.0f);  // rows 4..7

        for (int b = 0; b < nb; b++) {
            int8x16_t S[8];
            q3x8_decode_scales(w[b].scales, S);

            int32x4_t isum0 = vdupq_n_s32(0);
            int32x4_t isum1 = vdupq_n_s32(0);

            for (int t = 0; t < 4; t++) {
                const int8_t * q8 = a[b].qs + 64 * t;
                int8x16_t      A[8];
                for (int i = 0; i < 8; i++) {
                    A[i] = vreinterpretq_s8_s64(vld1q_dup_s64((const int64_t *) (q8 + 8 * i)));
                }
                // sub-blocks 4t..4t+3, int16 scales, rows 0..7 each
                const int16x8_t sg[4] = {
                    vmovl_s8(vget_low_s8(S[2 * t])),
                    vmovl_high_s8(S[2 * t]),
                    vmovl_s8(vget_low_s8(S[2 * t + 1])),
                    vmovl_high_s8(S[2 * t + 1]),
                };
                const uint8_t * qt = w[b].qs + (t * 4) * 48;

                for (int h = 0; h < 2; h++) {  // column pairs (0,1) -> rows 0..3, (2,3) -> rows 4..7
                    int8x16_t R[8];
                    int32x4_t accA[4], accB[4];

                    Q3X8_DECODE_TRIPLE(qt + (2 * h) * 48, R);
                    for (int gi = 0; gi < 4; gi++) {
                        accA[gi] = vdotq_s32(vdotq_s32(vdupq_n_s32(0), R[2 * gi], A[2 * gi]), R[2 * gi + 1], A[2 * gi + 1]);
                    }
                    Q3X8_DECODE_TRIPLE(qt + (2 * h + 1) * 48, R);
                    for (int gi = 0; gi < 4; gi++) {
                        accB[gi] = vdotq_s32(vdotq_s32(vdupq_n_s32(0), R[2 * gi], A[2 * gi]), R[2 * gi + 1], A[2 * gi + 1]);
                    }
                    // |sub-block dot| <= 16 * 4 * 128 = 8192: narrowing to int16 is exact.
                    for (int gi = 0; gi < 4; gi++) {
                        const int16x4_t p = vmovn_s32(vpaddq_s32(accA[gi], accB[gi]));
                        if (h == 0) {
                            isum0 = vmlal_s16(isum0, p, vget_low_s16(sg[gi]));
                        } else {
                            isum1 = vmlal_s16(isum1, p, vget_high_s16(sg[gi]));
                        }
                    }
                }
            }

            const float32x4_t da  = vdupq_n_f32(a[b].d);
            const float32x4_t dw0 = vcvt_f32_f16(vld1_f16((const __fp16 *) w[b].d));
            const float32x4_t dw1 = vcvt_f32_f16(vld1_f16((const __fp16 *) w[b].d + 4));
            accf0                 = vfmaq_f32(accf0, vcvtq_f32_s32(isum0), vmulq_f32(da, dw0));
            accf1                 = vfmaq_f32(accf1, vcvtq_f32_s32(isum1), vmulq_f32(da, dw1));
        }

        vst1q_f32(s + x * 8, accf0);
        vst1q_f32(s + x * 8 + 4, accf1);
    }
}

// ---------------------------------------------------------------------------------------------
// GEMM (prefill): 4 activation rows (block_q8_Kx4, blocklen 8) x 8 weight rows per tile, SMMLA.
// Per sub-block of 16 k: 16 SMMLA, ~15 decode ops (shared by both row pairs), 8 MLA, 4 LDR(scales).
// Accumulators stay in SMMLA lane order [c0r0, c0r1, c1r0, c1r1] until the final store.
// ---------------------------------------------------------------------------------------------
static void ggml_gemm_q3_K_8x8_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx,
                                    const void * GGML_RESTRICT vy, int nr, int nc) {
    const int nb = n / QK_K;
    assert(n % QK_K == 0);
    assert(nr % 4 == 0);
    assert(nc % 8 == 0);

    for (int y = 0; y < nr / 4; y++) {
        const block_q8_Kx4 * GGML_RESTRICT a = (const block_q8_Kx4 *) vy + y * nb;

        for (int x = 0; x < nc / 8; x++) {
            const block_q3_Kx8 * GGML_RESTRICT w = (const block_q3_Kx8 *) vx + x * nb;

            float32x4_t accf[4][2];
            for (int cp = 0; cp < 4; cp++) {
                accf[cp][0] = vdupq_n_f32(0.0f);
                accf[cp][1] = vdupq_n_f32(0.0f);
            }

            for (int b = 0; b < nb; b++) {
                // Per sub-block, per column pair: int32 [s_c0, s_c0, s_c1, s_c1] (SMMLA lane order).
                int32_t   sc32[16][4][4] __attribute__((aligned(16)));
                int8x16_t S[8];
                q3x8_decode_scales(w[b].scales, S);
                for (int g = 0; g < 16; g++) {
                    const int16x8_t s16 = vmovl_s8((g & 1) ? vget_high_s8(S[g / 2]) : vget_low_s8(S[g / 2]));
                    const int16x8_t z1  = vzip1q_s16(s16, s16);
                    const int16x8_t z2  = vzip2q_s16(s16, s16);
                    vst1q_s32(sc32[g][0], vmovl_s16(vget_low_s16(z1)));
                    vst1q_s32(sc32[g][1], vmovl_high_s16(z1));
                    vst1q_s32(sc32[g][2], vmovl_s16(vget_low_s16(z2)));
                    vst1q_s32(sc32[g][3], vmovl_high_s16(z2));
                }

                int32x4_t ia[4][2];
                for (int cp = 0; cp < 4; cp++) {
                    ia[cp][0] = vdupq_n_s32(0);
                    ia[cp][1] = vdupq_n_s32(0);
                }

                for (int t = 0; t < 4; t++) {
                    for (int cp = 0; cp < 4; cp++) {
                        int8x16_t R[8];
                        Q3X8_DECODE_TRIPLE(w[b].qs + (t * 4 + cp) * 48, R);
                        for (int gi = 0; gi < 4; gi++) {
                            const int8_t *  q8  = a[b].qs + (8 * t + 2 * gi) * 32;
                            const int8x16_t a01_0 = vld1q_s8(q8);
                            const int8x16_t a23_0 = vld1q_s8(q8 + 16);
                            const int8x16_t a01_1 = vld1q_s8(q8 + 32);
                            const int8x16_t a23_1 = vld1q_s8(q8 + 48);
                            const int32x4_t r01 =
                                vmmlaq_s32(vmmlaq_s32(vdupq_n_s32(0), R[2 * gi], a01_0), R[2 * gi + 1], a01_1);
                            const int32x4_t r23 =
                                vmmlaq_s32(vmmlaq_s32(vdupq_n_s32(0), R[2 * gi], a23_0), R[2 * gi + 1], a23_1);
                            const int32x4_t sc = vld1q_s32(sc32[4 * t + gi][cp]);
                            ia[cp][0]          = vmlaq_s32(ia[cp][0], r01, sc);
                            ia[cp][1]          = vmlaq_s32(ia[cp][1], r23, sc);
                        }
                    }
                }

                const float32x4_t dw0 = vcvt_f32_f16(vld1_f16((const __fp16 *) w[b].d));
                const float32x4_t dw1 = vcvt_f32_f16(vld1_f16((const __fp16 *) w[b].d + 4));
                const float32x4_t dwp[4] = { vzip1q_f32(dw0, dw0), vzip2q_f32(dw0, dw0), vzip1q_f32(dw1, dw1),
                                             vzip2q_f32(dw1, dw1) };
                const float32x4_t da     = vld1q_f32(a[b].d);
                const float32x4_t dap[2] = { vcombine_f32(vget_low_f32(da), vget_low_f32(da)),
                                             vcombine_f32(vget_high_f32(da), vget_high_f32(da)) };
                for (int cp = 0; cp < 4; cp++) {
                    for (int rp = 0; rp < 2; rp++) {
                        accf[cp][rp] = vfmaq_f32(accf[cp][rp], vcvtq_f32_s32(ia[cp][rp]), vmulq_f32(dap[rp], dwp[cp]));
                    }
                }
            }

            for (int rp = 0; rp < 2; rp++) {
                float * r0 = s + (y * 4 + 2 * rp) * bs + x * 8;
                float * r1 = r0 + bs;
                vst1q_f32(r0, vuzp1q_f32(accf[0][rp], accf[1][rp]));
                vst1q_f32(r0 + 4, vuzp1q_f32(accf[2][rp], accf[3][rp]));
                vst1q_f32(r1, vuzp2q_f32(accf[0][rp], accf[1][rp]));
                vst1q_f32(r1 + 4, vuzp2q_f32(accf[2][rp], accf[3][rp]));
            }
        }
    }
}
