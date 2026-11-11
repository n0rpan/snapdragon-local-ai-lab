# Integration sketch: Q3_K 8x8 repack into ggml-cpu (not applied, not built)

All line numbers refer to the pinned tree `~/.local/share/ai-stack/src/llama.cpp` at `954de24`. Paths are relative to `ggml/src/ggml-cpu/` unless stated otherwise. The model to copy is **Q6_K 8x8**, which has the same 16 signed sub-block scales, no mins, size-neutral blocks and the same `tensor_traits<…, 8, 8, GGML_TYPE_Q8_K>` instance. Kernel bodies come from `q3k_i8mm.h` in this directory. Porting them is mechanical: replace `fp16_to_fp32` with `GGML_CPU_FP16_TO_FP32`, drop `static` on the exported kernels and use `static_assert` in place of `_Static_assert`.

Nothing changes in llama (`src/`). `make_cpu_buft_list` (`src/llama-model.cpp:1228-1275`) already offers `CPU_REPACK` whenever extra buffer types are enabled, which is the default and is what the classic profile runs with. Weight placement asks `supports_op`, which works on MUL_MAT_ID for `ffn_*_exps` through `ggml_repack_get_optimal_repack_type`. As soon as step 4 returns a trait for Q3_K, the Q3_K expert tensors move into `CPU_REPACK`.

## 1. `repack.h`: block type and declarations

After `block_q6_Kx8`'s static_assert (line 98-99):

```cpp
struct block_q3_Kx8 {
    ggml_half d[8];       // super-block scales of the 8 rows
    uint8_t   scales[96]; // 128 raw 6-bit scales, nibble/2-bit planes (see q3k_i8mm.h)
    uint8_t   qs[768];    // 2048 signed 3-bit weights as 16 X/Y/Z triples
};
static_assert(sizeof(block_q3_Kx8) == 8 * (sizeof(ggml_half) + QK_K / 4 + QK_K / 8 + 12),
              "block_q3_Kx8 must be size-neutral (repack is in place)");
```

Declarations, each placed next to its q6_K counterpart:

- line 160: `void ggml_gemv_q3_K_8x8_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);`
- line 178: `void ggml_gemm_q3_K_8x8_q8_K(…same…);`
- line 216: `void ggml_gemv_q3_K_8x8_q8_K_generic(…same…);`
- line 234: `void ggml_gemm_q3_K_8x8_q8_K_generic(…same…);`

## 2. `repack.cpp`: generic kernels, block maker, repack function, template specializations

| where | add |
|---|---|
| after 1120 (end of `ggml_gemv_q6_K_8x8_q8_K_generic`) | `ggml_gemv_q3_K_8x8_q8_K_generic`, with the body from `q3k_i8mm.h` and the `q3x8_value` / `q3x8_raw_scale` / `q3x8_sext3` helpers made `static inline` above it |
| after 2217 (end of `ggml_gemm_q6_K_8x8_q8_K_generic`) | `ggml_gemm_q3_K_8x8_q8_K_generic` |
| after 3477 (end of `make_block_q6_Kx8`) | `q3k_raw_scales`, `q3k_fields`, `static block_q3_Kx8 make_block_q3_Kx8(block_q3_K * in, unsigned int blck_size_interleave)` that returns by value, the same as the other makers |
| after 3788 (end of `repack_q6_K_to_q6_K_8_bl`) | `repack_q3_K_to_q3_K_8_bl` (below) |
| after 4287 (`repack<block_q6_K, 8, 8>`) | `template <> int repack<block_q3_K, 8, 8>(…) { return repack_q3_K_to_q3_K_8_bl(t, 8, data, data_size); }` |
| after 4397 (`gemv<block_q6_K, 8, 8, …>`) | `template <> void gemv<block_q3_K, 8, 8, GGML_TYPE_Q8_K>(…) { ggml_gemv_q3_K_8x8_q8_K(n, s, bs, vx, vy, nr, nc); }` |
| after 4502 (`gemm<block_q6_K, 8, 8, …>`) | `template <> void gemm<block_q3_K, 8, 8, GGML_TYPE_Q8_K>(…) { ggml_gemm_q3_K_8x8_q8_K(n, s, bs, vx, vy, nr, nc); }` |

```cpp
static int repack_q3_K_to_q3_K_8_bl(struct ggml_tensor * t, int interleave_block, const void * GGML_RESTRICT data, size_t data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_Q3_K);
    GGML_ASSERT(interleave_block == 8);
    constexpr int nrows_interleaved = 8;

    block_q3_Kx8 *     dst = (block_q3_Kx8 *) t->data;
    const block_q3_K * src = (const block_q3_K *) data;
    block_q3_K         dst_tmp[8];
    int                nrow    = ggml_nrows(t);   // ne1*ne2: covers all experts of a 3-D tensor
    int                nblocks = t->ne[0] / QK_K;

    GGML_ASSERT(data_size == nrow * nblocks * sizeof(block_q3_K));

    if (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % QK_K != 0) {
        return -1;
    }
    for (int b = 0; b < nrow; b += nrows_interleaved) {
        for (int64_t x = 0; x < nblocks; x++) {
            for (int i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_q3_Kx8(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    return 0;
}
```

Because `ne[1] % 8 == 0`, 8-row groups never straddle two experts. This is the same assumption Q4_K and Q6_K make for 3-D expert tensors. `nb01`/`nb02` stay valid since sizes are unchanged: `src0_cur = data + cur_a*nb02` and `+ src0_cur_start*nb01` with `src0_cur_start % 8 == 0` land exactly on a `block_q3_Kx8` boundary.

## 3. `arch/arm/repack.cpp`: NEON kernels

- After 1769 (end of `ggml_gemv_q6_K_8x8_q8_K`), add `ggml_gemv_q3_K_8x8_q8_K` together with `q3x8_decode_scales` and `Q3X8_DECODE_TRIPLE`. Guard it with `#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)` and fall through to `ggml_gemv_q3_K_8x8_q8_K_generic`.
- After 5386 (end of `ggml_gemm_q6_K_8x8_q8_K`), add `ggml_gemm_q3_K_8x8_q8_K`. Guard it with `#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_MATMUL_INT8)` and fall through to `ggml_gemm_q3_K_8x8_q8_K_generic`.
- No SVE path is needed: Oryon has no SVE, and the pinned build passes `-U__ARM_FEATURE_SVE`.

## 4. Trait selection: `ggml_repack_get_optimal_repack_type` (`repack.cpp:4930`)

After line 4950 (`q6_K_8x8_q8_K` instance):

```cpp
    // instance for Q3_K
    static const ggml::cpu::repack::tensor_traits<block_q3_K, 8, 8, GGML_TYPE_Q8_K> q3_K_8x8_q8_K;
```

Before line 5066 (`} else if (cur->type == GGML_TYPE_Q6_K) {`):

```cpp
    } else if (cur->type == GGML_TYPE_Q3_K) {
        // gemm needs i8mm; on dotprod-only cores the gemm would fall back to scalar, so do not select
        if (ggml_cpu_has_neon() && ggml_cpu_has_matmul_int8()) {
            if (cur->ne[1] % 8 == 0) {
                return &q3_K_8x8_q8_K;
            }
        }
```

`work_size`, `forward_mul_mat` (which quantizes with `ggml_quantize_mat_t<8, GGML_TYPE_Q8_K>`, so the `q8_Kx4` blocklen 8 matches the gemm) and `forward_mul_mat_id` are templates and need no change. **MUL_MAT_ID support comes for free.** `extra_buffer_type::supports_op` (5213-5224) accepts 3-D src0 once the trait exists, and `forward_mul_mat_id` (4783) dispatches `gemv<block_q3_K, 8, 8, Q8_K>` for every routed row (4913).

## 5. `arch-fallback.h`: generic aliases for other architectures

The ARM section (lines 78-87) gets **no** entry, because ARM implements both kernels. Every other section aliases the generic version, next to its q6_K 8x8 line. The insertion points are after lines 53/71 (GGML_CPU_GENERIC), 100/113 (x86), 144/162 (POWER), 195/213 (LoongArch), 240/257 (RISC-V), 293/310 (s390x) and 350/368 (wasm):

```c
#define ggml_gemv_q3_K_8x8_q8_K_generic ggml_gemv_q3_K_8x8_q8_K
#define ggml_gemm_q3_K_8x8_q8_K_generic ggml_gemm_q3_K_8x8_q8_K
```

`ggml-cpu.c:314-319` (`type_traits_cpu[GGML_TYPE_Q3_K]`) stays unchanged. Repacked tensors never call `vec_dot`, and non-repacked Q3_K tensors such as attention or tensors with `--no-repack` keep using it.

## 6. Optional follow-up: gemm for MUL_MAT_ID (prefill)

Upstream `forward_mul_mat_id` only calls gemv, even when an expert has many routed rows. The sketch below replaces the `for (int ir1 …)` loop at `repack.cpp:4900-4917`. It applies to `PARAM_TYPE == GGML_TYPE_Q8_K && INTER_SIZE == 8` (Q3_K here, and also Q4_K/Q5_K/Q6_K 8x8 if enabled). It reuses the already-quantized q8_K rows in `wdata` through `q8_K_rows_to_q8_Kx4` (bitwise equal to `ggml_quantize_mat_q8_K_4x8`, as verified by the test), so the results stay bitwise identical to gemv.

```cpp
            int ir1 = 0;
            if constexpr (PARAM_TYPE == GGML_TYPE_Q8_K && INTER_SIZE == 8 && std::is_same_v<BLOC_TYPE, block_q3_K>) {
                const int64_t nb   = ne00 / QK_K;
                const int64_t ncol = src0_cur_end - src0_cur_start;
                block_q8_Kx4  tile[16];                 // ne00 <= 4096 -> <= 16 blocks; else fall back to gemv
                float         tmp[4 * 1024];            // 4 x ncol; GGML_ASSERT(ncol <= 1024) or chunk ncol
                if (nb <= 16 && ncol <= 1024) {
                    for (; ir1 + 4 <= nr1; ir1 += 4) {
                        const block_q8_K * rows[4];
                        for (int j = 0; j < 4; j++) {
                            const mmid_row_mapping rm = MMID_MATRIX_ROW(cur_a, ir1 + j);
                            rows[j] = (const block_q8_K *) (wdata + ((rm.i1 % ne11) * nbw1 + rm.i2 * nbw2));
                        }
                        q8_K_rows_to_q8_Kx4(rows, tile, nb);
                        gemm<BLOC_TYPE, INTER_SIZE, NB_COLS, PARAM_TYPE>(ne00, tmp, ncol,
                            src0_cur + src0_cur_start * nb01, tile, 4, ncol);
                        for (int j = 0; j < 4; j++) {
                            const mmid_row_mapping rm = MMID_MATRIX_ROW(cur_a, ir1 + j);
                            memcpy((float *) ((char *) dst->data + rm.i1 * nb1 + rm.i2 * nb2) + src0_cur_start,
                                   tmp + j * ncol, ncol * sizeof(float));
                        }
                    }
                }
            }
            for (; ir1 < nr1; ir1++) { /* existing gemv body unchanged */ }
```

For Ornith, `ne00` is 2048 (gate/up) or 512 (down), so `nb` ≤ 8. `ncol` is `ne01/nth`, at most 2048/1. Size `tmp` on the heap per thread, or chunk `ncol` into pieces of 512, before using this beyond Ornith. The benchmark's `--mode prefill` measures exactly this path (`new_gemm`), so the decision can be data-driven before writing the patch.

## 7. Validation plan once the patch exists (Codex, lease)

1. Build the patched `ggml-cpu` with the same flags as the pinned build (`-O3 -std=gnu11 -mcpu=oryon-1+…+dotprod+i8mm+nosve+nosme`).
2. Run `test-backend-ops -o MUL_MAT_ID` and `-o MUL_MAT` with `type_a=q3_K`, which exercises the CPU repack path through the extra buffer type, if that harness covers `CPU_REPACK` in this tree. Otherwise rely on step 3.
3. **Bitwise logits check.** Use the same prompt, `--temp 0` and the same thread counts, comparing the patched and current builds (or `llama-perplexity --kl-divergence-base`). Decode, and prefill through gemv, should give **KLD = 0 and identical tokens**: the integer sums are exact, the float FMA order is the same as `vec_dot` (shown bitwise in the test) and src1 is quantized with the same `from_float`. A non-zero KLD means a bug, not rounding.
4. Then run the tok/s A/B in the classic profile (4 decode / 10 batch threads), plus an RSS and memory.high check (see the risks in README.md and the report).

## Risks specific to integration

- **Memory residency.** Today the Q3_K expert tensors are file-backed (mmap); this is inferred, because no repack trait means they land in a mapped CPU buffer. After the change they are copied into anonymous `CPU_REPACK` memory: about 90 × 115 MB ≈ 10.4 GB, an estimate from 256 × 512 × 2048 × 3.4375/8 per tensor. Page cache can be reclaimed and re-read under memory.high, but this anonymous memory cannot. Check RSS against the profile's memory.high/max before shipping. Any expert-streaming path (bigmoe/Flash-style raw-offset reads) must not route Ornith Q3_K experts through repack, because those raw file offsets would no longer match the in-memory layout.
- **Load time.** Every Q3_K expert byte is repacked once at load. `make_block_q3_Kx8` decodes each row once and then packs. It is scalar but loop-vectorizable, and it has not been timed. If load time regresses noticeably, vectorize the packing step with NEON shifts and SLI, which are the inverse of the decode.
- **Maintenance.** This is a custom bit layout, not a byte interleave, so the patch has to be carried locally under `patches/` and rebased on every llama.cpp bump. Upstreaming it is possible, but expect review on the format.
