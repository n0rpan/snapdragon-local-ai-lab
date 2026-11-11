# Q3_K 8x8 repack with i8mm/dotprod kernels (prototype)

Status: the standalone prototype passed bitwise checks and48 order-balanced micro-runs. See [RESULTS.md](RESULTS.md). Real-engine patches are now candidates under test; no new engine is installed.
Pinned source: llama.cpp `954de24` (`~/.local/share/ai-stack/src/llama.cpp`, read only).

## Why

In Ornith-1.5-35B-A3B "APEX Compact" (`qwen35moe`, 256 experts, top 8, `n_embd` 2048, expert FFN 512), 90 of the 120 base expert tensors are Q3_K. Profiling put 26.8% of user cycles in Q3_K dot products and 23.1% in Q4_K GEMV.

Upstream already has ARM repack kernels for Q4_0, Q4_K, Q5_K, Q6_K, Q8_0 and IQ4_NL. **There is no Q3_K repack in any form at 954de24**: no `block_q3_Kx8`, no `make_block_q3_K*` and no `q3_K` branch in `ggml_repack_get_optimal_repack_type` (`ggml/src/ggml-cpu/repack.cpp:4930`). Q3_K experts therefore run `ggml_vec_dot_q3_K_q8_K` (`arch/arm/quants.c:2021`), one row per call with `nrows = 1` (`ggml-cpu.c:314-319`). The kernel does a horizontal `ADDV` and a scalar multiply-add for every 16 weights.

## Upstream design notes (what this mirrors)

- **Repack buffer.** `CPU_REPACK` (`repack.cpp:5120-5260`) is an extra buffer type. `init_tensor` attaches a `tensor_traits<BLOC, INTER, NB_COLS, PARAM>` and `set_tensor` repacks in place. The repacked layout must fit in `ggml_nbytes` because `get_alloc_size` is left at its default. For that reason `block_q6_Kx8` is exactly 8 × `block_q6_K`, and so is `block_q3_Kx8` here.
- **MUL_MAT** (`forward_mul_mat`, 4650). It quantizes `src1` into `block_q8_Kx4` (blocklen 8, `ggml_quantize_mat_q8_K_4x8`) in groups of 4 rows. It then calls `gemm` for each 4-row group and `gemv` for the leftover rows (4636-4647).
- **MUL_MAT_ID (MoE) does use repack.** `supports_op` accepts 3-D src0 (5213-5224) and `forward_mul_mat_id` (4783-4920) groups the routed rows by expert. It then calls **only `gemv`**, once per routed row (4913), even during prefill. Upstream MoE prefill on repacked experts therefore never reaches the i8mm gemm.
- **q4_K 8x8** (`arch/arm/repack.cpp:781` gemv, 4015 gemm) and **q6_K 8x8** (1570 gemv, 4984 gemm) use the same scheme: 8 rows interleaved in 8-byte chunks. gemv uses SDOT against a `vld1q_dup_s64` activation chunk, and the lanes are `[c0 k0-3, c0 k4-7, c1 k0-3, c1 k4-7]`, reduced with `ADDP`. gemm uses SMMLA with weights `(c0,c1)×8k` and activations `(r0,r1)×8k`, giving lanes `[c0r0, c0r1, c1r0, c1r1]`. Q6_K is the closest relative of Q3_K (16 signed sub-block scales and no mins), so `block_q3_Kx8` follows its blocking.

## block_q3_Kx8 (880 B = 8 × 110 B, 3.4375 bpw unchanged)

| field | bytes | content |
|---|---|---|
| `d[8]` | 16 | fp16 super-block scales of rows 0..7 (unchanged) |
| `scales[96]` | 96 | 128 raw 6-bit scales, `i = g*8 + c`. Low nibbles: `[i] = raw[i]&15 \| (raw[i+64]&15)<<4`. High 2 bits: `[64+i] = Σq (raw[i+32q]>>4)<<2q`. NEON decode is about 32 ops per super-block |
| `qs[768]` | 768 | 2048 signed 3-bit weights `v = q2 - (hbit?0:4)` in 16 triples (X,Y,Z) of 16-byte vectors |

Triple `(t, cp)` covers k ∈ [64t, 64t+64) for rows 2cp and 2cp+1. Lane L maps to row `2cp + L/8` and k offset `L%8`. Each field is `f_i = v(row, 64t+8i+kk) & 7` and is packed as follows:

```
X = f0<<5 | f1<<2 | f6[1:0]
Y = f2<<5 | f3<<2 | f6[2]    | f7[0]<<1
Z = f4<<5 | f5<<2 | f7[2:1]
```

Fifteen NEON ops (`SSHR #5`, `SHL+SSHR`, and `SHL+SLI+SSHR` for f6/f7) yield eight `int8x16` operands that are already sign-extended to [-4,3]. Each operand is exactly the SDOT 8x8 / SMMLA 2x8 operand for one 8-k chunk of a column pair. Because the values are signed, no `-4` bias or `bsums` correction is needed, unlike the q6_K kernels' `-32`/bsums path. The original layout needs about 4.5 ops per 16 weights (`BIC`, shift, `AND`, shift, `SUB`), because the 2-bit `qs` field and the `hmask` bit live in different bytes at different bit positions. The new layout needs about 1.9 ops per 16 weights.

This is a lossless re-encoding of hmask, qs and scales, not a byte interleave of them. The test checks every weight, every scale and every `d` against the original, and checks that the dequantized floats are bitwise equal to `dequantize_row_q3_K`.

## Kernels

- **`ggml_gemv_q3_K_8x8_q8_K`** (decode; dotprod). For each sub-block of 16 k and 8 rows: 8 SDOT, about 7.5 decode ops, 2 `ADDP`, 2 `XTN` and 2 `SMLAL` (the sub-block dot is at most 8192 in magnitude, so narrowing to int16 is exact). That is about 0.29 instructions per weight, against about 0.62 for upstream `vec_dot`, which also has an `ADDV`, a move to the general-purpose registers and a scalar `MADD` every 16 weights.
- **`ggml_gemm_q3_K_8x8_q8_K`** (prefill; i8mm). For each sub-block, on a 4-token × 8-row tile: 16 SMMLA, decode shared by both row pairs, and 8 `MLA` with a per-super-block table of int32 scales in SMMLA lane order. Accumulators stay in SMMLA lane order and are separated with `UZP1`/`UZP2` only at the store. That is about 0.10 instructions per weight per token, against about 0.62 for `vec_dot`.
- **Rounding.** Every kernel computes the same exact int32 super-block sum as upstream and then does `acc = fma(acc, float(isum), a.d * w.d)`. With the production flags (`-std=gnu11`, where fp-contract defaults to fast) the output is **bitwise identical** to `ggml_vec_dot_q3_K_q8_K`. With `-ffp-contract=off` it differs by at most about 5e-8 relative.
- **Generic scalar versions** (`*_generic`) of both kernels are included as the non-ARM fallback.
- **`q8_K_rows_to_q8_Kx4`** interleaves four already-quantized `q8_K` rows into `q8_Kx4`, bitwise equal to `ggml_quantize_mat_q8_K_4x8`. This is what lets MUL_MAT_ID use the gemm without quantizing again (INTEGRATION.md, step 6).

## Expected gains (estimate, untimed)

- **Bytes.** Q3_K stays at 3.4375 bpw (110 B per 256 weights), against 4.5 bpw for Q4_K. If a kernel is bandwidth-bound, Q3_K moves 24% fewer bytes per weight than Q4_K. Repacking adds no bytes.
- **Compute.** Decode is currently compute-bound in the Q3_K path, which is why it takes 26.8% of cycles. The new gemv issues about 2.1x fewer instructions per weight and removes the per-16-weight `ADDV` → general-register → `MADD` dependency chain. The expected Q3_K kernel speedup is **about 1.6-2.1x at 1 thread**. At 4-8 threads with cold experts, DRAM bandwidth may cap it: about 0.43 B/weight at roughly 20 GB/s per core puts 4 threads near the practical LPDDR5X limit.
- **End to end (Amdahl).** With 26.8% of cycles at s = 1.6-2.1x, 10-14% of CPU cycles are saved, so decode would be **about +11-16% tok/s if it stays CPU-bound**. Prefill on Q3_K experts gets the gemv gain immediately, because upstream MUL_MAT_ID repack only uses gemv. With the optional MUL_MAT_ID gemm (step 6), the Q3_K share of prefill could drop about 4-6x more, since 4 tokens share each weight decode through SMMLA.
- **Compared with Q4_K repack.** Q4_K's nibble unpack takes 1 op per 16 weights, so Q3_K repacked gemv will still cost somewhat more instructions per weight than Q4_K repacked gemv, while moving fewer bytes. Do not expect Q3_K to become cheaper than Q4_K per weight unless the kernel is memory-bound.

## Files

- `q3k_i8mm.h`: types, repack, scalar accessors, generic and NEON gemv/gemm, and the q8_K interleave helper.
- `upstream_ref.h`: verbatim upstream copies (`dequantize_row_q3_K`, `quantize_row_q8_K_ref`, generic and NEON `vec_dot_q3_K_q8_K`, `ggml_quantize_mat_q8_K_4x8_generic`) with file:line provenance.
- `test_q3k_i8mm.c`: correctness test (single thread, about 60 ms).
- `bench_q3k_i8mm.c`: micro-benchmark harness for Codex (compiled to check it builds; never run).
- `INTEGRATION.md`: patch sketch against the pinned tree.

## Test result (Claude lane, 2026-11-09; compiled with `nice -n 19 gcc -O2 -march=armv8.6-a+i8mm+dotprod`, run once on 1 core)

```
block_q3_K=110 B, block_q3_Kx8=880 B (8x=880)
shapes K×N×M: 256×8×4, 768×24×8, 2048×512×4 (gate/up), 512×2048×4 (down), 512×64×4 extreme
repack lossless: 0 value / 0 scale / 0 d / 0 dequant-bit mismatches (all shapes)
q8_K x4 interleave == ggml_quantize_mat_q8_K_4x8: bitwise (all shapes)
new gemv (NEON) vs upstream vec_dot NEON: 100% bitwise (all shapes)
new gemm (i8mm) vs upstream vec_dot NEON: 100% bitwise (all shapes)
generic gemv/gemm: 100% bitwise
vs upstream generic vec_dot: max_rel 6.2e-8 (same as upstream NEON vs generic)
vs float64 dequant reference: max_rel 4.6e-8
ALL PASSED (0 failures)
```

The same result (all bitwise) holds with `-O3 -std=gnu11 -mcpu=oryon-1+...+dotprod+i8mm+nosve+nosme`, which are the flags of the pinned build's `compile_commands.json`.

## Benchmark steps for Codex (under the inference lease only)

```sh
cd <lab checkout>/experiments/q3k-i8mm
OUT=$(mktemp -d)
FLAGS="-O3 -std=gnu11 -mcpu=oryon-1+rng+crc+nopauth+dotprod+i8mm+nosve+nosme"
gcc $FLAGS -o $OUT/test_q3k_i8mm test_q3k_i8mm.c -lm && $OUT/test_q3k_i8mm | tail -1   # must print ALL PASSED
gcc $FLAGS -pthread -o $OUT/bench_q3k_i8mm bench_q3k_i8mm.c -lm

# decode: one MUL_MAT_ID node per token, 8 of 256 experts (cold, about 231 MB of weights per shape)
for s in gate down; do for t in 1 4 8; do
  $OUT/bench_q3k_i8mm --shape $s --mode decode --threads $t --tokens 64 --reps 21 --warmup 3
done; done
# prefill: 512 tokens in one node; old vs gemv (current upstream MUL_MAT_ID) vs gemm (step 6)
for s in gate down; do for t in 1 8 10; do
  $OUT/bench_q3k_i8mm --shape $s --mode prefill --threads $t --tokens 512 --reps 7 --warmup 2
done; done
# optional, hot cache (L2-resident experts): add --experts 8
```

- Use the same CPU pinning and governor as the production classic profile (4 decode threads, 10 batch threads).
- Every run starts with a correctness gate that compares each kernel against `old_vec_dot` on the same work; it aborts if `max_abs_diff > 1e-5 × max|y|`.
- Collect the `RESULT {...}` lines. The `speedup` lines give old/new for median and min.
- **Go/no-go:** if `new_gemv` is at least 1.3x faster than `old_vec_dot` for decode at 4 threads with cold experts on both shapes, proceed to the integration patch in INTEGRATION.md.
