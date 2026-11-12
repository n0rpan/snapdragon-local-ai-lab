# Q3_K repack and MoE prefill gemm: applying and qualifying two ggml-cpu patches

Status: **candidate, not qualified.** Codex subsequently built with KleidiAI ON and passed graph tests at1/4/8 threads. `exact` passes the original-Compact4K model gate (reported KLD0,top-1 agreement100%). **Mode1 fails the model gate (KLD0.021678,agreement92.696%); do not use it until the bug is fixed.** Exact-only model ABBA improved pp512 at12 threads119.29→185.03tok/s and tg128 at8 threads28.90→31.16tok/s; protected-task validation is pending. The handoff below records the original patch design and expectations. Both patches build, and the ggml-level correctness tests below pass. Neither patch has been run with a model, timed, or installed. That work belongs to Codex, under the inference lease.

| file | commit (`<llama.cpp checkout>`, branch `claude/q3k-moe-gemm`) | sha256 |
|---|---|---|
| `llama-q3k-repack.patch` | `1c148ab093c0aaa43cefecba2ca091c1283b81c1` | `d3542b5c44fa0da82f8e557ccab41a5eb65ae5c4843c70b5241c7522e73f301c` |
| `llama-moe-repack-gemm.patch` | `0a3a67929be9a15b7bd7b1815358f11ed050a817` | `5e0659404858dd0a1b265635355c4276a5d566b0d10f2d0eaaa3ca4759d4759a` |

Base: llama.cpp `954de248ac6907dd05dc19f0f511ab18f2807a54`. Both patches touch only `ggml/src/ggml-cpu/{repack.h,repack.cpp,arch/arm/repack.cpp,arch-fallback.h}`, so they do not interact with `llama-small-batch-pool.patch`, which touches only `src/llama-context.cpp`. Apply the Q3_K patch first: the MoE patch's context depends on it.

## What each patch does

1. **`llama-q3k-repack.patch` (Q3_K repack).** Adds `block_q3_Kx8`, a lossless 8-row re-encoding of the same size (880 B = 8 × 110 B). It also adds the repack function, generic and NEON gemv (dotprod) and gemm (i8mm, with a dotprod fallback), the type traits and the arch-fallback aliases. Q3_K weights, both 2-D tensors and 3-D `ffn_*_exps`, are selected for `CPU_REPACK` on NEON+dotprod whenever `ne[1] % 8 == 0`. The kernels compute the same int32 super-block sums and the same FMA order as `ggml_vec_dot_q3_K_q8_K`, so their outputs are **bitwise identical** to the current Q3_K path.
2. **`llama-moe-repack-gemm.patch` (MoE gemm).** Upstream `forward_mul_mat_id` calls **only gemv**, one routed row at a time, so MoE prefill on repacked experts never reaches the i8mm gemm. With this patch, an expert with ≥ 4 routed rows has those rows gathered 4 at a time. The rows are already quantized by `from_float`, so nothing is quantized twice. Each group is interleaved into the `q8_0x4`/`q8_Kx4` gemm layout, run through gemm in chunks of 256 columns and scattered back. The 1-3-row remainder still goes through gemv. The path covers every repack trait whose activations are 4- or 8-wide q8_0/q8_K, and it is enabled on AArch64 only. Decode is unaffected: with MTP draft 2 a batch has at most 3 tokens, so every expert has ≤ 3 rows and stays on gemv.

## Environment toggles

Each toggle is read once per process. Set it in the environment of `llama-server`/`llama-bench`/`llama-perplexity`. No rebuild is needed.

| variable | value | effect |
|---|---|---|
| `GGML_CPU_REPACK_Q3K` | unset / anything but `0` (default) | Q3_K weights are repacked (patch 1 active) |
| | `0` | Q3_K weights stay in the plain CPU buffer (mmap) and use `vec_dot`, which is upstream behaviour |
| `GGML_CPU_MOE_GEMM` | unset / `1` (default) | MoE gemm for every supported trait |
| | `exact` | MoE gemm only for traits whose AArch64 gemm is bitwise equal to their gemv: Q3_K 8x8, Q4_0 4x8, Q6_K 8x8, IQ4_NL 4x4, MXFP4 4x4, Q8_0 4x8, Q1_0 4x8 |
| | `0` | gemv only (upstream behaviour) |

With `GGML_CPU_REPACK_Q3K=0 GGML_CPU_MOE_GEMM=0`, the patched binary computes exactly what the unpatched one does.

**Not bitwise:** the Q4_K 8x8 and Q5_K 8x8 gemm kernels round differently from their own gemv kernels (max about 3-4e-7 relative). This is the same difference that upstream MUL_MAT already has between its gemm and gemv. In mode `1`, MoE prefill on Q4_K/Q5_K experts therefore changes logits slightly. Mode `exact` excludes those two kernels.

Effect on the two target models (expert tensor types read from the GGUF headers):

| model | expert tensors | patch 1 | MoE gemm `exact` | MoE gemm `1` |
|---|---|---|---|---|
| Ornith-1.5-35B-A3B APEX MTP Compact | Q3_K ×90 (9.67 GiB), Q4_K ×30 (4.22 GiB), Q8_0 ×3 (0.80 GiB) | Q3_K experts repacked | Q3_K + Q8_0 experts use gemm | Q4_K experts also use gemm |
| Qwen3.6-35B-A3B MTP UD-Q4_K_M | Q4_K ×82, Q5_K ×38, Q6_K ×3 | no effect (no Q3_K) | only the 3 Q6_K tensors | Q4_K + Q5_K use gemm as well |

## Apply and build (separate tree and build directory; production release untouched)

Use the recipe in `docs/lab-notes/RUNTIME-SETUP.md` ("Source and engine identity") and add the two patches after the small-batch patch:

```sh
LAB=<your lab checkout>          # contains patches/
SRC=$(mktemp -d)/llama-source                     # fresh tree, never the pinned ~/.local/share/ai-stack/src/llama.cpp
git clone --no-hardlinks ~/.local/share/ai-stack/src/llama.cpp "$SRC"
git -C "$SRC" checkout 954de248ac6907dd05dc19f0f511ab18f2807a54
git -C "$SRC" apply "$LAB/patches/llama-small-batch-pool.patch"
git -C "$SRC" apply "$LAB/patches/llama-q3k-repack.patch"
git -C "$SRC" apply "$LAB/patches/llama-moe-repack-gemm.patch"
cmake -S "$SRC" -B "$SRC-build" -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=OFF -DGGML_NATIVE=OFF -DGGML_VULKAN=OFF \
  -DGGML_CPU_REPACK=ON -DGGML_CPU_KLEIDIAI=ON -DGGML_OPENMP=ON \
  -DGGML_CPU_ARM_ARCH=armv8.6-a+dotprod+i8mm+fp16+bf16+nosve+nosme
cmake --build "$SRC-build" -j 1 --target llama-server llama-bench llama-perplexity
sha256sum "$SRC-build/bin/llama-server"
```

`git apply --check` of all three patches on `954de24` was verified, and the resulting `ggml/` diff is identical to the branch. Build the unpatched reference (`954de24` + small-batch patch only) the same way, with the same `llama-bench` and `llama-perplexity` targets, so that every A/B uses matching tools. The production `llama-server` (`releases/llama-cpu-1bd3e439f357`) can serve as the unpatched server side.

Install a candidate only as a new `~/.local/share/ai-stack/releases/llama-cpu-<first 12 hex of sha256>/llama-server` after the acceptance below. Do not overwrite the existing release, and record the binary hash in a provenance JSON next to `llama-source-provenance.json`. Add the two patches to `patches/manifest.json` (the entries already present there are marked unqualified).

## Acceptance tests

Use the classic profile settings throughout: 4 decode threads, 10 batch threads, batch/ubatch 256, F16 KV, flash attention off, repack on. Unless noted, the A side is the unpatched build and the B side is the patched build.

### 0. Unit test (seconds, single-threaded, no model)

```sh
LLAMA_SRC="$SRC" LLAMA_BUILD="$SRC-build" \
  <lab checkout>/experiments/q3k-i8mm/run_test_ggml_repack_moe.sh
```

The build directory must contain `ggml/src/libggml-cpu.a` and `libggml-base.a`, which is the case with `BUILD_SHARED_LIBS=OFF`. Expected output: `ALL PASSED` in modes `0`, `1` and `exact`. Mode `0` vs `exact` differs only in the header line. Mode `0` vs `1` differs only in the four `tokens=61` lines of q4_K and q5_K. `GGML_CPU_REPACK_Q3K=0` reports `repacked=NO` for Q3_K. Reference output: `experiments/q3k-i8mm/test_ggml_repack_moe.out.txt`.

### 1. Identical tokens and KLD = 0

The bar here is exact equality. A non-zero KLD in a row that expects 0 means a bug, not rounding. Results do not depend on the thread count (each thread owns whole output columns), but keep the thread counts the same anyway.

```sh
# base logits once per model, unpatched build
A/llama-perplexity -m $MODEL -f $TEXT -c 512 -b 512 -ub 256 -t 10 --kl-divergence-base $OUT/base.kld
# each patched configuration
env $TOGGLES B/llama-perplexity -m $MODEL -f $TEXT -c 512 -b 512 -ub 256 -t 10 \
    --kl-divergence-base $OUT/base.kld --kl-divergence
```

Use `-ub 256` with 512-token chunks so that experts receive ≥ 4 rows and the gemm path runs.

| model | `$TOGGLES` | expected |
|---|---|---|
| Ornith | `GGML_CPU_REPACK_Q3K=0 GGML_CPU_MOE_GEMM=0` | KLD 0, same top-1 = 100% (sanity: the patched binary matches upstream) |
| Ornith | `GGML_CPU_MOE_GEMM=0` | **KLD 0** (Q3_K repack, gemv in decode and prefill) |
| Ornith | `GGML_CPU_MOE_GEMM=exact` | **KLD 0** (Q3_K and Q8_0 experts through gemm) |
| Ornith | default (`1`) | small non-zero KLD allowed (Q4_K experts through gemm, about 1e-7-relative logit noise); report the mean/max KLD and same top-1 |
| Qwen3.6 | `GGML_CPU_MOE_GEMM=0` and `exact` | **KLD 0** |
| Qwen3.6 | default (`1`) | small non-zero KLD allowed (Q4_K and Q5_K through gemm) |

Also check tokens end to end: use the same server settings, a fixed prompt longer than 256 tokens, `temperature 0`, `n_predict 128` and MTP on as in the profile, and compare the A and B completions token for token in modes `0` and `exact`. They must match exactly.

### 2. llama-bench A/B (pp512 / tg128)

Prefill and decode use different thread counts in the profile, so run them separately. Both machines must be in the same power and governor state, and runs must alternate between A and B:

```sh
for cfg in "A" "B GGML_CPU_REPACK_Q3K=0 GGML_CPU_MOE_GEMM=0" "B GGML_CPU_MOE_GEMM=0" "B GGML_CPU_MOE_GEMM=exact" "B"; do
  set -- $cfg; BIN=$1; shift
  env "$@" $BIN/llama-bench -m $MODEL -p 512 -n 0   -b 256 -ub 256 -t 10 -fa 0 -r 5 -o jsonl
  env "$@" $BIN/llama-bench -m $MODEL -p 0   -n 128 -t 4 -fa 0 -r 5 -o jsonl
done
```

- For **Ornith**, A and `Q3K=0 GEMM=0` must match within noise. `GEMM=0` isolates the Q3_K gemv gain on both pp512 and tg128; the prototype README estimates about +11-16% tg if decode stays CPU-bound (untimed). pp512 should improve further in `exact` (Q3_K/Q8_0 experts on i8mm gemm) and again in `1` (Q4_K experts too). tg128 must be the same in `0`, `exact` and `1`.
- For **Qwen3.6**, only the `GGML_CPU_MOE_GEMM` axis matters. tg128 must not change. pp512 is the MoE gemm gain on Q4_K/Q5_K (mode `1`).
- Then confirm with the classic real-server screen, which includes MTP. `llama-bench` does not run MTP.

### 3. Memory: RSS against the 28 GiB cap

Patch 1 moves Ornith's Q3_K experts from the file-backed mmap into anonymous `CPU_REPACK` memory: **+9.67 GiB anonymous**. The page cache for the same bytes becomes reclaimable. Before this patch, the only anonymous weights were the Q4_K/Q8_0 experts (5.0 GiB) plus the repacked 2-D Q4_K/Q6_K/Q8_0 weights. Afterwards, almost the whole 16.2 GiB model is anonymous, and the cgroup can no longer drop expert pages under pressure.

Run llama-server with the production unit limits, once with `GGML_CPU_REPACK_Q3K=0` and once with it unset. Load the model, run a 4K-token prompt plus 256 generated tokens, then record:

- `/proc/<pid>/status`: `VmHWM`, `RssAnon`, `RssFile`
- cgroup `memory.peak`, `memory.stat` (`anon`, `file`) and `memory.events` (`high`, `max`, `oom_kill`)

Pass condition: `memory.peak` stays clearly below the cap, `memory.events` `max` and `oom_kill` are 0, and `high` is not climbing during steady decode. Load is the critical phase. The loader reads each Q3_K tensor through the mmap (page cache) and copies it into the repack buffer, so anon and file-backed memory peak together until reclaim. Qwen3.6 has no Q3_K, so its RSS should not change apart from a few KB of per-thread work buffer.

### 4. Repack load time

Compare the time from llama-server start to `/health` = 200, or the `load time` line in the log, for Ornith with `GGML_CPU_REPACK_Q3K=0` and with it unset, using a cold and a warm page cache. The extra work is a single-threaded repack of 9.67 GiB (11.8 M `block_q3_Kx8`). The packing loop is vectorized (16-byte NEON shifts/ORs), but its throughput has **not** been measured. If load time regresses by more than a few seconds, vectorize `q3k_fields` and the scale gather with NEON intrinsics.

## Risks and limitations

- **Memory residency (main risk).** See test 3: +9.67 GiB anonymous for Ornith, with no reclaimable expert pages. Any expert-streaming (Flash/bigmoe) build must not take these patches, because its raw file-offset reads assume the GGUF layout. The patches belong to the classic engine only, which matches `CROSS-MODEL-OPTIMIZATIONS.md`.
- **Numerics.** Patch 1 and MoE gemm in mode `exact` are bitwise identical on AArch64 NEON with i8mm. Mode `1` is not, because of Q4_K/Q5_K (see above). The bitwise claims were verified with GCC 16.1.1 and `-march=armv8.6-a+dotprod+i8mm+fp16+bf16+nosve+nosme` (`-O3`, gnu++17). A different compiler or a flag change such as `-ffp-contract=off` can change the last bit in the generic fallbacks, but not in the NEON paths, which use explicit FMA intrinsics.
- **Scope of the MoE gemm.** It runs on AArch64 only, because x86/RISC-V/s390 override the 4xN activation quantizers with their own layouts. Traits with interleave 1 (RISC-V 16x1) keep gemv.
- **Work buffer and stack.** `MUL_MAT_ID` work size grows by `64 + n_threads × pad64(4 × q8 row size)`, about 9.3 KB per thread for Ornith's 2048-wide gate/up. The gemm scatter uses a 4 KB stack buffer. If the work buffer is ever too small, the code falls back to gemv instead of asserting.
- **Paths not exercised.** The tests are single-threaded, so the per-thread column offsets (`src0_cur_start > 0`) were covered only at the kernel level (columns 8..504) and by the 256-column chunking, not by multi-threaded graph runs. The dotprod-only gemm fallback was tested by linking a `-march=armv8.2-a+dotprod` build of `arch/arm/repack.cpp` into the test (bitwise PASS). The no-dotprod generic build compiles, but its kernels were not run. Non-ARM builds were not compiled (no cross compiler); the x86/POWER/etc. aliases are the q6_K pattern copied verbatim. KleidiAI was off in the test build. It does not claim Q3_K/Q4_K/Q5_K/Q6_K or MUL_MAT_ID, so the production `KLEIDIAI=ON` build should place weights the same way, but test 1 is what confirms it.
- **Maintenance.** `block_q3_Kx8` is a custom bit layout, not a byte interleave. Carry it as a local patch and re-run test 0 on every llama.cpp bump.

## Test evidence (Claude lane, 2026-11-09)

The `ggml-cpu` target was built once in `<llama.cpp checkout>/build-claude` (`nice -n 19`, `-j1`, 32 s, KleidiAI off, otherwise the RUNTIME-SETUP flags). Then `run_test_ggml_repack_moe.sh` was run, single-threaded and about 10 s per mode:

- **Q3_K, all modes.** Bitwise equal to `vec_dot`, with 0 mismatches in every case:
  - Shapes: Ornith shapes (2048→512 gate with `ne11 = 1`, 512→2048 down with `ne11 = 8`), 256 experts, top-8.
  - Routing: 1, 3 and 61 tokens; hot-expert counts 61/37/23/13/8/7/5/2, so remainders 1/1/3/1/0/3/1/2 plus 1-2-row cold experts.
  - Also covered: 2-D MUL_MAT with M = 1/3/7/37, and kernel-level gemm vs gemv.
- **MoE gemm actually runs.** Under gdb, `ggml_gemm_q3_K_8x8_q8_K` is hit 4 times in mode `0` (kernel test and MUL_MAT) and 470 times in modes `exact` and `1`.
- **Kernel gemm vs gemv, bitwise.** Q3_K, Q4_0 4x8, Q6_K 8x8, IQ4_NL 4x4, MXFP4 4x4, Q8_0 4x8 and Q1_0 4x8 are equal. Q4_K 8x8 and Q5_K 8x8 differ (max_rel 2.7e-7 and 2.8e-7).
- **MUL_MAT_ID output hashes, mode `0` vs `exact`.** Identical for all 9 types. Mode `0` vs `1`: only Q4_K and Q5_K at 61 tokens differ.
- **Activation interleave.** The interleave of `from_float` rows equals `ggml_quantize_mat_q8_K_4x8` and `ggml_quantize_mat_q8_0_4x{4,8}`, bitwise.
- **Q3_K toggle.** `GGML_CPU_REPACK_Q3K=0` makes Q3_K tensors unrepackable (`repacked=NO`), so they use the vec_dot path.
- **Other compile checks.** Both TUs compile without warnings under the build's `-Wall -Wextra -Wpedantic` set for armv8.6-a+i8mm, armv8.2-a+dotprod and armv8-a.
