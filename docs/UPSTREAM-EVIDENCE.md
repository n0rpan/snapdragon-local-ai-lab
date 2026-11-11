# Upstream patch hardware evidence

Measurement fact sheet for the owner; not upstream submission prose. No upstream PR/post or production change.

Hardware: Yoga Slim 7x 83ED, X1E80100 (12 Oryon cores), 32 GB, BIOS NHCN62WW, Linux 7.3.0-40-omarchy, MEMLAT DDR/LLCC active, GCC 16.1.1. MEMLAT-off is deferred because it requires a reboot; this is not a cheap extra arm.

## Method

Independent sources, built separately:
- master: `5f490d7bfe78844e6390c869454c149fd7e003a8`, clean tree. Build 198.0 s; binary identities in numeric receipts.
- moe-gemm: `104efec8f78f660593c7849d060661635e6a3a38`, clean tree. Build 198.5 s; binary identities in numeric receipts.
- q3k-repack: `32ccc2201907e9243db2d304bcf07cb26fcf45b3`, clean tree. Build 198.1 s; binary identities in numeric receipts.

Identical build flags:
```sh
nice -n 19 cmake -S SOURCE -B BUILD -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DGGML_NATIVE=OFF -DGGML_VULKAN=OFF -DGGML_BLAS=OFF -DGGML_OPENMP=ON -DGGML_CPU_REPACK=ON -DGGML_CPU_KLEIDIAI=OFF -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_EXAMPLES=OFF -DGGML_CPU_ARM_ARCH=armv8.6-a+dotprod+i8mm+fp16+bf16+nosve+nosme
nice -n 19 cmake --build BUILD -j 2 --target llama-bench llama-perplexity test-backend-ops test-cpu-repack
```
Master has no test-cpu-repack target. Builds held the lease and used two jobs. Primary native runs unset GGML_*, LLAMA_*, OXE_*, OMP_*, GOMP_* and KMP_*; no extra environment knobs. The initial explicitly PASSIVE pilot is retained separately and excluded from every primary number.

Each native job uses a fresh 28 GiB cgroup, MemoryHigh=infinity, MemorySwapMax=0, no CPU quota, the existing memory/battery guards, and the one inference lease. Models are never simultaneously loaded. No prompt reuse, persistent repack cache or MTP; in-memory CPU repacking is on. Native llama-bench random-token, short-context work is not finished-agent speed. Three repetitions per cell; master is repeated at the end of each model block. Master table values pool its six samples; candidates each have three. Report the opening/closing drift below rather than assuming identical thermals.

Exact native commands per cell (MODEL is the identical local GGUF for both sides; T=8 or 12 and MASK=0xff or 0xfff):
```sh
llama-bench -m MODEL -ngl 0 -dev none -nopo 1 --no-host 1 --repack 1 -lm mmap -fa off -ctk f16 -ctv f16 -b 256 -ub 256 -t T -C MASK --cpu-strict 1 -p 512 -n 128 -r 3 -o json --progress
test-backend-ops test -b CPU -o MUL_MAT,MUL_MAT_ID -p "type_a=(q3_K|q4_K|q5_K|q6_K)"
test-cpu-repack
```

## Per-patch timing

| Model | Threads | Work | Master tok/s | GEMM alone | Delta | Q3_K alone | Delta | Master open / close |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| ornith | 8 | pp512 | 93.88 | 94.73 | +0.9% | 136.16 | +45.0% | 95.78 / 92.44 (-3.5%) |
| ornith | 8 | tg128 | 30.98 | 30.75 | -0.8% | 32.87 | +6.1% | 30.91 / 31.10 (+0.6%) |
| ornith | 12 | pp512 | 121.14 | 120.58 | -0.5% | 165.23 | +36.4% | 121.36 / 120.57 (-0.6%) |
| ornith | 12 | tg128 | 27.27 | 27.41 | +0.5% | 27.93 | +2.4% | 27.22 / 27.31 (+0.3%) |
| qwen | 8 | pp512 | 143.80 | 180.56 | +25.6% | 143.47 | -0.2% | 144.18 / 143.42 (-0.5%) |
| qwen | 8 | tg128 | 26.18 | 25.95 | -0.9% | 26.27 | +0.4% | 26.31 / 26.07 (-0.9%) |
| qwen | 12 | pp512 | 172.33 | 210.52 | +22.2% | 172.09 | -0.1% | 175.64 / 172.18 (-2.0%) |
| qwen | 12 | tg128 | 20.98 | 21.47 | +2.3% | 21.36 | +1.8% | 20.70 / 21.56 (+4.2%) |

Interpretation: GEMM-only prefill improves Qwen by 22–26%; it is roughly flat on Ornith without Q3_K repacking. Q3_K-only improves Ornith prefill by 36–45% and eight-thread decode by 6.1%; Qwen is flat. Small decode differences for GEMM are not intrinsic GEMM wins: the one-token path does not use the new tile, and the 12-thread Qwen master control drift is 4.2%. These isolate existing contributions, not extra installed gains.

## MUL_MAT_ID GEMM alone

See the GEMM column above. This patch applies only to the existing exact repacked types; Ornith Q3_K experts stay on the old path in this single-patch branch. Generic backend tests do not exercise CPU_REPACK; the dedicated repack test checks the actual path.

## Q3_K repack alone

See Q3_K column above. No MoE GEMM patch is present. Qwen is a useful negative control because the selected expert tensor mix is Q4_K/Q5_K/Q6_K.

## Correctness

| Arm | Check | Result |
|---|---|---|
| master | backend-test | 435/435 tests passed |
| moe-gemm | backend-test | 435/435 tests passed |
| moe-gemm | repack-test | PASS (zero batch/single mismatches in repack test) |
| moe-gemm / ornith | 4 x 512-token model KLD | KLD 0; top-1 100% |
| q3k-repack / ornith | 4 x 512-token model KLD | KLD 0; top-1 100% |
| q3k-repack | backend-test | 435/435 tests passed |
| q3k-repack | repack-test | PASS (zero batch/single mismatches in repack test) |
| moe-gemm / qwen | 4 x 512-token model KLD | KLD 0; top-1 100% |
| q3k-repack / qwen | 4 x 512-token model KLD | KLD 0; top-1 100% |

KLD: fresh per-model master reference, synthetic quality-corpus.txt, 2,048 input tokens in four 512-token chunks, 1,020 scored positions. Same b/ub256, f16 KV, FA off, 8 decode/12 batch threads and CPU masks. Master command adds `-f CORPUS --save-all-logits BASE`; patch command adds `--kl-divergence --kl-divergence-base BASE`. Common options:
```sh
-m MODEL -ngl 0 --device none --no-op-offload --no-host --repack -fa off -ctk f16 -ctv f16 -c 512 -b 256 -ub 256 -t 8 --threads-batch 12 --cpu-mask 0xff --cpu-strict 1 --cpu-mask-batch 0xfff --cpu-strict-batch 1 --chunks 4 --no-warmup
```

Model identities: Ornith Compact a7373f724cf3693e8a6a5f8209ae9a06ff8071ae2af60659bdbdf8ff6f6c10df; Qwen3.6 UD-Q4_K_M 0b21525e972670ed59e1812e170b27c26355381f0656ecc4e25617ece7dac58b. All exact argv, binary hashes, samples, thermal/power and memory counters: numeric summary. Private raw receipts (not needed to read the numeric summary): $LAB_ROOT/local-ai-optimization/upstream-evidence-20261110.

## Memory and runtime limits

Peak CPU temperature 98.1 C; minimum host MemAvailable 6.54 GiB. The 28 GiB hard charge limit is configured on every scope; the largest recorded scope peak is 28.00002 GiB (24 KiB of kernel charge batching above the configured value), with no OOM. Process RSS can exceed the scope charge limit because shared clean file pages can be charged elsewhere.

Ornith's sampled anon peak rises from 5.61 GiB on master to 15.25 GiB on Q3_K-only (about +9.65 GiB). Sampled VmHWM is 21.71 versus up to 28.02 GiB; the source mmap and repacked copy overlap at startup. This upstream-only branch does not include our file-backed repack cache/source release. Qwen anon is about 20.22 GiB in all arms; its RSS peaks are about 28.5–28.6 GiB. These separate peaks are not additive.

No OOM or model swap in the recorded jobs; no CFS quota throttling. Per-job telemetry is retained. Scope memory peaks are charge accounting, not process RSS: clean mapped model pages may already be charged elsewhere. Sampled process VmHWM/RssAnon/RssFile are included where collected; do not sum their separate peaks. The Q3_K branch has no production repack sidecar/cache, so its new experts are anonymous memory.

## MUL_MAT_ID remainder padding

Measured below against fresh GEMM-only controls; not selected.

No production profile was promoted. These are single-laptop/one-compiler results, not Apple/x86/SVE coverage or a full upstream CI claim.

[Numeric samples and commands](../benchmarks/results/upstream-single-patches-20261110.json).

## Measured follow-up: remainder padding and combined patches

Same boot and primary flags/environment. Immutable sources:
- mmid-pad: `d45181f832a36caa45b0b7f17b087eabb8f20175`.
- combo: `08e56086bf1cabe91e2aeb04eb91c28f546c0b74`.
- combo-pad: `dc2999bc85415bc08661e04c81f373dba53f99f1`.

Built incrementally in the owned worktree, with a fresh configure per revision and identical flags. Each static executable was frozen by hash; original 104efec reference binaries were restored byte-for-byte. Padding is compared against fresh opening/closing GEMM-only controls. Each candidate has three native samples; each GEMM reference has six. Combined Q3_K+GEMM is measured on Ornith; Qwen’s lack of Q3_K was already checked as a negative control. No production or upstream submission.

| Model | Threads | Work | GEMM ref tok/s | + padding | Delta | Q3_K+GEMM | + padding | Padding on combo |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| ornith | 8 | pp512 | 96.84 | 94.82 | -2.1% | 154.81 | 153.23 | -1.0% |
| ornith | 8 | tg128 | 30.69 | 30.74 | +0.1% | 33.29 | 33.13 | -0.5% |
| ornith | 12 | pp512 | 123.26 | 120.54 | -2.2% | 187.92 | 186.98 | -0.5% |
| ornith | 12 | tg128 | 27.26 | 27.03 | -0.8% | 27.87 | 27.34 | -1.9% |
| qwen | 8 | pp512 | 181.50 | 181.24 | -0.1% | — | — | — |
| qwen | 8 | tg128 | 26.18 | 25.78 | -1.5% | — | — | — |
| qwen | 12 | pp512 | 218.52 | 211.88 | -3.0% | — | — | — |
| qwen | 12 | tg128 | 21.32 | 21.77 | +2.1% | — | — | — |

### Follow-up correctness

| Arm | Model | KLD | Top-1 |
|---|---|---:|---:|
| combo | ornith | 0 | 100% |
| combo-pad | ornith | 0 | 100% |
| mmid-pad | ornith | 0 | 100% |
| mmid-pad | qwen | 0 | 100% |

All three follow-up builds pass the dedicated repack test and 435/435 selected generic CPU operator cases. Model KLD reuses the fresh same-master 4 x 512 references from the first phase, with identical native arguments; 1,020 scored positions per model. The generic CPU tests still do not reach CPU_REPACK. No OOM, model swap or native guard failure. Raw job arguments/hashes/thermal/memory counters are retained in followup-summary.json beside the first-phase receipts.

This port deliberately leaves experts with fewer than four total routed rows on GEMV; Rust also pads those small experts during prefill. Do not transfer a speed claim between the two schedules. Padding is a separate candidate, not assumed beneficial because it is exact. Small decode changes are noise/layout effects: these patches leave one-token decode on GEMV.

**Decision: keep C++ remainder padding out.** Padding alone changes Ornith prefill by about −2.1/−2.2% and Qwen by −0.1/−3.0% at 8/12 threads. On the Ornith combined build it changes prefill by −1.0/−0.5%. It has no measured positive main-path gain. All four follow-up model checks are KLD 0 / top-1 100%; exactness alone is not a speed win. The existing Rust schedule is different and is not reverted based on this C++ experiment.

[Follow-up numeric receipts](../benchmarks/results/upstream-followups-20261110.json).
