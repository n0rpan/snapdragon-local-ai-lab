# Results

Every number on one page, with how well it is measured. Machine, models and conditions are in the [README](../README.md); how the engines work is in [ARCHITECTURE.md](ARCHITECTURE.md); how to rerun is in [REPRODUCE.md](REPRODUCE.md).

State: 2026-11-11. Earlier llama.cpp-era ladders (MEMLAT, threads, our llama.cpp patches, the first Rust milestones) are kept in [SPEEDUP-ATTRIBUTION.md](SPEEDUP-ATTRIBUTION.md) and summarised in section 4.

## 1. How to read this page

**Evidence class** (every row has one):

| Tag | Meaning |
|---|---|
| **[M id]** | Measured on the laptop, with a receipt. `id` is the run ticket (e.g. `063`, `G3′`, `QC-CTX2`); its public export is `benchmarks/results/public/<id>.json` (′ → `-prime`) for every README row and the QC-VL, QC-7, QC-LA, W4-J, W5-G and F-IMG-ATTN-1 rows; older ids (e.g. `008`, `036`) are in `benchmarks/results/runq*.json`. Unless the row says otherwise: warm runs, first run dropped, arms interleaved. |
| **[D]** | Derived by arithmetic from [M] numbers (tok/s = 1000 ÷ ms/token, ratios between two receipts, sums). Only as good as its inputs; the inputs are named. |
| **[E]** | Estimate: a model, a simulation, a projection or a third-party figure. Not a result. |
| **[S]** | A single run or a single pair, or a run known to be perturbed. Direction only. Used for older rows. |

**Quality class** (declared before the run, per the [quality policy](#22-quality-gates)):

- **lossless**: same maths, same order; identical greedy tokens or bit-identical logits/state.
- **reorder**: same maths, different floating-point order (batch width, summation order). Must be within the measured noise of the reference.
- **lossy**: quantisation or approximation. Needs a noise-level metric plus a real task check, and ships only as an opt-in profile.

"Installed" means it runs in the daily setup on the test laptop. "Candidate" means measured, not installed.

## 2. Method

### 2.1 Timing

- **Interleaved arms.** A/B comparisons run A B B A (or B A A B) in one block on the same binary build, after a discarded warm-up request. This laptop throttles about 25% within ~30 s of all-core load, per cluster (cluster 2 drops to ~2.5 GHz at ~95 °C while cluster 0 holds 3.4 GHz) and `cpufreq` does not show it; run order alone once moved a result by 15%. Sequential blocks are never pooled into a comparison.
- **The 3% rule.** A speed decision needs an effect of at least 3%. Smaller lossless effects go to a "small-gains stack" that is timed together as one arm; they are not credited one by one.
- **Decode after prefill.** A prefill-side change must also time the decode right after it: one cache-ordering change cut prefill reads by 21% and cost decode 4.4% (section 6).
- **Release-grade numbers** (README headlines) need ≥3 warm runs per arm with the first dropped, or a paired A B B A block; n is stated in each row here.
- **Memory numbers say what they are:** process peak (VmHWM), cgroup charge (includes some page cache) or host `MemAvailable` minimum.

### 2.2 Quality gates

From the project's quality policy:

1. **Gates are measured, never guessed.** Before a gate exists, measure the reference's own noise (1-ulp perturbation, same-binary repeat, cross-build). Gate = 2× that noise (≈ −6 dB for images) unless a stricter bar has a stated user-visible reason.
2. **Classes as in section 1.** Reorder-class changes: mean KLD no larger than 2× llama.cpp's own batched-vs-single noise on the same model (measured: Flash 5.2–5.8e-4, Ornith 0.027). Top-1 agreement is reported only with ≥2K positions; on a few hundred positions a near-tie flip is noise.
3. **Bit identity is a bug detector, not a gate.** When output diverges, classify first: rounding noise (near-tie, noise-level KLD) or a bug (state, rollback, position). Only bugs block.
4. **Production safety stays strict:** a protected OpenCode coding task (13 independent checks), a post-install regression through the installed service, a memory guard, and a one-command rollback written before the install.
5. **Opt-in fast profiles:** a lossy win that fails the default gate may ship as an opt-in profile with its measured cost shown.

## 3. Current results

### 3.1 Flash (Qwen3.8-Flash-Next, UD-IQ3_XXS, 83.6 GB, experts streamed from NVMe)

**Decode**

| Configuration | ms/token | tok/s [D] | Conditions | Quality | State | Evidence |
|---|---:|---:|---|---|---|---|
| BigMoeOnEdge (C++), production 12 GiB expert cache | 356.49 (range 355.72–360.61) | 2.8 | 3 warm 64-token requests, median of request wall ÷ output tokens | reference | replaced | [M G3] |
| Rust Flash engine, 16 GiB | 84.90 (84.47–85.75) | 11.8 | same request-wall metric, 3 retained warm requests | reorder: teacher-forced mean KLD 6.5e-4 vs exact path; 12 top-1 flips in 382 positions, all near-ties (max gap 0.31 nats) | installed 01:40, superseded | [M G3′, W4-G] |
| Rust Flash engine, 18 GiB | 82.55 | 12.1 | same metric; 18 GiB fails the host memory floor beside a running OpenCode harness | same | blocked by memory | [M G3′] |
| Rust, 14 GiB, prefill order 0 (plain) | 85.36 | 11.7 | decode after a 75-token prompt, 4 × 64 tokens, first dropped, B A A B | lossless vs order 2 | rolled forward | [M 063] |
| **Rust, 14 GiB, prefill order 2 (installed)** | **81.13** | **12.3** | same block; all 16 output texts identical | lossless | **installed 06:57** | [M 063] |
| Ratio, old engine → Rust | 4.2× | — | 356.49 ÷ 84.90; **cache size differs (12 vs 16 GiB)**, so not engine-only | — | — | [D G3, G3′] |
| Rust at 20 GiB | ~72–77 (expected) | — | not admissible on this desktop yet: needs 30.1 GiB free of 30.7 | — | ticket open | [E 064] |

The 14 GiB rows (063) and the 16/18 GiB rows (G3′) use different protocols and engine builds. Do not read a cache-size effect from them.

**Prefill**

| What | Result | Conditions | Quality | Evidence |
|---|---|---|---|---|
| Compute stack (dense GEMM, expert tiles, attention tiles) vs the previous Rust prefill | 30.28 → 45.32 tok/s pass 1 (+49.7%); 29.03 → 45.23 pass 2 | 2,048 tokens, 18 GiB, 11 compute threads + IO on core 11, six arms A C D D C A | lossless (state identical) | [M G4] |
| Later build, same shape | 62.3–66.4 tok/s cold, 64.8–67.0 repeat | 18 GiB, mirrored C O P K K P O C, 2 arms per setting | lossless (saved-state SHA identical in all 8 arms) | [M G4′] |
| Read-ahead on a repeated prompt | +12.9% on pass 2 in G4; **no gain** (0.99×) in G4′ on the later build | — | lossless | [M G4, G4′]: off by default |
| Prefill order 2 (residents first, then misses, each by last use) | repeat-pass expert reads 36.63 → 23.05 GiB (−37.06%); **no wall-time gain established** | 2,048 tokens, 14 GiB, 2 reps | lossless (states identical) | [M 063] |
| BigMoeOnEdge prefill | 13.5 tok/s at 156 tokens | different length; no matched-length comparison exists | — | [M, older] |

**Memory and task (installed 14 GiB configuration)**

| What | Value | Evidence |
|---|---|---|
| Protected OpenCode task | 13/13 checks in 182.33 s | [M 063] |
| Model cgroup peak | 22.28 GiB | [M 063] |
| Host `MemAvailable` minimum during the task | 3.64 GiB | [M 063] |
| OOM, model swap, guard stops | none | [M 063] |
| 16 GiB: cgroup peak / host minimum | 24.29 / 2.14 GiB | [M G3′] |
| 18 GiB beside OpenCode | host fell to 0.95 GiB, guard stopped the task | [M G3′] |
| Load-time admission need, 14 / 16 / 18 / 20 GiB tiers | 24.1 / 26.1 / 28.1 / 30.1 GiB free (cache + measured overhead + 0.8 GiB harness growth + 1 GiB floor) | [D F118] |

### 3.2 Ornith-1.5-35B-A3B (default) and Qwen3.6-35B-A3B on `oxe`

**Decode: MTP depth 1 with verify lanes vs plain decode**, installed binary, one server with per-request depth override, A B B A, 256 tokens per arm, one discarded warm-up per context, 8 decode threads (cores 0–7), 32K allocation. Token ids identical between arms at every context (lossless).

| Model | Context | Plain ms/token | MTP d1 ms/token | Time saved | tok/s [D] | Evidence |
|---|---:|---:|---:|---:|---|---|
| Ornith | 4,096 | 22.416 | 17.602 | 21.5% | 44.6 → 56.8 | [M QC-CTX2] |
| Ornith | 8,180 | 24.203 | 17.920 | 26.0% | 41.3 → 55.8 | [M QC-CTX2] |
| Ornith | 15,980 | 27.354 | 22.610 | 17.3% | 36.6 → 44.2 | [M QC-CTX2] |
| Ornith | 32,000 | 32.925 | 29.015 | 11.9% | 30.4 → 34.5 | [M QC-CTX2] |
| Qwen3.6 | 2,048 | 30.183 | 24.413 | 19.1% | 33.1 → 41.0 | [M QC-CTX2-Q] |
| Qwen3.6 | 8,180 | 32.610 | 25.073 | 23.1% | 30.7 → 39.9 | [M QC-CTX2-Q] |
| Qwen3.6 | 15,980 | 35.245 | 28.859 | 18.1% | 28.4 → 34.7 | [M QC-CTX2-Q] |

Contexts above 8K repeat a synthetic 8K token corpus: a matched decode screen, not a quality claim. No Qwen point at 32K was measured.

**Why MTP depth 1 now pays: verify-lane cost** (cost of checking 2 tokens in one pass, in plain decode steps, Ornith): 1.735× → 1.389× with the verify-lane kernels [M QC-VL, 2 arms each]. The second token now costs 0.39 of a step; Ornith's head lands about 1.8 tokens per verify.

**Prefill, 2,048 new tokens, 12 threads, native bench medians**

| Model | Prefix | mask 13 / group 1 | mask 1021 / group 2 | mask 1021 / group 4 | Quality | Evidence |
|---|---:|---:|---:|---:|---|---|
| Ornith | 0 | 198.13 tok/s | 276.77 | 283.58 | lossless: prefill logits and state bit-identical at 2,048 and 8,180 tokens | [M G1] |
| Ornith | 6,144 | 154.61 | 200.25 | — | same | [M G1] |
| Qwen3.6 | 0 | 177.71 | 261.64 (+47.2%) | — | state and first token identical across 8 arms | [M G2] |

**Whole requests and tasks (installed builds)**

| What | Before → after | Conditions | Evidence |
|---|---|---|---|
| Ornith, fixed 2K OpenCode request | 13.267 → 10.164 s (−23.4%) | previous installed build (MTP cut off above 2K) vs candidate 2; 3 per arm, medians | [M G0′] |
| Ornith, removing the 2K MTP cutoff | 10.261 → 9.819 s (−4.3%) | same request | [M G0′] |
| Qwen3.6, fixed 2K request | 12.115 → 8.361 s (−31.0%) | old Rust build without MTP vs installed; 3 per arm; identical token hashes | [M G2′] |
| Qwen3.6, protected task | 13/13 in 44.45 s, peak RSS 22.66 GiB, swap 0 | functional, not a timed pair | [M G2′] |
| Ornith, protected task | 13/13 (two runs, 57.71 s and 35.45 s) | functional, not a timed pair | [M G0′] |
| patched llama.cpp → Rust server, coding task | Ornith 77.70 → 60.42 s; Qwen 75.31 → 67.42 s | 3 per arm, medians; earlier Rust build | [M, RUST-SERVER-QUALIFICATION.md] |
| Rust → Rust with tiled attention | Ornith 60.06 → 52.13 s; Qwen 65.53 → 57.91 s | 3 per arm, medians | [M, RUST-SERVER-UPGRADE.md] |

### 3.3 Sessions: not recomputing

| What | Result | Quality | Evidence |
|---|---|---|---|
| Persistent prefix store (system prompt + tool schemas from disk) | first prompt 30.6 → 1.2 s on both MoEs (5,376 tokens restored, 168 new) | lossless | [S n=1 per model] |
| Parked sessions (conversation moved to RAM when a side request displaces it) | 32K-token session: 817 MiB, resumed in 42 ms, 1 new prompt token | lossless (state byte-identical) | [M 008] |
| Turn checkpoints, 8 instead of 4 | new prompt tokens over a recorded 42-request session 45,936 → 37,765 (−17.8%) | lossless | [M, deterministic count] |
| Session cache across idle unload | 2,304 of 2,304 tokens reused after reload; written in 183 ms, loaded in 15 ms | lossless | [M 038, G0′, G2′] |
| Cancel inside prefill | retry resumes 512 tokens instead of re-running 12,000 (73.9 s) | — | [S 039] |

Session wall-time gains on the current stack are **not** claimed: the last clean replay pair has not been run (the earlier one was perturbed by other jobs).

### 3.4 Images

| What | Result | Evidence |
|---|---|---|
| Installed: patched stable-diffusion.cpp, Z-Image Turbo Q8_0, 1024 px, CPU repack off → on | 775.5 → 488.1 s load-to-PNG | [S one pair] |
| Rust `oxi` port (FLUX.2 klein DiT), f32 reference vs stable-diffusion.cpp | parity image 35.58 dB vs the port's own 1-ulp noise floor 33.81 dB: **port OK** (within noise) | [M W5-G R2, R3] |
| `oxi` int8 fast path (int8 linears + int8 attention), 512 px, seed 42 | 22.75 dB vs a gate of 27.81 dB (2× own noise): **FAIL** | [M W4-J, W5-G R3] |
| `oxi` fast path with **f32 attention** (int8 linears, chunks 32/32), 512 px | seeds 42 / 43 / 44: 31.47 / 32.08 / 34.08 dB vs the 27.81 dB gate: **PASS** on all three | [M F-IMG-ATTN-1]; no timing claim yet |
| Fast DiT block kernels, synthetic | 4.1–5.3× per block vs the reference path | [M synthetic, model-free] |

The f32-attention fast path was then timed end to end through Studio at 512 px **[M IMG-SHIP-1]**: stable-diffusion.cpp 77.85 s → `oxi` 51.53 s per cold request (−33.8%; A B B A, mean of 2 per arm), 45.94 s with a cached prompt; quality 31.5 / 32.1 / 34.1 dB against the 27.8 dB gate, no fallback, OOM or swap. Studio routes 512 × 512 to `oxi` with automatic fallback; 1024 px and other sizes stay on sd.cpp (unqualified). Getting int8 attention through the gate, and 1024 px calibration, is [help-wanted item 2](HELP-WANTED.md#2-image-engine-int8-attention-and-chunk-256-calibration).

## 4. Earlier ladder (summary)

From [SPEEDUP-ATTRIBUTION.md](SPEEDUP-ATTRIBUTION.md); those rows keep their own evidence tags.

| Step | Before → after | Evidence |
|---|---|---|
| MEMLAT memory-frequency fix: CPU streaming read | 48.88 → 115.85 GB/s | [M n=3] |
| MEMLAT, Ornith server decode (llama.cpp, same binary) | 12.73 → 17.18 tok/s, outputs byte-identical | [M n=3] |
| 8 packed decode + 12 prefill threads, Ornith 16K | 16.54 → 30.12 tok/s | [M n=9 requests per arm]; three knobs changed together |
| Unmodified upstream llama.cpp `5f490d7`, tg128 / pp512 (12 threads) | Ornith 30.98 / 121.14; Qwen 26.18 / 172.33 tok/s | [M n=6] |
| Our Q3_K i8mm repack, Ornith pp512 | 121.14 → 165.23 tok/s, KLD 0 | [M n=3 vs 6] |
| Our routed-expert GEMM, Qwen pp512 | 172.33 → 210.52 tok/s, KLD 0 | [M n=3 vs 6] |
| Tiled attention, 2K on a 6K prefix (Rust) | Ornith 126.23 → 164.00; Qwen 126.32 → 166.57 tok/s | [M n=3] |
| Flash cache policy LFU + per-layer quotas vs Clock | −5.5% (20 GiB), −6.2% (18 GiB) token time | [M n=2 per arm] |
| Flash prefill: 11 compute threads + own IO core vs 12 sharing a core with IO | 15.3 → 23.0 tok/s | [M n=2]; two changes together |

There is still **no single matched "stock upstream → today" run**. The ladder is a chain of separate controlled steps measured on different days; multiplying them is not a measurement.

## 5. Compared with Apple Silicon

**No matched measurement exists.** The rules a Mac row must meet: same model file (SHA256) and prompt token ids; same day, ≥3 warm runs per side; a stock baseline on each machine (pinned upstream llama.cpp) plus the Mac's usual best stack, named and pinned; decode and prefill separately; context, threads, power and thermal state stated; receipt and command in the repository. [`tools/mac_reproduce.py`](../tools/mac_reproduce.py) does this; it has not yet run on a Mac.

Third-party Flash-Next figures we have seen [E: community posts, other quants, engines and prompts, not verified by us], listed so the gap is not hidden:

| Machine | Stack | Decode tok/s | Prefill tok/s |
|---|---|---:|---:|
| M1 Max 64 GB | llama.cpp, 3.84 bpw | 17.6 | 182 (pp512) |
| M5 Max 64 GB | vendor 85 GB quant | 36.0 | 518 (pp512) |
| M4 Max 128 GB | llama.cpp, UD-Q3_K_XL | 27.3 | — |
| M4 Max 128 GB | MLX, without / with MTP | 45.0 / 90.1 | — |
| M5 Pro 64 GB | SSD-streamed experts, without / with a draft head | 18 / 27.6 | — |

Our reading, unmeasured [E]: Max/Ultra chips with 64 GB+ hold Flash mostly or fully in RAM and have several times our 108–116 GB/s, so they win Flash decode and prefill clearly. Where we expect to be competitive is 35B-class MoE decode against base/Pro chips and warm agent sessions where exact state reuse removes most prefill. That stays a hypothesis until matched rows exist.

## 6. What did not work

Measured or proven dead ends. Each is re-opened only with new evidence against its reason.

| Idea | Result | Why | Evidence |
|---|---|---|---|
| **Flash MTP** (depth 1) | no gain: 114.7 plain vs 113.9–120.6 ms/token with MTP | 1.92 tokens per verify, but a 2-row verify costs 1.77× a plain step because the second row needs mostly *different* experts (decode is expert compute + IO bound). The code (~1,100 lines) is **not in the release `main`**; it is kept under the engine tag `archive/flash-mtp-w4c` | [M QC-7] |
| **Flash lookahead (expert prefetch) at 14 GiB** | 84.43 → 87.05 ms/token (slower); with fused phases 88.84 | At 18 GiB on a chat workload it saved 13.2% (chat) and 3.3% (code) of decode time [M QC-LA, 2 per arm]. At the installed 14 GiB on the short qualification prompt it lost; likely the pool is too small for speculative slots, and the workload differs. Router recall is ~63% | [M 062 block 2, QC-LA] |
| **Flash prefill "residents first" order** | −21…−48% prefill reads but **+4.4% decode** after a short prefill | The op's wave protection leaves only the residents evictable, so it evicts the hot experts decode needs next. Order 2 (by last use) keeps the read saving without the decode cost | [M 062, replay] |
| **int8 KV cache** (K-only, V-only, both) | fails the gate at 4K | Mean KLD at or over the headroom bound and top-1 <95%; int8 P·V was 1.5–2× slower than f16 | [M QC-4b/4c/4d] |
| **Adreno GPU for LLM prefill/decode** | 8.9–48.6 GMAC/s vs ~1,500 on the 12 CPU threads | Model-free Vulkan micro benchmark at the projection shape; ALU-only ceilings 97 (fp16) and 289–566 (int8); go line was 600. Submit cost (42 µs) is not the issue. An earlier 27B Vulkan prefill was slower (16.9 vs 27.2 tok/s) and starved CPU memory reads | [M GPU prefill study] |
| Hexagon NPU for LLM decode | parked | ~10.8 ms floor per published batch regardless of op size; FastRPC mapping failures around 1.1 and 2.26 GB (not a proven hard limit). Prefill on the NPU is open: [help wanted 1](HELP-WANTED.md#1-hexagon-npu-prefill-path) | [M NPU dispatch floor] |
| Flash own executor / scheduler | −0.18% | the decode graph is one dependency chain; only 12% of barriers removable | [M 017] |
| Flash fused phases | −2.08% | under the 3% line; time was not in the fused nodes. Kept in the small-gains stack | [M 036] |
| Flash 11 decode threads (vs 8) | 13.8% slower | decode is bandwidth- and barrier-bound; 11 threads is a prefill setting | [M 035] |
| Spreading Flash decode threads across clusters | no gain, IO latency +27% | the IO core's cluster must hold no compute threads | [M placement screens] |
| Flash eviction policies beyond LFU + quotas | — | online policies are ~2.1× the Belady optimum; the gap is prediction, not policy | [D IO-4 replay] |
| MTP depth ≥2 on Ornith | 5.4% / 19% slower | a MoE verify reads the union of the experts all its tokens route to | [M 018] |
| Prompt-lookup drafting on the MoEs | ≤ +3.4% simulated | below the bar at today's lane cost | [E simulation] |
| MoE: hot experts kept unpacked, 1–2-row fast paths, 3–4-row pair tiles, software prefetch in tiles | ≤0.2% or slower | unpack is per expert unit and already amortised | [M/D kernel screens] |
| Cluster-aware partitioning for MoE prefill | 0.55–0.93× at every core set | the 12-thread scaling loss is thermal, not L2 or bandwidth | [M kbench] |
| Pinning DDR/LLCC devfreq at max | nothing to gain | DDR is already at max frequency 100% of decode | [M] |
| Image DiT K-blocking, bigger tiles, batch 2/8, software prefetch; VAE blocked conv | no gain / 2× slower | — | [M] |
| Flash caches of 21–22 GiB | refused or stopped by the memory guard | 21 GiB was admitted at 26.09 GiB free and stopped 8 MiB under the floor during load | [M 035] |
| Profile-derived "overhead" estimates | wrong twice (executor projected −22 ms, got −0.18%; fusion projected −14…−23 ms, got −2.6 ms) | per-node profiles distort IO overlap | rule: decide on an A/B |

## 7. Claims we do not make

- A single "stock → now" speedup, or any product of rows from different tables.
- Any Mac comparison (section 5).
- Session wall-time gains on the current stack (no clean replay yet).
- Flash prefill faster than BigMoeOnEdge (no matched length).
- A cache-size effect from rows measured with different protocols (14 vs 16/18 GiB).
- Results for other X Elite / X Plus laptops, RAM sizes or firmware: one machine, and MEMLAT depends on its firmware.
- Thermal steady state over long sessions.
- Image speed: only 512 px is timed (IMG-SHIP-1); 1024 px and int8 attention are not qualified.
