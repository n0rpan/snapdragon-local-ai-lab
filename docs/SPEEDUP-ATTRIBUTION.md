# Speedup attribution: where every gain comes from

Status: draft for the public release, written 2026-11-10 from existing receipts. No model was run and nothing was built to write it. Updated 2026-11-10 evening with the day's measured steps (rows D14, P9, P13b, F6c, F11–F13, S7b, S9, S10; claims 30–34). Updated again 2026-11-10 late evening (release team): the installed-today block, rows D15, D16 and P14, section 1.6 (small-gains stack), the state of F7, F8, F11 and F13, claims 35–40, and section 5 items 8–9. Again no model was run; every new number was read from its receipt.

This document answers one question for each workload: starting from stock software on this laptop, which change bought how much, how well is that measured, and did it change the model's output?

**Read this first.** There is no single matched "stock upstream llama.cpp → today" run yet. The ladders below are chains of separate controlled steps. Each step has its own baseline, and the steps were measured on different days, tools and prompts. Do not multiply the rows into one number. The single matched stock-vs-now run is planned for just before launch (section 5).

## Machine

Lenovo Yoga Slim 7x Gen 9, Snapdragon X Elite X1E80100 (12 Oryon cores in three 4-core clusters, each cluster sharing its L2), 32 GB LPDDR5X, NVMe SSD, Linux ARM64. CPU only: no GPU or NPU is used for LLM inference in any number here.

Models:

| Short name | Model | Notes |
|---|---|---|
| Ornith | Ornith-1.5-35B-A3B, APEX-MTP Compact GGUF | Everyday default. Routed experts mostly Q3_K, dense Q4_K. 1.80 GB of weights read per token. |
| Qwen | Qwen3.6-35B-A3B MTP, UD-Q4_K_M | Selectable. Experts Q4_K/Q5_K/Q6_K, dense Q8_0. 2.63 GB per token. |
| Flash | Qwen3.8-Flash-Next, mixed UD-IQ3_XXS, four shards, 83.6 GB | 125B main weights, 512 experts, top 10 routed. Does not fit in RAM; experts are streamed from NVMe. |

## How to read the tables

Evidence class:

- **[M n=k]** measured, k runs per arm, on a quiet machine, with a receipt.
- **[S]** a single run or single pair, or a run known to be perturbed by other jobs. Direction only.
- **[E]** estimate or arithmetic from other numbers. Not a measurement.

Quality class:

- **lossless**: identical greedy tokens, or bit-identical logits and state, against the path it replaces.
- **reorder**: same math in a different floating-point order. Output can differ slightly. Stated with mean KLD and top-1 agreement against the exact path.
- **lossy**: changes the weights or drops work. None of the installed steps is lossy.

"Installed" means it runs in the daily setup on the test laptop today. "Candidate" means measured but not installed.

**Installed today (2026-11-10 evening).**

| Model | Engine in the daily setup | What is on |
|---|---|---|
| Ornith (default) | Rust engine `oxe`, combined build `259b31c`, server binary `a4662e2c…` | 32K context, parked sessions (2 GiB), 8 turn checkpoints, prefix store, controller request queue, the streaming tool-call parser fix (S9) and prefill mask 13 (P9, `OXE_PF_TILES=13`). MTP off. |
| Qwen | Rust engine, the earlier build (`707a0ec` / `649ac806`) | Unchanged today: no parser fix, no mask 13. Its checks are pending. |
| Flash | C++ streaming engine (BigMoeOnEdge), 12 GiB expert cache, in its own 29 GiB memory scope | Unchanged engine. No Rust Flash build is installed. |

The Ornith install passed its protected coding task (13/13 checks, 7 tool calls, 65.1 s). That was a functional run on a busy machine, not a timed pair: it supports "installed and working", not a speed figure. The previous Rust build and the patched llama.cpp remain one-command rollbacks.

Sources are repo-relative (`docs/…`, `benchmarks/…`, `experiments/…`) or a hardware ticket number (`runq NNN`, in the development handoff notes; their public receipts are `benchmarks/results/runqNNN-*.json` where they exist).

---

## 1. Ladders

### 1.1 Ornith and Qwen: decode

**Platform and settings (llama.cpp, our pinned build; same binary on both sides of each row)**

| # | Step | Before → after | Delta | Evidence | Quality | State | Source |
|---|---|---:|---:|---|---|---|---|
| D1 | MEMLAT device-tree fix (DDR/LLCC frequency votes) | CPU streaming read 48.88 → 115.85 GB/s | 2.37× | [M n=3] | lossless | installed | `docs/MEMLAT.md`, `docs/ROOFLINE.md` |
| D2 | Same fix, Ornith server decode, 4K ctx, 4 threads, MTP 2 | 12.73 → 17.18 tok/s | +35% | [M n=3], outputs byte-identical | lossless | installed | `docs/MEMLAT.md` |
| D3 | 8 packed decode threads + 12 prefill threads, MTP off (Ornith, 4K server) | 17.28 → 29.38 tok/s | +70% | [M n=9 requests per arm], outputs match | lossless | installed | `docs/SCOREBOARD.md` |
| D3b | Same, at the production 16K context | 16.54 → 30.12 tok/s | +82% | [M n=9 requests per arm] | lossless | installed | `docs/SCOREBOARD.md` |
| D3c | Same idea for Qwen, target-only llama-bench (4 unpinned → 8 packed) | 24.08 → 26.88 tok/s | +11.6% | [M n=3] | lossless | installed | `docs/SCOREBOARD.md` |

D3 changes three knobs together (thread count, placement, MTP). They were not separated. The 12.73 → 29.38 endpoint (2.31×) links D2 and D3; it is not one interleaved A/B.

**Stock upstream llama.cpp, pinned commit `5f490d7`, on the MEMLAT boot**

This is the only true unmodified-upstream baseline we have. It is llama-bench with random tokens at short context (tg128), 8 threads, so it is not agent speed.

| Model | Upstream master tg128 | Evidence |
|---|---:|---|
| Ornith | 30.98 tok/s | [M n=6] |
| Qwen | 26.18 tok/s | [M n=6] |

**Our llama.cpp patches against that upstream baseline (each patch alone)**

| # | Step | Before → after | Delta | Evidence | Quality | State | Source |
|---|---|---:|---:|---|---|---|---|
| D4 | Q3_K i8mm repack, Ornith tg128, 8 threads | 30.98 → 32.87 tok/s | +6.1% | [M n=3 vs 6] | lossless (KLD 0, top-1 100%) | installed in the llama.cpp fallback | `docs/UPSTREAM-EVIDENCE.md` |
| D4b | Same patch, Qwen (negative control: no Q3_K experts) | 26.18 → 26.27 | flat | [M n=3 vs 6] | lossless | — | same |
| D5 | Routed-expert GEMM (`MUL_MAT_ID`), decode | flat on both (−0.9% to +0.5% at 8 threads) | none | [M n=3 vs 6] | lossless | installed in the fallback | same |

The GEMM patch does nothing for decode by design: one token does not form a batch.

**Rust engine (`oxe`)**

| # | Step | Before → after | Delta | Evidence | Quality | State | Source |
|---|---|---:|---:|---|---|---|---|
| D6 | Rust engine vs our patched llama.cpp, Ornith, same 142-token code prompt, 1,024 output tokens, 16K ctx, 8 packed cores | 30.52 → 42.06 tok/s | +38% | [S] one screen per arm | not matched: the engines use different arithmetic and the continuations differ after the start | installed | `docs/RUST-M3.md` |
| D6b | Rust engine, Qwen, same screen, plain decode | 33.07 tok/s (no matched plain llama.cpp arm in that screen) | — | [S] | — | installed | `docs/RUST-M3.md` |
| D7 | Static plan (inside D6, not isolated): 171–211 barriers per token against 3,869 graph nodes per token in llama.cpp | part of D6 | not separated | [E] for the share | lossless by design | installed | `docs/SCOREBOARD.md` (Rust M0), `docs/RUST-M3.md` |
| D8 | Tiled attention, decode at 16K prompt, same binary legacy vs tiled | Ornith 30.53 → 37.30; Qwen 25.43 → 28.93 tok/s | +22% / +14% | [S] single pair each | lossless (same token IDs; bit-exactness checked on separate fixtures up to 8K) | installed | `docs/RUST-ATTENTION.md` |
| D8b | Same at 8K | Ornith 37.48 → 41.55; Qwen 28.68 → 30.16 | +11% / +5% | [S] single pair | lossless | installed | same |
| D8c | Same at 1K–4K, repeated | Qwen 1K +1.3%, 2K +1.5%, 4K +4.5%; Ornith 2K +3.2% | flat to small | [M n=2–4 per arm] | lossless | installed | same |
| D9 | 32K context instead of 16K, short-context decode (Ornith) | 22.608 → 22.581 ms/token | −0.12%, flat | [M n=2 per arm, ABBA] | lossless (same hash) | installed | runq 008 |
| D10 | Server path vs bench path, same token IDs | Ornith server 1.85% slower than bench; Qwen 6.57% | a gap, not a gain | [S] first-request outliers above the 2% spread rule | — | fix written, not installed | runq 004 |
| D11 | Qwen MTP depth 2 (native multi-token head, exact verify) | 33.07 → 43.05 tok/s | +30% | [S] one screen per arm | lossless (same output hash as MTP off) | **off in production**: prompt-side cost lost time on the real task | `docs/RUST-M3.md`, `docs/RUST-SERVER-QUALIFICATION.md` |
| D12 | Ornith MTP depth 1 (own head) | 42.06 → 42.86 tok/s | +1.9%, flat | [S] | lossless | off | `docs/RUST-M3.md` |
| D13 | Ornith with a grafted donor MTP head | 2.19–2.38 tokens per verify at depth 2 | no timing yet | [M] correctness only, 3 prompts | lossless (256/256 logit rows identical to MTP off) | candidate | runq 014 |
| D14 | Ornith MTP depth 1 with the donor head restricted to a 32,768-row draft vocabulary (ticket 018) | 22.66 → 21.28 and 22.67 → 21.25 ms/token (44.1 → 47.0 tok/s) | −6.1% / −6.3% time (+6.5% / +6.7% rate) | [M n=2 passes, forward and reversed]; **short screen only**: 53-token prompt, 2K context, 128 timed tokens | lossless (all 14 arms one output hash; target logit rows bit-identical on 3 prompts × 256 tokens) | candidate, in the night build; not installed | runq 018 |
| D14b | Same head, deeper drafts | depth 2: 23.88 / 23.91; depth 3: 27.02 / 26.96 ms/token | 5.4–5.5% and 18.9–19.3% **slower** | [M n=2 passes] | lossless | rejected | runq 018 |
| D14c | Unrestricted donor head | depth 1: 22.97 / 23.35; depth 2: 26.43 / 26.67; depth 3: 30.89 / 31.09 ms/token | 1.4–3.0%, 16.7–17.6%, 36.4–37.1% **slower** | [M n=2 passes] | lossless | rejected | runq 018 |
| D15 | Cost of one verify pass by number of lanes (tokens checked together), Ornith, `verify-bench` | 1 lane 22.49 ms; 2 lanes 35.70; 3 lanes 50.64; 5 lanes 80.21 ms = 1.000 / 1.587 / 2.252 / 3.566 plain steps | each extra lane costs 0.59–0.67 of a plain token | [S] one 14.8 s development quick check, not a timing run | n/a (a cost measurement) | explains D14/D14b; no change installed | quick check 045-lite |
| D16 | Prompt-lookup drafting (drafts copied from text already in the context), priced with D15 | simulated change in session wall time: −2.8% (default setting) to +3.4% (best setting) | below the 8% bar set for building it | [E] simulated on one recorded session; lanes beyond 2 extrapolated | lossless by design (exact verify) | **shelved** until a verify lane costs clearly less (reopen at ≤1.45 plain steps for 2 lanes) | handoff notes, 2026-11-10 |

Where decode stands today:

- Bench, short context: Ornith 22.6 ms/token (44 tok/s), Qwen 29.8 ms/token (34 tok/s) [M n=2 warm controls, runq 004].
- Real 42-request coding session: Ornith 35.9 tok/s overall (11,339 tokens in 316.2 s) [S]. Context length and heat (88–97 °C all session) explain most of the difference from the bench; the split is not measured.
- Bandwidth roof proxy for Ornith at this byte count: about 48–55 tok/s at 108 GB/s [E]. Decode is at roughly 80% of it.

### 1.2 Ornith and Qwen: prefill

**Platform and settings (our pinned llama.cpp)**

| # | Step | Before → after | Delta | Evidence | Quality | State | Source |
|---|---|---:|---:|---|---|---|---|
| P1 | MEMLAT, Ornith server, 4 decode / 10 batch threads | 94.74 → 107.70 tok/s | +13.7% | [M n=3] | lossless | installed | `docs/MEMLAT.md` |
| P2 | 12 prefill threads (with D3) | 107.70 → 119.86 tok/s | +11.3% | [M n=9 requests per arm] | lossless | installed | `docs/BEFORE-AFTER.md` |

**Stock upstream llama.cpp `5f490d7`, pp512, MEMLAT boot**

| Model | 8 threads | 12 threads | Evidence |
|---|---:|---:|---|
| Ornith | 93.88 | 121.14 tok/s | [M n=6] |
| Qwen | 143.80 | 172.33 tok/s | [M n=6] |

**Our llama.cpp patches against upstream (each alone, pp512, 12 threads)**

| # | Step | Before → after | Delta | Evidence | Quality | State | Source |
|---|---|---:|---:|---|---|---|---|
| P3 | Q3_K i8mm repack, Ornith | 121.14 → 165.23 tok/s | +36.4% (+45.0% at 8 threads) | [M n=3 vs 6] | lossless (KLD 0, top-1 100%) | installed in the fallback | `docs/UPSTREAM-EVIDENCE.md` |
| P4 | Routed-expert GEMM, Qwen | 172.33 → 210.52 tok/s | +22.2% (+25.6% at 8 threads) | [M n=3 vs 6] | lossless (KLD 0, top-1 100%) | installed in the fallback | same |
| P4b | Cross-checks: GEMM alone on Ornith, Q3_K alone on Qwen | flat (within ±1%) | none | [M n=3 vs 6] | lossless | — | same |
| P5 | Both patches in one engine, Ornith model ABBA, 12 threads | 119.29 → 185.03 tok/s | +55.1% | [M n=3 samples × 2 arms] | lossless (KLD 0) | installed in the fallback | `experiments/q3k-i8mm/RESULTS.md` |

A first, broader GEMM version failed its numeric check (KLD 0.0217, top-1 92.7%). It was a float accumulation-order bug, fixed before anything was installed. It is kept in the record.

**Rust engine**

| # | Step | Before → after | Delta | Evidence | Quality | State | Source |
|---|---|---:|---:|---|---|---|---|
| P6 | Rust batched prefill vs our patched llama.cpp, pp512, 12 threads, same token IDs | Ornith 193.9 → 208.7; Qwen 222.3 → 214.1 tok/s | +7.6% / **−3.7%** | [S] single ordered screen, 3 timed reps | first token matches; arithmetic differs | installed | `docs/RUST-M3.md` (prefill section) |
| P6b | Same at 4,096 tokens | Ornith 162.9 → 173.5; Qwen 180.6 → 175.7 | +6.5% / **−2.7%** | [S] | same | installed | same |
| P7 | Tiled attention, 2K extension on a 6K prefix, same binary | Ornith 126.23 → 164.00; Qwen 126.32 → 166.57 tok/s | +29.9% / +31.9% | [M n=3 timed reps per arm] | lossless (first token matches; bit-exactness checked separately) | installed | `docs/RUST-ATTENTION.md` |
| P8 | Grouped prefill (group 2), pp2048 | Ornith 202.31 → 207.46; Qwen 197.88 → 209.25 | +2.5% / +5.7% | [S] partial screen | first token matches | not installed | `docs/RUST-GROUPED-PREFILL.md` |
| P9 | Prefill kernel switches, ticket 032, mask 13 (Ornith only; see below) | see table | +19% to +45% | [M n=2 reps per arm, interleaved; both reps beat both baseline reps on every shape] | lossless (bit-exact logits, full state, decode rows and greedy tokens on 5 fixtures) | **installed for Ornith 2026-11-10** (`OXE_PF_TILES=13`); Qwen unchanged | runq 032, runq 016 |
| P14 | GPU prefill on the Adreno X1-85 through Vulkan: a model-free matrix-multiply micro benchmark at the attention-projection shape | GPU fp16 8.9 GMAC/s; GPU int8 18.5–48.6; arithmetic-only GPU ceilings 97 (fp16) and 289–566 (int8); the 12 CPU threads do about 1,500 GMAC/s on the same dense work | 30–170× short of the CPU; the go line fixed in advance was 600 | [M] micro benchmark with a plain shader on the Mesa Turnip driver; no model was run | n/a | **no-go; GPU LLM prefill stays parked** | handoff notes (GPU prefill study), 2026-11-10 |

P9 detail (Ornith, 16K context capacity, 12 threads unless noted):

| Shape (new tokens on prefix) | Mask 0 → mask 13, tok/s | Time saved | Rate change |
|---|---:|---:|---:|
| 2,048 on 0 | 200.33 → 237.99 | 15.8% | +18.8% |
| 2,048 on 6K | 142.83 → 183.10 | 22.0% | +28.2% |
| 1,024 on 12K | 112.94 → 150.86 | 25.1% | +33.6% |
| 24 on 12K (a tool-call turn) | 83.94 → 121.57 | 31.0% | +44.8% |
| 2,048 on 0, 8 threads | 165.62 → 202.42 | 18.2% | +22.2% |

Mask 13 is three switches: dense Q4_K projections in 8-token × 4-row tiles, attention scratch columns moved off the power-of-two stride, and single leftover expert rows through the GEMV. A fourth switch (position-blocked attention outputs) added nothing and is left out. Qwen has not been timed or checked with these switches. Two reps per arm is below the release rule of three.

**Not re-prefilling (largest prefill lever in real sessions)**

| # | Step | Before → after | Delta | Evidence | Quality | State | Source |
|---|---|---:|---:|---|---|---|---|
| P10 | Persistent prefix store (system prompt + tool schemas restored from disk) | First prompt of a session: Ornith 30.64 → 1.22 s; Qwen 31.25 → 1.22 s (5,376 tokens restored, 168 new) | −96% of first-prompt time | [S] n=1 per model | lossless (state restore) | installed | `docs/RUST-SERVER-UPGRADE.md` |
| P11 | Parked sessions: a conversation displaced by a title or subagent request is copied to RAM and restored | 32K-token session: 817 MiB, resumed in 42 ms, 1 new prompt token evaluated. 1.1K-token session: park 7 ms, resume 7 ms | replaces a full re-prefill | [M] real model, exactness on Ornith and on Qwen with MTP 2 | lossless (unparked state byte-identical; token IDs equal the uninterrupted session) | installed for Ornith | runq 008 |
| P11b | What that re-prefill would have cost | about 55–90 s for 10–16K tokens at ~180 tok/s | — | [E] | — | — | runq 008 summary |
| P12 | Turn checkpoints (8 instead of 4, at message boundaries) | New prompt tokens over the same 42 requests: 45,936 → 37,765 | −8,171 tokens (−17.8%) | [M] token count from one replay pair; the count is deterministic | lossless | installed for Ornith | batch-1 replay, `docs/MOE-BATCH1.md` |
| P12b | Prefill seconds in that replay pair | 344.7 → 287.8 s | −16.5% | [S] **both arms perturbed** | lossless | installed for Ornith | same |
| P13 | Session cache across idle unload (disk) | reload reused 0 of 2,304 tokens | **failed its benefit check** | [S] | identity passed only through the cold fallback | not deployed | handoff notes, 2026-11-10 |
| P13b | Same, after the fix (turn checkpoints are written to disk, not only the final session) | reload reused 2,304 of 2,304 tokens; 3 files, 340 MB written in 183 ms, loaded in 15 ms | replaces a re-prefill after idle unload | [S] n=1, synthetic 2,349-token transcript; expiry and corruption cases not rerun | lossless (same reply hash as the uninterrupted session) | candidate, in the night build; not installed | runq 038 |

Where prefill stood before P9 was installed (Ornith; no session measurement with P9 exists yet): 133 tok/s effective over a real 42-request session, 200–240 tok/s on a cold 512–2,048-token prompt, falling with prefix length (about 0.45 ms more per token per 1K positions of prefix) [S for the session figure, M for the bench shapes].

### 1.3 Flash: decode

Fixed workload unless stated: the same 156-token coding prompt, about 342–353 generated tokens, checked by an independent code oracle.

| # | Step | Before → after | Delta | Evidence | Quality | State | Source |
|---|---|---:|---:|---|---|---|---|
| F1 | Early SSD tuning in the C++ streaming engine: IO workers 2 → 4, demand mapping, expert reads overlapped with compute | request 382.0 → 227.3 s; decode 1.24 → 1.71 tok/s for the last step | −40.5% request time, first arm vs last arm | [S] first/last arms of separate pairs, not a combined A/B | lossless (byte-identical answer) | installed | `docs/BEFORE-AFTER.md` |
| F2 | Unexplained: the same untouched binary later ran faster | 1.707 → 3.753 tok/s | 2.2× | [S] | lossless | — | `docs/FLASH-NEXT-PROFILE.md` |
| F3 | Production baseline after MEMLAT (C++ streaming engine, 12 GiB expert cache, 4 decode threads) | **4.06 tok/s = 246.5 ms/token** | baseline for the rows below | [M] | exact answer | installed (this is what runs today) | `docs/SCOREBOARD.md` |
| F4 | Rust engine, exact path, 16 GiB cache, 4/6 threads | 246.5 → 193.9 ms/token | −21% | [S] n=1; cache size and engine both changed | lossless (353/353 tokens identical) | candidate | runq 001 |
| F5 | Rust fast path: 8-thread graph, fixed-chunk attention, shared-expert-first, 20 GiB cache | 193.9 → **123.07 ms/token (8.1 tok/s)** | −37% | [M n=3], all answers identical to each other | **reorder**: mean KLD 6.55e-4, top-1 99.72% vs the exact path; code oracle passes | candidate | runq 002, 003 |
| F6 | Cache size policy, real forward: 18 → 20 GiB | 136.29 → 123.07 ms/token | −9.7% | [M n=3 per size] | same path | candidate | runq 003 |
| F6b | Cache size, timing prototype (dummy activations): 16 → 18 → 20 GiB | 122.1 → 105.2 → 96.0 ms/token | — | [S] one screen per size; not real inference | — | — | `docs/FLASH-M0-TUNING.md` |
| F6c | 21 GiB cache with a 29 GiB scope and 1 GiB host floor | 20 GiB: 116.6 ms (hit rate 90.9%, minimum available RAM 2.46 GiB); 21 GiB: 112.35 ms/token (8.9 tok/s; 91.8%; 1.62 GiB) | −3.6% | [S] one run per size, in a memory test. Later the same day a 21 GiB load was stopped 8 MiB under the 1 GiB floor and 20 GiB needed a desktop reclaim: 21 GiB is not reliably admissible on this desktop | same path | candidate; 20 GiB is the working size | handoff notes, 2026-11-10; runq 035 |
| F7 | Static plan fixes (ticket 010) | 128.31 → 124.65 ms/token | −2.85% | [M n=2 per arm, ABBA]; below the 3% line | lossless (354 logit rows bitwise identical) | not credited alone; on by default in the Rust Flash candidate; in the small-gains stack (1.6) | runq 010 |
| F8 | Own executor (ticket 017), 18 GiB | 125.66 → 125.43 ms/token | −0.18%, flat | [M n=2 per arm, ABBA] | numeric check passed after a fix | no promotion; kept as a tracing harness; listed in 1.6 | runq 017 |
| F9 | Compute floor with zero SSD wait (timing prototype) | 65.3 ms/token (15.3 tok/s) | a bound, not a result | [M n=3], dummy activations | — | — | `docs/FLASH-M0.md` |
| F10 | MTP depth 2 on the C++ path | 4.27 → 5.48 tok/s | +28% | [S] and **unqualified**: the output differed from MTP off | failed the identity check | stopped | `docs/FLASH-MTP-ACCEPTANCE.md` |
| F11 | Cache policy: LFU with per-layer quotas instead of Clock, 8 threads (ticket 035) | 20 GiB: 123.20 / 129.40 → 117.56 / 121.14 ms/token; 18 GiB: 135.14 / 139.55 → 128.34 / 129.28 | −5.5% / −6.2% of mean token time | [M n=2 per arm, one screen per cache size]; both candidate runs beat both baseline runs at each size; the 20 GiB baseline drifted 5.0% first to last | same numeric path (cache policy only); code oracle passes on all 14 runs | candidate; the default cache policy for further Rust Flash candidate builds; not installed | runq 035 |
| F12 | 11 decode threads instead of 8 (ticket 035, 20 GiB) | 126.30 → 143.72 ms/token mean; with LFU 137.12; with LFU and two-pass attention 133.78 | 13.8%, 8.6%, 5.9% **slower** | [M n=2 per arm] | two-pass attention is reorder class (mean KLD 9.2e-4 on 354 positions) | rejected: Flash decode stays at 8 threads | runq 035 |
| F13 | Fused phases, `--fuse 31` vs 0 (ticket 036), 20 GiB, 8 threads | 123.55 / 125.17 → 121.05 / 122.49 ms/token | −2.08%, below the 3% line | [M n=2 per arm, ABBA]; per-phase arms all within 3% of the reference | lossless (teacher-forced logits byte-identical to each other and to the ticket-010 control; all 15 answers byte-identical) | not credited alone; **kept in the small-gains stack** (1.6) | runq 036 |

Tested and rejected for Flash decode, each one screen: NVMe queue depth 32 → 4 (−0.2%), kcryptd CPU mask (+0.6%), a 22 GiB cache (stopped by the memory guard), blanket Q4 dense requant (KLD 0.125).

Production 246.5 ms against the candidate 123.07 ms is 2.0×, but the two differ in engine, thread count, cache size and numeric path at once. The candidate also missed its own 105 ms gate. It is not installed.

### 1.4 Flash: prefill

| # | Step | Before → after | Delta | Evidence | Quality | State | Source |
|---|---|---:|---:|---|---|---|---|
| G1 | C++ engine, microbatch 128 → 512, 350-token prompt | 101.02 → 53.80 s | 1.88× | [S] one pair | — | installed | `docs/BEFORE-AFTER.md` |
| G2 | C++ engine, microbatch 512 → 4096, 922-token prompt | 140.14 → 102.97 s | 1.36× | [S] one pair | — | installed | same |
| G3 | Production reference, 156-token prompt | 11.55 s (13.5 tok/s) | baseline | [M] | — | installed | `docs/FLASH-MTP-ACCEPTANCE.md` |
| G4 | Rust exact engine, mapped prefill → streamed layer-major batched prefill, 156 tokens | 208.3 → 11.7–12.9 s | about 16–18× inside the Rust engine | [S] | lossless (state files bit-identical: streamed GEMM = streamed GEMV = mapped) | candidate | runq 001, 002, 003, 006 |
| G5 | Streamed batched prefill at length | 2,048 tokens: 20.39 / 20.49 tok/s; 5,500 tokens: 18.96 / 17.64 tok/s | — | [M n=2 passes] | lossless | candidate | runq 006 |
| G6 | Batched expert GEMM vs GEMV, 2,048 tokens | 16.85 / 16.43 → 20.39 / 20.49 tok/s | +21–25% | [M n=2 per arm] | lossless (identical state) | candidate | runq 006 |
| G7 | 11 compute threads with a dedicated IO core (vs 12 compute threads sharing a core with IO) | 15.30 / 15.45 → 23.02 / 22.68 tok/s | +49% | [M n=2 per arm]; thread count and IO sharing changed together | lossless (deterministic) | candidate | runq 011 |
| G8 | Microbatch 1024 instead of the default | 18.77 / 19.04 tok/s | slower | [M n=2] | — | rejected | runq 011 |

Honest reading: at 156 tokens the Rust streamed prefill (11.7–12.9 s) is no faster than the production engine (11.55 s). At 2K–5.5K tokens it reaches 18–23 tok/s, and there is no production measurement at a matched length, so no production-relative prefill gain is established. The 50 tok/s target was missed. The 2K profile says attention is about 36% of prefill time and the batched expert op about 22%.

### 1.5 Real OpenCode session wall time

All rows are a real OpenCode coding task with independent checks (13/13 in every arm unless noted). They are different tracks; a row's baseline is the row's own.

**Ornith and Qwen**

| # | Step | Before → after | Delta | Evidence | Quality | State | Source |
|---|---|---:|---:|---|---|---|---|
| S1 | Thread and profile retune (4/10 → 8/12 threads), load to finished task | Ornith 106.37 → 92.69 s; Qwen 119.02 → 106.06 s | −12.9% / −10.9% | [S] one pair each; file-cache order not controlled | same weights | installed | `docs/SCOREBOARD.md` |
| S2 | Patched llama.cpp (Q3_K repack, GEMM, repack cache, no prefetch), direct server | Ornith 99.77 → 64.00 s | −35.8% | [S] one pair; an earlier reference was 89.90 s | lossless (KLD 0) | installed as the fallback | `docs/SCOREBOARD.md` |
| S2b | Same engine through the real controller, 9-tool task | cold 121.41 → 113.77 s; cached 63.15 → 64.56 s | −6.3% / +2.2% (flat) | [S] one pair each | lossless | installed as the fallback | `docs/SCOREBOARD.md` |
| S3 | Prefix store, cold session vs second session | Ornith 121.41 → 63.15 s | −48% | [S] one pair; a caching effect, not a kernel speedup | lossless | installed | `docs/BEFORE-AFTER.md` |
| S4 | Patched llama.cpp → Rust server | Ornith 77.70 → 60.42 s; Qwen (MTP 2 both) 75.31 → 67.42 s | −22.2% / −10.5% | [M n=3 per arm], medians | passes the same checks; trajectories differ | installed (successor build) | `docs/RUST-SERVER-QUALIFICATION.md` |
| S5 | Rust → Rust with tiled attention and the MTP fixes | Ornith 60.06 → 52.13 s; Qwen 65.53 → 57.91 s | −13.2% / −11.6% | [M n=3 per arm], medians, same token counts | lossless (matches 3 native greedy references) | installed | `docs/RUST-SERVER-UPGRADE.md` |
| S6 | Installed llama.cpp → Rust, warm prefix, fresh process | Ornith 30.24 → 23.42 s; Qwen 70.54 → 29.33 s | −22.5% / −58.4% | [S] n=1; the Qwen baseline is an older engine with a 35 s start | trajectories differ | installed | `docs/RUST-SERVER-UPGRADE.md` |
| S7 | Batch 1 (32K context, parked sessions, 8 turn checkpoints), equal-work replay of a recorded 42-request session | Ornith 661.3 → 607.0 s | −8.2% | [S] **one pair, both arms perturbed by other jobs** | lossless | installed for Ornith | `docs/MOE-BATCH1.md`, handoff notes |
| S7b | Same replay on Qwen | invalid: 6 stream errors from a tool-call parser bug, fewer tokens generated | no number | — | — | fix installed for Ornith (S9); Qwen check pending | handoff notes |
| S8 | Controller request queue | nine HTTP 503 retries → zero; one subagent had started about 55 s late | not a tok/s gain | [M] one recording each | n/a | installed | `docs/MOE-BATCH1.md` |
| S9 | Streaming tool-call parser fix with the greedy fast path (ticket 016) | old stream exposed truncated invalid JSON arguments at a token cutoff and old non-stream dropped the call; new stream and non-stream agree | a correctness fix, not a speed step | [M] 5/5 assembled outputs identical fast vs legacy path; 3 prompts × 256 greedy tokens identical; protected task 13/13 in 65.1 s (functional run, no paired timing) | plain completions identical to the previous build; tool-call output changes on purpose | installed for Ornith 2026-11-10 | runq 016 |
| S10 | Cancel on client disconnect, with the partial prefill kept (ticket 039) | a cancelled 12K-token prefill: next request answered in 2.0 s; the retry resumes 512 tokens (the first build re-ran all 12,000 in 73.9 s) | removes wasted work; not a tok/s gain | [S] one live check per case | control and cache bookkeeping only | candidate, in the night build; not installed | runq 039 |

In the S7 baseline replay the session splits into 344.7 s prefill (52%) and 316.2 s decode (48%); everything else is 0.4 s.

Estimated and unmeasured session levers [E]: the ticket-032 kernels (P9) could save about 60–90 s of the 661 s session; avoiding the template's re-prefill of earlier think blocks about 40–90 s; decode attention about 15–20 s. None of these is a result.

**Flash**

| # | Step | Before → after | Delta | Evidence | Quality | State | Source |
|---|---|---:|---:|---|---|---|---|
| T1 | First completed protected repair (after F1) | deadline failure → 1175.2 s, 12/12 checks | no ratio: the earlier run never finished | [S] | — | installed | `docs/BEFORE-AFTER.md` |
| T2 | Same task later | 1175.2 → 768.1 s | 1.53× | [S]; **cause unknown, not credited to software** | 12/12 | — | `docs/FLASH-NEXT-PROFILE.md` |
| T3 | IO depth 4 → 8 | 768.5 → 736.8 s | −4.1% | [S] one pair, outputs differ, below the 5% gate | 12/12 | not installed | `docs/BEFORE-AFTER.md` |

The Rust Flash engine has no session-level measurement yet.

### 1.6 The small-gains stack

A lossless change that measures below the 3% line is not counted as a gain on its own, and it is not thrown away either. Such changes are collected and will be timed **together** as one arm against production, once the larger items are installed. The stack is kept if the combined arm is at least 3% faster; single items are then attributed by leaving one out. Until that run exists, the stack has **no measured total**, and the per-item numbers below must not be added up.

| Item | Engine | Measured alone | Quality | Evidence |
|---|---|---|---|---|
| Fused phases, `--fuse 31` (F13) | Flash, Rust | −2.08% | lossless (byte-identical) | [M n=2 per arm, ABBA], runq 036 |
| Static plan: graph reuse and state views (F7) | Flash, Rust | −2.85% | lossless (bit-identical) | [M n=2 per arm, ABBA], runq 010 |
| Own executor (F8) | Flash, Rust | −0.18%, flat | lossless | [M n=2 per arm, ABBA], runq 017 |
| Attention pre/post fusion, per-width graph cache, barrier padding | Flash, Rust | not measured alone | lossless (model-free bitwise tests) | — |
| Two-pass attention | Flash, Rust | not measured alone at 8 threads | reorder (mean KLD 9.2e-4) | numeric check only, runq 035 |
| Greedy fast path in the server (engine-side argmax, no logits copy) | Ornith/Qwen, Rust | about 1.4% / 1.1% | lossless (identical tokens) | [E] derived from a 0.33 ms per-token host cost in runq 004; installed for Ornith with S9 |
| Q6_K tiles, scratch-column pads, fused delta-net step in decode | Ornith/Qwen, Rust | not measured (kernel estimates of 0.2–3%) | lossless (bitwise tests) | [E] |

Why the fused-phases result was small: the plan projected 14–23 ms per token from counting graph nodes, and the measurement gave 2.6 ms. The time was not in the fused nodes. It is the second such miss after the executor (F8), and the reason derived "overhead" figures are tagged [E] in this document.

---

## 2. Why each technique fits this hardware

The chip in one paragraph: twelve identical Oryon cores in three clusters of four, each cluster with its own shared L2. NEON with `i8mm` (8-bit integer matrix multiply) and dot-product instructions; no SVE, no SME. LPDDR5X that delivers 108 GB/s of streaming reads to eight packed cores (115.85 GB/s to six spread cores) once the memory-frequency votes work. An NVMe drive that plateaus near 4.6 GB/s of direct reads at high queue depth. 32 GB of RAM.

A 35B MoE that routes about 3B parameters per token reads 1.8–2.6 GB of weights for every token. At that size decode is limited by memory bandwidth, and prefill by integer kernel throughput. That one fact drives most choices below.

**MEMLAT device-tree fix (D1, D2, P1).** The stock device tree did not expose the vendor SCMI protocol that lets the CPU ask for DDR and cache-controller frequency. Without those votes memory reads sat at about 49 GB/s whatever the thread count. A one-node overlay enables it. Because decode is bandwidth-bound, a 2.37× bandwidth gain is the largest single step. It is firmware-dependent and specific to this laptop model; credit to the Surface Pro 11 memlat work and the upstream RFC.

**Eight packed decode threads, twelve prefill threads (D3, P2).** Decode saturates the memory bus before it runs out of cores: 12 threads decoded 18% slower than 8 in a long ABBA, because the extra threads add synchronisation and no bandwidth. Prefill is compute-bound, so it uses all twelve. Packed placement (two whole clusters) keeps L2 sharing predictable.

**Q3_K repack with `i8mm` (D4, P3).** Upstream llama.cpp has fast repacked kernels for Q4_0 and Q4_K but none for Q3_K, the format Ornith's experts use. We repack Q3_K weights at load into rows the `SMMLA` instruction can consume directly. The kernel alone is 1.6–2.1× on GEMV and 3.9–4.8× on four-row GEMM; the whole model gains 36–45% in prefill and 6% in decode. Output is bit-for-bit the same as the unpacked path.

**Routed-expert GEMM (P4).** In an MoE, each expert sees only the tokens routed to it. Upstream runs those as separate vector products. Grouping the tokens that share an expert into a small matrix product reuses each decoded weight block across several tokens. Float accumulation follows the same order as the vector path, so the result is identical.

**Repack cache, no source prefetch (S2).** Repacking costs 15–25 s per load. A checksummed sidecar on disk makes it a one-time cost, and dropping the original pages after repacking avoids holding two copies in 32 GB.

**Rust engine with a static plan (D6, D7).** llama.cpp walks a general compute graph: 3,869 nodes per decoded token for these models, each with a thread barrier. Our engine supports only these architectures, lays out the work once at load, and crosses about 170–210 barriers per token. Nothing is allocated, locked or logged in the per-token loop. On a bandwidth-bound decode the time between useful memory reads is the overhead that matters, and this removes most of it.

**Tiled attention (D8, P7).** The KV cache grows 20.5 MB per 1K positions. The legacy path streamed all of it once per query token. The tiled path handles several query tokens per pass over each block of keys and values, so the same bytes are read from DRAM fewer times. It matters most where the prefix is long, which is exactly an agent session: +30% prefill on a 6K prefix, +22% decode at 16K, about flat below 4K.

**Batched prefill (P6).** Tokens go through each layer 256 at a time, so dense projections become matrix products and every weight block is read once per batch instead of once per token.

**Ticket-032 prefill switches (P9).** Three cache-shape fixes found by timing kernels in isolation:
- Dense Q4_K projections in 8-token × 4-row tiles with weights decoded once per 64-row chunk (kernel 100 → 162 GMAC/s). The decoded block stays in L1 while eight tokens use it.
- With a power-of-two context (16,384 or 32,768), the 32 score columns of one attention unit sat exactly one stride apart and all mapped to the same L1 cache set. Moving the scratch columns off that stride raised the score kernel from 19.8 to 56 GFMA/s. At a non-power-of-two context the problem does not exist.
- A single leftover expert row goes through the vector kernel instead of a padded four-row tile, which matters for 20–30-token tool-call turns where experts share almost no tokens.

**Prefix store, parked sessions, turn checkpoints (P10–P12).** Prefill runs at 130–240 tok/s here, so a 10K-token conversation costs about a minute to rebuild. An agent harness rebuilds it often: a title request, a subagent or a plan/build switch displaces the live conversation, and an edit rewinds it. State for these models is small (about 61 MiB plus 20.5 KB per token), and RAM copies run at tens of GB/s, so saving and restoring state costs milliseconds where recomputing costs a minute. This is the cheapest "speedup" on the machine and it is exact.

**Larger context at no speed cost (D9).** KV memory for 32K fits (Ornith peaks at 15.1 GiB), and short-context decode does not change, so the session hits context overflow and its full re-prefill less often.

**Flash: NVMe streaming for a 125B MoE in 32 GB (F1–F9, G4–G7).** The 83.6 GB model cannot be resident. What makes it workable:
- Dense weights (about 3.9–4.5 GB) and the router stay in RAM; they are read every token.
- The 512 experts per layer stay on SSD. Only the 10 routed per layer are needed per token, and routing is skewed, so a 16–21 GiB RAM cache gets a 91% hit rate. The misses cost about 84–118 MiB of reads per token.
- Reads use O_DIRECT at high queue depth on a dedicated IO core and overlap with compute on the other cores. The drive's 4.6 GB/s is enough that SSD wait is about 9–10 ms of a 123 ms token at 20 GiB.
- Cache size is the main policy knob, because every extra GiB turns SSD reads into RAM reads: 18 → 20 GiB is −9.7%. It is capped by what the desktop needs to stay alive, which is why the memory scope and host floor are part of the performance story.
- Prefill is layer-major: a whole batch goes through one layer while that layer's experts are resident, instead of each token dragging every layer through the cache.
- The zero-SSD-wait compute floor is 65 ms/token. The remaining gap to 123 ms is compute and synchronisation, not IO, which is why IO tuning stopped paying.

**Speculative multi-token decode (D11–D13, F10).** One verify pass over several drafted tokens reads the weights once for all of them, which sidesteps the bandwidth limit. It pays only when drafts chain: Qwen's head accepts about 2.8 tokens per verify (+30%); Ornith's own head about 1.8 (flat). It is exact when the verify equals the plain path, and it is off in production until its prompt-side cost is fixed.

The verify pass is not free, though, and D15 puts a number on it: on Ornith each extra token in a verify costs about 0.6 of a plain decode step, because a MoE verify reads the union of the experts all its tokens route to. Checking two tokens costs 1.59 steps, three cost 2.25. That one table accounts for D14 and D14b [E, arithmetic on D15]: a depth-1 draft lands about 1.8 tokens for about 1.65 steps and wins a little (measured 6%); a depth-2 draft lands about 2.3 tokens for about 2.4 steps and loses (measured 5% slower). It is also why prompt-lookup drafting (D16) is shelved: its drafts are long runs, and a five-token verify must land more than 3.6 tokens just to break even. Qwen's lane cost has not been measured; its +30% single screen (D11) should be re-read against such a table before it is quoted.

**What did not fit this hardware.** GPU prefill through Vulkan was slower than the CPU (16.9 vs 27.2 tok/s on a 27B model) and starved CPU memory reads. A later model-free micro benchmark (P14) says why it is not close: a plain compute shader on the Adreno X1-85 multiplies matrices at 9–49 GMAC/s where the twelve CPU threads reach about 1,500, the driver exposes no matrix-multiply extension, and it accelerates only unsigned or mixed-sign 8-bit dot products. Submitting work to the GPU is cheap (42 µs), so the limit is the arithmetic rate, not the hand-off. A tuned shader could be several times faster than the one measured; it would need about 70×. Flash decode at 11 threads instead of 8 is 14% slower (F12): decode waits on memory and on thread barriers, so more threads add waiting. Flash prefill is the opposite and gains from 11 compute threads with the IO thread on its own core (G7). The NPU has a per-dispatch floor near 10.8 ms and buffer-mapping failures around 1.1 GB and 2.26 GB. More than eight decode threads, pthread instead of OpenMP, and NVMe queue-depth tuning all measured flat or slower. See `docs/FAILED-PATHS.md`.

---

## 3. Mac comparison (placeholder: no matched measurement exists yet)

We have **no** Mac measurement of our own. The table lists what we hold on the Snapdragon side and what a fair row needs.

Release rules for any row in the published table:

1. Same model file (SHA256), same quant, same prompt token IDs and output length.
2. Same day, at least 3 warm runs per side, median and range reported.
3. A stock baseline on each machine (upstream llama.cpp at a pinned commit; on the Mac also its usual best stack, named and pinned).
4. Decode and prefill reported separately; context, threads, power source and thermal state stated.
5. Commit, binary SHA, receipt file and the reproduce command in the repo.
6. Hardware configuration (chip, RAM) stated for both sides; no prices.

| Workload | Snapdragon X Elite, what we have | Release-grade on our side? | Mac side | Status |
|---|---|---|---|---|
| Ornith decode, short context | 44 tok/s bench; 30.98 upstream tg128 | Partly: needs ≥3 runs with the production binary on one fixed prompt | TBD | not measured |
| Ornith decode in a real session | 35.9 tok/s | No: single perturbed replay | TBD | not measured |
| Qwen3.6-35B-A3B decode | 34 tok/s bench; 26.18 upstream tg128 | Partly | TBD | not measured |
| Ornith prefill, cold 2K | 238 tok/s with ticket 032 (installed); 200 before | No: n=2 reps per arm | TBD | not measured |
| Qwen prefill pp512 | 214 tok/s Rust; 172 upstream | Partly: single screen | TBD | not measured |
| 42-request OpenCode session replay | 607 s (Ornith) | No: perturbed | TBD (needs the same recording replayed) | not measured |
| Flash decode | 4.06 tok/s installed; 8.1 tok/s candidate | Installed figure yes; candidate no | TBD | not measured |
| Flash prefill | 13.5 tok/s installed; 18–23 candidate | No | TBD | not measured |

Third-party Mac numbers we have seen for Flash-Next (community posts; different quants, engines and prompts; not verified by us; listed only so the gap is not hidden):

| Machine | Stack | Decode tok/s | Prefill tok/s |
|---|---|---:|---:|
| M1 Max 64 GB | llama.cpp, 3.84 bpw | 17.6 | 182 (pp512) |
| M5 Max 64 GB | vendor 85 GB quant | 36.0 | 518 (pp512) |
| M4 Max 128 GB | llama.cpp, UD-Q3_K_XL | 27.3 | — |
| M4 Max 128 GB | MLX, without / with MTP | 45.0 / 90.1 | — |
| M5 Pro 64 GB | SSD-streamed experts, without / with a draft head | 18 / 27.6 | — |

**Where Macs win today**

- **Prefill.** Metal on the GPU plus higher memory bandwidth. Our working expectation for the 35B MoEs is 2–3× in the Mac's favour [E, no matched run]. On Flash the third-party figures above are roughly 9–25× our prefill.
- **Max and Ultra chips.** They have several times the memory bandwidth and enough RAM to hold Flash fully resident. A bandwidth-bound decode cannot beat that on 108 GB/s; on Flash the third-party figures are 2–11× our installed decode.
- **Flash in general.** Any Mac with 64 GB or more avoids most or all SSD streaming.

**Where we expect to be competitive, to be shown by matched runs**

- Decode on 35B-class MoEs against base and Pro chips with similar bandwidth.
- Warm agent sessions, where exact state reuse (prefix store, parked sessions, checkpoints) removes most prefill regardless of raw prefill speed.

No sentence of the form "faster than an M-series Mac" is supported until rows in the first table are filled under the six rules.

How the Mac rows get filled: we have no Mac, so the repo carries `tools/mac_reproduce.py` for a volunteer. It checks the model files against our SHA256 manifest, runs stock llama.cpp at the pinned commit (Metal, and CPU-only) and MLX, at least three warm runs with the first dropped, and writes receipts in the same format as ours, with no serial number, user name or path in them. Two limits are built into the comparison and must be stated next to any Mac row: MLX cannot load GGUF files, so an MLX row is never the same quant file; and the script has not yet been run on a Mac.

---

## 4. Claims we must NOT make yet

**Perturbed**

1. "Batch 1 makes real sessions 8.2% faster" (661.3 → 607.0 s) and "prefill 16.5% faster" (344.7 → 287.8 s). One pair, and both arms ran beside other CPU jobs. The 8,171 fewer prompt tokens is solid; the seconds are not. A clean back-to-back pair is queued.
2. Any Qwen session-replay time. The run was invalid (parser errors, unequal work).
3. Any per-turn tok/s read from those replays, including "35.9 tok/s real-session decode" as a headline and the "0.45 ms per 1K positions" slope. Use as diagnostics only.

**Single run or single pair**

4. "Rust is 38% faster at decode than llama.cpp" (30.52 → 42.06). One screen per arm, different arithmetic, different continuations.
5. "Tiled attention: +22% decode at 16K" and the 8K figure. Single pairs. The short-context repeats (flat to +4.5%) and the 6K+2K prefill (+30–32%, three timed reps) are better supported.
6. "Qwen MTP: +30%". One screen, and MTP is off in production.
7. "36% less task time from the llama.cpp patches" (99.77 → 64.00 s) without saying it is one pair with a second reference at 89.90 s, and that the real-controller cached session was flat.
8. Warm-prefix llama.cpp → Rust (30.24 → 23.42 s; 70.54 → 29.33 s). n=1, and the Qwen baseline is an older engine.
9. First prompt 30.6 → 1.2 s from the prefix store, as a general figure. n=1 per model.
10. Rust prefill vs llama.cpp pp512/pp4096. Single ordered screens. If quoted, quote both models: Qwen is 3–4% slower.
11. Thread-retune task gains (S1), Flash IO4 → IO8 (T3), Flash 21 GiB at 112 ms (F6c), Flash microbatch pairs (G1, G2). One pair each.
12. The Flash 1.53× (1175 → 768 s) and 2.2× (1.71 → 3.75 tok/s) changes. Cause unknown. Not ours to claim.
13. Flash MTP +28%. It failed the identity check.

**Below the release run count**

14. Ticket 032 "prefill +19% to +45%". Two reps per arm, Ornith only (installed 2026-11-10), Qwen not checked, and no session measurement with it. Needs three runs and a Qwen pass before it is a headline.
15. Flash static plan −2.85% and executor −0.18%. Both are below the 3% line: report as no gain.

**Estimates**

16. "About 2.5× Ornith decode vs stock" (2.31× × 1.078×). A product of two separate tracks.
17. Any multiplication of rows across ladders, for example MEMLAT × threads × Rust × attention.
18. Session savings from ticket 032 (60–90 s), think-block handling (40–90 s), decode attention (15–20 s), and the 55–90 s re-prefill cost that parking avoids. All estimates.
19. Bandwidth ceilings (48–55 tok/s for Ornith, the 36 ms dense floor for Flash) as achieved or achievable speeds. They are proxies from byte counts, not DRAM counters.
20. Flash targets (≤105 ms/token, 13–17 tok/s with MTP, 50 tok/s prefill). Targets, and the two that were tested were missed.
21. Mac prefill "2–3×" as a measured ratio, and any Mac comparison at all (section 3).

**Scope errors to avoid**

22. "Stock upstream llama.cpp → now" as one measured number. The only unmodified-upstream baseline is llama-bench pp512/tg128 on the MEMLAT boot. Rows D1–D3 and P1–P2 use our pinned build on both sides.
23. "2.0× Flash decode" (246.5 → 123.07 ms) without stating that the candidate is not installed, is reorder-class (KLD 6.55e-4), and differs from production in engine, threads and cache size at once.
24. Kernel-only ratios (1.6–4.8× for Q3_K, 100 → 162 GMAC/s, 19.8 → 56 GFMA/s) as model speedups.
25. Prefix-store or parked-session gains as tokens-per-second gains. They remove work; they do not speed it up.
26. "Flash prefill is faster in Rust than in production." Not shown at any matched length.
27. Bench tok/s as agent speed, or cached prompt tokens as newly processed tokens.
28. Results as valid for every X Elite or X Plus laptop. One machine, and MEMLAT depends on its firmware.
29. Thermal steady state. Runs reached 97–98 °C and the long-session thermal test has not been run.

**Added 2026-11-10 evening**

30. "MTP gives Ornith +7%" (D14). It is a short screen: 53-token prompt, 128 tokens, two passes. Long context is unmeasured, deeper drafts are slower, and it is not installed.
31. "LFU makes Flash 5.5–6.2% faster" (F11) as a release number. Two runs per arm with 5% baseline drift in the block; needs three warm interleaved runs.
32. Any gain from fused phases (F13, −2.08%). Below the 3% line: report as no gain.
33. A cache-size effect read from ticket 035 (its 20 GiB and 18 GiB blocks ran one after the other), or 21 GiB as the Flash configuration: it was refused at load the same afternoon.
34. The session cache (P13b) or cancel (S10) as speedups. One synthetic check each; they remove work in specific situations.

**Added 2026-11-10 late evening**

35. The verify-lane costs (D15: 1.59 / 2.25 / 3.57 plain steps for 2 / 3 / 5 lanes) as a measured property of the engine. They come from one 14.8-second development check on one build, Ornith only. Use them to explain D14; repeat them before quoting them.
36. "Prompt lookup does not work" or "gives up to +3.4%". D16 is a simulation on one recorded session with extrapolated lane costs. The honest statement is: shelved, because the simulated gain is below the bar.
37. "The GPU is useless for LLMs on Snapdragon." P14 measured a plain shader, one shape, one open-source driver version, model-free. The supported statement is narrower: on this laptop and driver, GPU prefill is 30–170× short of the CPU in that benchmark, so we stopped.
38. Any sum over the small-gains stack (1.6), for example "2.08% + 2.85% + …". The stack has not been timed as a whole, and several entries are estimates.
39. "Ornith got faster today" from the install run. Mask 13's prefill gain is P9 (n=2 reps per arm, bench shapes). The install itself was checked functionally (65.1 s, 13/13), not timed against the previous build, and no session replay with mask 13 exists.
40. "LFU is in production" or "Flash runs at 8 threads because of ticket 035" as descriptions of the installed setup. Flash production is still the C++ engine with a 12 GiB cache and 4 decode threads (F3); F11 and F12 are results on the Rust candidate.

---

## 5. What is needed to turn this into release claims

1. One matched run per model: stock device tree + upstream llama.cpp at the pinned commit vs today's installed setup, same day, same prompt IDs, ≥3 warm runs each, decode and prefill separate.
2. A clean batch-1 replay pair (baseline and candidate back to back, nothing else running), ≥3 runs if it is to be a headline, with thermal and CPU-throttle logging.
3. Ticket 032 with a third rep, the Qwen exactness fixtures and Qwen timing, then a session replay with the switches on.
4. Repeat the single-pair rows we want to quote (D6, D8 at 16K, P6, P10, S6) to n≥3.
5. The Qwen replay after the tool-call parser fix.
6. Flash: a matched-length prefill baseline on the production engine, and a session-level run on the Rust engine.
7. Mac rows under the six rules in section 3.
8. One stacked small-gains arm per engine against production (section 1.6), after the larger items are installed.
9. Public receipts for every quoted row, made with `tools/export_receipt.py` (numbers, hashes, argv and hardware only; it refuses to write if its path and secret scan finds anything). The hand-made extracts in `benchmarks/results/runq*.json` still carry private paths and are to be re-exported with it.
