# Rust M3: real-math decode screen (2026-11-09)

Production remains patched llama.cpp. M2 exactness uses native-layout llama.cpp with KleidiAI off; production KleidiAI uses different arithmetic, so cross-engine greedy identity is not claimed here.

## Realistic code continuation

Same synthetic chat-formatted LRU-cache implementation request, 142 prompt tokens, forced 1,024 output tokens, 16K context, 8 packed decode cores (0–7), memlat boot. llama.cpp uses 12 batch threads, F16 KV, FA off and engine `acd47ff1`; Rust uses frozen `e3ec3e84640b` from clean commit `18406e8`, file-backed weights, serial prefill, no MTP. Each arm ran alone under the canonical lease, 28 GiB cap and zero model swap. One screen per arm; no production promotion.

| Arm | Decode tok/s | Prompt time | Decode time | Inference time | Distinct output IDs |
|---|---:|---:|---:|---:|---:|
| llama.cpp MTP off | 30.52 | 0.737 s | 33.514 s | 34.252 s | 346 |
| llama.cpp MTP draft 2 | 25.91 | 0.869 s | 39.483 s | 40.353 s | 346 |
| Rust native | 42.06 | 10.300 s | 24.320 s | 34.620 s | 347 |

Rust's reported decode rate counts 1,023 actual engine steps for 1,024 emitted IDs; llama.cpp reports its own per-step rate (1,023 decode intervals). Rust timing is parsed from its two-decimal CLI timer. Its serial prefill includes first touches; llama.cpp mapped repack payloads were read and a 16-token warm-up ran before the timed request. This measures a steady continuation, not cold loading or a complete agent task. Rust's decode improvement is about 38%, but total inference time is approximately tied; batched prefill is still needed.

MTP accepted 489 / 1,067 drafts (45.8%) and was slower. MTP on/off produced identical 1,024-token sequences in this screen. Rust's native arithmetic follows a different continuation. Most frequent token fractions were below 9% in all arms. No OOM or service swap occurred. Public receipts: real code JSONL (raw rows `benchmarks/roofline/rust-m3-real-code.jsonl`, lab notebook, not in the public snapshot). Synthetic prompt/output and raw logs remain local under `local-ai-optimization/engine-m3-20261109/real-*`.

## Why the earlier 16.44 tok/s baseline is not the headline

The earlier Ornith MTP-off arm used the correct production SHA, 8 decode / 12 batch threads and masks `0xff` / `0xfff`, but context 2K, a 16-token pseudo-random prompt, 16 warm-up tokens and only 128 timed output tokens. It repeated one token almost entirely. Its 16.44 tok/s and perfect MTP acceptance do not represent coding use. The realistic run above recovered the earlier 28–31 tok/s range without changing the engine, thread count or placement. Context, routed workload, warm-up and cache-resume pattern differ; their individual causal contributions have not been isolated.

The frozen Rust bench CLI overloaded `--tokens` as both prompt IDs and generation length. The initial root screen omitted that flag to obtain the intended 16-token prompt and default 128-token decode, rather than silently benchmarking a one-token prompt. Realistic Rust uses `gen --tokens <same prompt IDs> --n 1024`, avoiding the ambiguity.

## Initial fixed decode screen

All use 16 pseudo-random prompt tokens, 16 warm-up and 128 timed tokens, 2K context. These are exploratory single screens, not matched agent-quality claims.

| Model / Rust arm | Median ms/token | Median-derived tok/s |
|---|---:|---:|
| Ornith file, spread | 21.882 | 45.70 |
| Ornith file, packed | 21.732 | 46.01 |
| Ornith file, profiling | 21.664 | 46.16 |
| Ornith anonymous, spread | 21.148 | 47.29 |
| Qwen file, spread | 34.624 | 28.88 |
| Qwen file, packed | 30.197 | 33.12 |

211 barriers/token. Every Rust arm within each model generated the same hash. Anonymous Ornith cost 6.31 s to load and peaked at 27.68 GiB process RSS; file-backed remains the default. The initial llama.cpp fixed screens gave Ornith 16.44 / 41.87 tok/s and Qwen 20.40 / 39.77 tok/s (MTP off/on). Perfect acceptance and repetitive output make them unsuitable as expected application rates. See Rust receipts (raw rows `benchmarks/roofline/rust-m3.jsonl`, lab notebook, not in the public snapshot) and llama receipts (raw rows `benchmarks/roofline/rust-m3-llama-baseline.jsonl`, lab notebook, not in the public snapshot).

## Retained failure

The first realistic attempt overlapped a Claude-lane batched oracle job despite root holding the lease. Root stopped its arm and excluded it; raw artifacts are preserved as `real-llama0-contaminated`. Claude stopped the conflicting job before the three valid isolated arms.

## Support work

Production-library oracle captures cover three synthetic fixtures per model, all 40 layer outputs, greedy IDs and full logit rows. Capture-on/off comparisons are byte-identical; raw data stays in the private local oracle cache. The opt-in OXE controller scaffold is source-only, disabled unless explicitly enabled and given a pinned server/profile. No live runtime, default or service changes were made.

Dynamic memory.low remains uninstalled: parallel-I/O trials had load/fault-pressure confounds and were stopped by the existing guard. A prototype cleanup comparison failed because cgroup values are page-rounded; all ancestor protections were reset to zero, then corrected cleanup was verified on pipe closure and child exit. No speed/stability benefit is claimed.

## Rust MTP: realistic code continuation

Same 142-token synthetic code prompt, 16K context, 8 packed cores, frozen binary 035dead4dd00. 1,024 requested timed tokens, zero warm-up continuation tokens; prompt prefill is outside the decode timer. MTP may evaluate one extra bonus token at the final burst; rates use the reported actual produced-token count. Off uses mean time per step; MTP uses total wall time / committed tokens. One screen per arm, no default changes.

| Model | Draft depth | Decode tok/s | Change vs off | Acceptance | Tokens/verify |
|---|---:|---:|---:|---:|---:|
| qwen | 0 | 33.07 | +0.0% | — | — |
| qwen | 1 | 41.76 | +26.3% | 96.0% | 1.96 |
| qwen | 2 | 43.05 | +30.2% | 90.7% | 2.81 |
| qwen | 3 | 42.88 | +29.7% | 88.3% | 3.65 |
| ornith | 0 | 42.06 | +0.0% | — | — |
| ornith | 1 | 42.86 | +1.9% | 77.5% | 1.77 |

All generated hashes match within each model. Qwen depth2 is the screen winner (+30.2%); depth3 is flat against it. Ornith depth1 is approximately flat (+1.9%), so there is no measured reason to enable it by default. This is a decode screen, not a finished-task promotion. The new binary embeds an older `0817f422f0af-dirty` build label; root records its exact SHA035dead4… and the clean delivery checkout342742b separately rather than treating the embedded commit label as verified provenance. Receipts (raw rows `benchmarks/roofline/rust-m3-real-mtp.jsonl`, lab notebook, not in the public snapshot).

### llama.cpp follow-up

Qwen3.6 on the patched production binary, MTP depth2, same 142-token prompt and 1,024-token continuation, gives **33.65 tok/s**. It accepts653/738 drafts (88.5%). Native Rust depth2 gives43.05 tok/s, about 28% higher on this code request, but arithmetic and resulting continuations differ; this is not an identical-output cross-engine speedup or a completed-task result.

A separate Ornith trace-only diagnostic (not the latency headline) reproduces the original MTP-on output hash and acceptance489/1067. llama.cpp reports accepted-at-position rates **75.7% first,15.9% second** per verify step; conditional second-given-first is about21%. The weak chain also occurs in llama.cpp. Its trace logging makes timing unsuitable as a clean production comparison; use the earlier 25.91tok/s result. Only timing and per-position lines were retained, not the full trace stream. Receipts (raw rows `benchmarks/roofline/rust-m3-llama-followup.jsonl`, lab notebook, not in the public snapshot).

## Batched prefill: measured same-token reference

Frozen Rust prefill binary `0847355e1abe` (delivery checkout `9833695`), one warm-up and three timed reset prefills, ubatch 256, context 8,192, same deterministic pseudo-random token IDs. Production llama.cpp reference is `acd47ff1`, 8 decode / 12 batch threads, F16 KV and FA off. Reference uses native `/completion` with one output token and no prompt-cache reuse; table reports prompt processing only. Arithmetic differs (KleidiAI on in production, native in Rust), so matching first tokens are a sanity check rather than full logit identity.

| Model | Prompt | Rust t8 | Rust t10 | Rust t12 | llama.cpp tb12 | Rust t12 vs reference |
|---|---:|---:|---:|---:|---:|---:|
| ornith | 512 | 178.9 | 206.1 | 208.7 | 193.9 | +7.6% |
| ornith | 4,096 | 151.3 | 170.7 | 173.5 | 162.9 | +6.5% |
| qwen | 512 | 184.9 | 210.2 | 214.1 | 222.3 | -3.7% |
| qwen | 4,096 | 154.2 | 174.3 | 175.7 | 180.6 | -2.7% |

First-token IDs are identical across all Rust thread arms and both engines for these prompts. All raw samples remain in the receipts, including a slow first Qwen t8 pp512 sample after model switching; no selective sample removal. The earlier ~650 tok/s verify-run estimate is not supported by this screen and is withdrawn pending Claude-lane reconciliation. Ornith gains about 7%; Qwen is about 3% slower at the current 12-thread settings. These are single ordered screens, not an ABBA or an end-to-end promotion verdict. Keep production unchanged; test the actual server/tool/protected-task path. Rust receipts (raw rows `benchmarks/roofline/rust-m3-prefill.jsonl`, lab notebook, not in the public snapshot), same-token llama reference (raw rows `benchmarks/roofline/rust-m3-prefill-llama.jsonl`, lab notebook, not in the public snapshot).

## Padded expert remainder follow-up

Frozen binary `7a0023a35772`, padded branch `04c1667`, same token IDs, context, ubatch, 12 threads, one warm-up and three measured prefills as above. Expert remainders use a padded four-row tile instead of GEMV. These are ordered screens, not an end-to-end promotion.

| Model | Prompt tokens | Previous Rust | Padded Rust | llama.cpp | Padded vs llama.cpp |
|---|---:|---:|---:|---:|---:|
| Ornith | 512 | 208.7 | 237.2 | 193.9 | +22.3% |
| Ornith | 4,096 | 173.5 | 183.8 | 162.9 | +12.8% |
| Qwen | 512 | 214.1 | 211.6 | 222.3 | −4.8% |
| Qwen | 4,096 | 175.7 | 182.9 | 180.6 | +1.2% |

Rates are tok/s. Ornith improves; Qwen has no general prefill win. The ~650 tok/s estimate was a misattributed 64-token fixture timing and is withdrawn. Padded receipts (raw rows `benchmarks/roofline/rust-m3-prefill-pad.jsonl`, lab notebook, not in the public snapshot).

The standalone server splits the prompt into a batched prefix and one final single-token prefill. Native oracle captures now match that schedule, rather than comparing against a differently rounded all-batched prompt. Six synthetic 128-token captures (three per model) completed without OOM or model swap; peak cgroup memory was 27.19 GiB. This prepares the reference, not a claim that the integrated server has passed. Raw prompts/IDs stay private. Capture metadata (raw rows `benchmarks/roofline/rust-server-split-oracles.jsonl`, lab notebook, not in the public snapshot).


## Integrated Rust server candidate, 2026-11-10

Three-trial protected-task medians: Ornith 77.70 → 60.42 s (22.2% less); Qwen MTP2 75.31 → 67.42 s (10.5% less). All 13/13 checks pass, native-reference greedy prompts match, no OOM/model swap. Source b477aec remains staged; no production switch. Recommend Qwen MTP off from the separate prompt-cost isolation; emitted-text differences are under investigation. [Full timings, token counts, sampling and limitations](RUST-SERVER-QUALIFICATION.md).
