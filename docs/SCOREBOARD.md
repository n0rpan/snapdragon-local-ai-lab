# Current-boot throughput scoreboard

Both MoEs now use installed bbc2990 with VL15, 32K context and MTP depth 1 without a context cutoff. In matched depth0/1 ABBA screens, Ornith decode time fell 11.9–26.0% across 4K–32K; Qwen fell 18.1–23.1% across 2K–16K. Final protected tasks, installed-unit regressions and zero-swap guards passed. [Exact scope, controls and rollback](MTP-CONTEXT.md). Historical tables below retain their original tools and dates.

Ornith candidate 2 is installed as of 2026-11-11. The matched fixed 2K-token controller request fell from 13.267 to 10.164 s (−23.4%). VL15 lowered short-prompt MTP decode time by 21.4%, with identical generated hashes. Both protected tasks and the installed-unit regression passed. [Configuration, controls and measured limits](lab-notes/MOE-CANDIDATE2.md).


Updated 2026-11-10. Yoga Slim 7x 83ED, BIOS NHCN62WW, kernel 7.3.0-40-omarchy, opt-in SCMI memlat boot. Measured CPU read ceiling: **115.85 GB/s**, three alternating repetitions. Ornith is now the approved everyday/OpenCode default; Qwen remains selectable. Service cap is 28 GiB with model swap disabled; see the promotion checkpoint below. [Methods and raw results](ROOFLINE.md).

## Wave 1 diagnostics — 2026-11-10

Production remains Rust oxe `649ac806`; the Flash fast candidate is **not qualified**. These are diagnostic paths, not equal-work agent-session headlines.

| Ticket / path | Result | Status |
|---|---|---|
| 001 exact Flash, 16 GiB, 4/6 threads | 353/353 identical; 193.91 ms/token; prefill 208.28 s; peak 28.26 GB | Reference passed |
| 002 original 7c51121, fast streamed prefill | Aborted before logits: per-token cap applied to 156-token batch | Runtime bug; fixed in 38d7fd9 |
| 002 corrected, 20 GiB, 8/8 threads | KLD 6.5453e-4; top-1 99.72%; 142.05 ms/token (profile arm); prefill 12.89 s; peak 26.84 GB | Failed original 5.2e-4 gate; passes the owner’s revised numeric bounds; code passed; speed gate unmet |
| Flash M3 six-run timing, candidate | 20 GiB: 122.75/123.07/123.68 ms/token; 18 GiB: 136.29/138.28/136.12; same 342-token answer SHA in all six | Numeric/code passed under revised bounds; 105 ms speed gate unmet |
| 006 prefill, exact state vs mapped | 2K streamed20.39/20.49 tok/s; 5.5K18.96/17.64; optional20GiB long context hit host floor | State gate passed; ≥50tok/s target unmet; profiler captured |
| 005 Ornith/Qwen head-input capture | 3,570 F32 result_norm[2048] vectors; all six greedy sequences identical to native oracle | Correctness passed; DEV timings excluded |
| 004 server/bench decode split | Ornith same-ID streaming 23.02 vs bench 22.60 ms/token; Qwen 31.72 vs stable warm bench 29.77 | Diagnostic 1.85% / 6.57% gaps; first-use variance prevents a blanket stable-gap claim; no promotion |
| 008 parked sessions and 32K RAM | Three real-model exactness tests pass; varied 32K peak RSS Ornith 15.12 GiB, Qwen 19.84 GiB, Qwen parked 20.71 GiB | No model swap/OOM; Ornith warm 16K/32K decode is flat (22.608/22.581 ms, −0.12%); no promotion |

004's host copy/sample/SSE costs only about 0.3–0.4 ms/token; the larger swings are engine-side. Both initial and single required repeat are retained, including unstable first requests. Chat uses a different prompt and is not a matched-ID transport comparison. Static byte estimates distinguish native oxe weights from KleidiAI expansion and allocated KV from actual used context. [004 split and safety receipt](../benchmarks/results/runq004-server-bench-split-20261110.json).

008 used 32,168 varied synthetic input IDs for the memory stress. The earlier repetitive probe understated expert residency. Its DEV wall times include builds and documented CPU pauses and are not speed measurements. Eight checkpoints were configured for the parked arm; raw completion does not populate every possible chat-turn checkpoint. [008 exactness, RAM and initial timing receipt](../benchmarks/results/runq008-sessions-context-20261110.json).

The exact/fast rows differ in cache size, threads, numerical path and profiling. Do not multiply them into a cumulative gain or claim deployment. Receipts: [001](../benchmarks/results/runq001-flash-m3-exact-20261110.json), [002 runtime failure](../benchmarks/results/runq002-flash-m3-prefill-abort-20261110.json), [002 numerical screen](../benchmarks/results/runq002-flash-m3-kld-20261110.json), [005](../benchmarks/results/runq005-head-dumps-20261110.json).

## Target-only decode and prefill

Native llama-bench, CPU-only qualified kernels, repack enabled, f16 KV, batch/ubatch256; three alternating repetitions. MoE decode uses 64 generated synthetic tokens; the bounded dense baseline uses 16. Baseline and candidate are on the **same memlat boot**, so these ratios attribute thread/placement tuning only.

| Model / GGUF | Four unpinned target tok/s | Best target tok/s | Change | Tensor sweep GB/token | Proxy GB/s / read ceiling | pp512 at best decode threads |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3.6-35B-A3B MTP UD-Q4_K_M | 24.08 | 26.88, eight packed | +11.6% | 2.626 | 70.6 / 60.9% | 145.67 tok/s |
| Ornith-1.5-35B-A3B APEX-MTP Compact | 24.60 | 30.56, eight packed | +24.3% | 1.797 | 54.9 / 47.4% | 93.57 tok/s |
| Qwen3.8-27B Q4_K_M | 5.151 | 5.405, six unpinned | +4.9% vs four unpinned | 15.821 | 85.5 /73.8% | pp128:21.65; pp512 pending |
| Arete-27B Q4_K_M | 4.929 | 5.358, six unpinned | +8.7% vs four unpinned | 16.477 | 88.3 /76.2% | pp128:21.78; pp512 pending |

Flash has a separate post-memlat fixed353 CLI baseline: **4.057 tok/s**, exact answer and independent oracle pass. It is not comparable to this target-only llama-bench table. [Streamer prerequisites](FLASH-MEMLAT-PREREQUISITES.md).

Tensor sweep is descriptor-derived target weights, including the separate output head, excluding the full input embedding and MTP block. It is **not a DRAM counter**; KV/activation/cache traffic is omitted. Never multiply an MTP accepted-token rate by this target-only proxy. The native benchmark does not use MTP. pp4k and pp16k remain outstanding on the new boot. Results across different models are not quality rankings.

## Why Ornith has both 30.56 and 17.18 tok/s

These numbers measure different execution paths. Both use the original Compact weights and qualified CPU kernels; neither is a protected-agent result.

| Measurement | Tool/API | Context capacity | Input / n_gen | Decode / batch threads | MTP | Median decode |
|---|---|---|---|---|---|---:|
| Stock-boot control | llama-server `/completion` |4096|128 fixed input /64 generated|4 unpinned /10|native MTP2, pool3|12.73 tok/s|
| Same control after memlat | same server and request harness |4096|128 /64|4 unpinned /10|native MTP2, pool3|17.18 tok/s|
| Memlat target-only thread winner | llama-bench-cpu-v2 |requested64, engine rounds allocation to 256|no text prefill,64 supplied synthetic decode tokens|8 packed /8|off, target graph only|30.56 tok/s|

The native benchmark times decoding a predetermined synthetic token stream. It does not select generated text or run a draft head. The server measures accepted output tokens from actual greedy completions, including the MTP verification/drafting work. Its4096-token allocation and 128-token starting prompt also differ. Thus30.56/17.18 is **not** a measured speedup from a single change.

Source check: native `to_llama_cparams()` requests `n_prompt+n_gen+n_depth`, and the matching context implementation pads capacity to 256. For pp512 the native capacity is 512; dense/Arete pp128 and tg16 are padded to 256. Server capacity 4096 is explicit. The earlier server request median 4.84 s includes prefill and generation but excludes server/model startup. The repeated server measurements below now cover both 4K and 16K contexts. The protected OpenCode pairs below additionally measure time to a correct finished task; controller lifecycle/cache qualification remains pending.

## MTP and request checks

Ordinary chat coding favoured Qwen MTP on and Ornith MTP off. Across the three different first-response prompts, median decode was 31.86→36.15 tok/s for Qwen (four unpinned MTP2→eight packed MTP2), and 18.52→28.18 tok/s for Ornith (four unpinned MTP2→eight packed MTP off). Outputs matched within each model. Two cases passed initially; the shared ASCII/Unicode validation failure was repaired in one extra turn in both baseline/candidate arms. Complete parse_port repair time was 8.46→7.62 s for Qwen and 13.04→9.72 s for Ornith. These are exploratory short cases, not protected-agent improvements or repeated timing estimates.

The separate matched memlat before/after control is Ornith MTP2 at four decode/ten batch threads: **12.73→17.18 tok/s (+35%)**, prefill 94.74→107.70, request6.30→4.84 s, with three exact output matches. This hardware change and the eight-thread change were measured separately; do not multiply their ratios to invent an unmeasured total.

## Qualification still required

Prompt microbatch 256→512 gave only 0.4% on Ornith and 1.4% on Qwen at 12 threads. Twelve prompt threads reached 121.29/172.87 tok/s at ub256; this is a separate prompt-pool candidate, with thermal/desktop checks outstanding. Dense and Arete short-screen baselines are complete; Flash and long prefill remain; attribute Ornith's remaining compute/synchronization limit before claiming convergence. Final candidates need the protected coding tasks, lifecycle/RAM-release checks and orchestrator approval before production-default promotion. Thermal peaks reached 98 °C; long-term and s2idle stability remain unverified. Historical production/agent/image results remain in the README and their original receipts.


## Repeated real-server control after the Q4_K attempt

Original Compact weights, qualified CPU llama-server, same raw128-token inputs and 64 output tokens as the memlat control, three repeats per prompt. All nine outputs per arm matched the corresponding baseline. These are generation requests, not a protected agent session; startup is excluded.

| Capacity | Baseline4 decode/10 batch, MTP2 | Candidate 8 packed/12 batch, MTP off | Median request before→after |
|---|---:|---:|---:|
|4096|17.275 tok/s|29.382 tok/s|4.902→3.216 s|
|16384|16.542 tok/s|30.123 tok/s|4.985→3.150 s|

Thus the selected settings also reach about 30 tok/s through the actual server at the production16K capacity. The 17.18 historical post-memlat control used four threads and MTP2, while the 30.56 target-only benchmark disabled MTP and used a much shorter context allocation. They are distinct measurements; the table now supplies a matched server comparison.

A sustained test completed 601.30 seconds of decode over 35 identical512-token requests (641.18 seconds wall time). First-three-window median 29.869→last-three29.720 tok/s (−0.5%); no material throughput degradation detected. All output hashes matched. The whole unit's highest sampled thermal reading was97.7 °C; CPU readings at window edges peaked92.8 °C. Reported CPU scaling/current maximum stayed3.4176 GHz; that is not a direct physical clock counter. Raw: server and sustained data (raw rows `benchmarks/roofline/ornith-real-server-and-sustained-memlat.jsonl`, lab notebook, not in the public snapshot).

## Protected real OpenCode pairs, current memlat boot

Original model weights; unchanged13-case protected checker. Baseline is the installed4-decode/10-batch settings; candidate uses 8 packed/12 batch, Qwen MTP2 or Ornith MTP off. Both arms use16K, the same 1200% CPU cap,27 GiB hard memory cap and current guards. All four runs passed 13/13, completed their final response and reaped native children. These are single final pairs, not repeated timing confidence intervals. They use direct llama-server transport; external controller PrefixStore is not exercised yet. Cold means a fresh native process and fresh logical context; OS file cache is uncontrolled.

| Model | Load before→after | Agent before→after | Load-to-finished task before→after | Change |
|---|---:|---:|---:|---:|
|Qwen MoE|34.93→30.18 s|82.20→73.88 s|119.02→106.06 s|10.9% less time|
|Ornith Compact|11.80→4.50 s|93.56→87.56 s|106.37→92.69 s|12.9% less time|

The load differences include cache/order effects and are not wholly attributed to thread tuning. Model choices remain opt-in/default unchanged until full staged lifecycle verification and the owner's approval. Actual run rows (raw rows `benchmarks/roofline/production-candidate-normal13.jsonl`, lab notebook, not in the public snapshot).

The same existing OpenCode logs already record per-turn SDK usage. Qwen reports 6,122 new input tokens,50,028 cached reads and 849 output tokens across 9 turns, identically in both arms. Ornith reports 6,066 new,35,825 cached and 479 output across 7 turns, also identical between arms. These native runs do not show45K **new** input tokens; cached reads count repeated prefix reuse across turns. Physical prefill timings and controller-specific cache behavior still need their own audit. Token-only metadata (raw rows `benchmarks/roofline/production-candidate-token-usage.jsonl`, lab notebook, not in the public snapshot).


## Candidate short-screen qualification (current memlat boot)

All eight real-server screens passed both code checks (with at most one repair) and five chat/tool/structured-JSON checks; native children were reaped. Installed settings are compared with the staged memlat settings, not a stock-hardware boot. Both use the original pinned weights, 16K context, CPUQuota=1200%, MemoryMax=27 GiB, and MemorySwapMax=0.

| Model | Baseline→candidate code decode median | Baseline→candidate code prefill median | Screen wall time | Promoted? |
|---|---:|---:|---:|---|
|Qwen MoE|31.76→38.07 tok/s|130.8→155.4 tok/s|65.37→49.72 s|No|
|Ornith Compact|20.02→30.22 tok/s|102.69→109.38 tok/s|42.84→27.99 s|No|
|Dense Qwen 27B|5.25→5.23 tok/s|30.00→31.31 tok/s|130.20→124.09 s|No|
|Arete 27B|5.11→5.12 tok/s|29.10→31.80 tok/s|134.81→128.99 s|No|

These are medians over the screen's code attempts, not a repeated fixed decode benchmark. Wall time includes model startup and functional checks. Uncontrolled file-cache/order effects confound startup gains; the table does not attribute them wholly to tuning. Dense and Arete show essentially flat decode on this screen despite the native microbenchmark winner. Protected tasks and controller/prefix/lifecycle checks determine qualification. Run data (raw rows `benchmarks/roofline/production-candidate-screens.jsonl`, lab notebook, not in the public snapshot).

Greedy output comparison of the completed screens also matched both code responses and three fixed chat responses in each of the four model pairs (five exact matches/model). No additional inference was needed. Output-hash comparisons (raw rows `benchmarks/roofline/production-candidate-parity.jsonl`, lab notebook, not in the public snapshot).


Dense protected pair on memlat, same original Q4_K_M weights: baseline4 decode/10 batch passed 13/13 in 401.35 s (33.91 s load,365.27 s agent); candidate 6 unpinned/12 strict batch passed 13/13 in 387.73 s (32.64 s load,352.96 s agent). This is 3.4% less total time and 3.4% less agent time in one pair. Both received per-file POSIX_FADV_DONTNEED before launch, under the exclusive lease; this advisory reduces order bias but does not prove identical cold hardware caches. Both report5,983 new input tokens and 651 output, with 29,818 cached reads across 6 turns. No controller/PrefixStore was involved. The gain is modest; the target-only benchmark increase is not an agent-speedup claim. Arete's protected pair and staged-controller lifecycle qualification remain pending.


Arete's protected pair also passed 13/13: stock settings38.18 s load+369.67 s agent =409.94 s total; candidate 36.18 s load+356.72 s agent =395.17 s total (3.6% less total,3.5% less agent time). Original Arete Q4_K_M, same memlat boot and per-file advice policy as dense. All eight protected runs across the four model pairs now pass. These qualify correctness and observed task-time direction; controller/prefix/switch/idle/RAM checks and explicit approval still precede a live profile update.


## Installed checkpoint and new kernel candidates, 2026-11-09

The owner approved and the controller now runs the memlat-aware winners: Ornith8 packed decode/12 strict batch, MTP off, PrefixStore on; Qwen8 packed decode/12 batch with MTP2; dense6 decode/12 batch. Stock-boot fallback values remain separate. Arete is retired and its payload deleted; its rows above are historical. The direct-task and short-screen tables above describe their original measurements, with the promoted settings now live.

Actual controller/OpenCode Ornith passed13/13 protected checks twice: **121.41 s cold /63.15 s cached second session**, both 9 tools. These use the actual controller transport and a different alias/tool trajectory from the direct-server table, so do not compare121.41 vs 92.69 as a matched regression. Controller prefix restore, all 3 retained classic model switches, idle unload after 123 s, lease exclusion/release and guard join passed. Live smoke then answered the fixed arithmetic check correctly. Promotion (raw rows `benchmarks/roofline/ornith-production-promotion.jsonl`, lab notebook, not in the public snapshot), lifecycle (raw rows `benchmarks/roofline/production-candidate-controller-lifecycle.jsonl`, lab notebook, not in the public snapshot).

The installed28 GiB cap passed a largest-resident Qwen MTP check at 16K:15,808 new prompt tokens,512 output,92.96 prefill/22.36 decode tok/s; no OOM kill or model swap. Peak charged current memory27.76 GiB, minimum host availability3.75 GiB, peak 98 °C. Early/late128-token windows21.66/22.86 tok/s. Browser was closed; the owner accepted the concurrent Codex/Claude/subagent load as the worst case and closed the extra browser condition, and there is no matched pre-swap full-context baseline. [Policy and limitations](HOST-MEMORY.md).

Standalone Q3_K kernel ABBA now clears the integration gate with exact outputs: at 4 threads, GEMV1.707× gate/up and 1.887× down; at 8 threads,1.595×/1.840×. Four-row prefill GEMM at 8–10 threads is 3.89–4.47× upstream. This is not a model-level win yet. [Kernel table/method](../experiments/q3k-i8mm/RESULTS.md). The real Q3_K+MUL_MAT_ID GEMM engine is being built separately with unchanged production source/binaries.

## Patched-engine qualification checkpoint (2026-11-09; NOT production)

| Profile | Load | Agent | Total elapsed | Protected checks |
|---|---:|---:|---:|---|
| Refreshed installed1bd engine,8/12,MTPoff |15.07s|83.63s|99.77s|13/13,6 tools|
| Corrected GEMM + Q3 repack + cache-v2,non-mmap,full verification |15.32s|60.60s|76.99s|13/13,6 tools|

One refreshed final pair:22.8% less elapsed time,27.5% less agent time. The older production reference was89.90s, so comparison against that gives14.4%; do not hide baseline variation or call either a confidence interval. Same original Compact weights,16K,8 decode/12 batch,1200% quota/28 GiB/model swap0, existing protected checker, per-source-file DONTNEED advisory. Cache source/load policies differ intentionally. A fresh sidecar is not the same as a cold model-file cache; startup conditions are stated rather than assumed.

The latest cache06f6 candidate uses sampled warm verification (the owner approved), fixes btrfs source identity and releases duplicated source pages. Cold/warm 4096-token checks both give KLD0/top1100%; warm load-to-evaluation11.02s (earlier content/full cache24.78s). These are perplexity contexts512/batch256, not server startup or finished-agent time. A prior16K mmap/cache-v2 run stopped on actual sustained PSI; it remains recorded. Final16K and no-prefetch qualification is pending. The installed engine remains1bd; do not count unpromoted kernels/cache gains in the headline stock-vs-now table.

### Final cache/no-prefetch candidate (still staged)

Final engine `acd47ff19a16` adds corrected GEMM, Q3 packing, sampled warm cache verification, btrfs file identity, source release and no source prefetch. Original Compact weights and 16K/8decode/12batch/MTPoff remain. Final4K numerical gate: KLD0/top1100%. Protected task: **load 4.02s + agent59.11s, total 64.00s**,13/13 checks,6 tools. Refreshed installed reference:15.07s +83.63s, total 99.77s; **35.8% less total time,29.3% less agent time**. Against the earlier89.90s reference the saving is28.8%; baseline variability is retained. These are successive single completed runs, not ABBA timing confidence intervals.

The sidecar was already populated by the numerical run. The model source received per-file DONTNEED advice; that does not make the separate sidecar physically cold. Cache creation is a one-time cost: the no-prefetch candidate's fresh-cache512-context KLD load was20.28s vs 24.02s without that loader patch. Context and tool differences prevent using these as the real16K cold-server load. Current installed1bd remains unchanged pending actual controller qualification and orchestrator approval. Final raw rows: cache/quality/task (raw rows `benchmarks/roofline/q3k-moe-cache.jsonl`, lab notebook, not in the public snapshot).

## Installed patched Ornith engine (2026-11-09)

The owner approved deployment after actual controller qualification. Installed engine `acd47ff19a16` is the final Q3_K/GEMV-order-MoE-GEMM/parallel-pack/cache/no-prefetch build. The actual live `/proc/<pid>/exe` digest, command line and selected cache environment were checked:8 decode/12 batch,16K,mmap,MTPoff,GEMM=1,Q3repack=1,VERIFY=once,DROP_SOURCE=1. Arithmetic smoke passed. The source-selected old1bd engine and 4/10 policy remain unchanged when MEMLAT is absent; Qwen/dense profiles are unchanged.

Direct protected task99.77→64.00s (−35.8%) is installed evidence, not a claim that every session accelerates36%. Actual controller9-tool task: cold 121.41→113.77s (−6.3%); cached 63.15→64.56s (+2.2%, effectively flat at this sample size). All 4 controller sessions pass 13/13 and have identical SDK token volumes:864 new input +1,104 output across 9 turns, first turn 5,487cached/62new. The direct6-tool task has 6,066 new input +479 output. Saved-prefix priming occurs before the SDK's first request; cold/cached timing differences include that work. The remaining controller time is not fully phase-attributed. Do not conflate cached reads with new prefill or infer task time from native tok/s.

Two clean controller unloads join the guard and cache-error reader;0cache errors,OOMs,service swap or cgroup max events. Idle unload observed 120.12s after cached-client completion, with a free canonical lease.280 source tests pass. Cache-error logging retains only bounded native cache ERROR metadata; prompts and responses are discarded by the reader. Full integrity checking remains available for debugging; sampled warm checks are the owner's approved btrfs policy.

One-command rollback (old engine and controller retained):

```sh
~/.local/share/ai-stack/rollbacks/latest-ornith-patched/rollback.py
```

Rollback copies the saved user runtime/unit and restarts the controller; it does not load a model. Deployment and actual-process evidence: promotion (raw rows `benchmarks/roofline/ornith-patched-production-promotion.jsonl`, lab notebook, not in the public snapshot), [token audit](../benchmarks/results/ornith-patched-controller-token-audit.json). The profiling summaries under `benchmarks/results/ornith-patched-final-*` include sampling overhead and are not ordinary throughput benchmarks.


## Decode threads after the patches: long workload

Same installed engine, 16K capacity, 128-token synthetic prompt, complete 1,024-token greedy generation, batch threads fixed at 12. Two balanced cycles give four observations per arm; filtering each challenger against the reference gives ABBA ordering. All 20 outputs are identical.

| Decode / placement | Median tok/s | Range | vs 8 packed | Promoted? |
|---|---:|---:|---:|---|
| 8 packed (reference) |28.273|25.815–28.748|—|Existing setting|
| 6 spread (2+2+2) |27.838|26.807–28.398|−1.5%|No|
| 8 spread (3+3+2) |28.529|27.908–28.861|+0.9%|No; within variation|
| 10 spread (4+3+3) |27.396|26.576–27.866|−3.1%|No|
| 12 full |23.117|22.057–23.493|−18.2%|No|

Keep eight packed decode threads; do not use this to lower the separate prefill pool. The first reference sample is slow and retained; dropping it would not establish a spread-placement win. The next test is OpenMP wait behavior and the pthread backend; cycle/frequency profiling at 8 vs 12 is queued to explain the loss. [Summary and counting method](../benchmarks/results/ornith-patched-long-sync.json). A preliminary native tg128 ABBA also found 6 vs 8 essentially flat; its prefill cases used the same thread count as decode and do not select the production batch pool.

### Patched pthread screen (not yet installed)

The 16-run balanced screen at 8 packed decode / 12 batch threads, 128 prompt tokens and 1,024 output tokens measured OpenMP **26.12** tok/s median; pthread poll 0/50/100 measured **27.14 / 29.32 / 29.08**. All 16 output digests match; separate 4K KLD is 0 with 100% top-1 agreement. Poll 50's raw median advantage is 12.2%, but the OpenMP references declined from 27.62 to 20.69 across the screen. Treat that figure as drift-confounded until paired/controller qualification; production remains OpenMP. [Per-run evidence](../benchmarks/results/ornith-patched-pthread.json).

The diagnostic graph has **3,869 nodes per decode**, and the profiled 8/12-thread runs reported 3.42 GHz on all three clusters. libgomp spin grows at 12 threads, supporting a synchronization explanation. These PMU counters do not establish calibrated DDR bandwidth; the perf parser's locale decimal commas were recovered explicitly. [Profile summary](../benchmarks/results/ornith-patched-thread-profile-summary.json).

### Rust M0 timing-only decision: GO, not an inference result

Fresh leased runs with real Ornith weights and approximate model numerics: 8 packed threads **24.35 ms/token**, 8 spread threads **23.16 ms/token** (43.17 tok/s), packed without instrumentation **23.48 ms/token**. The spread arm passes the stipulated ≤24 ms threshold, with **171 barriers/token**, 675 ns/empty barrier and **85.1 GB/s from plan byte accounting**. This justifies a correctness-tested engine prototype; it does not qualify a production model or demonstrate equal quality. Spread load was 5.16 s, process peak RSS 27.42 GiB (including shared source-file mappings), anonymous huge pages 10.56 GiB; cgroup peak 19.15 GiB, no OOM or model swap. Charges and process RSS differ because source pages may be owned by another cgroup. [All three runs](../benchmarks/results/rust-m0-decision.json).

### Corrected pthread repeat: decode gain disappears

After fixing caller affinity, an ABBA ×2 compared installed OpenMP 8/12 against pthread poll 50 8/8: **30.94 vs 30.83 tok/s decode** (−0.35%, flat), **161.69 vs 136.93 tok/s** on the 4K prompt after the first 512-token request (−15.3%). All three fixed-request output digests match across all 8 arms; the separate 4K KLD is 0 / top-1 100%. The first 512-token request is sensitive to lazy weight page faults and is explicitly not a steady compute-only result. The earlier 12.2% decode advantage did not survive this comparison. No promotion; controller cold/cached measurement follows. [Paired results](../benchmarks/results/ornith-pthread-fixed8-paired.json).

The unchanged controller/normal13 client also rejected pthread: cold **70.34 → 83.12 s** (+18.2%), cached **32.75 → 37.95 s** (+15.9%); all 13 cases and six tools passed in both arms, actual engine SHAs observed, clean unload and lease release. Both arms used the same isolated experimental alias; these absolute times do **not** replace the earlier production-alias baseline. Production remains acd47ff1. [Comparison and scope](../benchmarks/results/ornith-pthread-controller-comparison.json).


## Integrated Rust server candidate, 2026-11-10

Three-trial protected-task medians: Ornith 77.70 → 60.42 s (22.2% less); Qwen MTP2 75.31 → 67.42 s (10.5% less). All 13/13 checks pass, native-reference greedy prompts match, no OOM/model swap. Source b477aec is a qualified candidate; deployment is not staged and no production switch has occurred. Recommend Qwen MTP off from the separate prompt-cost isolation; emitted-text differences are under investigation. [Full timings, token counts, sampling and limitations](RUST-SERVER-QUALIFICATION.md).

## Rust tiled attention candidate

Same-binary long-prefix prefill: Ornith 126.23 → 164.00 tok/s (+29.9%), Qwen 126.32 → 166.57 (+31.9%). At 16K decode: 30.53 → 37.30 and 25.43 → 28.93 tok/s. Initial short Qwen losses do not reproduce: repeated 1K +1.3%, 2K +1.4%, 4K +4.5%. These are synthetic native screens, not installed or full-task results. [All pairs, faults and scope](RUST-ATTENTION.md).

## Integrated Rust upgrade, not installed

Three-trial Rust→Rust medians: Ornith 60.06→52.13 s (−13.2%), Qwen 65.53→57.91 s (−11.6%). Fair single warm-prefix sessions versus actual production CPP: Ornith 30.24→23.42 s, Qwen 70.54→29.33 s. Cold prefixes are compared separately. Qwen production still uses the older 1bd engine, so the large deployment delta includes its startup/repack cost. [Versions, phase attribution, memory, disk and scope](RUST-SERVER-UPGRADE.md).


## Rust production installed (2026-11-10)

Approved `9d9ccbe` / server `649ac806` is installed on MEMLAT: Ornith default, Qwen selectable, both MTP off, tiled threshold 0, persistent prefix store. Both chats and the real Ornith OpenCode task pass (7 tools, 13 checks, 55.39 s functional smoke). Existing matched qualification numbers stay unchanged. Hard guards active, model swap zero; dynamic memory.low remains separate. [Deployment and rollback](RUST-DEPLOYMENT.md). No Flash prototype timing is promoted as a model result.
