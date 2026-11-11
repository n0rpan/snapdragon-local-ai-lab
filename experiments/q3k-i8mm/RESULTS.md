# Q3_K standalone kernel measurements — 2026-11-09

These are kernel measurements, not llama-server throughput or a production promotion. All 48 runs passed bitwise checks. The standalone value/scale/repack test printed `ALL PASSED (0 failures)`. Four ABBA order-balanced invocations per shape/thread count used the same synthetic expert routes:256 experts,top8, original+repacked buffers about 231 MiB per shape, much larger than cache. No parameters or expert counts change.

| Mode | Shape | Threads / allowed packed CPUs | New GEMV / upstream | New four-row GEMM / upstream |
|---|---|---|---:|---:|
| decode | gate | 1 /0–0 | 1.955× | — |
| decode | gate | 4 /0–3 | 1.707× | — |
| decode | gate | 8 /0–7 | 1.595× | — |
| decode | down | 1 /0–0 | 2.147× | — |
| decode | down | 4 /0–3 | 1.887× | — |
| decode | down | 8 /0–7 | 1.840× | — |
| prefill | gate | 1 /0–0 | 1.962× | 4.676× |
| prefill | gate | 8 /0–7 | 1.941× | 4.232× |
| prefill | gate | 10 /0–9 | 1.971× | 3.894× |
| prefill | down | 1 /0–0 | 2.145× | 4.842× |
| prefill | down | 8 /0–7 | 2.129× | 4.467× |
| prefill | down | 10 /0–9 | 2.046× | 4.000× |

Decode uses 64 tokens,21 timed repetitions and 3 warmups per invocation; prefill uses 512 tokens,7 repetitions and 2 warmups. ABBA reverses the measured kernel order while preserving the old-kernel correctness reference. Reported ratios are the median of four within-invocation ratios; underlying timings/minima remain in the JSONL (raw rows `benchmarks/roofline/q3k-i8mm-micro.jsonl`, lab notebook, not in the public snapshot). Every invocation compares outputs to the old kernel before timing. CPU read ceiling is 115.85 GB/s on the active MEMLAT boot; compiler flags are in each row.

Both gate/down clear the requested ≥1.3× cold-expert four-thread GEMV threshold (1.707×/1.887×). At the production eight-thread count the kernel gains are 1.595×/1.840×. Prefill GEMM gains at 8–10 threads are 3.89–4.47×. These accelerate only the affected operations; whole-model gains, transient repack RAM, load time and protected-task time remain to be measured. The later real-engine patches are separate candidates and are not installed.

The new production profile uses 8 decode / 12 batch threads; the original prototype instructions describing4/10 are historical. Numeric source license: [llama.cpp MIT notice](LICENSE.llama.cpp).

## First real-engine result (v1; not promoted)

With KleidiAI ON, `exact` passes the original4K Compact logits baseline: reported KLD0,100% top-token agreement. The model ABBA (pp512/tg128,3 samples/test, two arms each) improved prefill at12 threads119.29→185.03tok/s (+55.1%) and decode at8 threads28.90→31.16tok/s (+7.8%). At12 decode threads the gain was only0.8%;8 remains faster. The ABBA's first wrapper was stopped during its third model load because it incorrectly reused the first load exemption; completed AB was retained and BA finished with a fresh existing load-aware guard per process. This interruption is recorded in failures.jsonl.

The protected13-case pair passes13/13 with six completed tools in both arms. Production:9.575s load,79.514s agent,89.899s total. Candidate:24.849s load,67.883s agent,94.460s total. Thus warm agent work improved14.6%, but cold finished time regressed5.1%; the15.3s extra repack/load cost prevents promotion. File DONTNEED advice was applied to both arms under the lease; it is advisory, not a guarantee of identical cache state.

Broader v1 GEMM failed KLD0.021678/top1 92.696%. Claude identified Q4_K/Q5_K float accumulation order (per256 vs per32), not an activation-bsums layout bug. The v2 patch uses GEMV-order accumulation for those routed GEMMs and is being requalified. A separate opt-in load patch parallelizes Q3_K 8-row groups without changing packed bytes, with 1 thread by default. These new changes and the planned persistent cache are excluded from installed gains until measured. The original v1 binaries are archived privately as exact-v1; their hashes remain in the run data.

### Corrected GEMM and hardened cache v2 (2026-11-09; candidate only)

The GEMV-order Q4_K/Q5_K fix passes the 4,096-token reference: mean KLD0, top-1 agreement100%, including the broader GEMM=1 mode. Cross-mode graph hashes match at1/4/8/10/12 threads with 8 packing threads and KleidiAI enabled. Cache patch4416edef builds and its unit test passes; the full payload checksum remains enabled in these measurements.

| Cache test | Load-to-evaluation | Total | KLD / top-1 | Result |
|---|---:|---:|---|---|
| Empty cache, 512-token contexts |17.77s|52.32s|0 /100%|310 entries written,15.63 GB,0 errors|
| Warm cache, same contexts |24.78s|53.07s|0 /100%|310 hits,0 errors|
| Real16K server, mmap |—|41.43s before stop|not completed|Sustained actual memory PSI12.1%/10s;28 GiB cap,0 OOM/own swap|
| Real16K server, load-mode none |14.31s|36.06s screen|all coding/chat/tool checks passed|cgroup peak9.25 GiB;310 hits,0 errors|

The minimal sandbox hid the original path shown by `/proc/self/maps`, so the KLD runs used content identity and hashed source bytes. This is not a production-like warm-load claim. The non-mmap screen intentionally also uses content identity. Next: expose the canonical source path in the sandbox, test cheap warm verification and source-page release when available, then the protected task. None of these kernels or cache options is installed in production. Two pre-load launch failures (missing sandbox mount point and invalid load-mode spelling) remain in the raw run log; neither loaded a model. Raw: `benchmarks/roofline/q3k-moe-cache.jsonl`.

### Final-cache core checkpoint

Final cache06f6eca4 fixes btrfs `/proc/maps` vs `stat()` device-number mismatch (source identity previously fell back to full tensor hashing), uses owner-approved sampled warm verification, and can discard clean source pages with `GGML_CPU_REPACK_DROP_SOURCE=1`. Full payload verification remains a debug option. Root cache unit and cold/warm 4K KLD both pass exactly0/top1100%, with 310 entries/no errors. Cold load-to-eval24.02s, warm 11.02s; same512-token perplexity contexts and cache identity, not server load. Real16K core screen with an empty canonical-path cache passed in46.97s including26.88s creation/load, without the prior PSI abort. Optional no-prefetch loader A/B remains pending.

The older content-cache/non-mmap/full-verification candidate completed the protected task in76.99s, versus a refreshed production reference99.77s; both 13/13 and 6 tools. Agent60.60 vs 83.63s. Its first earlier89.90s reference remains visible; do not conceal baseline/load variation. This is a candidate result, not a promoted stock-to-now gain. Perf's first sibling-process attach failed for lack of capabilities; no system permissions were changed. Scoped sudo perf will collect the final profile.

### Final protected candidate, before promotion

The final cache+no-prefetch engine (`acd47ff19a16`) passes4K KLD0/top1100% and the unchanged13-case OpenCode task, with 6 completed tools. Warm-sidecar load 4.0189s, agent59.1104s, total 63.9997s; refreshed installed reference15.0699s/83.6265s/99.7682s. Saving35.85% total and 29.32% agent;28.81% against the older89.90s reference. Keep both references visible. No source/kernel production promotion has happened yet. Core cache16K fixed screen cold 26.88s/warm 13.82s; no-prefetch's cached task loads in4.02s, but that comparison also differs in harness/workload, so it is not a pure loader-isolation ratio. Fresh-cache KLD no-prefetch20.28s vs core24.02s, identical outputs. All sampled warm checks remain enabled; full checks remain available for debugging.
