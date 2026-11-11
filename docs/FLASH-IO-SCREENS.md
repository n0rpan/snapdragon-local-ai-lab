# Flash I/O screens and current prerequisites

Both settings were tested separately on the same rebuilt binary after its 43 native tests and fresh original 353-token OFF parity passed. Decode 4/batch 6, CPU 1200%, IO8, cache 12 GiB, ctx 16K/F16/FA, original 512 experts/native top 10 and speculation off stayed fixed. No profile is promoted.

## Adjacent demand-read coalescing: small prefill signal

The OFF/ON/ON/OFF screen completed. All four answers retained the original 353 tokens and exact SHA, passed the independent code oracle, and completed EOF, normal exit, reaping and RAM recovery. Predictive/speculative traffic stayed zero. The two enabled arms exercised 19,560 coalesced pairs in total. [Curated arm metrics, counters and evidence hashes](../benchmarks/results/flash-coalescing-screen.json).

| ABBA aggregate | OFF | ON |
|---|---:|---:|
| Prefill time, sum | 23.071 s | 21.826 s |
| Complete request time, sum | 205.937 s | 203.947 s |
| Weighted decode rate | 3.870485 tok/s | 3.885998 tok/s |

Prefill was **5.396% shorter**, complete request time **0.967% shorter**, and weighted decode **0.401% higher**. The request result is below the 3% whole-request advance gate. It does not justify a standalone full task or promotion. Retain only a conditional prefill lead for an isolated check after the arithmetic configuration is selected; no follow-up starts automatically. Load-to-cleanup observations are retained separately and do not establish a full-agent gain.

The unit peaked at 19,465,940,992 bytes, below 21.5 GiB admission. Actual host reserve exceeded 2 GiB; own swap and hard memory/PID events remained zero. Quota readbacks, cleanup and canonical lease release passed.

## Predictive prefetch: rejected at the byte-waste gate

This screen stopped after A1/B1. Both answers retained the original 353-token SHA, passed the same code oracle and completed cleanup. The candidate then failed its preregistered byte budget; B2/A2 were not run. The failed terminal receipt remains failed. Its accepted-arm list contains A1 only, so the curated record separately binds the saved raw B1 receipt. [Exact stopped-screen evidence](../benchmarks/results/flash-predictive-prefetch-waste-stop.json).

| Candidate byte accounting | Aligned bytes |
|---|---:|
| Successful demand | 75,360,429,536 |
| Successful speculative | 26,719,830,016 |
| Useful speculative | 6,938,206,208 |
| Unused speculative | 19,781,623,808 |

Useful plus unused exactly equals successful speculative bytes. Unused bytes were **26.249% of successful demand bytes**, exceeding the 10% ceiling. Reject this tested setting without a full task, retry or more aggressive prediction. Prediction did not change the actual routing or correctness; this is a resource-efficiency failure. There is no completed ABBA or causal timing estimate.

The candidate performed 16,192 submitted/scored predictive jobs and consumed 9,021 results; failed reads and coalesced-pair counters stayed zero. The unit peaked at 18,804,957,184 bytes, own swap and hard memory/PID events stayed zero, and both children were reaped with RAM recovery and lease release. Those passing lifecycle checks do not override the byte-budget failure.

## io_uring: full native prerequisite passes

The separate io_uring branch now passes its focused fixture and **all 44 native tests**, with zero skips, a 974,819,328-byte peak, no memory/PID events and all children reaped. The owned fixture exercises eight genuine data SQEs before first reap, two files, out-of-order consumption of real CQEs, aligned short-read resubmission, buffered EOF tail, cancellation, two ready waiters and distinct generations. Actual completion-consumption order and later publication-cookie order are checked separately. [Curated native evidence](../benchmarks/results/flash-io-uring-native44.json).

These are generated-file correctness prerequisites. No large-model parity, backend ABBA, inference RAM fit or speedup is qualified. This branch has not been composed with the metadata-fix/repack candidate. Next establish its exact source-supported inference admission and fresh original-model OFF parity before the requested bounded QD8 comparison; preserve all failed build/capture/fixture receipts.

The fresh raw-engine OFF load reached READY, but its complete fixed-workload budget was **24,948,999,968 bytes**, above the **23,085,449,216-byte** cap. It stopped before GENERATE; process reaping, RAM recovery, resource checks and lease release passed. Whole file RSS and owned file cache are conservatively charged separately and may overlap; this is not measured generation RAM use or a throughput regression. That admission hold remains preserved; the later source-page-helper composition below has separate qualification and budget evidence. [Admission hold](../benchmarks/results/flash-io-uring-raw-admission.json).


## Ring plus source reclaim: complete screen is slower

The separate raw ring branch now composes the source-page helper and passes a **derived 45-test qualification** on C92's 40 binaries. The actual native run passed all 45 tests and hook/kernel/helper markers, then failed only its final Testing archive's 4 MiB per-file cap. Its failed receipt remains failed. A separate successful reconciliation executes **zero tests**, preserves the original evidence byte-for-byte with bounded 8 MiB files/16 MiB tree, and completes the omitted artifact postcheck. It does not claim another native run. [Exact qualification chain and READY/budget evidence](../benchmarks/results/flash-ring-source-reclaim-ready.json).

Fresh same-binary ring OFF/ON startup observations both reach READY and complete close, EOF, reaping, RAM recovery, distinct final snapshots, resource checks and lease release. Neither sends a generation request. Source reclaim stays on, speculation/repack/coalescing/prediction off; original 512 experts/native top 10, cache 12 GiB, IO8, ctx 16K/F16, decode 4/batch 6 and CPU 1200% stay fixed. A zero-job clean ring retirement is valid here and proves no generation QD8 activity.

The reviewed fixed 156/max 512-plus-EOS budgets are **22,747,842,336 bytes OFF** and **22,836,565,792 bytes ON**, leaving **337,606,880 / 248,883,424 bytes** below 21.5 GiB with the separate actual 2 GiB host reserve. Both retain conservative 64-head PLE; ON adds 88 MiB beyond measured READY residue. Individual and combined live component gates share 64 MiB drift once, with a separate ON-kernel ceiling relative to OFF. These are bounded engineering envelopes, not full-task admission or measured generation peaks. Allocation references bind 16 unchanged math files plus one exact model-load-only header-boolean equivalence; they do not claim 17 identical files. These budgets authorized the separately released parity and performance screen below; they did not themselves establish a speedup or promotion.

An early expert RANDOM advice candidate is prepared separately, default off and initially limited to raw/noSpec/source-reclaim-on/ring-off. Its fresh all-40 build, both owned fixtures and all 45 native tests now pass. [Native qualification](../benchmarks/results/flash-early-expert-random-native45.json). The startup-only comparison below now measures a memory/load tradeoff; generation remains untested. It has not been combined with this ring axis.

The separately released [complete OFF/ON/ON/OFF screen](../benchmarks/results/flash-io-uring-fixed-screen.json) now passes all four original 353-token SHA/oracle, counter, EOF/reap/RAM and resource gates. QD8 ring engagement is real: each ON arm retires 119,608 original CQEs and exactly matches native demand-byte totals; no worker fallback or discarded/uncompleted bytes. Ring requests are **11.030% longer**, prefill **18.053% longer**, and weighted decode falls **3.900→3.540 tok/s (−9.236%)**. Both matched request pairs lose. Close this setting with no full-task run or promotion; this result does not rule out every future ring implementation.

## Early expert RANDOM: lower startup peak, slower load

The [same-binary OFF/ON startup observations](../benchmarks/results/flash-early-expert-random-startup.json) both pass close/EOF/reaping, exact mapping/outcome, final snapshot, RAM recovery and resource gates. Peak charged memory is **20,543,688,704→10,654,334,976 bytes (−48.14%)**, while native load time is **12.268→16.203 s (+32.08%)**. Sampled file peaks fall 15,033,016,320→5,163,671,552 bytes; last process read counters fall 19,025,408,000→9,165,873,152 bytes, while sampled major faults rise 5,117→241,967. These are two startup observations, not ABBA timing or proof of the mechanism. All 144 inward advice calls and readonly identity/rr coverage pass; virtual advised bytes are not freed RAM. Neither run generates tokens. Retain this as a stability/memory tradeoff pending separate generation correctness and full admission; no production promotion or generation-speed claim.
