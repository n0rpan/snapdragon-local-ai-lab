# Flash decode threads: first six-thread candidate stopped

The first six-thread candidate passed the unchanged independent code oracle and lifecycle checks, but generated 358 tokens and a different answer SHA instead of the fixed workload's required 353 tokens. The strict matched-output gate stopped the run after the initial control and candidate. **This is not a completed ABBA, a quality failure or a speedup.** [Retained observations](../benchmarks/results/flash-decode6-output-identity-stop.json).

| Single-arm observation | Decode 4 | Decode 6 |
|---|---:|---:|
| Generated tokens | 353 | 358 |
| Reported decode rate | 3.914 tok/s | 3.916 tok/s |
| CPU seconds per token | 0.7534 | 0.9906 |
| Load-to-cleanup time | 110.351 s | 111.741 s |
| Original code oracle | Pass | Pass |

These unequal-output observations do not support a causal timing ratio. More threads did not show enough promise to justify extra work rescuing the six-thread candidate. The cause of output drift is not established; differences in floating-point evaluation order are only a hypothesis.

Both arms used the same fixed CPU1200 budget, batch6, IO8, cache12 GiB, original model/binary and numerical settings. Their answers and every failed gate remain recorded. EOF, normal process exit, reaping, RAM recovery, quota checks and lease release passed; no hard resource event or own swap occurred. Production stays at its qualified settings. The final bounded eight-thread candidate has now also stopped at the same strict gate, as recorded below.


## Final eight-thread candidate: strict identity stop

The final eight-thread candidate also passed the unchanged independent code oracle and lifecycle checks, but produced **342 tokens instead of 353** and a different correct program. The preregistered lossless gate stopped after A1/B1; B2/A2 were not run. This is neither a completed ABBA nor a code-quality failure, and it does not promote eight threads. [Retained observations and exact source/result hashes](../benchmarks/results/flash-decode8-output-identity-stop.json).

| Single-arm observation | Decode 4 | Decode 8 |
|---|---:|---:|
| Generated tokens | 353 | 342 |
| Reported decode rate | 3.905 tok/s | 3.754 tok/s |
| CPU seconds per token | 0.7537 | 1.2263 |
| Request time | 103.760 s | 102.924 s |
| Load-to-cleanup time | 112.598 s | 111.209 s |
| Original code oracle | Pass | Pass |

The different answers and lengths prevent a matched-work timing comparison; the slightly shorter request time cannot establish a speedup. Increased CPU cost is an observed per-token metric for these particular unequal outputs. The cause of output drift remains unproven.

Both arms retained CPU 1200%, batch 6, IO8, cache 12 GiB, ctx16K/F16, ubatch 4096, original 512 experts/native top-10 and the same binary/model settings. Normal exit, stdin/stdout EOF, reaping, RAM recovery, unchanged quota and lease release passed. Own swap remained 0; all hard memory and PID events were 0. Cgroup peak was 18,792,513,536 bytes, below the 21.5 GiB admission gate; sampled host available RAM stayed above 8.77 GB, exceeding the 2 GiB reserve. Exact resource counters and evidence hashes are retained in the linked result. No further thread rescue or full task follows automatically.


## Whole-process six-core placement: completed, candidate closed

The remaining placement ABBA completed with all four original 353-token answers, identical SHA, independent code oracles, lifecycle and resource gates passing. A allowed CPUs `0-5` (L2-group distribution 4+2+0); B allowed `0,1,4,5,8,9` (2+2+2). This restricted the whole native process, including batch and IO workers. It did not pin only decode threads. [Per-arm metrics, sampled masks and exact evidence hashes](../benchmarks/results/flash-balanced-placement-screen.json).

| Completed ABBA aggregate | A: contiguous six | B: spread six |
|---|---:|---:|
| Generated tokens | 706 | 706 |
| Weighted decode rate | 3.869421 tok/s | 3.768990 tok/s |
| Request time, sum | 208.228 s | 213.066 s |
| Prefill time, sum | 25.307 s | 25.311 s |

B's decode rate was **2.60% lower**, request time was **2.32% higher**, and prefill was effectively flat. Close this candidate without promotion or a full-task run. The experiment used the same six-CPU allowance count, fixed decode 4/batch6, IO8, cache12 GiB and CPU1200 outer quota in both conditions; each native child could use at most six CPU-equivalents. Neither condition is the previous unbound twelve-CPU configuration, so this screen does not establish a causal comparison against that configuration.

Root topology metadata confirms three four-core L2-sharing groups; cache sizes were unavailable. Placement also changes scheduler, IO and batch locality, so these results do not isolate an L2 effect. Each child reported its exact mask before native exec; 250 ms watchers checked live native task masks and joined after exit. These are sampled readbacks, not proof about every instant. All four arms retained EOF/reaping/RAM recovery; own swap and hard memory/PID events stayed zero, the quota was unchanged and the canonical lease was released. Detailed counts and caps are retained in the result.
