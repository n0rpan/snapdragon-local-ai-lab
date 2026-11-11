# Classic MoE quota: 800% versus 1200%

The completed fixed-screen ABBA reduced **prefill time 10.23%** and **model-request time 7.55%**. Decode was effectively flat. Raising the quota removed request-time CFS throttling while preserving every attempted answer. This is a screening survivor, not a deployed setting or a completed-agent speedup. [Numeric receipts](../benchmarks/results/classic-quota1200-screen.json).

| Metric, mean of two arms | Quota 800% | Quota 1200% | Change |
|---|---:|---:|---:|
| Native prefill | 54.148 s | 48.610 s | −10.23% |
| Native decode | 19.950 s | 19.895 s | −0.27% |
| All three model requests | 74.113 s | 68.520 s | −7.55% |
| Request-time throttled periods, sum of two arms | 1,039 | 0 | Cleared |

Only the quota changed. Both conditions used the same qualified Qwen3.6-35B-A3B UD-Q4_K_M/MTP package, server binary, decode 4/batch10, 16K context, 256 batch/microbatch, F16 KV, FA off, MTP2/pool 3, all 12 allowed CPUs, CPUWeight100 and Nice15. There were four fresh processes in baseline/candidate/candidate/baseline order. The exact engine/model/source hashes are in the numeric receipt.

The fixture contains two code tasks and permits one repair. Every arm made the same three calls: one task passed first try, and the other passed after its one allowed repair. All attempted answers, including the original failed answer, matched across all four arms. The unchanged final acceptance oracles passed; checker, diagnostic and repair costs remain counted.

**Loading confounds complete-wall comparisons.** Arm totals were 186.755, 97.941, 99.703 and 103.399 s; their load times were 112.595, 28.752, 30.389 and 28.420 s. The apparent 31.88% mean complete-time reduction is not a causal quota gain. Exact-file advisory cache release was applied in every arm, but it cannot guarantee identical cache/ownership conditions. The first control's paging motivates the separate loading-mode investigation, not an explanation established by this experiment.

All four processes exited normally and were reaped. Guards joined, RAM recovered, and the canonical lease was verified free after each arm. Peak unit memory was 24.0195 GiB; own swap and max/OOM/PID-limit events were zero. Memory.high events and measured pressure remain in the receipts. External power, the 20% battery cutoff and actual 2 GiB host reserve were retained; battery assistance was allowed and recorded. Scheduler throttle accounting is not an additive wall-time phase.

The earlier [unlimited/11-CPU screen](CLASSIC-QUOTA-UNLIMITED11.md) found a similar prefill benefit, but also changed affinity and weight and ran separately. These results support removing the quota/thread mismatch; they do not establish which policy is faster. Use a corrected scheduler for the finite thread/loading screens, then validate the selected combined configuration on the unchanged protected task and installed lifecycle before promotion. Long task comparisons remain deferred under the user's current sequencing.
