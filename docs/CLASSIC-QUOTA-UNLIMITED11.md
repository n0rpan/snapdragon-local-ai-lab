# Classic MoE scheduler screen: unlimited quota on 11 CPUs

One completed ABBA screen found a candidate worth a full protected-task comparison: mean complete wall fell **3.32%**, native prefill fell **10.09%**, and decode was effectively unchanged. This is a short coding screen, not a qualified full-agent improvement or production promotion. The [numeric receipt](../benchmarks/results/classic-quota-unlimited11-screen.json) contains raw receipt hashes and exact model/engine settings.

Both arms used the same pinned Unsloth Qwen3.6-35B-A3B-MTP UD-Q4_K_M weights and qualified CPU server, decode 4, batch threads 10, context 16K, batch/ubatch 256, MTP 2 and SMALL_BATCH_LIMIT 3. Baseline used quota 800%, CPUs 0–11 and CPU weight 100. Candidate used no quota, CPUs 0–10 and weight 50. Nice 15 stayed fixed. These three scheduler changes are a combined policy; the screen does not isolate their individual effects.

| ABBA position | Policy | Complete wall (s) | Native prefill (s) | Native decode (s) | Throttled periods |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | Baseline | 102.344 | 55.177 | 20.566 | 534 |
| 2 | Unlimited11 | 96.520 | 49.214 | 20.452 | 0 |
| 3 | Unlimited11 | 98.733 | 49.802 | 20.460 | 0 |
| 4 | Baseline | 99.613 | 54.945 | 20.351 | 538 |
| Mean | Baseline | 100.978 | 55.061 | 20.459 | — |
| Mean | Unlimited11 | 97.627 | 49.508 | 20.456 | — |

Candidate mean load readiness was slower (27.012 versus 24.923 s), so its wall advantage is not explained by faster loading. All four arms completed the same attempts and output token counts with identical answer hashes for every attempt. The initially failed answer and bounded repair remained counted. All four correctness, owned-process cleanup, RAM recovery and free-lease checks passed. Aggregate throttled time is scheduler accounting and must not be added to wall time or treated as a direct recoverable stall.

The frozen v6 screen retains external power, valid battery telemetry, a stop at 20%, live 2 GiB available-memory reserve, own swap zero and memory/OOM/PID checks. Battery assistance while plugged in is permitted; discharge trend is recorded without aborting. Each arm uses a fresh process/private HOME and exact-model-FD DONTNEED advice, without global cache drops or production prefix restoration. This advice cannot establish completely cold storage. One ABBA with two measurements per condition leaves cache, thermal, power and order uncertainty.

The [batch8 screen](CLASSIC-QUOTA-SCREEN.md) remains a separate result with no native compute improvement. Its outcome is not overwritten by this survivor. The next gate is a fresh baseline and candidate run of the unchanged protected three-module repair, requiring all 12 independent tests, complete final output and full lifecycle/resource checks. The prepared v6 full-task harness binds this exact passed pair and does not select or promote a winner itself. Full-task and normal-endpoint qualification are pending.
