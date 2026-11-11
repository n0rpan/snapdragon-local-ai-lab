# Flash Rust M0: authoritative timing only

2026-11-10. Rebuilt clean `flash-m0` branch `bf97cfdceee4de00a3380001aabda8c41f29a177`; binary SHA256 `f3ff478152a3048ae831ac1d1e101b95153e6ba28c72b974aa8e87b708c1913b`. Seven synthetic/trace tests pass. Each native run uses the supplied lease-owning helper and a 28 GiB, swap-disabled scope. Models/servers are stopped; no concurrent heavy work. Three runs per arm, rotating order, same boot.

**M0 uses dummy activations and replayed routes. It does not generate a correct answer.** Real weights, kernels, cache traffic and SSD reads are timed; its output token IDs have no quality meaning. The predictor is simulated, not learned or measured on hidden states.

| Arm | Median ms/token | Median-derived tok/s | Run-median range | Median run mean | Peak scope RAM |
|---|---:|---:|---:|---:|---:|
| none | 65.26 | 15.32 | 65.05–65.26 ms | 65.80 ms | 20.55 GiB |
| demand | 118.74 | 8.42 | 117.36–119.13 ms | 121.15 ms | 20.56 GiB |
| lookahead | 88.83 | 11.26 | 87.99–90.26 ms | 94.51 ms | 20.53 GiB |

Decode replays 353 trace tokens, excluding the first eight from statistics. Eight spread CPUs `0,1,2,4,5,6,8,9`, IO core 3, 16 GiB per-layer CLOCK expert cache warmed from prefill routes, dense vecdot, N-gram page cache 64 MiB, default IO QD32/prefetch QD16. Lookahead: recall 0.9, depth 2, false-positive reads **off**. The rebuilt commit label matches the frozen source; none of the earlier indicative timings is substituted into this table.

Demand-only median is **118.74 ms (~8.42 tok/s)** versus correct BigMoeOnEdge's **246.5 ms (4.057 tok/s)**: about 2.08× in this timing prototype. The optimistic lookahead median **88.83 ms (~11.26 tok/s)** meets the 90 ms line, but one run's median is 90.26 ms and its actual predictor assumptions remain unproven. Its worst token is **727 ms**, and median run mean is 94.51 ms: do not hide tails behind a median-derived throughput number. Compute-only is a lower-bound diagnostic, not a valid serving configuration or independent gate win.

The demand arm exposes ~55.1 ms/token SSD wait; optimistic lookahead ~19.5 ms, while dense compute rises from ~47.9 to ~53.4 ms. Resident-at-demand hit rates are 87.3% / 96.0%; both read about 118.2 MiB/token. The higher lookahead hit rate includes prefetched experts, not just extra cache capacity. All arms use 484 barriers/token. Component medians across runs are not expected to sum to a run's median wall time.

**Decision: proceed to exact-model implementation, not deployment.** Demand-only supports the architecture without relying on a predictor; 90 ms depends on optimistic lookahead. Required next work: real activations/routing/state; exact fixed353 output identity and independent code oracle; real prefill and protected task. Measure actual two-layer router recall and false-positive IO cost before claiming the lookahead rate. No dropped experts, model-quality claim, full-agent speedup or production switch follows from M0.

All nine native runs exit successfully, swap is disabled by the scope, memory peak stays below 20.56 GiB and the kernel experiment window has no OOM/panic. The usual production Flash admission and numerical policy are unchanged. No model or repacked sidecar payload is included in the repo.

Reproduce from the Flash engine worktree (build first at nice 19, two jobs, under the lease): run `tools/run_flash_m0.sh --io none`, `--io demand`, and `--io lookahead --recall 0.9 --depth 2 --false-pos off`, three times each. That helper owns the lease; do not wrap it in a second flock. Never overlap downloads, cache deletion or builds with these timings. [All nine numeric receipts, breakdowns and IO latency](../benchmarks/results/flash-m0-authoritative.json), [correct baseline and trace](FLASH-MEMLAT-PREREQUISITES.md).

Follow-up: [cache/QD/placement screens](FLASH-M0-TUNING.md) retain 20 GiB; QD/crypto ABBA gave no reliable gain. These remain timing-only, not deployed inference.
