# Flash hot-expert protection: no useful gain

Protecting up to 25% of the same 12 GiB cache passed every exact-output and lifecycle check, but did not improve this fixed workload. Complete wall time changed **0.52%** and decode throughput **-0.60%**. Those small differences are consistent with a flat result; they do not justify promotion. [Numeric evidence](../benchmarks/results/flash-hot-cache-screen.json).

| Order | Protected fraction | Complete wall | Native decode | Native cache hit |
|---|---:|---:|---:|---:|
|a1|0%|110.126 s|3.927 tok/s|77.8%|
|b1|25%|111.012 s|3.896 tok/s|77.9%|
|b2|25%|111.194 s|3.877 tok/s|77.9%|
|a2|0%|110.924 s|3.893 tok/s|77.8%|

The same compiled engine, IO8, 4 decode / 6 batch threads, 16K context, microbatch 4096 and diagnostics were used throughout. All four 353-token answers match the original code hash and independent oracle; all 512 experts and native top 10 remain intact. Protected payload stayed within 3 GiB, inside the original cache. Own swap and max/OOM/PID events stayed zero; all workers exited, RAM recovered and the lease released.

Before this ABBA, the known engine and patched engine with protection disabled passed a separate exact-answer/effective-policy parity gate. All 41 native regressions also passed. Those prerequisite passes establish correctness evidence, not performance.

HOT25 produced a small hit-rate/read-byte change and about 1.42 million eviction scans per arm. The extra retention/bookkeeping did not yield a useful time reduction. The policy stays disabled; no full-agent run is planned for it. This fixed single-prefill screen does not settle every possible multi-turn allocation policy, but the current candidate has not earned more benchmark time.
