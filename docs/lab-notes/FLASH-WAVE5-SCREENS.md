# Flash wave 5 screens, 2026-11-11

Production remains c59284a at a 16 GiB cache. These 18 GiB diagnostic screens do not establish a gain on the installed server.

## Router lookahead: keep as an integration candidate

Frozen d919600, plain expert sidecar, compute CPUs 0–7, IO CPU 8, spin fallback. The prior 354-row teacher-forced identity gate passed. This screen used OFF / ON / ON / OFF with two samples per setting:

| Workload | OFF / ON / ON / OFF, ms/token | Time reduction |
|---|---|---|
| Chat, 299 decode steps | 102.099 / 90.398 / 93.020 / 109.304 | 13.24% |
| Held-out code, steps 64 onward | 96.840 / 94.347 / 92.539 / 96.333 | 3.25% |

Pressure-skipped layer share was zero. All eight native arms passed memory and power guards, with no OOM or model swap. This meets the frozen decision rule; integration and its real-task gate remain pending. It is not a three-warm-run release claim. [Numeric receipt](../../benchmarks/results/flash-lookahead-screen-20261111.json).

The first attempt stopped on missing battery telemetry and is excluded. The fresh run completed all native arms, then its summary failed because a runner refactor missed an existing normalization step. Offline reanalysis checks the original native guard hashes, executable identity and frozen thresholds. The original failed-summary receipt is retained; no native run was repeated to repair the summary.

## Prefill policy: 2K exactness passes; long prompt blocked by memory

Frozen 4252846, 18 GiB cache, 11 compute threads. The mirrored C/O/P/K/K/P/O/C sequence ran two 2048-token passes per arm. All saved states were byte-identical, including repeats.

| Decision | Measured speed ratio | Result |
|---|---:|---|
| Resident expert ordering, cold | 0.996325 | Keep on; within the 3% no-regression band |
| Automatic read-ahead, cold / repeat | 0.992056 / 0.989370 | Keep off; no gain |
| Per-phase IO layout | 0.990507 | Keep the single layout |

Ordering reduced repeat-pass reads from 36.63 to 19.16 GiB. The control drifted (66.36 to 61.20 cold tokens/s), so these decision screens do not support a precise throughput headline. [Numeric receipt](../../benchmarks/results/flash-prefill-policy-screen-20261111.json).

The first 5500-token baseline hit the unchanged 1.5 GiB host floor: minimum 1.4203 GiB, scope peak 25.13 GiB, no OOM or model swap. It stopped after 18.11 seconds; the two candidate long-prompt arms did not run. Long-prompt policy qualification therefore remains blocked. A separately labelled retry at the deployed 16 GiB cache is pending; the guard is unchanged.
