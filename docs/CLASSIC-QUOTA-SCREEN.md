# Classic MoE batch-thread screen: no compute-time win

**The completed rerun removes throttling but shows no compute-time win; nothing is promoted.** The original interrupted run follows below.

## Completed rerun with battery assistance permitted

The new V6 A/B/B/A finished all four arms with identical answer hashes for every attempt and the unchanged independent oracles. External power and the 20% battery cutoff remain required; falling charge is telemetry. [Completed numeric record](../benchmarks/results/classic-quota-batch8-complete.json).

| Order / condition | Complete wall | Load/readiness | Native prefill | Native decode | Throttled periods |
|---|---:|---:|---:|---:|---:|
| 1 / baseline | 120.056 s | 42.679 s | 56.455 s | 20.194 s | 526 |
| 2 / batch8 | 110.096 s | 29.537 s | 59.420 s | 20.393 s | 0 |
| 3 / batch8 | 106.892 s | 27.564 s | 58.268 s | 20.517 s | 0 |
| 4 / baseline | 105.878 s | 27.203 s | 57.580 s | 20.703 s | 548 |

The candidate mean complete time is 3.96% lower, but the first control loaded much more slowly. Candidate prefill is 3.20% slower and decode is effectively unchanged (+0.03% time). The closing control finishes faster than either candidate. This does not establish a scheduling performance win; eliminating throttle counters alone is insufficient. Both candidates have zero throttled periods, versus 526/548 and 32.25/35.41 aggregate throttle seconds in the controls. Aggregate throttle seconds are not additive wall time.

All four processes were reaped, monitors joined, RAM recovered and the lease released. Own swap, max/OOM/kill and PID-limit gates passed. Full protected-task validation remains required for any later quota candidate that wins its completed screen. Production is unchanged.

## Earlier interrupted run, retained unchanged

**The earlier A/B/B/A stopped on its stricter power guard.** One baseline and both batch8 arms correctly finished the same two-case task. The second baseline was interrupted before its parse response returned. [Allowlisted numeric record](../benchmarks/results/classic-quota-batch8-incomplete.json).

| Order / condition | Finished or stopped wall | Load/readiness | Native prefill* | Native decode* | Outcome |
|---|---:|---:|---:|---:|---|
| 1 / baseline | 102.893 s | 27.941 s | 54.121 s | 19.952 s | Both cases pass; 1 repair |
| 2 / batch8 | 108.791 s | 32.007 s | 56.015 s | 19.820 s | Both cases pass; 1 repair |
| 3 / batch8 | 106.902 s | 29.801 s | 56.234 s | 20.021 s | Both cases pass; 1 repair |
| 4 / baseline | 94.900 s stopped | 27.217 s | 17.665 s | 8.424 s | Power interruption; unfinished |

*Native phase sums cover returned responses only. For arm 4 these include merge alone; the in-flight parse has no response/timing record. Its 94.900 s is stopped elapsed time, **not** finished-solution latency, and missing work is not zero work. Complete arm time includes loading, failed first answer, original and diagnostic checks, repair, cleanup and recovery.

The three complete arms produced the same 399 output tokens with identical hashes for **every** attempt:165 for merge,114 for the initially wrong parse answer,120 for its successful repair. Merge passed on its first answer; parse required one repair in every complete arm. The unchanged original oracle passed both final cases. This establishes these two synthetic functions under this feedback policy, not universal coding quality or the protected 12-test task.

The candidate finished in 108.791 / 106.902 s versus the sole completed control 102.893 s. That is **no observed finished-time win**, not a measured causal regression percentage: there is no completed closing control, conditions changed with order/power/heat, and load time differed. No result was extrapolated for the stopped arm.

## Same model and numerical settings

Target: **Unsloth Qwen3.6-35B-A3B-MTP, UD-Q4_K_M**, 22,663,387,424 bytes, SHA256 `0b21525e972670ed59e1812e170b27c26355381f0656ecc4e25617ece7dac58b`, publisher revision `5bc3e238d916f48a861bac2f8a1990a0e9b7e98d`. These are the qualified MTP weights, distinct from the ordinary non-MTP file. [Exact model registry](../dependencies/model-artifacts.json).

Both conditions used the qualified llama-server SHA256 `1bd3e439f357365c50fc8390cc2c7e4cde2f86f14fa1771a65f2323a307457b1`: CPU only, decode 4, 16K context, batch/microbatch 256, repack ON, FA OFF, F16 K/V, native draft-MTP maximum 2, LLAMA_CPU_SMALL_BATCH_LIMIT=3, reasoning disabled. OpenMP/GOMP/KMP overrides were unset in the verified running classic controller. No kernels, weights, KV format, expert policy or decode thread count changed.

The only numerical scheduling option changed was batch threads 10 → 8. Both retained CPUQuota=800%, CPUWeight=100, Nice=15, inherited CPUAffinity 0–11, MemoryHigh=24 GiB, MemoryMax=25 GiB, zero swap and 256 tasks. The user cgroup has no delegated leaf cpuset; actual process affinity and every observed native thread were checked instead, with ancestor 0–11 cpuset recorded. Start admission stayed model-size + 3 GiB.

Each fresh process used the same advisory POSIX_FADV_DONTNEED on only the verified model FD under the canonical lease, private HOME and fixed-request cache off. No real prefix store was accessed, model payload rehashed for advice or global cache/sysctl/swap setting changed. Advice is best effort and does not prove completely cold storage. This is a new controlled-loading screen, not the [normal-controller backend-cold/prefix-restored reference](MOE-NORMAL-REFERENCE.md).

## Throttle accounting changed; finished time did not show a win

| Arm | CPU usage (core-s) | Throttled / total periods | Aggregate throttled accounting | memory.high events | Peak charged cgroup |
|---|---:|---:|---:|---:|---:|
| 1 / baseline | 533.056 | 529 / 1026 | 36.730706 s | 56,622 | 24.016 GiB |
| 2 / batch8 | 525.361 | 1 / 1086 | 0.001385 s | 60,850 | 24.016 GiB |
| 3 / batch8 | 523.877 | 3 / 1067 | 0.002601 s | 59,306 | 24.015 GiB |
| 4 / baseline partial | 350.881 | 346 / 854 | 24.145151 s | 53,565 | 24.019 GiB |

CFS throttling was nearly absent in the two batch8 arms, while their returned-request prefill sums were slightly longer. Throttle counters alone therefore did not predict a finished-time win here. `throttled_usec` is aggregate scheduler accounting, **not** exclusive wall time to subtract or add to prefill/decode. CPU/core-seconds, PSI and wall overlap. The peak charge includes file cache; it is not an isolated model-resident footprint.

The dataset retains each returned attempt’s native prompt/decode rate, token counts, request/check wall, original oracle status, draft acceptance and cgroup CPU/throttle deltas, plus bounded numeric 1 Hz power/thermal/PSI/memory/CPU observations. Native decode rates are server-reported token rates (about 19–21 tokens/s here), not end-to-end finished-task throughput.

Memory pressure was present: 24 GiB soft-limit events rose in every arm. Max/OOM/kill and pids.max event deltas stayed zero; sampled own swap stayed zero. Minimum host MemAvailable across all arms was 4,672,544,768 bytes (4.352 GiB). All four native processes were reaped, monitors joined, RAM recovered to the existing 1 GiB tolerance with a 2 GiB floor, and the canonical lease was free after cleanup. These cleanup passes do not make the interrupted task complete.

## Power stop and retained correctness failures

External power remained reported online with battery status “Not charging”, but battery percentage fell 100.000 → 99.609 over the 414.217 s sequence. The final arm triggered the existing sustained-energy-loss guard: 0.15 percentage points over 61 s with the qualifying signed-power history. Its in-flight parse request disconnected after controlled stop; native termination ultimately used SIGKILL after TERM wait. That transport exception is attributed to the recorded **power intervention**, not a returned model answer or new oracle failure. No parse answer exists for that request.

Sampled maxima across all readable thermal zones were 96.5 / 96.5 / 96.9 / 97.3°C; these are not asserted to be a particular CPU diode. Signed battery power and percent history are retained without calling the power sign a charger measurement. Delivered charger watts were not established. Higher/unlimited-quota work and the protected follow-up were not run from this incomplete screen.

Earlier failures remain failures and are pinned in the dataset:

- V2 (`7083d52f…`): the one-shot parse oracle failed after merge passed. V2 omitted the failing response/oracle detail, so its exact cause cannot be reconstructed.
- V3 (`9ffa5924…`): retained evidence showed non-ASCII decimal input accepted by the generated parser; the original checker raised AssertionError.
- V4 (`a067f3d6…`): the original and one generic repair answer both failed the unchanged oracle.
- V5 (`f8dc5a9a…`): a separately preregistered workload permits at most one repair. Only an additional diagnostic assertion **message** names the failing input; inputs/assertions stay unchanged, and final acceptance re-runs the full original oracle. The failed first attempt remains failed; all its cost is counted. No reference implementation/helper hint is supplied, and remaining original inputs still reject a hardcoded one-input workaround.

The current source manifest is `f8dc5a9aa87fd55035a8bcc5d7a1c01216b9aa01aeb00ed895c87945bffd30bd`; raw incomplete pair receipt SHA256 is `aac803409bd4a3cff0f2f84c1ae54a038039f5e891fcd31ebdef5e8e898b8f58`. Full raw arm, original/diagnostic oracle and historical receipt hashes are in the numeric record. These are identity pins, not a bundled portable rerun command: this curation contains no prompts, generated code, response/tool payloads, user paths, PIDs, histories, auth or cache contents. The full-task gate rejects this incomplete pair. No installation/default changed or push occurred.
