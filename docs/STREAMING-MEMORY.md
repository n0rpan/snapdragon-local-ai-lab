# Streaming cache accounting and transparent huge pages

Flash-Next support is an experimental source preview. It remains unqualified for a complete coding-agent workflow. The following result is a bounded memory finding on a 32 GB Snapdragon X Linux laptop, not a general model-speed claim or a deployed Flash profile.

An 8 GiB expert-cache setting describes a logical resident-slice ledger. It does not cap the process's physical RAM. Dense/shared weights, context and recurrent state, staging, page rounding and file mappings consume additional memory. Admission must use observed process/cgroup memory and host headroom, alongside a conservative estimate.

## Why sparse expert banks can occupy more than their ledger

The reviewed BigMoe engine reserves a virtual bank per layer/projection and fills routed expert slices at fixed offsets. The POSIX allocator uses anonymous `MAP_NORESERVE` mappings; its reservation path does not request `MADV_NOHUGEPAGE`. Reservation size and logical cache bytes are therefore distinct from physically backed pages. [Pinned bank allocation](https://github.com/Helldez/BigMoeOnEdge/blob/930b2c9ebd8c2cface6566bd9f11075ac07adfe0/core/src/moe/expert_stream_source.cpp#L184-L210), [pinned virtual-memory primitives](https://github.com/Helldez/BigMoeOnEdge/blob/930b2c9ebd8c2cface6566bd9f11075ac07adfe0/core/src/io/platform_io.cpp#L307-L345)

Linux documents that touching a small part of a large anonymous region can obtain a huge page and use more memory than the touched base pages require. This makes THP amplification a plausible explanation for sparse banks exceeding the logical cache ledger. It is a source-supported hypothesis about the mechanism, rather than a matched attribution of every extra byte to one allocator. [Kernel THP behavior](https://docs.kernel.org/admin-guide/mm/transhuge.html#objective)

## The matched three-case observation

The same binary, weights, three fixed synthetic answer checks, 8K context, 8 GiB cache, microbatch 128, native expert routing and resource caps were used. The variant disabled THP for the experimental process; all three answers and RAM recovery passed in both runs. [Curated result and configuration](../benchmarks/results/flash-next-thp.json)

| Observation | Inherited THP | Process THP disabled |
|---|---:|---:|
| Minimum host available RAM | 4.66 GiB | 12.61 GiB |
| Three-case model run | 136.56 s | 135.77 s |
| Correct answers | 3/3 | 3/3 |
| RAM recovered after unload/exit |Yes |Yes |

Host headroom improved by about 7.94 GiB; total time was effectively neutral. Whole-job process/cgroup peak stayed around 22 GiB. Runs were sequential, observers had overhead and electrical/thermal conditions were not independently matched, so this is not a broad speedup or an isolated per-bank causal estimate. The export explicitly leaves power-condition verification false.

The THP-disabled observer recorded roughly 8.33 GiB of anonymous resident memory in noreserve banks, with zero `AnonHugePages` in those banks. Those are this run's sampled aggregates. Separate prefill VMA snapshots with 922 and 350 tokens helped identify the sparse-bank issue, but different workloads/sample times are **not** a matched bank-before/bank-after experiment. Do not subtract those snapshots to manufacture a THP-saving figure. The separate [350-token microbatch comparison](../benchmarks/results/flash-next-prefill.json) measures a different optimization.

Cold startup is another phase: the reviewed llama mapping path uses `MAP_POPULATE` when prefetch is requested without lazy ranges. Observed startup file-backed memory dominated the roughly 22 GiB whole-job peak, even when active anonymous bank memory improved. Inspect after-ready anonymous memory and headroom separately from the maximum across load, inference and unload. [Pinned mapping path](https://github.com/Helldez/llama.cpp/blob/7ac98f4a8089a36c0581b7c6e8973417a8fa175c/src/llama-mmap.cpp#L480-L497)

## Optional Linux process-scoped exec tool

[`tools/exec_no_thp.py`](../tools/exec_no_thp.py) is original MIT-licensed code, copied byte-for-byte from the tested candidate wrapper. It is not automatically installed, selected or added to a production launcher. It verifies an absolute resolved executable path and SHA256, disables THP in its own wrapper process, verifies the kernel flag and replaces itself with the requested command through `os.execv`.

```sh
python3 -I -B tools/exec_no_thp.py --sha256 EXPECTED_SHA256 -- /absolute/resolved/executable arguments
```

The command and its descendants inherit the process flag across exec. The calling shell, other processes and global THP controls remain unchanged. The separate classic dense profiles are unaffected. Dense weights inside the selected streamed process also lose THP, so a memory benefit need not be a speed benefit. No shell or `preexec_fn` is used, and PID/process-group identity is preserved. [Kernel process controls and inheritance](https://docs.kernel.org/admin-guide/mm/transhuge.html#process-thp-controls)

Portable Linux tests use only a separate 16 MiB anonymous-memory probe, executable-hash refusal and import/hash checks. They load no models and leave their own test parent/global settings unchanged. The prepared Flash integration remains disabled after the bounded agent trial missed its deadline. Native deployment would require a future successful qualification; it is not part of the verified core stack.

## Proposed narrower follow-up

A targeted `MADV_NOHUGEPAGE` policy on sparse expert-bank windows could retain THP for dense allocations while avoiding bank amplification. This is an upstream proposal to test, **not implemented or proven here**. It needs allocator/page-ownership review, correctness checks, measured physical memory and completed-task timing before adoption. No upstream issue, post or publication is implied by this local preparation.
