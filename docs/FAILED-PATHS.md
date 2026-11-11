# Disabled paths and what the measurements establish

The deployed core uses qualified CPU profiles. These experiments explain why the GPU, large-model NPU and losing dense DFlash/ngram configurations remain disabled. They are results from one Snapdragon X Elite laptop with 32 GB RAM, not universal hardware limits. The [allowlisted numerical record](../benchmarks/results/disabled-paths.json) contains scalar metrics and source-file hashes; it excludes prompts, answers, raw logs and local absolute paths. Evidence names below identify the original research files, which are not shipped wholesale in this public repository.

## Vulkan: slower 27B prefill; old cleanup failure fixed

On the same 525-token synthetic prompt, Qwen3.8-27B Q4_K_M processed **27.22 tokens/s on CPU versus 16.93 on Vulkan**. The GPU configuration offloaded 64 repeating layers; the output head could remain on CPU. Both used 8K context, six decode/ten prompt threads, batch 256, microbatch 64, F16 KV and flash attention off. The GPU runtime ported upstream PR31045 plus a packed-half safety correction; this was not an unmodified upstream build. GPU code generation measured 2.33 tokens/s, but the older roughly 2.76 tokens/s CPU decode sample changed power conditions and is not a controlled pair.

The original GPU test retained **5,338,804,224 bytes** of global shared memory after exit. **That cleanup failure was subsequently fixed:** the cleanup candidate's 4B load/swap cases left +20,480/+16,384 bytes; its 27B load/swap cases measured −3,813,376/−5,476,352 bytes. All four passed lifecycle safety checks with zero VM_BIND errors. Negative global deltas reflect other memory release, not negative model allocations. The 27B coding check passed; the 4B coding output failed one mutation check, separate from lifecycle safety.

GPU remains disabled because the measured large-model configuration was slower and GPU work interrupted desktop responsiveness. The pauses were observed but not quantitatively timed here. The cleanup-patched swaps did not repeat a matched CPU/GPU speed comparison. The fixed leak must not be cited as a current blocker.

Evidence: `llm-gpu-tests/comparison.json`, `llm-gpu-tests/RESULTS.txt` (historical pre-fix account), `llm-cleanup-model-tests/swap-summary.json` (supersedes its cleanup conclusion).

## Dense speculation: correct repair, longer finished time

One cold isolated OpenCode repair used the same Qwen3.8-27B Q4_K_M target, F16 KV, six decode/ten batch threads, batch 256/microbatch 64 and an eight-core CPU quota. Every row passed independent correctness checks. The DFlash2 draft was a separate 1,143,006,816-byte Q4_K_M file (SHA begins `1a25c56858e1`); the target weights were unchanged.

| Original build configuration | Verified finished time | Increase |
| --- | ---: | ---: |
| No speculation | 547.96 s | baseline |
| ngram-simple, maximum 8 draft tokens | 576.99 s | 5.30% |
| DFlash2, maximum 7 draft tokens | 590.72 s | 7.80% |

The original server SHA begins `2d402ce300ad`; its architecture target lacked explicit FP16 vector arithmetic. A matched FP16-enabled build (`2bc4b2a6b3af`, `armv8.6-a+dotprod+i8mm+fp16+nosve+nosme`) was therefore tested separately:

| FP16 build configuration | Verified finished time | New-prompt time | Generation time |
| --- | ---: | ---: | ---: |
| No speculation | 536.85 s | 378.025 s | 140.218 s |
| DFlash2 | 749.56 s | 611.634 s | 107.079 s |

Both processed 6,493 new prompt tokens and generated 345 tokens; 23,841 cached tokens were reported in each run. DFlash accepted 279 of 420 proposed tokens, yet total runtime increased **39.62%**. Faster generation did not offset added prompt/configuration costs. DFlash requires flash attention on while the baseline used it off, so these compare complete configurations rather than isolating drafting overhead. One fixture does not establish every speculation workload's behavior. Neither DFlash nor ngram was promoted.

Evidence: `moe-comparison/dense-speculation-comparison.json`, `dense-dflash-comparison.json`, `fp16-dflash-comparison.json`; underlying allowlisted profile fields in `moe-comparison/opencode-dense{,-ngram,-dflash}-result.json` and `llm-fp16-agent-tests/opencode-dense{,-dflash}-result.json`; `llm-cpu-mtp-tuning/fp16-build-result.json`.

## Hexagon GGUF: mapping failures and partial-offload loss

The b12184 Hexagon backend failed full Qwen3-4B Q4_K_M execution on buffers of **1,070,071,808 bytes**, and **1,060,175,872 bytes** on a no-host retry. With four layers offloaded, a short 16-token test completed cleanly, but was slower than CPU:

| Same 4B GGUF, 1K context, two host threads/200% CPU quota | Prompt tokens/s | Decode tokens/s |
| --- | ---: | ---: |
| CPU | 31.73 | 15.25 |
| Four Hexagon layers | 28.86 | 13.57 |

A separate dense 27B four-layer attempt failed to map **1,112,477,696 bytes** and aborted (return code −6). It produced no valid throughput result. These are observed failures for these runtime/allocation strategies, **not proof of a universal 1–2 GB NPU limit**. Sharding and additional DSP sessions can change what maps successfully.

Evidence: `npu-investigation/qwen4b-result.json`, `qwen4b-first.log`, `qwen4b-nohost.log`, `dense-qwen-four-layers/result.json`. Only scalar measurements are exported.

## SD 3.5 QNN: transformer never reached image generation

The native Linux AppBuilder bridge attempted an SC8380 Windows-targeted compiled SD 3.5 Medium graph bundle, exported for 1024×1024. Transformer initialization failed. A fresh process without prior text-encoder loading reproduced FastRPC ENOMEM for a **2,256,535,552-byte** mapping, followed by QNN error **1002**. Shutdown passed, cDSP remained running and **27,666,628,608 bytes** were available afterward. The precise mapping/driver constraint remains unresolved; this is not established ordinary system-RAM exhaustion.

There is no generated-image quality or speed result for this path. Its demo also omits T5 conditioning, so a successful run would still need separate quality validation. Compiled QNN graph precision must not be described with a GGUF quant label.

Evidence: `sd35-npu/artifacts/transformer-load-result.json`, `transformer-fresh-assessment.json`.

## Positive NPU control: small QAIRT helper remains experimental

A different native runtime, GenieX QAIRT plugin `34cc633` with QAIRT 2.50, successfully ran the official Qwen3-4B-Instruct-2507 **w4a16**, compiled for a 4K maximum context. A 64-token sample decoded at **21.76 tokens/s**; the CPU Q4_K_M control measured **16.45 tokens/s**. The roughly 950-token case reported NPU TTFT **795.81 ms** versus CPU prompt evaluation **11,132.38 ms**. These timing boundaries differ, and neither includes an identical loading boundary. NPU host quota was 200%; CPU control used six threads/600%.

Four load/generate/unload cycles completed; one simple function passed four checks. This is useful evidence for a helper, not qualification for a main coding agent or sustained service. The runtime recovered mapping warnings by using additional sessions. Different quantization, runtime and CPU quotas prevent an isolated accelerator comparison. Small sharded success neither proves nor rules out large-model support.

Evidence: `npu-investigation/qairt-results.json`. The helper remains experimental rather than part of the deployed core.

## Comparison limits and revisit criteria

These were short, sequential tests without randomized thermal control. AC connection does not establish matched delivered electrical power; battery assistance and thermal state were not controlled. Do not compound these results into one overall speedup or compare quantizations as if weights were identical. Revisit only with a specific runtime/driver change or bounded hypothesis, preserve correctness and memory-release gates, and measure total finished-task time before deployment.


## GPU LLM inference parked by the owner (2026-11-09)

No further GPU LLM offload, split-decode, bandwidth or DT experiments are queued. The historical GPU-only logical-read screen reached about 62.3GB/s, versus about 115.85GB/s for current memlat CPU reads. The historical simultaneous CPU/GPU screen showed strong contention, and prior Vulkan LLM runs were slower with desktop pauses. These are different workload/boot conditions, not proof of every GPU path's ceiling. Third-party OpenCL prefill reports are not a matched local benchmark. The owner explicitly chose to focus on CPU/NPU; the planned3-minute post-memlat GPU control is cancelled. Image Studio is unaffected. Source fixtures/results are retained for reproduction; no DT or boot change follows from this decision.

### Pthread 8/12 prefill: regression caught before deployment

The decode-only poll screen missed a batch-path problem. An actual 4K prefill timedout at 120 s; an 8-second profile attributed  >95% user cycles to `ggml_barrier`. Explicit batch mask 0xfff was already set; poll-batch 0/50 both measured about 21 tok/s on 512 tokens, vs OpenMP 93.5. The caller remained on CPU 7 alongside a worker, with CPU 11 unused. A per-dispatch affinity fix passed its targeted regression and 4K KLD 0 / top-1 100%; batch 12 prefill improved to 74.5 but remains slower thanOpenMP. The fixed 8/8 candidate is still being qualified; no production change. [Evidence](../benchmarks/results/ornith-pthread-affinity-diagnosis.json).
