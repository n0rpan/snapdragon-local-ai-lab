# NPU helper investigation

The main coding profiles remain the larger Qwen models on CPU. A small NPU profile may be useful for short chat, synthetic log explanations or narrow structured tasks while keeping CPU use low. It needs its own correctness, memory and lifecycle checks.

## Live recovery hold: Phi reused-I/O diagnostic

The isolated QNN option-17 placement candidate compiled, and its control relink reproduced the original core exactly. One constructor-only attempt then registered I/O for four contexts and 40 graphs, reporting eight unique fused allocations and a 309 MiB per-context declaration. These diagnostics do not prove inference correctness: no prompt or generation was sent.

The attempt encountered a harness failure. The SDK printed a `rpcmem_init` diagnostic on stdout before the expected READY line; the strict parser rejected it, and the failure path terminated the C++ probe before its normal unload/deinit calls. The native child is reaped, but host available memory remains about 7 GB below its starting value. A read-only kernel snapshot shows roughly 6.27 GB of DMA buffers attached to CDSP FastRPC context banks; there is no pre-run DMA snapshot, so this is not an exact attributed allocation delta. No process retains an open FastRPC device descriptor. The parent holds the canonical inference lease in strict recovery quarantine, blocking further model jobs.

The temporary heap permission has been restored to its exact original ACL and mode after confirming that no native child or device descriptor remains. That permission cleanup is explicitly separate from RAM recovery: the guardian has not released, and qualification has not passed. A narrow successor is being prepared to preserve the exact known SDK diagnostic and perform bounded graceful cleanup on errors/signals. Fatal termination during SDK work still requires independent recovery evidence; a destructor alone cannot guarantee driver reclamation.

One normal supported CDSP stop/start was then attempted under the retained lease. The recorded stop returned success and reached offline; kernel logs show CDSP booting and returning to running. The subsequent debugfs DMA-buffer listing read triggered a kernel OOPS in `dma_buf_debug_show` through string formatting. The trace identifies the faulting read; it does not prove the underlying stale-pointer mechanism or attribute it to the microphone module listed among loaded modules. No further DMA-buffer debug reads or hardware recovery attempts are permitted in this boot. RAM recovery remains unqualified, and a clean reboot is required before more model tests. No global sysctl change, model-generation request or relaxed gate followed.

## Current checkpoint: answers work, qualification failed

The actual fixed-prefix native run completed 15 of the required 20 requests and three real 60-second idle worker exits. The loaded FastRPC identity was verified after every response. It stopped on the unchanged monotonic memory-recovery guard: charged residuals rose from 6,170,624 to 8,724,480 to 9,613,312 bytes. Each exit still passed its individual 32 MiB charged / 128 MiB host tolerance, all workers were reaped, recovery completed and the lease was released. The exact temporary heap ACL was restored. This is a failed qualification, not an enabled helper. Source receipt SHA256: `79feacc9028b2f28854b7772ae1f18cbb9dba194ac17be2249e7ad9e0ed0220f`.

Manual review also found three factual failures among those 15 own synthetic responses. They ended normally within output limits; no truncation or template failure was established. The Python recorder retained telemetry objects and repeatedly serialized them, so parent growth is a plausible contributor to the residual trend, not proof that the whole increase belongs to the recorder or to the NPU. A bounded streaming successor adds parent RSS and cgroup attribution without subtracting memory or relaxing thresholds. It is source-tested, not natively qualified.

The subsequent bounded-recorder control persisted all 662 numeric samples (606,090 bytes) with only 20 recent records retained. It answered ten fixtures and completed one recorded idle cycle, then failed on a `BrokenPipeError` while polling status as the second worker retired. That worker logged clean retirement, but the interrupted cycle is not counted as passed. Guardian recovery/reaping completed and the exact heap ACL was restored. A source-tested successor handles that closing-socket race only during optional idle status polling; answer/metrics errors, clean exit, socket removal, deadlines and all recovery guards remain mandatory. No full-cycle stability or quality improvement is established by this interrupted run. [Curated failure](../benchmarks/results/npu-helper-streaming-recorder-control.json).

The helper worker is already Rust; rewriting its Python qualification harness is not an established generation-speed improvement. A separate model review selects the official X Elite QAIRT W4A16 Phi-4-mini-instruct bundle as the first alternative, with Qwen3-8B as a larger fallback. These published bundles remain unqualified on this Linux laptop. Phi acquisition is prepared with pinned object version and quarantined integrity checks; no SDK/firmware upgrade or profile activation follows automatically.

| Finish gate | Required evidence still missing |
|---|---|
| Small-task quality | All 20 fixtures: eight titles, six summaries, six short answers; exact backend prompt counts and manual semantic review |
| Native retirement | Four real 60-second idle exits, code 0, removed socket and all descendants reaped; unchanged 32 MiB charged / 128 MiB host-loss anchors with 20 stable recovery samples |
| Finished latency | Same-fixture CPU control and cold/warm small-job completion; target at least 2× median cold latency versus lazy Flash loading |
| Portable runtime/access | Qualified fixed-prefix FastRPC build, normal-user access, seat revocation and login/reboot checks |
| Co-residency | Actual per-profile aggregate within 21.5 GiB and live 2 GiB reserve, or explicit helper exclusion; no implicit main-model eviction |
| On-demand routing | Qualified lazy dispatcher, canonical-lease contention, idle exit/restart and full installed lifecycle |

Both `LOCAL_AI_HELPER_CACHE_ADMISSION_TOKENIZER=1` and `LOCAL_AI_HELPER_RETIRE_ON_IDLE=1` must be observed on the actual owned worker. The 4K context, AC/battery-above-20%, zero own swap/new page-out and recovery gates remain fixed. Counts, compilation and five historical titles cannot satisfy these native gates.

The [pinned FastRPC source](https://github.com/qualcomm/fastrpc/blob/d247519650fe5cb16de6c78edaa95bcc4be25073/configure.ac#L93) supports a compiled `--with-config-base-dir` prefix; the existing initial shell lookup is not repaired by a general search-path environment variable. The fixed-prefix CDSP-only rebuild must still pass separate native qualification before selection. The proposed exact-heap `uaccess` rule grants active-seat applications RW access, broader than the read-only diagnostic ACL. It remains a separate permission policy and is not a prerequisite for one temporary trial. No SDK/model assets or host-specific privileged installer are redistributed by this checkpoint.

The CDSP-only fixed-prefix FastRPC rebuild now compiles and passes separate static ABI, dependency and compiled-prefix checks. The original failed dependency-allowlist receipt stays failed; a distinct finalization verifies the baseline Linux loader and completes packaging. That static step did not load or select the library. The later native trial above did load the new unselected prefix, without changing the global default. [Static evidence](../benchmarks/results/npu-fastrpc-fixed-prefix-static.json). Native qualification and normal-user system-heap access still gate use.

The [unselected owned prefix stage](../benchmarks/results/npu-fixed-prefix-stage.json) now passes exact package placement and a temporary same-FD PROT_NONE identity probe. It observes the Btrfs stat-device/mapped-device distinction and proves probe removal. No library constructor, SDK, worker, model or hardware call ran; the previous runtime remains selected. The prepared native successor verifies the actual loaded new library after every response and avoids the old root alias, and the later native trial above used temporary identity-bound heap access with exact rollback.

## What already ran locally

The native Qualcomm GenieX QAIRT plugin with QAIRT 2.50 ran the official X Elite Qwen3-4B-Instruct-2507 w4a16 bundle. A 64-token generation measured 21.76 tok/s; the CPU control measured 16.45 tok/s. A roughly 950-token prompt had 795.81 ms NPU TTFT versus 11,132.38 ms CPU prompt processing. NPU TTFT excludes loading. Four load/generate/unload cycles completed and RAM recovered.

These are short single trials using the same base model but different quantization/runtime implementations. One simple coding function passed four cases; that is not evidence of full coding-agent quality or sustained energy efficiency.

The compiled bundle has a fixed 4K context. The ordinary OpenCode prefix is about 5.5K tokens, so this bundle cannot simply replace the main OpenCode profile.

## What npurun contributes

Primary sources: [repository](https://github.com/bpbonker/npurun), [installation](https://github.com/bpbonker/npurun/blob/main/docs/install.md), [benchmark notes](https://github.com/bpbonker/npurun/blob/main/docs/benchmarks.md). Reviewed at commit `9f6fa48522d3b1f1222b1f6d9c7fc130fb4b06e0`.

npurun currently targets Windows on ARM and links Windows SDK libraries. It wraps Qualcomm Genie with a model registry, verified bundles, HTTP APIs and multi-turn cache handling. It does not by itself make Genie compute faster. Its graph-switching/polling findings are worth understanding, but their settings must be checked against the actual runtime.

Our native [GenieX QAIRT plugin](https://github.com/qualcomm/geniex-qairt-plugin), tested at `34cc633df34c69acb478305c917d5256cafb3551`, calls QNN graphs directly. It is a different high-level implementation from npurun's Genie wrapper, so copying Genie configuration flags is not an established optimization.

## Bounded next step

Reuse the existing 4B bundle. Check normal-user FastRPC access, serve through a resident local API with idle unload, validate a few fixed chat/log/structured fixtures and multi-turn cache behavior, and measure cold/warm total latency plus RAM/power. Keep this as an optional lite profile. Concurrent residency with a large model requires a separate memory check and is not enabled by default.

Large dense/MoE NPU execution and SD 3.5 remain separate unresolved mapping problems; see [the exact disabled-path measurements](FAILED-PATHS.md). Proprietary QAIRT/SDK libraries and compiled model bundles are not included under this repository's MIT license or redistributed here.


## Implementation checkpoint: 2026-11-07

The separate Linux adapter branch now compiles with the standard ARM64 Rust toolchain. Twelve software-only Rust tests cover protocol limits, bounded queuing, same-user socket access and lease exclusion. The native Rust module and C bridge compile and link; mock socket and supervision checks also pass. This is not a measured NPU helper deployment.

The helper depends on external development components that are not redistributed with this project. This repository contains its own host and protocol sources and public runtime observations.

A resident main model must never be evicted implicitly to produce a title. Co-residency needs measured headroom, especially with the 22 GB MoE; total system RAM alone does not prove it is safe. The initial adapter holds the shared exclusive inference lease and remains optional.


Normal-user device access is now prepared on the test laptop through an active-seat `uaccess` rule matching only nonsecure `fastrpc-cdsp`. Rule verification and current-user access checks pass; ADSP and secure CDSP remain root-only. Reboot/login persistence still needs verification. The helper remains disabled; normal-user access does not establish native inference quality or resource safety.


## Current goal checkpoint: first native attempt failed before an answer

The corrected guardian's first actual standalone run verified its pinned artifacts, then the first title request failed while creating the QAIRT device: `14001`, followed by `backend_poisoned`. No fixture answer or complete lifecycle cycle passed. The guardian reaped the worker, confirmed its fixed-anchor recovery gates and CDSP running state, then released the canonical lease. [Failure and resource evidence](../benchmarks/results/npu-helper-native-device-failure.json). The cause is not established; runtime/device configuration must be compared with the earlier successful native pipeline before another bounded attempt. This does not supersede the historical 21.76 tok/s control or qualify the new helper.

The run's cgroup memory includes the explicit artifact integrity scan and file cache. It must not become a measured model/admission claim. Co-residency, semantic quality, token parity and the complete 20-fixture/four-cycle suite remain unproven. Nothing was installed or enabled.

One temporary legacy-prefix symlink, with the same binary/caps, advanced to context loading but then failed at context index 1 with `1002`. The captured PID-limit counter and memory-limit events were zero. Recovery passed and the exact temporary link was removed. [Trial and limits](../benchmarks/results/npu-helper-prefix-trial.json). This identifies a real portability defect without proving the remaining error's cause. A final release needs a portable FastRPC configuration/build path, not that compatibility link. The recorded error does not establish the failure cause.

The newly isolated helper bridge uses the actually measured native `LLMPipeline` / `libgeniex_core` path. The older Rust helper's SDK `libgeniex.so`/plugin path had compiled but had not run native inference; its compile result cannot inherit the earlier 21.76 tok/s measurement. The new bridge resets independent requests, counts tokens without an assumed added BOS, exposes numeric timing and adds summaries.

The corrected wrapper compiled in a guarded 3.998 s unit (CPU 3.950 s, peak 538.9 MiB, swap 0); all 13 Rust model-free tests passed. Eleven offline guard/log tests passed before build. Canonical lease/children/resource checks passed, and **no vendor API or model was executed by this compile**. Native binary SHA256 is `d1ded1e3f39aaf815139fd82d39b5557aa8048bb87389e6d1ca01237b2c4cdac`; compile input manifest `4bd064b322adc419affb738c5b784a383bc28154c2d4e4ba16ebb17f9ba2d25d`, receipt `2d4e95d8523a397897fdc369b77c257321a1e17a49f066c35f145755845ebfbd`.

The initial compile guard refused an existing safe owned 0755 control directory; its corrected variant retained the same canonical 0600 inode.

A guardian now retains the canonical lease across worker exit and external recovery. Native 20-fixture quality/latency/token parity, four load/idle/unload cycles, actual host/DSP RAM release and co-residency remain required before a qualification receipt or title routing can be enabled. New tests use 2 GiB actual MemAvailable headroom and 21.5 GiB aggregate admission; standalone diagnostics may reserve that full cap without pretending it is a measured peak. The compiled source remains disabled in the installed application. Current host read-only metadata confirms ordinary-user access to nonsecure CDSP, while ADSP and secure CDSP remain inaccessible.

## Diagnostic follow-up: IO guidance allocation

A separate, bounded callback-enabled bridge reached the same context-loading failure. No answer completed. The guardian recovered and released the lease; 86 external observations showed no own swap, memory max/OOM/kill or PID-limit events. The temporary legacy-prefix link was removed. [Diagnostic receipt](../benchmarks/results/npu-helper-guidance-buffer-failure.json). The recorded error does not establish the failure cause.

This narrows the failing stage; it does not establish a requested allocation size, physical NPU model limit or driver cause. The approximately 3.27 GiB cgroup peak includes file cache and integrity scans, and is not a DSP-residency measurement. The helper remains disabled. Next compare allocation/configuration behavior with the earlier successful pipeline and fix the portable FastRPC configuration path before another bounded trial.

The source/log comparison identified an additional access prerequisite: this FastRPC allocator opens `/dev/dma_heap/system` read-only before allocating shared buffers. The current node is root-only; the normal-user attempt logs access denied and falls back to a different allocation ioctl, which then fails. CDSP access alone is insufficient. `doctor` now reports the system-heap presence and read permission separately, without opening it or allocating memory. These fields are prerequisites, not an NPU-readiness claim.

A separate trial is prepared to grant temporary read access to this one node for the current UID, preserve all helper options and limits, then restore the captured ACL after terminal recovery. No persistent permission change or successful trial has occurred yet. The final installation also needs a portable FastRPC configuration path and verified login/reboot access.

## Temporary heap access: answers work, idle retirement required

The first access trial stopped before SDK execution because its old guard rejected normal host page-ins. The exact ACL and alias were restored. An idle counter check then measured 105 incoming pages, zero outgoing pages and about 27 GiB available with zero recent memory PSI. The separate V5 diagnostic permits existing host pages to return to RAM, rejects any new page-out, and retains zero helper swap, the 2 GiB reserve, admission, fixed recovery anchors and full cleanup. Its AC/battery policy matches the user's above-20% requirement. Old strict-policy failures stay failed. Four host mock groups passed before native execution.

With temporary read-only access to the one system heap, the unchanged native helper answered its first five title fixtures. Manual topic/conciseness checks passed; cold client latency was 4.706 s including a 3.842 s native load, and the four warm requests took 0.857–0.992 s. These short title results do not establish broader chat/coding quality. [Measurements, failed lifecycle and rollback proof](../benchmarks/results/npu-helper-heap-access-and-idle-failure.json).

The run remains **failed**: after 60 s idle it reported unloaded and had zero CDSP descriptors, but retained about 245 MiB of charged host memory. This exceeded the unchanged 32 MiB charge / 128 MiB host recovery tolerances. Worker exit recovered the original anchors without a reboot. All descendants were reaped, the guardian released, and the exact original ACL and mode were restored with the temporary alias removed. Only five of the required 20 answers ran; no full four-cycle or co-residency qualification is claimed.

The next implementation is graceful native-worker retirement after idle unloading, with a small dispatcher restarting it on demand. It must stop admission safely, finish accepted work, close its exact socket and exit cleanly. The original recovery thresholds stay in place; a wider allowance is not the fix.

The narrow retirement implementation compiled in 8.036 s and passed all 19 Rust tests. Two actual private-socket mock cases verified clean retirement and an accepted parser crossing the idle boundary, including socket removal and lease release. [Compile/mock evidence](../benchmarks/results/npu-helper-idle-retirement-build.json). These are model-free checks; the new native binary still needs all 20 fixtures and four native lifecycle cycles.

A separate worker-owned admission-tokenizer cache now compiles and passes **29 Rust tests**. It remains opt-in and requires idle retirement. Cold rejected requests and failed tokenizer allocations also participate in retirement, so CPU allocator state cannot stay resident merely because the model never loaded. The original fresh-tokenizer path remains the default. A stale runtime lock binding stopped the first attempt before compilation; a separately verified binding preserved the same source and identity checks. [Build and retained failure](../benchmarks/results/npu-helper-tokenizer-cache-build.json). No SDK tokenizer or native inference ran in this build; that build alone does not establish token-count parity, latency, added RAM or native retirement.

The subsequent [CPU SDK count check](../benchmarks/results/npu-helper-tokenizer-count-parity.json) passed all eight synthetic texts, three times each, with no added BOS. Reconstructing and running the fresh tokenizer averaged about 426 ms per count; cached short counts were below 0.1 ms and the 3,074-token case took 3.58 ms. Initial cache creation cost 363 ms. The process passed strict recovery with a 203.5 MiB unit peak and zero swap or limit events. This paired microdiagnostic establishes count parity and identifies removable CPU overhead; it is not ABBA model-performance evidence. Loaded-helper latency, additional resident RAM, full native generation quality and retirement remain pending.

## Phi-4-mini alternative: CPU template passes, native mapping fails

The official X Elite QAIRT W4A16 bundle was downloaded, checked against its published CRC, and safely extracted as data. Its vendor Python source stayed quarantined. The separate Phi worker compiled, and eight native CPU cases passed exact bundled-Jinja and full-token-ID parity, including special markers, Norwegian and context boundaries.

The first actual NPU quality attempt failed before an answer: QNN could not map shared weight buffers and a fused 125,829,120-byte KV buffer, reporting error 8003 and process-domain use around 1989 MB. The runtime also reported FastRPC error 0x1 and domain 27; no meaning is inferred from those numeric values. These observations establish a failed Linux/runtime mapping attempt, not a universal2GB device limit or a poor model-quality result. All descendants were reaped, strict recovery completed and the original heap ACL was restored. [Evidence](../benchmarks/results/npu-phi4mini-native-load-failure.json).

One source-supported follow-up is the context-level reused-IO placement budget, currently unset in this loader. That is a conditional constructor-only experiment, not a RAM reduction or an established fix. No larger helper download, SDK/firmware upgrade or profile activation follows from the failed load. A single generic factual-policy comparison on the working Qwen4B remains a separate fallback.

## Working Qwen4B: single factual-policy candidate closed

The one generic SYSTEM-policy experiment compiled, passed all 30 Rust checks, and passed exact cached/fresh counts for ten full A/B prompts. The native A–B–B–A screen completed only A–B–B: 15 answers, then the unchanged monotonic recovery guard stopped residuals of 2,113,536 → 2,129,920 → 2,138,112 bytes. Individual recovery bands passed; the 16 KiB/8 KiB increases are not attributed to DSP leakage without ownership evidence. Cleanup, reaping, guardian release and exact ACL restoration passed.

Both B repetitions removed the invented summary caveats/uncertainty and preserved the stated facts. Both also produced a four-sentence Norwegian summary against the unchanged maximum of three. The holdouts passed, but that format failure closes this candidate regardless of the incomplete ABBA. No helper qualification, causal speed gain or further prompt-tuning loop follows. [Closed screen](../benchmarks/results/npu-qwen-policy-screen-closed.json). The panel again reports the NPU as experimental.
