# Measured hardware ceilings

Measured 2026-11-09 on the Yoga Slim 7x 83ED, X Elite, kernel 7.3.0-40-omarchy. These are measured ceilings for the named workloads, not advertised RAM bandwidth or NPU TOPS. Qualified classic settings and the Ornith default have since been promoted; the historical hardware controls below retain their original boundaries.

## Current result: memlat enabled after the approved boot

The alternate Yoga UKI booted successfully on 2026-11-09, retaining kernel 7.3.0-40-omarchy. Firmware accepted SCMI vendor protocol 0x80; the built-in driver initialized three performance domains and `ddr`, `llcc`, `ddr-qos` appeared with the remote governor. The tested MEMLAT entry is now the explicitly approved default, with automatic verified rebuilds for edge updates and named stock fallback; see MEMLAT.md.

**CPU streaming bandwidth more than doubled for spread placements.** Same compiled binary, array, pass count, explicit affinity, three alternating repetitions and performance governor/3.4176 GHz. All 30 postboot checksums passed. This is a real new memory-throughput result; the matched Ornith inference screen below also improved, with exact output parity.

| Threads / placement | Before GB/s | Memlat GB/s | Ratio |
|---|---:|---:|---:|
| 2 / packed | 48.39 | 65.73 | 1.36× |
| 2 / spread | 50.31 | 110.90 | 2.20× |
| 4 / packed | 47.37 | 66.35 | 1.40× |
| 4 / spread | 48.83 | 98.73 | 2.02× |
| 6 / packed | 48.30 | 89.85 | 1.86× |
| 6 / spread | 48.88 | 115.85 | 2.37× |
| 8 / packed | 49.00 | 108.12 | 2.21× |
| 8 / spread | 49.00 | 112.55 | 2.30× |
| 12 / packed | 49.32 | 110.11 | 2.23× |
| 12 / spread | 49.21 | 109.52 | 2.23× |

The largest postboot arm median is **115.85 GB/s** (six spread threads), with a 115.57–116.00 GB/s range. Cluster placement now matters: four packed threads reach 66.35 GB/s versus 98.73 GB/s spread. The earlier flat sweep was performed with a different memory-governor state and cannot settle placement after memlat.

Postboot devfreq exposes DDR 200–4224 MHz and LLCC 300–1066 MHz; DDR-QOS has synthetic keys 1/100, not clock Hz. The driver `get_cur_freq` reads the maximum **firmware-voted** frequency across monitors. Treat that as a vote readback, not a direct physical clock counter.

Raw results: [`ram-memlat-runs.jsonl`](../benchmarks/roofline/ram-memlat-runs.jsonl). The earlier sections below preserve pre-memlat results and interpretation at the time; their ~50 GB/s denominator is superseded for the current boot.

## Before memlat: RAM streaming reads

2 GiB anonymous array, 32 passes (64 GiB read per run), eight independent NEON accumulation chains, independently checked per-thread and total checksums. Initialization, allocation and thread creation are excluded. Three alternating repetitions; decimal GB/s. Explicit thread affinity was read back and verified. Governor: performance. No CPU quota; service hard cap 5 GiB, swap prohibited.

Packed order: `0,1,2,3,4,5,6,7,8,9,10,11`. Spread order: `0,4,8,1,5,9,2,6,10,3,7,11`. L2 sharing groups observed from sysfs: 0–3 / 4–7 / 8–11. At 12 threads both arms use the same CPU set and are an ordering control.

| Threads | Packed median [min,max] GB/s | Spread median [min,max] GB/s |
|---:|---:|---:|
| 2 | 48.39 [48.24, 48.82] | 50.31 [49.54, 50.37] |
| 4 | 47.37 [47.34, 47.54] | 48.83 [48.61, 48.89] |
| 6 | 48.30 [48.12, 48.38] | 48.88 [48.07, 49.17] |
| 8 | 49.00 [48.97, 49.04] | 49.00 [48.59, 49.47] |
| 12 | 49.32 [48.73, 50.23] | 49.21 [49.02, 49.28] |

The largest arm median was **50.31 GB/s** (two spread threads); most configurations reached 48–49 GB/s. More threads did not increase streaming bandwidth. This screen does not establish a large per-cluster bandwidth bottleneck. Model thread placement still needs its own screen because quantized MoE computation and synchronization differ from sequential reads. Do not use the earlier unmeasured 100–120 GB/s estimate as the denominator.

This is a sequential read-plus-checksum ceiling for this OS/power configuration. It does not measure every memory access pattern or prove a universal physical maximum. Model bytes/token must include the actual active quantized tensors, shared layers and context-dependent KV traffic before estimating bandwidth utilization.

Raw runs: [`ram-runs.jsonl`](../benchmarks/roofline/ram-runs.jsonl). Source: [`memory_read.c`](../benchmarks/roofline/memory_read.c). Built with `cc -O3 -std=c11 -mcpu=oryon-1 -pthread -Wall -Wextra -Werror`. Each run logs before/after temperatures and power metadata; the largest sampled temperature in this screen was 57.3 °C. All 30 runs passed their checksum checks. These brief screens do not establish sustained thermal performance.

## NVMe random reads

120/120 read-only fio arms passed. io_uring, 2 seconds per arm, 3 repetitions; existing owned 49.95 GB shard larger than RAM. Decimal GB/s. The inference cgroup hard cap was 6 GiB (observed peak 3.37 GiB), so this buffered screen does not represent an unlimited page cache. No file creation or writes, no global cache dropping.

| Block KiB | Direct QD1 | QD4 | QD16 | QD32 |
|---:|---:|---:|---:|---:|
| 256 | 0.346 | 1.211 | 3.075 | 3.931 |
| 512 | 0.695 | 1.873 | 3.893 | 4.496 |
| 1024 | 1.065 | 2.673 | 4.524 | 4.581 |
| 2048 | 1.405 | 3.731 | 4.581 | 4.580 |
| 4096 | 1.891 | 4.399 | 4.585 | 4.560 |

Direct reads plateaued at about **4.58 GB/s**, reached by 1 MiB/QD32 or larger blocks with sufficient queue depth. At 256 KiB/QD1 the median was only 0.346 GB/s; concurrency and read size matter substantially. Buffered random-read medians ranged 0.44–1.62 GB/s and were slower here, not a faster SSD path. Physical-device counters and achieved-depth distributions are retained per arm; parent and partition counters must not be added.

Raw results: [`nvme-runs.jsonl`](../benchmarks/roofline/nvme-runs.jsonl). Source: [`nvme_read.py`](../benchmarks/roofline/nvme_read.py).

## Accelerator matrix throughput

Adreno Vulkan synchronized resident MUL_MAT passed scalar sanity checks: 1024² F16-weight/F32-activation median 267.05 GFLOP/s; 2048² F16/F32 325.73 GFLOP/s; 2048² Q4_0/F32 1566.33 effective GFLOP/s. One warmup plus three computes; allocation, transfers and initialization excluded. These are operator-path results, not advertised hardware FP16/TOPS or model KLD qualification. Raw results: `benchmarks/roofline/gpu-matmul-runs.jsonl`. Hexagon fixture and narrowly relocated FastRPC monitor library compiled successfully, but no new NPU device run occurred: priority 0 memlat investigation superseded it. NPU ceiling is still unmeasured, not zero.

## Reproduction and failures

The coordinator runs one heavy job under the existing inference lease. It applies the current MemAvailable/PSI memory policy, prohibits benchmark cgroup swap, logs temperatures, and permits small host zram activity. Failures are retained in `benchmarks/roofline/failures.jsonl` and failed run rows. The first RAM launch rejected the invalid systemd `CPUQuota=infinity` spelling before starting work; the corrected launch omits a quota.

## Memory-controller check and scope

The 48–50 GB/s result is the **CPU streaming-read ceiling for this workload**, not a proven whole-chip DDR limit. Disassembly confirms paired 128-bit NEON loads and independent accumulators; there is no timed memcpy or write stream. Numerator = bytes actually read once per pass.

A longer four-thread spread run read 512 GiB at 47.89 GB/s. Concurrent read-only ICC sampling observed the loaded LLCC BWMON maximum vote 16,896,000 kB/s and all three CPU BWMON maxima 17,064,000 kB/s. These match the loaded device-tree `opp-peak-kBps` maxima. The monitor did reach its highest requested-bandwidth OPP during the run. **This does not establish actual DDR frequency or 8448 MT/s**: `interconnect_summary` exposes aggregate/per-client bandwidth requests, not a DRAM clock. No direct DDR clock appeared in the checked standard CCF clock summary. See [kernel interconnect documentation](https://docs.kernel.org/driver-api/interconnect.html) and [BWMON driver](https://github.com/torvalds/linux/blob/master/drivers/soc/qcom/icc-bwmon.c). Raw sampling: [`icc-cpu-run.jsonl`](../benchmarks/roofline/icc-cpu-run.jsonl).

The CPU-only / Vulkan-read-only / simultaneous CPU+GPU controls below are complete; they show that 50 GB/s is not a whole-chip denominator. The simultaneous test is one coordinated benchmark under one lease. A positive aggregate result would justify investigating split decode; a flat result alone would not prove every accelerator decode path is bandwidth-useless.

## Priority 0: SCMI memlat boot experiment

Kernel `7.3.0-40-omarchy` has `CONFIG_QCOM_SCMI_GENERIC_EXT=y` and `CONFIG_SCMI_QCOM_MEMLAT_DEVFREQ=y`, including built-in registration symbols, but the live Yoga DT exposes only SCMI protocol 0x13. [Upstream experiment patch 151](https://github.com/ramonjesusmiranda-debug/sp11-el2/blob/main/kernel-patches/151-memlat-arm64-dts-qcom-hamoa-Enable-LLCC-DDR-DDR-QOS.patch) adds only `protocol@80 { reg = <0x80>; }`; LLCC/DDR/DDR_QOS configuration is in the driver, not additional Yoga DT properties. The Surface author's STREAM/memlat gains are a lead, not a result on this machine.

A dedicated tracefs instance captured `interconnect:icc_set_bw` during a 512 GiB CPU read run: 48.016 GB/s, all CPU BWMON peak votes reached 17,064,000 kB/s and LLCC reached 16,896,000 kB/s. No trace overruns/dropped events. This **does not reproduce the reported flat 4.8 million kB/s symptom** and does not confirm causation. The trace instance was removed afterward; global tracing was not changed.

A minimal overlay was applied offline to the stock Yoga DTB. Rebuilding the installed DTS with dtc `-@` exactly reproduced the stock embedded DTB (SHA256 `96506e59f410203dee572c25adf3c28cf1d398ed479d2ca86a06765ec77acc1b`). The candidate adds only the vendor protocol node, its new phandle and symbol; existing phandles are preserved. Its 224,999 bytes fit the existing 225,280-byte Yoga section slot. An independently reviewed copied UKI preserves every other section, kernel, initrd and command line. It is installed as **Local AI memlat test (7.3.0-40)**. The existing Omarchy image and default selection remain unchanged; Secure Boot is disabled. **This paragraph records the preboot preparation. The approved boot and measured gains are reported at the top of this document; the GPU minimum-clock control finished and restored 300 MHz.**

After the approved boot, check SCMI protocol acceptance, memlat binding and DDR/LLCC/DDR_QOS devfreq nodes; repeat reads and a resident MoE throughput screen with temperature/power logging. Firmware rejection means record the failure and return to the unchanged normal entry.

## Before memlat: CPU + GPU streaming controls

Three alternating repetitions, 512 MiB GPU input, 8-second windows, 32 full-array passes per submitted GPU batch; coherent UMA allocations, initialization/transfers/JIT/warmup excluded. Combined runs use 12 pinned CPU readers over a separate 512 MiB array and one synchronized start. GPU and CPU bytes use the same union of actual start/end times, including small launch/drain tails; rates from separate runs are never added.

| Control | Median GB/s | Min–max GB/s |
|---|---:|---:|
| GPU only | 62.33 | 62.10–63.01 |
| CPU + GPU aggregate | 69.24 | 68.58–69.28 |

The combined result exceeds the CPU streaming-read result. Therefore **~50 GB/s must not be used as a measured whole-chip ceiling**. This is sustained logical streaming bandwidth, not direct DRAM-controller-counter traffic; it does not prove independent buses, and the GPU heavily reduces concurrent CPU throughput. The integrated CPU control uses a smaller array/four accumulation chains, unlike the initial 2 GiB/eight-chain CPU-only screen. A model split still needs its own scheduling, correctness and throughput measurements.

Independent source review verified byte coverage, count arithmetic and timing. Every CPU pass is checked; GPU reference checks cover every output of the final 32-pass batch (earlier batches overwrite those slots). All runs passed these checks. Raw results: [`gpu-read-runs.jsonl`](../benchmarks/roofline/gpu-read-runs.jsonl); sources [`read_bandwidth.cpp`](../benchmarks/hardware/read_bandwidth.cpp), [`read_bandwidth.comp`](../benchmarks/hardware/read_bandwidth.comp).

## Resident-model byte accounting, preliminary

Actual GGUF descriptors, excluding the full input embedding matrix and MTP block but including the separate output head once, give a target tensor-sweep proxy of **2.626 GB/token for Qwen MoE**, **1.797 GB/token for Ornith**, **15.821 GB/token for dense Qwen**, and **16.477 GB/token for Arete**. These are not measured DRAM counters and omit KV/activation/cache effects. Qwen's base attention/SSM uses about 1.359 GB of Q8_0 tensors; Ornith uses about 0.722 GB of Q4_K tensors, so applying the same guessed 1.6 GB/token to both is misleading.

For historical non-MTP Qwen decode, 2.626 GB × 13.886 tok/s is a 36.47 GB/s tensor-sweep proxy. MTP accepted-token speed cannot be multiplied by the same number: verification amortizes target reads and drafting adds its own traffic. Proposal/acceptance and verification-batch counts are needed. The post-memlat real-model screens below supersede the historical four-thread result.

## GPU minimum-clock vote control before memlat boot

Requested GPU min_freq 300,000,000 versus 1,250,000,000 Hz; three alternating paired repetitions. GPU runtime-PM state was active and cur_freq matched the requested clock in both arms. Restored and verified 300,000,000 Hz afterward.

| Workload | 300 MHz median | 1.25 GHz median | Result |
|---|---:|---:|---|
| CPU NEON read, four spread threads | 49.025 GB/s | 48.936 GB/s | No improvement |
| Ornith MTP CPU decode, fixed 128-token prompt/64 output | 12.732 tok/s | 12.816 tok/s | +0.66%, small/noisy |
| Ornith prompt processing | 94.744 tok/s | 98.352 tok/s | +3.81%, overlapping ranges |
| Ornith whole request | 6.302 s | 6.219 s | −1.31%, small/noisy |

All three paired greedy model outputs matched. These synthetic raw completions compare the clock arms only; they must not replace prior agent-task numbers or be compared directly to unrelated historical decode prompts. Ornith used the qualified CPU-only engine, 4/10 threads, 4096 context, native MTP2/pool3 and no prefix cache. The initial 22.66 GB Qwen load hit sustained host memory full PSI and was stopped before model timings; all six earlier CPU-read arms had completed. The failure is retained. The bounded followup used smaller Ornith and passed.

This does not establish a useful DDR-vote workaround. A flat result also does not by itself establish the physical location of a bottleneck. Raw runs: `benchmarks/roofline/gpu-vote-runs.jsonl` (failed overall Qwen load, completed CPU components), `gpu-vote-ornith-runs.jsonl` (successful paired model screen).

The owner approved the one-time memlat reboot after this screen. Limine's one-shot identifier is `Local-AI-memlat-test--7.3.0-40-`; persistent default remains the normal Omarchy entry. Firmware acceptance and the repeated CPU/MoE controls are now verified above. GPU model decode remains untested after memlat; future GPU comparisons must check CPU-only versus Vulkan-enabled allocation paths and duplicate weight buffers. Desktop/video interference needs its own bounded measurement; the saturating read-kernel contention result does not quantify ordinary desktop overhead.

## Matched Ornith before/after memlat

Same qualified CPU-only engine and Compact MTP model, 4 decode / 10 batch threads, MTP2/pool3, 4096 context, three fixed raw prompts of 128 tokens and 64 generated tokens, no prefix cache. GPU minimum at its restored300 MHz.

| Metric | Before memlat median | After memlat median | Change |
|---|---:|---:|---:|
| Decode |12.7317 tok/s|17.1822 tok/s|+34.96%|
| Prefill |94.7439 tok/s|107.6971 tok/s|+13.67%|
| Whole short request |6.3015s|4.8364s|−23.25%|

All three postboot greedy outputs exactly matched their preboot counterparts. Postboot decode range 16.82–18.35 tok/s. Repeating the GPU1.25 GHz pin gave17.09 tok/s decode and 4.89s request, adding no win; restored300 MHz. Host full-memory PSI remained zero. This is a newly measured **short-screen** improvement; no protected full-task improvement or production-default promotion is claimed. Raw results: `benchmarks/roofline/gpu-vote-ornith-memlat-runs.jsonl`.

## Ornith thread/placement screen after memlat

84 native target-only throughput cases passed: 14 thread/placement combinations × prompt512/decode 64 × three alternating repetitions. Same qualified CPU archives; no MTP in llama-bench. Fixed synthetic token stream across arms. Full warmup per context, repack on, batch/ubatch256, CPU-only, f16 KV, no CPU quota at leaf or ancestors (zero throttling).

| Threads | Placement/mask | Decode median [range] tok/s | Prompt512 median tok/s |
|---:|---|---:|---:|
| 4 | unpinned / 0x0 | 24.60 [24.21–24.98] | 52.55 |
| 4 | spread / 0x113 | 24.47 [24.37–24.67] | 52.36 |
| 4 | packed / 0xf | 22.34 [22.33–22.52] | 53.42 |
| 6 | unpinned / 0x0 | 28.67 [28.63–28.91] | 76.36 |
| 6 | spread / 0x333 | 27.90 [27.84–28.17] | 74.68 |
| 6 | packed / 0x3f | 29.17 [29.04–29.18] | 73.65 |
| 8 | unpinned / 0x0 | 28.88 [28.54–29.04] | 95.05 |
| 8 | spread / 0x377 | 29.08 [29.02–29.38] | 93.55 |
| 8 | packed / 0xff | 30.56 [30.40–30.85] | 93.57 |
| 10 | unpinned / 0x0 | 27.96 [26.30–28.43] | 109.62 |
| 10 | packed / 0x3ff | 29.86 [29.71–29.95] | 109.42 |
| 10 | spread / 0x77f | 28.82 [28.82–28.87] | 108.55 |
| 12 | unpinned / 0x0 | 15.66 [15.57–16.04] | 115.29 |
| 12 | all cores, strict / 0xfff | 25.86 [24.63–26.91] | 118.03 |

Best target decode was eight packed threads,30.56 tok/s versus four unpinned24.60 (+24.3%); six packed was29.17. Prompt processing favoured more threads, up to 118.03 tok/s at 12strict. Spread placement was not the best real-model decode arm despite the synthetic RAM result. These are **synthetic target-only** results, not MTP accepted-token throughput or protected-task gains; do not compare them directly with the 17.18 tok/s raw-completion MTP screen.

Whole-matrix sampled thermal peak was97.7°C, which warrants sustained/thermal validation before selecting a production profile. Native benchmark code retained OpenMP affinity between configurations; the invalid first attempt was stopped/recorded. Corrected source restores every owned thread to inherited affinity between cases; actual task masks were observed, and the unpinned six-thread result recovered from the invalid singleton-core case. Patch: `benchmarks/roofline/bench-matched-masks.patch`. No engine kernels were rebuilt or production profiles changed.

Raw results: `benchmarks/roofline/ornith-thread-placement-memlat-v3.jsonl`. The identical Qwen screen completed successfully below. Readiness-aware loading PSI grace retains the hard memory floor, own OOM/swap checks and loading deadline.


## Qwen thread/placement screen after memlat

The same 84-case target-only screen passed on Qwen at its native quantization, with repacking enabled. There was no CPU quota. Each cell is the median of three alternating repetitions; prompt processing uses the listed thread count too, rather than the server's separate ten-thread batch pool.

| Threads | Unpinned decode / pp512 tok/s | Packed decode / pp512 | Spread decode / pp512 |
|---:|---:|---:|---:|
| 4 | 24.08 / 86.98 | 20.18 / 82.32 | 24.42 / 86.22 |
| 6 | 26.24 / 122.57 | 26.47 / 115.77 | 25.54 / 117.50 |
| 8 | 25.42 / 143.88 | 26.88 / 145.67 | 26.10 / 143.67 |
| 10 | 23.50 / 161.66 | 25.39 / 165.64 | 24.76 / 163.39 |
| 12 | 13.83 / 163.94 | 21.45 / 173.22 | same strict mask |

Eight packed threads improved target-only decode by 11.6% over four unpinned. Its 2.626 GB/token tensor-sweep proxy gives 70.6 GB/s, about 60.9% of the 115.85 GB/s read ceiling. Ornith's corresponding best 30.56 tok/s gives 54.9 GB/s, about 47.4%. These are tensor-sweep proxies, **not measured DRAM traffic**; they exclude KV traffic and do not apply to accepted-token MTP rates.

Qwen loaded successfully with bounded readiness-aware PSI grace. Minimum host MemAvailable was 6.48 GiB, own swap/OOM remained zero, and peak sampled CPU temperature reached 98.0°C. Repeated cgroup maximum events indicate file-page reclamation, not an OOM or an automatic reason to abort. Most observed anonymous packed weights already used transparent huge pages (about 20 GB), so disabling mmap cannot be assumed to add THP to all weights or save memory. No GPU inference job has run in this boot.

Raw results: [`qwen-thread-placement-memlat.jsonl`](../benchmarks/roofline/qwen-thread-placement-memlat.jsonl).

## MTP and ordinary coding cross-check

The raw 128-token/64-output thread screen preserved all three outputs, but its repetitive completion text greatly favours speculation. Its Qwen peak near 40 tok/s is **not an agent coding result**. Qwen accepted 42/42 drafts in each raw case; realistic coding must be measured separately.

Three ordinary chat requests (merge intervals, deduplicate IDs, parse an ASCII port) were independently checked inside the existing filesystem/network-isolated Python sandbox. Each tuning arm produced exactly the same three outputs as its baseline. Every first pass solved two cases and failed the same Unicode-digit edge case in parse_port. One bounded repair turn, given the failing assertion and input, corrected that case in all four compared arms.

| Host and configuration | Median decode across first responses | Three first-response times, total | parse_port including one repair |
|---|---:|---:|---:|
| Ornith: 4 unpinned, MTP2 | 18.52 tok/s | 17.14 s | 13.04 s, pass |
| Ornith: 8 packed, MTP2 | 24.64 tok/s | 13.47 s | not retested |
| Ornith: 8 packed, MTP off | 28.18 tok/s | 12.36 s | 9.72 s, pass |
| Qwen: 4 unpinned, MTP2 | 31.86 tok/s | 12.93 s | 8.46 s, pass |
| Qwen: 8 packed, MTP2 | 36.15 tok/s | 11.54 s | 7.62 s, pass |
| Qwen: 8 packed, MTP off | 24.85 tok/s | 15.63 s | not retested |

MTP policy is model-dependent: these coding prompts favour Qwen MTP on and Ornith MTP off. Coding rates are medians across **different prompts**, not repeated timing estimates. The repair comparison has one run per arm; it supports correctness and the candidate choice, not a statistically settled gain or full-task qualification. Server load is excluded from request timings. Generation/output differences across the two models are not a matched quality ranking. Production profiles remain unchanged.

Raw: [`mtp-thread-screens-memlat.jsonl`](../benchmarks/roofline/mtp-thread-screens-memlat.jsonl), [`mtp-code-screens-memlat.jsonl`](../benchmarks/roofline/mtp-code-screens-memlat.jsonl), [`mtp-code-repairs-memlat.jsonl`](../benchmarks/roofline/mtp-code-repairs-memlat.jsonl).

## Memlat thermal, idle and power observations

The complete native thread sweeps lasted about 10.7 minutes on Ornith and 8.1 minutes on Qwen, with sampled peaks of 97.7°C and 98.0°C. They completed without an observed crash or OOM. This does not establish long-term stability or acceptable sustained heat.

A later 30-second normal-desktop idle observation cooled from a maximum CPU sensor reading of 45.6°C to 39.9°C. DDR firmware votes ranged 200–4224 MHz (median 547 MHz); LLCC votes ranged 300–1066 MHz (median 466 MHz). Polling itself can wake the system. Battery energy stayed at 69.53 Wh; reported net battery power was zero while plugged in. There is **no exposed package-power counter**, so this is not a zero-watt package/wall-power measurement or a pre/post idle-power comparison.

Suspend remains untested: this boot offers s2idle and deep, with deep selected; the checked RTC exposes no programmable wake alarm. A manually coordinated wake is needed before testing s2idle. The alternate boot is still opt-in and the normal boot remains the default. Data for the orchestrator's [Omarchy issue #18254](https://github.com/omacom/omarchy/issues/18254): [`memlat-idle-power.jsonl`](../benchmarks/roofline/memlat-idle-power.jsonl).


## Prompt microbatch screen

Three alternating repetitions, pp512, logical batch 512, strict masks for 8/10/12 threads; microbatch 256 versus 512. All 18 cases passed per model. No changes to kernels, quantization or production settings.

| Model | Threads | ub256 median tok/s | ub512 median tok/s |
|---|---:|---:|---:|
| Ornith |8|92.75|92.71|
| Ornith |10|110.83|111.17|
| Ornith |12|121.29|121.77|
| Qwen |8|143.55|141.16|
| Qwen |10|166.78|164.84|
| Qwen |12|172.87|175.21|

Doubling microbatch did not produce a convincing improvement: +0.4% for Ornith and +1.4% for Qwen at 12 threads, with overlapping ranges. Keeping256 avoids using more workspace for this small return. Twelve prompt threads improved over ten by 9.4% for Ornith and 3.7% for Qwen at ub256. This remains a prompt-pool candidate to check on longer requests and desktop responsiveness; it does not imply12 decode threads, which lost the target-only screen. Early repetitions were faster on several arms, so sustained heat remains material.

Raw: [`ornith-microbatch-memlat.jsonl`](../benchmarks/roofline/ornith-microbatch-memlat.jsonl), [`qwen-microbatch-memlat.jsonl`](../benchmarks/roofline/qwen-microbatch-memlat.jsonl).


## Ornith remaining CPU cost: sampled coding workload

A separate, instrumented run attached perf only after the server's initial warmup, across the three ordinary coding requests, using eight packed threads and MTP off. There were no lost samples. This is mixed prefill/decode and is not a timed A/B or a claim about pure decode alone.

Sampled user-cycle shares: Q3_K×Q8_K dot products26.79%; repacked Q4_K GEMV23.14%; KleidiAI dot-product GEMV13.23%; repacked Q4_K GEMM4.06%; f32 dot products3.89%; memcpy3.21%. Another14.53% was in libgomp offset0x23b20, whose disassembly is a counter/shared-value polling loop. This identifies quantized computation and OpenMP waiting as concrete targets; it does not mean14.53% of wall time can be removed.

Next bounded candidate: lower the OpenMP spin count, preserving threads/weights/outputs, before considering a Q3_K storage-format change under the KLD gate. [GNU libgomp documents GOMP_SPINCOUNT](https://gcc.gnu.org/onlinedocs/libgomp/GOMP_005fSPINCOUNT.html) as controlling active waiting; the unset-policy default is 300,000 spins, while passive waiting uses zero. No system setting or production profile was changed.

Raw summary: [`ornith-coding-perf-summary-memlat.jsonl`](../benchmarks/roofline/ornith-coding-perf-summary-memlat.jsonl); request/check rows: [`ornith-perf-coding-memlat.jsonl`](../benchmarks/roofline/ornith-perf-coding-memlat.jsonl). The binary perf capture is retained in the private experiment folder.


The bounded Ornith descriptor read confirms90 base expert tensors in Q3_K and 30 in Q4_K, all with 268,435,456 elements. The qualified CPU source has a Q4_K i8mm repack path and no Q3_K repack branch. Converting just the Q3_K expert tensors to Q4_K would add3.209 GB on disk (17.438→20.647 GB total) and 0.100 GB to the target active tensor-sweep proxy per token, about 5.6%. This is an **unmeasured candidate**, not a quality-preserving conversion by assumption: it must pass the existing KLD/top-1 gate before any performance promotion. Descriptor/budget row: [`ornith-expert-format-budget.jsonl`](../benchmarks/roofline/ornith-expert-format-budget.jsonl).


## Dense Qwen after memlat

All 84 cases passed across the 14 configurations and three alternating repetitions, using pp128 and tg16 to bound this first baseline. Kernels, f16 KV and batch/ubatch256 match the MoE screens; sequence lengths differ, so this table is not a direct pp512 comparison.

| Threads | Unpinned decode / pp128 tok/s | Packed decode / pp128 | Spread decode / pp128 |
|---:|---:|---:|---:|
|4|5.151 /15.394|3.573 /13.965|5.085 /15.032|
|6|5.405 /21.650|5.176 /19.944|5.403 /20.053|
|8|5.334 /25.635|5.352 /25.301|5.312 /25.120|
|10|5.072 /28.957|5.249 /29.468|5.079 /28.914|
|12|3.752 /30.069|4.430 /30.542|same strict mask|

Six unpinned and six spread were effectively tied (5.4053 and 5.4031 tok/s); six unpinned remains the simple decode candidate. Its15.821 GB/token target-sweep proxy implies85.5 GB/s, about 73.8% of the current115.85 GB/s read ceiling. Four packed threads consistently reached only 3.573 tok/s. Four unpinned varied3.574–5.207, indicating why a single unpinned pass is insufficient. Eight unpinned ranged5.310–5.344; six spread5.373–5.426.

The full matrix took 671.1 seconds, peak sampled CPU temperature98.0°C, minimum host MemAvailable10.93 GiB, no guard failure. Historical dense rates near2.7 tok/s came from different pre-memlat runs/workloads; do not label this a controlled2× whole-task gain. New-boot pp512/4k/16k, greedy output parity and protected-agent qualification remain outstanding. Raw: [`dense-thread-placement-memlat.jsonl`](../benchmarks/roofline/dense-thread-placement-memlat.jsonl).


## Arete after memlat

The bounded followup covered four/six/eight unpinned threads, six spread and eight packed:30 pp128/tg16 cases across three alternating repetitions. All passed. Six unpinned won decode at 5.3581 tok/s [5.2300–5.4058]; six spread reached 5.3208 [5.2926–5.3825]; eight packed5.2241 [5.2215–5.2481]. Four unpinned varied4.2296–5.0377 (median 4.9287). Prefill 128 medians were15.597/21.776/26.246 tok/s for four/six/eight unpinned.

The 16.477 GB/token tensor-sweep proxy implies88.3 GB/s,76.2% of the current read ceiling. The sweep took 270.2 s, peaked at 97.3°C and kept at least10.73 GiB host MemAvailable, with no guard failure. This is a throughput screen, not a model-quality comparison or protected-agent qualification. Raw: [`arete-thread-placement-memlat.jsonl`](../benchmarks/roofline/arete-thread-placement-memlat.jsonl).


## OpenMP wait-policy screen: loss recorded

ABBAAB, three independent server starts per arm, eight packed decode threads, ten batch threads, MTP off. Default GOMP/OMP wait environment versus GOMP_SPINCOUNT=1000; all other request/engine settings unchanged. Each arm ran the three ordinary coding cases. All outputs matched across all 18 responses; the same two cases passed and the known parse_port first-pass failure remained unchanged.

Default per-case median decode was29.74–29.78 tok/s; the 1000-spin arm reached 17.86–17.89. Median total time for the three requests was11.803 s versus 17.966 s. Lowering spin count lost decisively and was not selected. The cycle profile's wait share is not free time: sleeping/waking workers can cost more than polling in this workload. This does not establish an optimal spin count for every model.

Raw: [`ornith-spin-screen-memlat.jsonl`](../benchmarks/roofline/ornith-spin-screen-memlat.jsonl). Production remains unchanged.


The matching quantize and perplexity drivers were compiled against the same qualified CPU archives. An initial link omitted the separate CLI main.cpp; that failure is retained. The corrected build succeeded. A quantizer dry run with explicit original types for all 753 tensors changed exactly the 90 intended base Q3_K expert tensors to Q4_K and no others. No model output was written. [`Build results`](../benchmarks/roofline/cpu-quant-tool-builds.jsonl), [`dry run`](../benchmarks/roofline/ornith-expert-format-dryrun.jsonl). Actual conversion, unchanged-tensor verification, KLD/top-1 comparison and performance remain outstanding.


## Q4_K expert conversion did not qualify

The separate20,646,503,712-byte model was produced in 239.6 seconds. All 753 tensor names/shapes and routing metadata remained unchanged. The 90 selected base Q3_K tensors became Q4_K; all 663 remaining tensors (7,045,968,384 payload bytes), including MTP payload, were independently byte-compared and identical. Only general.file_type changed in metadata. Experimental output SHA256:44f8e532c8c80b5402a0e2fc46870cb1d6ae841f523fd3d34e66d0a3a39da046.

A baseline log-probability reference completed on the first 4096 tokens of the committed mixed prose/Python corpus,512-token chunks, evaluating2040 last-half positions. The converted model's corresponding run stopped after 2/8 chunks under the unchanged sustained-PSI guard: host full avg10 reached 10.51% for10.3seconds, with 7.87GiB MemAvailable and no own swap/OOM. Its last **partial** cumulative statistics were KLD0.03054 and top-1 agreement89.216%; no completed full-corpus KLD/top-1 result exists. This is unqualified, and no speed or quality-preservation claim is made. Use original Compact weights for the production candidate.

[Conversion](../benchmarks/roofline/ornith-expert-format-conversion.jsonl), [tensor verification](../benchmarks/roofline/ornith-expert-format-verification.jsonl), [quality attempt/failure](../benchmarks/roofline/ornith-expert-format-quality.jsonl), [fixed corpus](../benchmarks/roofline/quality-corpus.txt). Candidate remains a separate experimental artifact; the original was untouched.
