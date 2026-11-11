# Flash prerequisites on the memlat boot

2026-11-10, Yoga Slim 7x 83ED, kernel 7.3.0-40-omarchy, BIOS NHCN62WW. No tuning or production change in P1/P4.

## P1: fixed-code baseline

Existing `ascii-port-code-v1`: 156 prompt tokens, 353 generated tokens, original answer SHA `460169e27739a86fbb94340a3d7ffcbace214719d46fcda2cc8bb3d0fc0a4093`. Independent Python oracle passes. Same instrumented CLI `ba88f3ba5f20`, 4 decode / 6 batch threads, IO4, 12 GiB cache, 16K context, ubatch 4096, F16 KV, native top-10 / all 512 experts, no speculation, overlap on, anonymous dense weights, mapped N-gram table, THP disabled. OpenMP 6 / PASSIVE / FALSE. Native CLI checkpoints default to 8 (installed server pins 2); this repeats the historical CLI screen, not the full installed HTTP path.

| Measurement | Result |
|---|---:|
| Decode |4.057 tok/s|
| Prefill |14.582 s /10.7 tok/s|
| Request |101.789 s|
| Time to first token |14.985 s|
| Load |11.414 s|
| Whole arm including close/oracle |114.844 s|
| Peak cgroup memory |18.54 GiB|
| Own swap /OOM /max events |0 /0 /0|
| Expert cache hit |77.8%|
| Exposed SSD wait |59.2 ms/token|
| CPU compute /cache management |171.2 /16.2 ms/token|

The older untouched CLI screen recorded 3.753 tok/s and 117.404 s complete time. This new single run is about 8.1% higher decode, 2.2% lower complete time; it is an observational cross-boot comparison, not an isolated memlat effect. Compute remains the largest reported decode bucket. No claim of a tuned Flash speedup yet. The canonical lease was held throughout load/run/unload. [Current raw-metric receipt](../benchmarks/roofline/flash-post-memlat-baseline.jsonl), [historical fixed screen](../benchmarks/results/flash-next-profile-fixed.json).

## P4: header audit

Four verified-download shard identities, 1,224 tensors. The first bounded-window parser could read a small leading portion of tensor data. It has been replaced by an unbuffered exact-header reader with a synthetic boundary check; final counts are re-audited with that reader. The 84 GB payload was not rehashed. Architecture qwen4exp, 48 blocks, hidden 2560, 512 experts / native top-10, expert FF 640. Stored routed experts 50.292 GB; other tensors 4.523 GB (includes the 0.521 GB token embedding, whose whole storage is not read per token); N-gram table 28.800 GB. Native top-10 routed weights total 0.982272 GB/token before runtime layouts,cache effects and other state. Types include IQ2_S,IQ3_XXS,IQ4_NL,IQ4_XS and Q2_0; no assumption of uniform IQ3.

**N-gram geometry correction:** `per_layer_token_embd.weight` is IQ4_NL `[160, 320001536]`, 90 bytes per physical row. Metadata has 16 head offsets/vocabulary sizes near 20M each. The earlier design estimate used 2560 values/row; that address/traffic arithmetic must be corrected before implementing pread. Runtime physical pages touched and residency remain unmeasured.

[Full header audit](../benchmarks/results/flash-post-memlat-header-audit.json), [reproduction tool](../experiments/flash-memlat-20261110/README.md).

## Next

Integrated Rust resident-model qualification takes priority. Then P2 captures expert IDs only (no prompt text) for consecutive-token overlap/lookahead recall, and P3 checks NVMe queue depth plus dm-crypt CPU cost. Flash retuning and R4/MTP come after the baseline, keeping the full-quality path and existing fallback.


## P2: exact-answer expert-ID trace

The existing `--route-trace` diagnostic was sufficient: no engine patch/build. A private FIFO reader discards weights and model strings, retaining only turn/phase/position/layer/rank/expert IDs. All 242,770 records are synthetic; the 353-token answer SHA and independent tests remain exact. Peak 19.15 GiB, zero own swap/OOM. Tracing adds graph nodes/barriers, so its timing is not a throughput result.

All 353 decode positions contain 48 layers × native top-10. Prefill has 7,333 cells rather than 156×48: the final MoE layer computes only the final prompt position (normal unused-output cropping). Consecutive-token expert-set recall is **34.38%**. This measures persistence, not an early-router predictor; early-router logits/accuracy remain unmeasured.

| Offline LRU budget | Decode hit | Expert payload read/token |
|---|---:|---:|
|8 GiB|72.44%|256.99 MiB|
|12 GiB|82.51%|162.57 MiB|
|13 GiB|84.50%|144.04 MiB|
|14 GiB|85.94%|130.59 MiB|
|16 GiB|88.70%|104.67 MiB|
|20 GiB|92.96%|65.41 MiB|

The 12 GiB replay's overall hit rate 77.81% matches native rounded 77.8%; read projections exclude alignment padding. Larger budgets are projections, not admitted/promoted settings. A 16 GiB cache would cut projected expert payload reads 35.6%, but this is not a 35.6% generation gain: CPU compute remains the largest bucket, and the 21.5 GiB admission still applies. The trace plus union-of-expert estimates are ready for the Flash M0 prototype. [Locality/cache metadata](../benchmarks/results/flash-post-memlat-route-locality.json), [native trace receipt](../benchmarks/roofline/flash-post-memlat-routes.jsonl). Full ID sequence stays in the private handoff/ops backup.


## P3: encrypted-storage queue depth and CPU cost

Thirty read-only random-read arms on the existing 49.95 GB GGUF, direct I/O, 2 seconds × 3 repeats per point, reverse queue order in repeat 2. No raw block-device reads or cache drops.

| Queue depth |512 KiB requests |2 MiB requests |
|---:|---:|---:|
|1|0.604 GB/s|1.517 GB/s|
|4|1.959 GB/s|3.698 GB/s|
|8|3.081 GB/s|4.533 GB/s|
|16|4.008 GB/s|4.605 GB/s|
|32|4.380 GB/s|4.589 GB/s|

Medians of 3. Two-MiB expert records reach 98.4% of the measured peak at QD8; smaller projection reads benefit from more outstanding I/O. This supports coalesced expert records plus moderate queue depth for the future engine; no native configuration was promoted.

Whole encrypted-filesystem path CPU cost at 2 MiB/high depth is about **0.38–0.40 kernel core-seconds/GB** after subtracting the one-second idle control. At 4.6 GB/s that is roughly 1.8 kernel cores, before inference/user-space cache work. Physical NVMe read counters track 98.98–99.72% of fio bytes (parent counted once), so the reported rate is not a RAM-hit result. This is achieved btrfs/dm-crypt throughput, not an independently measured raw-media limit.

A separate 2-second kernel-only CPU-clock profile finds AES decrypt leaf functions. Sample-period attribution gives about **0.30 core-seconds/GB** to those leaves. This is approximate and does not isolate the cipher from all mapping/checksum work. The instrumented profile ran at 3.64 GB/s; it is excluded from the throughput table. No stacks/registers/user-memory were collected. The first report parser rejected perf's extra IPC columns; existing samples were re-parsed without another measurement. [Kernel symbols](../benchmarks/results/flash-post-memlat-storage-kernel.json), [summary](../benchmarks/results/flash-post-memlat-storage-summary.json), [all 30 guarded runs](../benchmarks/roofline/flash-post-memlat-nvme.jsonl).

P1–P4 now have their requested baseline, header, route/locality and storage inputs. Actual early-router prediction accuracy remains unmeasured; P2 provides a persistence baseline only. Native/Rust Flash optimization and M0 remain next, not proven by these projections.
