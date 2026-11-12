# Flash-Next: SSD concurrency and expert-read overlap

Two separate coding screens found useful scheduling improvements on the Snapdragon X Elite CPU. Four SSD read workers reduced request time by **21.9%** against two workers on the original engine. On the separate demand-mapping candidate, overlapping expert reads with computation reduced a fresh request by **26.0%**, from **307.29 to 227.27 seconds**. Each pair produced the same complete 353-token Python answer and passed independent checks. These single-response pairs establish scheduling effects. The candidate later passed a separate protected full-agent repair and its planning profile is installed; complete normal planning/image-handoff acceptance also passed separately in 238.448 s. [Installed lifecycle evidence](../../benchmarks/results/flash-next-normal-lifecycle.json) The pairs alone do not establish that qualification. [Curated measurements](../../benchmarks/results/flash-next-overlap-code.json)

## Fixed workload and full routing

The synthetic prompt requested two Python functions for parsing one port and a list of ports, including validation and errors. The independent offline sandbox checked ASCII-only decimal digits, Unicode-digit rejection, whitespace, ranges, wrong types, ordering, duplicates and input nonmutation. Each arm received 156 prompt tokens and completed 353 generated tokens, below its 512-token cap. The four answers across the two experiments were byte-identical, 1,404 bytes each; response text is excluded from the public evidence. The overlap pair also independently read and compared both saved answer files after their separate oracles passed.

Both screens used the same pinned Unsloth `Qwen3.8-Flash-Next-GGUF` revision `084c631ac221016019d6e0a863afe92d90edd3b0`, mixed `UD-IQ3_XXS` weights, and CPU-only execution on a 12-core Snapdragon X Elite Linux ARM64 laptop with 32 GiB RAM. All 512 experts remained available and the native top-10 routing was preserved. No expert omission, substitution, route-ahead, prediction, speculative decoding or row streaming was enabled. Model file identities are recorded in the evidence.

The common runtime policy was four decode threads, six batch threads, 16K context, microbatch 4096, a 12 GiB logical expert LRU cache, O_DIRECT reads and process-scoped THP disabled. Ordinary dense weights stayed in anonymous RAM; the large per-layer embedding table stayed mapped. CPU repacking and KleidiAI remained off. The tests changed SSD scheduling; they did not change model weights or reduce the requested expert routing. [RAM/SSD layout and loader change](FLASH-NEXT-SPLIT.md)

## Pair 1: two versus four read workers

This pair used the original engine, with expert-read overlap **off** in both fresh processes. IO2 ran first, IO4 second. [Original worker-count evidence](../../benchmarks/results/flash-next-io-workers.json)

| Observation | IO2 | IO4 |
|---|---:|---:|
| Complete request, excluding load |382.005 s|298.250 s|
| Total including load and cleanup |444.806 s|356.673 s|
| Native prefill |42.168 s|22.871 s|
| Prefill serial flash wait |31.929 s|12.715 s|
| Native generation |1.040 tok/s|1.284 tok/s|
| Generation flash wall wait/token |0.3149 s|0.1723 s|
| Generation expert reads |57,734.1 MiB|57,734.1 MiB|
| Cumulative cache hit rate |77.8%|77.8%|
| Minimum host available RAM |7.63 GiB|8.11 GiB|

Request time fell by 83.75 seconds. The 19.21-second reduction in measured prefill flash wait closely matches its 19.30-second prefill reduction. Generation rate rose 23.5% with the same read volume and reported cache-hit rate. This supports a benefit from read concurrency in this workload; it does not establish that every saved second came from storage.

A 46 MB Git metadata copy occurred during the first arm's startup. Its load/total comparison is therefore confounded, separately from the request measurements. Maximum sampled thermal sensors were 73.3/66.3 C, without matched starting conditions or evidence establishing thermal throttling. No CPU quota throttling was recorded during either remaining-decode phase.

## Pair 2: overlap off versus on at IO4

Both arms used the **same demand-mapping candidate binary**, four read workers and fresh processes. Overlap off ran first, overlap on second. The candidate had already passed its separate original-versus-demand-mapped startup pair. That earlier experiment found a startup improvement and essentially neutral generation; it is not reused as the overlap baseline. [Startup experiment](FLASH-NEXT-SPLIT.md), [overlap evidence](../../benchmarks/results/flash-next-overlap-code.json)

| Observation | Candidate IO4, overlap off | Same candidate IO4, overlap on |
|---|---:|---:|
| Complete request, excluding load |307.293 s|227.270 s|
| Total including load and cleanup |318.470 s|238.349 s|
| First token |24.317 s|21.086 s|
| Native prefill |23.015 s|20.168 s|
| Native generation |1.243 tok/s|1.707 tok/s|
| Generation expert reads |57,734.1 MiB|57,734.1 MiB|
| Cumulative cache hit rate |77.8%|77.8%|
| Generation flash **wall wait**/token |0.1811 s|0.0712 s|
| Generation `io_s_tok` |0.1811 s, serial wall wait|0.3958 s, aggregate lane busy time|
| Generation compute residual/token |0.5797 s|0.4723 s|
| Generation cache management/token |0.0435 s|0.0422 s|
| Generation CPU core-seconds/token |1.8432|1.6352|
| Minimum host available RAM |8.04 GiB|7.84 GiB|

Overlap saved 80.02 seconds of request time, a 26.0% reduction. Total time fell 25.2% and native generation rate rose 37.3%. Prompt processing improved by 2.85 seconds. Both arms still read the same prefill expert volume, 19,960.4 MiB, and generated the same exact answer.

**The two `io_s_tok` values have different meanings.** In serial mode it measures wall time blocked on expert reads. Under overlap it sums the lanes' busy times; simultaneous reads can make that sum exceed elapsed wall time. Use the mode-selected `flash_wait_s_tok` for the wall comparison: 0.1811 to 0.0712 seconds per token. The larger aggregate figure of 0.3958 does not mean overlap caused a larger wall stall. Similarly, overlap prefill reported 42.494 seconds of aggregate lane busy time within 20.168 seconds of prefill wall time; its measured wall stall was 7.366 seconds, against 12.911 seconds of serial prefill wait. [Pinned telemetry definitions](https://github.com/Helldez/BigMoeOnEdge/blob/930b2c9ebd8c2cface6566bd9f11075ac07adfe0/docs/telemetry.md)

The entire improvement cannot be assigned solely to hidden I/O. Generation wall wait fell by 0.1099 seconds per token, **and the compute residual fell by 0.1074 seconds**. That residual includes costs not separately timed and is not a direct matrix-kernel measurement. Summed CPU time also fell. These observations support a scheduling benefit but do not identify every underlying cause. Prefill CPU time moved in the other direction, 69.749 to 91.596 core-seconds, while prefill wall time improved.

No CPU quota throttling was recorded in the remaining-decode phases. The prefill-plus-first-token phase recorded about 0.0045 seconds of throttling with overlap off and 0.5687 seconds with overlap on. These bounded CPU conditions are part of the observation, not an unrestricted twelve-core benchmark. Maximum sampled sensors were 66.3/67.4 C; those measurements do not prove the runs had equivalent thermal behavior.

## Source and validation boundaries

The original source pins were BigMoeOnEdge `930b2c9ebd8c2cface6566bd9f11075ac07adfe0` and llama.cpp `7ac98f4a8089a36c0581b7c6e8973417a8fa175c`. The original CLI SHA256 was `7b1574cafd7ca418d1d3a31a871595adbe0b7f084a48cca1b09ac6f665aece44`.

The overlap pair used the isolated demand-mapping parent `40c0ebb5e766efe5f07a91484abc83114c81735e`, llama fork `04ff7dfdfceee3e34b2d76372db1bc286b8fc179`, and CLI SHA256 `18bfe8ff6ad94fe685945b1aede845b93ece3aaedbaea265b62faca122bbbbac`. The fork changes only the loader's eager mapping-prefetch argument; the parent changes only that llama gitlink. These local provenance commits are not advertised as publicly fetchable upstream commits. The [startup note](FLASH-NEXT-SPLIT.md) describes reconstruction from the public base, and the evidence records source/build/runner manifest hashes.

The candidate's source, binary and passed mapping receipt were verified before and after the overlap experiment. The original mapping validator and shared screen remained unchanged. Each fresh arm inherited THP disabled, completed strict native request/close/EOF handling, exited with code 0, was reaped, passed its offline oracle and recovered RAM within the allowed threshold. No reboot was needed.

Runtime limits were a 600% CPU quota, 22 GiB `memory.high`, 24 GiB `memory.max`, no swap, 160 tasks, 600 seconds per arm and a 1,500-second unit ceiling. An exclusive lease prevented concurrent model jobs; each arm required external power and at least 25.5 GiB available before startup, with a live 4 GiB reserve. Process RSS and host available memory are different observations; neither is interchangeable with unique steady physical RAM. Cgroup peaks and PSI totals span the shared unit and are not presented as independent per-arm peaks.

Each comparison is one sequential pair, without randomized repeats, globally cleared caches or matched power/thermal conditions. The original IO4 control and the later mapping-candidate IO4 control are separate runs with separate baselines. **Do not multiply their ratios**, the startup ratio or earlier JSON/prompt/cache improvements into a general speedup. Complete single-response coding is stronger evidence than a repeated integer array, but it does not qualify multi-turn tools, long-session behavior or a completed OpenCode repair. The pairs alone changed no deployed profile or default. The later protected full-agent qualification and installed policy are recorded separately in [final repair evidence](../../benchmarks/results/flash-next-final-agent.json) and [planning policy](FLASH-NEXT-PLANNING.md); complete normal-workflow acceptance subsequently passed separately. [Installed lifecycle evidence](../../benchmarks/results/flash-next-normal-lifecycle.json)
