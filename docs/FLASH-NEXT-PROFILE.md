# Flash-Next: measured attribution before tuning

Phase 1 is complete. The corrected protected task passed in **768.123 seconds**, with all 12 independent tests, a complete final response, complete phase counters, explicit unload, child/worker reaping, RAM recovery and resource checks. The Phase 1 measurements below preceded the later I/O experiments, summarized at the end of this report. Baseline repository rollback: `e103673ce75e932882220691862164ea49d0ea45`. Nothing pushed.

[Complete numeric profile](../benchmarks/results/flash-next-profile-agent.json) includes all 48 layers in each phase, every request, per-core estimates, source/model hashes and counter definitions. Raw receipt SHA256: `6b7cd9b6643fe88dff07c81d85b4644fb22deb243e5991b97efeb02c30df5e78`. Offline analysis SHA256: `374da847c928e93f862188dfbd10edf0e9a5a2ae2bd52fffd43402ca24ee633a`.

## Where the full task spends time

| Wall component | Prefill seconds | Decode seconds | Whole-task share |
|---|---:|---:|---:|
| Exposed SSD wait |35.727|65.838|13.22% combined |
| Cache management |17.311|20.839|4.97% combined |
| Remaining graph/compute wall |369.675|241.355|79.55% combined |
| Native phase total |422.713|328.032|97.74% combined |
| Loading, client, verification, unloading and cleanup | | |17.379 s /2.26% |
| Complete load-to-cleanup task | | |**768.123 s** |

The residual is **not directly measured pure kernel wall time**: it includes computation and other untimed graph/scheduler work after subtracting measured exposed waits and management. Measuring every node synchronously would change overlap. The replacement evidence is process CPU core-time plus owned-process user-space PC sampling, kept separate from wall time. Process CPU was **2293.945 core-seconds in prefill** and **909.605 in decode**, approximately 5.43 and 2.77 occupied cores. Core-seconds may exceed wall and must not be stacked onto this table.

Prefill is 55.03% of total; decode 42.71%. Prefill residual is 48.13% of total and is the largest remaining component. Removing all exposed SSD waits with everything else fixed could save at most 13.22% of this task. That is an arithmetic scope bound, not a promised storage improvement; storage changes may also affect management and CPU work.

The native wall partitions reconcile within floating-point precision. Native and trace aligned byte totals match exactly in both phases. Native-versus-graph boundary differences remain visible: 0.45955 s prefill and 0.000902 s decode; neither is silently assigned to SSD or CPU.

## CPU, cache and traffic

| Measurement | Prefill | Decode |
|---|---:|---:|
| New input / generated output tokens |5583|1173|
| Count-weighted projected cache hit rate |15.650%|83.085%|
| Application aligned read bytes |194,287,627,616|195,107,168,672|
| Aligned MiB per phase token |33.188/input|158.626/output|
| Aligned MiB/s across complete graph wall |438.806|567.229|
| Inner positioned-read lane seconds, **nonadditive** |332.290|268.460|
| Outer read-call lane seconds, **nonadditive** |411.995|341.313|

Cache denominators are projection/graph demand lookups. Batch staging shares experts within a prompt batch outside that cache-hit denominator; 15.65% does not imply every input token misses 84% of its experts. All 48 layers, lookups/hits and logical/aligned reads are exported. Alignment adds only about 0.603% bytes.

Inner and outer I/O clocks have different scopes. Pinned `FileReader` source clocks the positioned-read syscall loop before its following copy; the outer expert-source clock includes staging/setup/copy/control elapsed. Their difference is not isolated memcpy CPU. Both sum concurrent lanes and are **not exposed wall wait**.

Owned-child sampling uses `cpu-clock:u` at 49 Hz, without raw stacks/register memory or system-wide capture. Dot-product kernels dominate user CPU; the earlier same-task profile measured 83.37% of prefill and 82.93% of decode samples in the `ggml_vec_dot_*` family. Q6_K×Q8_K was the largest individual family. This family includes F32/BF16/F16 and inlined unpacking; leaf PCs cannot identify a tensor or separate unpack from arithmetic. The corrected run's numeric sampled categories are exported independently rather than copying those earlier percentages into it.

Approximately 1 Hz host counters cover all 12 cores. Per-thread user/system estimates are assigned by graph overlap; retiring-thread tails and interval boundary allocation are limitations. Host busy percentages include other programs and do not prove model thread placement. Native CPU core-time is the stronger phase total. The corrected run additionally saved two owned-cgroup CPU/pressure snapshots: 1627 throttled periods and 33.426 s of aggregate throttle accounting during the client interval, with zero memory-pressure totals. These counters are not another additive wall bucket or a per-phase throttle trace.

## Storage reference and observer control

The separate read-only sequential filesystem reference measured **679.36/2808.46/3985.79 MiB/s** at 1/4/8 blocking O_DIRECT lanes. It read existing pinned-shard bytes in a repeated 4 GiB window, 10 seconds per arm, without writes, cache dropping or raw-device access. Worker cleanup, shard identity and the canonical lease passed. [Reference and limitations](../benchmarks/results/flash-next-storage-reference.json).

These are not verified NAND bandwidth or hardware queue occupancy. Effective application bytes divided by whole graph wall are 15.6%/20.2% of the QD4 reference; those ratios are not SSD utilization or an attainable speedup. Request sizes/order, warm-up, temperature and concurrent compute differ. Process I/O counters also differ from application completed-byte counters.

The fixed 156-input/353-output code screen passed its exact answer and independent oracle. Profiling OFF/ON took 117.735/119.542 s: **1.535% observer overhead**, below the 3% gate. Both arms used identical numerical policy/environment/argv apart from tracing. An untouched original CLI control took 117.404 s and 3.753 tok/s. [Fixed profile](../benchmarks/results/flash-next-profile-fixed.json).

## Historical 1175-second task: faster now, cause unestablished

| Same protected task | Historical | Corrected fresh run |
|---|---:|---:|
| Complete seconds |1175.218|768.123|
| Prefill seconds |445.359|422.713|
| Decode seconds |713.079|328.032|
| Generated tokens |1173|1173|
| Decode read bytes |195,107,168,672|195,107,168,672|
| Decode CPU core-seconds |1956.075|909.605|
| Native decode major-fault counts |7473|7592|

Historical reconstruction: [derived numeric record](../benchmarks/results/flash-next-historical-attribution.json), [qualified task](../benchmarks/results/flash-next-final-agent.json). Historical decode exposed wait was 83.467 s, management 50.348 s and residual 579.264 s; historical prefill direct counters were not exported. That missing historical split cannot be recovered by inventing values from the fresh run.

The fresh run is **1.530× faster /34.64% lower elapsed** than historical, reaching the 1.5× target. Current fixed decode 3.753 tok/s exceeds the 3.414 target against historical 1.707. **These are observations, not a newly established software tuning gain.** The untouched original CLI is also faster now. Decode CPU core-time fell roughly 2.15× while mean occupied cores stayed similar; equal token/read counts and unchanged explicit OMP6/PASSIVE/FALSE settings do not establish the cause. The three independently checked implementation files are also [byte-identical](../benchmarks/results/flash-next-profile-output-parity.json) between historical and corrected runs. Major-fault counts are similar, but counts alone do not measure their service latency. Historical frequency/thermal/contending-work data are insufficient. Do not credit the telemetry fix, RAM reserve change or SSD cache warming with an unexplained difference.

## Exact configuration, memory and provenance

Same four pinned mixed UD-IQ3_XXS shards, 512 experts/native top 10, no dropped/substituted experts, 12 GiB LRU, 4/6 decode/batch threads, IO4, overlap, 4096 microbatch, 16K context, F16 K/V, THP disabled and speculation/thinking off. [Exact models](../dependencies/model-artifacts.json). Flash attention was already enabled by AUTO resolution; [allowlisted feature evidence](../benchmarks/results/flash-next-runtime-feature-status.json). Native prefix reuse and two hybrid checkpoints/minimum step 256 remain configured.

The user reduced live MemAvailable headroom from 4 to**2 GiB**; model admission remains**21.5 GiB**, start threshold 23.5 GiB. The corrected run peaked at 20,616,577,024 cgroup bytes (19.20 GiB), minimum available 6,621,859,840 bytes (6.17 GiB), and recovered 26,899,791,872 available bytes. Swap 0, CPU 600%, bounded unit/process counts, external power and battery guards remained active. Production policy is unchanged pending validated rollout; older 4 GiB proofs are preserved.

The isolated [attribution patch](../patches/flash-next-attribution.patch) and [provenance](../patches/flash-next-attribution-provenance.json) add phase/layer counters without new graph barriers or GGML changes. Native config/params/mapping/file-reader tests and byte-identity gates passed on three tiny architectures with the expert-ready hook required. The [serializer correction](../patches/flash-next-summary-counters.patch) adds four existing prefill counters; actual C++ old/new tests verified existing JSON unchanged plus exact finite additions/zero/cancelled cases. Its build took 7.233 s/355 MiB, changing only one archive member while 32 other members, every engine/GGML object and the CLI stayed byte-identical. [Serializer provenance](../patches/flash-next-summary-counters-provenance.json).

The corrected harness passed 27 tests, including real Unix transport and actual native serializer JSON through final rich-counter persistence, then a model-free provenance preflight. It requires one complete finite rich summary per completed request. Fixed CLI observer evidence is reused only through an explicit serializer-only identity bridge; no forged native proof or historical receipt was changed. Actual corrected server SHA256 is `801dcc1b49e3f841dc3c44f4343292a5e54a225a61c604f99f35bcb494cfc4a2`; the earlier base server hash remains separately identified in the dataset. This diagnostic runtime is not installed.

## Preserved failures and next experiments

The original fixed pair passed OFF but stopped before ON on the older 4 GiB start threshold. A new 2 GiB harness initially failed on missing legacy bindings; a preserved corrected variant resolved it. The first protected attempt stopped on an overlong Unix-socket path before coding; a held-directory-fd alias passed real transport tests. The next task passed in 771.371 s but the recorder discarded its rich phase counters: [preserved incomplete receipt](../benchmarks/results/flash-next-profile-agent-incomplete.json). Own synthetic logs/database did not retain those counters. This successful 768.123 s recollection supersedes its attribution completeness, not its historical task/lifecycle result.

A first build target typo, a missing NumPy fixture interpreter and a partial Git history-copy failure also remain recorded. Existing verified dependencies/current-checkout copies resolved them without changing model policy.

The [ranked plan](PLAN.md) now starts with a cheap IO4/IO8 fixed comparison, then CPU thread tests with the selected I/O policy fixed. Cache pinning, exact prefetch and speculation follow their measured scope and cost. CPU-heavy prefill makes GPU backend feasibility relevant, but direct movable dense/attention time and exact streamed-expert support still require proof. An earlier operational >50%-of-total trigger was stricter than the user's wording and is superseded; 48.1% residual alone neither proves movable GPU work nor justifies discarding the GPU question. KV precision/reasoning quality changes remain separate from lossless I/O tests.


## First post-attribution experiment

The separate [IO4/IO8 ABBA screen](../benchmarks/results/flash-next-io-depth-abba.json) completed with exact 353-token output and all correctness/resource/release checks. IO8 reduced median complete fixed-task time 119.167→110.282 s (7.46%), raised weighted decode 3.711→3.914 tok/s (5.46%), and reduced median prefill 15.260→11.387 s. Exposed decode wait fell approximately 21.16→14.60 s. This is a measured configuration comparison after the attribution gate, separate from the unexplained historical/current difference. The later protected IO4→IO8 pair completed in 768.484→736.838 s (4.118% less total time), with 12/12 checks and complete finals in both arms. Outputs differed, and the gain missed the 5% full-task gate. IO8 remains a provisional CPU-screen baseline; installed IO4 is unchanged. [Protected pair and limitations](../benchmarks/results/flash-next-io-depth-agent.json).

All four arms drew battery assistance: 96.49→95.25% across 7 m 43 s despite external power. The raw power reading was mainly negative; the observed charge decline, rather than a guessed sign convention, flags assistance. Later arms were warmer than the first (maxima across recorded zones 75.7–87.5°C). The ABBA order, both same-direction candidate wins and complete numeric samples constrain order drift but do not provide a confidence interval or perfectly matched thermal conditions.
