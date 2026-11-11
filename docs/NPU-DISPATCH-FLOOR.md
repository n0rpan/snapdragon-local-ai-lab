# NPU dispatch-floor environment checks

2026-11-10, same memlat boot, b12184 Hexagon backend and unchanged monitor-patched libcdsprpc. Five sequential one-session runs under the canonical inference lease and `npu_guard`, explicit MBUF `128M,128M,1G`, VMEM unset. No DSPQUEUE monitor change, DSP rebuild, device/mapping error, recovery or reboot. Temporary DMA-heap ACL and path alias are restored after every run. The fixture validates every output against CPU row sums.

| Check | Median wall time | DSP time / interpretation |
|---|---:|---|
| Tiny SUM_ROWS, PROFILE=1 |10.7786 ms|10 μs per op; batch time reported below|
| Same tiny op, PROFILE unset |10.8601 ms|profiling does not explain the floor|
| Tiny op, OPPOLL=1 |10.8419 ms|−0.17%; host response polling does not remove it|
| Eight independent nodes, default OPBATCH |10.8721 ms|one published batch|
| Eight independent nodes, OPBATCH=1 |86.3570 ms|approximately eight dispatch floors|

The multi-node graph uses a 128-byte source and 1/2/4/8 SUM_ROWS nodes, 20 timed graphs per size after warmup. Default wall medians: **10.8647 / 10.8244 / 10.8574 / 10.8721 ms**. OPBATCH=1: **10.8857 / 21.5592 / 43.4590 / 86.3570 ms**. The latter slope is **10.7902 ms per node/publish**, versus essentially flat default batching. These tiny inputs are deliberately cache resident: any reported bandwidth from their noisy near-zero slope is meaningless.

The profile-enabled tiny fixture uses 200 timed graphs after 20 warmups. DSP op median is 10 μs, while host wall is ~10.8 ms. The verbose per-batch profile is also retained numerically in the [receipt](../benchmarks/results/npu-dispatch-env.json). DSP profiling ends before the suspected trailing idle poll, so wall minus DSP time is not proof that buffer mapping caused it.

**Conclusion:** the expensive boundary is per published batch. OPBATCH=1 is a diagnostic loser; OPPOLL is neutral and adds host spinning. Neither is a production optimization. Batch more useful operations per publish. Source analysis points to the DSP callback's 99 × 100 μs trailing poll inside the synchronous queue-monitor call; these env checks support that explanation but do not time method 9 directly or prove the shipped skel's constants. Do not disable the queue monitor. A skel/compiler or host monitor-pump change is separate work requiring review; no such change is installed here.

Fixture SHA256: `56ead3474d002270d145d3aa14c21039d4f7938174682529dadf8a311de5a155`. Reproduce with the existing `ggml_sumrows_bw`: `--tiny --verbose` for profile, `--tiny` for default/poll, and `--k 32 --rows 1 --runs 20` for OPBATCH default/1. Keep every other environment setting and library identity matched to the last good run; use the existing safety launcher, never raw unchecked device calls.

[Measurements](../benchmarks/results/npu-dispatch-env.json), [earlier NPU limits and throughput](NPU-MEASUREMENTS.md). No model-level speedup or production change is claimed.
