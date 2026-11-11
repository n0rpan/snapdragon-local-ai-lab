# NPU measurements — 9 November 2026

On the Yoga Slim 7x 83ED, the Linux v73 path works at conservative mapping sizes. The tiny hybrid model reaches 565–567 prefill tok/s against a **two-thread** CPU control at 193. Large resident matmuls reach roughly 10 effective TOPS after dispatch overhead is removed. Neither result establishes 1,000 tok/s on Ornith or a successful large-model NPU implementation.

Production stays on CPU. Per-layer decode offload is parked by the owner; NPU prefill remains a research option.

## Stage 1: one small model

Memlat boot, kernel 7.3.0-40-omarchy; Llama-3.2-1B-Instruct mixed Q4_0/Q4_1/Q6_K GGUF, SHA256 `fa0390e7c043f89ae1847bd6682d748041a99d4ef3de0e0b27d33b6af97a8be8`. b12184 server SHA256 `af24b15a6c01edbb50e31336f4515537d3407233ba622127ae6720cd2f73f74f` (prebuilt commit c0ab6e5c0). One synthetic 512-token prompt, one greedy output, 2 CPU threads, context 1,024, batch/ubatch 512, FA off. This checks mapping, output and teardown; it is not a comparison with the best CPU thread configuration.

| Arm | Prefill time | Prefill tok/s | Top-1 token |
|---|---:|---:|---:|
| CPU control, 2 threads | 2.659 s | 192.52 | 9013 |
| Hybrid HTP, first successful run | 0.903 s | 566.70 | 9013 |
| Hybrid HTP, verbose placement check | 0.906 s | 565.19 | 9013 |

The verbose loader reports 16/17 layers offloaded, including output plus 15 repeating layers. Logical NPU buffers are 709.80 MiB model + 30 MiB KV + 72 MiB compute, about 812 MiB. CPU retains 215.50 MiB model, 23.62 MiB repack and host compute storage. These are buffer totals, not single mapping sizes. Both native NPU runs exited cleanly, CDSP stayed running, no new kernel device/fault messages appeared, and host available RAM recovered within 128 MiB (about 1 MiB difference on the verbose run).

Explicit configuration: `GGML_HEXAGON_DEVICES=HTP0:0`, `GGML_HEXAGON_MBUF=128M,128M,1G`, VMEM unset, `GGML_HEXAGON_OPFILTER=FLASH_ATTN_EXT`, `--no-host`, `-ngl 16`. The working FastRPC compatibility library is SHA256 `557d88433d3451f2e27b0be2d234b10bd823001b11ef2edd4d83b4b3b80d906f`, with the existing opt-in queue-monitor workaround. No persistent system setting changed.

The initial preflight incorrectly assumed the tied output would stay on CPU. The corrected conservative ledger, including all block tensors, a tied-output copy and 256 MiB extras, is 985.74 MiB, below 1 GiB. The largest source tensor is 205.49 MiB, below the standing 256 MiB cap but above the requested 128 MiB chunk. MBUF is not a promise to split every large tensor. Raw receipts retain both the underestimate and its correction.

## RPC and the 10.8 ms graph cost

| Measurement | Result | Includes |
|---|---:|---|
| Direct FastRPC hwinfo reply, median | 307.50 µs | Synchronous invoke and constant query work, 24-byte reply |
| Direct invoke with PM QoS requested, median | 271.77 µs | Same, 2,000 calls after 200 warm-ups |
| Tiny SUM_ROWS graph, host median | 10.809 ms | Queue submission and completion |
| Tiny SUM_ROWS, DSP profile | 10 µs | Operation execution on DSP |

Direct calls reuse one open handle; open/close (roughly 27–34 / 3 ms) are excluded. The hwinfo method and wire layout were checked against the shipped b12184 stub and matching DSP source. This is a small-method RPC measurement, not a zero-work transport primitive.

Large-graph instrumentation separates setup from each timed call. Graph construction is roughly 0.02–0.08 ms, allocation up to 6.5 ms and upload up to 4.8 ms, once outside the timed region. Async submission takes about 0.01–0.03 ms; nearly all remaining time is in completion synchronization. Linear fits retain an approximately 11 ms intercept. Mapping and graph construction therefore do not explain a repeated 10.8 ms cost. The exact queue/notification cause remains under source investigation; it is not a hardware RPC-latency limit. The required compatibility workaround was not disabled.

## Prefill-shaped matmul ceiling

K=2,048, M=512 (expert-like) or 8,192 (dense), N=512/1,024/2,048. Resident dynamic buffers; five timed graphs after two warm-ups, with 1/4/8/16 distinct output nodes. No output aliasing or omitted computation. The 16-node dense N=2,048 graph would exceed the 1 GiB session budget, so that slope uses 1/4/8 nodes. The largest logical tensor allocation set is about 545 MiB, split by the backend into requested <=128 MiB buffers.

Effective TOPS below is `2*K*M*N / slope_seconds_per_op`; it removes the fixed per-graph intercept. It is an isolated, resident operator rate rather than a whole-model rate.

| Type | M | N=512 | N=1,024 | N=2,048 |
|---|---:|---:|---:|---:|
| q4_0 | 512 | 7.01 | 7.05 | 7.96 |
| q4_0 | 8,192 | 10.14 | 10.18 | 10.34 |
| q8_0 | 512 | 6.19 | 6.84 | 7.23 |
| q8_0 | 8,192 | 10.13 | 10.12 | 10.31 |
| iq4_nl | 512 | 6.50 | 8.25 | 7.97 |
| iq4_nl | 8,192 | 10.05 | 10.11 | 10.44 |

The best measured whole-graph rate is about 8.6 TOPS including dispatch. DSP profile records sometimes cover fused groups of four matmuls: their duration must be divided by four before comparing with a per-op slope. Profile duration and wall slope agree after accounting for grouping. A separate rerun of all three largest dense shapes checked 39 distinct outputs against sampled f64 references: maximum relative RMSE 0.000512 (0.052%). The initial 96-case shape/type screen also passed a 3% sampled relative-RMSE bound. These are arithmetic sanity checks, not a model KLD/quality qualification.

Header-based normalization gives approximately 60.95 million active parameters per Ornith transformer layer, or 2.946 billion including all 40 layers and output head. Ignoring non-matmul work, 6.2–10.4 TOPS corresponds to roughly **51,000–85,000 layer-equivalent tokens/s**, or **1,050–1,770 full-model compute-equivalent tokens/s**. These are FLOP normalizations, not predicted serving speeds. Real expert batches are often much smaller than N=512, the full weights do not fit this session, and window changes, memory reads, routing, recurrent state and dispatch can dominate. No 1,000 tok/s Ornith prefill claim is justified yet.

## Read throughput and shared bandwidth

64 MiB F32 SUM_ROWS, graph sizes 1/2/4/8: wall-time slope 2.940 ms per operation gives **22.83 GB/s**, while DSP profiling gives **22.10 GB/s**. This is effective streaming-read operator throughput, a lower bound rather than a raw DRAM hardware ceiling; reduction work remains. The custom scalar read skel was not built because the Hexagon-capable clang driver is absent. Its installation is with the owner/Claude for approval. No new DSP module was loaded.

A separate SCALE over 32 MiB input and 32 MiB output gives 5.71 GB/s including graph overhead. Eight independent operations per graph raise effective read+write throughput to 27.77 GB/s.

| Concurrent screen | CPU pure reads, 8 packed cores | NPU effective read+write |
|---|---:|---:|
| CPU controls | 106.48 / 106.41 GB/s | — |
| NPU alone | — | 27.77 GB/s |
| Both active | 80.52 / 80.98 GB/s | 28.31 GB/s |

CPU controls use a 2 GiB array and 128 passes with validated checksums. The CPU intervals overlap about 8 seconds of the 15-second NPU run; the NPU rate is a whole-run average, so their sum is approximate. CPU reads decline about 24%. This supports shared-bus contention, not an independent extra memory channel. The requested CPU-decode/NPU/NVMe triple run is deferred following the decision to park decode offload and prioritize the prefill ceiling and Rust qualification.

## Flash implications and limits

Header audit: qwen4exp, 48 blocks, hidden width 2,560, expert FFN 640, 512 experts/top-10. Layer-0 gate/up experts are IQ2_S, down IQ4_NL; dense qkv is [2560,10240] Q6_K. Q4_0, Q8_0, MXFP4, IQ4_NL and Q6_K ran on this v73 device in the initial screen. Four Q6_K cases with K=640 were skipped before allocation because the format requires block-256 alignment. IQ2_S/IQ3_XXS experts do not have a claimed direct Hexagon path.

Small Flash expert/projection calls incur the queue floor even at batches 1/4/16. Per-layer Flash NPU decode remains parked. Prefill may have compute headroom, but no sliding-window implementation, full MoE MUL_MAT_ID throughput screen or large-model mapping was attempted. CPU Rust batched prefill is measured next; practical value must be compared with that improved CPU path.

## Safety and retained failures

Every native operation ran alone under the canonical inference lease, with explicit one-session MBUF, no VMEM probing, the existing watchdog, a 2 GiB host floor and zero service swap. Temporary heap ACLs and the legacy library-path alias were restored on every exit. CDSP is running after the wave; no mapping or kernel fault occurred, so no CDSP reset or reboot was performed.

Retained host-side failures: a lease wait expired before any launch; a missing `env=env` caused driver discovery to fail before a session (corrected retry explicitly approved); suppressed placement logging caused a post-run assertion; SUM_ROWS initially used the dynamic-plugin loader against a static-style backend DSO. The latter was corrected to the exported registry API used by the successful matmul fixture, before any HTP session or mapping. This follows the clarified stop rule targeting device/mapping errors. Every failure remains in its raw receipt.

No successful external armbrusl fork run log exists locally, so no historical 3.5–4.5x claim is repeated. Raw logs and synthetic inputs stay under `local-ai-optimization/npu-stage1-20261109*` and `npu-micro-20261109/`. Scalar receipts (raw rows `benchmarks/roofline/npu-20261109.jsonl`, lab notebook, not in the public snapshot). Experimental source is under [experiments/npu-20261109](../experiments/npu-20261109/README.md); it requires local proprietary/runtime dependencies and is not auto-installed.

## Dispatch-floor follow-up

The env-only checks confirm a per-published-batch cost: eight tiny nodes cost 10.87 ms by default, 86.36 ms with OPBATCH=1. OPPOLL is neutral (10.84 vs 10.86 ms), and DSP op time is just 10 μs. All five runs pass and restore temporary access. [Protocol, limits and numerical evidence](NPU-DISPATCH-FLOOR.md). No production knob is changed.
