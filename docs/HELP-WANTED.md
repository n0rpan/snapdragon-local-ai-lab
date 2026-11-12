# Help wanted

Concrete problems where outside help would move this project. Each has the context we have, what is already ruled out, a first step and acceptance criteria. Open a "help wanted" issue to claim one or ask questions, and read [CONTRIBUTING.md](../CONTRIBUTING.md) first; the measurement rules there are what makes a result usable.

Effort: **S** about a day, **M** a few days, **L** a week or more. Hardware: what you need to work on it.

---

## 1. Hexagon NPU prefill path

**Effort L · Hardware: Snapdragon X Elite/Plus with a working FastRPC/CDSP stack · Area: NPU, engine**

Prefill is int8-compute-bound on the CPU (MoE ~200–280 tok/s at 2K, Flash ~62–66 tok/s). The Hexagon v73 NPU has far more int8 throughput on paper, but decode on it is not viable here and nothing large has run.

What we know:
- Every published batch costs a floor of ~10.8 ms of wall time regardless of op size (8 tiny ops in one batch: 10.87 ms; one per batch: 86.4 ms). A prefill path must batch whole layers, not ops. NPU-DISPATCH-FLOOR.md (lab note omitted from this release)
- FastRPC mappings failed around 1.07 GB and 2.26 GB on the backends we tried. That is an observed failure, not a proven hard limit: a QAIRT runtime recovered by opening more sessions, and upstream ggml-hexagon now has a sliding-window mapper. [FAILED-PATHS.md](FAILED-PATHS.md)
- A small Qwen3-4B (w4a16) prefill through QAIRT ran ~1,190 tok/s vs ~85 on the CPU (different quantisation, not a controlled pair). NPU-MEASUREMENTS.md (lab note omitted from this release)
- Attention + dense parts are only ~25–35% of active parameters in a 3B-active MoE, so an NPU path that skips the routed experts gives at most ~1.2–1.3× end-to-end prefill [E].
- A crash that leaves DMA buffers mapped can leak several GB until the CDSP is restarted. **Never read the dma_buf debugfs** (it oopsed for us).

First step: a model-free micro benchmark of one Ornith layer's dense + attention prefill shapes (2,048 tokens, real head counts) on the NPU, batch-published, with a go number fixed in advance.

Acceptance:
- [ ] A model-free benchmark with its go/no-go number stated *before* the run.
- [ ] If go: a prefill path for one 35B MoE that is ≥1.3× faster on 2,048 new tokens than the CPU path, interleaved A B B A, with mean KLD within the reorder gate ([RESULTS.md §2.2](RESULTS.md#22-quality-gates)).
- [ ] Clean teardown proven: after a killed process, `MemAvailable` recovers without a reboot.
- [ ] No Qualcomm binaries or SDK files committed; document where to obtain them.

## 2. Image engine: int8 attention and chunk-256 calibration

**Effort M · Hardware: any ARMv8.6 machine with i8mm (Snapdragon X, Apple M2+ under Linux, Graviton 3) · Area: `oxi` (engine repo, image branch)**

The Rust port of FLUX.2 klein is correct (its reference is within its own 1-ulp noise of stable-diffusion.cpp) and its int8 linears are 4–5× faster per DiT block. With int8 attention too, the 512 px image scores **22.75 dB against a gate of 27.81 dB** (2× own noise). With **f32 attention** it passes on seeds 42/43/44 (31.47 / 32.08 / 34.08 dB), at an attention cost not yet timed. The question is how much of the int8 attention speed can be kept inside the gate.

What we know:
- With both linears and attention in int8, the error grows in the double-stream blocks. Chunk sizes 32/32 (20.88 dB) and 64/256 (22.75 dB) both fail.
- `--attn-prec int8|f16|f32` (int8 linears kept) exists; f32 passes the gate, f16 has not been run end to end, int8 fails.
- For 1024 px, sorted chunk-256 activation calibration gives 0.0079–0.0087 relative error against a 0.0198 limit — activation-only; weight-aware calibration is not done.

First step: time `--attn-prec f32` vs `int8` per image at 512 px (interleaved), then try `f16` and per-head / outlier-aware int8 scales for q·kᵀ and softmax·v.

Acceptance:
- [ ] An attention setting faster than f32 that still passes the gate (min(30 dB, own noise − 6 dB)) on seeds 42, 43 and 44 at 512 px.
- [ ] Wall time per image for f32, the new setting and stable-diffusion.cpp on the same machine, interleaved.
- [ ] For 1024 px: a weight-aware chunk-256 calibration with its per-layer error, then the same 3-seed gate.

## 3. Flash: expert prediction at small caches

**Effort M–L · Hardware: 32 GB ARM64 Linux machine with NVMe (prototyping can be model-free on recorded routes) · Area: Flash engine**

Flash decode is ~81 ms/token at a 14 GiB expert cache. Eviction is solved as far as eviction goes: LFU with per-layer quotas, and online policies are already ~2.1× the Belady optimum. The gap is **knowing which experts the next layer/token will need before the router says so**.

What we know:
- A learned lookahead (prefetch from the previous layer's state) reaches ~63% recall. At 18 GiB on a chat workload it cut decode time 13.2% (code: 3.3%). At 14 GiB on the short qualification prompt it was **slower** (84.43 → 87.05 ms/token): speculative prefetches evict experts that are needed.
- A per-layer top-K prefetch issues 48 × K reads per token; K = 1 already wastes ~34–47 reads. A gate must be stated as net ms/token, not recall alone.
- Recorded routes from real sessions can be replayed model-free against the cache simulator (`order_sim` style), so most of the work needs no weights.

First step: on recorded routes, compute the decode misses/token of an oracle-free predictor at pool sizes for 14 and 16 GiB, including the evictions its prefetches cause.

Acceptance:
- [ ] Replay result: fewer total misses/token *including prefetch-caused evictions* at the 14 GiB pool than LFU alone, by ≥5%.
- [ ] Then an engine A B B A at 14 GiB on two workloads (chat, code): ≥3% less decode time on both, identical output text.
- [ ] No change to admission: the prefetch budget fits inside the existing cache.

## 4. A shared-expert verify for MoE speculative decoding

**Effort L · Hardware: any; design + model-free kernel work first · Area: `oxe`, Flash**

Multi-token prediction pays on the 35B MoEs only at depth 1 and does not pay at all on Flash: Flash's head lands 1.92 tokens per verify, but a 2-row verify costs 1.77× a plain step, because the second row routes to mostly *different* experts and Flash decode is bound by expert compute + IO. On Ornith, verify lanes brought a 2-token verify from 1.74× to 1.39×.

The open design question: a verify whose cost grows sublinearly in the number of rows when their experts differ. Ideas we have not tried: draft-aware routing constraints, sharing the expert pass across rows where top-K sets overlap partially, verifying only on the shared expert plus a cheap correction, tree drafts restricted to expert-overlapping branches.

Acceptance:
- [ ] A cost model (rows × expert-overlap → plain-step multiples) validated against measured verify costs on Ornith (1.389× for 2 lanes) and Flash (1.77×).
- [ ] A prototype that verifies 2 rows on Flash at ≤1.45× a plain step, exact (identical tokens to plain greedy) or within the reorder gate.

## 5. Adreno GPU: re-open the no-go?

**Effort M · Hardware: Snapdragon X with a recent Mesa Turnip or the vendor OpenCL stack · Area: research**

We stopped GPU work for LLMs on this laptop. A model-free Vulkan benchmark at an attention-projection shape gave **8.9 GMAC/s (fp16) and 18.5–48.6 GMAC/s (int8)** on the Adreno X1-85, against ~1,500 GMAC/s from the 12 CPU threads; ALU-only ceilings were 97 (fp16) and 289–566 (int8). The driver exposed no cooperative-matrix extension and accelerates only unsigned or mixed-sign 8-bit dot products. Submission is cheap (42 µs), so the limit is arithmetic rate. An earlier 27B Vulkan prefill was slower than the CPU (16.9 vs 27.2 tok/s) and starved CPU memory reads while it ran.

A tuned shader could be several times faster than ours; it would need about 30–70× to matter. Re-opening needs new evidence, for example:

Acceptance:
- [ ] A model-free GEMM at a real prefill shape (e.g. 2,048 × 2,048 × 512, int8 or fp16) on the Adreno at ≥600 GMAC/s, with the shader and driver version, reproducible from a script.
- [ ] Measured CPU memory bandwidth while the GPU kernel runs (the earlier failure mode).

## 6. Numbers from other Snapdragon X laptops

**Effort S · Hardware: any Snapdragon X Elite / Plus laptop on Linux · Area: benchmarks**

Everything here is one Lenovo Yoga Slim 7x. We would like the same measurements on other machines (X1E-78/80/84, X1P, 16 GB / 32 GB / 64 GB), especially:

- CPU streaming read bandwidth with and without memory-frequency votes (is MEMLAT needed on your firmware?).
- `oxe` decode and prefill on Ornith or Qwen3.6 with the same files (SHA-256 from the manifest).
- Throttling: clock per cluster under 2 minutes of all-core load.

Acceptance:
- [ ] A filled [benchmark report](../.github/ISSUE_TEMPLATE/community-benchmark.yml) with device, SKU, RAM, kernel, firmware notes, engine commit, model SHA-256, threads/placement, context, temperature and power source.
- [ ] Interleaved arms where two settings are compared; first run dropped.

## 7. A matched Mac comparison

**Effort S–M · Hardware: an Apple Silicon Mac with ≥32 GB (≥64 GB for Flash) · Area: benchmarks**

We claim nothing against Macs because we have no matched measurement. [`tools/mac_reproduce.py`](../tools/mac_reproduce.py) checks the same model files by SHA-256, runs upstream llama.cpp at the pinned commit (Metal and CPU-only) and MLX, and writes receipts with no serial, user name or path in them. It has never run on a Mac, so the first job is to make it work.

Acceptance:
- [ ] `plan`, `verify-models` and `run` complete on a Mac; fixes sent as a pull request.
- [ ] Receipts for at least one 35B MoE: decode and prefill separately, ≥3 warm runs, chip and RAM stated, power source stated.
- [ ] MLX rows labelled as a different quantisation (MLX does not load GGUF).

## 8. Fuzz the tool-call parser

**Effort S–M · Hardware: none (model-free) · Area: `chat` crate (engine repo), controller**

Agent harnesses depend on the Qwen3-Coder XML tool-call parser. We found real bugs in it in production: Python-style `True`/`False`/`None` in a boolean parameter dropped the whole call; the streamed path once exposed truncated invalid JSON at a token cutoff while the non-streamed path silently dropped the same call.

Acceptance:
- [ ] A fuzz target (`cargo fuzz` or property tests) over tool-call text: random truncation points, nested JSON in string parameters, quoted numbers, Python literals, unicode, unclosed tags, multiple calls per turn.
- [ ] Invariants checked: streamed and non-streamed parses agree; no panic; a schema-typed parameter is coerced or the call is reported as invalid, never silently dropped.
- [ ] Every crash or disagreement found becomes a regression test.

---

Also welcome, smaller: thermal measurements over a long session (we have none), a portable installer for the controller, and making the CI workflow pass from a fresh clone.
