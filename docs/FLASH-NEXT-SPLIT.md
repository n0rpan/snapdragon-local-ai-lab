# Flash-Next: preserve the RAM/SSD split during startup

Avoiding eager mmap population reduced model load from **60.89 to 11.70 seconds** in one Snapdragon X Elite comparison. Total time fell from **370.46 to 320.89 seconds**, while request time and sustained generation were effectively unchanged. Both processes produced the same complete 353-token coding answer, passed the independent oracle and released RAM after exit. This pair establishes the startup effect only. The candidate later passed a separate protected full-agent repair and is installed as a selectable planning profile; complete normal planning/image-handoff acceptance subsequently passed separately. [Installed lifecycle evidence](../benchmarks/results/flash-next-normal-lifecycle.json) [Curated evidence](../benchmarks/results/flash-next-demand-mapping.json)

## The model is already split by use

The target is Unsloth's V3.1 `Qwen3.8-Flash-Next-GGUF`, revision `084c631ac221016019d6e0a863afe92d90edd3b0`, `UD-IQ3_XXS`, four files totaling 83,626,438,426 bytes. The label describes a mixed quantization: routed projections use IQ2_S, IQ3_XXS, IQ4_NL and Q2_0; ordinary dense tensors include Q6_K, Q8_0, F32 and BF16. Treating the entire file as uniform three-bit weights would misdescribe its CPU and storage costs. All four file sizes and SHA256 identities are in the evidence.

| Weight group | Stored tensor bytes | Runtime policy retained in both arms |
|---|---:|---|
| Ordinary dense weights | 4,522,949,120 | Anonymous RAM: attention, routers, shared experts, output, hyper-connection weights and ordinary token embedding |
| Routed expert bank | 50,292,326,400 | SSD plus a fixed 12 GiB global LRU cache; O_DIRECT reads on misses |
| N-gram embedding table | 28,800,138,240 | Read-only mmap; actual graph hashes gather rows, with random-access advice |

The earlier 45.4 GB expert estimate covered shard 2 only. Shard 3 contains the final 13 ordinary expert tensors from layers 43–47, adding about 4.87 GB. It is not an unused MTP head. Headers and padding explain the difference between tensor bytes and total file bytes.

The expert cache already adapts to reuse across all layers through global LRU. It has no supported permanent hot-expert list or per-layer quota flag. Shared experts remain resident dense weights. The n-gram table is explicitly mapped because only selected rows are needed. `--row-stream` can serve the ordinary token embedding, but cannot take over this table's indices computed inside the graph; it stayed off in this experiment.

No expert was omitted, replaced or pruned. Both arms kept all 512 experts per layer and the model's native top-10 routing. There was no route-ahead, cold-expert dropping, substitution, predictive expert prefetch or speculative decoding.

## The one-line loader change

In the pinned engine, explicit `dense-mapped` disables llama.cpp's lazy-table mechanism. The loader nevertheless calls `init_mappings(true, ...)`. With no lazy ranges, Linux mmap can use `MAP_POPULATE`, and the mapping code issues `WILLNEED` advice over the non-lazy ranges. This eager work happens before BigMoe installs its selective anonymous-dense and streamed-expert policies.

The isolated llama fork changes only this argument in `src/llama-model.cpp`:

```diff
-    ml.init_mappings(true, use_mlock ? &pimpl->mlock_mmaps : nullptr);
+    ml.init_mappings(false, use_mlock ? &pimpl->mlock_mmaps : nullptr);
```

Mapping remains read-only with the same tensor bytes, offsets and shapes. The change suppresses eager population/prefetch; required pages can still fault in, and BigMoe still loads its anonymous dense set and reads selected expert slices. It changes neither graph arithmetic nor routing. Lower startup reads are consistent with this mechanism; the experiment does not attribute each read byte to a particular kernel operation.

Source bases are BigMoeOnEdge `930b2c9ebd8c2cface6566bd9f11075ac07adfe0` and its llama submodule `7ac98f4a8089a36c0581b7c6e8973417a8fa175c`. See the pinned [engine setup](https://github.com/Helldez/BigMoeOnEdge/blob/930b2c9ebd8c2cface6566bd9f11075ac07adfe0/core/src/engine/session.cpp), [model loader](https://github.com/Helldez/llama.cpp/blob/7ac98f4a8089a36c0581b7c6e8973417a8fa175c/src/llama-model.cpp), [mapping implementation](https://github.com/Helldez/llama.cpp/blob/7ac98f4a8089a36c0581b7c6e8973417a8fa175c/src/llama-mmap.cpp) and [row-table policy](https://github.com/Helldez/BigMoeOnEdge/blob/930b2c9ebd8c2cface6566bd9f11075ac07adfe0/docs/row-gathered-tables.md).

To reconstruct the source experiment, make the change in a standalone llama fork at that pin, commit it there, then advance only a new BigMoe parent's gitlink to that commit. Do not edit the parent vendor tree in place. The observed local fork/parent commits were `04ff7dfdfceee3e34b2d76372db1bc286b8fc179` and `40c0ebb5e766efe5f07a91484abc83114c81735e`; these private provenance hashes are not advertised as publicly fetchable upstream commits. The evidence records the patch, source/build manifests and binary identities. Reconstructing equivalent source does not imply an identical binary from another compiler or build environment.

The build used Release/Ninja, existing tools, one job and these identical runtime flags in both builds:

```text
-mcpu=oryon-1+rng+crc+nopauth+dotprod+i8mm+nosve+nosme
-U__ARM_FEATURE_SVE -U__ARM_FEATURE_SME
GGML_CPU=ON, GGML_NATIVE=OFF, GGML_CPU_ALL_VARIANTS=OFF
GGML_CPU_REPACK=OFF, GGML_CPU_KLEIDIAI=OFF, GGML_OPENMP=ON
BUILD_SHARED_LIBS=OFF, GGML_BACKEND_DL=OFF, BMOE_BACKEND_MODULES=OFF
GGML_VULKAN=OFF, GGML_HEXAGON=OFF, GGML_OPENCL=OFF
GGML_CUDA=OFF, GGML_HIP=OFF, GGML_METAL=OFF
```

CLI, server and tests were enabled; desktop, tools and backend modules were disabled. The build had a 4 GiB memory cap, no swap and a 200% CPU cap. Four model-free software checks passed. Model-based byte-identity gates were built but not executed in that build window; the later real-model pair below supplies its own exact-answer and lifecycle checks. An unchanged GCC 16 warning in server `json_rpc.cpp` remains recorded rather than being mixed into the mapping patch.

## One fresh-process pair

The hardware class was a 12-core Snapdragon X Elite Linux ARM64 laptop with 32 GB RAM, using the CPU only. Both arms used four SSD read workers, four decode/six batch threads, 16K context, microbatch 4096, a 12 GiB cache, F16 K/V, greedy sampling with seed 42 and process-scoped THP disabled. Overlap stayed off. The 156-token synthetic prompt requested one complete coding answer; it was not a multi-turn agent task.

Runtime limits were 600% CPU, `memory.high` 22 GiB, `memory.max` 24 GiB, swap disabled, 160 tasks, a 600-second per-arm deadline and a 1,500-second unit ceiling. Each arm waited for at least 25.5 GiB available before starting, with a 4 GiB available-memory guard. External power remained online in recorded samples. Baseline ran first, candidate second, in fresh native processes.

| Observation | Baseline IO4 | Demand-mapped startup IO4 |
|---|---:|---:|
| Engine load | 60.888 s | 11.700 s |
| Load-phase process `read_bytes` | 101,926,719,488 | 18,622,251,008 |
| Request, excluding load | 307.497 s | 307.123 s |
| Native prefill | 23.571 s | 23.461 s |
| Native generation | 1.245 tok/s | 1.246 tok/s |
| Total including load and cleanup | 370.462 s | 320.890 s |
| Sampled peak process RSS | 22.45 GiB | 18.75 GiB |
| Minimum host available memory | 7.60 GiB | 8.22 GiB |

Load time fell 80.8%, load-phase accounted reads fell 81.7%, and complete elapsed time fell 13.4%. The 0.12% request difference and 0.08% generation-rate difference are neutral observations. Both arms reported the same 77.8% cache hit rate, 57,734.1 MiB of generation reads and 19,960.4 MiB of prefill reads. This supports a startup benefit without evidence of faster sustained decoding.

Both answers had the same SHA256, 353 generated tokens and 1,404 characters. Independent sandboxed checks passed for ASCII ports, Unicode rejection, types, ranges and input nonmutation. EOF, process exit, reaping and RAM recovery passed. Baseline/candidate binary and source identities remained unchanged after the pair.

## Limits of the result

This was one sequential pair without clearing global caches or interleaving repeated controls. Cache/order effects remain, and thermal conditions were measured rather than matched: maximum first-sample sensor readings were 43.9/42.3 C, with recorded maxima 69.6/72.0 C. These are sensor maxima, not controlled temperatures or energy measurements. No broad or multiplicative speedup claim follows.

Linux process `read_bytes` is an accounting observation, not a direct whole-device hardware trace. Sampled peak RSS includes mapped, shared and reclaimable pages; its reduction is not proof of an equal reduction in unique physical memory or required steady RAM. Host available memory is a separate observation. The cgroup peak was cumulative across both arms, so it is not reported as two independent per-arm peaks.

The candidate later completed the protected full-agent repair in 1175.218 s with 12/12 tests and a complete final response. Its admitted planning profile is now installed; the complete normal planning/image workflow also passed its corrected 238.448 s installed rerun. [Installed lifecycle evidence](../benchmarks/results/flash-next-normal-lifecycle.json) Ordinary coding retains explicit MoE. [Final repair evidence](../benchmarks/results/flash-next-final-agent.json), [installed policy](FLASH-NEXT-PLANNING.md). Compare this startup result with [prompt/cache experiments](FLASH-NEXT-PREFILL.md), while keeping their workloads and ratios separate.
