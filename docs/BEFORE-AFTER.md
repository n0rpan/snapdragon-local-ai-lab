# Per-model before/after and exact tested weights

Both MoEs now use installed bbc2990 with VL15, 32K context and MTP depth 1 without a context cutoff. In matched depth0/1 ABBA screens, Ornith decode time fell 11.9–26.0% across 4K–32K; Qwen fell 18.1–23.1% across 2K–16K. Final protected tasks, installed-unit regressions and zero-swap guards passed. [Exact scope, controls and rollback](MTP-CONTEXT.md). Historical tables below retain their original tools and dates.

Ornith candidate 2 is installed as of 2026-11-11. The matched fixed 2K-token controller request fell from 13.267 to 10.164 s (−23.4%). VL15 lowered short-prompt MTP decode time by 21.4%, with identical generated hashes. Both protected tasks and the installed-unit regression passed. [Configuration, controls and measured limits](MOE-CANDIDATE2.md).


These are measured outcomes on a Snapdragon X Elite Linux ARM64 laptop with 32 GB RAM. The tables separate same-weight runtime changes, cache/lifecycle gains, model substitutions and different tasks so each result can be reproduced. Lower seconds and higher tokens/s are better. For the selected/headline models, image assets and ordinary MoE control, exact filenames, publisher revisions, sizes and every model/shard SHA256 are in [model-artifacts.json](../dependencies/model-artifacts.json).

## Stock DT/settings → current setup (matched tracks)

“Stock” in the hardware track means the same 7.3.0-40 Omarchy kernel with its original Yoga DTB (no SCMI protocol0x80) and the earlier4-decode/10-prompt settings. The controlled model tests already used our pinned CPU engine; this is **not** an unmodified upstream llama.cpp binary comparison. Same weights, request fixtures, context and measured boundary are required for each row. Individual links below are measured controls; rows from different tracks must not be multiplied.

| Track / change | Before → after | Isolated measured contribution | State |
|---|---:|---|---|
| MEMLAT DT fix, same six spread readers |48.88 →115.85 GB/s|2.37× CPU streaming-read bandwidth; same binary/2 GiB/32passes/three repeats|Default on this Yoga; automatic edge-update rebuild + stock fallback|
| MEMLAT DT fix, Ornith same 4K server/MTP2/4+10 threads |12.73 →17.18 decode tok/s;94.74 →107.70 prefill tok/s|+35.0% decode,+13.7% prefill; three exact output matches|Installed platform change|
| Threads/placement + MTP off, Ornith same 4K server on MEMLAT |17.275 →29.382 decode tok/s;107.695 →119.858 prefill|+70.1% decode,+11.3% prefill; nine exact output matches per arm|Installed8decode/12prompt; these knobs were measured together|
| Same settings change at actual 16K capacity |16.542 →30.123 decode tok/s|+82.1%; same request harness, startup excluded|Installed; ten-minute decode drift−0.5%|
| Qwen MTP pool-limit control (separate historical engine) |10.98 →16.86 decode tok/s|+53.5%; same MTP weights and experimental binary|Patch carried; not an additional Ornith multiplier|
| Current-boot thread/profile changes, Qwen protected task |119.02 →106.06 s|10.9% less load-to-finished time;13/13 checks|Installed fallback settings; one pair/cache-order limits|
| Current-boot thread/profile changes, Ornith protected task |106.37 →92.69 s|12.9% less load-to-finished time;13/13 checks|Installed default settings; one pair/cache-order limits|
| Current-boot thread/profile changes, dense protected task |401.35 →387.73 s|3.4% less load-to-finished time;13/13 checks|Installed selectable profile; one pair|
| Actual controller/OpenCode Ornith prefix/session reuse |121.41 →63.15 s|48.0% less finished time in cold/cached second-session pair;13/13 each|Installed prefix store; includes load/session cache effects|
| Dense saved prefix across full model reload (earlier fixture) |432.89 →201.60 s|53.4% less finished time|Installed mechanism; separate single-file fixture|
| Qwen saved prefix across full reload (earlier fixture) |264.73 →210.21 s|20.6% less finished time|Installed mechanism; separate three-module fixture|
| Q3_K + exact MoE GEMM |Model KLD0,100% top-token agreement; speed qualification underway|Standalone kernel gains are not whole-model or protected-task gains|Candidate only|
| Broader Q4_K/Q5_K MoE GEMM |KLD0.021678,92.696% top-token agreement on Ornith|Fails the numeric gate; no gain credited|Rejected pending bug fix; production unchanged|

The linked4K Ornith endpoints are **12.73 stock-DT/old-settings →29.38 current-settings tok/s (2.31×,+131%)**, and prefill 94.74 →119.86 tok/s (+26.5%). They use the same model/binary, raw128-token requests and 64-token outputs across the two controlled tracks. This is an endpoint summary of linked measurements at different times, not a new interleaved all-changes A/B; the post-MEMLAT baseline was17.182 in the boot control and 17.275 in the later repeated control. The 16K production server's30.12 is a separate matched-capacity result. **No fully matched stock-DT → current protected-agent total has been measured.**

Ornith becoming the everyday default is a **model choice**, not another same-model speed multiplier. Qwen and Ornith both passed the protected fixture; that small suite does not establish universal quality superiority. Likewise, prefix reuse changes the amount of new work and cannot be added to a decode-kernel percentage. The experimental Q3_K/GEMM work is excluded from installed totals until protected-task and lifecycle qualification pass.

Evidence: [hardware/DT controls](ROOFLINE.md), [matched server + task table](SCOREBOARD.md), [actual controller promotion](../benchmarks/roofline/ornith-production-promotion.jsonl), [kernel microbench](../experiments/q3k-i8mm/RESULTS.md), [model numerical checks](../benchmarks/roofline/q3k-moe-model-kld.jsonl). Credit: the Surface Pro11 memlat work and upstream RFC identified the missing protocol; [Omarchy issue#18254](https://github.com/omacom/omarchy/issues/18254) tracks this Yoga finding. [MEMLAT.md](MEMLAT.md) has credits, exact patch/boot controls, auto-update verification and rollback.

## Finished results at a glance

| Exact profile family | Before/control | After/selected result | What this establishes |
|---|---|---|---|
| Qwen3.6-35B-A3B MTP, normal endpoint repair | Cold 264.73 s | Full reload/prefix restore 210.21 s | 20.6% (about 21%) less task time; same three-module fixture |
| Qwen3.8-27B DevQuasar Q4_K_M, normal endpoint repair | Cold 432.89 s | Full reload/prefix restore 201.60 s | 53.4% (about 53%) less task time; same single-file fixture |
| Qwen3.6-35B-A3B MTP, standalone pool control | Pool limit 1: 10.98 tok/s | Pool limit 3: 16.86 tok/s | +53.5% decode throughput on the same experimental binary; not agent completion |
| Z-Image Turbo Q4_K, native 1024 | Repack off 1226.06 s | Repack on 897.74 s |26.8% less load-to-PNG time in one matched 600% quota pair |
| Z-Image Turbo Q8_0, native 1024 | Repack off 775.52 s | Repack on 488.06 s |37.1% less load-to-PNG time in one matched 1000% quota pair |
| Flash-Next mixed UD-IQ3_XXS, same complete code across SSD stages |444.806 s total|238.349 s total|46.4% lower observed total; matched code, separate sequential stages (limits below) |
| Flash-Next mixed UD-IQ3_XXS, full agent | Earlier 1800 s client deadline failed; no finished-task time |1175.218 s total, 1161.700 s client; 12/12 and complete final response | New full synthetic repair passed; **no speedup ratio against the timeout** |

The selected core MoE also completed the normal installed three-module repair in 264.73 s cold and 210.21 s after full reload/prefix restore. The dense normal single-file port repair completed in 432.89/201.60 s. Those are different fixtures, so they do not rank the models against each other. Actual Image Studio Q8 requests measured 20.07 s for a 256 draft and 526.61 s for a 1024 portrait. [Normal MoE](../benchmarks/results/normal-moe-opencode.json), [normal dense](../benchmarks/results/normal-dense-opencode.json), [Studio](../benchmarks/results/final-image-comparison.json).

The Flash full repair is now qualified as a bounded synthetic task after an audited checker-state reconciliation. The planning profile, settings, controls and UI are installed. Complete normal planning/image-handoff qualification also passed in a separate 238.448 s installed run. [Installed lifecycle evidence](../benchmarks/results/flash-next-normal-lifecycle.json) [Final Flash evidence](../benchmarks/results/flash-next-final-agent.json)

## Fresh Flash attribution controls — software gain not established

| Same workload | Historical reference | Fresh observation | Limit |
|---|---:|---:|---|
| Protected complete agent |1175.218 s|768.123 s, 12/12 checks|1.530× faster observed, 34.640% lower time; same 1173 output tokens and decode read bytes. Not a matched new software A/B. |
| Exact 353-token code decode |1.707 tok/s|3.753 tok/s on untouched original CLI|2.199× higher observed rate. Existing binary now runs faster; cause unestablished. |

The fixed observer pair passed with 1.535% overhead and exact output identity. The first fresh full task passed correctness, unload, RAM recovery and process cleanup, but the harness failed to save rich native phase counters; it is preserved as incomplete attribution. A telemetry-only correction and new recorder passed software/preflight gates; recollection passed in 768.123 s with complete counter reconciliation. No current-phase tuning result is promoted. [Full evidence, source identities and limitations](FLASH-NEXT-PROFILE.md).

## Exact models: use the file and checksum, not just the label

| Artifact used | Publisher repository | Pinned revision / exact publisher file |
|---|---|---|
| Core MoE selected MTP UD-Q4_K_M | `unsloth/Qwen3.6-35B-A3B-MTP-GGUF` | `5bc3e238d916f48a861bac2f8a1990a0e9b7e98d` / `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` |
| Ordinary MoE historical control | `unsloth/Qwen3.6-35B-A3B-GGUF` | `a483e9e6cbd595906af30beda3187c2663a1118c` / `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` |
| Dense 27B Q4_K_M | `DevQuasar/Qwen.Qwen3.8-27B-GGUF` | `b677217fc76a6823c000b4903d3ff2d2665dac16` / `Q4_K_M/Qwen.Qwen3.8-27B.f16.gguf.Q4_K_M.gguf` |
| Flash-Next mixed UD-IQ3_XXS | `unsloth/Qwen3.8-Flash-Next-GGUF` | `084c631ac221016019d6e0a863afe92d90edd3b0` / all four `UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-0000N-of-00004.gguf` shards |
| Image diffusion Q4_K and Q8_0 | `leejet/Z-Image-Turbo-GGUF` | `c61c0e422dc8b541b7548cf33a4ef8302b0f8085` / `z_image_turbo-Q4_K.gguf` and `z_image_turbo-Q8_0.gguf` |
| Image text encoder Q4_K_M | `unsloth/Qwen3-4B-Instruct-2507-GGUF` | `a06e946bb6b655725eafa393f4a9745d460374c9` / `Qwen3-4B-Instruct-2507-Q4_K_M.gguf` |
| Compatible image VAE | `Comfy-Org/z_image_turbo` | `6fc90a3b1b653e935a0d175e260736de25b84df5` / `split_files/vae/ae.safetensors` |

The selected MoE file is 22,663,387,424 bytes, SHA256 `0b21525e972670ed59e1812e170b27c26355381f0656ecc4e25617ece7dac58b`; the ordinary control is a different 22,134,528,992-byte file. The MTP repository name carries the head distinction even though its publisher filename does not contain MTP. The local descriptive name does. Do not substitute the ordinary file when reproducing an MTP result.

The dense file is 16,547,400,064 bytes, SHA256 `9f27cff95dd7cfcb721aa7a581504fdba72d1b9cad7aaf58b4b332f7d38184f3`. Its historical URL used `main`; the pinned revision above was subsequently verified by publisher tree/LFS metadata to contain those exact bytes. This does not claim that the historical download recorded that revision. **The tested dense weights are Q4_K_M; “FP16” in older result names describes a CPU kernel/build, not FP16 model weights.** Unsloth's UD-Q4_K_M is a different artifact and is not a replacement for this control.

The Flash label is mixed: expert projections use IQ2_S, IQ3_XXS, IQ4_NL and Q2_0; dense tensors include Q6_K, Q8_0, F32 and BF16. Its four files total 83,626,438,426 bytes. The manifest includes every shard's filename/size/SHA, plus image diffusion/encoder/VAE checksums and direct pinned publisher links. VAE content equality establishes compatible bytes, not its historical download origin. Model licensing remains separate from the repository's MIT wrapper license.

The manifest covers these selected/headline artifacts and the ordinary MoE control; it is not a catalog of every historical failed experiment. Separate Q4_0, draft/MTP, DFlash, GPU and NPU trials retain their available identities and limits in the [failed-path record](FAILED-PATHS.md) and linked datasets. An absent historical identity is not supplied by substituting a similarly named current download.

## Core MoE: what changed

The engine uses CPU/Oryon quantized kernels, an adaptive small-batch MTP thread-pool patch, four generation/ten prompt threads and batch/microbatch 256/256 in the qualified coding profile. Cross-session leading-prefix save/restore avoids reprocessing instructions and tool definitions after a full unload. Titles were disabled while coding messages and the full tool set were preserved. These are separate mechanisms; no combined kernel speedup is inferred. [CPU patch provenance](../patches/llama-source-provenance.json), [cache lifecycle](../benchmarks/results/controller-cache.json)

| Comparison | Before | After | Boundary |
|---|---:|---:|---|
| Ordinary file versus MTP file + adaptive pool |13.89 tok/s|16.86 tok/s|Different weights; both 7/8 on small fixed suite |
| MTP pool limit 1 versus 3 |10.98 tok/s|16.86 tok/s|Same experimental MTP file/binary; runtime setting only |
| Ordinary versus adaptive-MTP simple repair |about 133 s|about 133 s|No useful completion gain established |
| Protected harder repair, qualified MoE |265.21 s cold|213.70 s after unload/reload|12/12 each; cache/lifecycle comparison |
| Normal installed three-module repair |264.73 s cold|210.21 s after unload/reload|12/12 each; full load/prime counted |

The pool-limit pair used the **same experimental binary**, SHA256 `b8d6df90c7cc5308488f09889af5bc8ed70ed62c3bbab85a3653ef4b8fbdb63b`, and the same MTP weights, four generation/ten batch threads, repack enabled and draft length 2. The limit-1 arm omitted `LLAMA_CPU_SMALL_BATCH_LIMIT`, preserving its stock-like default; the limit-3 arm set it to `3`. This is a runtime pool-selection control, **not an unmodified-stock versus newly compiled patched-build comparison**. The later qualified joint CPU engine has a different binary identity, recorded in its coding results; the standalone pool ratio is not its finished-agent gain.

The harder same-fixture dense controls timed out at 900 s with 11/12 tests in the frozen patch. That supports the default choice on this task, not universal model superiority. Dense plan→MoE build→dense review passed in 1242.99 s, a separate three-role workflow, not an optimization of the single-model 265 s repair. The normal MoE ASCII/Unicode port repair also passed in 177.99 s; a separate direct-API screen still failed Unicode rejection. Both outcomes remain in the evidence. [Hard comparison](../benchmarks/results/final-coding-comparison.json), [port repair](../benchmarks/results/normal-moe-port-opencode.json), [one-shot failure](../benchmarks/results/portable-moe-screen.json)

### Fresh current-profile MoE reference

The new normal-controller `atomic-reservations-v1` reference finished **488.191 s and 184.589 s**, both 12/12 with complete finals and successful unload/RAM/lease checks. Both first turns already restored 5490 cached/60 new tokens (98.919%): first backend-cold/prefix-restored, then full unload/reload/prefix restore. Outputs 2670/1816 tokens and 11/7 steps differed; this is a current reference with variance, not an optimization A/B. Memory.high increased 89,590 times with nonzero PSI despite zero max/OOM events. Normal CPU 800%/24 GiB-high/25 GiB-max settings remained unchanged and are not matched to Flash's resource allocation. [Current model/settings, pressure and limitations](MOE-NORMAL-REFERENCE.md), [numeric evidence](../benchmarks/results/moe-normal-reference.json).

## Dense 27B: what improved and what did not

| Same-weight observation | Before | After | Interpretation |
|---|---:|---:|---|
| Early short prefill, original versus pure CPU/ten batch threads |6.74 tok/s|19.61 tok/s|Historical 74-token prompt; not an agent speedup |
| Two-request prefill, microbatch 64→256 |14.88 tok/s|21.08 tok/s|Same 4267 new input tokens; preliminary, power/reboot confounded |
| Verified simple repair, original versus FP16-kernel build |547.96 s|536.85 s|2.0% observed reduction; one trial cannot establish repeatability |
| Cold versus new session on resident backend |425.22 s|155.29 s|95.4% first-turn prefix reuse; no unload between these two sessions |
| Normal installed repair, cold versus full reload/restore |432.89 s|201.60 s|Separate single-file fixture with protected assertions |

The useful changes were prompt batching, four generation/ten batch threads and prefix reuse across isolated sessions/full reloads. The larger thread/affinity sweep found no clear gain worth carrying forward. The joint KleidiAI/BF16/Q4_0 experiment did not earn promotion on finished-task time; Q4_0 and draft-head files are separate artifacts, not the selected DevQuasar Q4_K_M above. Both original and joint dense builds timed out on the harder 12-test repair. [Measured components](../benchmarks/results/before-after.json), [preliminary screen](../benchmarks/results/cpu-screen-preliminary.json), [hard task](../benchmarks/results/final-coding-comparison.json)

## Images: matched repack gains, separately for Q4_K and Q8_0

The same-weight controls enable CPU weight repacking and the supported dispatch path, retaining weights, fixed synthetic scene, seed 42, eight steps and ten CPU threads. Native 1024 Q4 used 600% CPU quota; native 1024 Q8 used 1000%. Quota was held equal within each pair. Do not use those two pairs to rank quants.

| Exact diffusion quant / workload | Repack off | Repack on | Less load-to-PNG time | Samples per mode |
|---|---:|---:|---:|---:|
| Q4_K 256×256 |54.92 s|37.21 s|32.2%|2 |
| Q4_K 1024×1024,600% quota |1226.06 s|897.74 s|26.8%|1 |
| Q8_0 256×256 |59.88 s|28.05 s|53.2%|2 |
| Q8_0 1024×1024,1000% quota |775.52 s|488.06 s|37.1%|1 |

These CLI times include engine load, text encoding, sampling, VAE decoding and PNG output. One prompt/seed had limited visual checks; outputs differ and broad quality equivalence is not established. An older Q4 native run already took 893.04 s under different resource conditions, so the 26.8% matched gain is **not** a 26.8% claim over the earliest application configuration. [Paired evidence](../benchmarks/results/before-after.json), [native Q8](../benchmarks/results/native-q8-runtime.json)

Actual Studio requests, with repack enabled, 1000% quota, CFG1, diffusion flash attention, tiled VAE 512 and image metadata off, measured:

| Real app request | Q4_K | Q8_0 | Meaning |
|---|---:|---:|---|
|256×256 bicycle draft |26.36 s|20.07 s|Different quantized weights, same app workload |
|1024×1024 portrait |615.04 s|526.61 s|Q8 selected after limited two-scene visual review |

Q4 remains available for lower RAM use. Sampled peak RSS in the actual native app pair was 7.95 GiB Q4 versus 10.43 GiB Q8. RAM recovered and no image job remained active. These app results are model/quant selection, not another same-weight repack speedup. [Studio comparison](../benchmarks/results/final-image-comparison.json), [image source/asset identity](DEPENDENCIES.md)

## Flash-Next SSD/RAM: measured components and the completed full task

The engine leaves the full expert bank on SSD, stages misses into a 12 GiB LRU cache, keeps ordinary dense weights in RAM and maps the large per-layer embedding table. Full native 512/top 10 routing stays intact. Changes included process-scoped THP disable for memory headroom, wider prefill sharing, cache 8→12 GiB, four SSD read workers, suppressing eager mapping population, and overlapping expert reads with compute. The final agent retained the same four slim tool schemas and checkpoint 2/minstep 256 as the earlier failed retry. [Memory policy](STREAMING-MEMORY.md), [startup split](FLASH-NEXT-SPLIT.md), [SSD scheduling](FLASH-NEXT-SSD-SCHEDULING.md)

| Separate fixed-input pair | Before | After | What can be claimed |
|---|---:|---:|---|
|350-token prefill, microbatch 128→512 |101.02 s|53.80 s|1.88× prefill; three-token answer only |
|922-token prefill, microbatch 512→4096 |140.14 s|102.97 s|1.36× prefill; different pair |
| Repetitive 246-token JSON output, cache 8→12 GiB |1.752 tok/s|2.447 tok/s|Structured-output screen, not coding throughput |
| Complete 353-token code, IO2→IO4 |382.005 s request|298.250 s request|21.9% less request time; original binary |
| Same code, eager→demand mapping at IO4 |60.888 s load|11.700 s load|80.8% less load; request/decode neutral |
| Same mapping candidate/code, overlap off→on at IO4 |307.293 s request|227.270 s request|26.0% less request; 1.243→1.707 tok/s |

### Current data-gated I/O-depth screen

After complete attribution, IO4/IO8 in order 4/8/8/4 reduced median fixed-task 119.167→110.282 s (**7.46% less time**) and increased weighted decode 3.711→3.914 tok/s (**5.46%**). Median prefill 15.260→11.387 s improved 25.4%. All four outputs were the same 353-token tested code; the separate fresh full-agent result follows below. Every arm used battery assistance and later arms warmed, both flagged in [the complete numeric screen](../benchmarks/results/flash-next-io-depth-abba.json). These are separate current controls, not a ratio to multiply into older stages.

The subsequent **fresh protected IO4→IO8 pair finished 768.484→736.838 s (4.118% less time)**, with 12/12 tests, complete finals and all cleanup/RAM/resource checks in both conditions. Prefill 423.271→410.809 s and decode 332.476→312.612 s included exposed waits 34.627→10.227 s and 65.341→44.797 s. Outputs differed 1193→1173 tokens, so the whole-time change is not a pure I/O causal estimate. Thermal maxima across sampled zones were 96.9/98.1 °C; net battery percentage rose in both, with transient IO4 declines. **Below the 5% full-task gate: IO8 is only a provisional CPU-screen baseline; installed IO4 is unchanged.** [Complete protected-pair counters, provenance and limits](../benchmarks/results/flash-next-io-depth-agent.json)

### Observed first-to-last SSD tuning total: same code fixture

The first IO2/eager-mapping/overlap-off arm took **444.806 s total**; the last IO4/demand-mapping/overlap-on arm took **238.349 s total**. That is **46.4% lower observed total time**, computed directly from those two recorded totals, not by multiplying the intervening component ratios. Their request-only times were 382.005 and 227.270 s (40.5% lower observed). [First worker-count pair](../benchmarks/results/flash-next-io-workers.json), [final overlap pair and original IO control](../benchmarks/results/flash-next-overlap-code.json)

The fixture is exactly matched: the same 156-token coding prompt, all 353 generated tokens and the byte-identical 1,404-byte answer with independently passed ASCII/Unicode/range/type/nonmutation checks. Weights, native 512/top-10 routing, 16K context, 4096 microbatch, 12 GiB expert cache, decode 4/batch 6, O_DIRECT and process THP disabled stayed fixed. The combined changes were **IO workers 2→4, eager mapping population disabled in the separate one-line loader fork, and expert-read overlap off→on**.

This is a comparison of the **first recorded arm of the first pair with the last arm of a later pair**, not a repeated combined A/B. The binary/source changed for demand mapping, intervening tests/build activity occurred, global caches were not cleared, and thermal/power/order conditions were not matched. A 46 MB Git metadata copy also occurred during the first startup. The exact fixed code controls output-length/correctness variation; it does not remove those experimental confounders. The 46.4% observation is not a universal speedup, an independent combined optimization estimate or a completed-agent comparison. Normal workflow verification subsequently passed separately; it supplies no new controlled speedup ratio.

The overlap `io_s_tok` sum of lane-busy time is not wall wait. Use `flash_wait_s_tok`: 0.1811→0.0712 s/token; the compute residual also fell 0.5797→0.4723 s/token. Do not attribute all savings solely to hidden I/O. Nor should these separate component ratios be multiplied.

| Full protected OpenCode repair | Earlier retry | Final candidate |
|---|---|---|
| Client window |1800 s|1800 s |
| Completed task time |**No completed time: deadline failure**|1175.218 s total; 1161.700 s client |
| Correctness |Correct artifact observed while final response remained unfinished|12/12; all three modules changed; protected files unchanged; complete 929-character final response |
| Native model load |60.514 s|8.864 s |
| Lifecycle/privacy |Cleanup and RAM recovery passed|All cleanup/EOF/reaping/RAM/privacy/resource gates passed |

This is a transition from an unsuccessful bounded workflow to a completed one, not a measured percentage speedup against a completed baseline. The old 1864.24 s cleanup-inclusive elapsed time and 1818.14 s early artifact observation are not converted into successful completion times.

The final raw checker initially reported overall false because it expected the controller label `unloaded`; the native server correctly reported HTTP 200/state `empty`. A **new** strict derived receipt preserved the original failed record, checked pinned unload/session-reset/model-null source semantics, repeated current binary/source gates, re-parsed completion and reran the original protected 12-test oracle offline. No inference was rerun and no native timings changed. Original SHA `37230ee4…`, derived `ba3e3aa0…`, proof `c62ec204…` and admission review `8411631b…` are recorded in the [curated full-task evidence](../benchmarks/results/flash-next-final-agent.json).

Raw peak RSS 21,159,288,832 bytes and actual final cgroup peak 20,843,126,784 bytes remain unchanged. The conservative resident estimate is 22,776,560,128 bytes. Admission review changed only rounding granularity from 1 GiB to 512 MiB, producing a 21.5 GiB budget with the same 4 GiB host reserve; this is not a measured RAM reduction or another model run. The profile and controls are installed; complete normal planning/image-handoff qualification passed separately. [Installed lifecycle evidence](../benchmarks/results/flash-next-normal-lifecycle.json)

## Reproducing and interpreting the measurements

Use the exact artifact manifest, engine commit/patch provenance and per-row settings, not current publisher main branches or a similarly named quant. The original/core CPU engine is pinned in [llama provenance](../patches/llama-source-provenance.json); image source is pinned in [SOURCE-PROVENANCE.md](SOURCE-PROVENANCE.md); Flash's isolated mapping fork and all binary/source/manifest hashes are in its datasets. Compatible source reconstruction is not a claim of bit-identical binaries across toolchains.

All figures are workload-specific. Historical pairs were not uniformly charger/thermal controlled; external power did not prevent battery assistance. Single pairs, OS cache/order, prompt/output length and different deadlines limit generalization. Correct source snapshots are not completed agent sessions, prefix caching is not kernel acceleration, and different models/quants are not same-weight optimization controls. See [benchmark method and portable runner](BENCHMARKS.md) and [preserved failed GPU/NPU/speculation paths](FAILED-PATHS.md). Model weights and raw prompts/images/logs are not bundled in this MIT repository.


## Patched Ornith installed, separate workload tracks

The final Q3_K/MoE GEMM/cache engine is now installed on MEMLAT, with a retained stock-boot old-engine fallback. Matched direct protected repair:99.77→64.00s (35.8% less elapsed),13/13 and 6 tools in both. Earlier89.90s baseline is retained; startup/cache variation is not hidden. Real controller is a different9-tool track: cold 121.41→113.77s (6.3% less); cached 63.15→64.56s (2.2% slower in one pair, no verified warm-session gain). Controller uses 864 new SDK input/1,104 output, whereas direct uses 6,066/479; controller's5,487-token prefix is handled outside the SDK's new-token count. [Installed settings, phase limits and rollback](SCOREBOARD.md#installed-patched-ornith-engine-2026-11-09).


## Integrated Rust server candidate, 2026-11-10

Three-trial protected-task medians: Ornith 77.70 → 60.42 s (22.2% less); Qwen MTP2 75.31 → 67.42 s (10.5% less). All 13/13 checks pass, native-reference greedy prompts match, no OOM/model swap. Source b477aec remains staged; no production switch. Recommend Qwen MTP off from the separate prompt-cost isolation; emitted-text differences are under investigation. [Full timings, token counts, sampling and limitations](RUST-SERVER-QUALIFICATION.md).

## New experimental attention step (not installed)

A separate same-binary Rust screen improves 6K-prefix + 2K-extension prefill by 29.9% Ornith / 31.9% Qwen, and 16K decode by 22.2% / 13.8%. The repeated short-context Qwen check is flat at 1K/2K and +4.5% at 4K. These ratios are not stacked onto the installed stock-to-today chain. [Exact versions, receipts and limits](RUST-ATTENTION.md).

## Qualified integrated Rust package candidate

Not installed. Separate matched Rust-build upgrade: −13.2% Ornith / −11.6% Qwen task medians (n=3). Separate fair warm-prefix production CPP→Rust pairs: 30.24→23.42 s Ornith, 70.54→29.33 s Qwen (n=1; different trajectories, Qwen CPP is the older installed fallback). These are not multiplied into stock-to-today ratios. [Cold/warm measurements and memory](RUST-SERVER-UPGRADE.md).


## Rust production installed (2026-11-10)

Approved `9d9ccbe` / server `649ac806` is installed on MEMLAT: Ornith default, Qwen selectable, both MTP off, tiled threshold 0, persistent prefix store. Both chats and the real Ornith OpenCode task pass (7 tools, 13 checks, 55.39 s functional smoke). Existing matched qualification numbers stay unchanged. Hard guards active, model swap zero; dynamic memory.low remains separate. [Deployment and rollback](RUST-DEPLOYMENT.md). No Flash prototype timing is promoted as a model result.
