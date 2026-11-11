# Lab notebook: the README of 2026-11-11 morning

> Kept for its links into the earlier llama.cpp-era evidence. Current results are in [RESULTS.md](RESULTS.md) and the [README](../README.md); where they differ, those win. Some rows below describe states that have since been superseded (for example the Flash engine, which is now the Rust engine).

## Stock → today

**A lot more AI from the same 32 GB Snapdragon laptop.** No cloud inference in these runs.

| Measured track | Before → after | What that means |
|---|---:|---|
| CPU memory reads |48.88 → 115.85 GB/s (**2.37×**)|Matched six-thread spread test; MEMLAT DT fix. [Evidence](MEMLAT.md)|
| Ornith, actual server decode |12.73 → 29.38 tok/s (**2.31×**)|Same 4K request track, linked steps: stock DT/settings → MEMLAT/tuned settings; not one combined A/B. [Step-by-step](BEFORE-AFTER.md#stock-dtsettings--current-setup-matched-tracks)|
| Ornith, CPU kernel screen (historical snapshot) |pp512:119.3 → 185.0 tok/s (**+55%**); tg128:28.9 → 31.2 (**+7.8%**)|Native bench ABBA of the earlier exact-mode snapshot, unchanged weights. The installed successor also adds cache/no-prefetch; these are not its fresh native-bench numbers. [Exact snapshot/method](../experiments/q3k-i8mm/RESULTS.md)|
| Ornith, patched llama.cpp **fallback installed** |99.77 → 64.00 s (**36% less time**)|Refreshed direct-server pair,13/13 independent checks and 6 tools each; includes load. Earlier reference variation is disclosed. Installed on the MEMLAT boot. [Evidence](SCOREBOARD.md#patched-engine-qualification-checkpoint-2026-11-09-not-production)|
| Dense 27B, persistent agent prefix |432.89 → 201.60 s (**53% less time**)|Same repair after a full model reload with the saved prefix; a warm-prefix benefit. [Evidence](BEFORE-AFTER.md#dense-27b-what-improved-and-what-did-not)|
| Z-Image Turbo Q8, 1024px |775.52 → 488.06 s (**37% less time**)|Matched CPU repack off/on, same weights and resolution, one pair. [Evidence](BEFORE-AFTER.md)|

**Installed Rust engine (MEMLAT):** three-trial upgrade medians are Ornith **60.06 → 52.13 s** and Qwen **65.53 → 57.91 s** versus our earlier Rust build. Fair **warm-prefix** sessions against installed llama.cpp take **30.24 → 23.42 s Ornith**, **70.54 → 29.33 s Qwen** (one session each; versions and token counts disclosed). All protected checks pass. [Cold/warm comparison, memory and limits](RUST-SERVER-UPGRADE.md).

**Attention evidence, now included in the Rust profile:** same-binary tests improve 2K prefill on a 6K prefix by **30–32%**, and 16K decode by **14–22%**, across Ornith/Qwen. Short-context Qwen losses did not reproduce in warm repeats. [Measurements and limits](RUST-ATTENTION.md).

**Estimate from linked measurements: about 2.5× Ornith decode versus the stock DT/settings, including the earlier measured kernel uplift.** Method: the 2.31× linked server track × the 1.078× native kernel uplift. Different contexts, tools and timestamps make this an estimate, not a measured combined speedup or a claim about protected-task time. [Inputs and limits](../experiments/q3k-i8mm/RESULTS.md). The single matched stock/current run will replace this estimate immediately before public launch, after all optimization work.

These are separate tracks; do not multiply unrelated model, MTP, prefix or image gains. **Installed**: MEMLAT, tuned CPU profiles, Rust Ornith default/Qwen selectable with persistent prefixes, the patched llama.cpp fallback/repack cache, lazy load/unload and the Omarchy panel. Stock-boot fallback keeps the old engine and settings.

Earlier fixed-workload evidence: the standalone Rust engine reaches 42.06 vs 30.52 decode tok/s on one synthetic code request; experimental Qwen MTP reaches 43.05 tok/s with identical output to native MTP-off. Prefill results and arithmetic differences are reported separately in [RUST-M3](RUST-M3.md). The [bounded NPU measurements](NPU-MEASUREMENTS.md) show working small-model prefill and substantial per-graph queue overhead, without claiming large-model serving speed.

Live verification (2026-11-11): both MoEs use installed bbc2990, 32K context, VL15 and MTP depth 1 without a context cutoff. Both passed protected OpenCode tasks and installed-unit regressions with no model swap/OOM. The matched fixed 2K request fell 10.261 → 9.819 s for Ornith’s cutoff change and 12.115 → 8.361 s for Qwen’s full upgrade. [Attribution, context screens and one-command rollback](MTP-CONTEXT.md).

## The whole story

We fixed the CPU's missing memory-latency votes ([Omarchy #18254](https://github.com/omacom/omarchy/issues/18254)), retuned threads and placement, measured when MTP helps, made Ornith the everyday default, and kept prefix reuse across unloads. The installed Ornith engine repacks Q3_K weights, batches routed-expert prefill into GEMM, and caches packed weights on disk. Each step has its own controls: [stock-to-now chain](BEFORE-AFTER.md), [scoreboard](SCOREBOARD.md), [failed paths](FAILED-PATHS.md). The platform fix is tracked in [Omarchy#18254](https://github.com/omacom/omarchy/issues/18254), with upstream/Surface credits in [MEMLAT](MEMLAT.md).

This is one laptop, not a claim about every X Elite/X Plus or every coding task. Vendor model scores are separate from our independent local checks. A single stock-boot vs fully-tuned protected-task launch comparison is planned after all optimizations, immediately before public launch; we do not have that number yet.

- **Purpose:** local coding, planning and image generation on Snapdragon X Linux.
- **Test machine:** Lenovo Yoga Slim 7x Gen 9 (83ED / 14Q8X9), 12-core Snapdragon X Elite, 32 GB RAM, Adreno X1 and Hexagon v73, running Omarchy / Linux ARM64.
- **Working here:** CPU coding and image profiles, SSD-backed Flash planning, lazy loading, idle unload and an Omarchy control panel.
- **Evidence:** fixed synthetic tasks, independent correctness checks, per-model results and preserved failures; large-model GPU/NPU paths remain experimental.
- **Start safely:** try the three commands below. Full runtime setup is manual; weights and vendor runtimes are separate downloads.

## Measured before → after

| Model / same-workload comparison | Before → after | Result |
|---|---:|---|
| Qwen3.6-35B-A3B MTP UD-Q4_K_M, pool limit 1 → 3 | 10.98 → 16.86 tok/s | **~54% higher decode throughput**; same weights and experimental binary |
| Ornith Compact MTP, matched CPU profile, stock → memlat boot | 12.73 → 17.18 tok/s | **35% higher short-screen decode**; exact outputs; matched boot control (MEMLAT now default on this laptop) |
| Qwen MoE, cold repair → full reload with saved prefix | 265 → 210 s | **21% less finished repair time** |
| Qwen3.8-27B DevQuasar Q4_K_M, cold repair → full reload with saved prefix | 433 → 202 s | **53% less finished repair time** |
| Z-Image Turbo Q4_K, native 1024, CPU repack off → on | 1226 → 898 s | **27% less load-to-PNG time** |
| Z-Image Turbo Q8_0, native 1024, CPU repack off → on | 776 → 488 s | **37% less load-to-PNG time** |
| Flash-Next mixed UD-IQ3_XXS, eager → demand mapping | 61 → 12 s load | **81% less startup time** in the direct pair; request/decode neutral |
| Flash, same complete code across sequential SSD tuning stages | 445 → 238 s total | **46% lower observed time**; separate first/last pairs, identical tested code |
| Flash, current IO4 → IO8 fixed-code ABBA screen | 119.167 → 110.282 s median | **7.5% less fixed-task time**; all 353-token outputs identical |

Each row has its own controls. These percentages are not a combined speedup or a ranking between models. The MTP result is **53.47% measured**, rounded here to about 54%; it is a decode screen, not agent completion. Image rows are matched within each quant, with one native-resolution pair each. [Exact measurements, settings and limits](BEFORE-AFTER.md).

Dense 27B has a measured prefix-reuse gain above. On the new opt-in memlat boot, its target-only short-context baseline reaches **5.41 tok/s**, and Arete reaches **5.36 tok/s**. Historical decode near 2.7 tok/s used different workloads, so this is not a matched whole-task speedup. [Current-boot scoreboard and thread screens](SCOREBOARD.md). A separate preliminary microbatch 64 → 256 screen raised prefill from 14.88 → 21.08 tok/s, with power/reboot confounds. [Dense improvements and unsuccessful experiments](BEFORE-AFTER.md#dense-27b-what-improved-and-what-did-not).

A separate protected IO4 → IO8 agent pair took **768.484 → 736.838 s (4.1% less time)**, with 12/12 tests and complete finals in both arms. Outputs differed. The gain missed the original 5% full-task gate; the later acceptance rule requires an improvement, so IO8 is now eligible on performance, pending installed lifecycle validation. Installed IO4 remains unchanged. [Fixed-screen evidence](../benchmarks/results/flash-next-io-depth-abba.json), [protected pair and limits](../benchmarks/results/flash-next-io-depth-agent.json).

Flash completed a protected OpenCode repair with a complete final response and **12/12 independent tests** in 19 min 35 s. A later attribution run took **12 min 48 s** on the same task; the original engine also runs faster now, so the unexplained **1.53× difference is not credited to a software change**. [Phase timings and current I/O experiments](FLASH-NEXT-PROFILE.md).

The selected Flash shards already contain **Unsloth Dynamic 3.1**, updated on 4 November. All four local full-file hashes match the published artifacts; the release announcement is not an additional upgrade over these files. [Exact identity audit](../benchmarks/results/flash-dynamic31-identity.json).

## Qualified profiles on the memlat boot

The next CPU settings also pass the protected OpenCode repair (13/13 independent checks in every arm). The settings are now installed after controller/cache lifecycle qualification and the owner's approval. Ornith is the everyday/OpenCode default; Qwen MoE is the fallback. Arete is retired (its row is historical):

| Original pinned model | Current settings → candidate, load to finished task | Observed time saved |
|---|---:|---:|
|Qwen3.6-35B-A3B UD-Q4_K_M + MTP|119.02 → 106.06 s|10.9%|
|Ornith 1.5 Compact|106.37 → 92.69 s|12.9%|
|Dense Qwen3.8-27B Q4_K_M|401.35 → 387.73 s|3.4%|
|Arete27B Q4_K_M (retired; historical)|409.94 → 395.17 s|3.6%|

Those direct comparison pairs used original weights and the same memlat boot,16K context,1200% quota and 27 GiB memory cap. The promoted service now has an approved28 GiB cap and model swap disabled; host pages may use disk-backed swap. Each is one pair. The first two have uncontrolled file-cache/order effects; dense/Arete receive per-file DONTNEED advice in both arms (not a guarantee of identical cold caches). Decode improvements are larger than task-time improvements: the matched16K Ornith server control is 16.54 → 30.12 tok/s. Its sustained ten-minute decode window changed−0.5%, with a97.7 °C whole-unit peak. [Settings, scope and raw evidence](SCOREBOARD.md), [new versus cached prompt tokens](PREFILL-AUDIT.md).

Actual controller/OpenCode Ornith also passed13/13 independent checks cold (**121.41 s**) and in the cached second session (**63.15 s**). This is a separate cold/cached pair, not a matched speed comparison with the direct-server rows. Saved prefixes survive model unloads. [Host memory policy and rollback](HOST-MEMORY.md).

The matched4K Ornith track goes from **12.73 tok/s with the stock DT/old settings to 29.38 tok/s with MEMLAT + tuned settings (2.31×)**; actual 16K production capacity reaches 30.12 tok/s. CPU reads at the same six-thread spread placement rose 48.88→115.85 GB/s (2.37×). These are linked controlled measurements, not one all-changes agent A/B. [Stock→now steps, limits and prefix gains](BEFORE-AFTER.md#stock-dtsettings--current-setup-matched-tracks). The older native kernel snapshot is kept separate; the installed successor and its direct/controller task measurements are reported above. Credit and platform tracking: [Omarchy#18254](https://github.com/omacom/omarchy/issues/18254).

The actual controller/OpenCode path is a separate 9-tool workload: patched cold 113.77s vs 121.41s before (6.3% less), cached 64.56s vs 63.15s (2.2% slower in one pair; no verified cached-session gain). Both pass 13/13. The matched64.00s direct task above uses 6 tools and must not be presented as the cached-controller result. The controller reuses 5,487 initial prefix tokens and generates 1,104 output tokens; the direct task newly processes6,066 input tokens and generates 479. [Token audit](../benchmarks/results/ornith-patched-controller-token-audit.json), [deployment and rollback](SCOREBOARD.md#installed-patched-ornith-engine-2026-11-09).

## Upstream contributions

| Contribution | Status |
|---|---|
| CPU MEMLAT platform fix | Measured on this Yoga; [Omarchy#18254](https://github.com/omacom/omarchy/issues/18254). [Upstream/Surface sources and credits](MEMLAT.md). |
| llama.cpp Q3_K repack and routed-expert MUL_MAT_ID GEMM | Locally tested patches; upstream submissions are being prepared. No public PR claimed yet. [Code, validation and failures](../experiments/q3k-i8mm/Q3K-MOE-GEMM.md). |
| Snapdragon kernel/firmware findings | [Reproducible MEMLAT component](MEMLAT.md) and [parked/failed paths](FAILED-PATHS.md); further reports stay draft until approved. |

For llama.cpp submissions, the owner writes PR descriptions and commit messages himself and discloses AI assistance as the project requires. Links/status will be updated when submissions actually open. Community review and independent reproductions are welcome.

## Try the tools

From a checkout, using Python 3:

```sh
PYTHONPATH=src python3 -B -m snapdragon_ai_lab doctor
PYTHONPATH=src python3 -B -m unittest discover -s tests
PYTHONPATH=src python3 -B -m snapdragon_ai_lab install-plugin --dry-run
```

These run the read-only diagnostic, offline software checks and panel installation preflight. They do not load models, download weights or install the runtime. Panel preflight requires an existing Omarchy/local-ai setup and reports missing prerequisites. See [panel installation and rollback](OMARCHY-PLUGIN.md) before installing, and [manual runtime setup](RUNTIME-SETUP.md) for the complete stack.

## What runs on the laptop

OpenCode uses the smaller MoE by default. Dense and Flash are selectable; Flash has a dedicated **Plan (read-only)** launch. Q8 CPU image presets offer a draft and native 1024 output, with Q4 also selectable. A shared inference lease keeps heavy jobs separate, and chat models unload after 120 seconds idle. The panel exposes selection, state, RAM and launch/unload controls.

This installed laptop is qualified separately from the portable source preview. [Deployment status, memory policy and evidence limits](DEPLOYMENT-AND-EVIDENCE.md) retain the detailed lifecycle checks, failed attempts and provenance. Current plan, [Flash planning](FLASH-NEXT-PLANNING.md) and [app settings](APP-SETTINGS-AUDIT.md) distinguish installed defaults from experiments.

[Ornith additional MoE comparison](ORNITH-MOE-COMPARISON.md) has passed its bounded screen and protected normal task. The [opt-in beta profile](ORNITH-BETA.md) is installed on the test laptop and passed its load, chat/tool, idle-unload and model-switch checks. Qwen remains the default. Other installations must explicitly stage and qualify this optional profile.

## Reusable OpenCode prefix cache

The shared runtime includes [prefix selection](../src/snapdragon_ai_lab/runtime/prefix_plan.py), [private bounded storage](../src/snapdragon_ai_lab/runtime/prefix_store.py) and the controller's native slot save/restore integration. The [example configuration](../examples/runtime/llm-runtime.json) enables a 3 GiB disk cache for the leading instruction/tool prefix. Exact model/build, context, engine settings, template and tool-schema identities determine reuse; changed inputs safely miss. This persists across full model unload/reload, while unloading still releases model RAM.

Each installation creates its own owner-only cache locally. User turns and tool results are excluded from the saved prefix. The first priming cost is counted in cold-task measurements; later reuse produced the dense 433 → 202 s and MoE 265 → 210 s observations above. [Setup and lifecycle checks](RUNTIME-SETUP.md), [synthetic cache tests](../tests/test_prefix_cache.py). Flash currently has in-RAM context reuse only; persistent Flash hybrid state remains an experiment.

## Known platform issues

- **SCMI memlat is experimental and firmware-dependent.** This Yoga has a tested opt-in alternate boot that exposes DDR/LLCC devfreq and improved bounded RAM-read and Ornith screens. The owner approved it as the default on this tested Yoga, with automatic verified rebuilds for daily edge kernels and a named stock fallback; other installations remain opt-in. Thread sweeps reached 98 °C; a ten-minute Ornith decode window showed no material slowdown, and the CPU candidates passed protected coding checks. Two deep-suspend cycles and the post-resume bandwidth check passed; s2idle itself remains untested. Setup, rollback and measurements: [SCMI memlat experiment](MEMLAT.md) and [current-boot scoreboard](SCOREBOARD.md). [Omarchy issue #18254](https://github.com/omacom/omarchy/issues/18254).

## Help wanted

- **FastRPC / QNN buffers:** observed mapping failures around 1.1 GB and 2.26 GB need a reproducible allocation/driver explanation. A small 4B NPU control worked; large-model NPU execution is unproven. These failures do not establish a hard hardware limit. [Measurements](FAILED-PATHS.md), [helper status](NPU-HELPER.md).
- **Adreno OpenCL (community research):** local LLM GPU work is parked by the owner; the CPU memlat path has higher measured read bandwidth, and earlier concurrent GPU work starved CPU reads. OpenCL is OFF in our tested profiles. Independent results still need correctness, admission and cleanup evidence; earlier Vulkan results do not qualify OpenCL. Image Studio is unaffected.
- **Other Snapdragon SKUs and RAM sizes:** repeat a fixed workload with exact engine/model identities, resource limits and complete-task correctness. Use the [community benchmark issue form](../.github/ISSUE_TEMPLATE/community-benchmark.yml).
- **The unexplained 1.53× Flash change:** equal output/read counts but much lower CPU core-time warrant controlled frequency, temperature and contention evidence. Historical telemetry is insufficient. [What is known](FLASH-NEXT-PROFILE.md).

Read [CONTRIBUTING.md](../CONTRIBUTING.md) for the test and privacy rules. Share numeric results from synthetic fixtures, not private prompts, answers, images, credentials or device serials. Community post drafts are drafts only; nothing has been posted.

## Reproduce and inspect

Use the [exact model manifest](../dependencies/model-artifacts.json) for repositories, immutable revisions, quantized filenames, sizes and every selected shard SHA256. The MIT code does not include model weights, compiled engines or proprietary vendor runtimes; their licenses remain separate.

The finished release must let another supported Snapdragon/Omarchy owner reproduce this setup: modified engines and tested settings, selected model downloads, local chat and image interfaces, OpenCode integration, and the always-visible Local AI panel with load/switch/unload controls. It must include dependency and license links, checksums, setup checks and rollback, without relying on this laptop's files. Optional profiles require their own qualification; experimental results are not installed defaults.

The [fresh-install parity checklist](BOOTSTRAP-PARITY.md) tracks this release requirement. The [selected-setup reproducibility audit](SELECTED-SETUP-REPRODUCIBILITY.md) distinguishes exported runtime source from missing installation and qualification pieces. Staging and public Flash source reconstruction are implemented; complete installation parity remains pending. See the [Flash source/build instructions](FLASH-REBUILD.md), [runtime dependencies and settings](RUNTIME-SETUP.md), [model downloads](../dependencies/model-artifacts.json) and [panel installation](OMARCHY-PLUGIN.md).

See the [benchmark method](BENCHMARKS.md), [source provenance](SOURCE-PROVENANCE.md), [patch manifest](../patches/manifest.json), [failed paths](FAILED-PATHS.md) and [release checklist](RELEASE.md). The [engine opportunities review](RESEARCH-ENGINE-OPPORTUNITIES.md) ranks bounded next tests from npurun/QNN, ARM kernels, DS4, oMLX and Sushi. Their hardware-specific rates are not Snapdragon performance claims.
