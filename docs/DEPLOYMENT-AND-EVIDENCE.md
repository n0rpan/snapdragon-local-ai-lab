# Deployment status and interpreting the evidence

This page keeps the detailed deployment, provenance and comparison limits behind the shorter [README](../README.md). These are observations from the test laptop; the portable package remains a development preview.

The test machine is a Lenovo Yoga Slim 7x Gen 9 with Snapdragon X Elite and 32 GB RAM, running Linux ARM64 and Omarchy. It runs local coding, planning and image generation. Core profiles have passed normal reload, caching, idle-unload and image-handoff tests. The improved SSD-backed Flash-Next completed a protected OpenCode repair in **19 min 35 s**, including a final response and **12/12 independent tests**. Its installed planning workflow also passed browser chat, real idle unload, MoE switching, Q8 image handoff and RAM recovery. Portable packaging remains a development preview.

The primary measurement is time to a correct, independently verified finished result, including model load, prompt processing, tool use and retries. Prefill and generation rates are reported separately.

Current attribution work measured a fresh protected Flash repair at **12 min 48 s** (12/12 checks) and the fixed decode control at **3.75 tok/s**. These are 1.53× and 2.20× the historical 1175-second/1.707 tok/s references. An untouched original-engine control also reaches 3.75 tok/s now, so this difference is **not attributed to a new software optimization**. The corrected attribution run passed; exposed SSD waits are 13.2% of total, and CPU work is the larger remaining cost. The later I/O screen and protected pair are recorded in the [current profile](FLASH-NEXT-PROFILE.md); CPU and cache-policy tests remain separate experiments. [Current profile and limitations](FLASH-NEXT-PROFILE.md).

## Exact models and runtime changes

Use [the model artifact manifest](../dependencies/model-artifacts.json) to reproduce these results. It identifies publisher repositories, immutable revisions, exact quantized filenames, byte sizes and SHA256 checksums, including all four Flash shards and the image encoder/VAE. A similarly named quant is not necessarily the same model file. Weights are downloaded separately; their licenses are separate from this repository's MIT code.

The work combines ARM quantized kernels and weight repacking, adaptive MTP threading, measured prompt batching, safe cross-session prefix caching, and lazy model switching. For Flash, it adds a bounded RAM expert cache, demand-based mapping, parallel SSD reads and read/compute overlap while preserving native expert routing. [Runtime and patch provenance](SOURCE-PROVENANCE.md) records the tested implementations.

RAM is used for speed where measurements support it: Flash's **8 → 12 GiB expert cache** raised the fixed JSON screen from **1.75 → 2.45 tok/s**, and Q8 images use more RAM than Q4. [Memory and performance](MEMORY-AND-PERFORMANCE.md) distinguishes weight residency, working memory, context and disk caches. [App settings](APP-SETTINGS-AUDIT.md) explains the defaults and remaining experiments.

The installed laptop uses these defaults:

- **Coding:** ordinary OpenCode starts with MoE. The panel's Dense/MoE selection applies to OpenCode launched from that panel.
- **Planning:** select Flash-Next and press **Plan (read-only)**, or run `local-ai plan`. This dedicated launch permits file reads; large cold prompts can take minutes.
- **Images:** Q8 CPU presets provide a 256-pixel draft and a native 1024-pixel final image; Q4 remains selectable.

Models load on demand, share one inference lease and release chat RAM after 120 seconds idle. Flash retains its tested 12 GiB expert cache while loaded. The normal workflow check observed release after **123.18 seconds**, completed a Q8 draft with chat excluded, and returned to MoE selected/unloaded with **26.4 GiB available**. See the [current plan and follow-up](PLAN.md) for the separate task and lifecycle evidence.

Follow the [current plan and evidence](PLAN.md), [Omarchy panel](OMARCHY-PLUGIN.md), [Flash planning workflow](FLASH-NEXT-PLANNING.md) and [NPU helper review](NPU-HELPER.md). The target is a useful offline agent and private local images, with reproducible experiments that other Snapdragon/Omarchy users can improve.

The [manual runtime setup guide](RUNTIME-SETUP.md) includes portable configuration, user-unit and OpenCode v2 examples. It is a source preview; models, local qualification and manual installation remain your responsibility.

## Available now

- A read-only hardware and power diagnostic.
- An allowlist exporter for public benchmark metrics.
- Fixed synthetic coding fixtures with independent tests, plus an explicit local-API runner with offline sandbox checks.
- Preliminary measurements with explicit comparison limits.
- Small, opt-in engine patches with upstream license notices and provenance.

```sh
PYTHONPATH=src python3 -m snapdragon_ai_lab doctor
PYTHONPATH=src python3 -m unittest discover -s tests
```

For the optional panel, `install-plugin --dry-run` checks installation prerequisites. Existing personal clones can be migrated with `install-plugin --replace-plugin ID --dry-run`; see the [panel installation and rollback instructions](OMARCHY-PLUGIN.md) before omitting `--dry-run`.

These commands do not load models, install packages, change drivers or modify power settings. The diagnostic reports architecture, RAM, device availability and battery telemetry. It does not inspect prompts, chats, images, browser state, hostnames or credentials.

## Test method

See [the benchmark method](BENCHMARKS.md), [preliminary CPU screen](../benchmarks/results/cpu-screen-preliminary.json), [patch provenance](../patches/manifest.json), [ideas from other engines](TRANSFERRED-IDEAS.md), [the release checklist](RELEASE.md) and [disabled GPU/speculation/NPU paths with numbers](FAILED-PATHS.md).

Current CPU measurements are the practical baseline. GPU measurements were slower for the large model and caused desktop pauses; the earlier GPU cleanup problem was separately fixed and tested. Native Qualcomm NPU inference worked for a compiled 4B model, while larger-model buffer mapping remains unresolved. These findings apply to the tested stack and do not establish limits for all Snapdragon hardware.

Only one heavy build or inference job runs at a time. The laptop may use its battery briefly while plugged in. Benchmarks require a starting reserve and stop at the 20% reserve threshold. Long image runs require more starting reserve. RAM recovery is checked after processes exit.

## Scope

The test-machine deployment combines a fast MoE coding default, selectable dense and Flash planning profiles, Image Studio, lazy loading, idle unload and an always-visible Omarchy control panel. A manual runtime guide and disabled-by-default examples are available. Flash's complete repair and normal planning lifecycle are qualified separately. This source preview provides building blocks and panel installation, with manual setup for the complete stack.

Model weights, compiled engines, proprietary Qualcomm/Windows runtime files, user configurations, prompts, images and raw logs are excluded. Model licensing is separate from this repository's MIT license.

## Contributing

Reports from other Snapdragon machines are useful. Include the model revision and quantization, engine commit and patch list, context and batch sizes, thread placement, power conditions, correctness results, complete task timing and RAM recovery. Read [CONTRIBUTING.md](../CONTRIBUTING.md) before submitting data.

Related work reviewed for ideas includes [oMLX](https://github.com/jundot/omlx) and [Sushi](https://github.com/beamivalice/sushi). Their Apple-specific kernels and reported rates are not performance claims for Snapdragon.
