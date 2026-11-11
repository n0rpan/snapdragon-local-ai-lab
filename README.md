# Snapdragon Local AI Lab

Large language models and image generation running locally on a **Snapdragon X Elite laptop under Linux**, on the CPU, with a receipt behind every speed claim.

This repository holds the controller, the OpenAI-compatible API, the harness integrations, the benchmark and qualification tools, the measurements and the failed attempts. The inference engines live in the companion repository **snapdragon-engine** (<https://github.com/n0rpan/snapdragon-engine>): `oxe` for the 35B mixture-of-experts models, a streaming Flash engine for a 125B MoE that does not fit in RAM, and the experimental `oxi` image engine.

It is one laptop and one firmware. Read the numbers as "what this machine did and how to check it", not as a ranking of chips.

## The machine

| | |
|---|---|
| Laptop | Lenovo Yoga Slim 7x (Gen 9) |
| Chip | Snapdragon X Elite X1E80100: 12 Oryon cores in three clusters of four, each cluster sharing an L2 |
| Memory | 32 GB LPDDR5X (30.7 GiB visible to Linux) |
| Storage | NVMe SSD |
| OS | Linux ARM64 (Arch-based Omarchy) |
| Used for inference | **CPU only.** The Adreno GPU and the Hexagon NPU are not used in any number below ([why](docs/RESULTS.md#what-did-not-work)) |

## Stock → today

Same laptop, stock software on one side, **today's installed stack** (MEMLAT on, Rust engines, MTP + verify lanes, installed profiles) on the other. One row per workload, measured 2026-11-10/11. `[M id]` = measured, receipt `benchmarks/results/public/<id>.json` (′ → `-prime`); tok/s values are 1000 ÷ ms/token. **Where the two sides do not share a metric or a context, the row says so; where no stock run exists, the row compares with the previous production instead.** Rows are separate tracks: do not multiply them.

| Workload | Stock | Today (installed) | Gain | Same metric and conditions? | Evidence |
|---|---|---|---|---|---|
| CPU memory reads (2 GiB NEON read, 6 spread threads) | 48.88 GB/s (stock device tree) | **115.85 GB/s** (MEMLAT overlay) | 2.37× | yes: same binary, 3 repeats | [M] [ROOFLINE](docs/ROOFLINE.md) |
| **Ornith** 35B-A3B decode, short prompt | 12.73 tok/s: pinned llama.cpp server, stock device tree, old 4 decode / 10 prompt threads, MTP 2; 128-token prompt, 64 output | **54.4 tok/s** (18.38 ms/token): Rust `oxe`, VL15 + MTP depth 1, 8 decode cores; 53-token prompt, 256 timed tokens | 4.3× | close, not identical: both short-prompt decode, different harness and output length; today's arm is candidate `24caf62`, the profile installed as `bbc2990` | [M] stock: [ROOFLINE](docs/ROOFLINE.md) · today: G1′ |
| Ornith decode, 8K context | not measured at 8K | **55.8 tok/s** (17.92 ms/token) installed `bbc2990`; plain decode on the same binary 41.3 tok/s (24.20 ms) | 1.35× vs plain | yes for today vs plain (A B B A, token ids identical); **no stock run** | [M] QC-CTX2 (G1′ on the candidate: 19.43 ms) |
| Ornith prefill, cold | 121.14 tok/s: unmodified upstream llama.cpp `5f490d7`, `llama-bench` pp512, 12 threads, MEMLAT on | **276.77 tok/s**: 2,048 tokens, kernel mask 1021, **group 2** (283.58 with group 4; the installed profile uses `pf_group=auto`), 12 threads | 2.3× | **no**: 512 vs 2,048 tokens. Same-length comparison is vs previous production: 198.13 → 276.77 (mask 13 / group 1 → mask 1021 / group 2) | [M] stock: [RESULTS §4](docs/RESULTS.md#4-earlier-ladder-summary) · today: G1 |
| **Qwen3.6** 35B-A3B decode | 26.18 tok/s: upstream llama.cpp `5f490d7`, `llama-bench` tg128 (empty context), MEMLAT on | **41.0 tok/s** at 2K (24.41 ms/token), 39.9 at 8K (25.07), 34.6 at 16K (28.86): installed Rust, VL15 + MTP depth 1 | 1.57× | **no**: tg128 at empty context vs decode at 2K+. Same binary plain → MTP: 30.18 → 24.41 ms at 2K | [M] stock: [RESULTS §4](docs/RESULTS.md#4-earlier-ladder-summary) · today: QC-CTX2-Q |
| Qwen3.6 prefill, cold | 172.33 tok/s: upstream llama.cpp `llama-bench` pp512 | **261.64 tok/s**: 2,048 tokens, mask 1021 / group 2 | 1.52× | **no**: 512 vs 2,048 tokens. Same length vs previous production: 177.71 → 261.64 | [M] stock: [RESULTS §4](docs/RESULTS.md#4-earlier-ladder-summary) · today: G2 |
| **Flash** (Qwen3.8-Flash-Next 125B MoE, 83.6 GB, experts streamed from SSD) decode | 356.49 ms/token (2.8 tok/s): BigMoeOnEdge, its production **12 GiB** expert cache; request wall ÷ output tokens, 3 warm 64-token requests | **81.13 ms/token (12.3 tok/s)**: Rust Flash engine, **14 GiB** cache, prefill order 2; decode after a 75-token prompt, 4 × 64 tokens, first dropped | 4.4× | **no**: cache size and metric differ. Same metric as stock: Rust 84.90 ms/token at **16 GiB** (4.2×) | [M] G3 → 063 (G3′) |
| Real coding-agent request (OpenCode, fixed 2K request, whole request, median of 3) | stock not measured; vs previous production Rust `a219347`: Ornith **13.27 s** | Ornith **10.16 s** (`bbc2990`, MTP with a 2K cutoff) → **9.82 s** installed (no cutoff); Qwen 12.11 → **8.36 s** (old Rust build without MTP → installed) | −26% Ornith, −31% Qwen | yes: same request, same controller, token hashes identical | [M] G0′, QC-CTX2, G2′ |
| Images, 512 px (FLUX.2 klein 4B Q8), whole request through Studio | 77.85 s: stable-diffusion.cpp (`sd-cli`) | **51.53 s**: experimental `oxi` fast path (32/32 chunks, f32 attention), automatic fallback to sd.cpp; **45.94 s** with a cached prompt; installed Studio live smoke **47.42 s** | −33.8% | yes: same Studio path, prompt and seeds, A B B A, mean of 2 cold requests per arm; quality gate passed (31.5 / 32.1 / 34.1 dB vs a 27.8 dB gate); 1024 px and other sizes stay on sd.cpp | [M] IMG-SHIP-1, F-IMG-ATTN-1; live smoke: [image-ship-20261111.json](benchmarks/results/image-ship-20261111.json) |

Conditions that apply to every row: CPU only; the MEMLAT memory-frequency fix active ([docs/MEMLAT.md](docs/MEMLAT.md)); decode on 8 packed threads (cores 0–7), prefill on 12 (MoE) or 11 + IO core (Flash). **Thermal state:** sustained runs on this laptop sit at 94–98 °C on the hottest CPU sensor and all-core work throttles about 25% within 30 s, which is why every speed decision here uses interleaved arms. **Power source (AC or battery) was not recorded** for these runs. A long-session thermal test has not been done. The session cost model (prefix reuse across turns) is not receipted yet and is not in this table.

The receipt ids name the run tickets of the development notes. Each one has a public, scrubbed export at [`benchmarks/results/public/<id>.json`](benchmarks/results/public/) (′ is written `-prime`, e.g. `G3-prime.json`): the numbers, digests and sanitised command lines of the private run receipts, with prompts reduced to their length and SHA-256 and every path, log and free-text note dropped ([how](tools/export_receipt.py)).

### Compared with a Mac

**We have no Mac and no matched Mac measurement, so we make no "faster than" or "slower than" claim.** Third-party Mac numbers for these models exist, but with other quants, engines and prompts; they are listed with that caveat in [docs/RESULTS.md](docs/RESULTS.md#compared-with-apple-silicon). On Flash they are well ahead of us (a 64 GB+ Mac holds most or all of the model in RAM). If you have an Apple Silicon Mac, [`tools/mac_reproduce.py`](tools/mac_reproduce.py) runs the same model files by SHA256 and writes receipts in our format; see [help wanted](docs/HELP-WANTED.md#7-a-matched-mac-comparison).

## What made the difference

- **Memory-frequency votes (MEMLAT).** The stock device tree did not let the CPU ask for DDR/LLCC frequency; reads sat near 49 GB/s. A one-node overlay brings them to 116 GB/s. Decode is bandwidth-bound, so this is the largest single step. Firmware-dependent.
- **A purpose-built Rust engine (`oxe`).** A static plan per model instead of a general graph (about 170–210 thread barriers per token instead of ~3,900 graph nodes), tiled attention, batched prefill with repacked int8 (`i8mm`) kernels, and power-of-two stride fixes found at production shapes.
- **Speculative decode that pays on a MoE.** MTP depth 1 with "verify lanes": a two-token verify costs 1.39 plain steps instead of 1.74, so depth 1 now wins at every measured context (to 32K on Ornith, 16K on Qwen). Deeper drafts lose on these models.
- **Not recomputing.** Prefix store on disk, parked sessions in RAM, turn checkpoints. In an agent session this removes more prefill than any kernel speeds up. Exact.
- **SSD streaming for a model 2.7× larger than RAM.** Dense weights stay resident; 512 experts per layer stay on NVMe behind a 14–16 GiB LFU cache with per-layer quotas; O_DIRECT reads on a dedicated IO core overlap compute; memory admission is measured, not guessed.

Why each technique fits this chip: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). What we tried and dropped, with the measurement: [docs/RESULTS.md#what-did-not-work](docs/RESULTS.md#what-did-not-work).

## The whole story (history)

How we got here, step by step. Each row has its own baseline; the per-step evidence is in [docs/RESULTS.md](docs/RESULTS.md), [docs/BEFORE-AFTER.md](docs/BEFORE-AFTER.md) and [docs/SPEEDUP-ATTRIBUTION.md](docs/SPEEDUP-ATTRIBUTION.md).

**Per-change measurements of the current engines (2026-11-10/11):**

| Workload | Before → after | Conditions | Evidence |
|---|---|---|---|
| **Flash** (Qwen3.8-Flash-Next, 125B MoE, 83.6 GB, experts streamed from SSD): decode, old engine → our Rust engine | **356.49 → 84.90 ms/token (2.8 → 11.8 tok/s, 4.2×)** | BigMoeOnEdge at its production 12 GiB expert cache vs Rust at 16 GiB; 3 warm 64-token requests each, request wall time ÷ output tokens; short prompts. **Cache size differs between the arms**, so this is engine + configuration, not engine alone. | [M] G3, G3′ |
| Flash decode, installed configuration today | **81.13 ms/token (12.3 tok/s)**; 4.95% less than the previous order rule (85.36) | 14 GiB expert cache, `--flash-pf-order 2`, decode after a 75-token prompt, 4 × 64 tokens with the first dropped, interleaved B A A B; all 16 output texts identical | [M] 063 |
| Flash prefill, 2,048 new tokens, cold | **62.3–66.4 tok/s** | 18 GiB cache, 11 compute threads + 1 IO core, 2 arms per setting, saved state identical across arms | [M] G4′ |
| **Ornith** 1.5 35B-A3B (default model): decode with MTP depth 1 + verify lanes vs plain | 4K **22.42 → 17.60 ms/token** (44.6 → 56.8 tok/s) · 8K 24.20 → 17.92 · 16K 27.35 → 22.61 · 32K 32.93 → 29.02 | Same installed binary, A B B A, 256 tokens per arm, 8 decode threads on cores 0–7; **token ids identical** in both arms | [M] QC-CTX2 |
| Ornith prefill, 2,048 tokens, cold | **198.13 → 276.77 tok/s** (283.58 with group 4) | Kernel mask 13 / group 1 → mask 1021 / group 2, 12 threads; prefill logits and state bit-identical | [M] G1 |
| Ornith, fixed 2K OpenCode request, whole request | **13.27 → 10.16 s (−23.4%)** | Previous installed build vs candidate 2, 3 runs per arm, medians | [M] G0′ |
| **Qwen3.6** 35B-A3B: prefill, 2,048 tokens, cold | **177.71 → 261.64 tok/s (+47%)** | Same mask/group change, 12 threads, state and first token identical | [M] G2 |
| Qwen3.6 decode with MTP depth 1 + verify lanes vs plain | 2K **30.18 → 24.41 ms/token** · 8K 32.61 → 25.07 · 16K 35.25 → 28.86 | A B B A, 256 tokens, token ids identical | [M] QC-CTX2-Q |
| Qwen3.6, fixed 2K request, whole request | **12.11 → 8.36 s (−31%)** | Old Rust build (no MTP) vs installed, 3 runs per arm, identical token hashes | [M] G2′ |

**Earlier milestones (2026-11-09/10, before the Rust engines were installed):**

| Measured track | Before → after | What that means |
|---|---|---|
| CPU memory reads | 48.88 → 115.85 GB/s (2.37×) | Matched six-thread spread test; MEMLAT device-tree fix. [MEMLAT](docs/MEMLAT.md) |
| Ornith, server decode on llama.cpp | 12.73 → 29.38 tok/s (2.31×) | Linked steps: stock device tree/settings → MEMLAT + tuned threads; not one combined A/B. [BEFORE-AFTER](docs/BEFORE-AFTER.md#stock-dtsettings--current-setup-matched-tracks) |
| Ornith, CPU kernel screen (patched llama.cpp) | pp512 119.3 → 185.0 tok/s (+55%); tg128 28.9 → 31.2 (+7.8%) | Native bench A B B A of the exact-mode Q3_K snapshot, unchanged weights. [Method](experiments/q3k-i8mm/RESULTS.md) |
| Ornith, patched llama.cpp fallback, protected task | 99.77 → 64.00 s (36% less time) | Direct-server pair, 13/13 checks, includes load. [SCOREBOARD](docs/SCOREBOARD.md) |
| Dense 27B, persistent agent prefix | 432.89 → 201.60 s (53% less time) | Same repair after a full reload with the saved prefix. [BEFORE-AFTER](docs/BEFORE-AFTER.md) |
| Z-Image Turbo Q8, 1024 px, CPU repack off → on | 775.52 → 488.06 s (37% less time) | Same weights and resolution, one pair. [BEFORE-AFTER](docs/BEFORE-AFTER.md) |

## Supported models

| Model | Role | Engine | Context | Weights |
|---|---|---|---|---|
| Ornith-1.5-35B-A3B, APEX-MTP Compact GGUF | everyday default | `oxe` | 32K | `mudler/Ornith-1.5-35B-A3B-APEX-MTP-GGUF` |
| Qwen3.6-35B-A3B MTP, UD-Q4_K_M | selectable | `oxe` | 32K allocated (decode measured to 16K) | `unsloth/Qwen3.6-35B-A3B-MTP-GGUF` |
| Qwen3.8-Flash-Next, UD-IQ3_XXS (4 shards, 83.6 GB) | "think hard" model; temperature 0, ≤2,048 output tokens | Flash (Rust, SSD streaming) | 16K | `unsloth/Qwen3.8-Flash-Next-GGUF` |
| Qwen3.8-27B dense, Q4_K_M | fallback | patched llama.cpp | 16K | `DevQuasar/Qwen.Qwen3.8-27B-GGUF` |
| Z-Image Turbo Q8_0 / Q4_K | images | patched stable-diffusion.cpp | — | `leejet/Z-Image-Turbo-GGUF` |

Pinned revisions, file names, sizes and SHA256 of every file: [`dependencies/model-artifacts.json`](dependencies/model-artifacts.json). **No weights are in this repository.** Each model keeps its publisher's licence.

## Quick start

The model-free checks run anywhere with Python 3:

```sh
git clone https://github.com/n0rpan/snapdragon-local-ai-lab && cd snapdragon-local-ai-lab
PYTHONPATH=src python3 -B -m snapdragon_ai_lab doctor              # read-only system check
PYTHONPATH=src python3 -B -m unittest discover -s tests            # offline tests
```

Running the models needs the engines, the weights and a manual setup. The MoE engine builds on its own; the Flash engine first needs two ggml libraries from a BigMoeOnEdge build (public sources, exact pins and commands in [docs/REPRODUCE.md](docs/REPRODUCE.md#2-build-the-engines)). Details: [docs/REPRODUCE.md](docs/REPRODUCE.md) (build, download, verify, run, benchmark, quality gates). There is no one-command installer yet.

## Use it from any client

The controller serves every model on one loopback port in the OpenAI wire format:

```sh
curl http://127.0.0.1:8092/v1/models
curl http://127.0.0.1:8092/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model": "local-default", "messages": [{"role": "user", "content": "Hello"}]}'
```

Base URL `http://127.0.0.1:8092/v1`, any API key (e.g. `local`), model id `local-default` (the selected model), `local-heavy` (Flash) or a full name. Streaming, tool calls and `usage` are supported. Point any OpenAI-compatible client at that base URL. Tested and documented:

| Harness | Guide |
|---|---|
| OpenCode | the protected qualification task runs through it; [docs/HARNESSES.md](docs/HARNESSES.md) |
| pi | [docs/HARNESS-PI.md](docs/HARNESS-PI.md) |
| dsh (DeepSeek Harness) | [docs/HARNESS-DSH.md](docs/HARNESS-DSH.md) |
| Anything else (Python/JS SDKs, Aider, Continue, Open WebUI, …) | [docs/OPENAI-API.md](docs/OPENAI-API.md) |

One model is loaded at a time and requests are served one at a time (others queue). Give a harness one model id for all its roles, or a model switch reloads weights. The generic API surface (aliases, `unknown_model`, request ledger) is in the source here; on the test laptop the installed controller is the previous release until that package is qualified.

## Safety and rollback

- **Memory:** the inference service runs in its own systemd scope with a hard cap and no swap. Loads are admitted only if measured headroom allows (Flash needs cache + measured overhead + harness allowance + a 1 GiB host floor), and a guard stops the engine below the floor instead of letting the OOM killer pick a victim. [docs/HOST-MEMORY.md](docs/HOST-MEMORY.md), [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#memory-admission).
- **Every production change** passes a protected OpenCode coding task (13/13 independent checks), a post-install regression through the installed service, and ships with a one-command rollback script written before the install. [docs/POST-INSTALL-REGRESSION.md](docs/POST-INSTALL-REGRESSION.md).
- **MEMLAT** installs as a separate boot entry and leaves the stock entry alone. It is specific to this laptop model; read [docs/MEMLAT.md](docs/MEMLAT.md) before trying it elsewhere.
- **Loopback only.** The API listens on 127.0.0.1 and has no setting to listen elsewhere.

## Help wanted

We would like help on concrete, scoped problems: a Hexagon NPU prefill path, image int8 attention calibration, Flash expert prediction at small caches, a shared-expert verify for MoE speculation, numbers from other Snapdragon X laptops, a matched Mac comparison, and fuzzing the tool-call parser. Each comes with context and acceptance criteria in [docs/HELP-WANTED.md](docs/HELP-WANTED.md). Read [CONTRIBUTING.md](CONTRIBUTING.md) first.

## More

- [docs/RESULTS.md](docs/RESULTS.md): all measurements, method, what did not work
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): engines, controller, caching, memory admission
- [docs/REPRODUCE.md](docs/REPRODUCE.md): build, models, benchmarks, quality gates
- [docs/LAB-NOTEBOOK.md](docs/LAB-NOTEBOOK.md): the previous README, with the earlier llama.cpp-era results and their links
- [docs/FAILED-PATHS.md](docs/FAILED-PATHS.md), [docs/SPEEDUP-ATTRIBUTION.md](docs/SPEEDUP-ATTRIBUTION.md): older detailed ledgers

## Credits

The memory-frequency work builds on the Surface Pro 11 memlat patches and the upstream SCMI memlat RFC ([docs/MEMLAT.md](docs/MEMLAT.md); platform tracking in [Omarchy#18254](https://github.com/omacom/omarchy/issues/18254)). The engines build on [llama.cpp](https://github.com/ggml-org/llama.cpp) and ggml (vendored kernels, ported tokenizer and chat templates), [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) and [BigMoeOnEdge](https://github.com/Helldez/BigMoeOnEdge), whose SSD-streaming design started the Flash work.

## Licence

MIT, copyright Oscar Nordström (n0rpan). Vendored and patched third-party code keeps the licence of its source; see [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES) and `third_party/`. Model weights, engine binaries and vendor runtimes are not included and have their own terms.
