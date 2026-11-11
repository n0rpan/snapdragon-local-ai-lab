<!-- DRAFT for the public v0.1 README. Private branch only. Not the live README.
     Every number below is copied from docs/SPEEDUP-ATTRIBUTION.md with its evidence tag.
     Rows marked "pending" must be filled by the release measurements before this replaces README.md. -->

# Snapdragon Local AI Lab

Large local language models on a Snapdragon X Elite Linux laptop, on the CPU, with every speed claim backed by a receipt. This repository holds the tools, patches, measurements and failed attempts from getting 35B-class mixture-of-experts models, and one 125B model that does not fit in RAM, to run usefully for agentic coding on a 32 GB machine.

It is one laptop and one firmware. Read the numbers as "what this machine did, and how to check it", not as a ranking of chips.

## Hardware

| | |
|---|---|
| Laptop | Lenovo Yoga Slim 7x Gen 9 (83ED) |
| Chip | Snapdragon X Elite X1E80100: 12 Oryon cores in three clusters of four |
| Memory | 32 GB LPDDR5X |
| Storage | NVMe SSD |
| OS | Linux ARM64 (Omarchy) |
| Used for LLM inference | CPU only. The GPU and the NPU are not used in any number here |

## Models

| Short name | Model | Quant | Role |
|---|---|---|---|
| Ornith | Ornith-1.5-35B-A3B | APEX-MTP Compact GGUF | everyday default |
| Qwen | Qwen3.6-35B-A3B MTP | UD-Q4_K_M | selectable |
| Flash | Qwen3.8-Flash-Next | mixed UD-IQ3_XXS, four shards, 83.6 GB | "think hard" model; experts are streamed from SSD |

Repositories, revisions, file names, sizes and SHA256 for every file: [`dependencies/model-artifacts.json`](../dependencies/model-artifacts.json). No weights are in this repository.

## Results

**The matched "stock → now" table is pending.** The release rule for a headline number is: same model file, quant and prompt tokens; same boot and day; at least three warm runs with median and range; baseline is unmodified upstream llama.cpp at commit `5f490d7`; decode and prefill reported separately; binary SHA, receipt and reproduce command in the repository. No row meets all of that yet, so there is no headline table yet.

What is measured today, with how well:

| What | Result | Evidence |
|---|---|---|
| Memory-frequency fix (device-tree overlay), CPU streaming read | 48.88 → 115.85 GB/s | measured, n=3 |
| Same fix, Ornith decode, same binary and settings | 12.73 → 17.18 tok/s, outputs byte-identical | measured, n=3 |
| Unmodified upstream llama.cpp `5f490d7` on the fixed boot, llama-bench tg128 / pp512 at 8 / 12 threads | Ornith 30.98 / 121.14 tok/s; Qwen 26.18 / 172.33 tok/s | measured, n=6 |
| Our Q3_K kernel patch on that baseline, Ornith prefill | 121.14 → 165.23 tok/s, output identical | measured, n=3 vs 6 |
| Our routed-expert batching patch on that baseline, Qwen prefill | 172.33 → 210.52 tok/s, output identical | measured, n=3 vs 6 |
| Rust engine, Ornith, real coding task end to end against patched llama.cpp | 77.70 → 60.42 s | measured, n=3 per arm, medians |
| Rust engine prefill kernels ("mask 13"), Ornith, 2,048 new tokens on a 6K prefix | 142.83 → 183.10 tok/s, bit-exact | measured, n=2 per arm (below the release count) |
| Flash, installed engine, decode | 4.06 tok/s | measured |
| Flash, Rust candidate engine, decode | 8.1 tok/s (123 ms per token); not installed; output differs within a stated bound (mean KLD 6.55e-4) | measured, n=3 |

Each row has its own baseline. **Do not multiply rows.** The full ladders, including every step that measured flat or slower, are in [`docs/SPEEDUP-ATTRIBUTION.md`](SPEEDUP-ATTRIBUTION.md); its section 4 lists the claims we do not make.

## What we changed

- **Memory-frequency votes.** The stock device tree did not let the CPU ask for memory frequency, so reads were stuck near 49 GB/s. A one-node overlay fixes it. Decode on these models is limited by memory bandwidth, so this is the largest single step. It depends on this laptop's firmware. [`docs/MEMLAT.md`](MEMLAT.md)
- **Thread layout.** Eight packed threads for decode, twelve for prefill. More decode threads are slower.
- **Quant kernels for llama.cpp.** A repacked Q3_K kernel using the `i8mm` instructions, and batching of the tokens routed to the same expert. Both give identical output. [`patches/`](../patches), [`docs/UPSTREAM-EVIDENCE.md`](UPSTREAM-EVIDENCE.md)
- **A small Rust engine for these architectures.** A fixed plan per model instead of a general graph, tiled attention, batched prefill.
- **Not recomputing.** A prefix store on disk, parked sessions in RAM and turn checkpoints, all exact. In an agent session this removes more prefill than any kernel speeds up.
- **SSD streaming for the 125B model.** Dense weights stay in RAM; the experts stay on SSD behind a RAM cache.

## What did not work

GPU prefill through Vulkan, the NPU for LLM decode, speculative decoding deeper than one token on the 35B models, more than eight decode threads, and several IO tunings. Each has a measurement: [`docs/FAILED-PATHS.md`](FAILED-PATHS.md) and the "rejected" rows of the attribution document.

## Compared with a Mac

**We have no Mac and no matched Mac measurement.** We therefore make no claim of the form "faster than" or "slower than" an M-series Mac.

What we expect, unmeasured: Macs with Max or Ultra chips have several times the memory bandwidth and enough RAM to hold the 125B model, and Metal makes prefill faster. On those machines we expect to lose clearly, on prefill most of all. Against base and Pro chips on the 35B models we do not know.

If you have an Apple Silicon Mac you can fill the table. [`tools/mac_reproduce.py`](../tools/mac_reproduce.py) checks the same model files by SHA256, runs unmodified llama.cpp at the same commit (Metal and CPU-only) and MLX, and writes receipts with no serial number, user name or path in them:

```
python3 tools/mac_reproduce.py plan
python3 tools/mac_reproduce.py verify-models --models-dir DIR
python3 tools/mac_reproduce.py run --models-dir DIR --llama-dir LLAMA_CPP_CHECKOUT --out receipts/
```

Two limits: the script has not yet been run on a Mac, and MLX cannot load GGUF files, so an MLX row is a different quantisation of the same model and is labelled as such.

## Reproduce

1. **Models.** Download the files named in `dependencies/model-artifacts.json` and check their SHA256.
2. **Baseline.** Build unmodified llama.cpp at `5f490d7bfe78844e6390c869454c149fd7e003a8`. The bench command and settings are in [`docs/UPSTREAM-EVIDENCE.md`](UPSTREAM-EVIDENCE.md).
3. **Patched llama.cpp.** Apply the patches in `patches/` to the commit named in `patches/manifest.json`.
4. **Memory-frequency fix.** Read [`docs/MEMLAT.md`](MEMLAT.md) first. It installs a separate boot entry and leaves the default alone. It is specific to this laptop model.
5. **Runtime.** [`docs/RUNTIME-SETUP.md`](RUNTIME-SETUP.md). Setup is manual; there is no installer.
6. **Checks.** `PYTHONPATH=src python3 -m unittest discover -s tests` runs the model-free tests.

Receipts live in `benchmarks/results/`. Paths in them are written with placeholder roots (`$LAB_ROOT`, `$AI_STACK`, `$HOME`); they are names, not variables anything reads. New receipts are made with `tools/export_receipt.py`, which keeps numbers, hashes, argv and hardware and refuses to write anything that looks like a path, user, host, serial or key.

<!-- pending: one command that reruns the headline table and writes receipts. It does not exist yet. -->
<!-- pending: where the Rust engine lives (same repository or its own) and its build line. -->

## Limits

- One laptop. Other X Elite and X Plus machines, other RAM sizes and other firmware are untested.
- Runs reached 97–98 °C. A long-session thermal test has not been done.
- Tasks are synthetic coding tasks with independent checks, not a broad quality benchmark.
- Several results rest on one or two runs. The attribution document marks each one.

## Credits

The memory-frequency work builds on the Surface Pro 11 memlat patches and the upstream SCMI memlat RFC (links in `docs/MEMLAT.md`). The engines build on [llama.cpp](https://github.com/ggml-org/llama.cpp) and ggml, [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) and [BigMoeOnEdge](https://github.com/Helldez/BigMoeOnEdge).

## Licence

MIT, copyright Oscar Nordström (n0rpan). Vendored and patched third-party code keeps the licence of its source; see `THIRD_PARTY_NOTICES` and `third_party/`. Model weights and vendor runtimes are not included and have their own terms.
