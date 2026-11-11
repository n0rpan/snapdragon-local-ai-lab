# Reproduce

How to build the engines, get the models, run the benchmarks and apply the quality gates. Setup is manual today; where a step has no tool yet, this page says so.

**Hardware assumed:** a Snapdragon X Elite (X1E) laptop with 32 GB RAM and an NVMe SSD, Linux ARM64. Other Snapdragon X machines should build and run; their numbers are exactly what we would like to see ([help wanted 6](HELP-WANTED.md#6-numbers-from-other-snapdragon-x-laptops)).

## 1. Prerequisites

| Need | Why |
|---|---|
| Linux ARM64, kernel with cgroup v2 and systemd user services | the controller uses systemd scopes for memory limits |
| Rust stable (edition 2021 workspace) | engines |
| GCC 16 (`cc`, `c++`) | the vendored ggml kernels are compiled with `-O3 -march=armv8.6-a+dotprod+i8mm+fp16+bf16+nosve+nosme -fno-fast-math`; bit-exactness against llama.cpp was established with GCC 16 |
| Python ≥3.12 | controller, tools and tests |
| ~90 GB free on a fast unencrypted or lightly encrypted NVMe partition | Flash weights + its expert sidecar |
| Optional: the MEMLAT boot entry | without it CPU memory reads sit near 49 GB/s instead of ~116 GB/s on the test laptop, and every decode number is far lower. Laptop- and firmware-specific: read [MEMLAT.md](MEMLAT.md) first |

## 2. Build the engines

From a checkout of the engine repository (snapdragon-engine):

```sh
# MoE engine (Ornith, Qwen3.6): oxe-server + tools
CARGO_BUILD_JOBS=2 nice -n 19 cargo build --release -p server -p oxe-cli

# Flash engine (needs the flash feature and the qwen4exp crate)
CARGO_BUILD_JOBS=2 nice -n 19 cargo build --release -p qwen4exp -p server --features flash
```

**Flash build prerequisite (BigMoeOnEdge ggml).** `crates/qwen4exp/build.rs` links the Flash graph port against the *production* ggml static libraries (`libggml-base.a`, `libggml-cpu.a`) of a BigMoeOnEdge build, so the numerics match the engine Flash replaced bit for bit. Build them once (no model needed):

1. Reconstruct the source exactly as in [FLASH-REBUILD.md](FLASH-REBUILD.md), "Reconstruct in a new directory": public BigMoeOnEdge `930b2c9ebd8c2cface6566bd9f11075ac07adfe0` (Apache-2.0) + the llama.cpp fork `7ac98f4a8089a36c0581b7c6e8973417a8fa175c` (MIT) + [`patches/flash-demand-mapping.patch`](../patches/flash-demand-mapping.patch) as a local commit (tree `4f0c6498e9ac5a2b098c99a0d8398dfc738fbdef`). Our development notes call that commit llama.cpp `04ff7df`; its ggml sources are those of `7ac98f4`.
2. Configure with the CMake command in FLASH-REBUILD.md ("Build with the qualified CPU configuration": Release, `-mcpu=oryon-1+rng+crc+nopauth+dotprod+i8mm+nosve+nosme`, OpenMP on, LLAMAFILE on, CPU_REPACK and KleidiAI off, static), and build just the two libraries: `cmake --build "$FLASH_WORK/build" --parallel 2 --target ggml-base ggml-cpu`.
3. From the engine checkout: `BIGMOE_BUILD="$FLASH_WORK/build" tools/sync_ggml_prod.sh "$FLASH_WORK/parent"`. It copies the libraries and headers with a SHA-256 MANIFEST into `$OXF_GGML_PROD` (default `~/.cache/local-ai/snapdragon-engine/ggml-prod`); without them the Flash build stops with "production ggml missing".

Production reference (GCC 16.1.1, CMake 4.4.4, Ninja 1.13.2): `libggml-base.a` sha256 `bb0ca3f811548f7b51c01f0e145225a04f16de6466a2d45e9db30cf6120ba067`, `libggml-cpu.a` `adf384be7a9de1c0a4b3cdaf39efa84975df945ac9af8909a811579f8b8e1d01`. Another compiler gives other hashes; `flash-m3 selftest` and the Flash identity gates decide. The MoE engine and the image crates do not need any of this. Licences: BigMoeOnEdge is Apache-2.0 and llama.cpp/ggml MIT; keep their notices when you build or redistribute.

`.cargo/config.toml` sets `target-cpu=oryon-1`. On a different ARM core change it, and expect the bit-exactness fixtures to need regenerating.

Model-free tests (no weights needed):

```sh
cargo test --release -p kernels-sys -p model -p sched -p tokenizer -p chat -p server
cargo test --release -p server --lib --features flash      # Flash server lib
cargo test --release -p qwen4exp --lib                     # Flash graph, cache, ordering
./target/release/flash-m3 selftest                         # Flash kernel self-tests (synthetic)
```

Lab repository (this one):

```sh
PYTHONPATH=src python3 -B -m unittest discover -s tests
PYTHONPATH=src python3 -B -m snapdragon_ai_lab doctor
```

## 3. Get the models

No weights are in either repository. Every file is pinned by repository, revision, file name, size and SHA-256 in [`dependencies/model-artifacts.json`](../dependencies/model-artifacts.json). Example with the Hugging Face CLI:

```sh
huggingface-cli download unsloth/Qwen3.6-35B-A3B-MTP-GGUF Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --revision 5bc3e238d916f48a861bac2f8a1990a0e9b7e98d --local-dir models/
sha256sum models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf   # compare with the manifest
```

| Model | Repository | Files |
|---|---|---|
| Ornith 1.5 35B-A3B | `mudler/Ornith-1.5-35B-A3B-APEX-MTP-GGUF` | `Ornith-1.5-35B-A3B-APEX-MTP-Compact.gguf` |
| Qwen3.6 35B-A3B MTP | `unsloth/Qwen3.6-35B-A3B-MTP-GGUF` | `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` |
| Qwen3.8 Flash-Next | `unsloth/Qwen3.8-Flash-Next-GGUF` | `UD-IQ3_XXS/…-0000{1..4}-of-00004.gguf` (83.6 GB) |
| Qwen3.8 27B dense | `DevQuasar/Qwen.Qwen3.8-27B-GGUF` | `Q4_K_M/…Q4_K_M.gguf` |
| Z-Image Turbo | `leejet/Z-Image-Turbo-GGUF` + text encoder + VAE | see the manifest |

Each model keeps its publisher's licence; read it before use.

**Flash expert sidecar.** The Flash engine reads experts from a repacked sidecar file, built once from the four shards (heavy IO, about the size of the expert weights):

```sh
./target/release/flash-m0 sidecar     # see M0-FLASH.md in the engine repository for paths and options
```

Put it on the fastest partition you have; encryption costs read latency ([ARCHITECTURE.md §3](ARCHITECTURE.md#3-flash-ssd-streaming-for-a-model-27-larger-than-ram)).

## 4. Run a server by hand

The controller normally starts engines; to try one directly:

```sh
# Ornith, as the installed profile runs it (decode cores 0-7, prefill 0-11, 32K context, MTP depth 1)
./target/release/oxe-server --engine oxe --model models/Ornith-1.5-35B-A3B-APEX-MTP-Compact.gguf \
  --alias ornith --host 127.0.0.1 --port 8080 --ctx-size 32768 \
  --oxe-cpus 0-7 --oxe-prefill-cpus 0-11 --spec-type draft-mtp --spec-draft-n-max 1

# Flash, installed tier: fixed 14 GiB expert cache, prefill order 2
./target/release/oxe-server --engine flash --model models/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00004.gguf \
  --flash-sidecar /path/to/sidecar --flash-cache-gib 14 --flash-pf-order 2 --port 8081
```

The exact installed argv (kernel mask, verify lanes, groups, draft head) is recorded per release in the stage receipts; `oxe-server` with no arguments prints its usage. Run Flash only with enough free RAM: [ARCHITECTURE.md §7](ARCHITECTURE.md#7-memory-admission). Then point any OpenAI client at `http://127.0.0.1:8080/v1`.

For the full controller (one port, lazy loading, idle unload, memory guard, prefix store, harness configs), follow [RUNTIME-SETUP.md](RUNTIME-SETUP.md).

## 5. Benchmarks

| What | How | Notes |
|---|---|---|
| MoE decode, fixed prompt | `tools/run_bench.sh` in the engine repo; `oxe bench` | real maths, fixed prompt token ids |
| MoE prefill | `oxe bench` prefill shapes (2,048 new tokens on prefix 0 / 6,144) | report the kernel mask and group |
| Flash decode / prefill | `flash-m3` in the engine repo (`prefill`, `export`, `bench-dense`) and timed requests against `oxe-server --engine flash` | always state the cache size and the free RAM at launch |
| Stock baseline | upstream llama.cpp at `5f490d7bfe78844e6390c869454c149fd7e003a8`, `llama-bench -p 512 -n 128` at 8 and 12 threads | [UPSTREAM-EVIDENCE.md](UPSTREAM-EVIDENCE.md) |
| Mac comparison | `python3 tools/mac_reproduce.py plan` / `verify-models` / `run` | same model files by SHA256; untested on a Mac |
| Your own receipt | `python3 -I tools/export_receipt.py export …` | keeps numbers, hashes, argv and hardware; refuses to write anything that looks like a path, user, host, serial or key |

Rules that make a number comparable with ours (details in [RESULTS.md §2](RESULTS.md#2-method)):

1. Warm up, drop the first run, compare medians; never put a cold run in a timing arm.
2. Interleave arms (A B B A) in one block. This laptop throttles ~25% within 30 s of all-core load; sequential blocks are not comparable.
3. One change per arm.
4. State context length, threads and placement, cache size (Flash), temperature and power source.
5. Effects under 3% are noise for a decision.

**Not available yet:** a single command that reruns every README row and writes receipts. The README rows were measured by the project's internal batch runner against the frozen binaries named in each receipt.

## 6. Quality gates

| Class | Gate | How it is checked |
|---|---|---|
| lossless | bit-identical logits/state or identical greedy tokens on ≥3 fixtures | engine `oxe compare` / `verify`, state-file `cmp`, token-id hashes |
| reorder | mean KLD ≤ 2× the reference's own batched-vs-single noise (Flash ~5.5e-4, Ornith 0.027); top-1 only reported with ≥2K positions | teacher-forced logits against the exact path; the llama.cpp reference dumper in `tools/oracle/` of the engine repo, `tools/engine_oracle.py` here |
| lossy | a noise-level metric + a real task check, opt-in only | protected task / image seeds |
| images | PSNR gate = min(30 dB, own 1-ulp noise − 6 dB) on ≥3 seeds | parity image vs stable-diffusion.cpp |
| everything installed | protected OpenCode task 13/13, post-install regression, rollback | [POST-INSTALL-REGRESSION.md](POST-INSTALL-REGRESSION.md) |

The protected OpenCode task ("normal13") is a small repair job with 13 independent checks; fixture, oracle and runner are in [`benchmarks/classic/`](../benchmarks/classic/) (`normal13.py`, `controller_normal13.py`, see `CONTROLLER_NORMAL13.md`). A pass means the harness completed real tool calls, edits and a final answer through the local model.
