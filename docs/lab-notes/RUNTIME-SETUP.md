# Manual runtime setup — source preview

> **Historical.** This is the early manual source-preview setup guide for the llama-server and sd-cli stack. It is kept for the record. Its profiles and resource limits (120 s idle, 16K and 4-thread profiles, old memory limits) are out of date; do not copy them. For the current setup, see [REPRODUCE.md](../REPRODUCE.md); for which engine runs, see [ENGINE-SELECTION.md](../ENGINE-SELECTION.md).

The coding/image core is installed and lifecycle-verified on the test Yoga Slim 7x. This repository is a **manual source preview**, not a tested one-click runtime/model installer for another laptop. The optional panel installer installs the panel and its small planning launcher; it requires the existing control command. The recipe below prepares the same contracts with portable paths; building, installation and native qualification were **not rerun for this document**. Flash-Next and the NPU helper are experimental and are not ready profiles in these examples.

Use one CPU llama-server build for both coding profiles and one CPU sd-cli build for images. Keep model weights, binaries, private state and generated images outside the Git repository. Model licenses are separate from this repository's MIT license.

## Source and engine identity

The selected llama.cpp source is `954de248ac6907dd05dc19f0f511ab18f2807a54` plus [llama-small-batch-pool.patch](../../patches/llama-small-batch-pool.patch). [Audited provenance](../../patches/llama-source-provenance.json) records the exact source diff and tested binary SHA `1bd3e439f357365c50fc8390cc2c7e4cde2f86f14fa1771a65f2323a307457b1`. KleidiAI is v1.32.0. The tested CMake cache used Release, static libraries, CPU repack/OpenMP/KleidiAI on, native auto-detection and Vulkan off, and:

```text
GGML_CPU_ARM_ARCH=armv8.6-a+dotprod+i8mm+fp16+bf16+nosve+nosme
```

A portable reconstruction starting point, run serially and with sufficient power/RAM:

```sh
git clone https://github.com/ggml-org/llama.cpp.git llama-source
git -C llama-source checkout 954de248ac6907dd05dc19f0f511ab18f2807a54
git -C llama-source apply /path/to/snapdragon-local-ai-lab/patches/llama-small-batch-pool.patch
cmake -S llama-source -B llama-build -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=OFF -DGGML_NATIVE=OFF -DGGML_VULKAN=OFF \
  -DGGML_CPU_REPACK=ON -DGGML_CPU_KLEIDIAI=ON -DGGML_OPENMP=ON \
  -DGGML_CPU_ARM_ARCH=armv8.6-a+dotprod+i8mm+fp16+bf16+nosve+nosme
cmake --build llama-build -j 1 --target llama-server
```

Confirm the resulting configuration enables no unwanted GPU/NPU backend on your build. The listed options come from the audited cache; they do not pin every host dependency or compiler artifact. Record your resulting executable SHA. A different rebuilt hash requires your own qualification and an updated profile identity, not copying the tested hash to a different binary.

For the image engine, follow [SOURCE-PROVENANCE.md](../SOURCE-PROVENANCE.md) to reconstruct stable-diffusion.cpp `5a1ba5d895bec4e07deab80d6d791721cd59d9c2`, its pinned ggml and the six-file repack patch. The audited image recipe used Release, `SD_VULKAN=OFF`, `GGML_VULKAN=OFF`, `SD_BUILD_EXAMPLES=ON`, `BUILD_SHARED_LIBS=OFF`, `GGML_NATIVE=OFF`, `GGML_CPU_REPACK=ON`, `GGML_CPU_KLEIDIAI=OFF`, `GGML_OPENMP=ON`, with both C/C++ flags:

```text
-mcpu=oryon-1+rng+crc+nopauth+dotprod+i8mm+nosve+nosme -U__ARM_FEATURE_SVE -U__ARM_FEATURE_SME
```

Build only `sd-cli`, serially, using the required platform dependencies/sysroot for your toolchain. The audited deployment's hash is `e59a180c339266ff8b29e31c3ed5ae6b68ddb05ea1de9e199bdbf9ecea44f606`. A bit-identical independent rebuild is **not** claimed for either engine.

## Models: pin and verify separately

The coding profiles expect these exact tested identities; filenames in the examples are local portable names and may be renamed without changing bytes.

| Profile | Public source | Bytes | SHA256 |
| --- | --- | ---: | --- |
| Dense Qwen3.8-27B Q4_K_M (retired in v0.1.2; historical identity only, no launch path) | [DevQuasar catalog](https://huggingface.co/DevQuasar/Qwen.Qwen3.8-27B-GGUF), `Q4_K_M/Qwen.Qwen3.8-27B.f16.gguf.Q4_K_M.gguf` | 16547400064 | `9f27cff95dd7cfcb721aa7a581504fdba72d1b9cad7aaf58b4b332f7d38184f3` |
| MoE Qwen3.6-35B-A3B MTP UD-Q4_K_M | [pinned Unsloth revision](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/tree/5bc3e238d916f48a861bac2f8a1990a0e9b7e98d), `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` | 22663387424 | `0b21525e972670ed59e1812e170b27c26355381f0656ecc4e25617ece7dac58b` |
| Image diffusion Q8 | [leejet Z-Image-Turbo catalog](https://huggingface.co/leejet/Z-Image-Turbo-GGUF), `z_image_turbo-Q8_0.gguf` | 6577440704 | `df1c5baa86d1398c979495a6072dbcee79444fdb884a2445582ba0769c44e9a1` |

The dense catalog's historical download used `main`; its checksum above is the immutable identity gate. Resolve a current public commit/LFS SHA and verify it matches before downloading rather than assuming current `main` is identical. Do the same for image weights. No downloader is run by this guide or example renderer.

Image Studio additionally expects these pipeline dependencies under `$HOME/.local/share/ai-stack/models/images/`:

```text
z_image_turbo-Q4_K.gguf                     # optional selectable Q4
z_image_turbo-Q8_0.gguf
Qwen3-4B-Instruct-2507-Q4_K_M.gguf            # image text encoder, not the coding server
vae/vae.safetensors
```

Use the compatible text encoder/VAE listed by the [pinned image engine's model instructions](https://github.com/leejet/stable-diffusion.cpp/tree/5a1ba5d895bec4e07deab80d6d791721cd59d9c2), not a guessed similarly named model. For each dependency record publisher repository/revision/file, expected byte count and SHA256 from its public catalog/LFS metadata, then verify downloaded bytes with `sha256sum` and `stat -c %s`. The [dependency inventory](../DEPENDENCIES.md) now pins public Q4/text-encoder identities and the compatible VAE. Completed serialized full checksums now identify local Q4, encoder and VAE content against that manifest; Q8 has its prior full-checksum receipt. Historical VAE download origin is not claimed. A new installation must verify its own files. Do not set a profile validated until its exact full pipeline passes local checks.

The controller checks model existence/size at load and uses recorded SHA identities for prefix caching; it does **not** rehash every model at each request. Manual full-file verification is required before accepting `ready:true`.

## Stage portable configuration without installing it

From the repository root:

```sh
python3 tools/render_runtime_examples.py /tmp/local-ai-example-stage
```

The destination must not exist. This helper only replaces `@HOME@` in [example JSON](../../examples/runtime/llm-runtime.json), validates JSON syntax and writes a new staging directory. It does not install services, enable profiles, download models or start inference. JSON itself does not expand `$HOME`; use rendered absolute paths. Units use systemd `%h`. If you use custom XDG roots, edit the staged JSON and unit environment paths consistently; the runtime uses explicit paths, not universal automatic XDG discovery.

Review the staging directory. Both coding profiles deliberately have `ready:false`; the optimized image profile has `enabled:false, validated:false`. Populate verified binary/model identities and run your own direct synthetic checks before changing these gates. Do not treat the tested laptop's validation as your device's receipt.

The selected coding settings in the examples are 16K context, four decode/ten batch threads, batch/microbatch 256/256, CPU repacking, F16 KV, flash attention off, four recurrent checkpoints, thinking off and one slot. The explicit classic OpenCode model limits allow 4096 output tokens within 16K context and 12K input; Flash planning and the dynamic selected-model alias are capped at 2048 output, so switching that alias to Flash cannot request an unsupported 4096-token answer. MoE alone uses `draft-mtp` with two draft tokens and `LLAMA_CPU_SMALL_BATCH_LIMIT=3`. The ordinary `qwen-local` alias selects MoE. The dense 27B was retired in v0.1.2: `local-ai dense` and a request for `qwen3.8-27b-local` now answer with the retired message. MTP is retained as the measured profile, not claimed to improve every complete agent task.

The derived prefix store is bounded to 3 GiB on disk. It saves only a leading instruction/tool prefix, excludes user turns/tool results, uses owner-only files and identities tied to exact model/build/settings, and clears independently of unloading RAM. Native `--cache-ram 0` does not disable this separate controller-managed store. Changed or unsuitable prefixes safely miss.

## Manual installation after review

Back up existing user units/configuration/CLI and keep a rollback copy. Do not overwrite a working setup while an image, inference or benchmark job is active. For a new installation, create owner-only private directories, copy the chat runtime files and the image worker source, place your qualified binaries/weights and copy the **reviewed rendered** configuration:

```sh
install -d -m 700 "$HOME/.local/share/ai-stack/runtime" \
  "$HOME/.local/share/ai-stack/releases/llama-cpu" \
  "$HOME/.local/share/ai-stack/releases/sd-cpu-repack" \
  "$HOME/.config/local-ai/llm" "$HOME/.config/local-ai/images" \
  "$HOME/.local/state/local-ai" "$HOME/.config/systemd/user" "$HOME/.local/bin"
install -m 600 src/snapdragon_ai_lab/runtime/{controller.py,classic_beta.py,prefix_plan.py,prefix_store.py,bigmoe_adapter.py,bigmoe_runtime.py,exec_no_thp.py,slim_request.py,studio} \
  "$HOME/.local/share/ai-stack/runtime/"
install -m 700 src/snapdragon_ai_lab/runtime/local-ai "$HOME/.local/bin/local-ai"
install -m 700 src/snapdragon_ai_lab/runtime/launch_local_plan.py "$HOME/.local/bin/launch_local_plan.py"
# Copy your reviewed rendered llm-runtime.json as ~/.config/local-ai/llm/runtime.json;
# copy images/*.json and units/*.service to the corresponding directories above.
# Copy qualified llama-server and sd-cli to the two release directories above, mode700.
```

The control command opens OpenCode in `LOCAL_AI_CODE_DIRECTORY`, defaulting to `$HOME/Work`. For a manual install, choose an existing project directory or create that directory before the first launch. Source bootstrap staging already declares `Work`; the manual file-copy commands above do not create it. Set the override in the environment used to launch `local-ai` or the panel, rather than assuming a service-only environment reaches desktop actions.

OpenCode context presets (MoE launches only: Qwen3.6 MoE and Ornith). Both modes use context 131072 and output 16384. `LOCAL_AI_OPENCODE_MODE=fast` (the default when unset or empty, O-105) sets input 65536, so OpenCode 2.2.7 compacts at about 49536 tokens. `deep` (explicit) sets input 120000, so it compacts at about 102000 tokens; it keeps more history but is slower on heavy tool turns at depth. Both modes keep the last 8000 tokens after compaction. Any other value stops the launch with an error. Set the variable in the same environment as `LOCAL_AI_CODE_DIRECTORY`. OpenCode 2.2.7 has only one compaction setting per instance: it cannot set compaction per model, and it ignores the legacy provider `whitelist`/`blacklist` keys. Keeping 8000 tokens is unsafe for a 16K model, so a MoE launch also marks every small-context local model (`qwen-local`, `qwen3.8-27b-local`, `qwen3.8-flash-next-local`) as `disabled`. Those models then disappear from that instance's model list, and a session pinned to one fails with "Model unavailable" instead of running with the MoE setting. Flash and planning launches are unchanged; open Flash in a new launch. Both modes keep the last 8000 tokens after compaction. Any other value stops the launch with an error. Set the variable in the same environment as `LOCAL_AI_CODE_DIRECTORY`. OpenCode 2.2.7 has only one compaction setting per instance: it cannot set compaction per model, and it ignores the legacy provider `whitelist`/`blacklist` keys. Keeping 8000 tokens is unsafe for a 16K model, so a MoE launch also marks every small-context local model (`qwen-local`, `qwen3.8-27b-local`, `qwen3.8-flash-next-local`) as `disabled`. Those models then disappear from that instance's model list, and a session pinned to one fails with "Model unavailable" instead of running with the MoE setting. Flash and planning launches are unchanged; open Flash in a new launch.

All participating engines must use **the same lock inode** at `$HOME/.local/state/local-ai/inference.lock`. On a fresh idle installation, create it owner-only once. If migrating an older active lock, stop all participating jobs first or preserve it by a hard link on the same filesystem; never replace a locked file with a new inode and assume exclusion remains intact. Do not run chat and image engines independently around the controller's lease.

The example chat unit caps CPU at 800%, MemoryHigh/Max at 24/25 GiB; Studio uses 1000%, 21/22 GiB (historical). The installed envelope on the test laptop (2026-11-12) is: chat `CPUQuota=1200%`, `MemoryHigh=infinity`, `MemoryMax=28G`; Studio `CPUQuota=1200%`, 21/22 GiB; both `MemorySwapMax=0`. The installed controller runs from a content-addressed `~/.local/share/ai-stack/runtime-releases/<id>/`, not `runtime/`. Both disable swap and core dumps and set owner-only umask. These are the tested device's resource envelopes, not permission to run both models simultaneously or proof another laptop has enough RAM. Reassess on different hardware.

After local gates are deliberately enabled and files reviewed:

```sh
systemctl --user daemon-reload
systemctl --user enable --now local-ai-chat.service prompt-studio-app.service
```

These start lightweight frontends, not model weights. The controller binds public loopback `8092`, launches its private llama backend on `8093` on demand, then unloads after the idle timeout (`LOCAL_LLM_IDLE_SECONDS`, 600 s by default; this guide's original 120 s is historical). Studio binds loopback `8090` and starts a separate image worker per request. Status polling must not wake a sleeping model. The image handoff asks only an idle controller to unload; active jobs keep their lease. Never expose these unauthenticated local frontends to a network.

## Private engine port conflicts

The controller checks its private engine port before loading weights. An occupied
port returns `Local engine port N is in use by PROCESS`, with the owning process
name when readable. It never connects to, stops or signals the other application.
An already loaded engine is reused without probing its own listener.

Choose another private port through the chat service's environment:
`LOCAL_LLM_BACKEND_PORT` controls the llama/oxe engine (default 8093), and
`LOCAL_FLASH_BACKEND_PORT` controls the legacy streamed BigMoe backend (default
8094). The two ports must differ. For example, set
`Environment=LOCAL_LLM_BACKEND_PORT=8193` in a reviewed user-service drop-in,
reload the user units and restart an idle chat service. The client-facing port
does not change. A program can still claim a port between preflight and native
launch; the engine's bind remains authoritative.

## Image presets and private files

The examples select Q8 for draft 256×256 and final 1024×1024, eight steps, ten threads, diffusion flash attention on, CPU backend, CFG 1, tiled VAE with tile size 512 and image metadata disabled. Q4 remains selectable when its assets are present. The optimized profile is limited to the checked text-to-image dimensions and eight steps; it validates the installed release binary hash and enables `SD_CPU_REPACK_EXPERIMENT=1` per worker. Other settings may fall back to the base CPU executable and require separate validation. GPU stays disabled.

Studio opens on the 256 draft, can draw 1–4 seed variations, and offers **Quick upscale ×4** (ESRGAN only) and **Final (refine)** (img2img of the chosen draft) on a chosen image; photo edits use Light touch / Restyle / Reimagine strength presets. See [STUDIO-UPSCALE.md](../STUDIO-UPSCALE.md) for the flow, the optional `upscale.json`, the ESRGAN model download and the calibration still required.

Use `SD_HOME`, `SD_CLI`, `LOCAL_IMAGE_CONFIG_DIR`, `LOCAL_AI_LEASE` and `IMG_OUTDIR` from the unit examples. Default images are written to `$HOME/Pictures/sd`; uploads persist in its `_uploads` subdirectory. They are private user files and are not automatically erased after generation. No prompt classifier/blocklist was added to this Studio. Disabled PNG prompt metadata is not a promise about every external client or diagnostic log; consult the source and your own retention policy.

## OpenCode v2 and panel

[opencode-v2.json](../../examples/runtime/opencode-v2.json) uses the tested v2 `providers` / `settings` / `agents` schema. Merge this local provider/model selection into your own reviewed configuration; **do not replace unrelated providers or settings wholesale**. It targets `http://127.0.0.1:8092/v1`, uses a local-only placeholder credential, keeps title generation disabled and pins ordinary coding to `local-qwen/qwen3.6-35b-a3b-local`; the dynamic `qwen-local` alias remains available for explicitly selected chat. No real service/auth/session configuration was used to make this example. The exact working standalone version was OpenCode 2.2.6; other versions need their own schema/protocol check.

OpenCode opened normally uses explicit MoE and triggers lazy loading. The panel's OpenCode action follows a selected MoE or qualified Ornith model through a per-launch model overlay, preserving ordinary agent permissions; selected Flash uses the read-only planning launch. The qualified Flash planning launcher is described in [FLASH-NEXT-PLANNING.md](FLASH-NEXT-PLANNING.md). It need not be started through the panel. Install the optional [Omarchy panel](../OMARCHY-PLUGIN.md) only after the CLI works, using explicit `--replace-plugin ID` for a personal clone. It stays in the always-visible right section and does not install the runtime. `local-ai status --json` is read-only; `local-ai moe`, `ornith`, `flash`, `plan`, `load`, `unload`, `images`, `code` and `clear-cache` are explicit actions.

The optional [Ornith integration](../ORNITH-BETA.md) requires its own passed screen and protected 13-case task, exact artifact identity and explicit review. It keeps the Qwen default unchanged and disables the external prefix store for that beta profile.

## End-to-end checks and rollback

Use fixed synthetic inputs. Check both models through the configured OpenCode provider with independent protected tests; count load, prefix priming, prompt processing, tool calls and retries. Check fresh-session prefix reuse across full unload/reload, status polling through the real idle timer, cache clearing while a model remains loaded, image handoff/exclusion, chat resumption and RAM recovery without reboot. Verify an actual Q8 draft and final through Studio, including limited visual review. See [BENCHMARKS.md](../BENCHMARKS.md), [measured progress](../BEFORE-AFTER.md), [disabled paths](../FAILED-PATHS.md) and release checklist (private release notes). Power/thermal and fixture limits still apply.

For rollback, finish or cancel only your own active job, stop the two user services, restore backed-up units/configuration/runtime/CLI, preserve the canonical lease inode, reload user units and restart the reviewed prior frontends. Restore the panel via its documented backup procedure. Keep models and private images intact; deleting weights, chats or uploads is not part of rollback.

Offline checks performed for this recipe: example JSON syntax/placeholder rendering, actual controller alias resolution without constructing a manager, actual safe-prefix planner excluding a synthetic user sentinel, disabled example gates and v2 provider/preset shape. No native build, systemd installation, model run, OpenCode launch or desktop/configuration edit was executed for this documentation task. Portable benchmark orchestration and final public-file/license review still need release work; the image content-identity inventory is complete.
