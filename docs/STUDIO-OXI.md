# Image Studio on oxi (FLUX.2 klein 512 px)

> **Experimental.** `oxi` is experimental. Persistent prompt conditioning is **off by default**;
> opt in with `{"oxi": {"prompt_cache": true}}` in `$LOCAL_IMAGE_CONFIG_DIR/runtime-settings.json`.
> The cache contains text-encoder outputs derived from prompts, even though it contains no prompt text.
> Set the Studio engine to `sdcpp` to use stable-diffusion.cpp only.

Studio's text-to-image with **FLUX.2 klein 4B Q8 at 512 × 512** runs on `oxi`, the lab's Rust image
engine, on its fast path: int8 linears, scale chunks 32/32, f32 attention. That setting passed the
image gate on seeds 42/43/44 (31.47 / 32.08 / 34.08 dB against sd.cpp's own image; gate 27.81 dB =
measured noise floor minus 6 dB; `runq/F-IMG-ATTN-1`). Everything else still runs on sd-cli.

## Routing

| Request | Engine |
|---|---|
| klein Q8, 512 × 512, engine `auto` or `oxi` | oxi |
| klein at any other size (768, 1024, non-square) | sd.cpp (labelled slower in the UI) |
| klein Q4 | sd.cpp (the gate ran on Q8 only) |
| Edit (instruction), masked edit, refine, quick upscale, Z-Image Q4/Q8 | sd.cpp, unchanged |
| oxi binary or probe sd-cli not installed | sd.cpp |
| oxi fails (any exit code, missing or wrong-size image, exception) | `auto`: the whole request reruns on sd.cpp, the job says why; `oxi`: the job fails with oxi's error |
| engine `sdcpp` | sd.cpp for everything |

The engine setting lives in `$LOCAL_IMAGE_CONFIG_DIR/engine.json` (`{"engine": "auto"}`; missing or
invalid = `auto`). The Generate panel shows a selector for klein models; `GET /engine` reports it
and `POST /engine {"engine": "auto"|"oxi"|"sdcpp"}` changes it (same Host/Origin protections as the
other POSTs). Finished jobs report `engine` ("oxi" or "sd.cpp"), `engine_reason`, and on a fallback
`engine_note`; each image carries `backend`.

## How a request runs

oxi has no text encoder yet. For a prompt it has not seen:

1. `oxi generate --cond-prompt-file …` exits 3 (no cached conditioning for this prompt, model, size, steps).
2. The **probe** sd-cli (sd.cpp with the `SD_PROBE` dump patch) runs Studio's own klein argv with
   `SD_PROBE=1 SD_PROBE_DUMP_CALL=0`. Studio stops it at the first probe line after `dump.pe call=0`,
   that is right after text encoding and before any DiT work (about 7 s).
3. `oxi cond-put --sigmas computed --steps 4` stores the text context, RoPE table and the 4-step
   FLUX.2 schedule (checked against the dumped first sigma). Offline check on the seed 42 dumps: the
   stored files are bit-identical to the ones from a full 4-step hand-over.
4. `oxi generate` runs again and writes the PNG.

With persistence enabled, a repeated prompt can skip steps 2–3 while its matching conditioning entry
is retained. With the public default OFF, each job uses a fresh owner-only temporary cache, shared by
that job's variations and removed after success, failure or fallback. It never reads the persistent
conditioning directory. Variations still encode once within a job, then run one `oxi generate` per seed.
The reported 45.94 s warm result requires an opt-in persistent cache hit; it is not the default cold path.

`--noise sdcpp` draws sd-cli's own initial latent (a port of sd.cpp's default Philox RNG, bit-identical
to sd-cli's step-0 DiT input on the seed 42/43/44 dumps). So a seed gives the same picture on either
engine, within the image gate, and the sd.cpp fallback reproduces the oxi image's composition.

The prompt never reaches argv, a named file or a log: both engines read it from an anonymous memory
file (`--prompt-file` / `--cond-prompt-file /proc/self/fd/N`).

## Files and settings

| Env (unit) | Default | What |
|---|---|---|
| `OXI_BIN` | `$SD_HOME/releases/oxi/oxi` | frozen oxi release binary |
| `SD_CLI_PROBE` | `$SD_HOME/releases/sd-probe/sd-cli` | probe sd-cli for the prompt hand-over |
| `OXI_COND_CACHE` | `$SD_HOME/cache/oxi-cond` | opt-in persistent conditionings, newest 16 kept (~17 MB each); overridden per job when OFF |
| `OXI_WORK` | `$SD_HOME/cache/oxi-handover` | per-request dump directory, removed after each request |
| `STUDIO_PORT` | 8090 | lets a candidate run beside the installed service |

`runtime-settings.json` can hold
`{"oxi": {"threads": 12, "prompt_cache": false, "vae_conv": "f32"}}`.
The default thread count is 12, the gated setting. Only boolean `true` enables persistent conditioning;
missing, invalid or false settings keep it off. The approved lab machine explicitly opts in with
`"prompt_cache": true`; that local choice does not change the public default.

`vae_conv` accepts `f32`, `winograd` or `int8` independently of `prompt_cache`.
The public default is `f32`; any other value requires its own image gate (runq/VAE-1).
Removing `vae_conv` or setting an unknown value goes back to f32.

Starting with engine `claude/image-vae`, oxi frees the DiT and its pinned worker pool before the VAE
runs. Earlier builds (IMG-SHIP 339f185) left the main thread pinned to cpu 0, so all 12 VAE threads
shared one core: 21.9 s instead of about 3.5 s at 512 px. The pixels are the same either way.

**Privacy:** the conditioning cache holds the text encoder's output for recent prompts (not the
prompt text: the key is a SHA-256 and `meta.json` stores only the prompt's hash). Embeddings are
derived from the prompt, so the cache sits in the private stack directory (umask 0077) and is pruned
to 16 entries when enabled. Turning it off leaves existing persistent entries untouched and unused;
remove the directory separately if you want to erase them. Deleting `OXI_COND_CACHE` is always safe.

## Native text encoder (TE product binding, opt-in)

`runtime-settings.json` `{"oxi": {"text_encoder": ...}}`:

- `"sdcpp"` or absent (default): a new prompt runs the probe sd-cli up to model call 0, then `oxi cond-put`.
- `"native"`: a new prompt runs one `oxi cond-native` process: Qwen3 context in Rust with sd.cpp's own ggml
  Q8_0 kernels (`--te-backend cpu-ggml`), native FLUX.2 RoPE table and sigmas. No sd-cli, no DiT load for the prompt.
- `"native-npu"`: as `native`, but an NPU job runs `cond-native` through `image_npu_worker.py`, which pins the
  arithmetic (fp16, 3 sessions, 6 GiB weight budget). The encoder releases its sessions before it exits, so the DiT
  places its weights afterwards in its own process, under the same image permit (one at a time). The worker receipt
  records the encoder's actual DMA allocation; admission keeps the DiT's (larger) qualified figure.

Fallback: `cond-native` exit 4 (emphasis/escape syntax, BREAK, special-token or LoRA strings, non-UTF-8, more than
512 tokens) or any other failure drops the request to the sd-cli hand-over and legacy cache keys. NPU worker
recovery (70) never starts sd-cli.

Cache provenance: native entries are keyed and labelled with their encoder identity
(`qwen3-te/1;<arith>;<TE file>;<size>`); legacy entries keep their exact old key and `"encoder": "sd.cpp-handover"`.
Neither can replace or be read as the other. Rollback = remove the `text_encoder` key: legacy entries are found
again, native entries age out of the 16-entry prune.

## v0.1.3: stall watchdog, per-size gate, image-slot recovery

**Stall watchdog.** oxi, probe sd-cli and NPU-worker runs are stopped when they print no output line
for `runtime-settings.json` `{"oxi": {"stall_seconds": N}}` seconds (default 180, `0` disables, max
86400), plus a 10 s grace after the stop signal. The job then fails (engine `oxi`) or reruns on sd.cpp
(engine `auto`) like any other oxi failure. It stops the direct child only. The 180 s default has
not yet been measured against a cold 1024 px load or the NPU setup (hardware batch).

**Per-size routing gate.** A klein size other than 512 x 512 runs on oxi only when
`image_size_gate.py judge` wrote a PASS record for exactly that width x height, made with the oxi
binary Studio would run (sha256) and the same arithmetic settings. Records live in
`$OXI_SIZE_GATE_DIR`, else `<$LOCAL_IMAGE_CONFIG_DIR or ~/Work/prompt-studio>/oxi-size-gates/<W>x<H>.json`.
Without a record, after a FAIL, with another binary or after an unreadable record, the size stays on
sd.cpp. Gated sizes always run on the CPU (the NPU profile is qualified for 512 px only). This release
ships no record, so no size changes route. The tool is installed with the runtime
(`python3 -m snapdragon_ai_lab.runtime.image_size_gate judge|run`, see its `--help`): the gate is
`min(30, N - 6)` dB against the sd.cpp one-ULP noise floor N, on at least three seeds.

**Image slot broker.** A broker socket left behind by a crashed controller is removed at start-up
only after a re-probe shows that no process is listening on it. A socket with a wider mode than
expected is still refused and has to be removed by hand. A hold longer than
`LOCAL_AI_IMAGE_SLOT_WARN_SECONDS` (default 600) logs one warning.

## Tests

`tests/test_studio_oxi.py` (model-free): routing table, engine setting, oxi argv, hand-over stop
rule, the new-prompt and cached-prompt pipelines, every fallback trigger, the `oxi` setting without
fallback, cache pruning, the HTTP endpoints, and one real subprocess run of the stop + memfd path.
