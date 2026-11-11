# Image Studio on oxi (FLUX.2 klein 512 px)

> **Experimental.** `oxi` and its prompt-conditioning cache (`OXI_COND_CACHE`: text-encoder outputs of the last 16 prompts on disk, owner-only permissions, keyed by SHA-256, no prompt text) are experimental. Whether the cache stays on disk by default is an open decision. Set the Studio engine to `sdcpp` to use stable-diffusion.cpp only.

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

A repeated prompt (a new seed, more variations) skips steps 2-3. Variations encode once and then run
one `oxi generate` per seed.

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
| `OXI_COND_CACHE` | `$SD_HOME/cache/oxi-cond` | cached conditionings, newest 16 kept (~17 MB each) |
| `OXI_WORK` | `$SD_HOME/cache/oxi-handover` | per-request dump directory, removed after each request |
| `STUDIO_PORT` | 8090 | lets a candidate run beside the installed service |

`runtime-settings.json` can hold `{"oxi": {"threads": N}}` (default 12, the gated setting).

**Privacy:** the conditioning cache holds the text encoder's output for recent prompts (not the
prompt text: the key is a SHA-256 and `meta.json` stores only the prompt's hash). Embeddings are
derived from the prompt, so the cache sits in the private stack directory (umask 0077) and is pruned
to 16 entries. Deleting `OXI_COND_CACHE` is always safe.

## Tests

`tests/test_studio_oxi.py` (model-free): routing table, engine setting, oxi argv, hand-over stop
rule, the new-prompt and cached-prompt pipelines, every fallback trigger, the `oxi` setting without
fallback, cache pruning, the HTTP endpoints, and one real subprocess run of the stop + memfd path.
