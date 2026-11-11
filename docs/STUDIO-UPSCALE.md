# Studio: draft, choose, upscale or refine

Image Studio (`src/snapdragon_ai_lab/runtime/studio`) now starts with a 256×256 draft. You can ask for 1–4 variations, pick one, and then either enlarge **that** image or refine it at a larger size. This page covers what the old draft→final did, how the new flow calls the pinned stable-diffusion.cpp (`5a1ba5d`), which numbers are measured and which are estimates, and the checks Codex must run before the estimates or defaults are trusted.

Everything here was written and unit-tested offline with a stubbed engine. **No image was generated, upscaled or timed for this change.**

## What the old draft→final actually did

I checked this in the source before the change:

- **Fast draft** and **Final image** only filled in the form (model, size, steps) from `presets.json`. Neither button generated anything. **Generate** then ran one independent text-to-image `sd-cli` job.
- The final had no link to the draft. It was a new text-to-image run at 1024×1024 from the prompt. It never used the draft image as input.
- When the seed field was blank, Studio passed no `--seed`, so `sd-cli` used its default of **42** (`examples/common/common.h`: `int64_t seed = 42`). The "random" placeholder was wrong: every blank-seed draft and final used seed 42. Even with the same seed, a different resolution gives a different noise tensor, so the 1024 final is a **new image**. It is not an enlargement of the draft.
- On first load, if a 1024 profile or benchmark was marked good, the page opened with the Final preset selected.

## New flow

1. **Draft.** The page now always opens on the Draft preset (256×256). The **variations** selector (1–4) runs one `sd-cli` process with `-b N --seed S`, and image *i* gets seed `S+i` (`GenerationRequest::seed_for_batch`). If the seed is blank, `S` is 42, so a blank-seed single draft passes exactly the same command as before (no `--seed`, no `-b`). Outputs are `stem-%d.png`, which gives `stem-0.png` … `stem-(N-1).png`. Studio keeps a record of each finished image in memory (path, seed, model, size, steps and the prompt). The prompt never appears in `/job`, argv or files; it still reaches `sd-cli` only through the memfd `--prompt-file`.
2. **Choose.** A single result is selected automatically. With several variations, click the one you want. Each result lists the actions that are available for it and their estimated time.
3. **Quick upscale ×4.** This is ESRGAN only: `sd-cli -M upscale -i draft.png --upscale-model <x4 model> --upscale-tile-size 128 --threads N --backend cpu -o draft-x4.png --disable-image-metadata`. No diffusion model is loaded. A 256 draft becomes 1024. The pixels match the draft exactly, but the result looks soft and has no new detail.
4. **Final (refine).** This is img2img from the chosen image, with the same prompt and seed:
   - It enlarges the image first. With an ESRGAN model it runs the same upscale-mode command into `OUT/_work/…-esrgan.png`, an intermediate file that is deleted when the job ends. Without an ESRGAN model it skips this step and passes the draft directly (plain resize).
   - It then runs `sd-cli … -W T -H T --steps 8 --seed <draft seed> -i <enlarged or draft> --strength 0.45 --image-preprocess "target=init,mode=stretch,filter=lanczos"`. The pinned tree resizes init images with **nearest** by default (`docs/image_preprocessing.md`), so Lanczos is requested explicitly. This covers both the plain-resize fallback (256→T) and fitting the ×4 ESRGAN output to T.
   - The target is 768 or 1024 (`refine_targets` / `refine_target` in `upscale.json`; the default is 1024). The long edge is scaled to the target on the 64-pixel grid, so non-square drafts keep their aspect ratio.
   - Optional **ESRGAN ×2 after** appears only when `esrgan_x2_model` is configured (see the caveat below).

Strength 0.45 at 8 steps samples **4** steps. The pinned `src/pipeline/image.cpp` computes `t_enc = int(steps·strength)` and samples `t_enc+1` steps (`sampled_steps()` in Studio mirrors this), so refine costs about half a full pass at the target size.

### Why not native hires fix (`--hires`, PR #1451)

The pinned tree has `--hires`, `--hires-upscaler <name>`, `--hires-upscalers-dir`, `--hires-width/height`, `--hires-steps` and `--hires-denoising-strength` (`examples/common/common.cpp`; `src/pipeline/image.cpp` runs it after base sampling). It does **not** take a chosen finished image as input. It re-samples a base image from noise in the same process and then upscales and refines that. To use it here, Studio would have to replay the draft and count on bit-identical reproduction across binaries: drafts may use the repack release binary, while refine falls back to the base binary. It would also redo the 256 base pass. The explicit two-process path works on the actual chosen PNG, and it loads the diffusion model only once, because the ESRGAN step is a separate `-M upscale` process that loads no DiT. Codex may compare hires (`--hires --hires-upscaler 4x-UltraSharp --hires-width 1024 --hires-height 1024 --hires-steps 8 --hires-denoising-strength 0.45`, same seed, from a txt2img 256 base) as an experiment. It is not used by Studio.

### Engine facts verified in the pinned source (read-only)

| Need | Pinned `5a1ba5d` | Where |
|---|---|---|
| ESRGAN only | `-M upscale` needs `-i` and `--upscale-model`; `--upscale-repeats`, `--upscale-tile-size` (default 128) | `examples/cli/main.cpp:887`, `examples/common/common.cpp:910,1254` |
| Upscale failure | logs `upscale failed` / `new_upscaler_ctx failed` but **exits 0 and saves the unscaled image** | `examples/cli/main.cpp:943-980` |
| img2img | `-i/--init-img`, `--strength` (default 0.75) | `common.cpp:1137,1341` |
| Init resize | default nearest; `--image-preprocess target=init,mode=…,filter=lanczos` | `docs/image_preprocessing.md`, `src/runtime/image_preprocess.cpp` |
| Mask | `--mask` (grayscale, rounded; 1 = denoise, 0 = keep), applied for any model as a denoise mask | `common.cpp:1147`, `src/pipeline/diffusion_engine.cpp:2345,2863` |
| Batch seeds | `-b N` → seed+i; `-o name-%d.png` numbers from 0 | `src/pipeline/request.cpp:231`, `examples/cli/main.cpp:516` |
| ESRGAN scale | read from the weights: `conv_up2` → ×4, only `conv_up1` → ×2; old (`model.N`) and RealESRGAN key layouts are mapped | `src/model/upscaler/esrgan.hpp:62-75`, `src/name_conversion.cpp:1517` |

Because `-M upscale` exits 0 even when ESRGAN fails, Studio checks the PNG header size after every ESRGAN step. A mismatch fails the job with "…produced W×H, expected …".

**×2 caveat.** `RealESRGAN_x2plus.pth` is an x4 network with pixel-unshuffle input (12 channels). The pinned ESRGAN graph has no pixel-unshuffle, so it will probably be misdetected as ×4 and fail. That is why `esrgan_x2_model` defaults to `null` and the "ESRGAN ×2 after" checkbox stays hidden. Only configure it with a model that `sd-cli -M upscale` has been checked to double exactly.

## Model download (Codex/owner, not done here)

Put the file under `$SD_HOME/models/images/upscale/`. Studio only accepts upscaler paths inside `models/images` with suffix `.pth`, `.safetensors` or `.gguf`. Anything else counts as "not installed".

| Model | Source | Size (bytes) | sha256 | License |
|---|---|---:|---|---|
| **RealESRGAN_x4plus.pth** (default) | https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth (same bytes at HF `lllyasviel/Annotators`) | 67040989 | `4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1` (HF LFS oid; the GitHub asset has the same size) | BSD-3-Clause (Real-ESRGAN repo) |
| 4x-UltraSharp.pth (alternative, sharper) | HF `Kim2091/UltraSharp` | 66961958 | `a5812231fc936b42af08a5edba784195495d303d5b3248c24489ef0c4021fe01` | **CC BY-NC-SA 4.0 (non-commercial)** |
| 4x-UltraSharp.safetensors | HF `Kim2091/UltraSharp` | 66864028 | `36a340b5509b699d2c06cb445ddc1d3d39199ac734d889ed6d7915f60e05bcbc` | CC BY-NC-SA 4.0 |

The hashes come from the Hugging Face tree API (`/api/models/<repo>/tree/main`, `lfs.oid`) on 2026-11-09. Check after download with `sha256sum`. `RealESRGAN_x4plus.pth` stores weights under `params_ema`, and I did not confirm that the pinned `.pth` loader unwraps that key. If `-M upscale` reports a load error, use 4x-UltraSharp (check the license first) and set `"esrgan_model": "upscale/4x-UltraSharp.safetensors"`.

Install steps:

```sh
mkdir -p "$SD_HOME/models/images/upscale"
curl -L -o "$SD_HOME/models/images/upscale/RealESRGAN_x4plus.pth" \
  https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
echo "4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1  $SD_HOME/models/images/upscale/RealESRGAN_x4plus.pth" | sha256sum -c
cp examples/runtime/images/upscale.json "$LOCAL_IMAGE_CONFIG_DIR/upscale.json"   # then edit; review first
```

With no model installed, Studio still works. `/models` reports `upscale.esrgan: false`, the **Quick upscale** button is hidden and `POST /upscale` is refused, and **Refine** passes the draft straight to img2img with a Lanczos resize.

## Configuration: `upscale.json`

The file is `$LOCAL_IMAGE_CONFIG_DIR/upscale.json`, with an example in `examples/runtime/images/upscale.json`. Any invalid field silently keeps its default.

| Key | Default | Meaning |
|---|---|---|
| `esrgan_model` | `upscale/RealESRGAN_x4plus.pth` | x4 model, relative to `models/images` |
| `esrgan_scale` | 4 | the model's scale (2 or 4); used for the size check |
| `esrgan_x2_model` | `null` | optional post-refine ×2 model |
| `esrgan_tile_size` | 128 | `--upscale-tile-size` (32–1024) |
| `refine_targets` / `refine_target` | `[768, 1024]` / 1024 | long-edge sizes offered / preselected |
| `refine_strength` | 0.45 | default img2img strength (0.05–0.95) |
| `estimate.*` | see below | time-model constants; `calibrated: false` labels every UI estimate "uncalibrated" |

`cpu-profile.json` also accepts an **opt-in** `"img2img_validated": true` plus `"img2img_dimensions": [[768,768],[1024,1024]]`. Only with those set do refine and photo edits use the repack release binary (`SD_CPU_REPACK_EXPERIMENT=1`). Without them, refine and edits keep the old rule that the base `sd-cli` is used whenever an init image is present.

## Measured versus estimated

The estimate model (Studio `estimate_seconds`) is:

`seconds = fixed_s + count · s_per_mpx_step · MPx^pixel_exponent · sampled_steps (+ vae_encode_s_per_mpx · MPx for img2img)`

The whole result is multiplied by `unoptimized_factor` when the repack profile does not apply. ESRGAN is `esrgan_fixed_s + esrgan_s_per_output_mpx · output MPx`.

| Action (Q8, 8 steps) | Number | Status |
|---|---:|---|
| 256 draft | 20.07 s | **measured** (Studio request, repack on; BEFORE-AFTER.md) |
| 1024 native final | 526.61 s | **measured** (same) |
| Q4 256 / 1024 | 26.36 / 615.04 s | **measured** |
| repack off/on, Q8 1024 txt2img | 775.52 / 488.06 s (×1.59) | **measured** CLI pair, source of `unoptimized_factor` |
| 256 draft ×4 variations | ~44 s | estimate (one load, four samplings) |
| 512 / 768 native | ~76 / ~229 s | estimate (two-point fit, exponent 1.5) |
| Quick upscale 256→1024 | ~24 s | **placeholder** |
| Refine → 768, strength 0.45, base binary | ~201 s (+24 s ESRGAN) | **placeholder** |
| Refine → 1024, strength 0.45, base binary | ~445 s (+24 s ESRGAN) | **placeholder** |
| Refine → 1024 with img2img repack validated | ~280 s (+24 s) | **placeholder** |

`fixed_s` = 12 and the pixel exponents are a fit through the two measured anchors and nothing else. The fixed load/encode share, the VAE share, the img2img cost, ESRGAN speed and the 1.59 factor applied to img2img are all unmeasured.

**Expected outcome to confirm.** On the base binary, refine to 1024 (~470 s) may barely beat the native 1024 final (527 s). The clear win is refine to **768**, or refine to 1024 *after* img2img is validated on the repack binary. Decide the default `refine_target` from the calibration below.

## Calibration and quality checks for Codex

Take `inference.lock` through the normal Studio lease, use Q8, one fixed synthetic prompt (for example the BEFORE-AFTER bicycle scene), seed 42, the 1000% quota unit and no other jobs. Record wall time from POST to `done`.

1. **Phase timing anchor.** Run one 256 draft directly with `sd-cli … -v` and note the load, text-encode, sampling and VAE-decode timestamps. Set `estimate.q8.fixed_s` to load + text-encode.
2. **Time per action**, three runs each, median:
   - txt2img 256, 768 and 1024 (refit `s_per_mpx_step` and `pixel_exponent` from all three)
   - 256 with `variations` 4 (confirms the batch shares one load)
   - Quick upscale 256→1024 (set `esrgan_fixed_s` and `esrgan_s_per_output_mpx`)
   - Refine 256→768 and 256→1024 at strength 0.45, base binary, both with ESRGAN and with the plain-resize fallback (temporarily rename the model)
   - If the repack binary is to be used for img2img: the same refine runs with `img2img_validated` set, then set `unoptimized_factor` from the img2img off/on ratio.

   Write the fitted values into `upscale.json` → `estimate`, set `"calibrated": true`, and put the raw timings in `benchmarks/results/` (new `studio-upscale.json`).
3. **Batch determinism.** Variation *i* of a `-b 4 --seed 42` draft must match a single draft with `--seed 42+i` (pixel-identical, or a PSNR above 40 dB if threading makes it nondeterministic). Do this on the repack binary, which has only been validated for single images.
4. **ESRGAN loading.** `sd-cli -M upscale -i draft.png --upscale-model …RealESRGAN_x4plus.pth -o x4.png` must print no `upscale failed`, and `x4.png` must be 1024×1024. Repeat with 4x-UltraSharp if needed.
5. **Composition check (refine keeps the layout).** For 3 prompts × 2 seeds, compare the draft with its refine at 768 and 1024, strength 0.45 (also 0.3 and 0.6 for one prompt). Pass: the same subjects in the same places, the same horizon and pose, and colours within one shade. Automated aid: downscale the refine to 256 with Lanczos and compute SSIM against the draft; expect ≥ 0.6 at 0.45 (calibrate the threshold on the first pair you approve by eye). Also check that the refine adds real detail over Quick upscale by looking at both at 100% crop.
6. **RAM peak.** Sample `VmHWM`/cgroup `memory.peak` for: 256×4 variations, Quick upscale (ESRGAN tile 128), refine 1024 (base binary and, if enabled, repack). Each must stay under the Studio unit's `MemoryHigh` 21 GiB with headroom (the earlier native 1024 Q8 peak was 10.43 GiB RSS). If ESRGAN peaks high, lower `esrgan_tile_size`.
7. **Edit presets** (see below): one portrait photo at Light touch / Restyle / Reimagine. Light touch should keep the face recognisable. Record times as img2img estimates.

## Editing

Photo edits are img2img. `--strength` sets how far back toward noise the photo is pushed before Z-Image Turbo denoises it again under the prompt. At strength *s* with 8 steps, the sampler re-runs `int(8·s)+1` steps starting from a partly noised copy of the photo. Low strength leaves the coarse structure (layout, face geometry, identity) in place and changes mainly texture, lighting and colour. High strength destroys that structure, so the model redraws it from the prompt. Faces change first because identity sits in fine detail that even moderate noise wipes out. The old default of 0.65 re-ran 6 of 8 steps, which is enough to redraw a face.

Z-Image Turbo is a **text-to-image** model. It cannot follow instructions like "make the sky orange" or "remove the hat". The prompt should describe the whole desired picture, not the change. Instruction-style editing would need a separate edit model (for example Qwen-Image-Edit or Flux Kontext), which this Studio does not use.

The edit control is now a preset selector, with the slider moved behind an **advanced** toggle:

| Preset | Strength | Sampled steps (of 8) | Use |
|---|---:|---:|---|
| **Light touch** (default) | 0.30 | 3 | keep faces and composition, adjust the look |
| Restyle | 0.55 | 5 | new style or material, layout mostly kept |
| Reimagine | 0.80 | 7 | loose reinterpretation of the photo |
| advanced slider | 0.05–0.95 | | custom strength (API: `edit_preset: "custom"`, `strength`) |

The API also still accepts a bare `strength` without a preset, as old clients sent it.

**Mask (inpainting).** The pinned tree supports `--mask` for any model. The mask becomes a latent denoise mask: after every step, the denoised latent is blended back to the original wherever the mask is black. After choosing a photo, an optional mask upload appears. **White = change, black = keep**, and the mask should have the same aspect ratio as the output; it is resized with nearest to the generation size. Studio passes `--mask` only together with a photo. Follow-ups: an in-browser brush instead of uploading a file, and a visual check that seams at the mask edge are acceptable at Light touch / Restyle. The mask is applied at latent resolution (1/8), so edges are coarse.

## Integration steps for Codex

1. Review and merge branch `claude/studio-upscale`. No runtime, systemd or `~/.config` file was touched.
2. Download and verify the ESRGAN model (above) into `$SD_HOME/models/images/upscale/`.
3. Copy `examples/runtime/images/upscale.json` to `$LOCAL_IMAGE_CONFIG_DIR/upscale.json` (fresh bootstrap stages it automatically).
4. Deploy the new `studio` the way the runtime is normally deployed, and restart `prompt-studio-app.service`.
5. Run checks 1–7 above. Write the constants and `"calibrated": true`. Pick the default `refine_target` (768 unless 1024 is clearly worth it). Optionally validate img2img on the repack binary and add `img2img_validated` / `img2img_dimensions` to `cpu-profile.json`.
6. Record the results in `docs/BEFORE-AFTER.md` and in `benchmarks/results/`.
