# Studio: Edit (instruction)

Upload a photo, type what to change ("change the jacket colour to dark green"),
press Edit. Engine: FLUX.2 [klein] 4B (Apache-2.0) in stable-diffusion.cpp,
reference-image edit mode (`-r`). Nothing here has been run on the machine yet:
every time below is an uncalibrated estimate until Codex runs the checks at the
end of this file.

## Files

Directory `~/.local/share/ai-stack/models/images/flux2-klein/`.

| File | Size (bytes) | sha256 | Source / license |
|---|---|---|---|
| `flux-2-klein-4b-Q8_0.gguf` | 4300629440 | `0bba6951258ec8f92d51114a8fa13e66828297bfff58a738f52729b3ef66fa28` | leejet/FLUX.2-klein-4B-GGUF, Apache-2.0 |
| `flux-2-klein-4b-Q4_0.gguf` | 2460378560 | `d1023499ef3f2f82ff7c50e6778495195c1b6cc34835741778868428111f9ff4` | same repo, Apache-2.0 (smaller, lower quality) |
| `Qwen3-4B-Q8_0.gguf` | 4280405792 | `eed555233267a33c7e8ee31682762cc7751b3f6d224039086e0e846f05fffa5d` | unsloth/Qwen3-4B-GGUF, Apache-2.0 (text encoder) |
| `flux2-vae.safetensors` | 336211292 | `868fe7b343cc8f3a19dbcfcafbc3d5f888802be3f89bd81b65b3621a066ce8f3` | Comfy-Org/Flux-2-Klein-4B `split_files/vae/`, Apache-2.0 |

sha256 values are the Hugging Face API `lfs.oid` values, checked against the
downloaded files.

* Text encoder: klein 4B uses the original **Qwen3-4B** (sd.cpp docs/flux2.md,
  `--llm qwen_3_4b`). The existing `Qwen3-4B-Instruct-2507-Q4_K_M.gguf` is a
  different fine-tune that the klein model was not trained against, and its
  Q4 quantisation would add error; it is not used. Klein 9B would need Qwen3-8B.
* VAE: the FLUX.2 autoencoder (`flux2_ae.safetensors` in the docs). The
  black-forest-labs/FLUX.2-dev copy is gated; the Comfy-Org klein repo carries
  the ungated `flux2-vae.safetensors` of the same size (336211292 bytes).
* Download recipe (low priority, resumable; verify with `sha256sum`):
  `ionice -c3 nice -n19 curl -L --limit-rate 25M -C - -o FILE https://huggingface.co/<repo>/resolve/main/<path>`.

## Pinned sd.cpp

Pinned tree `master-1053-5a1ba5d` (`~/.local/share/ai-stack/src/stable-diffusion.cpp`).
`docs/flux2.md` has the klein 4B edit example, `docs/edit.md` lists
"Flux.2 [Dev] / Flux.2 [Klein]" with preset `flux2`, and
`src/model/diffusion/model.hpp` defines it (`pass_to_vlm=false`,
`pass_to_dit=true`, `ref_index_mode=increase`, `resize_before_vae=true`). So the
pinned commit supports klein edit mode; no newer commit is needed. The preset is
auto-detected from the weights, so no `--ref-image-args` is passed.

Command line Studio builds (instruction goes through the memfd `--prompt-file`
path of `run_engine`, never argv):

```
sd-cli --diffusion-model flux-2-klein-4b-Q8_0.gguf --llm Qwen3-4B-Q8_0.gguf \
  --vae flux2-vae.safetensors -r INPUT.png --cfg-scale 1.0 --sampling-method euler \
  --steps 4 -W W -H H -o OUT.png --threads N --flash-attn off,diffusion=off \
  --vae-tiling --vae-tile-size 512 --disable-image-metadata --backend cpu --seed S
```

* Distilled klein: 4 steps, cfg 1.0. Do not use the Base model (20 steps, cfg 4).
* `--offload-to-cpu` from the docs is omitted: the whole run is on the CPU backend.
* Output size: long edge 512 (default) or 768, aspect kept, both sides rounded to
  a multiple of 16 (the FLUX.2 latent grid). The reference is VAE-encoded and
  resized by sd.cpp to about the output area.

Not supported / limits in the pinned tree:

* `--mask` is an inpainting option of the init-image path (`-i`); with `-r` edits
  it is not applied. Studio therefore composites the mask itself (below).
* Several references (`-r` repeated) work in sd.cpp but are not exposed in Studio.
* Refine (Z-Image img2img) is disabled for edited images; Quick upscale (ESRGAN)
  still works.

## Studio behaviour

* Panel "Edit (instruction)": photo, optional PNG mask, instruction text, size
  (512 default, 768), model (Q8 default, Q4), seed (blank = 42, sd-cli default).
  Endpoint `POST /edit`; `GET /models` has an `edit` section with readiness and
  per-size estimates.
* Same lease, same job pipeline and the same `run_engine` boundary as the other
  modes; tests replace `run_engine`.
* Mask (white = change, black = keep): see "Masked edits: crop-edit-paste"
  below. Masked edits need the photo and mask as PNG and at most about 4
  megapixels (the PNG reader/resizer is pure Python, no Pillow on this machine).
  Unmasked edits accept PNG or JPEG.
* Time estimates: `fixed_s + s_per_mpx_step * mpx^1.5 * steps + vae_encode * mpx`
  with placeholders in `EDIT_ESTIMATE_DEFAULTS` (Z-Image fit scaled by 4B/6B and
  by two image sequences). Always shown as "uncalibrated estimate" until
  `edit.json` in the Studio config dir sets `"estimate": {"calibrated": true, ...}`
  with measured constants (keys: fixed_s, s_per_mpx_step, pixel_exponent,
  vae_encode_s_per_mpx, composite_s_per_mpx).
* Optional `runtime-settings.json` key `klein` (`threads`, `flash_attention`).

## Masked edits: crop-edit-paste

Nothing here has been run on the machine; the behaviour is covered only by the
stubbed unit tests in `tests/test_studio_crop_edit.py`. Times are uncalibrated
estimates.

Before this change a masked edit downsized the whole photo to a 512/768 working
copy, edited it and composited at that size, so the output lost the original
resolution. Now, when a mask is given (the default), Studio edits only the masked
area at native resolution:

1. The native photo and the mask are read (a mask of a different size is
   resized to the photo). The box around the mask pixels > 0 is found. An empty
   mask is refused.
2. The box is padded by 25% of its larger side on every side (at least 32 px)
   and clamped to the photo.
   Each side is then widened to at least 256 px (centred, shifted inside the photo; capped at
   the largest 16-multiple inside the photo when the photo side is shorter, so a 200 px side
   becomes 192 px and a 240 px side is used whole). If the padded long edge is at most the requested
   size (512 or 768), the crop is edited 1:1 with both sides rounded up to
   multiples of 16 (grown inside the photo, shifted if needed). Otherwise the
   crop is scaled so its long edge equals the size, and the short side lands on
   a multiple of 16. Geometry: `crop_box_for_mask(mask_bbox, image_w, image_h,
   work_size)` in the Studio source.
3. The crop (scaled if needed) is written as the klein reference PNG and edited
   exactly as in the whole-image path (`build_edit_args` with the crop's working
   size).
4. The edited crop is resized back to the native crop size if it was scaled and
   pasted into the **original** photo through the native mask, feathered by a box
   blur of radius `max(2, crop long edge // 128)` applied to the mask inside the
   crop only. Where the feathered mask is 0 the output is byte-identical to the
   photo. The output PNG has the photo's size.

The job image record carries `crop: {box: [x0, y0, x1, y1], work: [w, h], scale}`.
`scale` is the working size divided by the native crop long edge (1.0 for 1:1).

Request field `"crop": false` keeps the previous whole-image behaviour (working
copy, composite at working size). The page sends no such field, so the page uses
the crop path. Unmasked edits are unchanged.

Limits and notes:

* The 4-megapixel limit (`MAX_MASK_PIXELS`) still applies to the whole photo and
  the whole mask, and is checked from the PNG headers before any pixel is read.
* A padded box that cannot be held at a multiple of 16 inside a photo whose side
  is not a multiple of 16 is refused with a message, rather than cutting into the mask.
* The crop size stays at 512 or 768; a 1024 working size is not offered.
* Estimates use the crop's working size (`estimate_seconds` in the `/edit`
  response, `estimate_seconds` field) plus the composite term. They are
  placeholders until `edit.json` is calibrated.
* Checking a result: `python3 -I tools/studio_mask_check.py ORIGINAL.png RESULT.png MASK.png [--work-size 512]`
  prints how many pixels outside the feathered mask differ from the original and
  exits 0 only when that count is 0.

## Calibration and quality checks for Codex

Take `inference.lock` as usual. Run through Studio (or the printed command).

1. Portrait edit: one portrait photo, instruction "change jacket colour to dark
   green", seed 42, 512 then 768, Q8. Judge: is the face and identity kept
   (same person, same expression, same background), is only the jacket changed.
   Repeat with Q4 once. Repeat once with a mask over the jacket to confirm the
   outside is untouched.
2. Time: wall time at 512 and 768 for load, sampling (4 steps) and VAE, and the
   masked-composite overhead. Fill `edit.json` and set `calibrated: true`.
3. RAM: peak RSS (Q8 diffusion 4.3 GB + Qwen3 Q8 4.3 GB + VAE and activations)
   against the memory cap; if close, try Q4 diffusion or a smaller Qwen3 GGUF
   (unsloth Q4_K_M 2.5 GB, checked against quality).
4. If identity is weak, the runner-up options are in
   `image-edit-models-research.md` (LongCat Image Edit, Qwen-Image-Edit-2511).

Tests: `python3 -I -m unittest discover -s tests -p "test_studio_*.py"` (engine
stubbed; no model files, sd-cli or services needed).
