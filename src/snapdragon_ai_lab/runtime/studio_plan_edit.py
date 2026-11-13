"""Image Studio planners for FLUX.2 klein on sd-cli: instruction edit (whole image, masked, cropped)
and klein text-to-image, with their estimates, readiness and argument lists (docs/STUDIO-EDIT.md).

Every function takes the loaded Studio's context first (studio_context.py) and reads that Studio's
paths, settings, job state and collaborators as `ctx.NAME`; nothing here is per-process state.
"""
import json
import time
import uuid
from pathlib import Path

EDIT_STEPS = 4                   # distilled klein: 4 steps, cfg 1.0 (not the Base model)
# What `studio` binds to its context and exposes under the same names (StudioContext.bind).
FUNCTIONS = (
    'edit_estimate_settings', 'edit_estimate_seconds', 'edit_missing_files', 'klein_info', 'edit_info',
    'prepare_edit_inputs', 'build_edit_args', 'build_klein_t2i_args', 'plan_klein_text', 'plan_edit',)


def edit_estimate_settings(ctx):
    estimate = dict(ctx.EDIT_ESTIMATE_DEFAULTS)
    try:
        configured = json.loads((ctx.PROJECT / 'edit.json').read_text()).get('estimate', {})
    except (OSError, ValueError, AttributeError):
        configured = {}
    if isinstance(configured, dict):
        for key, value in configured.items():
            if key == 'calibrated' and isinstance(value, bool):
                estimate[key] = value
            elif key in estimate and key != 'calibrated' and type(value) in (int, float) and 0 <= value <= 10000:
                estimate[key] = float(value)
    return estimate


def edit_estimate_seconds(ctx, width, height, steps=EDIT_STEPS, masked=False, t2i=False, count=1):
    """Edit (two sequences: reference + target, VAE-encode of the reference) or, with t2i,
    text-to-image (one sequence, no VAE-encode term). Uncalibrated until edit.json says otherwise."""
    e = ctx.edit_estimate_settings()
    mpx = width * height / 1e6
    sampling = e['s_per_mpx_step'] * mpx ** e['pixel_exponent'] * steps
    if t2i:
        sampling *= 0.5
        seconds = e['fixed_s'] + count * sampling
    else:
        seconds = e['fixed_s'] + sampling + e['vae_encode_s_per_mpx'] * mpx
    if masked:
        seconds += e['composite_s_per_mpx'] * mpx
    return round(seconds, 1)


def edit_missing_files(ctx, model='q8'):
    return [f.name for f in (ctx.KLEIN_FILES[model], ctx.KLEIN_LLM, ctx.KLEIN_VAE) if not f.is_file()]


def klein_info(ctx):
    """FLUX.2 klein text-to-image readiness (files present) and estimates for the Generate flow."""
    ready = {f'klein-{key}': not ctx.edit_missing_files(key) for key in ctx.KLEIN_FILES}
    calibrated = ctx.edit_estimate_settings()['calibrated']
    return {'ready': any(ready.values()), 'models': ready, 'steps': ctx.EDIT_STEPS,
            'sizes': [ctx.KLEIN_T2I_SIZE_MIN, ctx.KLEIN_T2I_SIZE_MAX],
            'estimates': {str(size): ctx.edit_estimate_seconds(size, size, t2i=True) for size in (512, 768, 1024)},
            'estimate_label': 'estimate' if calibrated else 'uncalibrated estimate',
            'estimates_calibrated': calibrated}


def edit_info(ctx):
    ready = {key: not ctx.edit_missing_files(key) for key in ctx.KLEIN_FILES}
    return {'ready': any(ready.values()), 'models': ready, 'sizes': list(ctx.EDIT_SIZES),
            'default_size': ctx.DEFAULT_EDIT_SIZE, 'steps': ctx.EDIT_STEPS,
            'estimates': {str(size): ctx.edit_estimate_seconds(size, size) for size in ctx.EDIT_SIZES},
            'estimates_calibrated': ctx.edit_estimate_settings()['calibrated']}


def prepare_edit_inputs(ctx, source, mask, work_size, ref_out):
    """Write the working-size reference PNG the engine and the composite both use."""
    width, height, rgb = ctx.read_png(source)
    if width * height > ctx.MAX_MASK_PIXELS:
        raise ValueError('Masked edits take images up to about 4 megapixels; downscale it first')
    ref = ctx.resize_pixels(rgb, width, height, work_size[0], work_size[1], 3)
    ctx.write_png(ref_out, work_size[0], work_size[1], ref)
    mw, mh, mgray = ctx.read_png(mask, gray=True)
    return ref, ctx.resize_pixels(mgray, mw, mh, work_size[0], work_size[1], 1)


def build_edit_args(ctx, binary, diffusion, ref, out, width, height, steps, threads, fa='off', seed=None):
    """sd-cli FLUX.2 klein edit command (instruction is supplied separately via memfd).

    Distilled klein: 4 steps, cfg 1.0, euler (pinned docs/flux2.md). The flux2
    ref-image preset is auto-detected from the model weights; the reference is
    VAE-encoded and fed to the DiT, the Qwen3 encoder sees text only.
    """
    args = [str(binary), "--diffusion-model", str(diffusion), "--llm", str(ctx.KLEIN_LLM), "--vae", str(ctx.KLEIN_VAE),
            "-r", str(ref), "--cfg-scale", "1.0", "--sampling-method", "euler", "--steps", str(steps),
            "-W", str(width), "-H", str(height), "-o", str(out), "--threads", str(threads),
            "--flash-attn", "off,diffusion=" + fa, "--vae-tiling", "--vae-tile-size", "512",
            "--disable-image-metadata", "--backend", "cpu"]
    if seed is not None:
        args += ["--seed", str(int(seed))]
    return args


def build_klein_t2i_args(ctx, binary, diffusion, width, height, out, threads, fa='off', seed=None, batch=1):
    """sd-cli FLUX.2 klein text-to-image: build_edit_args without the reference image (-r).

    Same distilled settings (4 steps, cfg 1.0, euler). The prompt is supplied separately via memfd.
    """
    args = [str(binary), "--diffusion-model", str(diffusion), "--llm", str(ctx.KLEIN_LLM), "--vae", str(ctx.KLEIN_VAE),
            "--cfg-scale", "1.0", "--sampling-method", "euler", "--steps", str(ctx.EDIT_STEPS),
            "-W", str(width), "-H", str(height), "-o", str(out), "--threads", str(threads),
            "--flash-attn", "off,diffusion=" + fa, "--vae-tiling", "--vae-tile-size", "512",
            "--disable-image-metadata", "--backend", "cpu"]
    if seed is not None:
        args += ["--seed", str(int(seed))]
    if batch > 1:
        args += ["-b", str(batch)]
    return args


def plan_klein_text(ctx, body):
    """FLUX.2 klein 4B text-to-image from the Generate flow (base sd-cli, 4 steps, no photo)."""
    prompt = body.get("prompt")
    prompt = prompt.strip() if isinstance(prompt, str) else ""
    if not prompt:
        raise ctx.StudioError("empty prompt", 200)
    model = body.get("model")
    if model not in ctx.KLEIN_MODELS:
        raise ctx.StudioError("Choose FLUX.2 klein Q8 or Q4")
    variant = model[len('klein-'):]
    width, height = body.get("width", 512), body.get("height", 512)
    if not all(type(v) is int and ctx.KLEIN_T2I_SIZE_MIN <= v <= ctx.KLEIN_T2I_SIZE_MAX and v % 16 == 0 for v in (width, height)):
        raise ctx.StudioError("FLUX.2 klein sizes are multiples of 16 from 256 to 1024")
    # The step count is fixed at 4 for the distilled model; a client-sent value is ignored.
    variations = body.get("variations", 1)
    if type(variations) is not int or not 1 <= variations <= ctx.MAX_VARIATIONS:
        raise ctx.StudioError(f"Variations must be 1–{ctx.MAX_VARIATIONS}")
    seed = body.get("seed")
    if seed is not None:
        try:
            if isinstance(seed, bool) or int(seed) != float(seed):
                raise ValueError()
            seed = int(seed)
            if not 0 <= seed <= 2**62:
                raise ValueError()
        except (TypeError, ValueError, OverflowError):
            raise ctx.StudioError("Seed must be a whole number of 0 or more") from None
    if body.get("init"):
        raise ctx.StudioError("FLUX.2 klein text-to-image does not take a photo; use the Edit panel for photos")
    if not bool(body.get("cpu", True)):
        raise ctx.StudioError("FLUX.2 klein runs on CPU only; keep CPU selected")
    missing = ctx.edit_missing_files(variant)
    if missing:
        raise ctx.StudioError('Missing klein model files: ' + ', '.join(missing) + ' (see docs/STUDIO-EDIT.md)', 200)
    tuning = ctx.runtime_tuning(model)
    threads = int(tuning.get('threads', 6))
    fa = tuning.get('flash_attention', 'off')
    stem = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    base_seed = ctx.SDCPP_DEFAULT_SEED if seed is None else seed
    if variations == 1:
        out = ctx.OUT / f"{stem}.png"
        outputs = [out]
        seed_arg = seed
    else:
        out = ctx.OUT / f"{stem}-%d.png"
        outputs = [ctx.OUT / f"{stem}-{i}.png" for i in range(variations)]
        seed_arg = base_seed
    args = ctx.build_klein_t2i_args(ctx.SDCLI, ctx.KLEIN_FILES[variant], width, height, out, threads, fa,
                                seed=seed_arg, batch=variations)
    results = [{'path': str(p), 'prompt': prompt, 'seed': base_seed + i, 'model': model,
                'width': width, 'height': height, 'steps': ctx.EDIT_STEPS, 'kind': 'image', 'engine': 'klein-t2i',
                'backend': 'sd.cpp'}
               for i, p in enumerate(outputs)]
    plan = {'steps': [{'args': args, 'outputs': outputs, 'label': 'generating', 'prompt': True}],
            'environment': None, 'prompt': prompt, 'results': results,
            'model': model, 'device': 'cpu', 'scratch': []}
    setting = ctx.engine_setting()
    engine, reason = ctx.choose_t2i_engine(setting, model, width, height, ctx.oxi_missing())
    plan['engine'] = {'setting': setting, 'engine': engine, 'reason': reason}
    if engine == 'oxi':
        spec = {'setting': setting, 'width': width, 'height': height, 'outputs': outputs, 'results': results,
                'seeds': [base_seed + i for i in range(variations)], 'sd_args': args, 'threads': threads, 'fa': fa}
        for record in results:
            record['backend'] = 'oxi'
        plan['steps'] = [{'run': lambda job, text: ctx.run_oxi_t2i(job, text, spec), 'outputs': outputs,
                          'label': 'generating', 'prompt': True, 'expect_size': (width, height)}]
    return plan


def plan_edit(ctx, body):
    """Validate an instruction-edit request and build its engine step(s).

    With a mask (default) only the masked area is edited: the native photo is cropped around
    the mask, the crop is edited at its own working size, and the result is pasted back at the
    photo's full resolution through the feathered native mask. "crop": false keeps the older
    whole-image working copy, which is also what an unmasked edit uses.
    """
    instruction = (body.get('instruction') or '').strip() if isinstance(body.get('instruction'), str) else ''
    if not instruction:
        raise ctx.StudioError('Type what to change first', 200)
    if len(instruction) > ctx.MAX_INSTRUCTION:
        raise ctx.StudioError(f'Instruction is longer than {ctx.MAX_INSTRUCTION} characters')
    crop = body.get('crop', True)
    if type(crop) is not bool:
        raise ctx.StudioError('crop must be true or false')
    use_crop = crop and bool(body.get('mask'))
    size = body.get('size', ctx.DEFAULT_EDIT_SIZE)
    if type(size) is not int or size not in ctx.EDIT_SIZES:
        raise ctx.StudioError('Edit size must be ' + ' or '.join(map(str, ctx.EDIT_SIZES)))
    model = body.get('model', 'q8')
    if model not in ctx.KLEIN_FILES:
        raise ctx.StudioError('Choose Q8 or Q4')
    steps = body.get('steps', ctx.EDIT_STEPS)
    if type(steps) is not int or not 1 <= steps <= 8:
        raise ctx.StudioError('Edit steps must be 1-8 (the distilled model is made for 4)')
    seed = body.get('seed')
    if seed is not None:
        try:
            if isinstance(seed, bool) or int(seed) != float(seed):
                raise ValueError()
            seed = int(seed)
            if not 0 <= seed <= 2**62:
                raise ValueError()
        except (TypeError, ValueError, OverflowError):
            raise ctx.StudioError('Seed must be a whole number of 0 or more') from None
    missing = ctx.edit_missing_files(model)
    if missing:
        raise ctx.StudioError('Missing edit model files: ' + ', '.join(missing) + ' (see docs/STUDIO-EDIT.md)', 200)
    source = Path(body['init']) if isinstance(body.get('init'), str) and body.get('init') else None
    if source is None or not source.is_file() or not source.resolve().is_relative_to(ctx.UPLOADS.resolve()):
        raise ctx.StudioError('Upload the photo to edit first')
    dims = ctx.image_size(source)
    if dims is None:
        raise ctx.StudioError('The photo must be a PNG or JPEG')
    if min(dims) < 64:
        raise ctx.StudioError('The photo is too small to edit')
    mask = None
    if body.get('mask'):
        mask = Path(body['mask']) if isinstance(body['mask'], str) else None
        if mask is None or not mask.is_file() or not mask.resolve().is_relative_to(ctx.UPLOADS.resolve()):
            raise ctx.StudioError('Upload the mask again')
        if ctx.png_size(source) is None or ctx.png_size(mask) is None:
            raise ctx.StudioError('A mask needs the photo and the mask as PNG files (convert JPEG to PNG first)')
        if dims[0] * dims[1] > ctx.MAX_MASK_PIXELS or ctx.png_size(mask)[0] * ctx.png_size(mask)[1] > ctx.MAX_MASK_PIXELS:
            raise ctx.StudioError('Masked edits take images up to about 4 megapixels; downscale it first')
    tuning = ctx.runtime_tuning('klein')
    threads = int(tuning.get('threads', 6))
    fa = tuning.get('flash_attention', 'off')
    base_seed = ctx.SDCPP_DEFAULT_SEED if seed is None else seed
    stem = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}-edit"
    final = ctx.OUT / f"{stem}.png"
    work = ctx.OUT / '_work'
    ref, raw = work / f"{stem}-ref.png", work / f"{stem}-raw.png"
    scratch = []
    state = {}
    crop_info = None
    if mask is None or not use_crop:
        width, height = ctx.edit_work_size(dims[0], dims[1], size)
        step = {'label': 'editing', 'prompt': True, 'expect_size': (width, height)}
        if mask is None:
            step['args'] = ctx.build_edit_args(ctx.SDCLI, ctx.KLEIN_FILES[model], source, final, width, height, steps, threads, fa, seed=base_seed)
            step['outputs'] = [final]
        else:
            scratch += [str(ref), str(raw)]

            def pre():
                work.mkdir(parents=True, exist_ok=True)
                state['ref'], state['mask'] = ctx.prepare_edit_inputs(source, mask, (width, height), ref)

            def post():
                ctx.composite_files(state['ref'], state['mask'], raw, final, (width, height))

            step.update(args=ctx.build_edit_args(ctx.SDCLI, ctx.KLEIN_FILES[model], ref, raw, width, height, steps, threads, fa, seed=base_seed),
                        outputs=[raw], pre=pre, post=post)
        out_width, out_height = width, height
    else:
        # Crop-edit-paste at native resolution (docs/STUDIO-EDIT.md, "Masked edits").
        try:
            mask_gray = ctx.read_mask_at(mask, dims[0], dims[1])
            bbox = ctx.mask_bounding_box(mask_gray, dims[0], dims[1])
            if bbox is None:
                raise ctx.StudioError('The mask has no white area; paint the part to change in white')
            x0, y0, x1, y1, width, height = ctx.crop_box_for_mask(bbox, dims[0], dims[1], size)
        except (ValueError, OSError) as error:
            raise ctx.StudioError(str(error)) from None
        cw, ch = x1 - x0, y1 - y0
        scale = max(width, height) / max(cw, ch)
        scratch += [str(ref), str(raw)]

        def pre():
            work.mkdir(parents=True, exist_ok=True)
            _, _, rgb = ctx.read_png(source)
            crop_rgb = ctx.crop_pixels(rgb, dims[0], x0, y0, cw, ch, 3)
            if (width, height) != (cw, ch):
                crop_rgb = ctx.resize_pixels(crop_rgb, cw, ch, width, height, 3)
            ctx.write_png(ref, width, height, crop_rgb)
            state['rgb'] = rgb

        def post():
            ctx.composite_crop(state['rgb'], dims[0], dims[1], (x0, y0, cw, ch), mask_gray, raw, (width, height), final)

        step = {'label': 'editing', 'prompt': True, 'expect_size': (width, height), 'outputs': [raw],
                'pre': pre, 'post': post,
                'args': ctx.build_edit_args(ctx.SDCLI, ctx.KLEIN_FILES[model], ref, raw, width, height, steps, threads, fa, seed=base_seed)}
        out_width, out_height = dims
        crop_info = {'box': [x0, y0, x1, y1], 'work': [width, height], 'scale': round(scale, 4)}
    result = {'path': str(final), 'prompt': instruction, 'seed': base_seed, 'model': 'klein-' + model,
              'width': out_width, 'height': out_height, 'steps': steps, 'kind': 'edit', 'engine': 'klein'}
    if crop_info is not None:
        result['crop'] = crop_info
    return {'steps': [step], 'environment': None, 'prompt': instruction, 'results': [result],
            'model': 'klein-' + model, 'device': 'cpu', 'scratch': scratch,
            'estimate': ctx.edit_estimate_seconds(width, height, masked=mask is not None)}
