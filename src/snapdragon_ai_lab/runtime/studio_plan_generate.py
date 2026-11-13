"""Image Studio planners for Z-Image: generate, quick upscale and refine. Installed CPU/GPU profile
checks, presets, upscale settings, time estimates and the sd-cli argument lists.

Every function takes the loaded Studio's context first (studio_context.py) and reads that Studio's
paths, settings, job state and collaborators as `ctx.NAME`; nothing here is per-process state.
"""
import hashlib
import json
import os
import time
import uuid
from pathlib import Path

# What `studio` binds to its context and exposes under the same names (StudioContext.bind).
FUNCTIONS = (
    'optimized_gpu_profile', 'optimized_cpu_profile', 'image_presets', 'models_relative_file', 'valid_edge',
    'upscale_settings', 'edit_strength', 'sampled_steps', 'estimate_seconds', 'build_generation_args',
    'build_upscale_args', 'runtime_tuning', 'action_estimates', 'plan_generation', 'chosen_image',
    'plan_quick_upscale', 'refine_source', 'plan_refine',)


def optimized_gpu_profile(ctx, model, width, height):
    path = ctx.PROJECT / 'gpu-profile.json'
    if not path.is_file():
        return None
    try:
        profile = json.loads(path.read_text())
        if not isinstance(profile, dict):
            raise ValueError('GPU profile is incomplete; use CPU')
        if not profile.get('enabled'):
            return None
        if not profile.get('validated'):
            raise ValueError('GPU profile has not passed validation; use CPU')
        limits = profile.get('models', {}).get(model)
        if limits is None:
            raise ValueError('GPU preview is not validated for this model; use CPU')
        if width > int(limits['max_width']) or height > int(limits['max_height']):
            raise ValueError(f"GPU preview is validated up to {limits['max_width']}×{limits['max_height']}; use CPU for larger images")
        binary = Path(profile['binary']).resolve()
        if not binary.is_relative_to((ctx.STACK / 'releases').resolve()) or not binary.is_file() or not os.access(binary, os.X_OK):
            raise ValueError('Installed GPU profile is unavailable; use CPU')
        expected = profile['sha256']
        stat = binary.stat()
        cache_key = (str(binary), stat.st_size, stat.st_mtime_ns, expected)
        if cache_key not in ctx.GPU_BINARY_CHECKS:
            digest = hashlib.sha256()
            with binary.open('rb') as stream:
                while data := stream.read(1024 * 1024): digest.update(data)
            if digest.hexdigest() != expected:
                raise ValueError('Installed GPU binary failed its checksum; use CPU')
            ctx.GPU_BINARY_CHECKS[cache_key] = True
        environment = profile.get('environment', {})
        if not isinstance(environment, dict) or any(not k.startswith('GGML_') or not isinstance(v, str) for k, v in environment.items()):
            raise ValueError('GPU profile environment is invalid; use CPU')
        return profile
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError('GPU profile is incomplete; use CPU') from error


def optimized_cpu_profile(ctx, model, width, height, steps, has_init=False):
    """Select only explicitly validated text-to-image cases from an installed release.

    Image-to-image (refine/edit) is excluded unless the profile separately opts in
    with "img2img_validated": true and lists the checked "img2img_dimensions".
    """
    if model not in ('q4', 'q8') or steps != 8:
        return None
    path = ctx.PROJECT / 'cpu-profile.json'
    if not path.is_file():
        return None
    try:
        profile = json.loads(path.read_text())
        if model not in profile.get('models', ['q4']):
            return None
        if not profile.get('enabled'):
            return None
        if not profile.get('validated'):
            raise ValueError('CPU optimization has not passed validation')
        if has_init:
            if profile.get('img2img_validated') is not True or [width, height] not in profile.get('img2img_dimensions', []):
                return None
        elif [width, height] not in profile.get('dimensions', []):
            return None
        binary = Path(profile['binary']).resolve()
        if not binary.is_relative_to((ctx.STACK / 'releases').resolve()) or not binary.is_file() or not os.access(binary, os.X_OK):
            raise ValueError('Installed CPU optimization is unavailable')
        stat = binary.stat()
        key = (str(binary), stat.st_size, stat.st_mtime_ns, profile['sha256'])
        if key not in ctx.GPU_BINARY_CHECKS:
            digest = hashlib.sha256()
            with binary.open('rb') as stream:
                while data := stream.read(1024 * 1024): digest.update(data)
            if digest.hexdigest() != profile['sha256']:
                raise ValueError('Installed CPU binary failed its checksum')
            ctx.GPU_BINARY_CHECKS[key] = True
        return profile
    except json.JSONDecodeError as error:
        if model == 'q8': return None
        raise ValueError('CPU optimization profile is incomplete') from error
    except (OSError, KeyError, TypeError, AttributeError) as error:
        raise ValueError('CPU optimization profile is incomplete') from error


def image_presets(ctx):
    defaults = {'draft': {'model': 'q4', 'width': 256, 'height': 256, 'steps': 8},
                'final': {'model': 'q4', 'width': 1024, 'height': 1024, 'steps': 8},
                # FLUX.2 klein 512 draft ("Draft 512 (klein)"); shown only when the klein files exist.
                'draft512': {'model': 'klein-q8', 'width': 512, 'height': 512, 'steps': ctx.EDIT_STEPS}}
    try:
        configured = json.loads((ctx.PROJECT / 'presets.json').read_text())
        for name in ('draft', 'final', 'draft512'):
            value = configured.get(name, {})
            w, h, steps = (value.get(k) for k in ('width', 'height', 'steps'))
            if name == 'draft512':
                valid = (value.get('model') in ctx.KLEIN_MODELS and all(type(v) is int for v in (w, h, steps))
                         and ctx.KLEIN_T2I_SIZE_MIN <= w <= ctx.KLEIN_T2I_SIZE_MAX and ctx.KLEIN_T2I_SIZE_MIN <= h <= ctx.KLEIN_T2I_SIZE_MAX
                         and w % 16 == h % 16 == 0 and steps == ctx.EDIT_STEPS)
            else:
                valid = (value.get('model') in ('q4', 'q8') and all(type(v) is int for v in (w, h, steps))
                         and 256 <= w <= 1536 and 256 <= h <= 1536 and w % 64 == h % 64 == 0
                         and w * h <= 1536 * 1024 and 1 <= steps <= 50)
            if valid:
                defaults[name] = {key: value[key] for key in ('model', 'width', 'height', 'steps')}
                seconds = value.get('expected_seconds')
                if type(seconds) in (int, float) and 0 < seconds <= 14400:
                    defaults[name]['expected_seconds'] = seconds
    except (OSError, ValueError, AttributeError, TypeError):
        pass
    return defaults


def models_relative_file(ctx, value):
    """An upscaler path relative to (or inside) the image model directory, or None."""
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = ctx.MODELS / path
    try:
        resolved = path.resolve()
    except OSError:
        return None
    if (not resolved.is_relative_to(ctx.MODELS.resolve()) or resolved.suffix.lower() not in ctx.UPSCALE_SUFFIXES
            or not resolved.is_file()):
        return None
    return resolved


def valid_edge(ctx, value):
    return type(value) is int and 256 <= value <= 1536 and value % 64 == 0


def upscale_settings(ctx):
    """Read PROJECT/upscale.json; any invalid field keeps its default."""
    settings = {key: (list(value) if isinstance(value, list) else value) for key, value in ctx.UPSCALE_DEFAULTS.items()}
    estimate = json.loads(json.dumps(ctx.ESTIMATE_DEFAULTS))
    try:
        configured = json.loads((ctx.PROJECT / 'upscale.json').read_text())
        if not isinstance(configured, dict):
            configured = {}
    except (OSError, ValueError):
        configured = {}
    for key in ('esrgan_model', 'esrgan_x2_model'):
        if key in configured and (configured[key] is None or isinstance(configured[key], str)):
            settings[key] = configured[key]
    if configured.get('esrgan_scale') in (2, 4):
        settings['esrgan_scale'] = configured['esrgan_scale']
    tile = configured.get('esrgan_tile_size')
    if type(tile) is int and 32 <= tile <= 1024:
        settings['esrgan_tile_size'] = tile
    targets = configured.get('refine_targets')
    if isinstance(targets, list) and targets and all(ctx.valid_edge(t) for t in targets):
        settings['refine_targets'] = sorted(set(targets))
    target = configured.get('refine_target')
    if ctx.valid_edge(target):
        settings['refine_target'] = target
    if settings['refine_target'] not in settings['refine_targets']:
        settings['refine_targets'] = sorted(set(settings['refine_targets'] + [settings['refine_target']]))
    strength = configured.get('refine_strength')
    if type(strength) in (int, float) and 0.05 <= strength <= 0.95:
        settings['refine_strength'] = float(strength)
    supplied = configured.get('estimate')
    if isinstance(supplied, dict):
        for key, value in supplied.items():
            if key == 'calibrated' and isinstance(value, bool):
                estimate[key] = value
            elif key in ('q4', 'q8') and isinstance(value, dict):
                for name in ('fixed_s', 's_per_mpx_step', 'pixel_exponent'):
                    number = value.get(name)
                    if type(number) in (int, float) and 0 <= number <= 10000:
                        estimate[key][name] = float(number)
            elif key in estimate and type(value) in (int, float) and type(estimate[key]) is float and 0 <= value <= 10000:
                estimate[key] = float(value)
    settings['estimate'] = estimate
    settings['esrgan_path'] = ctx.models_relative_file(settings['esrgan_model'])
    settings['esrgan_x2_path'] = ctx.models_relative_file(settings['esrgan_x2_model'])
    return settings


def edit_strength(ctx, body):
    """Map the photo-edit preset (or the advanced slider value) to img2img strength."""
    preset = body.get('edit_preset')
    if preset in ctx.EDIT_PRESETS:
        return ctx.EDIT_PRESETS[preset]
    if preset not in (None, '', 'custom'):
        raise ValueError('Choose Light touch, Restyle, Reimagine or a custom strength')
    value = body.get('strength')
    if value is None or value == '':
        return ctx.EDIT_PRESETS[ctx.DEFAULT_EDIT_PRESET]
    if type(value) not in (int, float) and not isinstance(value, str):
        raise ValueError('Edit strength must be a number between 0.05 and 0.95')
    try:
        value = float(value)
    except ValueError:
        raise ValueError('Edit strength must be a number between 0.05 and 0.95') from None
    if not 0.05 <= value <= 0.95:
        raise ValueError('Edit strength must be a number between 0.05 and 0.95')
    return value


def sampled_steps(ctx, steps, strength=None):
    """Steps sd.cpp actually samples (pinned src/pipeline/image.cpp img2img path)."""
    if strength is None or strength >= 1:
        return steps
    t_enc = int(steps * strength)
    if t_enc == steps:
        t_enc -= 1
    return t_enc + 1


def estimate_seconds(ctx, settings, kind, model='q8', width=256, height=256, steps=8, strength=None,
                     count=1, optimized=True, input_size=None, scale=4):
    """Rough wall time for one action; placeholders until calibrated."""
    e = settings['estimate']
    if kind == 'esrgan':
        w, h = input_size or (width, height)
        return round(e['esrgan_fixed_s'] + e['esrgan_s_per_output_mpx'] * (w * scale) * (h * scale) / 1e6, 1)
    m = e.get(model) or e['q8']
    mpx = width * height / 1e6
    per_image = m['s_per_mpx_step'] * mpx ** m['pixel_exponent'] * ctx.sampled_steps(steps, strength)
    if kind == 'img2img':
        per_image += e['vae_encode_s_per_mpx'] * mpx
    seconds = m['fixed_s'] + per_image * count
    if not optimized:
        seconds *= e['unoptimized_factor']
    return round(seconds, 1)


def build_generation_args(ctx, binary, diffusion, width, height, steps, out, threads, fa, cpu=True,
                          seed=None, gpu_profile=None, cpu_profile=None, init=None, strength=None,
                          mask=None, batch=1, init_filter=None):
    """sd-cli image generation command (prompt is supplied separately via memfd)."""
    args = [str(binary), "--diffusion-model", str(diffusion), "--llm", str(ctx.LLM),
            "--vae", str(ctx.VAE), "--cfg-scale", "1.0",
            "--steps", str(steps),
            "-W", str(width),
            "-H", str(height),
            "-o", str(out), "--threads", str(threads), "--flash-attn", "off,diffusion=" + fa,
            "--vae-tiling", "--vae-tile-size", "512", "--disable-image-metadata"]
    if cpu_profile:
        args += ["--auto-fit", "off"]
    if seed is not None:
        args += ["--seed", str(int(seed))]
    if batch > 1:
        args += ["-b", str(batch)]
    if cpu:
        args += ["--backend", "cpu"]
    else:
        args += ["--backend", "te=cpu,vae=cpu,diffusion=vulkan0"]
        if gpu_profile:
            args += gpu_profile.get('extra_args', [])
    if init is not None:
        args += ["-i", str(init), "--strength", str(strength)]
        if init_filter:
            # Default init resampling in the pinned tree is nearest; ask for a real filter.
            args += ["--image-preprocess", f"target=init,mode=stretch,filter={init_filter}"]
        if mask is not None:
            args += ["--mask", str(mask)]
    return args


def build_upscale_args(ctx, binary, model_path, source, out, threads, tile_size):
    """sd-cli ESRGAN-only run (no diffusion model is loaded)."""
    return [str(binary), "-M", "upscale", "-i", str(source), "--upscale-model", str(model_path),
            "--upscale-tile-size", str(tile_size), "--threads", str(threads), "--backend", "cpu",
            "-o", str(out), "--disable-image-metadata"]


def runtime_tuning(ctx, model):
    # Every klein variant shares the one "klein" tuning block.
    key = 'klein' if model.startswith('klein') else model
    try:
        return json.loads((ctx.PROJECT / 'runtime-settings.json').read_text()).get(key, {})
    except (OSError, ValueError, AttributeError):
        return {}


def action_estimates(ctx, record, settings=None):
    """Per-image action availability and estimated seconds for the UI."""
    settings = settings or ctx.upscale_settings()
    w, h = record['width'], record['height']
    result = {'upscale': None, 'refine': {}, 'post_upscale': settings['esrgan_x2_path'] is not None}
    scale = settings['esrgan_scale']
    if settings['esrgan_path'] is not None and w * scale <= 4096 and h * scale <= 4096:
        result['upscale'] = ctx.estimate_seconds(settings, 'esrgan', input_size=(w, h), scale=scale)
    for target in ([] if record.get('engine') == 'klein' else settings['refine_targets']):
        try:
            tw, th = ctx.refine_size(w, h, target)
        except ValueError:
            continue
        refine_model, refine_steps = ctx.refine_source(record)
        try:
            optimized = ctx.optimized_cpu_profile(refine_model, tw, th, refine_steps, True) is not None
        except ValueError:
            optimized = False
        seconds = ctx.estimate_seconds(settings, 'img2img', refine_model, tw, th, refine_steps,
                                   settings['refine_strength'], optimized=optimized)
        if settings['esrgan_path'] is not None:
            seconds += ctx.estimate_seconds(settings, 'esrgan', input_size=(w, h), scale=scale)
        result['refine'][str(target)] = round(seconds, 1)
    return result


def plan_generation(ctx, body):
    """Validate a text-to-image / photo-edit request and build its engine step."""
    prompt = (body.get("prompt") or "").strip()
    if not prompt:
        raise ctx.StudioError("empty prompt", 200)
    model = body.get("model", "q4")
    if model in ctx.KLEIN_MODELS:
        return ctx.plan_klein_text(body)
    diffusion = ctx.MODEL_FILES.get(model)
    if diffusion is None:
        raise ctx.StudioError("Choose Q4 or Q8, or FLUX.2 klein")
    try:
        width, height = int(body.get("width", 512)), int(body.get("height", 512))
        steps = int(body.get("steps", 8))
        if not (256 <= width <= 1536 and 256 <= height <= 1536 and width % 64 == height % 64 == 0 and width * height <= 1536 * 1024 and 1 <= steps <= 50):
            raise ValueError()
    except (TypeError, ValueError):
        raise ctx.StudioError("Use dimensions 256–1536 in multiples of 64, at most 1536×1024 pixels, and 1–50 steps") from None
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
    for f in (diffusion, ctx.LLM, ctx.VAE):
        if not f.exists():
            raise ctx.StudioError(f"missing model {f.name}", 200)

    cpu = bool(body.get('cpu', True))
    tuning = ctx.runtime_tuning(model)
    fa = tuning.get('flash_attention', 'off') if cpu else 'on'
    threads = int(tuning.get('threads', 6))
    init = mask = strength = None
    if body.get("init"):
        ip = Path(body["init"])
        if ip.exists() and ip.resolve().is_relative_to(ctx.UPLOADS.resolve()):
            init = ip
            try:
                strength = ctx.edit_strength(body)
            except ValueError as error:
                raise ctx.StudioError(str(error)) from None
            if body.get("mask"):
                mp = Path(body["mask"])
                if not (mp.exists() and mp.resolve().is_relative_to(ctx.UPLOADS.resolve())):
                    raise ctx.StudioError("Upload the mask again")
                mask = mp
    gpu_profile = None
    environment = None
    if not cpu:
        try:
            gpu_profile = ctx.optimized_gpu_profile(model, width, height)
        except ValueError as error:
            raise ctx.StudioError(str(error)) from None
        if not gpu_profile:
            raise ctx.StudioError('GPU generation is paused; choose CPU.')
        environment = os.environ.copy()
        environment.update(gpu_profile.get('environment', {}))
    cpu_profile = None
    if cpu:
        try:
            # The photo-edit path keeps its original rule: no repack profile for init images.
            cpu_profile = ctx.optimized_cpu_profile(model, width, height, steps, init is not None)
        except ValueError as error:
            raise ctx.StudioError(str(error)) from None
        if cpu_profile:
            environment = os.environ.copy()
            environment['SD_CPU_REPACK_EXPERIMENT'] = '1'
    binary = cpu_profile['binary'] if cpu_profile else gpu_profile['binary'] if gpu_profile else ctx.SDCLI
    stem = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    if variations == 1:
        out = ctx.OUT / f"{stem}.png"
        outputs = [out]
        seed_arg = seed
    else:
        # sd-cli writes stem-0.png … stem-(n-1).png and uses seed+i for image i.
        out = ctx.OUT / f"{stem}-%d.png"
        outputs = [ctx.OUT / f"{stem}-{i}.png" for i in range(variations)]
        seed_arg = ctx.SDCPP_DEFAULT_SEED if seed is None else seed
    args = ctx.build_generation_args(binary, diffusion, width, height, steps, out, threads, fa, cpu,
                                 seed=seed_arg, gpu_profile=gpu_profile, cpu_profile=cpu_profile,
                                 init=init, strength=strength, mask=mask, batch=variations)
    base_seed = ctx.SDCPP_DEFAULT_SEED if seed is None else seed
    results = [{'path': str(p), 'prompt': prompt, 'seed': base_seed + i, 'model': model,
                'width': width, 'height': height, 'steps': steps,
                'kind': 'edit' if init else 'image'}
               for i, p in enumerate(outputs)]
    return {'steps': [{'args': args, 'outputs': outputs, 'label': 'generating'}],
            'environment': environment, 'prompt': prompt, 'results': results,
            'model': model, 'device': 'cpu' if cpu else 'gpu', 'scratch': []}


def chosen_image(ctx, body):
    with ctx.LOCK:
        record = ctx.IMAGES.get(body.get('image')) if isinstance(body.get('image'), str) else None
    if record is None or not Path(record['path']).is_file():
        raise ctx.StudioError('That image is no longer available here; generate it again')
    return record


def plan_quick_upscale(ctx, body):
    """ESRGAN-only enlargement of a chosen image (no diffusion pass)."""
    record = ctx.chosen_image(body)
    settings = ctx.upscale_settings()
    if settings['esrgan_path'] is None:
        raise ctx.StudioError('Quick upscale needs an ESRGAN model; see docs/STUDIO-UPSCALE.md')
    scale = settings['esrgan_scale']
    w, h = record['width'] * scale, record['height'] * scale
    if w > 4096 or h > 4096:
        raise ctx.StudioError('Image is too large for a ×%d upscale' % scale)
    source = Path(record['path'])
    out = ctx.OUT / f"{source.stem}-x{scale}.png"
    if out.exists():
        out = ctx.OUT / f"{source.stem}-x{scale}-{uuid.uuid4().hex[:6]}.png"
    threads = int(ctx.runtime_tuning(record['model']).get('threads', 6))
    args = ctx.build_upscale_args(ctx.SDCLI, settings['esrgan_path'], source, out, threads, settings['esrgan_tile_size'])
    result = dict(record, path=str(out), width=w, height=h, kind='upscale')
    return {'steps': [{'args': args, 'outputs': [out], 'label': 'ESRGAN upscale', 'prompt': False,
                       'expect_size': (w, h)}],
            'environment': None, 'prompt': None, 'results': [result],
            'model': record['model'], 'device': 'cpu', 'scratch': []}


def refine_source(ctx, record):
    """Z-Image model and step count used by refine. A klein text-to-image result refines with
    Z-Image Turbo at its 8 steps (klein's 4-step count would leave 2 sampled steps at strength 0.45)."""
    if record.get('engine') == 'klein-t2i':
        return record['model'][len('klein-'):], 8
    return record['model'], record['steps']


def plan_refine(ctx, body):
    """Enlarge a chosen image, then img2img it at the target size with its prompt and seed."""
    record = ctx.chosen_image(body)
    if record.get('engine') == 'klein':
        raise ctx.StudioError('Refine uses the Z-Image model; use Quick upscale on an edited image')
    settings = ctx.upscale_settings()
    target = body.get('target', settings['refine_target'])
    if target not in settings['refine_targets']:
        raise ctx.StudioError('Choose a refine size of ' + ' or '.join(map(str, settings['refine_targets'])))
    strength = body.get('strength', settings['refine_strength'])
    if type(strength) not in (int, float) or not 0.05 <= strength <= 0.95:
        raise ctx.StudioError('Refine strength must be between 0.05 and 0.95')
    strength = float(strength)
    try:
        width, height = ctx.refine_size(record['width'], record['height'], target)
    except ValueError as error:
        raise ctx.StudioError(str(error)) from None
    post = bool(body.get('post_upscale'))
    if post and settings['esrgan_x2_path'] is None:
        raise ctx.StudioError('ESRGAN ×2 needs a configured esrgan_x2_model')
    model, sample_steps = ctx.refine_source(record)
    diffusion = ctx.MODEL_FILES[model]
    for f in (diffusion, ctx.LLM, ctx.VAE):
        if not f.exists():
            raise ctx.StudioError(f"missing model {f.name}", 200)
    tuning = ctx.runtime_tuning(model)
    fa = tuning.get('flash_attention', 'off')
    threads = int(tuning.get('threads', 6))
    source = Path(record['path'])
    stem = f"{source.stem}-refine{target}-{uuid.uuid4().hex[:6]}"
    steps, scratch = [], []
    init = source
    if settings['esrgan_path'] is not None:
        scale = settings['esrgan_scale']
        work = ctx.OUT / '_work'
        work.mkdir(parents=True, exist_ok=True)
        init = work / f"{stem}-esrgan.png"
        scratch.append(str(init))
        steps.append({'args': ctx.build_upscale_args(ctx.SDCLI, settings['esrgan_path'], source, init, threads,
                                                 settings['esrgan_tile_size']),
                      'outputs': [init], 'label': 'ESRGAN upscale', 'prompt': False,
                      'expect_size': (record['width'] * scale, record['height'] * scale)})
    try:
        cpu_profile = ctx.optimized_cpu_profile(model, width, height, sample_steps, True)
    except ValueError as error:
        raise ctx.StudioError(str(error)) from None
    environment = None
    if cpu_profile:
        environment = os.environ.copy()
        environment['SD_CPU_REPACK_EXPERIMENT'] = '1'
    binary = cpu_profile['binary'] if cpu_profile else ctx.SDCLI
    out = ctx.OUT / f"{stem}.png"
    # Lanczos covers both cases: the plain-resize fallback (draft -> target)
    # and fitting the ESRGAN output to the target size.
    args = ctx.build_generation_args(binary, diffusion, width, height, sample_steps, out, threads, fa, True,
                                 seed=record['seed'], cpu_profile=cpu_profile, init=init,
                                 strength=strength, init_filter='lanczos')
    steps.append({'args': args, 'outputs': [out], 'label': 'refining', 'prompt': True,
                  'environment': environment, 'expect_size': (width, height)})
    final, final_w, final_h = out, width, height
    if post:
        final = ctx.OUT / f"{stem}-x2.png"
        final_w, final_h = width * 2, height * 2
        steps.append({'args': ctx.build_upscale_args(ctx.SDCLI, settings['esrgan_x2_path'], out, final, threads,
                                                 settings['esrgan_tile_size']),
                      'outputs': [final], 'label': 'ESRGAN ×2', 'prompt': False,
                      'expect_size': (final_w, final_h)})
    result = dict(record, path=str(final), width=final_w, height=final_h, kind='refine')
    return {'steps': steps, 'environment': None, 'prompt': record['prompt'], 'results': [result],
            'model': model, 'device': 'cpu', 'scratch': scratch}
