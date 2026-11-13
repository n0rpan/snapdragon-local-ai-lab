"""Image Studio's oxi route for klein text-to-image: the engine setting, per-size gate records,
routing, oxi argument lists, the prompt conditioning (native or sd-cli hand-over) and the run with
its sd.cpp fallback (docs/STUDIO-OXI.md). The constants and their notes are in `studio`.

Every function takes the loaded Studio's context first (studio_context.py) and reads that Studio's
paths, settings, job state and collaborators as `ctx.NAME`; nothing here is per-process state.
"""
import hashlib
import json
import os
import re
import shutil
import tempfile
import uuid
from pathlib import Path
from stat import S_ISREG

OXI_COND_KEEP = 16                       # cached prompt conditionings (~17 MB each at 512 px)
# What `studio` binds to its context and exposes under the same names (StudioContext.bind).
FUNCTIONS = (
    'engine_setting', 'save_engine_setting', 'oxi_threads', 'oxi_prompt_cache_enabled', 'oxi_npu_settings',
    'oxi_text_encoder', 'oxi_te_arith', 'oxi_te_key_args', 'oxi_cache_environment', 'oxi_vae_conv',
    'oxi_missing', 'oxi_binary_sha256', 'oxi_fast_setting', 'oxi_size_gate_passed', 'oxi_gated_sizes',
    'choose_t2i_engine', 'engine_info', 'oxi_estimate_seconds', 'oxi_generate_args', 'oxi_cond_native_args',
    'oxi_cond_put_args', 'handover_stop', 'prune_cond_cache', 'make_conditioning', 'oxi_generate_all',
    'run_oxi_t2i',)


def engine_setting(ctx):
    """The Studio engine choice for klein text-to-image: auto (default), oxi or sdcpp."""
    try:
        value = json.loads((ctx.PROJECT / 'engine.json').read_text()).get('engine')
    except (OSError, ValueError, AttributeError):
        value = None
    return value if value in ctx.ENGINE_CHOICES else 'auto'


def save_engine_setting(ctx, value):
    if value not in ctx.ENGINE_CHOICES:
        raise ctx.StudioError('Engine must be auto, oxi or sdcpp')
    ctx.PROJECT.mkdir(parents=True, exist_ok=True)
    tmp = ctx.PROJECT / f'.engine.json.{os.getpid()}.tmp'
    tmp.write_text(json.dumps({'engine': value}) + '\n')
    os.replace(tmp, ctx.PROJECT / 'engine.json')
    return value


def oxi_threads(ctx):
    try:
        value = json.loads((ctx.PROJECT / 'runtime-settings.json').read_text()).get('oxi', {}).get('threads')
    except (OSError, ValueError, AttributeError):
        value = None
    return value if type(value) is int and 1 <= value <= 64 else ctx.OXI_DEFAULT_THREADS


def oxi_prompt_cache_enabled(ctx):
    """Experimental persistent conditioning is opt-in; malformed/non-boolean values stay off."""
    try:
        value = json.loads((ctx.PROJECT / 'runtime-settings.json').read_text()).get('oxi', {}).get('prompt_cache')
    except (OSError, ValueError, AttributeError):
        value = None
    return value is True


def oxi_npu_settings(ctx):
    """Public default is off; a local owner-reviewed package supplies the pins."""
    try:
        value = json.loads((ctx.PROJECT / 'runtime-settings.json').read_text()).get('oxi', {})
        path = value.get('npu_profile')
        return (value.get('npu') is True,
                path if isinstance(path, str) and Path(path).is_absolute() else None)
    except (OSError, ValueError, AttributeError):
        return False, None


def oxi_text_encoder(ctx):
    """Prompt encoder for oxi conditioning (TE product binding; docs/STUDIO-OXI.md).

    'sdcpp' (default, and the rollback: remove the key): the probe sd-cli hand-over.
    'native': `oxi cond-native` on the CPU (no sd-cli, no DiT load for the prompt).
    'native-npu': `oxi cond-native` through the checked NPU worker when the job holds an NPU permit,
    else the CPU native encoder. Native entries live under their own encoder-identity cache keys, so
    legacy entries are never replaced and a rollback finds them again."""
    try:
        value = json.loads((ctx.PROJECT / 'runtime-settings.json').read_text()).get('oxi', {}).get('text_encoder')
    except (OSError, ValueError, AttributeError):
        value = None
    return value if value in ctx.OXI_TEXT_ENCODERS else 'sdcpp'


def oxi_te_arith(ctx, job):
    """Encoder arithmetic of this job's native conditioning, or None for the sd-cli hand-over."""
    mode = ctx.oxi_text_encoder()
    if mode == 'sdcpp' or not ctx.KLEIN_LLM.is_file():
        return None
    if mode == 'native-npu' and job.get('npu_requested'):
        return ctx.OXI_TE_NPU_ARITH
    return ctx.OXI_TE_CPU_ARITH


def oxi_te_key_args(ctx, spec):
    """generate/cond-native flags naming the encoder namespace of the conditioning cache."""
    arith = spec.get('te_arith')
    return [] if arith is None else ['--te-gguf', str(ctx.KLEIN_LLM), '--te-arith', arith]


def oxi_cache_environment(ctx, cache_dir):
    return None if cache_dir is None else dict(os.environ, OXI_COND_CACHE=str(cache_dir))


def oxi_vae_conv(ctx):
    try:
        value = json.loads((ctx.PROJECT / 'runtime-settings.json').read_text()).get('oxi', {}).get('vae_conv')
    except (OSError, ValueError, AttributeError):
        value = None
    return value if value in ctx.OXI_VAE_CONVS else ctx.OXI_DEFAULT_VAE_CONV


def oxi_missing(ctx):
    """What the oxi path still needs (empty = ready): its two binaries and the klein Q8 files."""
    missing = [p.name if p.name != 'sd-cli' else 'sd-cli (probe)' for p in (ctx.OXI_BIN, ctx.SD_PROBE_CLI)
               if not (p.is_file() and os.access(p, os.X_OK))]
    return missing + ctx.edit_missing_files('q8')


def oxi_binary_sha256(ctx, path=None):
    """sha256 of the oxi binary, cached by (path, size, mtime_ns); None when unreadable."""
    path = Path(ctx.OXI_BIN if path is None else path)
    try:
        info = path.stat()
        key = (str(path), info.st_size, info.st_mtime_ns)
        if key not in ctx.OXI_SHA_CACHE:
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                while data := stream.read(1024 * 1024):
                    digest.update(data)
            ctx.OXI_SHA_CACHE.clear()
            ctx.OXI_SHA_CACHE[key] = digest.hexdigest()
        return ctx.OXI_SHA_CACHE[key]
    except OSError:
        return None


def oxi_fast_setting(ctx, flag):
    return ctx.OXI_FAST_ARGS[ctx.OXI_FAST_ARGS.index(flag) + 1]


def oxi_size_gate_passed(ctx, width, height):
    """True only for a well-formed PASS record of this exact size and the current oxi binary."""
    path = ctx.OXI_GATE_DIR / f'{int(width)}x{int(height)}.json'
    try:
        info = path.lstat()
        if not S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_size > 1024 * 1024:
            return False
        record = json.loads(path.read_text())
        if not isinstance(record, dict) or record.get('schema') != ctx.OXI_GATE_SCHEMA:
            return False
        if type(record.get('width')) is not int or type(record.get('height')) is not int:
            return False
        if (record['width'], record['height']) != (width, height) or record.get('verdict') != 'PASS':
            return False
        seeds = record.get('seeds')
        if not isinstance(seeds, list) or not seeds or not all(isinstance(e, dict) and e.get('pass') is True for e in seeds):
            return False
        expected = record.get('oxi_sha256')
        if not isinstance(expected, str) or not re.fullmatch(r'[0-9a-f]{64}', expected):
            return False
        settings = record.get('oxi_settings') or {}
        if not isinstance(settings, dict):
            return False
        current = {'chunk1': ctx.oxi_fast_setting('--chunk1'), 'chunk2': ctx.oxi_fast_setting('--chunk2'),
                   'attn_prec': ctx.oxi_fast_setting('--attn-prec'), 'vae_conv': ctx.oxi_vae_conv()}
        for key, value in current.items():
            if settings.get(key) is not None and str(settings[key]) != value:
                return False
    except (OSError, ValueError, TypeError, AttributeError):
        return False
    return ctx.oxi_binary_sha256() == expected


def oxi_gated_sizes(ctx):
    """Sizes beyond OXI_SIZES that currently pass their size gate (for the engine panel)."""
    sizes = []
    try:
        names = sorted(p.name for p in ctx.OXI_GATE_DIR.iterdir())
    except OSError:
        return sizes
    for name in names:
        m = re.fullmatch(r'(\d{2,4})x(\d{2,4})\.json', name)
        if m:
            size = (int(m.group(1)), int(m.group(2)))
            if size not in ctx.OXI_SIZES and ctx.oxi_size_gate_passed(*size):
                sizes.append(size)
    return sizes


def choose_t2i_engine(ctx, setting, model, width, height, missing):
    """Route one klein text-to-image request: ('oxi' | 'sdcpp', reason for the UI)."""
    if setting == 'sdcpp':
        return 'sdcpp', 'engine setting: sd.cpp'
    if model not in ctx.OXI_MODELS:
        return 'sdcpp', 'oxi runs FLUX.2 klein Q8 only; this model runs on sd.cpp'
    gated = (width, height) not in ctx.OXI_SIZES
    if gated and not ctx.oxi_size_gate_passed(width, height):
        return 'sdcpp', f'oxi fast path is 512 × 512 only; {width} × {height} runs on sd.cpp (slower)'
    if missing:
        return 'sdcpp', 'oxi is not installed (' + ', '.join(missing) + '); using sd.cpp'
    if gated:
        return 'oxi', f'oxi ({width} × {height} passed its size gate)'
    return 'oxi', 'oxi fast path (512 × 512)'


def engine_info(ctx):
    missing = ctx.oxi_missing()
    return {'engine': ctx.engine_setting(), 'choices': list(ctx.ENGINE_CHOICES), 'labels': ctx.ENGINE_LABELS,
            'oxi_ready': not missing, 'missing': missing, 'oxi_sizes': [f'{w}x{h}' for w, h in ctx.OXI_SIZES],
            'oxi_gated_sizes': [f'{w}x{h}' for w, h in ctx.oxi_gated_sizes()],
            'note': '1024 px and other sizes stay on sd.cpp (slower); edits, refine and upscale use sd.cpp'}


def oxi_estimate_seconds(ctx, count, width=512, height=512):
    # Gated sizes: scaled by pixels^0.75 (768: ~1.8x, 1024: ~2.8x, as in the one-seed 768/1024
    # screen); uncalibrated like the 512 px figure.
    scale = (width * height / (512 * 512)) ** 0.75
    return round(ctx.OXI_ESTIMATE['handover_s'] + count * ctx.OXI_ESTIMATE['image_s'] * scale, 1)


def oxi_generate_args(ctx, spec, seed, out, cache_dir=None):
    return [str(ctx.OXI_BIN), 'generate', '--gguf', str(ctx.KLEIN_FILES['q8']), '--vae', str(ctx.KLEIN_VAE),
            '--width', str(spec['width']), '--height', str(spec['height']), '--steps', str(ctx.EDIT_STEPS),
            '--seed', str(int(seed)), '--cache', str(ctx.OXI_COND_CACHE if cache_dir is None else cache_dir), *ctx.OXI_FAST_ARGS,
            '--vae-conv', ctx.oxi_vae_conv(), '--threads', str(ctx.oxi_threads()), '--out', str(out),
            *ctx.oxi_te_key_args(spec)]


def oxi_cond_native_args(ctx, spec, cache_dir=None):
    """New prompt without sd-cli: native context + native RoPE + computed sigmas (exit 4 = keep sd-cli)."""
    args = [str(ctx.OXI_BIN), 'cond-native', '--gguf', str(ctx.KLEIN_FILES['q8']),
            '--width', str(spec['width']), '--height', str(spec['height']), '--steps', str(ctx.EDIT_STEPS),
            '--max-tokens', '512', '--cache', str(ctx.OXI_COND_CACHE if cache_dir is None else cache_dir),
            *ctx.oxi_te_key_args(spec)]
    if spec.get('te_arith') == ctx.OXI_TE_CPU_ARITH:
        args += ['--te-backend', 'cpu-ggml', '--threads', str(ctx.oxi_threads())]
    return args


def oxi_cond_put_args(ctx, spec, handover, cache_dir=None):
    return [str(ctx.OXI_BIN), 'cond-put', '--handoff', str(handover), '--gguf', str(ctx.KLEIN_FILES['q8']),
            '--width', str(spec['width']), '--height', str(spec['height']), '--steps', str(ctx.EDIT_STEPS),
            '--sigmas', 'computed', '--cache', str(ctx.OXI_COND_CACHE if cache_dir is None else cache_dir)]


def handover_stop(ctx):
    """Stop predicate for the probe sd-cli: the first SDPROBE line after `dump.pe call=0`.

    The probe prints a tag's line and then writes its file on the same thread; the next probe op
    starts only after that, so by the following line the four hand-over files are complete."""
    seen = {'pe': False}

    def stop(line):
        if not line.startswith('SDPROBE '):
            return False
        if seen['pe']:
            return True
        if line.startswith('SDPROBE dump.pe call=0 '):
            seen['pe'] = True
        return False
    return stop


def prune_cond_cache(ctx, keep=OXI_COND_KEEP, cache_dir=None):
    cache_dir = ctx.OXI_COND_CACHE if cache_dir is None else Path(cache_dir)
    try:
        entries = sorted((d for d in cache_dir.iterdir() if d.is_dir()), key=lambda d: d.stat().st_mtime)
    except OSError:
        return
    for old in entries[:max(0, len(entries) - keep)]:
        shutil.rmtree(old, ignore_errors=True)


def make_conditioning(ctx, job, prompt, spec, cache_dir=None):
    """Encode a new prompt once.

    Native encoder (setting 'native'/'native-npu'): one `oxi cond-native` process, no sd-cli. Any
    refusal (exit 4: emphasis/escape/BREAK/special/LoRA/non-UTF-8 or > 512 tokens) or failure drops
    this job to the legacy route for the rest of the request (`spec['te_arith'] = None`, so generate
    reads legacy keys): probe sd-cli up to model call 0, then `oxi cond-put`. An encoder run that
    may have left native work behind (NPU worker recovery) is never followed by a CPU retry."""
    if spec.get('te_arith') is not None:
        job['state'] = 'encoding prompt'
        job['engine_binary'] = 'oxi (text encoder)'
        rc = ctx.run_engine(ctx.oxi_cond_native_args(spec, cache_dir), job, ctx.oxi_cache_environment(cache_dir), prompt)
        if rc == 0:
            ctx.prune_cond_cache(cache_dir=cache_dir)
            return
        if spec['te_arith'] == ctx.OXI_TE_NPU_ARITH and rc == 70:
            raise ctx.OxiNpuRecovery('NPU text encoder cleanup requires recovery; sd-cli was not started')
        spec['te_arith'] = None
        job['te_note'] = ('prompt needs the sd.cpp text encoder' if rc == ctx.OXI_NEEDS_REFERENCE
                          else f'native text encoder exited {rc}; sd.cpp hand-over used')
    work = ctx.OXI_WORK / uuid.uuid4().hex
    dump, handover = work / 'dump', work / 'handover'
    try:
        dump.mkdir(parents=True, mode=0o700)
        handover.mkdir(mode=0o700)
        job['state'] = 'encoding prompt'
        job['engine_binary'] = 'sd-cli (probe)'
        environment = dict(ctx.oxi_cache_environment(cache_dir) or os.environ, SD_PROBE='1',
                           SD_PROBE_DUMP_DIR=str(dump), SD_PROBE_DUMP_CALL='0')
        args = ctx.build_klein_t2i_args(ctx.SD_PROBE_CLI, ctx.KLEIN_FILES['q8'], spec['width'], spec['height'],
                                    work / 'unused.png', spec['threads'], spec['fa'], seed=spec['seeds'][0])
        ctx.run_engine(args, job, environment, prompt, stop=ctx.handover_stop())
        for tag in ctx.HANDOVER_TAGS:
            found = list(dump.glob(f'{tag}.call0.*.bin'))
            if len(found) != 1:
                raise ctx.OxiFailure('the prompt hand-over from sd-cli is incomplete')
            shutil.copyfile(found[0], handover / found[0].name)
        job['engine_binary'] = 'oxi'
        rc = ctx.run_engine(ctx.oxi_cond_put_args(spec, handover, cache_dir), job, ctx.oxi_cache_environment(cache_dir), prompt)
        if rc != 0:
            raise ctx.OxiFailure(f'oxi cond-put exited {rc}')
        ctx.prune_cond_cache(cache_dir=cache_dir)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def oxi_generate_all(ctx, job, prompt, spec, cache_dir=None):
    spec['te_arith'] = ctx.oxi_te_arith(job)
    for index, (seed, out) in enumerate(zip(spec['seeds'], spec['outputs'])):
        if len(spec['outputs']) > 1:
            job['step'] = f'image {index + 1} of {len(spec["outputs"])}'
        args = ctx.oxi_generate_args(spec, seed, out, cache_dir)
        job['engine_binary'] = 'oxi'
        job['oxi_forwards'] = 0
        rc = ctx.run_engine(args, job, ctx.oxi_cache_environment(cache_dir), prompt, prompt_flag='--cond-prompt-file')
        if rc == ctx.OXI_NO_CONDITIONING and index == 0:
            ctx.make_conditioning(job, prompt, spec, cache_dir)
            args = ctx.oxi_generate_args(spec, seed, out, cache_dir)  # the namespace may have fallen back
            job['engine_binary'] = 'oxi'
            job['oxi_forwards'] = 0
            job['state'] = 'starting'
            rc = ctx.run_engine(args, job, ctx.oxi_cache_environment(cache_dir), prompt, prompt_flag='--cond-prompt-file')
        if rc != 0:
            if job.get('npu_requested'):
                if rc in (69, 71):
                    raise ctx.OxiNpuFallback('NPU package refused or latched; finished on CPU')
                raise ctx.OxiNpuRecovery('NPU worker cleanup requires recovery; CPU generation was not started')
            raise ctx.OxiFailure(f'oxi exited {rc}')
        if ctx.png_size(out) != (spec['width'], spec['height']):
            raise ctx.OxiFailure('oxi did not write a %d × %d image' % (spec['width'], spec['height']))
    job.pop('step', None)


def run_oxi_t2i(ctx, job, prompt, spec):
    """The oxi step of a klein 512 px request; returns an exit code like run_engine.

    "auto": any oxi failure removes oxi's images and runs the whole request on sd-cli instead
    (same seeds, so the same pictures within the image gate). "oxi": the failure is the job's error.
    """
    job['engine'] = 'oxi'
    try:
        def generate(cache_dir=None):
            try:
                ctx.oxi_generate_all(job, prompt, spec, cache_dir)
            except ctx.OxiNpuFallback as error:
                for out in spec['outputs']:
                    Path(out).unlink(missing_ok=True)
                job['npu_requested'] = False
                job['engine_note'] = str(error)
                ctx.oxi_generate_all(job, prompt, spec, cache_dir)
        if ctx.oxi_prompt_cache_enabled():
            generate()
        else:
            with tempfile.TemporaryDirectory(prefix='oxi-cond-') as cache_dir:
                generate(Path(cache_dir))
        if id(job) in ctx.IMAGE_LEASES:
            backend = 'oxi-npu' if job.get('npu_requested') else 'oxi-cpu'
            job['engine'] = backend
            job['device'] = 'npu' if job.get('npu_requested') else 'cpu'
            for record in spec['results']:
                record['backend'] = backend
        return 0
    except ctx.OxiNpuRecovery:
        raise  # Never overlap a CPU retry with a possibly surviving native child.
    except Exception as error:  # noqa: BLE001 - every oxi failure is a fallback reason
        reason = str(error) or type(error).__name__
    for out in spec['outputs']:
        Path(out).unlink(missing_ok=True)
    if spec['setting'] == 'oxi':
        raise ctx.OxiFailure(f'oxi failed: {reason} (engine setting "oxi" has no sd.cpp fallback)')
    job['engine'] = 'sd.cpp'
    job['engine_note'] = f'oxi failed ({reason}); finished on sd.cpp'
    job['engine_binary'] = 'sd-cli'
    job.pop('step', None)
    job['phase'] = None
    job['state'] = 'starting sd.cpp'
    for record in spec['results']:
        record['backend'] = 'sd.cpp'
        record['fallback'] = reason
    return ctx.run_engine(spec['sd_args'], job, None, prompt)
