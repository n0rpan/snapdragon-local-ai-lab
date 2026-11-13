#!/usr/bin/env python3
"""Per-size image gate for oxi (F-IMG-SHIP5): judge sd.cpp-reference / one-ULP-control / oxi
candidate PNG triples and write the PASS/FAIL record Studio's router reads.

Recipe (docs/IMAGE-LARGE-SCREEN.md, docs/IMAGE-ATTENTION.md), per size and per seed (42/43/44):

  N = PSNR(sd.cpp reference, sd.cpp one-ULP perturbed control)   the measured noise floor
  G = min(30, N - 6) dB                                         the frozen per-size gate
  P = PSNR(oxi candidate, sd.cpp reference)
  pass iff P >= G for EVERY seed, with at least MIN_SEEDS distinct seeds.

PSNR is RGB-normalised: all R, G, B samples of both images scaled to [0, 1], one MSE over them,
PSNR = 10 log10(1 / MSE). Identical images have infinite PSNR (stored as null with
"identical": true). Alpha is ignored.

  judge  compares PNG files that already exist and writes the record.
  run    renders the triples with command templates (sd-cli reference, sd-cli + one-ULP control,
         oxi candidate) from a JSON recipe or the command line, then calls judge.

The record (<record dir>/<W>x<H>.json, written temp+rename, mode 0600) holds no prompt text and
no paths: binaries and images are named by sha256 only. Studio routes a non-512 size to oxi only
for a PASS record of exactly that size whose oxi_sha256 equals the binary it would run. The record
dir defaults to Studio's: $OXI_SIZE_GATE_DIR, else <$LOCAL_IMAGE_CONFIG_DIR or
~/Work/prompt-studio>/oxi-size-gates. A newer judgement of a size replaces the older record, so a
FAIL withdraws an earlier PASS.

    python3 -m snapdragon_ai_lab.runtime.image_size_gate judge --width 768 --height 768 \\
        --oxi PATH/oxi --seed 42 REF CTL CAND --seed 43 ... --seed 44 ...
    python3 -m snapdragon_ai_lab.runtime.image_size_gate run --recipe recipe.json

Exit status: 0 PASS, 1 FAIL, 2 usage or input error.

-----------------------------------------------------------------------------------------------
sd.cpp parity-emulation gaps (F-W5G-3)
-----------------------------------------------------------------------------------------------
oxi's parity mode is NOT a bit-exact re-run of sd.cpp:
  * ggml's SIMD summation order is not reproduced (residual ~0.037 % relative per block);
  * repack / NEON round-half-even ties ARE reproduced;
  * flash attention with F16 K/V, sage attention and KleidiAI kernels are NOT emulated. If
    sd.cpp's production arguments enable any of them, the parity reference must enable them too
    before any comparison.
The gate therefore compares against sd.cpp's own image plus its measured one-ULP noise floor
(N above), never against bit parity.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import struct
import subprocess
import sys
import tempfile
import zlib

SCHEMA = 'oxi-size-gate/1'
TOOL_VERSION = '1'
GATE_CAP_DB = 30.0
GATE_MARGIN_DB = 6.0
MIN_SEEDS = 3
DEFAULT_SEEDS = (42, 43, 44)
SETTING_KEYS = ('threads', 'chunk1', 'chunk2', 'attn_prec', 'vae_conv')
PLACEHOLDER = re.compile(r'\{(seed|out|width|height|prompt_file)\}')


class GateError(Exception):
    """Unusable input (missing/odd PNG, wrong size, failed render); no record is written."""


def default_record_dir():
    explicit = os.environ.get('OXI_SIZE_GATE_DIR')
    if explicit:
        return Path(explicit)
    project = os.environ.get('LOCAL_IMAGE_CONFIG_DIR', str(Path.home() / 'Work/prompt-studio'))
    return Path(project) / 'oxi-size-gates'


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        while data := stream.read(1024 * 1024):
            digest.update(data)
    return digest.hexdigest()


def _unfilter(raw, width, height, step):
    stride = width * step
    if len(raw) < (stride + 1) * height:
        raise GateError('truncated PNG')
    out = bytearray(stride * height)
    previous = bytearray(stride)
    for y in range(height):
        base = y * (stride + 1)
        mode = raw[base]
        row = bytearray(raw[base + 1:base + 1 + stride])
        if mode == 1:
            for i in range(step, stride):
                row[i] = (row[i] + row[i - step]) & 255
        elif mode == 2:
            row = bytearray((a + b) & 255 for a, b in zip(row, previous))
        elif mode == 3:
            for i in range(stride):
                left = row[i - step] if i >= step else 0
                row[i] = (row[i] + ((left + previous[i]) >> 1)) & 255
        elif mode == 4:
            for i in range(stride):
                a = row[i - step] if i >= step else 0
                b, c = previous[i], (previous[i - step] if i >= step else 0)
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                row[i] = (row[i] + (a if pa <= pb and pa <= pc else b if pb <= pc else c)) & 255
        elif mode != 0:
            raise GateError('corrupt PNG filter')
        out[y * stride:(y + 1) * stride] = row
        previous = row
    return out


def read_png_rgb(path):
    """(width, height, RGB bytes) of an 8-bit, non-interlaced RGB or RGBA PNG."""
    data = Path(path).read_bytes()
    if data[:8] != b'\x89PNG\r\n\x1a\n':
        raise GateError('not a PNG')
    pos, idat, header = 8, [], None
    while pos + 8 <= len(data):
        length, kind = struct.unpack('>I4s', data[pos:pos + 8])
        body = data[pos + 8:pos + 8 + length]
        pos += 12 + length
        if kind == b'IHDR':
            header = struct.unpack('>IIBBBBB', body)
        elif kind == b'IDAT':
            idat.append(body)
        elif kind == b'IEND':
            break
    if header is None or not idat:
        raise GateError('unreadable PNG')
    width, height, depth, ctype, _, _, interlace = header
    if depth != 8 or interlace or ctype not in (2, 6):
        raise GateError('unsupported PNG (8-bit RGB/RGBA, non-interlaced only)')
    step = 3 if ctype == 2 else 4
    try:
        raw = zlib.decompress(b''.join(idat))
    except zlib.error as error:
        raise GateError('corrupt PNG data') from error
    pixels = _unfilter(raw, width, height, step)
    if step == 4:
        rgb = bytearray(width * height * 3)
        rgb[0::3], rgb[1::3], rgb[2::3] = pixels[0::4], pixels[1::4], pixels[2::4]
        pixels = rgb
    return width, height, bytes(pixels)


def psnr_rgb(a, b):
    """RGB-normalised PSNR in dB between two equal-length RGB byte strings (inf if identical)."""
    if len(a) != len(b) or not a:
        raise GateError('images differ in size')
    squared = sum((x - y) * (x - y) for x, y in zip(a, b))
    if squared == 0:
        return math.inf
    mse = squared / (len(a) * 255.0 * 255.0)
    return 10.0 * math.log10(1.0 / mse)


def gate_db(noise_db):
    return min(GATE_CAP_DB, noise_db - GATE_MARGIN_DB)


def _db(value):
    return None if math.isinf(value) else round(value, 4)


def judge_seed(seed, reference, control, candidate, width, height):
    images = {}
    for role, path in (('reference', reference), ('control', control), ('candidate', candidate)):
        w, h, rgb = read_png_rgb(path)
        if (w, h) != (width, height):
            raise GateError(f'seed {seed}: {role} image is {w}x{h}, expected {width}x{height}')
        images[role] = rgb
    noise = psnr_rgb(images['reference'], images['control'])
    gate = gate_db(noise)
    score = psnr_rgb(images['candidate'], images['reference'])
    entry = {'seed': int(seed), 'n_db': _db(noise), 'gate_db': _db(gate), 'p_db': _db(score),
             'pass': bool(score >= gate),
             'images_sha256': {role: sha256_file(path) for role, path in
                               (('reference', reference), ('control', control), ('candidate', candidate))}}
    if math.isinf(noise):
        entry['control_identical'] = True   # the one-ULP control did not perturb anything
    if math.isinf(score):
        entry['identical'] = True
    return entry


def write_record(record, record_dir):
    record_dir = Path(record_dir)
    record_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    target = record_dir / f"{record['width']}x{record['height']}.json"
    fd, temporary = tempfile.mkstemp(prefix='.gate-', suffix='.tmp', dir=record_dir)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, 'w') as stream:
            json.dump(record, stream, indent=2, sort_keys=True)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    return target


def judge(width, height, oxi_bin, triples, settings=None, record_dir=None, now=None, write=True):
    """Judge `triples` = [(seed, reference_png, control_png, candidate_png), ...] for one size.

    Returns the record (also written to `record_dir` unless write=False). Raises GateError on
    unusable input, in which case nothing is written."""
    width, height = int(width), int(height)
    if not (16 <= width <= 4096 and 16 <= height <= 4096):
        raise GateError('size out of range')
    if not triples:
        raise GateError('no seeds to judge')
    seeds = [judge_seed(seed, ref, ctl, cand, width, height) for seed, ref, ctl, cand in triples]
    distinct = len({entry['seed'] for entry in seeds})
    verdict = 'PASS' if distinct >= MIN_SEEDS and distinct == len(seeds) and all(e['pass'] for e in seeds) else 'FAIL'
    oxi_settings = {key: (settings or {}).get(key) for key in SETTING_KEYS
                    if (settings or {}).get(key) is not None}
    record = {'schema': SCHEMA, 'tool_version': TOOL_VERSION, 'width': width, 'height': height,
              'oxi_sha256': sha256_file(oxi_bin), 'oxi_name': Path(oxi_bin).name,
              'oxi_settings': oxi_settings, 'seeds': seeds, 'verdict': verdict,
              'rule': f'P >= min({GATE_CAP_DB:g}, N - {GATE_MARGIN_DB:g}) dB for every seed, '
                      f'>= {MIN_SEEDS} distinct seeds',
              'created_at': (now or datetime.now(timezone.utc)).strftime('%Y-%m-%dT%H:%M:%SZ')}
    if distinct < MIN_SEEDS or distinct != len(seeds):
        record['reason'] = f'needs {MIN_SEEDS} distinct seeds'
    if write:
        write_record(record, default_record_dir() if record_dir is None else record_dir)
    return record


def expand(template, values):
    """Fill {seed} {out} {width} {height} {prompt_file} in each argv element (no shell)."""
    if not isinstance(template, list) or not template or not all(isinstance(a, str) for a in template):
        raise GateError('a command template must be a non-empty list of strings')
    return [PLACEHOLDER.sub(lambda m: str(values[m.group(1)]), arg) for arg in template]


def _render(template, environment, values, timeout):
    command = expand(template, values)
    env = dict(os.environ, **{str(k): str(v) for k, v in (environment or {}).items()})
    try:
        # Output is discarded: engines may echo the prompt. Only the exit status is reported.
        rc = subprocess.run(command, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL, timeout=timeout).returncode
    except subprocess.TimeoutExpired as error:
        raise GateError(f'{Path(command[0]).name} timed out after {timeout:g} s') from error
    except OSError as error:
        raise GateError(f'{Path(command[0]).name} could not start ({error.strerror})') from error
    if rc != 0:
        raise GateError(f'{Path(command[0]).name} exited {rc}')
    if not Path(values['out']).is_file():
        raise GateError(f'{Path(command[0]).name} wrote no image')


def run(recipe, record_dir=None, work_dir=None, keep=False, now=None):
    """Render reference / control / candidate per seed from command templates, then judge.

    recipe keys: width, height, oxi_bin, reference, control, candidate (argv templates),
    optional seeds (default 42/43/44), reference_env / control_env / candidate_env (dicts),
    prompt_file, settings (threads/chunk1/chunk2/attn_prec/vae_conv), timeout_seconds."""
    if not isinstance(recipe, dict):
        raise GateError('recipe must be a JSON object')
    try:
        width, height = int(recipe['width']), int(recipe['height'])
        oxi_bin = recipe['oxi_bin']
        templates = {role: recipe[role] for role in ('reference', 'control', 'candidate')}
    except (KeyError, TypeError, ValueError) as error:
        raise GateError('recipe needs width, height, oxi_bin, reference, control and candidate') from error
    seeds = recipe.get('seeds', list(DEFAULT_SEEDS))
    if not isinstance(seeds, list) or not seeds or not all(type(s) is int for s in seeds):
        raise GateError('seeds must be a list of integers')
    timeout = float(recipe.get('timeout_seconds', 3600))
    owned = None
    if work_dir is None:
        owned = tempfile.TemporaryDirectory(prefix='oxi-size-gate-')
        work_dir = owned.name
    work = Path(work_dir)
    work.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        triples = []
        for seed in seeds:
            outputs = {}
            for role in ('reference', 'control', 'candidate'):
                out = work / f'{role}-{width}x{height}-{seed}.png'
                out.unlink(missing_ok=True)
                values = {'seed': seed, 'out': out, 'width': width, 'height': height,
                          'prompt_file': recipe.get('prompt_file', '')}
                _render(templates[role], recipe.get(f'{role}_env'), values, timeout)
                outputs[role] = out
            triples.append((seed, outputs['reference'], outputs['control'], outputs['candidate']))
        return judge(width, height, oxi_bin, triples, recipe.get('settings'), record_dir, now)
    finally:
        if owned is not None and not keep:
            owned.cleanup()


def summary(record):
    def show(value):
        return 'inf' if value is None else f'{value:.2f}'
    lines = [f"{record['width']}x{record['height']}: {record['verdict']} (oxi {record['oxi_sha256'][:12]})"]
    for entry in record['seeds']:
        lines.append(f"  seed {entry['seed']}: N {show(entry['n_db'])} dB, G {show(entry['gate_db'])} dB, "
                     f"P {show(entry['p_db'])} dB -> {'pass' if entry['pass'] else 'FAIL'}")
    if record.get('reason'):
        lines.append('  ' + record['reason'])
    return '\n'.join(lines)


def _settings_from(args):
    return {key: getattr(args, key) for key in SETTING_KEYS if getattr(args, key, None) is not None}


def main(argv=None):
    parser = argparse.ArgumentParser(prog='image_size_gate', description=__doc__.split('\n\n')[0])
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('judge', 'run'):
        sub = commands.add_parser(name)
        sub.add_argument('--record-dir', type=Path, default=None)
        sub.add_argument('--threads', type=int)
        sub.add_argument('--chunk1')
        sub.add_argument('--chunk2')
        sub.add_argument('--attn-prec', dest='attn_prec')
        sub.add_argument('--vae-conv', dest='vae_conv')
        sub.add_argument('--json', action='store_true', help='print the record instead of a summary')
        if name == 'judge':
            sub.add_argument('--width', type=int, required=True)
            sub.add_argument('--height', type=int, required=True)
            sub.add_argument('--oxi', required=True, help='the oxi binary the candidates came from')
            sub.add_argument('--seed', nargs=4, action='append', required=True,
                             metavar=('SEED', 'REFERENCE', 'CONTROL', 'CANDIDATE'))
        else:
            sub.add_argument('--recipe', type=Path, help='JSON recipe (see run())')
            sub.add_argument('--width', type=int)
            sub.add_argument('--height', type=int)
            sub.add_argument('--oxi')
            sub.add_argument('--seeds', help='comma-separated, default 42,43,44')
            sub.add_argument('--prompt-file', dest='prompt_file')
            for role in ('reference', 'control', 'candidate'):
                sub.add_argument(f'--{role}-cmd', dest=f'{role}_cmd',
                                 help='argv template (shell-quoted, not run by a shell); '
                                      'placeholders {seed} {out} {width} {height} {prompt_file}')
            sub.add_argument('--work-dir', type=Path)
            sub.add_argument('--keep', action='store_true', help='keep rendered images in --work-dir')
    args = parser.parse_args(argv)
    try:
        if args.command == 'judge':
            triples = []
            for seed, reference, control, candidate in args.seed:
                try:
                    seed = int(seed)
                except ValueError:
                    raise GateError('seed must be an integer') from None
                triples.append((seed, reference, control, candidate))
            record = judge(args.width, args.height, args.oxi, triples, _settings_from(args), args.record_dir)
        else:
            recipe = {}
            if args.recipe is not None:
                try:
                    recipe = json.loads(args.recipe.read_text())
                except (OSError, ValueError) as error:
                    raise GateError('recipe is not readable JSON') from error
                if not isinstance(recipe, dict):
                    raise GateError('recipe must be a JSON object')
            for key in ('width', 'height', 'prompt_file'):
                if getattr(args, key) is not None:
                    recipe[key] = getattr(args, key)
            if args.oxi is not None:
                recipe['oxi_bin'] = args.oxi
            if args.seeds is not None:
                try:
                    recipe['seeds'] = [int(s) for s in args.seeds.split(',') if s.strip()]
                except ValueError:
                    raise GateError('seeds must be integers') from None
            for role in ('reference', 'control', 'candidate'):
                if getattr(args, f'{role}_cmd') is not None:
                    recipe[role] = shlex.split(getattr(args, f'{role}_cmd'))
            settings = dict(recipe.get('settings') or {}, **_settings_from(args))
            if settings:
                recipe['settings'] = settings
            record = run(recipe, args.record_dir, args.work_dir, args.keep)
    except (GateError, OSError) as error:
        message = error.strerror if isinstance(error, OSError) and error.strerror else str(error)
        print(f'image_size_gate: {message}', file=sys.stderr)
        return 2
    print(json.dumps(record, indent=2, sort_keys=True) if args.json else summary(record))
    return 0 if record['verdict'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
