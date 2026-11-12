#!/usr/bin/env python3
"""Generate the v0.1.2 FINAL install stage from an uninstalled base stage; never install.

Writes ONE fresh directory (it must not exist; the base stage is only read) with
FIVE guarded live payloads, bound by stage_rollback_guard and installed by
tools/install_staged_release.py (copies of both are put in <stage>/tools):

  unit      local-ai-chat.service   re-pointed at the content-addressed release of
                                    the lab runtime at --lab-rev; keeps
                                    LOCAL_AI_IMAGE_WATCHDOG_KILL_CLIENT=1 (O-100)
  runtime   runtime.json            the base runtime (Flash 18 GiB / prefix store /
                                    PSI off, MoE 128K / checkpoints 8, cache 64 GB /
                                    TTL 7 d, sidecar, model SHAs) with the chosen
                                    oxe-server bound; Ornith without "(beta)"
  opencode  opencode.json           canonical MoE limit.input 65536 (FAST, O-105);
                                    no global `compaction` (keep never leaks to Flash)
  cli       local-ai                from --lab-rev
  launcher  launch_local_plan.py    from --lab-rev (default fast, deep 120000 explicit)

Every live preimage is read once (sha256 + mode, nothing else is written there):
the four base keys must still equal the base stage's recorded preimages, and the
launcher's bytes become rollback/launcher.original. The generated rollback.py
restores each file with its captured preimage mode (0700 for the CLI and the
launcher). The O-100 N9 chmods are a separate guarded script (n9/n9_chmod.py:
dirs 0700, files 0600, identity-checked, nothing deleted, no service control).

The generator itself never calls systemctl, never writes outside --out, never
starts a model. Use --verify STAGE to recheck a written stage's seal.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import time

TOOLS = Path(__file__).resolve().parent
sys.dont_write_bytecode = True  # the copy in <stage>/tools must not add files to a sealed stage
sys.path.insert(0, str(TOOLS))
import stage_rollback_guard as guard  # noqa: E402
# stage_controller_release / stage_flash_memory are imported inside generate(): the copy
# bundled in <stage>/tools only needs the guard for --verify.

RUNTIME_SRC = 'src/snapdragon_ai_lab/runtime'
KEYS = ('unit', 'runtime', 'opencode', 'cli', 'launcher')
STAGED = {'unit': 'local-ai-chat.service', 'runtime': 'runtime.json', 'opencode': 'opencode.json',
          'cli': 'local-ai', 'launcher': 'launch_local_plan.py'}
MOE_MODELS = ('ornith-35b-a3b-local', 'qwen3.6-35b-a3b-local')
FLASH = 'qwen3.8-flash-next-local'
FAST_INPUT, DEEP_INPUT, MOE_CONTEXT, MOE_OUTPUT = 65536, 120000, 131072, 16384
CACHE = {'disk_budget_bytes': 64000000000, 'ttl_secs': 604800}
WATCHDOG = 'Environment=LOCAL_AI_IMAGE_WATCHDOG_KILL_CLIENT=1'
CARRY = ('T1-PRIMERS.json', 'T1-primer-ornith-35b-a3b-local.json', 'T1-primer-qwen3.6-35b-a3b-local.json',
         'harness-registry')
SEAL = 'SEAL.json'
SEAL_SKIP = ('receipts', 'retired.json', 'n9/receipts', 'rollback/receipts', SEAL)
LIVE_LIMIT = 8 * 1024 ** 2


class StageError(Exception):
    pass


def sha(data):
    return hashlib.sha256(data).hexdigest()


def git(repo, *args):
    return subprocess.run(['git', '-C', str(repo), *args], check=True, capture_output=True).stdout


def export_tree(repo, rev, prefix, dest):
    """Files under `prefix` at commit `rev` (git objects only; the worktree is ignored)."""
    commit = git(repo, 'rev-parse', '--verify', rev + '^{commit}').decode().strip()
    data = git(repo, 'archive', '--format=tar', commit, '--', prefix)
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        tar.extractall(dest, filter='data')
    return commit, Path(dest) / prefix


def capture_live(path):
    """sha256, mode and bytes of one live file; a regular file owned by us, never followed."""
    path = Path(path)
    if not path.is_absolute():
        raise StageError('live path must be absolute: %s' % path)
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    except OSError as exc:
        raise StageError('live file unreadable: %s (%s)' % (path, exc.strerror)) from None
    with os.fdopen(fd, 'rb') as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_size > LIVE_LIMIT:
            raise StageError('live file must be a bounded regular file we own: %s' % path)
        data = stream.read(LIVE_LIMIT + 1)
    return {'sha256': sha(data), 'mode': info.st_mode & 0o7777, 'bytes': data}


def dump_json(obj):
    return (json.dumps(obj, indent=2, ensure_ascii=False) + '\n').encode()


def same_format(raw):
    """The base files are indent-2, non-ASCII kept, trailing newline: keep that byte for byte."""
    if dump_json(json.loads(raw)) != raw:
        raise StageError('base JSON is not in the canonical indent-2 layout; refusing to reformat it')
    return json.loads(raw)


def arg_after(args, flag):
    hits = [i for i, a in enumerate(args) if a == flag]
    if len(hits) != 1 or hits[0] + 1 >= len(args):
        return None
    return args[hits[0] + 1]


def walk_keys(obj, path=()):
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield path + (key,)
            yield from walk_keys(value, path + (key,))
    elif isinstance(obj, list):
        for value in obj:
            yield from walk_keys(value, path)


def check_runtime(runtime, base_runtime, requirements, server_sha):
    errors = []
    if runtime.get('cache') != CACHE:
        errors.append('runtime cache must be %s' % CACHE)
    profiles = runtime.get('profiles') or {}
    for alias, want in requirements.items():
        profile = profiles.get(alias) or {}
        oxe = ((profile.get('memlat_overrides') or {}).get('oxe')) or {}
        args = oxe.get('oxe_args') or []
        if profile.get('sha256') != want['model_sha256']:
            errors.append(alias + ': model sha256 changed')
        if '(beta)' in str(profile.get('display_name', '')):
            errors.append(alias + ': display_name still carries "(beta)" (O-100)')
        if oxe.get('context_size') != want['context_size'] or arg_after(args, '--ctx-size') != '{context}':
            errors.append(alias + ': oxe context must be %d' % want['context_size'])
        for flag, value in (('--ctx-checkpoints', want['checkpoints']), ('--session-ram-bytes', want['session_ram_bytes']),
                            ('--mtp-max-ctx', want['mtp_max_ctx']), ('--park-disk-min-bytes', 536870912)):
            if arg_after(args, flag) != str(value):
                errors.append('%s: %s must be %s' % (alias, flag, value))
        if oxe.get('server_sha256') != server_sha:
            errors.append(alias + ': oxe server_sha256 is not the staged server')
        if 'OXE_ATTN' in (oxe.get('environment') or {}):
            errors.append(alias + ': OXE_ATTN set in production (fused attention is gate-only)')
    flash = (((profiles.get(FLASH) or {}).get('memlat_overrides') or {}).get('oxe_flash')) or {}
    fargs = flash.get('oxe_args') or []
    if flash.get('cache_pref') != '18':
        errors.append('flash: cache_pref must be "18" (18 GiB)')
    if (flash.get('prefix_store') or {}).get('enabled') is not True:
        errors.append('flash: prefix store must be on')
    if arg_after(fargs, '--flash-psi-some') != '0' or arg_after(fargs, '--flash-psi-full') != '0':
        errors.append('flash: PSI throttling must stay off (--flash-psi-some/full 0)')
    if not flash.get('sidecar_path') or arg_after(fargs, '--flash-sidecar') != flash.get('sidecar_path'):
        errors.append('flash: sidecar must be bound (--flash-sidecar == sidecar_path)')
    base_flash = (base_runtime.get('profiles') or {}).get(FLASH) or {}
    if (profiles.get(FLASH) or {}) != base_flash:
        errors.append('flash: profile differs from the base stage (it must be carried byte-identical)')
    for alias, profile in (base_runtime.get('profiles') or {}).items():
        if (profiles.get(alias) or {}).get('sha256') != profile.get('sha256'):
            errors.append(alias + ': model sha256 differs from the base stage')
    for keys in walk_keys(runtime):
        if 'oxi' in keys and keys[-1] == 'text_encoder':
            errors.append('runtime: oxi.text_encoder present (the native text encoder must stay inert)')
    return errors


def check_opencode(opencode, base_opencode):
    errors = []
    if 'compaction' in opencode:
        errors.append('opencode: a global compaction block would leak keep to Flash/16K models')
    models = (((opencode.get('providers') or {}).get('local-qwen') or {}).get('models')) or {}
    base_models = (((base_opencode.get('providers') or {}).get('local-qwen') or {}).get('models')) or {}
    for name in MOE_MODELS:
        if '(beta)' in str((models.get(name) or {}).get('name', '')):
            errors.append('opencode: %s name still carries "(beta)" (O-100)' % name)
        if (models.get(name) or {}).get('limit') != {'context': MOE_CONTEXT, 'input': FAST_INPUT, 'output': MOE_OUTPUT}:
            errors.append('opencode: %s limit must be context %d / input %d / output %d'
                          % (name, MOE_CONTEXT, FAST_INPUT, MOE_OUTPUT))
    for name, entry in models.items():
        if name not in MOE_MODELS and entry != base_models.get(name):
            errors.append('opencode: non-MoE model %s changed' % name)
        if entry.get('disabled'):
            errors.append('opencode: %s disabled in the base config (only launch overlays disable)' % name)
    if set(models) != set(base_models):
        errors.append('opencode: model set changed')
    return errors


def check_unit(text, release_dir):
    errors = []
    lines = text.splitlines()
    if lines.count(WATCHDOG) != 1 or sum('LOCAL_AI_IMAGE_WATCHDOG_KILL_CLIENT' in l for l in lines) != 1:
        errors.append('unit: exactly one ' + WATCHDOG + ' (installed controller unit only)')
    if 'Restart=on-failure' not in lines:
        errors.append('unit: Restart=on-failure missing')
    for name in ('LOCAL_LLM_CONFIG', 'LOCAL_AI_LEASE'):
        if not any(l.startswith(('Environment="%s=' % name, 'Environment=%s=' % name)) for l in lines):
            errors.append('unit: %s must be set explicitly (audit M4b)' % name)
    if 'WorkingDirectory=' + release_dir not in lines or \
            'ExecStart=/usr/bin/python3 "%s/controller.py"' % release_dir not in lines:
        errors.append('unit: not pointed at the staged release')
    return errors


def launcher_facts(source_dir):
    """Constants and the two MoE overlays, computed by the launcher module itself (isolated child)."""
    code = ('import json,sys; sys.path.insert(0, sys.argv[1]); import launch_local_plan as L; '
            'print(json.dumps({"default": L.DEFAULT_OPENCODE_MODE, "inputs": L.OPENCODE_MODE_INPUT, '
            '"keep": L.MOE_KEEP_TOKENS, "fast": L.apply_moe_preset({}, "fast"), '
            '"deep": L.apply_moe_preset({}, "deep")}))')
    out = subprocess.run([sys.executable, '-I', '-B', '-c', code, str(source_dir)], check=True,
                         capture_output=True, text=True, env={'PATH': os.environ.get('PATH', '/usr/bin')})
    facts = json.loads(out.stdout)
    errors = []
    if facts['default'] != 'fast':
        errors.append('launcher: DEFAULT_OPENCODE_MODE must be fast (O-105)')
    if facts['inputs'] != {'deep': DEEP_INPUT, 'fast': FAST_INPUT}:
        errors.append('launcher: OPENCODE_MODE_INPUT must be deep %d / fast %d' % (DEEP_INPUT, FAST_INPUT))
    return facts, errors


N9_TEMPLATE = r'''#!/usr/bin/python3
"""O-100 N9: chmod-only privacy fix for the inventoried paths (dirs 0700, files 0600).

Default: check only. `--apply` chmods, `--restore` puts the recorded modes back.
Each path must still be the inventoried object (same dev/inode/uid, not a symlink,
the expected type) and at either its recorded mode or the target mode, or the
whole run refuses before anything changes (exit 3). Nothing is deleted, no
service is touched, contents are never read. Receipts go to n9/receipts/.
"""
import json, os, stat, sys, time
from pathlib import Path

BINDING = json.loads(@BINDING@)
root = Path(__file__).resolve().parent


def refuse(message):
    print('N9 REFUSED (nothing changed): ' + message, file=sys.stderr)
    sys.exit(3)


def open_checked(item):
    want_dir = item['target_mode'] == 0o700
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK | (os.O_DIRECTORY if want_dir else 0)
    try:
        fd = os.open(item['path'], flags)
    except OSError as exc:
        refuse('%s: cannot open without following links (%s)' % (item['path'], exc.strerror))
    info = os.fstat(fd)
    kind_ok = stat.S_ISDIR(info.st_mode) if want_dir else stat.S_ISREG(info.st_mode)
    if not kind_ok or (info.st_dev, info.st_ino, info.st_uid) != (item['dev'], item['inode'], item['uid']):
        os.close(fd)
        refuse('%s: not the inventoried object (type/dev/inode/uid changed)' % item['path'])
    return fd, info.st_mode & 0o7777


def plan(direction):
    todo = []
    for item in BINDING['files']:
        fd, mode = open_checked(item)
        os.close(fd)
        source, target = (item['mode'], item['target_mode']) if direction == 'apply' else (item['target_mode'], item['mode'])
        if mode == target:
            continue
        if mode != source:
            refuse('%s: mode %04o is neither the recorded %04o nor the target %04o'
                   % (item['path'], mode, item['mode'], item['target_mode']))
        todo.append((item, target))
    return todo


def main():
    args = sys.argv[1:]
    if args not in ([], ['--check'], ['--apply'], ['--restore']):
        print('usage: n9_chmod.py [--check | --apply | --restore]', file=sys.stderr)
        return 2
    direction = 'restore' if args == ['--restore'] else 'apply'
    todo = plan(direction)
    if args in ([], ['--check']):
        print('N9 check OK: would chmod %d of %d path(s): %s' % (len(todo), len(BINDING['files']),
              ', '.join('%s -> %04o' % (i['path'], t) for i, t in todo) or 'none'))
        return 0
    done = []
    for item, target in todo:
        fd, mode = open_checked(item)
        try:
            os.fchmod(fd, target)
        finally:
            os.close(fd)
        done.append({'path': item['path'], 'from': '%04o' % mode, 'to': '%04o' % target})
    receipts = root / 'receipts'
    receipts.mkdir(mode=0o700, exist_ok=True)
    stamp = time.strftime('%Y%m%dT%H%M%S')
    (receipts / ('n9-%s-%s-%d.json' % (direction, stamp, os.getpid()))).write_text(
        json.dumps({'action': 'n9-' + direction, 'time': stamp, 'changed': done}, indent=2) + '\n')
    print('N9 %s: changed %d path(s); nothing deleted, no service touched' % (direction, len(done)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
'''


def render_n9(inventory):
    files = []
    for item in inventory.get('files') or []:
        target = item.get('target_mode')
        if target not in (0o700, 0o600):
            raise StageError('N9: target_mode must be 0700 (dir) or 0600 (file): %s' % item.get('path'))
        if not Path(str(item.get('path'))).is_absolute():
            raise StageError('N9: absolute paths only')
        files.append({k: item[k] for k in ('path', 'dev', 'inode', 'uid', 'mode', 'target_mode', 'kind')})
    if not files or inventory.get('chmod_only') is not True or inventory.get('applied'):
        raise StageError('N9: inventory must be chmod_only, unapplied and non-empty')
    text = N9_TEMPLATE.replace('@BINDING@', repr(json.dumps({'version': 1, 'files': files}, sort_keys=True)))
    compile(text, 'n9_chmod.py', 'exec')
    return text.encode(), files


def seal(stage):
    stage = Path(stage)
    out = {}
    for path in sorted(stage.rglob('*')):
        rel = path.relative_to(stage).as_posix()
        if path.is_file() and '__pycache__' not in rel.split('/') and \
                not any(rel == s or rel.startswith(s + '/') for s in SEAL_SKIP):
            out[rel] = sha(path.read_bytes())
    return out


def verify(stage):
    stage = Path(stage)
    recorded = json.loads((stage / SEAL).read_text())
    actual = seal(stage)
    bad = sorted(set(recorded['files']) ^ set(actual)) + sorted(
        k for k in recorded['files'] if k in actual and recorded['files'][k] != actual[k])
    if bad:
        raise StageError('stage seal mismatch: ' + ', '.join(bad))
    return len(actual)


def generate(*, base, out, lab_repo, lab_rev, engine_rev=None, server=None, server_source=None,
             oxe=None, oxe_source=None, launcher_path=None, n9_inventory=None, systemctl='systemctl',
             forbidden_roots=None):
    import stage_controller_release as controller_release
    import stage_flash_memory as flash_memory
    base, out = Path(base), Path(out)
    if not out.is_absolute() or out.exists() or out.is_symlink():
        raise StageError('--out must be a fresh absolute path')
    for live in forbidden_roots if forbidden_roots is not None else (Path.home() / '.config', Path.home() / '.local'):
        if out == live or out.is_relative_to(live):
            raise StageError('stage must not be under live configuration: %s' % live)
    if out.is_relative_to(base):
        raise StageError('stage must not be inside the base stage')
    if guard.is_retired(base):
        raise StageError('base stage is retired')
    manifest_raw = (base / 'manifest.json').read_bytes()
    bm = json.loads(manifest_raw)
    if bm.get('installed') is not False:
        raise StageError('base stage must be an uninstalled stage (installed: false)')
    if (server is None) != (server_source is None) or (oxe is None) != (oxe_source is None):
        raise StageError('--server/--oxe each need their --*-source commit')
    for src in (server_source, oxe_source, engine_rev):
        if src is not None and not re.fullmatch(r'[0-9a-f]{40}', src):
            raise StageError('engine sources must be full 40-hex commit ids')

    # 1. live preimages (read once: sha + mode + bytes); the base binding must still hold
    paths = {k: Path(bm['paths'][k]) for k in ('unit', 'runtime', 'opencode', 'cli')}
    paths['launcher'] = Path(launcher_path) if launcher_path else paths['cli'].with_name('launch_local_plan.py')
    live = {k: capture_live(p) for k, p in paths.items()}
    preimages, modes = {}, {}
    for key in ('unit', 'runtime', 'opencode', 'cli'):
        if live[key]['sha256'] != bm['preimage_sha256'][key]:
            raise StageError('%s: live sha256 %s is not the base preimage %s; restage from the live state'
                             % (key, live[key]['sha256'][:12], bm['preimage_sha256'][key][:12]))
        original = (base / 'rollback' / (key + '.original')).read_bytes()
        if sha(original) != live[key]['sha256']:
            raise StageError('%s: base rollback/%s.original is not the live preimage' % (key, key))
        preimages[key] = original
    preimages['launcher'] = live['launcher']['bytes']
    for key in KEYS:
        modes[key] = live[key]['mode']
        if modes[key] & ~0o777 or modes[key] & 0o600 != 0o600:
            raise StageError('%s: unexpected live mode %04o' % (key, modes[key]))

    with tempfile.TemporaryDirectory(prefix='v012-final-src-') as tmp:
        lab_commit, source_dir = export_tree(lab_repo, lab_rev, RUNTIME_SRC, tmp)
        sources = controller_release.release_sources(source_dir)
        facts, errors = launcher_facts(source_dir)
    if errors:
        raise StageError('; '.join(errors))

    # 2. engine artifacts: new server/oxe only when given, else the base files with their own source
    base_server = (base / 'engine' / 'oxe-server').read_bytes()
    base_oxe = (base / 'engine' / 'oxe').read_bytes()
    if sha(base_server) != bm['server_sha256'] or sha(base_oxe) != bm['oxe_sha256']:
        raise StageError('base engine files do not match the base manifest')
    server_bytes = Path(server).read_bytes() if server else base_server
    oxe_bytes = Path(oxe).read_bytes() if oxe else base_oxe
    server_sha, oxe_sha = sha(server_bytes), sha(oxe_bytes)
    old_engine_dir = bm['engine_dir']
    engine_dir = (str(Path(old_engine_dir).parent / ('oxe-v012-final-' + server_sha[:12])) if server else old_engine_dir)
    artifacts = {
        'oxe-server': {'sha256': server_sha, 'source': server_source or bm['engine_source'],
                       'origin': 'supplied' if server else 'carried from base stage (not rebuilt)'},
        'oxe': {'sha256': oxe_sha, 'source': oxe_source or bm['engine_source'],
                'origin': 'supplied' if oxe else 'carried from base stage (not rebuilt)'}}

    # 3. payloads
    source_sha = {name: sha(data) for name, data in sources.items()}
    release_id = sha(json.dumps(source_sha, sort_keys=True).encode())[:12]
    release_dir = str(Path(bm['release_dir']).parent / ('v012-final-' + release_id))
    base_unit = (base / bm['staged_files']['unit']).read_text()
    unit = flash_memory.rewrite_unit(base_unit, release_dir)

    def bind_runtime(raw, server_path_old, server_path_new):
        text = raw.decode().replace(server_path_old, server_path_new).replace(bm['server_sha256'], server_sha)
        obj = same_format(text.encode())
        for alias in MOE_MODELS:
            oxe_cfg = obj['profiles'][alias]['memlat_overrides']['oxe']
            oxe_cfg['validation'] = {'kind': 'v012-final-staged-unqualified', 'engine_source': artifacts['oxe-server']['source'],
                                     'server_sha256': server_sha, 'qualified': False}
        return obj

    base_runtime_raw = (base / 'runtime.json').read_bytes()
    base_runtime = same_format(base_runtime_raw)
    runtime = bind_runtime(base_runtime_raw, old_engine_dir + '/oxe-server', engine_dir + '/oxe-server')
    gate = bind_runtime((base / 'runtime.gate.json').read_bytes(), str(base / 'engine' / 'oxe-server'),
                        str(out / 'engine' / 'oxe-server'))
    base_opencode = same_format((base / 'opencode.json').read_bytes())
    opencode = json.loads(json.dumps(base_opencode))
    opencode.pop('compaction', None)
    for name in MOE_MODELS:
        entry = opencode['providers']['local-qwen']['models'][name]
        if isinstance(entry.get('name'), str):
            entry['name'] = entry['name'].replace(' (beta)', '').replace('(beta)', '').strip()
        opencode['providers']['local-qwen']['models'][name]['limit'] = {
            'context': MOE_CONTEXT, 'input': FAST_INPUT, 'output': MOE_OUTPUT}

    errors = (check_runtime(runtime, base_runtime, bm['profile_requirements'], server_sha)
              + check_runtime(gate, base_runtime, bm['profile_requirements'], server_sha)
              + check_opencode(opencode, base_opencode) + check_unit(unit, release_dir))
    for name, overlay in (('fast', facts['fast']), ('deep', facts['deep'])):
        limit = overlay['providers']['local-qwen']['models']['ornith-35b-a3b-local']['limit']['input']
        if limit != (FAST_INPUT if name == 'fast' else DEEP_INPUT) or overlay['compaction']['keep'] != {'tokens': 8000}:
            errors.append('launcher overlay %s is not the expected preset' % name)
        if not overlay['providers']['local-qwen']['models'][FLASH].get('disabled'):
            errors.append('launcher overlay %s does not isolate Flash' % name)
    if errors:
        raise StageError('; '.join(errors))
    payload = {'unit': unit.encode(), 'runtime': dump_json(runtime), 'opencode': dump_json(opencode),
               'cli': sources['local-ai'], 'launcher': sources['launch_local_plan.py']}
    for key in KEYS:
        if payload[key] == preimages[key] and key != 'cli':
            raise StageError('%s: payload equals the live preimage; nothing to install' % key)

    # 4. write the stage (fresh dir, O_EXCL everywhere)
    n9_script = n9_files = None
    if n9_inventory:
        n9_script, n9_files = render_n9(json.loads(Path(n9_inventory).read_text()))
    old_umask = os.umask(0o077)
    try:
        out.mkdir(mode=0o700)

        def put(name, data, mode=0o600):
            target = out / name
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, mode)
            with os.fdopen(fd, 'wb') as stream:
                stream.write(data)
            os.chmod(target, mode)
        for key in KEYS:
            put(STAGED[key], payload[key], 0o700 if key in ('cli', 'launcher') else 0o600)
            put('rollback/%s.original' % key, preimages[key])
        put('runtime.gate.json', dump_json(gate))
        put('opencode-mode-overlay.fast.json', dump_json(facts['fast']))
        put('opencode-mode-overlay.deep.json', dump_json(facts['deep']))
        put('engine/oxe-server', server_bytes, 0o700)
        put('engine/oxe', oxe_bytes, 0o700)
        for name, data in sources.items():
            put('runtime-source/' + name, data, 0o700 if name in flash_memory.RUNTIME_FILES else 0o600)
        for name in CARRY:
            src = base / name
            if src.is_dir():
                for f in sorted(p for p in src.rglob('*') if p.is_file()):
                    put(name + '/' + f.relative_to(src).as_posix(), f.read_bytes())
            elif src.is_file():
                put(name, src.read_bytes())
        for name in ('install_staged_release.py', 'stage_rollback_guard.py', 'stage_v012_final.py'):
            put('tools/' + name, (TOOLS / name).read_bytes(), 0o700 if name != 'stage_rollback_guard.py' else 0o600)
        if n9_script:
            put('n9/n9_chmod.py', n9_script, 0o700)
            put('n9/inventory.json', Path(n9_inventory).read_bytes())
        doc = 'Restore the exact pre-change unit, runtime, opencode, cli and launcher (with their modes), then restart the controller.'
        message = 'Restored the previous unit, runtime, opencode, cli and launcher (modes kept); no model loaded.'
        put('rollback/rollback.py', guard.render_rollback(paths, payload, preimages, staged_files=STAGED, doc=doc,
                                                          message=message, systemctl=systemctl, modes=modes), 0o700)
        probe = ('import json,sys; sys.path.insert(0, sys.argv[1]); import oxe_runtime as R; '
                 'print(json.dumps({"state_cache": bool(R.engine_has_state_cache(sys.argv[2])), '
                 '"state_preload": bool(R.engine_has_state_preload(sys.argv[2]))}))')
        try:  # the staged release's own probe on the staged binary, in an isolated child
            capabilities = json.loads(subprocess.run(
                [sys.executable, '-I', '-B', '-c', probe, str(out / 'runtime-source'), str(out / 'engine/oxe-server')],
                check=True, capture_output=True, text=True, timeout=120,
                env={'PATH': os.environ.get('PATH', '/usr/bin'), 'HOME': os.environ.get('HOME', '/nonexistent')}).stdout)
        except (subprocess.SubprocessError, ValueError) as exc:  # recorded, never silently true
            capabilities = {'probe_error': type(exc).__name__}
        put('CAPABILITIES.json', dump_json(capabilities))
        try:
            gen_commit = git(TOOLS.parent, 'rev-parse', 'HEAD').decode().strip()
            gen_dirty = bool(git(TOOLS.parent, 'status', '--porcelain', '--', 'tools').strip())
        except (subprocess.CalledProcessError, OSError):
            gen_commit, gen_dirty = None, None
        diff = {k: ('changed' if payload[k] != preimages[k] else 'unchanged') for k in KEYS}
        manifest = {
            **guard.manifest_binding(paths, payload, preimages, STAGED, modes=modes),
            'installed': False, 'qualified': False, 'models_started': False,
            'stage': str(out), 'base_stage': str(base), 'base_manifest_sha256': sha(manifest_raw),
            'created': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
            'lab_source': lab_commit, 'engine_source': engine_rev, 'engine_artifacts': artifacts,
            'server_sha256': server_sha, 'oxe_sha256': oxe_sha, 'engine_dir': engine_dir,
            'release_dir': release_dir, 'release_id': release_id, 'runtime_source_sha256': source_sha,
            'unit_sha256': sha(payload['unit']), 'runtime_json_changed': payload['runtime'] != preimages['runtime'],
            'payload_vs_live': diff,
            'payload_vs_base_stage': {k: ('same' if (base / bm['staged_files'][k]).read_bytes() == payload[k] else 'changed')
                                      for k in ('unit', 'runtime', 'opencode', 'cli')},
            'profile_requirements': bm['profile_requirements'],
            'model_sha_verification_receipt': bm.get('model_sha_verification_receipt'),
            'opencode': {'canonical_moe_input': FAST_INPUT, 'deep_override_input': DEEP_INPUT,
                         'launcher_default_mode': facts['default'], 'global_compaction': 'absent',
                         'moe_keep_tokens_launch_overlay_only': facts['keep']},
            'invariants_checked': ['flash cache_pref 18 / prefix store on / PSI off / sidecar bound / profile == base',
                                   'MoE ctx 131072 / checkpoints 8 / session RAM / mtp-max-ctx / park 512 MiB',
                                   'cache 64e9 bytes / ttl 604800', 'model sha256 == base', 'no "(beta)" label',
                                   'no oxi.text_encoder (native TE inert)', 'watchdog env once in controller unit',
                                   'no global compaction in opencode.json'],
            'O100': {'runtime_label': True, 'watchdog_kill_client_installed_only': True,
                     'studio_restart_recheck_required_at_install': True,
                     'N9_chmod_step': 'n9/n9_chmod.py' if n9_script else None,
                     'N9_paths': len(n9_files) if n9_files else 0, 'N9_chmod_applied': False},
            'native_text_encoder': 'inert: not part of v0.1.2 (no oxi.text_encoder in any payload)',
            'compaction_overlay_scope': 'MoE launch overlay only (launch_local_plan.apply_moe_preset); never global',
            'carried_from_base': [n for n in CARRY if (base / n).exists()],
            'not_carried': ['gate-tools-* (bound to older lab sources; take gate tools from lab_source)'],
            'install': ['copy runtime-source/ to release_dir (and engine/ to engine_dir when absent)',
                        'tools/install_staged_release.py <stage>   (dry run; --apply after the gates)',
                        'daemon-reload + restart (root only, as for every stage)',
                        'n9/n9_chmod.py --check, then --apply (separate step)'],
            'rollback': 'rollback/rollback.py (restores bytes AND the captured preimage modes)',
            'generator': {'path': 'tools/stage_v012_final.py', 'commit': gen_commit, 'tools_dirty': gen_dirty,
                          'sha256': sha((TOOLS / 'stage_v012_final.py').read_bytes())},
        }
        put('manifest.json', dump_json(manifest))
        put('PREPARATION.json', dump_json({
            'stage': str(out), 'lab_source': lab_commit, 'engine_source': engine_rev,
            'server_sha256': server_sha, 'oxe_sha256': oxe_sha, 'release_dir': release_dir,
            'supersedes_preparation': str(base), 'qualified': False, 'installed': False,
            'binary_binding': 'per-file sources in manifest.engine_artifacts; no install'}))
        put(SEAL, dump_json({'note': 'sha256 of every stage file at generation; receipts excluded',
                             'files': seal(out)}), 0o400)
        for path in out.rglob('*'):  # immutable payload: read-only files, writable dirs for receipts only
            if path.is_file() and not path.is_symlink():
                os.chmod(path, 0o500 if path.stat().st_mode & 0o100 else 0o400)
    finally:
        os.umask(old_umask)
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--verify', type=Path, metavar='STAGE', help='only recheck a written stage seal')
    ap.add_argument('--base', type=Path)
    ap.add_argument('--out', type=Path)
    ap.add_argument('--lab-repo', type=Path)
    ap.add_argument('--lab-rev', help='lab commit (e.g. the FINAL-3 candidate tip)')
    ap.add_argument('--engine-rev', help='FINAL engine commit (recorded; full sha)')
    ap.add_argument('--server', type=Path); ap.add_argument('--server-source')
    ap.add_argument('--oxe', type=Path); ap.add_argument('--oxe-source')
    ap.add_argument('--launcher-path', type=Path, help='live launcher (default: next to the live CLI)')
    ap.add_argument('--n9-inventory', type=Path)
    ap.add_argument('--systemctl', default='systemctl', help='baked into rollback.py (tests use a fake)')
    args = ap.parse_args(argv)
    try:
        if args.verify:
            print('seal OK: %d files' % verify(args.verify))
            return 0
        if not all((args.base, args.out, args.lab_repo, args.lab_rev)):
            ap.error('--base, --out, --lab-repo and --lab-rev are required')
        m = generate(base=args.base, out=args.out, lab_repo=args.lab_repo, lab_rev=args.lab_rev,
                     engine_rev=args.engine_rev, server=args.server, server_source=args.server_source,
                     oxe=args.oxe, oxe_source=args.oxe_source, launcher_path=args.launcher_path,
                     n9_inventory=args.n9_inventory, systemctl=args.systemctl)
    except (StageError, guard.GuardError, ValueError, KeyError, OSError, subprocess.CalledProcessError) as exc:
        print('REFUSED: %s' % exc, file=sys.stderr)
        return 2
    print(json.dumps({k: m[k] for k in ('stage', 'lab_source', 'engine_source', 'server_sha256', 'oxe_sha256',
                                        'release_dir', 'preimage_sha256', 'preimage_mode', 'installed_sha256')},
                     indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
