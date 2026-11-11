#!/usr/bin/env python3
"""Stage the MEMLAT-only Rust Flash (oxe --engine flash) switch for one alias; never install.

Writes a fresh staging directory with the patched runtime.json, a unit that
points at a content-addressed runtime release and sets LOCAL_AI_ENABLE_OXE_FLASH=1
(an existing LOCAL_AI_ENABLE_OXE=1 line is kept), the runtime source to copy into
that release, a copy of the frozen engine binary (engine/oxe-server), exact
rollback preimages plus a one-command rollback.py, a manual install.py and a
manifest.json. In the default qos mode it also writes system/ (the root part of the
idle-QoS hook). The runtime.json names the engine at its content-addressed install
path under --engine-releases-root. It never edits live files, installs anything,
starts services, or reads model weights: the Flash shards and expert sidecar are
only stat()ed.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys

ROOT = Path(__file__).resolve().parents[1]
# Root part of the idle-QoS hook (helper, system unit template, polkit rule, README); copied in qos mode.
QOS_PACKAGING = ROOT / 'packaging' / 'flash-qos'
QOS_HELPER, QOS_UNIT, QOS_POLKIT = 'local-ai-flash-qos', 'local-ai-flash-qos@.service', '49-local-ai-flash-qos.rules'
IO_MODES = ('qos', 'spin')
IDLE_QOS = {'unit': 'local-ai-flash-qos@{pid}.service', 'cpu': 8, 'latency_us': 100}
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tools')]
import stage_rollback_guard as guard  # noqa: E402
from snapdragon_ai_lab.runtime import bigmoe_runtime, oxe_runtime, profile_policy  # noqa: E402

ALIAS = 'qwen3.8-flash-next-local'
# Pinned after the Rust Flash qualification ticket passes; None refuses staging.
QUALIFIED_FLASH_SERVER_SHA = None
CONTEXT = 16384
THREADS = 8
CACHE_GIB = 20
ADMITTED_BUDGET_BYTES = 28 * 1024**3
# 4 checkpoints x 128 MiB (118 MB measured per checkpoint, rounded up)
SESSION_RESERVE_BYTES = 4 * 128 * 1024**2
ENGINE_LIMIT_BYTES = 256 * 1024**2
FLASH_ARGS = ['--engine', 'flash', '--model', '{model}', '--alias', '{alias}', '--ctx-size', '{context}',
              # IO thread on core 8; compute threads avoid it (see ENVIRONMENT GOMP_CPU_AFFINITY).
              '--flash-threads', '8', '--flash-threads-batch', '11', '--flash-io-cpu', '8',
              '--flash-cache-gib', '20', '--ctx-checkpoints', '4',
              '--session-ram-bytes', '0', '--reasoning', 'off']
# NO warm-up in this package (2026-11-10): `Stream::warm` (hot-set restore) queues prefetch-class reads, and a
# demand hit on a pending prefetch is re-issued into the SAME slot, so a decode can compute on a buffer that is
# being rewritten (found by the Flash-prefill lane; root fix assigned to F-IO). The server calls `Stream::warm`
# only from `--flash-hotset` (process start) and from loading a stored state while the cache is cold
# (`--prefix-cache`, `--session-cache`, unparking a session). None of those flags may appear in FLASH_ARGS until
# the fix is qualified; RAM parking is already off. Cost: after a reload the system prompt is prefilled again.
# FLASH-PREFIX-1: the prefix store is switched on only through the profile key oxe_flash.prefix_store (the controller
# adds the flags, oxe_runtime refuses them raw), so this argv check stays as it is.
WARMUP_FLAGS = ('--flash-hotset', '--prefix-cache', '--prefix-cache-min-tokens', '--session-cache')
# Affinity: decode's 8 threads take cores 0-7, prefill's 11 threads take 0-7 and 9-11, and core 8 is the
# IO thread's (--flash-io-cpu 8), so no compute thread shares the IO core.
ENVIRONMENT = {'GOMP_CPU_AFFINITY': '0-7 9-11', 'OMP_WAIT_POLICY': 'ACTIVE', 'OMP_PROC_BIND': 'TRUE',
               'LOCAL_AI_ENGINE_LEDGER': '1'}
# Default ('qos') parks the IO thread by default and the idle-QoS hook covers its idle time. The fallback ('spin',
# --io-mode spin) keeps OXF_IO (F-IO lineage, M3-FLASH-IO.md): one IO thread that does not park inside a decode.
# An engine without F-IO ignores the variable.
SPIN_ENVIRONMENT = {'OXF_IO': 'spins=1000000'}
ENABLE_LINE = 'Environment="%s=1"' % profile_policy.OXE_FLASH_ENABLE_ENV
RUNTIME_FILES = ('local-ai', 'studio')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_bound(path, wanted, limit=8 * 1024**2):
    path = Path(path)
    if not path.is_absolute():
        raise ValueError('absolute_path_required')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    with os.fdopen(fd, 'rb') as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_size > limit:
            raise ValueError('bounded_owned_regular_file_required')
        data = stream.read(limit + 1)
    if wanted is not None and sha(data) != wanted:
        raise ValueError('identity_changed: ' + path.name)
    return data


def engine_install_path(engine_releases_root, server_sha256):
    """Content-addressed install path of the frozen engine; nothing is written here by staging."""
    return Path(engine_releases_root) / ('oxe-flash-' + server_sha256[:12]) / 'oxe-server'


def check_no_warmup(args):
    if any(arg.split('=', 1)[0] in WARMUP_FLAGS for arg in args) or _session_ram(args) != '0':
        raise ValueError('flash_warmup_paths_must_stay_off')


def _session_ram(args):
    return args[args.index('--session-ram-bytes') + 1] if '--session-ram-bytes' in args else None


def flash_override(server_binary, server_sha256, *, sidecar_path, sidecar_bytes, measured_peak_bytes, cache_gib,
                   fallback_peak_bytes=None, budget_bytes=ADMITTED_BUDGET_BYTES, qualified=True, io_mode='qos',
                   minimum_available_bytes=None):
    if io_mode not in IO_MODES:
        raise ValueError('flash_io_mode_invalid')
    args = [str(cache_gib) if i and FLASH_ARGS[i - 1] == '--flash-cache-gib' else arg
            for i, arg in enumerate(FLASH_ARGS)]
    check_no_warmup(args)
    # the engine reads the staged sidecar (it must match sidecar_path: the controller stats that file at load)
    args += ['--flash-sidecar', str(sidecar_path)]
    extra = {} if fallback_peak_bytes is None else {'fallback': {'cache_gib': 18, 'admission': {
        'measured_peak_bytes': fallback_peak_bytes, 'admitted_budget_bytes': budget_bytes}}}
    # qos: the idle-QoS hook switches on (the root helper owns the knob); spin: the fallback, no hook.
    qos = {'idle_qos': copy.deepcopy(IDLE_QOS)} if io_mode == 'qos' else {}
    environment = {**ENVIRONMENT, **({} if io_mode == 'qos' else SPIN_ENVIRONMENT)}
    kind = 'oxe-rust-flash-qualification' if qualified else 'oxe-rust-flash-candidate'
    admission = {'measured_peak_bytes': measured_peak_bytes, 'admitted_budget_bytes': budget_bytes}
    if minimum_available_bytes is not None:
        admission['minimum_available_bytes'] = minimum_available_bytes
    return {**extra, **qos, 'server_binary': server_binary, 'server_sha256': server_sha256, 'threads': THREADS,
            'oxe_args': args, 'environment': environment,
            'session_reserve_bytes': SESSION_RESERVE_BYTES,
            'validation': {'kind': kind, 'server_sha256': server_sha256,
                           'report': 'docs/FLASH-RUST-PRODUCTION.md', 'qualified': qualified},
            'cache_gib': cache_gib, 'sidecar_path': str(sidecar_path), 'sidecar_bytes': sidecar_bytes,
            'admission': admission}


def add_flash_override(runtime, server_binary, server_sha256, *, sidecar_path, measured_peak_bytes,
                       cache_gib=CACHE_GIB, qualified_sha=None, candidate_sha256=None, fallback_peak_bytes=None,
                       io_mode='qos', minimum_available_bytes=None):
    """Return a detached runtime with one nested memlat 'oxe_flash' override for the Flash alias.

    The binary must match the qualified hash (qualified=True) or, when given, exactly
    candidate_sha256 (qualified=False, for a staged but unqualified candidate).
    """
    if qualified_sha is None:
        qualified_sha = QUALIFIED_FLASH_SERVER_SHA
    if qualified_sha is not None and server_sha256 == qualified_sha:
        qualified = True
    elif candidate_sha256 is not None and server_sha256 == candidate_sha256:
        qualified = False
    else:
        raise ValueError('unqualified_rust_flash_server')
    if cache_gib not in (16, 18, 20, 21):
        raise ValueError('flash_cache_gib_must_be_16_18_20_or_21')
    info = os.stat(sidecar_path)
    if not stat.S_ISREG(info.st_mode):
        raise ValueError('flash_sidecar_must_be_regular_file')
    updated = copy.deepcopy(runtime)
    profile = updated.get('profiles', {}).get(ALIAS)
    if not isinstance(profile, dict) or profile.get('backend_kind') != bigmoe_runtime.KIND:
        raise ValueError('flash_profile_required: ' + ALIAS)
    if profile.get('context_size') != CONTEXT:
        raise ValueError('flash_16k_profile_required: ' + ALIAS)
    shards = (profile.get('bigmoe') or {}).get('shards')
    if not isinstance(shards, list) or [(row.get('bytes'), row.get('sha256')) for row in shards] != list(bigmoe_runtime.PINNED):
        raise ValueError('flash_shards_must_match_pinned_identity')
    overrides = profile.setdefault('memlat_overrides', {})
    if 'oxe_flash' in overrides:
        raise ValueError('oxe_flash_override_already_present: ' + ALIAS)
    overrides['oxe_flash'] = flash_override(str(server_binary), server_sha256, sidecar_path=sidecar_path,
                                            sidecar_bytes=info.st_size, measured_peak_bytes=measured_peak_bytes,
                                            cache_gib=cache_gib, fallback_peak_bytes=fallback_peak_bytes,
                                            # the profile's own 29 GiB scope (tools/stage_flash_memory.py), if staged
                                            budget_bytes=oxe_runtime.flash_limits(profile)[1], qualified=qualified,
                                            io_mode=io_mode, minimum_available_bytes=minimum_available_bytes)
    # Validate both resolutions: stock when the Flash switch is off, Rust when it is on.
    merged = {**{k: v for k, v in updated.items() if k != 'profiles'}, **profile}
    if profile_policy.resolve(merged, available=True, oxe_flash_enabled=False) != merged:
        raise ValueError('flash_stock_resolution_changed: ' + ALIAS)
    effective = profile_policy.resolve(merged, available=True, oxe_flash_enabled=True)
    if effective['backend_kind'] != 'oxe' or effective['prefix_cache']['enabled'] is not False:
        raise ValueError('flash_resolution_failed: ' + ALIAS)
    oxe_runtime._validate(effective)
    return updated


def rewrite_unit(text, release_dir):
    """Point WorkingDirectory/ExecStart at release_dir and enable the Flash switch once."""
    if profile_policy.OXE_FLASH_ENABLE_ENV in text:
        raise ValueError('unit_already_mentions_oxe_flash')
    lines, done = [], set()
    for line in text.splitlines():
        if line.startswith('WorkingDirectory='):
            line = 'WorkingDirectory=' + release_dir; done.add('cwd')
        elif line.startswith('ExecStart='):
            line = 'ExecStart=/usr/bin/python3 "%s/controller.py"' % release_dir; done.add('exec')
        lines.append(line)
        if line.startswith('Environment="LOCAL_LLM_CONFIG='):
            lines.append(ENABLE_LINE); done.add('env')
    if done != {'cwd', 'exec', 'env'}:
        raise ValueError('unexpected_unit_layout')
    return '\n'.join(lines) + '\n'


ROLLBACK_DOC = 'Restore the pre-Flash-switch runtime.json and unit, then restart.'
ROLLBACK_MESSAGE = ('Restored the previous runtime and controller; Flash is back on BigMoeOnEdge; no model loaded. '
                    'The engine and release directories were left in place (content-addressed, harmless).')

# Placeholders are replaced with repr() values, so the template needs no %-escaping.
INSTALL = '''#!/usr/bin/python3
"""Install the staged Flash switch by hand; loads no model.

Exit 3 (nothing changed) when the live runtime.json or unit is no longer the staged
preimage. Exit 4 when a different engine already sits at the install path. Exit 5
when a staged file changed. Exit 6 when the staged engine is an unqualified candidate. The engine and release directories are content-addressed.
"""
from pathlib import Path
import hashlib, os, shutil, subprocess, sys

SYSTEMCTL = 'systemctl'
root = Path(__file__).resolve().parent
LIVE = {'runtime': Path(__RUNTIME_PATH__), 'unit': Path(__UNIT_PATH__)}
PREIMAGE = {'runtime': __RUNTIME_PREIMAGE__, 'unit': __UNIT_PREIMAGE__}
STAGED = {'runtime': root / 'runtime.json', 'unit': root / 'local-ai-chat.service'}
STAGED_SHA = {'runtime': __RUNTIME_STAGED__, 'unit': __UNIT_STAGED__}
ENGINE_SRC = root / 'engine' / 'oxe-server'
ENGINE_SHA = __ENGINE_SHA__
QUALIFIED = __QUALIFIED__
ENGINE_ROOT = Path(__ENGINE_ROOT__)
ENGINE_DIR = Path(__ENGINE_DIR__)
ENGINE_BIN = ENGINE_DIR / 'oxe-server'
RELEASE_SRC = root / 'runtime-source'
RELEASE_DIR = Path(__RELEASE_DIR__)
ROLLBACK = root / 'rollback' / 'rollback.py'
ROOT_INSTALL = __ROOT_INSTALL__


def digest(path):
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def fail(code, message):
    print(message, file=sys.stderr)
    sys.exit(code)


def replace_bytes(target, source, mode):
    tmp = target.with_name(target.name + '.install')
    tmp.unlink(missing_ok=True)
    shutil.copyfile(source, tmp)
    os.chmod(tmp, mode)
    os.replace(tmp, target)


def main():
    if not QUALIFIED and '--allow-unqualified-candidate' not in sys.argv[1:]:
        fail(6, 'this stage holds an UNQUALIFIED candidate engine; restage after the qualification ticket '
                'passes (or pass --allow-unqualified-candidate for a supervised test); nothing changed')
    for name, path in LIVE.items():
        if digest(path) != PREIMAGE[name]:
            fail(3, 'live ' + name + ' no longer matches the staged preimage; nothing changed')
    for name, path in STAGED.items():
        if digest(path) != STAGED_SHA[name]:
            fail(5, 'staged ' + path.name + ' changed since staging; nothing changed')
    if digest(ENGINE_SRC) != ENGINE_SHA:
        fail(5, 'staged engine changed since staging; nothing changed')
    if ENGINE_BIN.exists():
        if digest(ENGINE_BIN) != ENGINE_SHA:
            fail(4, 'a different engine already sits at ' + str(ENGINE_BIN) + '; nothing changed')
    else:
        ENGINE_ROOT.mkdir(parents=True, exist_ok=True)
        ENGINE_DIR.mkdir(mode=0o700, exist_ok=True)
        replace_bytes(ENGINE_BIN, ENGINE_SRC, 0o700)
    if not RELEASE_DIR.exists():
        RELEASE_DIR.parent.mkdir(parents=True, exist_ok=True)
        tmp = RELEASE_DIR.with_name(RELEASE_DIR.name + '.install')
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.copytree(RELEASE_SRC, tmp)
        os.rename(tmp, RELEASE_DIR)
    subprocess.run([SYSTEMCTL, '--user', 'stop', 'local-ai-chat.service'], check=True)
    for name, path in LIVE.items():
        replace_bytes(path, STAGED[name], 0o600)
    subprocess.run([SYSTEMCTL, '--user', 'daemon-reload'], check=True)
    subprocess.run([SYSTEMCTL, '--user', 'start', 'local-ai-chat.service'], check=True)
    print('Installed the staged Flash switch; no model loaded.')
    print('Rollback: python3 ' + str(ROLLBACK))
    if ROOT_INSTALL:
        print('Root part (optional): sudo ' + str(root / ROOT_INSTALL) + '; without it the controller logs the '
              'idle-QoS hook as unavailable and the engine runs with default idle states.')


if __name__ == '__main__':
    main()
'''

# Root part of the idle-QoS hook: installs three files and starts nothing (run once, as root).
INSTALL_ROOT = r'''#!/bin/sh
# Root part of the Flash idle-QoS hook. Installs the helper, the system unit template and the polkit rule.
# Starts no unit. Temporary rendered files are removed on exit. Removal: rollback-root.sh in this directory.
set -eu
if [ "$(id -u)" -ne 0 ]; then
    echo "install-root.sh must run as root: sudo $0 [--user USER]" >&2
    exit 1
fi
if [ "$#" -eq 0 ]; then
    installing_user=${SUDO_USER:-}
elif [ "$#" -eq 2 ] && [ "$1" = --user ]; then
    installing_user=$2
else
    echo "usage: sudo $0 [--user USER]" >&2
    exit 2
fi
if ! printf '%s\n' "$installing_user" | LC_ALL=C grep -Eq '^[A-Za-z_][A-Za-z0-9_.-]*[$]?$'; then
    echo "installing user must be a valid account name (use --user USER when root)" >&2
    exit 2
fi
account=$(getent passwd "$installing_user") || {
    echo "installing user was not found by getent" >&2; exit 2;
}
installing_uid=$(id -u -- "$installing_user")
canonical_user=$(id -un -- "$installing_user")
case "$installing_uid" in ''|*[!0-9]*|0) echo "installing user must have a non-root numeric UID" >&2; exit 2;; esac
if [ "$canonical_user" != "$installing_user" ] ||
   [ "$(printf '%s\n' "$account" | cut -d: -f1)" != "$installing_user" ] ||
   [ "$(printf '%s\n' "$account" | cut -d: -f3)" != "$installing_uid" ]; then
    echo "getent/id disagree about the installing user" >&2
    exit 2
fi
dir=$(dirname "$(readlink -f "$0")")
umask 077
rendered=$(mktemp -d)
trap 'rm -rf -- "$rendered"' EXIT HUP INT TERM
python3 -I - "$dir" "$installing_user" "$installing_uid" "$rendered" <<'PYQOS'
import json, re, sys
from pathlib import Path
source, user, uid, output = Path(sys.argv[1]), sys.argv[2], sys.argv[3], Path(sys.argv[4])
if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_.-]*\$?', user) or not uid.isdecimal() or int(uid) <= 0:
    raise SystemExit('invalid installing user identity')
for name, original, replacement in (
    ('local-ai-flash-qos', 'ALLOWED_UID = None', 'ALLOWED_UID = ' + str(int(uid))),
    ('49-local-ai-flash-qos.rules', 'var ALLOWED_USER = null;', 'var ALLOWED_USER = ' + json.dumps(user) + ';'),
):
    text = (source / name).read_text()
    if text.count(original) != 1:
        raise SystemExit('unconfigured QoS template required: ' + name)
    (output / name).write_text(text.replace(original, replacement))
PYQOS
install -D -o root -g root -m 0755 "$rendered/local-ai-flash-qos" /usr/local/libexec/local-ai-flash-qos
install -o root -g root -m 0644 "$dir/local-ai-flash-qos@.service" /etc/systemd/system/local-ai-flash-qos@.service
install -o root -g root -m 0644 "$rendered/49-local-ai-flash-qos.rules" /etc/polkit-1/rules.d/49-local-ai-flash-qos.rules
systemctl daemon-reload
echo "Installed local-ai-flash-qos for $installing_user (uid $installing_uid); no unit started. Removal: rollback-root.sh"
'''

ROLLBACK_ROOT = '''#!/bin/sh
# Root part of the Flash idle-QoS hook: stop its units, reset the knob, remove the three installed files.
# Run before the user rollback (rollback/rollback.py).
set -eu
if [ "$(id -u)" -ne 0 ]; then
    echo "rollback-root.sh must run as root: sudo $0" >&2
    exit 1
fi
systemctl stop 'local-ai-flash-qos@*.service' || true
if [ -e /usr/local/libexec/local-ai-flash-qos ]; then
    /usr/local/libexec/local-ai-flash-qos reset || true
fi
rm -f /usr/local/libexec/local-ai-flash-qos /etc/systemd/system/local-ai-flash-qos@.service /etc/polkit-1/rules.d/49-local-ai-flash-qos.rules
systemctl daemon-reload
echo "Removed local-ai-flash-qos (helper, system unit template, polkit rule). The user rollback is separate: rollback/rollback.py"
'''


def render(template, values):
    for key, value in values.items():
        template = template.replace('__%s__' % key, repr(value))
    return template


def qos_packaging_files():
    """Regular files of packaging/flash-qos by name; the three installed files must all be present."""
    if not QOS_PACKAGING.is_dir():
        raise ValueError('flash_qos_packaging_missing')
    files = {}
    for path in sorted(QOS_PACKAGING.iterdir()):
        if stat.S_ISREG(os.lstat(path).st_mode):
            files[path.name] = path.read_bytes()
    if not {QOS_HELPER, QOS_UNIT, QOS_POLKIT} <= set(files):
        raise ValueError('flash_qos_packaging_missing')
    return files


def stage(destination, *, runtime_path, runtime_sha, unit_path, unit_sha, server_binary, releases_root,
          engine_releases_root, sidecar_path, measured_peak_bytes, cache_gib=CACHE_GIB, qualified_sha=None,
          candidate_sha256=None, fallback_peak_bytes=None, io_mode='qos', minimum_available_bytes=None):
    destination, releases_root, engine_releases_root = Path(destination), Path(releases_root), Path(engine_releases_root)
    if not destination.is_absolute() or destination.exists() or not releases_root.is_absolute():
        raise ValueError('fresh_absolute_stage_and_releases_root_required')
    if not engine_releases_root.is_absolute():
        raise ValueError('absolute_engine_releases_root_required')
    for live in (Path.home() / '.config', Path.home() / '.local'):
        if destination == live or destination.is_relative_to(live):
            raise ValueError('stage_must_not_be_live_configuration')
    runtime_raw = read_bound(runtime_path, runtime_sha)
    unit_raw = read_bound(unit_path, unit_sha)
    engine_raw = read_bound(server_binary, None, limit=ENGINE_LIMIT_BYTES)
    server_sha = sha(engine_raw)
    install_path = engine_install_path(engine_releases_root, server_sha)
    updated = add_flash_override(json.loads(runtime_raw), str(install_path), server_sha,
                                 sidecar_path=sidecar_path, measured_peak_bytes=measured_peak_bytes,
                                 cache_gib=cache_gib, qualified_sha=qualified_sha,
                                 candidate_sha256=candidate_sha256, fallback_peak_bytes=fallback_peak_bytes,
                                 io_mode=io_mode, minimum_available_bytes=minimum_available_bytes)
    # Checked before anything is written: a qos stage without its root package is refused outright.
    qos_files = qos_packaging_files() if io_mode == 'qos' else None
    qualified = updated['profiles'][ALIAS]['memlat_overrides']['oxe_flash']['validation']['qualified']
    source_dir = ROOT / 'src/snapdragon_ai_lab/runtime'
    sources = {p.name: p.read_bytes() for p in sorted(source_dir.iterdir())
               if p.is_file() and (p.suffix == '.py' or p.name in RUNTIME_FILES)}
    release_id = sha(json.dumps({k: sha(v) for k, v in sources.items()}, sort_keys=True).encode())[:12]
    release_dir = str(releases_root / ('flash-oxe-production-' + release_id))
    unit = rewrite_unit(unit_raw.decode(), release_dir)
    runtime_text = (json.dumps(updated, indent=2) + '\n').encode()
    unit_bytes = unit.encode()
    destination.mkdir(mode=0o700)

    def put(name, data, mode=0o600):
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, mode)
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
    put('runtime.json', runtime_text)
    put('local-ai-chat.service', unit_bytes)
    put('engine/oxe-server', engine_raw, 0o700)
    for name, data in sources.items():
        put('runtime-source/' + name, data, 0o700 if name in RUNTIME_FILES else 0o600)
    put('rollback/runtime.original', runtime_raw)
    put('rollback/unit.original', unit_raw)
    # The shared guard (W4-3, F77): rollback refuses a stale, tampered or retired stage.
    paths = {'runtime': Path(runtime_path), 'unit': Path(unit_path)}
    payload = {'runtime': runtime_text, 'unit': unit_bytes}
    originals = {'runtime': runtime_raw, 'unit': unit_raw}
    put('rollback/rollback.py', guard.render_rollback(paths, payload, originals, doc=ROLLBACK_DOC,
                                                      message=ROLLBACK_MESSAGE), 0o700)
    root_install = None
    if qos_files is not None:
        root_install = 'system/install-root.sh'
        for name, data in qos_files.items():
            put('system/' + name, data, 0o700 if name == QOS_HELPER else 0o600)
        put('system/install-root.sh', INSTALL_ROOT.encode(), 0o700)
        put('system/rollback-root.sh', ROLLBACK_ROOT.encode(), 0o700)
    put('install.py', render(INSTALL, {
        'RUNTIME_PATH': str(runtime_path), 'UNIT_PATH': str(unit_path),
        'RUNTIME_PREIMAGE': sha(runtime_raw), 'UNIT_PREIMAGE': sha(unit_raw),
        'RUNTIME_STAGED': sha(runtime_text), 'UNIT_STAGED': sha(unit_bytes),
        'ENGINE_SHA': server_sha, 'QUALIFIED': qualified, 'ENGINE_ROOT': str(engine_releases_root),
        'ENGINE_DIR': str(install_path.parent), 'RELEASE_DIR': release_dir,
        'ROOT_INSTALL': root_install}).encode(), 0o700)
    manifest = {**guard.manifest_binding(paths, payload, originals), 'stage': str(destination), 'release_dir': release_dir, 'alias': ALIAS, 'io_mode': io_mode,
                'root_install': root_install,
                'server_sha256': server_sha, 'engine_install_path': str(install_path),
                'engine_source': str(server_binary), 'cache_gib': cache_gib,
                'fallback_cache_gib': None if fallback_peak_bytes is None else 18,
                'measured_peak_bytes': measured_peak_bytes, 'fallback_measured_peak_bytes': fallback_peak_bytes,
                'minimum_available_bytes': minimum_available_bytes,
                'session_reserve_bytes': SESSION_RESERVE_BYTES, 'qualified': qualified, 'warmup': False,
                'runtime_sha256': {'preimage': sha(runtime_raw), 'staged': sha(runtime_text)},
                'unit_sha256': {'preimage': sha(unit_raw), 'staged': sha(unit_bytes)},
                'installed': False, 'models_started': False}
    put('manifest.json', (json.dumps(manifest, indent=2, sort_keys=True) + '\n').encode())
    return manifest


def sha256_hex(text):
    if not re.fullmatch(r'[0-9a-f]{64}', text):
        raise argparse.ArgumentTypeError('lowercase_sha256_hex_required')
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--runtime-sha256', required=True)
    parser.add_argument('--unit', type=Path, required=True)
    parser.add_argument('--unit-sha256', required=True)
    parser.add_argument('--server-binary', type=Path, required=True,
                        help='frozen engine build output (the source; copied into the stage as engine/oxe-server)')
    parser.add_argument('--engine-releases-root', type=Path, required=True,
                        help='absolute root; the engine installs at <root>/oxe-flash-<sha12>/oxe-server')
    parser.add_argument('--releases-root', type=Path, required=True)
    parser.add_argument('--sidecar', type=Path, required=True)
    parser.add_argument('--measured-peak-bytes', type=int, required=True)
    parser.add_argument('--minimum-available-bytes', type=int, help='optional measured host admission floor')
    parser.add_argument('--cache-gib', type=int, choices=(16, 18, 20, 21), default=CACHE_GIB)
    parser.add_argument('--fallback-measured-peak-bytes', type=int,
                        help='measured peak at an 18 GiB cache: the controller starts with 18 when 20 is not admissible')
    parser.add_argument('--candidate-sha256', type=sha256_hex,
                        help='stage exactly this engine hash as an unqualified candidate (qualified false)')
    parser.add_argument('--io-mode', choices=IO_MODES, default='qos',
                        help='qos (default): default IO parking plus the idle-QoS hook and system/ root part; '
                             'spin: the fallback, OXF_IO=spins=1000000 and no hook')
    args = parser.parse_args()
    print(json.dumps(stage(args.destination, runtime_path=args.runtime, runtime_sha=args.runtime_sha256,
                           unit_path=args.unit, unit_sha=args.unit_sha256, server_binary=args.server_binary,
                           releases_root=args.releases_root, engine_releases_root=args.engine_releases_root,
                           sidecar_path=args.sidecar, measured_peak_bytes=args.measured_peak_bytes,
                           cache_gib=args.cache_gib, candidate_sha256=args.candidate_sha256,
                           fallback_peak_bytes=args.fallback_measured_peak_bytes, io_mode=args.io_mode,
                           minimum_available_bytes=args.minimum_available_bytes),
                     indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
