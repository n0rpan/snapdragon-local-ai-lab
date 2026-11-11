#!/usr/bin/env python3
"""Stage the Flash-only memory policy (own 29 GiB scope, 1 GiB host floor); never install.

Writes a fresh staging directory with the patched runtime.json (one ``flash_memory``
entry on the Flash alias, nothing else changed), a unit that points at a
content-addressed runtime release (its MemoryMax and every other line unchanged:
the MoE models stay under the unit's limit), the runtime source to copy into that
release, and exact rollback preimages plus a one-command rollback.py. It never
edits live files, starts services, creates scopes or reads model weights.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import stat
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from snapdragon_ai_lab.runtime import bigmoe_runtime, memory_policy  # noqa: E402
sys.path.insert(0, str(ROOT / 'tools'))
import stage_rollback_guard as guard  # noqa: E402

ALIAS = 'qwen3.8-flash-next-local'
POLICY = {'scope_max_bytes': memory_policy.FLASH_SCOPE_MAX_BYTES,
          'host_floor_bytes': memory_policy.FLASH_HOST_FLOOR_BYTES}
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


def add_flash_memory(runtime):
    """Return a detached runtime whose Flash profile carries the policy; all else equal."""
    updated = copy.deepcopy(runtime)
    profile = updated.get('profiles', {}).get(ALIAS)
    if not isinstance(profile, dict) or profile.get('backend_kind') != bigmoe_runtime.KIND:
        raise ValueError('flash_profile_required: ' + ALIAS)
    if 'flash_memory' in profile or 'flash_memory' in updated:
        raise ValueError('flash_memory_already_present')
    for alias, other in updated.get('profiles', {}).items():
        if alias != ALIAS and isinstance(other, dict) and 'flash_memory' in other:
            raise ValueError('flash_memory_on_other_model: ' + alias)
    profile['flash_memory'] = dict(POLICY)
    if memory_policy.flash_policy(profile) != POLICY:
        raise ValueError('flash_memory_resolution_failed')
    return updated


def rewrite_unit(text, release_dir):
    """Re-point WorkingDirectory/ExecStart only; the unit keeps its one MemoryMax line as is."""
    if len(unit_limits(text)) != 1:
        raise ValueError('unit_must_have_exactly_one_memory_limit')
    lines, done = [], set()
    for line in text.splitlines():
        if line.startswith('WorkingDirectory='):
            line = 'WorkingDirectory=' + release_dir; done.add('cwd')
        elif line.startswith('ExecStart='):
            line = 'ExecStart=/usr/bin/python3 "%s/controller.py"' % release_dir; done.add('exec')
        lines.append(line)
    if done != {'cwd', 'exec'}:
        raise ValueError('unexpected_unit_layout')
    return '\n'.join(lines) + '\n'


def unit_limits(text):
    return [line for line in text.splitlines() if line.startswith('MemoryMax=')]


ROLLBACK_DOC = 'Restore the runtime.json and unit from before the Flash memory policy, then restart.'
ROLLBACK_MESSAGE = 'Restored the previous runtime and controller; Flash runs inside the unit limit again; no model loaded.'


def stage(destination, *, runtime_path, runtime_sha, unit_path, unit_sha, releases_root):
    destination, releases_root = Path(destination), Path(releases_root)
    if not destination.is_absolute() or destination.exists() or not releases_root.is_absolute():
        raise ValueError('fresh_absolute_stage_and_releases_root_required')
    for live in (Path.home() / '.config', Path.home() / '.local'):
        if destination == live or destination.is_relative_to(live):
            raise ValueError('stage_must_not_be_live_configuration')
    runtime_raw = read_bound(runtime_path, runtime_sha)
    unit_raw = read_bound(unit_path, unit_sha)
    updated = add_flash_memory(json.loads(runtime_raw))
    source_dir = ROOT / 'src/snapdragon_ai_lab/runtime'
    sources = {p.name: p.read_bytes() for p in sorted(source_dir.iterdir())
               if p.is_file() and (p.suffix == '.py' or p.name in RUNTIME_FILES)}
    if 'exec_in_scope.py' not in sources:
        raise ValueError('scope_wrapper_missing_from_release')
    release_id = sha(json.dumps({k: sha(v) for k, v in sources.items()}, sort_keys=True).encode())[:12]
    release_dir = str(releases_root / ('flash-memory-' + release_id))
    unit = rewrite_unit(unit_raw.decode(), release_dir)
    destination.mkdir(mode=0o700)

    def put(name, data, mode=0o600):
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, mode)
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
    paths = {'runtime': runtime_path, 'unit': unit_path}
    originals = {'runtime': runtime_raw, 'unit': unit_raw}
    payload = {'runtime': (json.dumps(updated, indent=2) + '\n').encode(), 'unit': unit.encode()}
    for key, data in payload.items():
        put(guard.DEFAULT_STAGED[key], data)
    for name, data in sources.items():
        put('runtime-source/' + name, data, 0o700 if name in RUNTIME_FILES else 0o600)
    put('rollback/runtime.original', runtime_raw)
    put('rollback/unit.original', unit_raw)
    put('rollback/rollback.py', guard.render_rollback(paths, payload, originals, doc=ROLLBACK_DOC,
                                                      message=ROLLBACK_MESSAGE), 0o700)
    return {**guard.manifest_binding(paths, payload, originals), 'stage': str(destination), 'release_dir': release_dir, 'alias': ALIAS, 'flash_memory': dict(POLICY),
            'unit_limit_unchanged': unit_limits(unit)[0], 'runtime_sha256': sha((json.dumps(updated, indent=2) + '\n').encode()),
            'unit_sha256': sha(unit.encode()), 'installed': False, 'models_started': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--runtime-sha256', required=True)
    parser.add_argument('--unit', type=Path, required=True)
    parser.add_argument('--unit-sha256', required=True)
    parser.add_argument('--releases-root', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(stage(args.destination, runtime_path=args.runtime, runtime_sha=args.runtime_sha256,
                           unit_path=args.unit, unit_sha=args.unit_sha256, releases_root=args.releases_root), indent=2))


if __name__ == '__main__':
    main()
