#!/usr/bin/env python3
"""Stage the MEMLAT-only Rust (oxe) switch for the two MoE aliases; never install.

Writes a fresh staging directory with the patched runtime.json, a unit that
points at a content-addressed runtime release and sets LOCAL_AI_ENABLE_OXE=1,
the runtime source to copy into that release, and exact rollback preimages
plus a one-command rollback.py. It never edits live files, starts services,
or reads model weights (only the small engine binary is hashed).
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
from snapdragon_ai_lab.runtime import profile_policy  # noqa: E402
sys.path.insert(0, str(ROOT / 'tools'))
import stage_rollback_guard as guard  # noqa: E402

# Exact model identities qualified with the Rust server (RUST-M3 / RUST-SERVER-QUALIFICATION).
QUALIFIED_MODELS = {
    'ornith-35b-a3b-local': 'a7373f724cf3693e8a6a5f8209ae9a06ff8071ae2af60659bdbdf8ff6f6c10df',
    'qwen3.6-35b-a3b-local': '0b21525e972670ed59e1812e170b27c26355381f0656ecc4e25617ece7dac58b',
}
QUALIFIED_SERVER_SHA = '649ac80670e58d5a4806355543f67a8bce0ce6034cf63e380dbf0c2a3f3ccf69'
CONTEXT = 16384
THREADS = 8
# Matches controller-integrated-*-oxe runtime.json except MTP is explicitly off.
OXE_ARGS = ['--engine', 'oxe', '--model', '{model}', '--alias', '{alias}', '--ctx-size', '{context}',
            '--oxe-cpus', '0-7', '--oxe-prefill-cpus', '0-11', '--ubatch-size', '256',
            '--ctx-checkpoints', '4', '--reasoning', 'off', '--spec-type', 'none',
            '--prefix-cache', 'default']
ENABLE_LINE = 'Environment="%s=1"' % profile_policy.OXE_ENABLE_ENV
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


def oxe_override(server_binary, server_sha256):
    return {'server_binary': server_binary, 'server_sha256': server_sha256, 'threads': THREADS,
            'oxe_args': list(OXE_ARGS), 'environment': {'OXE_ATTN_TILED_MIN_KV': '0'},
            'validation': {'kind': 'oxe-rust-server-qualification', 'engine_source': '707a0ec',
                           'server_sha256': server_sha256, 'report': 'docs/RUST-SERVER-UPGRADE.md'}}


def add_oxe_overrides(runtime, server_binary, server_sha256):
    """Return a detached runtime with a nested memlat 'oxe' override per alias."""
    if server_sha256 != QUALIFIED_SERVER_SHA:
        raise ValueError('unqualified_rust_server')
    updated = copy.deepcopy(runtime)
    for alias, model_sha in QUALIFIED_MODELS.items():
        profile = updated.get('profiles', {}).get(alias)
        if not isinstance(profile, dict) or profile.get('sha256') != model_sha:
            raise ValueError('qualified_model_identity_required: ' + alias)
        if profile.get('context_size') != CONTEXT or profile.get('backend_kind') not in (None, 'llama'):
            raise ValueError('classic_16k_profile_required: ' + alias)
        overrides = profile.setdefault('memlat_overrides', {})
        if 'oxe' in overrides:
            raise ValueError('oxe_override_already_present: ' + alias)
        overrides['oxe'] = oxe_override(server_binary, server_sha256)
        # Validate both resolutions: classic MEMLAT when disabled, Rust when enabled.
        merged = {**{k: v for k, v in updated.items() if k != 'profiles'}, **profile}
        profile_policy.resolve(merged, available=True, oxe_enabled=False)
        effective = profile_policy.resolve(merged, available=True, oxe_enabled=True)
        if effective['backend_kind'] != 'oxe' or effective['prefix_cache']['enabled'] is not False:
            raise ValueError('oxe_resolution_failed: ' + alias)
    return updated


def rewrite_unit(text, release_dir):
    """Point WorkingDirectory/ExecStart at release_dir and enable oxe once."""
    if profile_policy.OXE_ENABLE_ENV in text:
        raise ValueError('unit_already_mentions_oxe')
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


ROLLBACK_DOC = 'Restore the pre-oxe runtime.json and unit (classic engines), then restart.'
ROLLBACK_MESSAGE = 'Restored the previous runtime, controller and classic engines; no model loaded.'


def stage(destination, *, runtime_path, runtime_sha, unit_path, unit_sha, server_binary, releases_root):
    destination, releases_root = Path(destination), Path(releases_root)
    if not destination.is_absolute() or destination.exists() or not releases_root.is_absolute():
        raise ValueError('fresh_absolute_stage_and_releases_root_required')
    for live in (Path.home() / '.config', Path.home() / '.local'):
        if destination == live or destination.is_relative_to(live):
            raise ValueError('stage_must_not_be_live_configuration')
    runtime_raw = read_bound(runtime_path, runtime_sha)
    unit_raw = read_bound(unit_path, unit_sha)
    server_sha = sha(read_bound(server_binary, None, limit=256 * 1024**2))
    updated = add_oxe_overrides(json.loads(runtime_raw), str(server_binary), server_sha)
    source_dir = ROOT / 'src/snapdragon_ai_lab/runtime'
    sources = {p.name: p.read_bytes() for p in sorted(source_dir.iterdir())
               if p.is_file() and (p.suffix == '.py' or p.name in RUNTIME_FILES)}
    release_id = sha(json.dumps({k: sha(v) for k, v in sources.items()}, sort_keys=True).encode())[:12]
    release_dir = str(releases_root / ('oxe-production-' + release_id))
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
    return {**guard.manifest_binding(paths, payload, originals), 'stage': str(destination), 'release_dir': release_dir, 'server_sha256': server_sha,
            'aliases': sorted(QUALIFIED_MODELS), 'installed': False, 'models_started': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--runtime-sha256', required=True)
    parser.add_argument('--unit', type=Path, required=True)
    parser.add_argument('--unit-sha256', required=True)
    parser.add_argument('--server-binary', type=Path, required=True)
    parser.add_argument('--releases-root', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(stage(args.destination, runtime_path=args.runtime, runtime_sha=args.runtime_sha256,
                           unit_path=args.unit, unit_sha=args.unit_sha256, server_binary=args.server_binary,
                           releases_root=args.releases_root), indent=2))


if __name__ == '__main__':
    main()
