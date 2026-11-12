#!/usr/bin/env python3
"""Stage a controller-only release (the runtime sources of this tree); never install.

Changes one live file when installed: the user unit, re-pointed at a content-addressed
copy of `src/snapdragon_ai_lab/runtime`. runtime.json, opencode.json and every model
profile stay as they are, so the engines, models and their arguments do not change.

Writes a fresh staging directory outside git with:
  local-ai-chat.service   the live unit with WorkingDirectory/ExecStart on release_dir
  runtime-source/         the controller release to copy into release_dir
  rollback/               the exact unit preimage plus the guarded rollback.py
  manifest.json           paths, preimage and installed hashes (stage_rollback_guard binding)

Install (by the hardware lane, after review): copy runtime-source/ to release_dir, then
`tools/install_staged_release.py <stage> --apply`, daemon-reload, restart. Rollback:
`<stage>/rollback/rollback.py` (refuses a stale, tampered or retired stage).
It never edits live files, starts services, or reads model weights.
"""
import argparse
import json
import os
from pathlib import Path
import py_compile
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tools')]
import stage_flash_memory as base  # noqa: E402
import stage_rollback_guard as guard  # noqa: E402

ROLLBACK_DOC = 'Restore the unit from before this controller release, then restart.'
ROLLBACK_MESSAGE = ('Restored the previous controller release; runtime.json was never changed; no model loaded. '
                    'The new release directory was left in place (content-addressed, harmless).')
# The standalone controller imports its siblings by name; a release without one cannot start.
REQUIRED = ('controller.py', 'openai_api.py', 'harness_compat.py', 'harness_registry.py', 'request_ledger.py',
            'model_catalog.py', 'storage_view.py', 'explicit_route.py', 'engine_ledger.py', 'oxe_runtime.py',
            'profile_policy.py', 'memory_policy.py', 'exec_in_scope.py', 'cache_turn.py', 'cache_miss.py', 'state_cache.py')


def release_sources(source_dir):
    sources = {p.name: p.read_bytes() for p in sorted(Path(source_dir).iterdir())
               if p.is_file() and (p.suffix == '.py' or p.name in base.RUNTIME_FILES)}
    missing = [name for name in REQUIRED if name not in sources]
    if missing:
        raise ValueError('release_is_missing: ' + ', '.join(missing))
    with tempfile.TemporaryDirectory() as tmp:  # a syntax error must not reach a stage
        for name, data in sources.items():
            if name.endswith('.py'):
                probe = Path(tmp) / name
                probe.write_bytes(data)
                py_compile.compile(str(probe), cfile=str(probe) + 'c', doraise=True)
    return sources


def stage(destination, *, unit_path, unit_sha, releases_root, source_dir=None, label='controller-w4'):
    destination, releases_root = Path(destination), Path(releases_root)
    if not destination.is_absolute() or destination.exists() or not releases_root.is_absolute():
        raise ValueError('fresh_absolute_stage_and_releases_root_required')
    for live in (Path.home() / '.config', Path.home() / '.local'):
        if destination == live or destination.is_relative_to(live):
            raise ValueError('stage_must_not_be_live_configuration')
    unit_raw = base.read_bound(unit_path, unit_sha)
    sources = release_sources(source_dir or ROOT / 'src/snapdragon_ai_lab/runtime')
    source_sha = {name: base.sha(data) for name, data in sources.items()}
    release_id = base.sha(json.dumps(source_sha, sort_keys=True).encode())[:12]
    release_dir = str(releases_root / (label + '-' + release_id))
    unit = base.rewrite_unit(unit_raw.decode(), release_dir).encode()
    if unit == unit_raw:
        raise ValueError('unit_already_points_at_this_release')
    destination.mkdir(mode=0o700)

    def put(name, data, mode=0o600):
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, mode)
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
    paths, payload, originals = {'unit': Path(unit_path)}, {'unit': unit}, {'unit': unit_raw}
    put(guard.DEFAULT_STAGED['unit'], unit)
    for name, data in sources.items():
        put('runtime-source/' + name, data, 0o700 if name in base.RUNTIME_FILES else 0o600)
    put('rollback/unit.original', unit_raw)
    put('rollback/rollback.py', guard.render_rollback(paths, payload, originals, doc=ROLLBACK_DOC,
                                                      message=ROLLBACK_MESSAGE), 0o700)
    manifest = {**guard.manifest_binding(paths, payload, originals), 'stage': str(destination),
                'release_dir': release_dir, 'release_id': release_id, 'runtime_source_sha256': source_sha,
                'runtime_json_changed': False, 'unit_sha256': base.sha(unit), 'installed': False, 'models_started': False}
    put('manifest.json', (json.dumps(manifest, indent=2, sort_keys=True) + '\n').encode())
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--unit', type=Path, required=True)
    parser.add_argument('--unit-sha256', required=True)
    parser.add_argument('--releases-root', type=Path, required=True)
    parser.add_argument('--label', default='controller-w4')
    args = parser.parse_args()
    manifest = stage(args.destination, unit_path=args.unit, unit_sha=args.unit_sha256,
                     releases_root=args.releases_root, label=args.label)
    print(json.dumps({k: v for k, v in manifest.items() if k != 'runtime_source_sha256'}, indent=2))


if __name__ == '__main__':
    main()
