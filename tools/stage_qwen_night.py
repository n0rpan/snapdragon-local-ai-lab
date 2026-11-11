#!/usr/bin/env python3
"""Prepare tomorrow's Qwen-only upgrade and rollback; never install or load it.

The stage contains credentials copied from the live configs: owner-only files,
outside git. Regenerate it after tonight's Ornith install; preimages must match.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tools')]
import stage_flash_memory as base  # noqa: E402
from stage_idle_output_defaults import ROLLBACK_DOC, ROLLBACK_MESSAGE  # noqa: E402
import stage_rollback_guard as guard  # noqa: E402
from snapdragon_ai_lab.runtime import profile_policy  # noqa: E402

ALIAS = 'qwen3.6-35b-a3b-local'
MODEL_SHA = '0b21525e972670ed59e1812e170b27c26355381f0656ecc4e25617ece7dac58b'
SERVERS = {'0e1c41eae6f9a422c5e104b34517a12da0e15007fdbcb63ee8b529f7b1e26ddc': 'a219347',
           'c8213805571f7caf4e702ef286c399e2e796c6d392e4dc6a6c01986880222e1f': 'e831814'}


def patch_runtime(runtime, binary, digest, sessions):
    if digest not in SERVERS:
        raise ValueError('unknown_night_binary')
    updated = copy.deepcopy(runtime)
    profile = updated.get('profiles', {}).get(ALIAS)
    if not isinstance(profile, dict) or profile.get('sha256') != MODEL_SHA:
        raise ValueError('qwen_model_identity_required')
    old = profile.get('memlat_overrides', {}).get('oxe')
    if not isinstance(old, dict):
        raise ValueError('existing_qwen_oxe_profile_required')
    profile['default_max_tokens'] = 16384
    profile['memlat_overrides']['oxe'] = {
        'server_binary': str(binary), 'server_sha256': digest, 'context_size': 32768, 'threads': 8,
        'oxe_args': ['--engine', 'oxe', '--model', '{model}', '--alias', '{alias}',
                     '--ctx-size', '{context}', '--oxe-cpus', '0-7', '--oxe-prefill-cpus', '0-11',
                     '--ubatch-size', '256', '--ctx-checkpoints', '8', '--reasoning', 'off',
                     '--spec-type', 'none', '--prefix-cache', 'default',
                     '--session-ram-bytes', str(2 * 2**30), '--session-cache', str(sessions)],
        'environment': {'OXE_ATTN_TILED_MIN_KV': '0', 'OXE_PF_TILES': '13',
                        'LOCAL_AI_ENGINE_LEDGER': '1'},
        'validation': {'kind': 'qwen-night-staged-unqualified', 'engine_source': SERVERS[digest],
                       'server_sha256': digest, 'qualified': False,
                       'required': ['qwen-fixtures-mask13', 'tool-parser', 'cancel', 'session-resume',
                                    'protected-task', '32k-memory-no-swap', 'no-regression']}}
    shared = {k: v for k, v in updated.items() if k != 'profiles'}
    effective = profile_policy.resolve({**shared, **profile, 'model_alias': ALIAS}, available=True, oxe_enabled=True)
    if effective['context_size'] != 32768 or effective['backend_kind'] != 'oxe':
        raise ValueError('candidate_resolution_failed')
    return updated


def patch_opencode(config):
    updated = copy.deepcopy(config)
    entry = updated.get('providers', {}).get('local-qwen', {}).get('models', {}).get(ALIAS)
    if not isinstance(entry, dict) or not isinstance(entry.get('limit'), dict):
        raise ValueError('qwen_opencode_limit_required')
    entry['limit'].update(context=32768, input=24576, output=16384)
    return updated


def stage(destination, *, runtime, unit, opencode, server, preimages, releases_root, sessions):
    destination = Path(destination)
    if not destination.is_absolute() or destination.exists():
        raise ValueError('fresh_absolute_stage_required')
    for live in (Path.home() / '.config', Path.home() / '.local', ROOT):
        if destination == live or destination.is_relative_to(live):
            raise ValueError('stage_must_be_outside_live_config_and_git')
    if not Path(releases_root).is_absolute() or not Path(sessions).is_absolute():
        raise ValueError('absolute_release_and_session_paths_required')
    if Path(sessions).resolve().is_relative_to('/var/lib/local-ai-models'):
        raise ValueError('private_sessions_must_stay_on_encrypted_storage')
    inputs = {'runtime': runtime, 'unit': unit, 'opencode': opencode}
    raw = {key: base.read_bound(path, preimages[key]) for key, path in inputs.items()}
    engine = base.read_bound(server, None, limit=256 * 2**20); digest = base.sha(engine)
    engine_dir = Path(releases_root) / ('oxe-qwen-night-' + digest[:12])
    updated = patch_runtime(json.loads(raw['runtime']), engine_dir / 'oxe-server', digest, sessions)
    oc = patch_opencode(json.loads(raw['opencode']))
    source = ROOT / 'src/snapdragon_ai_lab/runtime'
    sources = {p.name: p.read_bytes() for p in sorted(source.iterdir())
               if p.is_file() and (p.suffix == '.py' or p.name in base.RUNTIME_FILES)}
    release_id = base.sha(json.dumps({k: base.sha(v) for k, v in sources.items()}, sort_keys=True).encode())[:12]
    release_dir = Path(releases_root) / ('qwen-controller-' + release_id)
    text = base.rewrite_unit(raw['unit'].decode(), str(release_dir))
    destination.mkdir(mode=0o700)

    def put(name, data, mode=0o600):
        target = destination / name; target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, mode)
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
    payload = {'runtime': (json.dumps(updated, indent=2) + '\n').encode(), 'unit': text.encode(),
               'opencode': (json.dumps(oc, indent=2, ensure_ascii=False) + '\n').encode()}
    for key, content in payload.items():
        put(guard.DEFAULT_STAGED[key], content)
    put('engine/oxe-server', engine, 0o700)
    for name, content in sources.items():
        put('runtime-source/' + name, content, 0o700 if name in base.RUNTIME_FILES else 0o600)
    for key, content in raw.items():
        put('rollback/' + key + '.original', content)
    put('rollback/rollback.py', guard.render_rollback(inputs, payload, raw, doc=ROLLBACK_DOC,
                                                      message=ROLLBACK_MESSAGE), 0o700)
    manifest = {'qualified': False, 'installed': False, 'models_started': False, 'alias': ALIAS,
                **guard.manifest_binding(inputs, payload, raw), 'server_sha256': digest,
                'release_dir': str(release_dir), 'engine_dir': str(engine_dir),
                'note': 'STAGE ONLY. Qwen gate and separate promotion approval required. Regenerate after Ornith install.'}
    put('manifest.json', (json.dumps(manifest, indent=2) + '\n').encode())
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    for key in ('runtime', 'unit', 'opencode'):
        parser.add_argument('--' + key, type=Path, required=True)
        parser.add_argument('--' + key + '-sha256', required=True)
    parser.add_argument('--server', type=Path, required=True)
    parser.add_argument('--releases-root', type=Path, required=True)
    parser.add_argument('--sessions', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(stage(args.destination, runtime=args.runtime, unit=args.unit, opencode=args.opencode,
                           server=args.server, releases_root=args.releases_root, sessions=args.sessions,
                           preimages={k: getattr(args, k + '_sha256') for k in ('runtime', 'unit', 'opencode')}), indent=2))


if __name__ == '__main__':
    main()
