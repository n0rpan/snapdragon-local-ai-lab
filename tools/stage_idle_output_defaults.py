#!/usr/bin/env python3
"""Stage the approved idle/output defaults (2026-11-10); never install.

Writes a fresh staging directory outside git with:
  runtime.json            default_max_tokens=16384 on the oxe Ornith and Qwen 3.6 profiles
  local-ai-chat.service   points at a content-addressed runtime release, idle unload 600 s
  opencode.json           provider output cap 2048/4096 -> 16384 (bounded by each alias's context)
  runtime-source/         the controller release to copy into release_dir
  rollback/               exact preimages of all three files plus rollback.py

opencode.json holds credentials: this tool only touches the documented limit
keys, writes files 0600 in a 0700 directory, and never prints file contents.
It never edits live files, starts services, or reads model weights.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'tools'))
from snapdragon_ai_lab.runtime import oxe_runtime, profile_policy  # noqa: E402
import stage_oxe_production as base  # noqa: E402
import stage_rollback_guard as guard  # noqa: E402

DEFAULT_MAX_TOKENS = 16384
IDLE_SECONDS = 600
OUTPUT_ALIASES = ('ornith-35b-a3b-local', 'qwen3.6-35b-a3b-local')  # oxe-backed in production
OPENCODE_PROVIDER = 'local-qwen'
OPENCODE_ALIASES = ('qwen-local', *OUTPUT_ALIASES)  # qwen-local resolves to the selected model
IDLE_LINE = 'Environment="LOCAL_LLM_IDLE_SECONDS=%d"' % IDLE_SECONDS


def add_output_defaults(runtime):
    updated = copy.deepcopy(runtime)
    shared = {k: v for k, v in updated.items() if k != 'profiles'}
    for alias in OUTPUT_ALIASES:
        profile = updated.get('profiles', {}).get(alias)
        if not isinstance(profile, dict):
            raise ValueError('profile_required: ' + alias)
        if 'default_max_tokens' in profile:
            raise ValueError('default_max_tokens_already_present: ' + alias)
        profile['default_max_tokens'] = DEFAULT_MAX_TOKENS
        effective = profile_policy.resolve({**shared, **profile, 'model_alias': alias}, available=True, oxe_enabled=True)
        if effective.get('backend_kind') != oxe_runtime.KIND or oxe_runtime.default_output_tokens(effective) is None:
            raise ValueError('oxe_resolution_required: ' + alias)
    if 'default_max_tokens' in shared:
        raise ValueError('shared_default_max_tokens_refused')
    return updated


def raise_opencode_output(opencode):
    updated = copy.deepcopy(opencode)
    models = updated.get('providers', {}).get(OPENCODE_PROVIDER, {}).get('models', {})
    for alias in OPENCODE_ALIASES:
        limit = models.get(alias, {}).get('limit')
        if not isinstance(limit, dict) or type(limit.get('context')) is not int or type(limit.get('output')) is not int:
            raise ValueError('opencode_limit_required: ' + alias)
        if limit['output'] >= DEFAULT_MAX_TOKENS:
            raise ValueError('opencode_output_already_raised: ' + alias)
        limit['output'] = min(DEFAULT_MAX_TOKENS, limit['context'])
    return updated


def rewrite_unit(text, release_dir):
    if 'LOCAL_LLM_IDLE_SECONDS' in text:
        raise ValueError('unit_already_sets_idle')
    lines, done = [], set()
    for line in text.splitlines():
        if line.startswith('WorkingDirectory='):
            line = 'WorkingDirectory=' + release_dir; done.add('cwd')
        elif line.startswith('ExecStart='):
            line = 'ExecStart=/usr/bin/python3 "%s/controller.py"' % release_dir; done.add('exec')
        lines.append(line)
        if line.startswith('Environment="LOCAL_LLM_CONFIG='):
            lines.append(IDLE_LINE); done.add('env')
    if done != {'cwd', 'exec', 'env'}:
        raise ValueError('unexpected_unit_layout')
    return '\n'.join(lines) + '\n'


ROLLBACK_DOC = 'Restore the exact pre-change runtime.json, unit and opencode.json, then restart the controller.'
ROLLBACK_MESSAGE = 'Restored previous runtime.json, unit (controller release, idle) and opencode.json; no model loaded.'


def stage(destination, *, runtime_path, runtime_sha, unit_path, unit_sha, opencode_path, opencode_sha, releases_root):
    destination, releases_root = Path(destination), Path(releases_root)
    if not destination.is_absolute() or destination.exists() or not releases_root.is_absolute():
        raise ValueError('fresh_absolute_stage_and_releases_root_required')
    for live in (Path.home() / '.config', Path.home() / '.local', ROOT):
        if destination == live or destination.is_relative_to(live):
            raise ValueError('stage_must_be_outside_live_configuration_and_git')
    runtime_raw = base.read_bound(runtime_path, runtime_sha)
    unit_raw = base.read_bound(unit_path, unit_sha)
    opencode_raw = base.read_bound(opencode_path, opencode_sha)
    runtime = add_output_defaults(json.loads(runtime_raw))
    opencode = raise_opencode_output(json.loads(opencode_raw))
    source_dir = ROOT / 'src/snapdragon_ai_lab/runtime'
    sources = {p.name: p.read_bytes() for p in sorted(source_dir.iterdir())
               if p.is_file() and (p.suffix == '.py' or p.name in base.RUNTIME_FILES)}
    release_id = base.sha(json.dumps({k: base.sha(v) for k, v in sources.items()}, sort_keys=True).encode())[:12]
    release_dir = str(releases_root / ('idle-output-' + release_id))
    unit = rewrite_unit(unit_raw.decode(), release_dir)
    destination.mkdir(mode=0o700)

    def put(name, data, mode=0o600):
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, mode)
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
    paths = {'runtime': runtime_path, 'unit': unit_path, 'opencode': opencode_path}
    originals = {'runtime': runtime_raw, 'unit': unit_raw, 'opencode': opencode_raw}
    payload = {'runtime': (json.dumps(runtime, indent=2) + '\n').encode(), 'unit': unit.encode(),
               'opencode': (json.dumps(opencode, indent=2, ensure_ascii=False) + '\n').encode()}
    for key, data in payload.items():
        put(guard.DEFAULT_STAGED[key], data)
    for name, data in sources.items():
        put('runtime-source/' + name, data, 0o700 if name in base.RUNTIME_FILES else 0o600)
    put('rollback/runtime.original', runtime_raw)
    put('rollback/unit.original', unit_raw)
    put('rollback/opencode.original', opencode_raw)
    put('rollback/rollback.py', guard.render_rollback(paths, payload, originals, doc=ROLLBACK_DOC,
                                                      message=ROLLBACK_MESSAGE), 0o700)
    return {**guard.manifest_binding(paths, payload, originals), 'stage': str(destination), 'release_dir': release_dir, 'idle_unload_seconds': IDLE_SECONDS,
            'default_max_tokens': {a: DEFAULT_MAX_TOKENS for a in OUTPUT_ALIASES},
            'opencode_output_raised': list(OPENCODE_ALIASES), 'installed': False, 'models_started': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('destination', type=Path)
    for name in ('runtime', 'unit', 'opencode'):
        parser.add_argument('--' + name, type=Path, required=True)
        parser.add_argument('--%s-sha256' % name, required=True)
    parser.add_argument('--releases-root', type=Path, required=True)
    a = parser.parse_args()
    print(json.dumps(stage(a.destination, runtime_path=a.runtime, runtime_sha=a.runtime_sha256, unit_path=a.unit,
                           unit_sha=a.unit_sha256, opencode_path=a.opencode, opencode_sha=a.opencode_sha256,
                           releases_root=a.releases_root), indent=2))


if __name__ == '__main__':
    main()
