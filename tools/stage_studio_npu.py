#!/usr/bin/env python3
"""Stage the qualified controller + Studio NPU pair; never install or start it."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tools')]
from snapdragon_ai_lab.runtime import image_npu, image_npu_worker
import stage_controller_release as source
import stage_rollback_guard as guard


def rewrite_unit(raw, release, entry, *, enable=False):
    lines, section, seen = [], '', set()
    for line in raw.decode().splitlines():
        if line.startswith('['):
            section = line
        if 'LOCAL_AI_IMAGE_NPU=' in line or 'LOCAL_AI_IMAGE_SOCKET=' in line:
            raise ValueError('image permit environment already configured; review before restaging')
        if section == '[Service]' and line.startswith('ExecStart='):
            if line == 'ExecStart=':
                # A service drop-in resets the main unit's command before
                # supplying its replacement. Preserve that required reset.
                lines.append(line)
                continue
            if 'exec' in seen:
                raise ValueError('multiple service commands')
            line = 'ExecStart=/usr/bin/python3 "%s/%s"' % (release, entry)
            seen.add('exec')
        elif section == '[Service]' and line.startswith('WorkingDirectory='):
            line = 'WorkingDirectory=' + str(release)
        lines.append(line)
        if line == '[Service]' and enable:
            lines.append('Environment=LOCAL_AI_IMAGE_NPU=1')
    if seen != {'exec'}:
        raise ValueError('unexpected service layout')
    return ('\n'.join(lines) + '\n').encode()


def stage(destination, *, paths, expected, releases_root, profile_path, profile_sha, source_dir=None):
    destination = Path(destination)
    if not destination.is_absolute() or destination.exists():
        raise ValueError('fresh absolute staging directory required')
    if not Path(releases_root).is_absolute() or not Path(profile_path).is_absolute() or not all(Path(p).is_absolute() for p in paths.values()):
        raise ValueError('all deployment paths must be absolute')
    if destination.is_relative_to(Path.home() / '.config') or destination.is_relative_to(Path.home() / '.local'):
        raise ValueError('stage must be outside live configuration')
    image_npu.verify_file(profile_path, profile_sha)
    image_npu_worker.load_profile(profile_path)
    originals = {k: source.base.read_bound(path, expected[k]) for k, path in paths.items()}
    if set(originals) != {'controller_unit', 'studio_unit', 'image_settings'}:
        raise ValueError('controller, Studio and image settings must be staged together')
    settings = json.loads(originals['image_settings'])
    if not isinstance(settings, dict) or not isinstance(settings.get('oxi', {}), dict):
        raise ValueError('invalid image settings')
    settings.setdefault('oxi', {}).update(npu=True, npu_profile=str(profile_path))
    files = source.release_sources(source_dir or ROOT / 'src/snapdragon_ai_lab/runtime')
    for name in ('studio', 'studio_page.py', 'studio_pixels.py', 'studio_context.py', 'studio_worker.py',
                 'studio_plan_generate.py', 'studio_plan_edit.py', 'studio_oxi.py', 'image_slot.py', 'image_npu.py', 'image_npu_worker.py', 'image_size_gate.py'):  # v0.1.3 H: the per-size gate tool ships with Studio
        if name not in files:
            raise ValueError('incomplete image runtime')
    hashes = {name: guard.sha(data) for name, data in files.items()}
    release_id = guard.sha(json.dumps(hashes, sort_keys=True).encode())[:12]
    release = Path(releases_root) / ('studio-npu-w1-' + release_id)
    payload = {
        'controller_unit': rewrite_unit(originals['controller_unit'], release, 'controller.py', enable=True),
        'studio_unit': rewrite_unit(originals['studio_unit'], release, 'studio'),
        'image_settings': (json.dumps(settings, indent=2) + '\n').encode()}
    names = {'controller_unit': 'local-ai-chat.service', 'studio_unit': 'prompt-studio-app.service',
             'image_settings': 'runtime-settings.json'}
    destination.mkdir(mode=0o700)
    def put(name, data, mode=0o600):
        p = destination / name
        p.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, mode)
        with os.fdopen(fd, 'wb') as output:
            output.write(data)
    for key, data in payload.items():
        put(names[key], data)
        put('rollback/' + key + '.original', originals[key])
    for name, data in files.items():
        put('runtime-source/' + name, data, 0o700 if name in ('studio', 'local-ai') else 0o600)
    put('rollback/rollback.py', guard.render_rollback(paths, payload, originals, staged_files=names,
        services=['local-ai-chat.service', 'prompt-studio-app.service'],
        doc='Restore the previous controller, Studio and image settings together.'), 0o700)
    manifest = {**guard.manifest_binding(paths, payload, originals, names),
                'release_dir': str(release), 'runtime_source_sha256': hashes,
                'npu_profile': str(profile_path), 'npu_profile_sha256': profile_sha,
                'runtime_json_changed': False, 'installed': False}
    put('manifest.json', (json.dumps(manifest, indent=2) + '\n').encode())
    return manifest


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--releases-root', type=Path, required=True)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--profile-sha256', required=True)
    for name in ('controller-unit', 'studio-unit', 'image-settings'):
        parser.add_argument('--' + name, type=Path, required=True)
        parser.add_argument('--' + name + '-sha256', required=True)
    args = parser.parse_args()
    names = ('controller_unit', 'studio_unit', 'image_settings')
    manifest = stage(args.destination, paths={k: getattr(args, k) for k in names},
        expected={k: getattr(args, k + '_sha256') for k in names}, releases_root=args.releases_root,
        profile_path=args.profile, profile_sha=args.profile_sha256)
    print(json.dumps({'stage': str(args.destination), 'release_dir': manifest['release_dir'], 'installed': False}))


if __name__ == '__main__':
    main()
