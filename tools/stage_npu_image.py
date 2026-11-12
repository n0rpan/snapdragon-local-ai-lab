#!/usr/bin/env python3
"""Create a pinned local NPU image package. Never installs or enables Studio.

The runtime is supplied by the user and remains outside Git/release archives.
Only our executable, wrapper and device kit are copied into a new directory.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from snapdragon_ai_lab.runtime import image_npu, image_npu_worker


def create(destination, binary, binary_sha, wrapper, wrapper_sha, kit, kit_sha, runtime, runtime_sha):
    image_npu.verify_file(binary, binary_sha)
    image_npu.verify_file(wrapper, wrapper_sha)
    image_npu.verify_file(kit / 'build/SHA256SUMS', kit_sha)
    image_npu.verify_file(runtime / 'manifest.json', runtime_sha)
    manifest = json.loads((runtime / 'manifest.json').read_text())
    pins = {}
    for relative, digest in manifest['files'].items():
        path = runtime / relative
        if Path(relative).is_absolute() or '..' in Path(relative).parts:
            raise ValueError('unsafe runtime manifest path')
        image_npu.verify_file(path, digest)
        pins[str(path)] = digest
    links = {str(runtime / name): target for name, target in manifest['links'].items()}
    inventory = {}
    for line in (kit / 'build/SHA256SUMS').read_text().splitlines():
        digest, name = line.split()
        name = name.lstrip('*')
        if Path(name).name != name:
            raise ValueError('unsafe device kit inventory')
        image_npu.verify_file(kit / 'build' / name, digest)
        inventory[name] = digest
    if not {'oxn_bench', 'liboxn_skel_v73.so'} <= set(inventory):
        raise ValueError('device kit incomplete')
    destination.mkdir(mode=0o700)  # Refuse every pre-existing destination.
    (destination / 'kit/build').mkdir(mode=0o700, parents=True)
    copies = [(binary, destination / 'oxi', binary_sha, 0o700),
              (wrapper, destination / 'run-npu-image.sh', wrapper_sha, 0o700),
              (kit / 'build/SHA256SUMS', destination / 'kit/build/SHA256SUMS', kit_sha, 0o600)]
    copies += [(kit / 'build' / name, destination / 'kit/build' / name, digest,
                0o700 if name == 'oxn_bench' else 0o600) for name, digest in inventory.items()]
    for source, target, digest, mode in copies:
        shutil.copyfile(source, target)
        target.chmod(mode)
        image_npu.verify_file(target, digest)
        pins[str(target)] = digest
    profile = dict(version=1, binary=str(destination / 'oxi'), wrapper=str(destination / 'run-npu-image.sh'),
                   kit=str(destination / 'kit'), runtime=str(runtime),
                   runtime_sha256=manifest['library_sha256'], files=pins, links=links)
    path = destination / 'profile.json'
    path.write_text(json.dumps(profile, indent=2) + '\n')
    path.chmod(0o600)
    image_npu_worker.load_profile(path)
    return path


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('destination', 'binary', 'wrapper', 'kit', 'runtime'):
        parser.add_argument('--' + key, type=Path, required=True)
    for key in ('binary', 'wrapper', 'kit', 'runtime'):
        parser.add_argument('--' + key + '-sha256', required=True)
    a = parser.parse_args()
    for p in (a.destination, a.binary, a.wrapper, a.kit, a.runtime):
        if not p.is_absolute():
            parser.error('all paths must be absolute')
    path = create(a.destination, a.binary, a.binary_sha256, a.wrapper, a.wrapper_sha256,
                  a.kit, a.kit_sha256, a.runtime, a.runtime_sha256)
    print(json.dumps({'staged': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                      'enabled': False, 'services_changed': False}))


if __name__ == '__main__':
    main()
