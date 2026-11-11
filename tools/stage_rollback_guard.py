#!/usr/bin/env python3
"""Guarded stage rollback: bind, check, retire. Never installs or touches a service itself.

A stage's `rollback/rollback.py` is generated from ROLLBACK_TEMPLATE with a
binding baked in: for every key the live path, the sha256 this stage installs
and the sha256 of the preimage (`rollback/<key>.original`). The generated
script, in this order and BEFORE it stops the service:

  1. refuses a retired stage (exit 7);
  2. verifies every `.original` against its bound sha256, and the binding
     against manifest.json when the manifest carries the same keys (exit 5);
  3. refuses when any live file is neither this stage's installed bytes nor
     its preimage, naming the sibling stage whose payload is live (exit 3).

Only then: stop, restore atomically, verify, daemon-reload, start, receipt.
A mixed state (some files installed, some already at the preimage: a failed
partial install or an interrupted rollback) is restored; all files already at
the preimage is a no-op that leaves the service alone. `--check` runs the
checks and changes nothing.

Commands:
  bind STAGE --path key=/live/path ...   guard an existing stage (replaces its
                                         rollback.py; the old one is kept
                                         non-executable as rollback.py.unguarded)
  check STAGE                            run the stage's rollback.py --check
  retire STAGE [--superseded-by NAME]    mark retired; keeps receipts/preimages
  reinstate STAGE                        undo retire once the stage is the current
                                         install again (newer stage rolled back)
  status STAGE                           print binding state as JSON

Contents of bound files are never printed, only hashes and paths.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

DEFAULT_STAGED = {'runtime': 'runtime.json', 'unit': 'local-ai-chat.service', 'opencode': 'opencode.json'}
SERVICE = 'local-ai-chat.service'
RETIRED_MARKER = 'retired.json'
EXIT_STALE, EXIT_TAMPERED, EXIT_RETIRED = 3, 5, 7


class GuardError(Exception):
    pass


def sha(data):
    return hashlib.sha256(data).hexdigest()


def sha_file(path):
    try:
        return sha(Path(path).read_bytes())
    except OSError:
        return None


ROLLBACK_TEMPLATE = r'''#!/usr/bin/python3
"""@DOC@

Guarded (stage_rollback_guard): refuses, before the service is stopped, when the
stage is retired (exit 7), a preimage or the manifest binding changed (exit 5), or
a live file is no longer what this stage installed (exit 3). `--check` only checks.
"""
from pathlib import Path
import hashlib, json, os, subprocess, sys, time

BINDING = json.loads(@BINDING@)
SYSTEMCTL = BINDING['systemctl']
SERVICE = BINDING['service']
root = Path(__file__).resolve().parent
stage = root.parent


def digest(path):
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def refuse(code, message, note='nothing changed, service not touched'):
    print('ROLLBACK REFUSED (' + note + '): ' + message, file=sys.stderr)
    sys.exit(code)


def owner_of(name, live_sha):
    """Name the sibling stage whose staged payload for `name` is the live file."""
    staged_name = BINDING['files'][name]['staged']
    try:
        siblings = sorted(p for p in stage.parent.iterdir() if p.is_dir() and p != stage)
    except OSError:
        return None
    for other in siblings:
        names = {staged_name}
        try:
            manifest = json.loads((other / 'manifest.json').read_text())
            if (manifest.get('installed_sha256') or {}).get(name) == live_sha:
                return other.name
            names.add((manifest.get('staged_files') or {}).get(name, staged_name))
        except (OSError, ValueError, AttributeError):
            pass
        if any(digest(other / n) == live_sha for n in names):
            return other.name
    return None


def checks():
    marker = stage / 'retired.json'
    manifest = None
    try:
        manifest = json.loads((stage / 'manifest.json').read_text())
    except FileNotFoundError:
        pass
    except (OSError, ValueError):
        refuse(5, 'manifest.json is unreadable')
    if marker.exists() or (isinstance(manifest, dict) and manifest.get('retired')):
        by = ''
        try:
            by = json.loads(marker.read_text()).get('superseded_by') or ''
        except (OSError, ValueError, AttributeError):
            pass
        refuse(7, 'stage ' + stage.name + ' is retired' + (' (superseded by ' + by + ')' if by else '')
               + '; roll back the current stage instead')
    for name, item in BINDING['files'].items():
        if digest(root / (name + '.original')) != item['preimage_sha256']:
            refuse(5, 'preimage ' + name + '.original does not match its bound sha256 '
                   + item['preimage_sha256'][:12] + ' (missing or modified)')
        if isinstance(manifest, dict):
            for field, key in (('preimage_sha256', 'preimage_sha256'), ('installed_sha256', 'installed_sha256')):
                recorded = manifest.get(field)
                if isinstance(recorded, dict) and name in recorded and recorded[name] != item[key]:
                    refuse(5, 'manifest ' + field + '.' + name + ' differs from the binding in rollback.py')
    return live_state()


def live_state():
    todo, stale = [], []
    for name, item in BINDING['files'].items():
        live = digest(item['path'])
        if live == item['preimage_sha256']:
            continue
        if live == item['installed_sha256']:
            todo.append(name)
            continue
        owner = owner_of(name, live) if live else None
        stale.append('%s (%s) is %s, not the %s this stage installed: %s' % (
            name, item['path'], live[:12] if live else 'missing', item['installed_sha256'][:12],
            'that is the payload of the newer install ' + owner + '; roll that stage back first' if owner
            else 'changed by a newer install or by hand; no sibling stage matches'))
    if stale:
        refuse(3, 'stale rollback of ' + stage.name + ': ' + '; '.join(stale))
    return todo


def restore(name):
    item = BINDING['files'][name]
    path = Path(item['path'])
    data = (root / (name + '.original')).read_bytes()
    if hashlib.sha256(data).hexdigest() != item['preimage_sha256']:
        raise SystemExit('preimage ' + name + ' changed during rollback; stopped')
    tmp = path.with_name(path.name + '.rollback')
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(data); stream.flush(); os.fsync(stream.fileno())
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)
    if digest(path) != item['preimage_sha256']:
        raise SystemExit('post-restore sha256 mismatch for ' + name + '; stopped, service left down')


def main():
    check_only = '--check' in sys.argv[1:]
    todo = checks()
    if check_only:
        print('rollback check OK for ' + stage.name + ': would restore ' + (', '.join(todo) or 'nothing (already at the preimage)'))
        return
    if not todo:
        print('Nothing to roll back: every file is already at the preimage; service not touched.')
        return
    subprocess.run([SYSTEMCTL, '--user', 'stop', SERVICE], check=True)
    if live_state_changed(todo):
        subprocess.run([SYSTEMCTL, '--user', 'start', SERVICE], check=True)
        refuse(3, 'a live file changed while the service was stopping', 'no file changed, service started again')
    for name in todo:
        restore(name)
    subprocess.run([SYSTEMCTL, '--user', 'daemon-reload'], check=True)
    subprocess.run([SYSTEMCTL, '--user', 'start', SERVICE], check=True)
    receipts = root / 'receipts'
    receipts.mkdir(mode=0o700, exist_ok=True)
    stamp = time.strftime('%Y%m%dT%H%M%S')
    (receipts / ('rollback-' + stamp + '-' + str(os.getpid()) + '.json')).write_text(json.dumps(
        {'action': 'rollback', 'stage': stage.name, 'time': stamp, 'restored': todo,
         'sha256': {n: BINDING['files'][n]['preimage_sha256'] for n in todo}}, indent=2) + '\n')
    print(@MESSAGE@)


def live_state_changed(todo):
    for name, item in BINDING['files'].items():
        live = digest(item['path'])
        expected = item['installed_sha256'] if name in todo else item['preimage_sha256']
        if live != expected:
            return True
    return False


if __name__ == '__main__':
    main()
'''


def render_rollback(paths, installed, preimages, *, staged_files=None, message=None, doc=None,
                    systemctl='systemctl', service=SERVICE):
    """Return the guarded rollback.py source (bytes).

    paths: key -> live path; installed: key -> bytes this stage installs;
    preimages: key -> exact bytes of rollback/<key>.original.
    """
    keys = list(paths)
    if not keys or set(keys) != set(installed) or set(keys) != set(preimages):
        raise GuardError('binding_keys_must_match')
    names = {**DEFAULT_STAGED, **(staged_files or {})}
    files = {}
    for key in keys:
        if not re.fullmatch(r'[a-z][a-z0-9_-]*', key) or not Path(str(paths[key])).is_absolute():
            raise GuardError('bad_binding_key_or_relative_path: ' + key)
        files[key] = {'path': str(paths[key]), 'staged': names.get(key, key),
                      'installed_sha256': sha(installed[key]), 'preimage_sha256': sha(preimages[key])}
    binding = {'version': 1, 'systemctl': str(systemctl), 'service': service, 'files': files}
    doc = doc or 'Restore the exact pre-change %s, then restart the controller.' % ', '.join(keys)
    if '"""' in doc or '\\' in doc:
        raise GuardError('bad_doc')
    message = message or 'Restored the previous %s; no model loaded.' % ', '.join(keys)
    text = (ROLLBACK_TEMPLATE.replace('@DOC@', doc)
            .replace('@BINDING@', repr(json.dumps(binding, sort_keys=True)))
            .replace('@MESSAGE@', repr(message)))
    compile(text, 'rollback.py', 'exec')
    return text.encode()


def manifest_binding(paths, installed, preimages, staged_files=None):
    """Keys a stage manifest carries so install_staged_release and rollback agree."""
    names = {**DEFAULT_STAGED, **(staged_files or {})}
    return {'paths': {k: str(v) for k, v in paths.items()},
            'preimage_sha256': {k: sha(preimages[k]) for k in paths},
            'installed_sha256': {k: sha(installed[k]) for k in paths},
            'staged_files': {k: names.get(k, k) for k in paths}}


def read_binding(stage):
    """Binding baked into a guarded stage's rollback.py; GuardError when unguarded."""
    script = Path(stage) / 'rollback' / 'rollback.py'
    try:
        text = script.read_text()
    except OSError as exc:
        raise GuardError('no rollback script: %s' % script) from exc
    match = re.search(r'^BINDING = json\.loads\((.+)\)$', text, re.M)
    if not match:
        raise GuardError('rollback script is not guarded: %s (run `bind`)' % script)
    binding = json.loads(ast.literal_eval(match.group(1)))
    if binding.get('version') != 1 or not isinstance(binding.get('files'), dict):
        raise GuardError('unknown binding version in %s' % script)
    return binding


def is_retired(stage):
    stage = Path(stage)
    if (stage / RETIRED_MARKER).exists():
        return True
    try:
        return bool(json.loads((stage / 'manifest.json').read_text()).get('retired'))
    except (OSError, ValueError, AttributeError):
        return False


def is_current(binding):
    """True when every file this stage changes is live with this stage's bytes."""
    changed = [i for i in binding['files'].values() if i['installed_sha256'] != i['preimage_sha256']]
    return bool(changed) and all(sha_file(i['path']) == i['installed_sha256'] for i in changed)


def status(stage):
    stage = Path(stage)
    binding = read_binding(stage)
    files = {}
    for key, item in binding['files'].items():
        live = sha_file(item['path'])
        files[key] = {'path': item['path'],
                      'live': 'installed' if live == item['installed_sha256'] else
                              'preimage' if live == item['preimage_sha256'] else 'other',
                      'preimage_ok': sha_file(stage / 'rollback' / (key + '.original')) == item['preimage_sha256']}
    return {'stage': stage.name, 'retired': is_retired(stage), 'files': files}


def _write_private(path, data, mode=0o600):
    tmp = path.with_name(path.name + '.tmp')
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, mode)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(data); stream.flush(); os.fsync(stream.fileno())
    os.chmod(tmp, mode)
    os.replace(tmp, path)


def retire(stage, *, superseded_by=None, reason=None, force=False):
    """Mark a superseded stage retired. Deletes nothing; its rollback then refuses.

    Refuses while any bound live file still holds this stage's installed bytes
    without being the preimage too (the stage is still live for that file):
    retiring it would remove the one-command rollback of the current state.
    """
    stage = Path(stage)
    if not (stage / 'rollback').is_dir():
        raise GuardError('not a stage with a rollback directory: %s' % stage)
    if (stage / RETIRED_MARKER).exists():
        raise GuardError('already retired: %s' % stage.name)
    try:
        binding = read_binding(stage)
    except GuardError:
        binding = None  # unguarded legacy stage: retiring it is still safe and useful
    if binding and not force and is_current(binding):
        raise GuardError('stage %s is the CURRENT install (live files match it); not retired' % stage.name)
    kept = sorted(str(p.relative_to(stage)) for p in (stage / 'rollback').rglob('*') if p.is_file())
    kept += sorted(str(p.relative_to(stage)) for p in (stage / 'receipts').rglob('*') if p.is_file()) \
        if (stage / 'receipts').is_dir() else []
    record = {'retired': True, 'stage': stage.name, 'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
              'superseded_by': superseded_by, 'reason': reason, 'kept': kept}
    _write_private(stage / RETIRED_MARKER, (json.dumps(record, indent=2) + '\n').encode())
    manifest_path = stage / 'manifest.json'
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, ValueError):
        manifest = None
    if isinstance(manifest, dict):
        manifest['retired'] = True
        if superseded_by:
            manifest['superseded_by'] = superseded_by
        _write_private(manifest_path, (json.dumps(manifest, indent=2) + '\n').encode(),
                       manifest_path.stat().st_mode & 0o777)
    return record


def reinstate(stage):
    """Undo retire for a stage that is the current install again (the newer stage was rolled back)."""
    stage = Path(stage)
    marker = stage / RETIRED_MARKER
    if not is_retired(stage):
        raise GuardError('not retired: %s' % stage.name)
    if not is_current(read_binding(stage)):
        raise GuardError('stage %s is not the current install; stays retired' % stage.name)
    if marker.exists():
        os.replace(marker, stage / (RETIRED_MARKER + '.reinstated-' + time.strftime('%Y%m%dT%H%M%S')))
    manifest_path = stage / 'manifest.json'
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, ValueError):
        manifest = None
    if isinstance(manifest, dict) and manifest.get('retired'):
        manifest['retired'] = False
        manifest.pop('superseded_by', None)
        _write_private(manifest_path, (json.dumps(manifest, indent=2) + '\n').encode(),
                       manifest_path.stat().st_mode & 0o777)
    return {'stage': stage.name, 'retired': False}


def bind(stage, paths, *, staged_files=None, systemctl='systemctl', service=SERVICE, message=None):
    """Guard an existing stage in place: bind its payload and preimages, replace rollback.py.

    The stage's own files are the source: installed sha = sha of the staged
    payload, preimage sha = sha of rollback/<key>.original. When manifest.json
    records preimage_sha256 / paths for a key they must agree. The old script is
    kept as rollback.py.unguarded (mode 0600, not executable).
    """
    stage = Path(stage)
    rollback_dir = stage / 'rollback'
    if not rollback_dir.is_dir():
        raise GuardError('not a stage with a rollback directory: %s' % stage)
    if is_retired(stage):
        raise GuardError('stage is retired: %s' % stage.name)
    try:
        manifest = json.loads((stage / 'manifest.json').read_text())
    except FileNotFoundError:
        manifest = None
    if manifest is not None and not isinstance(manifest, dict):
        raise GuardError('manifest.json is not an object')
    names = {**DEFAULT_STAGED, **((manifest or {}).get('staged_files') or {}), **(staged_files or {})}
    installed, preimages = {}, {}
    for key, live in paths.items():
        try:
            installed[key] = (stage / names.get(key, key)).read_bytes()
            preimages[key] = (rollback_dir / (key + '.original')).read_bytes()
        except OSError as exc:
            raise GuardError('%s: staged payload or preimage missing (%s)' % (key, exc.filename)) from exc
        recorded = ((manifest or {}).get('preimage_sha256') or {})
        if isinstance(recorded, dict) and key in recorded and recorded[key] != sha(preimages[key]):
            raise GuardError('%s: rollback/%s.original does not match manifest preimage_sha256' % (key, key))
        known = ((manifest or {}).get('paths') or {}).get(key)
        if known is not None and str(known) != str(live):
            raise GuardError('%s: --path differs from manifest paths' % key)
    extra = sorted(p.name[:-9] for p in rollback_dir.glob('*.original') if p.name[:-9] not in paths)
    if extra:
        raise GuardError('preimages without a --path: ' + ', '.join(extra))
    script = render_rollback(paths, installed, preimages, staged_files=names, systemctl=systemctl,
                             service=service, message=message)
    target = rollback_dir / 'rollback.py'
    if target.exists() and not (rollback_dir / 'rollback.py.unguarded').exists():
        try:
            read_binding(stage)
        except GuardError:
            _write_private(rollback_dir / 'rollback.py.unguarded', target.read_bytes(), 0o600)
    _write_private(target, script, 0o700)
    binding = manifest_binding(paths, installed, preimages, names)
    if manifest is not None:
        manifest.update(binding)
        _write_private(stage / 'manifest.json', (json.dumps(manifest, indent=2) + '\n').encode(),
                       (stage / 'manifest.json').stat().st_mode & 0o777)
    return binding


def find_superseded(stage, plan_preimages):
    """Sibling stages whose installed payload equals this stage's preimage for every shared key."""
    stage = Path(stage)
    found = []
    try:
        siblings = sorted(p for p in stage.parent.iterdir() if p.is_dir() and p != stage)
    except OSError:
        return found
    for other in siblings:
        if is_retired(other):
            continue
        try:
            files = read_binding(other)['files']
        except GuardError:
            continue
        shared = [k for k in files if k in plan_preimages]
        if shared and all(files[k]['installed_sha256'] == plan_preimages[k] for k in shared):
            found.append(other)
    return found


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('bind'); p.add_argument('stage', type=Path)
    p.add_argument('--path', action='append', required=True, metavar='KEY=/live/path')
    p = sub.add_parser('check'); p.add_argument('stage', type=Path)
    p = sub.add_parser('retire'); p.add_argument('stage', type=Path)
    p.add_argument('--superseded-by'); p.add_argument('--reason')
    p.add_argument('--force', action='store_true', help='retire even when the stage is the current install')
    p = sub.add_parser('reinstate'); p.add_argument('stage', type=Path)
    p = sub.add_parser('status'); p.add_argument('stage', type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == 'bind':
            paths = dict(item.split('=', 1) for item in args.path)
            print(json.dumps(bind(args.stage, paths), indent=2))
        elif args.command == 'check':
            read_binding(args.stage)
            return subprocess.run([sys.executable, str(args.stage / 'rollback' / 'rollback.py'), '--check']).returncode
        elif args.command == 'retire':
            record = retire(args.stage, superseded_by=args.superseded_by, reason=args.reason, force=args.force)
            print('retired %s; kept %d rollback/receipt files' % (record['stage'], len(record['kept'])))
        elif args.command == 'reinstate':
            print('reinstated %s' % reinstate(args.stage)['stage'])
        else:
            print(json.dumps(status(args.stage), indent=2))
    except GuardError as exc:
        print('REFUSED: %s' % exc, file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
