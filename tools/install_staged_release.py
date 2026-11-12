#!/usr/bin/env python3
"""Verify-preimage and atomically install a staged release (runtime.json, unit, opencode.json).

Stage layout: <stage>/manifest.json with `paths` (key -> live path) and
`preimage_sha256` (key -> sha256 of the live file before the change). The staged
replacement for each key is read from <stage>/<name>, where name is
runtime.json, unit or opencode.json for the keys runtime, unit and opencode
(override with an optional manifest `staged_files` map).

Default is a dry run: every check runs, nothing is written. `--apply` installs.
Any mismatch (missing staged file, live preimage hash differs, post-install hash
differs) refuses or stops. Contents are never printed, only hashes and paths.
This script does not stop/start services; do that around it.

Rollback binding (stage_rollback_guard): a retired stage is refused; when the
stage has rollback/<key>.original it must hash to the recorded preimage, and a
manifest `installed_sha256` must match the staged file, so the stage's guarded
rollback is known to work before anything is written. A failure part-way through
`--apply` restores the files already replaced from the verified preimage bytes;
if that is interrupted too, the stage's rollback.py restores the mixed state.
Each apply leaves <stage>/receipts/install-*.json. When the manifest records
`preimage_mode` (key -> octal string) the live mode must equal it before
anything is written, and the installed file keeps it (the rollback restores it). After a successful apply the
superseded sibling stage is named (`--retire-superseded` retires it; its
receipts and preimages are kept).
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stage_rollback_guard as guard  # noqa: E402

DEFAULT_STAGED = {'runtime': 'runtime.json', 'unit': 'unit', 'opencode': 'opencode.json'}


class RefuseError(Exception):
    pass


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load_plan(stage):
    stage = Path(stage)
    manifest = json.loads((stage / 'manifest.json').read_text())
    if guard.is_retired(stage):
        raise RefuseError(f'stage {stage.name} is retired; stage a fresh release')
    names = {**DEFAULT_STAGED, **manifest.get('staged_files', {})}
    plan = []
    for key, live in manifest['paths'].items():
        if key not in manifest.get('preimage_sha256', {}):
            raise RefuseError(f'{key}: no recorded preimage hash')
        if key not in names:
            raise RefuseError(f'{key}: no staged file name')
        src = stage / names[key]
        if not src.is_file():
            raise RefuseError(f'{key}: staged file missing: {src}')
        item = {'key': key, 'live': Path(live), 'src': src,
                'preimage': manifest['preimage_sha256'][key], 'new': sha256_file(src)}
        mode = (manifest.get('preimage_mode') or {}).get(key)
        if mode is not None:
            try:
                item['mode'] = int(str(mode), 8)
            except ValueError:
                raise RefuseError(f'{key}: bad preimage_mode {mode!r}') from None
        bound_new = (manifest.get('installed_sha256') or {}).get(key)
        if bound_new is not None and bound_new != item['new']:
            raise RefuseError(f'{key}: staged file changed since staging (manifest installed_sha256)')
        original = stage / 'rollback' / (key + '.original')
        if (stage / 'rollback').is_dir():
            if not original.is_file() or sha256_file(original) != item['preimage']:
                raise RefuseError(f'{key}: rollback/{key}.original missing or not the recorded preimage; '
                                  'rollback would not be exact')
        plan.append(item)
    return manifest, plan


def verify_preimages(plan):
    for item in plan:
        if not item['live'].is_file():
            raise RefuseError(f"{item['key']}: live file missing: {item['live']}")
        data = item['live'].read_bytes()
        actual = hashlib.sha256(data).hexdigest()
        if actual == item['preimage']:
            live_mode = item['live'].stat().st_mode & 0o7777
            if 'mode' in item and live_mode != item['mode']:
                raise RefuseError(f"{item['key']}: live mode {live_mode:04o} != recorded preimage mode "
                                  f"{item['mode']:04o}; the rollback would not restore it exactly")
            item['state'] = 'preimage-ok'
            item['preimage_bytes'] = data  # small config files; used to undo a partial install
        elif actual == item['new']:
            raise RefuseError(f"{item['key']}: already at the new hash (installed?); refusing")
        else:
            raise RefuseError(f"{item['key']}: live sha256 {actual[:12]} != recorded preimage {item['preimage'][:12]}")


def install_atomic(item):
    live = item['live']
    mode = live.stat().st_mode & 0o777
    fd, tmp = tempfile.mkstemp(prefix=live.name + '.', suffix='.new', dir=live.parent)
    try:
        with os.fdopen(fd, 'wb') as out:
            out.write(item['src_bytes'] if 'src_bytes' in item else item['src'].read_bytes())
            out.flush()
            os.fsync(out.fileno())
        os.chmod(tmp, mode)
        if sha256_file(tmp) != item['new']:
            raise RefuseError(f"{item['key']}: temp file hash mismatch before rename")
        os.replace(tmp, live)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def recover_partial(touched):
    """Put back the verified preimage bytes of every file a failed apply may have replaced."""
    restored, failed = [], []
    for item in touched:
        try:
            current = sha256_file(item['live'])
            if current not in (item['preimage'], item['new']):
                raise RefuseError('live file changed outside this install; refusing to overwrite it')
            if current != item['preimage']:
                install_atomic({**item, 'src_bytes': item['preimage_bytes'], 'new': item['preimage']})
            if sha256_file(item['live']) != item['preimage']:
                raise RefuseError('post-restore hash mismatch')
            restored.append(item['key'])
        except Exception as exc:  # keep going: restore as much as possible
            failed.append(f"{item['key']} ({exc})")
    return restored, failed


def write_receipt(stage, record):
    try:
        receipts = Path(stage) / 'receipts'
        receipts.mkdir(mode=0o700, exist_ok=True)
        stamp = time.strftime('%Y%m%dT%H%M%S')
        path = receipts / f'install-{stamp}-{os.getpid()}.json'
        path.write_text(json.dumps({'time': stamp, **record}, indent=2) + '\n')
        return path
    except OSError:
        return None


def run(stage, apply, out=None, retire_superseded=False):
    out = out or sys.stdout
    manifest, plan = load_plan(stage)
    verify_preimages(plan)
    for item in plan:
        print(f"{item['key']}: {item['live']} preimage {item['preimage'][:12]} -> new {item['new'][:12]}", file=out)
    if not apply:
        print('dry run: all preimages match; nothing written (use --apply)', file=out)
        return 0
    hashes = {i['key']: {'preimage': i['preimage'], 'installed': i['new']} for i in plan}
    done, touched = [], []
    try:
        for item in plan:
            touched.append(item)
            install_atomic(item)
            done.append(item)
            actual = sha256_file(item['live'])
            if actual != item['new']:
                raise RefuseError(f"{item['key']}: post-install sha256 {actual[:12]} != expected {item['new'][:12]}")
            if 'mode' in item and item['live'].stat().st_mode & 0o7777 != item['mode']:
                raise RefuseError(f"{item['key']}: post-install mode changed")
    except BaseException as exc:
        restored, failed = recover_partial(touched)
        write_receipt(stage, {'action': 'install-failed', 'stage': Path(stage).name, 'error': str(exc),
                              'replaced_before_failure': [d['key'] for d in done], 'restored': restored,
                              'restore_failed': failed, 'sha256': hashes})
        if failed:
            print(f"FAILED after installing {[d['key'] for d in done]}: {exc}; automatic restore FAILED for "
                  f"{failed}; run the stage rollback (it restores a partial install)", file=out)
        else:
            print(f"FAILED after installing {[d['key'] for d in done]}: {exc}; every file is back at its "
                  'preimage (verified); nothing is installed', file=out)
        if isinstance(exc, Exception) and not isinstance(exc, RefuseError):
            raise RefuseError(f'install failed: {exc}') from exc
        raise
    print('installed and verified: ' + ', '.join(d['key'] for d in done), file=out)
    write_receipt(stage, {'action': 'install', 'stage': Path(stage).name, 'installed': [d['key'] for d in done],
                          'sha256': hashes})
    for old in guard.find_superseded(stage, {i['key']: i['preimage'] for i in plan}):
        if retire_superseded:
            guard.retire(old, superseded_by=Path(stage).name, reason='superseded by a newer install')
            print(f'retired superseded stage {old.name} (receipts and preimages kept)', file=out)
        else:
            print(f'superseded stage: {old.name}; its rollback now refuses as stale. Retire it with: '
                  f'tools/stage_rollback_guard.py retire {old} --superseded-by {Path(stage).name}', file=out)
    print('manifest `installed` flag NOT modified; restart services and run qualification', file=out)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('stage', help='stage dir containing manifest.json')
    ap.add_argument('--apply', action='store_true', help='really install (default is a dry run)')
    ap.add_argument('--dry-run', action='store_true', help='explicit dry run (the default)')
    ap.add_argument('--retire-superseded', action='store_true',
                    help='after a successful --apply, retire the sibling stage this one supersedes')
    args = ap.parse_args(argv)
    if args.apply and args.dry_run:
        ap.error('--apply and --dry-run are mutually exclusive')
    try:
        return run(args.stage, args.apply, retire_superseded=args.retire_superseded)
    except RefuseError as exc:
        print(f'REFUSED: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
