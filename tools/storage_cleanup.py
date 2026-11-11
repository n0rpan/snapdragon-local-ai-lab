#!/usr/bin/env python3
"""Local AI storage maintenance; dry run by default, skips a busy inference lease.

Only generated cache/run/build directories are candidates. Models and runtime
paths are preserved. Oracle dataset indexes keep the whole latest fixture set.
"""
import argparse
from collections import defaultdict
import contextlib
import fcntl
import io
import json
import os
from pathlib import Path
import re
import shutil
import stat
import time
import repack_cache

DAY = 86400


def path_references(value):
    """Protect configured assets, including image binaries and split GGUFs."""
    result = set()
    if isinstance(value, dict):
        values = value.values()
    elif isinstance(value, list):
        values = value
    elif isinstance(value, str):
        if '\0' not in value and '\n' not in value:
            path = Path(value).expanduser()
            if path.is_absolute(): result.update((path, path.resolve()))
        return result
    else:
        return result
    for item in values: result.update(path_references(item))
    return result


def runtime_references(document):
    """Include stock and MEMLAT variants, not just the currently selected model."""
    caches = defaultdict(lambda: {'engines': set(), 'models': set()})
    configured_models = set()
    protected = path_references(document)
    base = {k: v for k, v in document.items() if k != 'profiles'}
    for raw in [base, *document.get('profiles', {}).values()]:
        p = {**base, **raw}
        overrides = p.get('memlat_overrides', {})
        variants = [p, {**p, **overrides, 'environment': {**p.get('environment', {}), **overrides.get('environment', {})}}]
        if isinstance(overrides.get('oxe'), dict):
            variants.append({**p, **overrides['oxe']})
        for variant in variants:
            model = variant.get('sha256')
            if isinstance(model, str) and re.fullmatch('[0-9a-f]{64}', model):
                configured_models.add(model)
            for key in ('model_path', 'server_binary'):
                if isinstance(variant.get(key), str):
                    path = Path(variant[key]).expanduser()
                    protected.update((path, path.resolve()))
            env = variant.get('environment', {})
            value = env.get('GGML_CPU_REPACK_CACHE')
            if not value: continue
            path = Path(value).expanduser()
            if not path.is_absolute(): raise ValueError('relative_runtime_cache_path')
            protected.add(path)
            entry = caches[path.resolve()]
            for key, target in [('server_sha256', 'engines'), ('sha256', 'models')]:
                value = variant.get(key)
                if isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value): entry[target].add(value)
    # A selectable fallback can reuse a shared cache even if its present engine
    # does not populate it. Keep every configured model, not only the default.
    for identities in caches.values():
        identities['models'].update(configured_models)
    return dict(caches), protected


def inside(path, parent):
    return path == parent or parent in path.parents


def tree_info(path):
    logical = allocated = 0
    root = path.lstat()
    if path.is_symlink() or root.st_uid != os.getuid():
        raise PermissionError('tree_not_owned_regular_directory')
    newest = root.st_mtime
    def unreadable(error):
        raise error
    for base, dirs, files in os.walk(path, followlinks=False, onerror=unreadable):
        directory = Path(base).lstat()
        if directory.st_uid != os.getuid():
            raise PermissionError('foreign_owned_tree')
        newest = max(newest, directory.st_mtime)
        dirs[:] = [d for d in dirs if not (Path(base)/d).is_symlink()]
        for name in files:
            s = (Path(base)/name).lstat()
            if stat.S_ISLNK(s.st_mode):
                continue
            if s.st_uid != os.getuid() or not stat.S_ISREG(s.st_mode):
                raise PermissionError('foreign_or_special_file_in_tree')
            logical += s.st_size; allocated += s.st_blocks * 512
            newest = max(newest, s.st_mtime, s.st_atime)
    return {'logical_bytes': logical, 'allocated_bytes': allocated, 'last_touch': newest,
            'device': root.st_dev, 'inode': root.st_ino}


def active_paths():
    """Only paths, never prompt/memory contents; used solely to avoid deletion."""
    result = set()
    for proc in Path('/proc').iterdir():
        if not proc.name.isdecimal(): continue
        try:
            if proc.stat().st_uid != os.getuid(): continue
            for name in ('cwd', 'exe'):
                try: result.add(Path(os.readlink(proc/name)))
                except OSError: pass
            try:
                for fd in (proc/'fd').iterdir():
                    try:
                        target = os.readlink(fd)
                        if target.startswith('/'): result.add(Path(target))
                    except OSError: pass
            except OSError: pass
            for line in (proc/'maps').read_text().splitlines():
                fields = line.split(None, 5)
                if len(fields) == 6 and fields[5].startswith('/'): result.add(Path(fields[5]))
        except (OSError, ProcessLookupError): pass
    return result


def oracle_keeps(root):
    keep = set()
    # A model's latest dataset contains control/capture for ALL three fixtures.
    indexes = list(root.glob('*-production-summary.json'))
    for path in indexes:
        data = json.loads(path.read_text())
        if data.get('passed') is not True: raise ValueError('incomplete_oracle_index')
        for case in data['fixtures']:
            for field in ('control', 'capture'):
                p = Path(case[field]).resolve()
                if not inside(p, root.resolve()): raise ValueError('oracle_index_outside_root')
                keep.add(p)
    return keep


def build_directories(work):
    def visit(path, depth):
        if depth > 4 or path.is_symlink(): return
        try:
            if path.lstat().st_uid != os.getuid(): return
            if (path.name == 'target' and (path.parent/'Cargo.toml').is_file() or
                    path.name.startswith('build') and (path/'CMakeCache.txt').is_file()):
                yield path; return
            children = list(path.iterdir())
        except OSError: return
        for child in children:
            if child.is_dir() and not child.is_symlink() and child.name not in ('.git', 'models', 'node_modules', 'artifacts'):
                yield from visit(child, depth+1)
    if work.is_dir():
        for project in work.iterdir():
            if project.is_dir() and not project.is_symlink(): yield from visit(project, 0)


def plan(cache, work, references, protected, busy, now, age_days=7, retire_qualifiers=False):
    actions, skipped = [], []
    protected = protected | set(references)
    def inspect(path):
        try: return tree_info(path)
        except OSError as error:
            skipped.append({'path': str(path), 'reason': 'unreadable-or-changed-tree',
                            'error_type': type(error).__name__})
            return None
    for path in cache.glob('repack*'):
        if not path.is_dir() or path.is_symlink() or path.resolve() in references: continue
        info = inspect(path)
        if info is None: continue
        retired = retire_qualifiers and path.name in ('repack-qualified-final', 'repack-qualified-v2')
        if not retired and now - info['last_touch'] <= age_days*DAY: continue
        if any(inside(p, path.resolve()) for p in protected | busy):
            skipped.append({'path': str(path), 'reason': 'referenced-or-in-use'}); continue
        actions.append({'path': str(path), 'kind': 'unreferenced-repack', **info})
    for path in build_directories(work):
        info = inspect(path)
        if info is None: continue
        if now - info['last_touch'] <= age_days*DAY: continue
        if any(inside(p, path.resolve()) for p in protected | busy):
            skipped.append({'path': str(path), 'reason': 'referenced-or-in-use'}); continue
        actions.append({'path': str(path), 'kind': 'old-build', **info})
    oracle = cache/'engine-oracle'
    if oracle.is_dir() and not oracle.is_symlink():
        keeps = oracle_keeps(oracle)
        indexed_prefixes = [p.name.removesuffix('-summary.json') + '-' for p in oracle.glob('*-production-summary.json')]
        # Unknown/unindexed runs are preserved until an explicit model dataset
        # index exists. Never treat the shared repack directory as an oracle run.
        if keeps:
            for path in oracle.iterdir():
                if not path.is_dir() or path.is_symlink() or path.name.startswith('repack'): continue
                if not any(path.name.startswith(prefix) for prefix in indexed_prefixes): continue
                if not ((path/'summary.json').is_file() or any(path.glob('*/manifest.json'))): continue
                if any(inside(p, path.resolve()) for p in keeps | protected | busy): continue
                info = inspect(path)
                if info is not None and now - info['last_touch'] > age_days*DAY:
                    actions.append({'path': str(path), 'kind': 'old-oracle', **info})
    return actions, skipped


def append_log(path, report):
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 1024**2:
        path.replace(path.with_suffix('.previous.jsonl'))
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'a') as out: out.write(json.dumps(report)+'\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--apply', action='store_true')
    ap.add_argument('--wait-lease', action='store_true', help='wait for the lease (manual validation only; scheduled jobs skip)')
    ap.add_argument('--runtime', type=Path, default=Path.home()/'.config/local-ai/llm/runtime.json')
    ap.add_argument('--retire-qualifiers-now', action='store_true', help='retire the two explicitly approved qualification caches now')
    args = ap.parse_args(); os.umask(0o077)
    cache = Path.home()/'.cache/local-ai'; work = Path.home()/'Work/local-ai'
    lease = Path.home()/'.local/state/local-ai/inference.lock'
    lease.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(lease, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
            raise ValueError('owned_regular_inference_lease_required')
        try: fcntl.flock(fd, fcntl.LOCK_EX | (0 if args.wait_lease else fcntl.LOCK_NB))
        except BlockingIOError:
            print(json.dumps({'skipped': 'inference lease busy', 'freed_bytes': 0})); return
        refs, protected = runtime_references(json.loads(args.runtime.read_text()))
        image_config = Path.home()/'.config/local-ai/images'
        for name in ('cpu-profile.json', 'gpu-profile.json', 'runtime-settings.json'):
            config = image_config/name
            if config.exists():
                protected.update(path_references(json.loads(config.read_text())))
        busy = active_paths()
        actions, skipped = plan(cache, work, refs, protected, busy, time.time(), retire_qualifiers=args.retire_qualifiers_now)
        before = shutil.disk_usage(cache).free
        report = {'kind': 'local-ai-storage-maintenance', 'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                  'apply': args.apply, 'actions': actions, 'skipped': skipped, 'cache_prunes': [],
                  'planned_logical_bytes': sum(a['logical_bytes'] for a in actions)}
        for path, identities in refs.items():
            if not path.is_dir() or path.is_symlink(): continue
            if any(inside(p, path) for p in busy):
                report['skipped'].append({'path': str(path), 'reason': 'mapped-by-process'}); continue
            # Never age current runtime entries. Filter obsolete model/engine
            # identities as a union, preserving every configured profile.
            ns = argparse.Namespace(dir=str(path), older_than=None, build=None, patch=None,
                                    engine=sorted(identities['engines']), model_sha=sorted(identities['models']),
                                    keep_latest=True, dry_run=not args.apply)
            pre = tree_info(path); text = io.StringIO()
            with contextlib.redirect_stdout(text): repack_cache.cmd_prune(ns)
            post = tree_info(path)
            report['cache_prunes'].append({'path': str(path), 'logical_bytes_before': pre['logical_bytes'],
                                           'logical_bytes_after': post['logical_bytes'], 'details_tail': text.getvalue()[-2048:]})
        if args.apply:
            for action in actions:
                path = Path(action['path'])
                if path.is_symlink() or not (inside(path, cache) or inside(path, work)): raise ValueError('unsafe_cleanup_path')
                current = tree_info(path)
                if (current['device'], current['inode'], current['last_touch']) != (action['device'], action['inode'], action['last_touch']):
                    report['skipped'].append({'path': str(path), 'reason': 'changed-since-plan'})
                    continue
                shutil.rmtree(path)
            report['filesystem_free_bytes_after'] = shutil.disk_usage(cache).free
            report['observed_free_delta_bytes'] = report['filesystem_free_bytes_after'] - before
            append_log(Path.home()/'.local/state/local-ai/storage-cleanup.jsonl', report)
        print(json.dumps(report, indent=2))
    finally: os.close(fd)


if __name__ == '__main__': main()
