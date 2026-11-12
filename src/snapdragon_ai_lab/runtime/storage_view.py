"""Panel Storage view: engine caches, sizes and safe deletion of unreferenced ones.

Lists the Rust weight arenas, the Rust prefix store, the Flash expert sidecar
and the llama.cpp prefix/repack caches under ~/.cache/local-ai. A cache is
deletable only when no configured profile (stock, MEMLAT or Rust variant)
references it and no process has it open or mapped. The Flash sidecar is
always kept. Weekly maintenance (tools/storage_cleanup.py) stays separate.
Never reads weights, prompt text or prefix state; only names, sizes and the
bounded identity headers used by model_catalog.
"""
import hashlib
import os
from pathlib import Path
import re
import stat

if __package__:
    from . import model_catalog
else:
    import model_catalog

ID_RE = re.compile('[a-z]+(?:-[a-z]+)*-[0-9a-f]{16}')
ARENA_RE = re.compile(r'([0-9a-f]{64})-[A-Za-z0-9]{1,16}-[0-9a-f]{7,40}-[0-9a-f]{8}(?:-mtp)?\.bin')
REPACK_RE = re.compile(r'(.{1,200}\.gguf)-[0-9a-f]{12}')
NAME_RE = re.compile(r'[A-Za-z0-9._-]{1,96}')
KINDS = ('rust-weights', 'rust-prefix', 'flash-sidecar', 'llama-prefix', 'llama-repack')


def cache_root(home=None):
    return Path(home or Path.home()) / '.cache/local-ai'


def variants(document):
    """Every effective variant a profile may resolve to (stock, MEMLAT, Rust)."""
    base = {k: v for k, v in document.items() if k != 'profiles'}
    profiles = document.get('profiles') if isinstance(document.get('profiles'), dict) else {}
    result = []
    for alias, raw in [(base.get('model_alias'), base), *profiles.items()]:
        if not isinstance(raw, dict):
            continue
        profile = {**base, **raw} if raw is not base else dict(base)
        overrides = profile.get('memlat_overrides') if isinstance(profile.get('memlat_overrides'), dict) else {}
        classic = {**profile, **{k: v for k, v in overrides.items() if k != 'oxe'}}
        result.append((alias, 'llama', profile))
        result.append((alias, 'llama', classic))
        rust = model_catalog.profile_policy.rust_overrides(profile)[1]
        if 'oxe' in rust:
            result.append((alias, 'rust', {**profile, **rust['oxe'], 'backend_kind': 'oxe'}))
        if profile.get('backend_kind') == 'bigmoe-cpu-stream':
            result.append((alias, 'stream', profile))
    return result


def references(document):
    """Model shas, model file names, and (model, engine) pairs per engine."""
    refs = {'models': {}, 'files': {}, 'rust': {}, 'llama': {}}
    for alias, kind, profile in variants(document):
        model, server = profile.get('sha256'), profile.get('server_sha256')
        if isinstance(model, str) and model_catalog.SHA_RE.fullmatch(model):
            refs['models'].setdefault(model, alias)
            if isinstance(server, str) and kind in ('rust', 'llama'):
                refs[kind].setdefault((model, server), alias)
        if isinstance(profile.get('model_path'), str):
            refs['files'].setdefault(Path(profile['model_path']).name, alias)
    return refs


def active_paths(proc=Path('/proc')):
    """Paths open or mapped by this user's processes; contents are never read."""
    result = set()
    try:
        entries = list(proc.iterdir())
    except OSError:
        return result
    for entry in entries:
        if not entry.name.isdecimal():
            continue
        try:
            if entry.stat().st_uid != os.getuid():
                continue
            for fd in (entry / 'fd').iterdir():
                try:
                    target = os.readlink(fd)
                    if target.startswith('/'):
                        result.add(target.removesuffix(' (deleted)'))
                except OSError:
                    pass
            with open(entry / 'maps', encoding='utf-8', errors='replace') as maps:
                for line in maps:
                    fields = line.split(None, 5)
                    if len(fields) == 6 and fields[5].startswith('/'):
                        result.add(fields[5].rstrip('\n').removesuffix(' (deleted)'))
        except OSError:
            continue
    return result


def _size(paths):
    total = 0
    for path in paths:
        try:
            info = path.lstat()
        except OSError:
            continue
        if stat.S_ISREG(info.st_mode):
            total += info.st_size
        elif stat.S_ISDIR(info.st_mode):
            for base, dirs, files in os.walk(path, followlinks=False):
                for name in files:
                    try:
                        child = os.lstat(os.path.join(base, name))
                        if stat.S_ISREG(child.st_mode):
                            total += child.st_size
                    except OSError:
                        pass
    return total


def _id(kind, key):
    return kind + '-' + hashlib.sha256(f'{kind}\0{key}'.encode()).hexdigest()[:16]


def _model_key(alias):
    return model_catalog.MODELS.get(alias, {}).get('key', 'other') if alias else ''


def _entry(kind, key, paths, *, name, owner, protected=False):
    return {'id': _id(kind, key), 'kind': kind, 'name': name if NAME_RE.fullmatch(name or '') else '',
            'paths': [str(p) for p in paths], 'bytes': _size(paths),
            'model': _model_key(owner), 'state': 'protected' if protected else
            ('referenced' if owner else 'unreferenced')}


def _scan(root, refs):
    rows = []
    engine = root / 'snapdragon-engine'
    try:
        names = sorted(os.listdir(engine)) if engine.is_dir() and not engine.is_symlink() else []
    except OSError:
        names = []
    for name in names:
        match = ARENA_RE.fullmatch(name)
        if match and (engine / name).is_file() and not (engine / name).is_symlink():
            rows.append(_entry('rust-weights', name, [engine / name], name=name,
                               owner=refs['models'].get(match.group(1))))
    prefix = engine / 'prefix'
    for pair, paths in sorted(model_catalog.prefix_identities(prefix, 'rust').items()):
        rows.append(_entry('rust-prefix', '%s:%s' % pair, paths, name=pair[0][:12] + '-' + pair[1][:12],
                           owner=refs['rust'].get(pair)))
    sidecar = engine / 'flash'
    if sidecar.is_dir() and not sidecar.is_symlink():
        rows.append(_entry('flash-sidecar', 'flash', [sidecar], name='flash', owner='qwen3.8-flash-next-local',
                           protected=True))
    classic = root / 'prefixes'
    for pair, paths in sorted(model_catalog.prefix_identities(classic, 'llama').items()):
        files = [p for meta in paths for p in (meta.with_suffix('.bin'), meta) if p.is_file() and not p.is_symlink()]
        rows.append(_entry('llama-prefix', '%s:%s' % pair, files, name=pair[0][:12] + '-' + pair[1][:12],
                           owner=refs['llama'].get(pair)))
    repack = root / 'repack'
    try:
        names = sorted(os.listdir(repack)) if repack.is_dir() and not repack.is_symlink() else []
    except OSError:
        names = []
    for name in names:
        match = REPACK_RE.fullmatch(name)
        if match and (repack / name).is_dir() and not (repack / name).is_symlink():
            rows.append(_entry('llama-repack', name, [repack / name], name=name,
                               owner=refs['files'].get(match.group(1))))
    return rows


def inventory(document, *, home=None, busy=None):
    """Rows with in-use marking; `paths` are kept internal for deletion."""
    root = cache_root(home)
    busy = active_paths() if busy is None else busy
    rows = _scan(root, references(document))
    for row in rows:
        if row['state'] != 'protected' and any(
                b == p or b.startswith(p.rstrip('/') + '/') for p in row['paths'] for b in busy):
            row['state'] = 'in-use'
        row['deletable'] = row['state'] == 'unreferenced'
    return rows


def public(rows):
    """Panel/API shape: no absolute paths."""
    return [{k: v for k, v in row.items() if k != 'paths'} for row in rows]


def delete(document, entry_id, *, home=None, busy=None):
    """Delete one unreferenced, unused cache; re-planned at call time.

    The caller must hold the inference lease (or own it through a loaded
    model) so no other job can start using the cache meanwhile.
    """
    if not isinstance(entry_id, str) or not ID_RE.fullmatch(entry_id):
        raise ValueError('invalid_storage_id')
    root = cache_root(home).resolve()
    rows = [row for row in inventory(document, home=home, busy=busy) if row['id'] == entry_id]
    if len(rows) != 1:
        raise LookupError('storage_entry_not_found')
    row = rows[0]
    if not row['deletable']:
        raise PermissionError('storage_entry_' + row['state'].replace('-', '_'))
    for raw in row['paths']:
        path = Path(raw)
        if path.is_symlink() or root not in path.resolve().parents:
            raise PermissionError('unsafe_storage_path')
        info = path.lstat()
        if info.st_uid != os.getuid():
            raise PermissionError('foreign_storage_path')
    freed = 0
    for raw in row['paths']:
        path = Path(raw)
        if path.is_dir():
            for base, dirs, files in os.walk(path, topdown=False, followlinks=False):
                for name in files:
                    child = os.path.join(base, name)
                    freed += os.lstat(child).st_size
                    os.unlink(child)
                for name in dirs:
                    child = os.path.join(base, name)
                    if os.path.islink(child):
                        os.unlink(child)
                    else:
                        os.rmdir(child)
            os.rmdir(path)
        else:
            freed += path.lstat().st_size
            path.unlink()
    return {'deleted': entry_id, 'kind': row['kind'], 'freed_bytes': freed}
