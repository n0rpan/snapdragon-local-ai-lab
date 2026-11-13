"""CACHE-PERFECT P2 disk state cache, controller side: settings, usage, budget eviction, clear.

One ROOT for all models (default ~/.cache/local-ai/snapdragon-engine/state, inside the encrypted
home), written by oxe-server --state-cache ROOT:
  <model_key>/identity.json                     model_key = 16 hex
  <model_key>/blocks/<hh>/<h>.<c>.kv            immutable 256-token KV blocks (h 64 hex, c 16 hex)
  <model_key>/snaps/<key>.<c>.st                snapshots
  <model_key>/sessions/<sid>.json               manifests (blocks [[h, c]], snaps [{key, c, ...}], hint)
  <model_key>/echo/<hint>.jsonl                 echo records of a conversation: hint = the manifest's
                                                `hint` (32 hex of SHA-256 of the X-Oxe-Session value);
                                                forks share it, so the file goes with the last of them.
                                                Records without a session hint use the hash of their
                                                group name (`k:<key>`): no manifest names that file, so
                                                gc() removes it once it is older than the TTL
                                                (engine: state_store::hint_hash, echo.rs group_stem)
  *.pending                                     in-flight writes (ignored for 10 min)
The walk is read-only and works without an engine. While an oxe engine owns the store the
controller forwards budget and clear to it instead (controller.py). Eviction is the engine's:
TTL-expired sessions first, then lowest value = Cost(n_tokens) / bytes x 1/(1 + age_hours);
sessions used in the last 10 min (or pinned) stay; then every block or snapshot no remaining
manifest references is removed (shared blocks go last by themselves). Deletes go through
directory fds opened O_NOFOLLOW, check lstat for a regular file and touch only the names above.
File contents are never read except manifests (counts and hashes, no text). Never touches RAM state.
"""
import hashlib
import json
import os
import re
import stat
import time
from pathlib import Path

ROOT_PARTS = ('.cache', 'local-ai', 'snapdragon-engine', 'state')
DEFAULT_BUDGET = 64_000_000_000
MIN_BUDGET = 4_000_000_000
MAX_BUDGET = 1 << 50
DEFAULT_TTL = 604800
MIN_TTL, MAX_TTL = 60, 10 * 365 * 86400
PIN_S = 600       # a session used in the last 10 min is kept
PENDING_S = 600   # a .pending (or unreferenced) file younger than this may be an in-flight write
MANIFEST_MAX = 8 << 20
MODEL = re.compile(r'[0-9a-f]{16}')
HEX2 = re.compile(r'[0-9a-f]{2}')
H64, C16 = re.compile(r'[0-9a-f]{64}'), re.compile(r'[0-9a-f]{16}')
SID = re.compile(r'[A-Za-z0-9_-]{1,128}')
BLOCK = re.compile(r'([0-9a-f]{64})\.([0-9a-f]{16})\.kv')
SNAP = re.compile(r'([A-Za-z0-9_-]{1,128})\.([0-9a-f]{16})\.st')
SESSION = re.compile(r'([A-Za-z0-9_-]{1,128})\.json')
ECHO = re.compile(r'([A-Za-z0-9_-]{1,128})\.jsonl')
HINT = re.compile(r'[0-9a-f]{32}')
PENDING = re.compile(r'[A-Za-z0-9._-]{1,240}\.pending')
SUBDIRS = {'snaps': SNAP, 'sessions': SESSION, 'echo': ECHO}
SCOPES = ('all', 'model', 'session')


def cost_ms(p):
    """Re-prefill cost of p tokens (the engine's eviction price)."""
    return 3.0 * p + 0.112 * p * p / 1000


def valid_budget(n):
    return type(n) is int and (n == 0 or MIN_BUDGET <= n <= MAX_BUDGET)


def valid_ttl(n):
    return type(n) is int and MIN_TTL <= n <= MAX_TTL


def settings(runtime):
    """runtime.json `cache`: disk_budget_bytes (default 64 GB, min 4 GB, 0 = disk tier off) and
    ttl_secs (default 7 d). An invalid stored value reads as its default and is named in `invalid`."""
    cache = runtime.get('cache') if isinstance(runtime, dict) and isinstance(runtime.get('cache'), dict) else {}
    budget, ttl = cache.get('disk_budget_bytes', DEFAULT_BUDGET), cache.get('ttl_secs', DEFAULT_TTL)
    invalid = [k for k, ok in (('disk_budget_bytes', valid_budget(budget)), ('ttl_secs', valid_ttl(ttl))) if not ok]
    return {'disk_budget_bytes': budget if valid_budget(budget) else DEFAULT_BUDGET,
            'ttl_secs': ttl if valid_ttl(ttl) else DEFAULT_TTL, 'invalid': invalid}


ROOT_ENV = 'LOCAL_AI_STATE_CACHE_ROOT'
BASE_PARTS = ROOT_PARTS[:-1]  # ~/.cache/local-ai/snapdragon-engine: weights (arenas, Flash sidecar) stay here
# Conversation stores below the base. An override moves all of them, never the weight caches.
STORES = {'state': 'state', 'prefix': 'prefix', 'prefix-flash': 'prefix-flash', 'sessions': 'sessions'}


def _inside_home(path, home):
    real_home, real = os.path.realpath(home), os.path.realpath(path)
    return real != real_home and os.path.commonpath([real_home, real]) == real_home


def override(runtime=None, environ=None):
    """The raw conversation-store root override, or None: env LOCAL_AI_STATE_CACHE_ROOT wins over
    runtime.json `cache.root`. An empty env value counts as unset."""
    environ = os.environ if environ is None else environ
    raw = environ.get(ROOT_ENV)
    if raw:
        return raw
    cache = runtime.get('cache') if isinstance(runtime, dict) and isinstance(runtime.get('cache'), dict) else {}
    return cache.get('root') if 'root' in cache else None


def base(home=None, runtime=None, environ=None):
    """The directory the conversation stores live in: the override when set, else
    ~/.cache/local-ai/snapdragon-engine. An override must be an absolute path that resolves
    (symlinks of its existing part followed) inside the user's home and is not the home itself;
    anything else raises ValueError('state_cache_root_override_invalid'), never a silent fallback."""
    home = Path(home if home is not None else Path.home())
    raw = override(runtime, environ)
    if raw is None:
        return home.joinpath(*BASE_PARTS)
    if (not isinstance(raw, str) or not raw or any(ord(c) < 32 for c in raw) or not os.path.isabs(raw)
            or not home.is_absolute() or not _inside_home(raw, home)):
        raise ValueError('state_cache_root_override_invalid')
    return Path(raw)


def store_dir(name, home=None, runtime=None, environ=None):
    """One conversation store's directory (STORES), refused (ValueError) outside the home."""
    home = Path(home if home is not None else Path.home())
    path = base(home, runtime, environ) / STORES[name]
    if not home.is_absolute() or not _inside_home(path, home):
        raise ValueError('state_cache_root_outside_home')
    return path


def root(home=None, runtime=None, environ=None):
    """The disk state store's ROOT under the user's home; refused (ValueError) anywhere else."""
    return store_dir('state', home, runtime, environ)


def make_private(path, home=None):
    """Create `path` and its missing parents 0700 (existing directories are left as they are);
    refused when it resolves outside the home afterwards. Returns the path."""
    home = Path(home if home is not None else Path.home())
    missing = []
    p = Path(path)
    while not p.exists() and not p.is_symlink() and p != p.parent:
        missing.append(p)
        p = p.parent
    for d in reversed(missing):
        os.mkdir(d, 0o700)
        os.chmod(d, 0o700)  # the umask may have masked bits off; never wider than 0700
    if not _inside_home(path, home) or not Path(path).is_dir():
        raise ValueError('state_cache_root_override_invalid')
    return Path(path)


def engine_args(config, home=None):
    """oxe-server flags of the disk state store for a non-Flash oxe launch; [] when the budget is 0."""
    s = settings(config)
    if s['disk_budget_bytes'] == 0:
        return []
    return ['--state-cache', str(root(home, config)), '--state-budget-bytes', str(s['disk_budget_bytes']),
            '--state-ttl-secs', str(s['ttl_secs'])]


def valid_rel(rel):
    """Only the store's own names (relative to a model dir) may be counted or deleted."""
    parts = rel.split('/')
    if len(parts) == 1:
        return parts[0] == 'identity.json' or bool(PENDING.fullmatch(parts[0]))
    if len(parts) == 2:
        d, n = parts
        return (d in SUBDIRS and bool(SUBDIRS[d].fullmatch(n))) or (d in SUBDIRS and bool(PENDING.fullmatch(n)))
    if len(parts) == 3 and parts[0] == 'blocks' and HEX2.fullmatch(parts[1]):
        m = BLOCK.fullmatch(parts[2])
        return bool(m and m.group(1)[:2] == parts[1]) or bool(PENDING.fullmatch(parts[2]))
    return False


def _open_dir(name, dir_fd=None):
    return os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=dir_fd)


def _walk(fd, prefix, out, depth):
    """Regular files under fd with a valid store name -> (size, mtime); symlinks never followed."""
    for name in os.listdir(fd):
        try:
            st = os.stat(name, dir_fd=fd, follow_symlinks=False)
        except OSError:
            continue
        rel = prefix + name
        if stat.S_ISREG(st.st_mode) and valid_rel(rel):
            out[rel] = (st.st_size, st.st_mtime)
        elif stat.S_ISDIR(st.st_mode) and depth and (name in SUBDIRS or name == 'blocks' or (prefix == 'blocks/' and HEX2.fullmatch(name))):
            try:
                sub = _open_dir(name, fd)
            except OSError:
                continue
            try:
                _walk(sub, rel + '/', out, depth - 1)
            finally:
                os.close(sub)


def _read_manifest(model_fd, sid):
    try:
        sfd = _open_dir('sessions', model_fd)
    except OSError:
        return None
    try:
        fd = os.open(sid + '.json', os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=sfd)
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode) or os.fstat(fd).st_size > MANIFEST_MAX:
                return None
            with os.fdopen(os.dup(fd), 'rb') as f:
                return json.loads(f.read(MANIFEST_MAX + 1))
        finally:
            os.close(fd)
    except (OSError, ValueError, RecursionError):
        return None
    finally:
        os.close(sfd)


def _num(v):
    return v if type(v) in (int, float) and v == v and abs(v) != float('inf') else None


def parse_manifest(value):
    """{'refs', 'n_tokens', 'atime', 'pin'} of a version-1 manifest, or None (then the model is
    treated as opaque: nothing in it is swept as unreferenced)."""
    if not isinstance(value, dict) or value.get('version') != 1:
        return None
    blocks, snaps = value.get('blocks'), value.get('snaps', [])
    if not isinstance(blocks, list) or not isinstance(snaps, list) or _num(value.get('atime')) is None:
        return None
    refs = set()
    for b in blocks:
        if not (isinstance(b, list) and len(b) == 2 and all(isinstance(x, str) for x in b)
                and H64.fullmatch(b[0]) and C16.fullmatch(b[1])):
            return None
        refs.add('blocks/%s/%s.%s.kv' % (b[0][:2], b[0], b[1]))
    for s in snaps:
        if not (isinstance(s, dict) and isinstance(s.get('key'), str) and isinstance(s.get('c'), str)
                and SNAP.fullmatch('%s.%s.st' % (s['key'], s['c']))):
            return None
        refs.add('snaps/%s.%s.st' % (s['key'], s['c']))
    n = value.get('n_tokens')
    hint = value.get('hint')
    return {'refs': refs, 'n_tokens': n if type(n) is int and n >= 0 else 0, 'atime': value['atime'],
            'pin': value.get('pin') is True,
            # only a well-formed hint names an echo file; anything else owns none
            'hint': hint if isinstance(hint, str) and HINT.fullmatch(hint) else None}


def scan(path):
    """{model_key: {'files': {rel: (size, mtime)}, 'sessions': {sid: manifest}, 'opaque': bool}}."""
    try:
        root_fd = _open_dir(str(path))
    except FileNotFoundError:
        return {}
    models = {}
    try:
        for key in os.listdir(root_fd):
            if not MODEL.fullmatch(key):
                continue
            try:
                fd = _open_dir(key, root_fd)
            except OSError:
                continue  # a symlink or a file: never entered
            try:
                files = {}
                _walk(fd, '', files, 2)
                sessions, opaque = {}, False
                for rel in files:
                    m = SESSION.fullmatch(rel[len('sessions/'):]) if rel.startswith('sessions/') else None
                    if m:
                        parsed = parse_manifest(_read_manifest(fd, m.group(1)))
                        if parsed is None:
                            opaque = True
                        else:
                            sessions[m.group(1)] = parsed
                models[key] = {'files': files, 'sessions': sessions, 'opaque': opaque}
            finally:
                os.close(fd)
    finally:
        os.close(root_fd)
    return models


def _counted(rel, entry, now):
    return not rel.endswith('.pending') or now - entry[1] > PENDING_S


def model_bytes(model, now):
    return sum(e[0] for rel, e in model['files'].items() if _counted(rel, e, now))


def usage(path, budget, ttl, now=None):
    """GET /_local/cache: budget, used bytes, per-model bytes and session counts (no names)."""
    now = time.time() if now is None else now
    models = scan(path)
    atimes = [s['atime'] for m in models.values() for s in m['sessions'].values()]
    per = {k: model_bytes(m, now) for k, m in sorted(models.items())}
    return {'budget_bytes': budget, 'ttl_secs': ttl, 'used_bytes': sum(per.values()), 'models': per,
            'sessions': {'count': len(atimes), 'oldest_atime': min(atimes) if atimes else None,
                         'newest_atime': max(atimes) if atimes else None}}


def _manifest_file(sid):
    return 'sessions/%s.json' % sid


def hint_hash(text):
    """The engine's echo/manifest stem for a session hint or echo group: the first 32 hex of
    SHA-256(text) (state_store::hint_hash). A session group `s:<hint>` uses hint_hash(hint)."""
    return hashlib.sha256(text.encode()).hexdigest()[:32]


def echo_stem(group):
    """echo.rs group_stem: `s:<hint>` -> hint_hash(hint); any other group -> hint_hash(group)."""
    return hint_hash(group[2:] if group.startswith('s:') else group)


def _echo_file(hint):
    """echo/<hint>.jsonl for a valid 32-hex hint, else None (a null or odd hint owns no file)."""
    return 'echo/%s.jsonl' % hint if isinstance(hint, str) and HINT.fullmatch(hint) else None


def plan(models, budget, ttl, now):
    """Sessions to evict, in order: TTL-expired, then lowest value until used <= budget. Budget 0
    (disk tier off) keeps nothing, pins included."""
    used = sum(model_bytes(m, now) for m in models.values())
    refcount = {}
    for key, m in models.items():
        for s in m['sessions'].values():
            # an echo file is shared by the manifests carrying its hint, like a block
            for rel in s['refs'] | ({_echo_file(s['hint'])} - {None}):
                refcount[(key, rel)] = refcount.get((key, rel), 0) + 1
    size = lambda key, rel: models[key]['files'].get(rel, (0, 0))[0]
    expired, rest = [], []
    for key, m in models.items():
        for sid, s in m['sessions'].items():
            shared = s['refs'] | ({_echo_file(s['hint'])} - {None})
            own = size(key, _manifest_file(sid))
            exclusive = own + sum(size(key, r) for r in shared if refcount[(key, r)] == 1)
            total = own + sum(size(key, r) for r in shared)
            age_h = max(0.0, now - s['atime']) / 3600
            value = cost_ms(s['n_tokens']) / max(1, exclusive or total) / (1 + age_h)
            if budget and (s['pin'] or now - s['atime'] < PIN_S):
                continue
            (expired if now - s['atime'] > ttl else rest).append((value, key, sid))
    victims = []
    for i, (_value, key, sid) in enumerate(expired + sorted(rest)):
        if i >= len(expired) and used <= budget:
            break
        victims.append((key, sid))
        s = models[key]['sessions'][sid]
        used -= size(key, _manifest_file(sid))
        for r in s['refs'] | ({_echo_file(s['hint'])} - {None}):
            refcount[(key, r)] -= 1
            if refcount[(key, r)] == 0:
                used -= size(key, r)
    return victims


def _unlink(model_fd, rel):
    """Unlink one store file below model_fd; returns the bytes freed (0 when absent or refused)."""
    if not valid_rel(rel):
        return 0
    *dirs, name = rel.split('/')
    fds = [model_fd]
    try:
        for d in dirs:
            fds.append(_open_dir(d, fds[-1]))
        st = os.stat(name, dir_fd=fds[-1], follow_symlinks=False)
        if not stat.S_ISREG(st.st_mode):
            return 0
        os.unlink(name, dir_fd=fds[-1])
        return st.st_size
    except OSError:
        return 0
    finally:
        for fd in fds[1:]:
            os.close(fd)


def _prune_dirs(model_fd):
    """Remove the model's empty store directories (rmdir fails on anything non-empty)."""
    try:
        bfd = _open_dir('blocks', model_fd)
        try:
            for name in os.listdir(bfd):
                if HEX2.fullmatch(name):
                    try: os.rmdir(name, dir_fd=bfd)
                    except OSError: pass
        finally:
            os.close(bfd)
    except OSError:
        pass
    for name in (*SUBDIRS, 'blocks'):
        try: os.rmdir(name, dir_fd=model_fd)
        except OSError: pass


def _apply(path, models, victims, now, everything=()):
    """Delete the victims' manifests, then their echo files that no remaining manifest's hint
    names, then the blocks and snapshots of their models that no remaining manifest references
    (just released, or older than PENDING_S), or every file of the models in `everything`.
    Returns freed bytes."""
    freed = 0
    try:
        root_fd = _open_dir(str(path))
    except FileNotFoundError:
        return 0
    try:
        by_model = {}
        for key, sid in victims:
            by_model.setdefault(key, []).append(sid)
        for key in set(by_model) | set(everything):
            model = models[key]
            try:
                fd = _open_dir(key, root_fd)
            except OSError:
                continue
            try:
                if key in everything:
                    for rel in model['files']:
                        freed += _unlink(fd, rel)
                    _prune_dirs(fd)
                    try:
                        os.rmdir(key, dir_fd=root_fd)
                    except OSError:
                        pass
                    continue
                released, hints = set(), set()
                for sid in by_model[key]:
                    freed += _unlink(fd, _manifest_file(sid))
                    victim = model['sessions'].pop(sid)
                    released |= victim['refs']
                    hints.add(victim['hint'])
                if model['opaque']:
                    continue  # an unreadable manifest may reference anything: sweep nothing
                alive = {s['hint'] for s in model['sessions'].values()}
                for hint in hints - alive:
                    rel = _echo_file(hint)
                    if rel is not None:
                        freed += _unlink(fd, rel)
                kept = set().union(*(s['refs'] for s in model['sessions'].values()))
                for rel, (_size, mtime) in model['files'].items():
                    if (rel.startswith(('blocks/', 'snaps/')) and rel not in kept and
                            (rel in released or now - mtime > PENDING_S)):
                        freed += _unlink(fd, rel)
            finally:
                os.close(fd)
    finally:
        os.close(root_fd)
    return freed


def gc(path, budget, ttl, now=None):
    """Evict down to `budget` (POST /_local/cache/budget without an engine). Returns freed bytes.
    Echo files that no manifest names (groups without a session hint) go once older than `ttl`."""
    now = time.time() if now is None else now
    models = scan(path)
    freed = _apply(path, models, plan(models, budget, ttl, now), now)
    return freed + _sweep_orphan_echo(path, scan(path), now - ttl)


def _sweep_orphan_echo(path, models, before):
    """Unlink echo/<stem>.jsonl files older than `before` that no manifest's hint names.
    Models with an unreadable manifest are left alone. Returns freed bytes."""
    freed = 0
    try:
        root_fd = _open_dir(str(path))
    except FileNotFoundError:
        return 0
    try:
        for key, model in models.items():
            if model['opaque']:
                continue
            named = {_echo_file(s['hint']) for s in model['sessions'].values()} - {None}
            stale = [rel for rel, (_size, mtime) in model['files'].items()
                     if rel.startswith('echo/') and rel.endswith('.jsonl') and rel not in named and mtime < before]
            if not stale:
                continue
            try:
                fd = _open_dir(key, root_fd)
            except OSError:
                continue
            try:
                for rel in stale:
                    freed += _unlink(fd, rel)
            finally:
                os.close(fd)
    finally:
        os.close(root_fd)
    return freed


def clear(path, scope='all', model=None, session=None, now=None):
    """Unlink the state in scope: all, one model (16-hex key) or one session (in `model` or any).
    Returns freed bytes. ValueError for a bad scope or name."""
    now = time.time() if now is None else now
    if scope not in SCOPES or (model is not None and not (isinstance(model, str) and MODEL.fullmatch(model))):
        raise ValueError('state_cache_scope_invalid')
    if scope == 'model' and model is None:
        raise ValueError('state_cache_model_required')
    if scope == 'session' and not (isinstance(session, str) and SID.fullmatch(session)):
        raise ValueError('state_cache_session_required')
    models = scan(path)
    if scope == 'all':
        return _apply(path, models, [], now, everything=tuple(models))
    if scope == 'model':
        return _apply(path, models, [], now, everything=(model,) if model in models else ())
    victims = [(k, session) for k, m in models.items() if (model is None or k == model) and session in m['sessions']]
    return _apply(path, models, victims, now)
