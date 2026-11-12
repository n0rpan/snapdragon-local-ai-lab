"""Model chooser facts for the panel, OpenCode and /v1/models; metadata only.

Never loads a model, hashes a binary or reads prompt content. Prefix warmth is
decided from bounded entry headers (Rust) or metadata JSON (llama.cpp), which
hold only model/engine identities. API alias ids are stable; only display
names and hints live here.
"""
import hashlib
import json
import os
from pathlib import Path
import re

if __package__:
    from . import flash_cache_pref, profile_policy
else:
    import flash_cache_pref
    import profile_policy

GIB = 1024**3
DEFAULT_ALIAS = 'ornith-35b-a3b-local'
# Order is the chooser order. Strength/speed copy is translated in the panel
# by `key`; the controller ships only ids, enums and numbers.
MODELS = {
    'ornith-35b-a3b-local': {'key': 'ornith', 'name': 'Ornith 1.5 35B A3B', 'order': 0},
    'qwen3.6-35b-a3b-local': {'key': 'moe', 'name': 'Qwen 3.6 35B MoE', 'order': 1},
    'qwen3.8-flash-next-local': {'key': 'flash', 'name': 'Qwen3.8 Flash-Next', 'order': 2},
    'qwen3.8-27b-local': {'key': 'dense', 'name': 'Qwen 3.8 27B Dense', 'order': 3},
}
KEYS = {row['key']: alias for alias, row in MODELS.items()}
SELECTED_ALIAS_NAME = 'Selected local model (Local AI)'
# Admission rules mirrored from the controller (classic/Rust: model + 3 GiB
# available) and the Flash adapter (admitted budget + 4 GiB OS headroom).
CLASSIC_HEADROOM_BYTES = 3 * GIB
FLASH_OS_HEADROOM_BYTES = 4 * GIB
FLASH_START_BATTERY_PERCENT = 30
RUST_PREFIX_DEFAULT = Path('.cache/local-ai/snapdragon-engine/prefix')
RUST_PREFIX_MAGIC = b'OXEPFX01'
MAX_PREFIX_ENTRIES = 512
MAX_IDENTITY_BYTES = 4096
SHA_RE = re.compile('[0-9a-f]{64}')
RUST_IDENTITY_RE = re.compile(r'model=sha256:([0-9a-f]{64}) server=sha256:([0-9a-f]{64}) ')


def display_name(alias, config=None):
    """Catalog name for known aliases; otherwise the profile's own name."""
    if alias in MODELS:
        return MODELS[alias]['name']
    name = (config or {}).get('display_name')
    return name if isinstance(name, str) and 0 < len(name) <= 128 else alias


FLASH_ALIAS = 'qwen3.8-flash-next-local'
MAX_RECEIPT_BYTES = 16 * 1024**2
_RECEIPT_CACHE = {}


def _receipts(validation):
    """(path, sha256) pairs a validation block names: its own report and any proof sources."""
    rows = []
    if isinstance(validation.get('report'), str) and isinstance(validation.get('receipt_sha256'), str):
        rows.append((validation['report'], validation['receipt_sha256']))
    sources = validation.get('proof_sources')
    if isinstance(sources, dict):
        for source in sources.values():
            if isinstance(source, dict) and isinstance(source.get('path'), str) and isinstance(source.get('sha256'), str):
                rows.append((source['path'], source['sha256']))
    return rows


def _receipt_ok(path, digest):
    """One small receipt file matches its recorded SHA-256; cached by (inode, size, mtime)."""
    if not SHA_RE.fullmatch(digest):
        return False
    try:
        info = Path(path).stat()
        key = (path, info.st_ino, info.st_size, info.st_mtime_ns)
        if key not in _RECEIPT_CACHE:
            if not Path(path).is_file() or info.st_size > MAX_RECEIPT_BYTES:
                return False
            _RECEIPT_CACHE[key] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        return _RECEIPT_CACHE[key] == digest
    except OSError:
        return False


def qualification(resolved):
    """M3: the one derived production status of a resolved profile.

    'qualified' only when the profile's own validation says qualified for exactly the engine
    hash it will launch, and every receipt it names exists with its recorded SHA-256.
    Otherwise 'unqualified' (no qualified validation), 'stale' (validation names another
    engine hash) or 'receipt-missing'. Reads small receipt files only (cached); never a model
    or an engine binary.
    """
    validation = resolved.get('validation') if isinstance(resolved, dict) else None
    if not isinstance(validation, dict) or validation.get('qualified') is not True:
        return 'unqualified'
    if validation.get('server_sha256') != resolved.get('server_sha256'):
        return 'stale'
    receipts = _receipts(validation)
    if not receipts or not all(_receipt_ok(path, digest) for path, digest in receipts):
        return 'receipt-missing'
    return 'qualified'


def flash_phase(weights_verified, qualified):
    """Storage-row phase once every Flash shard is verified: 'ready' only when qualified."""
    if not weights_verified:
        return None
    return 'ready' if qualified == 'qualified' else 'weights-ready-testing'


def engine(config):
    kind = config.get('backend_kind') or 'llama'
    return {'oxe': 'rust', 'llama': 'llama', 'bigmoe-cpu-stream': 'stream'}.get(kind, 'unknown')


def engine_status(raw_profile, resolved):
    """production: Rust engine; fallback: Rust configured but classic resolved."""
    if engine(resolved) == 'rust':
        return 'production'
    if 'oxe' in profile_policy.rust_overrides(raw_profile)[1]:
        return 'fallback'
    return 'classic'


def memory_bytes(config):
    """Classic start requirement, or the qualified Rust Flash working set (see memory_basis)."""
    try:
        if engine(config) == 'stream':
            budget = config['bigmoe']['admission']['admitted_budget_bytes']
            return budget + FLASH_OS_HEADROOM_BYTES if type(budget) is int and budget > 0 else None
        flash = config.get('oxe_flash')
        if engine(config) == 'rust' and isinstance(flash, dict):
            if flash_cache_pref.enabled(os.environ, flash):
                return None  # MemTotal admission is not a free-RAM requirement.
            admission = flash.get('admission', {})
            peak = admission.get('measured_peak_bytes', admission.get('admitted_budget_bytes'))
            return peak if type(peak) is int and peak > 0 else None
        size = config.get('expected_bytes')
        return size + CLASSIC_HEADROOM_BYTES if type(size) is int and size > 0 else None
    except (KeyError, TypeError):
        return None


def rust_prefix_directory(config, home=None):
    """Directory from an explicit --prefix-cache flag, or None when disabled."""
    args = config.get('oxe_args')
    if not isinstance(args, list):
        return None
    value = None
    for index, arg in enumerate(args):
        if arg == '--prefix-cache' and index + 1 < len(args):
            value = args[index + 1]
        elif isinstance(arg, str) and arg.startswith('--prefix-cache='):
            value = arg.split('=', 1)[1]
    if not isinstance(value, str) or not value:
        return None
    if value == 'default':
        return Path(home or Path.home()) / RUST_PREFIX_DEFAULT
    path = Path(value)
    return path if path.is_absolute() else None


def read_rust_identity(path):
    """(model_sha, server_sha) from an OXEPFX01 header; never reads tokens/state."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    except OSError:
        return None
    with os.fdopen(fd, 'rb') as stream:
        head = stream.read(12)
        if len(head) != 12 or head[:8] != RUST_PREFIX_MAGIC:
            return None
        length = int.from_bytes(head[8:], 'little')
        if not 0 < length <= MAX_IDENTITY_BYTES:
            return None
        identity = stream.read(length)
    match = RUST_IDENTITY_RE.match(identity.decode('ascii', 'replace') + ' ')
    return (match.group(1), match.group(2)) if match else None


def read_llama_identity(path):
    """(model_sha, server_sha) from a controller PrefixStore metadata file."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
        with os.fdopen(fd, 'rb') as stream:
            data = stream.read(65537)
        value = json.loads(data) if len(data) <= 65536 else None
    except (OSError, ValueError):
        return None
    if not isinstance(value, dict):
        return None
    pair = value.get('model_sha256'), value.get('server_sha256')
    return pair if all(isinstance(v, str) and SHA_RE.fullmatch(v) for v in pair) else None


def prefix_identities(directory, kind):
    """{(model, server): [entry paths]} for one store; bounded and symlink-free."""
    result = {}
    try:
        directory = Path(directory)
        if directory.is_symlink() or not directory.is_dir():
            return result
        names = sorted(os.listdir(directory))[:MAX_PREFIX_ENTRIES * 2]
    except OSError:
        return result
    suffix = '.bin' if kind == 'rust' else '.json'
    for name in names:
        if not name.endswith(suffix) or not SHA_RE.fullmatch(name[:-len(suffix)]):
            continue
        path = directory / name
        pair = read_rust_identity(path) if kind == 'rust' else read_llama_identity(path)
        if pair is not None:
            result.setdefault(pair, []).append(path)
    return result


def prefix_state(config, home=None):
    """warm: a reusable entry for this exact model+engine exists; cold; off."""
    kind = engine(config)
    if kind == 'rust':
        directory, store = rust_prefix_directory(config, home), 'rust'
    elif kind == 'llama' and isinstance(config.get('prefix_cache'), dict) and config['prefix_cache'].get('enabled') is True:
        directory, store = config['prefix_cache'].get('directory'), 'llama'
    else:
        return 'off'
    if not isinstance(directory, (str, Path)) or not Path(directory).is_absolute():
        return 'off'
    pair = (config.get('sha256'), config.get('server_sha256'))
    return 'warm' if pair in prefix_identities(directory, store) else 'cold'


def row(alias, raw_profile, resolved, *, selected, loaded_alias, state, home=None, flash_cache=None):
    """One chooser row: ids, enums and numbers only."""
    memory = memory_bytes(resolved)
    loaded = alias == loaded_alias
    cache = flash_cache_pref.catalog_status(resolved, flash_cache, os.environ)
    return {**({'flash_cache': cache} if cache else {}),
            **({'memory_basis': 'qualified_working_set'} if isinstance(resolved.get('oxe_flash'), dict) and memory else {}),
            'id': alias, 'key': MODELS.get(alias, {}).get('key', ''),
            'name': display_name(alias, resolved),
            'engine': engine(resolved), 'engine_status': engine_status(raw_profile, resolved),
            'qualification': qualification(resolved) if engine(resolved) == 'rust' else None,
            'memory_gib': round(memory / GIB, 1) if memory else None,
            'battery_start_percent': FLASH_START_BATTERY_PERCENT if engine(resolved) == 'stream' else None,
            'prefix': prefix_state(resolved, home),
            'selected': alias == selected, 'default': alias == DEFAULT_ALIAS,
            'state': (state if state in ('ready', 'loading') else 'unloaded') if loaded else 'unloaded'}


def sort_key(row_value):
    return MODELS.get(row_value['id'], {}).get('order', len(MODELS)), row_value['id']
