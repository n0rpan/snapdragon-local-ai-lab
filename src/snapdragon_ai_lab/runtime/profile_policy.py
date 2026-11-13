"""Read-only MEMLAT detection and explicit, per-profile execution overrides."""
import copy
import hashlib
import os
from pathlib import Path
import re

if __package__:
    from . import flash_cache_pref, harness_registry, memory_policy
else:
    import flash_cache_pref
    import harness_registry
    import memory_policy

FLAGS = {'threads_batch': ('--threads-batch', '-tb'),
         'cpu_mask': ('--cpu-mask', '-C'), 'cpu_mask_batch': ('--cpu-mask-batch', '-Cb'),
         'cpu_strict': ('--cpu-strict',), 'cpu_strict_batch': ('--cpu-strict-batch',),
         'spec_type': ('--spec-type',), 'spec_draft_n_max': ('--spec-draft-n-max',)}
ENGINE_KEYS = ('server_binary', 'server_sha256')
KEYS = {'threads', 'small_batch_limit', 'prefix_cache_enabled', 'environment', 'extra_args', 'oxe', 'oxe_flash', *ENGINE_KEYS, *FLAGS}
# Rust engine switches (M4). The Rust engines are the default: unset or '1' means on; only an
# explicit '0' (or off/false/no) selects the classic llama.cpp / BigMoe fallback.
OXE_ENABLE_ENV = 'LOCAL_AI_ENABLE_OXE'
SWITCH_OFF_VALUES = ('0', 'off', 'false', 'no')
# Top-level per-profile Rust engine block: {"oxe": {...}} or {"oxe_flash": {...}}, same schema as the
# nested memlat_overrides entries. Unlike those (historical name, applied only on a MEMLAT boot), it
# applies whenever its switch is on, so a clean install runs the Rust engine or fails loudly.
RUST_ENGINE_KEY = 'rust_engine'
RUST_KEYS = ('oxe', 'oxe_flash')
OXE_KEYS = {'server_binary', 'server_sha256', 'oxe_args', 'threads', 'context_size', 'environment', 'validation', 'rss_proof'}
OXE_REQUIRED = {'server_binary', 'server_sha256', 'oxe_args'}
OXE_MODEL_SHA_ENV = 'OXE_MODEL_SHA256'
# Separate MEMLAT-only switch for the Rust Flash engine; independent of LOCAL_AI_ENABLE_OXE.
OXE_FLASH_ENABLE_ENV = 'LOCAL_AI_ENABLE_OXE_FLASH'
FLASH_KIND = 'bigmoe-cpu-stream'
OXE_FLASH_KEYS = {'server_binary', 'server_sha256', 'oxe_args', 'threads', 'environment', 'validation',
                  'cache_gib', 'sidecar_path', 'sidecar_bytes', 'admission', 'idle_unload_seconds', 'fallback',
                  'session_reserve_bytes', 'idle_qos', 'cache_pref', 'desktop_floor_gib',
                  'swap_tolerant_guard', 'prefix_store'}
OXE_FLASH_REQUIRED = {'server_binary', 'server_sha256', 'oxe_args', 'cache_gib', 'sidecar_path', 'sidecar_bytes', 'admission'}
FLASH_DENSE_LOWER_BOUND = 4522949120
# Absolute bound; the 28 GiB limit without a flash_memory policy is enforced at launch (oxe_runtime).
FLASH_CAP_BYTES = 29 * 1024**3
LEGACY_ENV_PREFIXES = ('GGML_', 'LLAMA_')

# V013 lane C: 256K is "RSS-proven or not offered" (LONG-CTX-DESIGN sec. 3/7). An Ornith-class oxe
# profile may run above the 131072 standard cap, up to 262144, only when its oxe block carries the
# receipt of a measured RSS ladder for exactly the engine it launches:
#   "rss_proof": {"context": 262144, "peak_rss_bytes": <= 27 GiB, "engine_sha256": <server_sha256>,
#                 "date": "YYYY-MM-DD"}
# A malformed record is a profile error; a well-formed one for another engine (after an engine
# update) or another context only makes the 256K mode unavailable, with a reason.
BASE_CONTEXT_CAP = harness_registry.BASE_CAP
LONG_CONTEXT_CAP = harness_registry.LONG_CAP
RSS_PROOF_PEAK_MAX = 27 * 1024**3
RSS_PROOF_KEYS = frozenset({'context', 'peak_rss_bytes', 'engine_sha256', 'date'})
ORNITH_CLASS = frozenset({'ornith-35b-a3b-local'})
# The models OpenCode's MoE preset (and therefore the fast/deep/max modes) covers.
MODE_MODELS = ('ornith-35b-a3b-local', 'qwen3.6-35b-a3b-local')
CONTEXT_MODES_KEY = 'context_modes'
NEEDS_RSS_PROOF = 'needs RSS proof'


def validate_rss_proof(proof):
    """The record's shape; ValueError('oxe_rss_proof_invalid') when malformed."""
    if (not isinstance(proof, dict) or set(proof) != RSS_PROOF_KEYS
            or type(proof['context']) is not int or not BASE_CONTEXT_CAP < proof['context'] <= LONG_CONTEXT_CAP
            or type(proof['peak_rss_bytes']) is not int or proof['peak_rss_bytes'] <= 0
            or not isinstance(proof['engine_sha256'], str) or not re.fullmatch(r'[0-9a-f]{64}', proof['engine_sha256'])
            or not isinstance(proof['date'], str) or not re.fullmatch(r'20[0-9]{2}-[01][0-9]-[0-3][0-9]', proof['date'])):
        raise ValueError('oxe_rss_proof_invalid')
    return proof


def rss_proof_reason(config, context=LONG_CONTEXT_CAP):
    """None when `config` (a resolved oxe profile) may run at `context`; else a short reason.
    Contexts up to 131072 need no proof."""
    if not isinstance(config, dict):
        return NEEDS_RSS_PROOF
    if type(context) is not int or context > LONG_CONTEXT_CAP:
        return 'above 262144'
    if context <= BASE_CONTEXT_CAP:
        return None
    if config.get('model_alias') not in ORNITH_CLASS:
        return 'Ornith only'
    if config.get('backend_kind') != 'oxe' or isinstance(config.get('oxe_flash'), dict):
        return 'Rust engine only'
    proof = config.get('rss_proof')
    try:
        validate_rss_proof(proof)
    except ValueError:
        return NEEDS_RSS_PROOF
    if proof['engine_sha256'] != config.get('server_sha256'):
        return 'RSS proof is for another engine'
    if proof['context'] < context:
        return NEEDS_RSS_PROOF
    if proof['peak_rss_bytes'] > RSS_PROOF_PEAK_MAX:
        return 'RSS above 27 GiB'
    return None


def stored_mode(runtime, alias):
    """The persisted mode of `alias` (runtime.json `context_modes.models`), else the stored
    `context_modes.default`, else fast (O-105). Unknown values read as the default."""
    block = runtime.get(CONTEXT_MODES_KEY) if isinstance(runtime, dict) else None
    block = block if isinstance(block, dict) else {}
    default = block.get('default') if block.get('default') in harness_registry.MODES else harness_registry.DEFAULT_MODE
    models = block.get('models') if isinstance(block.get('models'), dict) else {}
    mode = models.get(alias)
    return mode if mode in harness_registry.MODES else default


def with_stored_mode(runtime, alias, mode):
    """A copy of runtime.json with `alias`'s mode set (other keys untouched)."""
    if mode not in harness_registry.MODES or not isinstance(alias, str):
        raise ValueError('unsupported_context_mode')
    value = copy.deepcopy(runtime)
    block = value.get(CONTEXT_MODES_KEY)
    block = dict(block) if isinstance(block, dict) else {}
    models = dict(block['models']) if isinstance(block.get('models'), dict) else {}
    models[alias] = mode
    block['models'] = models
    value[CONTEXT_MODES_KEY] = block
    return value


def mode_rows(config):
    """[{id, compact_at, cap, available, reason}] for a profile with modes, else []."""
    if not isinstance(config, dict) or config.get('model_alias') not in MODE_MODELS:
        return []
    rows = []
    for mode in harness_registry.MODES:
        cap = harness_registry.MODE_CAP[mode]
        reason = rss_proof_reason(config, cap)
        rows.append({'id': mode, 'compact_at': harness_registry.compact_at(mode), 'cap': cap,
                     'available': reason is None, 'reason': reason})
    return rows


def effective_mode(config, mode):
    """`mode` when this profile can run it, else deep (a stored max whose proof went stale
    falls back to the 131072 cap); None for a profile without modes."""
    rows = {row['id']: row for row in mode_rows(config)}
    if not rows:
        return None
    return mode if mode in rows and rows[mode]['available'] else harness_registry.DEFAULT_MODE


def apply_mode(config, mode):
    """The resolved profile with its context mode applied: `context_mode` set and, for max, the
    engine context raised to 262144. Anything without modes is returned unchanged."""
    chosen = effective_mode(config, mode)
    if chosen is None:
        return config
    result = dict(config)
    result['context_mode'] = chosen
    if chosen == 'max':
        result['context_size'] = harness_registry.MODE_CAP['max']
    elif type(result.get('context_size')) is int and result['context_size'] > BASE_CONTEXT_CAP:
        result['context_size'] = BASE_CONTEXT_CAP  # fast/deep never run above the standard cap
    return result


def context_status(config, mode):
    """`status.context` (V013-PLAN sec. 3): {cap, mode, modes}. cap is the engine context the
    model runs in this mode (the profile's own context, never above 131072 outside max)."""
    rows = mode_rows(config)
    if not rows:
        cap = config.get('context_size') if isinstance(config, dict) else None
        return {'cap': cap if type(cap) is int else None, 'mode': None, 'modes': []}
    applied = apply_mode(config, mode)
    cap = applied.get('context_size')
    return {'cap': cap if type(cap) is int else harness_registry.MODE_CAP[applied['context_mode']],
            'mode': applied['context_mode'], 'modes': rows}


def switch_on(name, env=None):
    """M4: a Rust engine switch is on unless explicitly set to 0/off/false/no."""
    value = (os.environ if env is None else env).get(name)
    return value is None or value.strip().lower() not in SWITCH_OFF_VALUES


def rust_overrides(profile):
    """(source, {'oxe'|'oxe_flash': block}) for a raw profile; source is 'rust_engine',
    'memlat_overrides' or None. Metadata only; validation happens in resolve()."""
    if not isinstance(profile, dict):
        return None, {}
    top = profile.get(RUST_ENGINE_KEY)
    if isinstance(top, dict) and any(isinstance(top.get(k), dict) for k in RUST_KEYS):
        return RUST_ENGINE_KEY, {k: top[k] for k in RUST_KEYS if isinstance(top.get(k), dict)}
    nested = profile.get('memlat_overrides')
    if isinstance(nested, dict) and any(isinstance(nested.get(k), dict) for k in RUST_KEYS):
        return 'memlat_overrides', {k: nested[k] for k in RUST_KEYS if isinstance(nested.get(k), dict)}
    return None, {}


def memlat_available(root=Path('/sys/class/devfreq')):
    """Fail closed on absent, partial, unreadable or non-remote domains."""
    found = set()
    try:
        for device in Path(root).iterdir():
            # Unrelated incomplete devfreq devices do not hide valid domains.
            try:
                name = (device / 'name').read_text().strip().lower()
                if name in ('ddr', 'llcc') and (device / 'governor').read_text().strip() == 'remote':
                    found.add(name)
            except (OSError, UnicodeError):
                continue
    except (OSError, UnicodeError):
        return False
    return found == {'ddr', 'llcc'}


def _validate_oxe_flash(value):
    if not isinstance(value, dict) or set(value) - OXE_FLASH_KEYS or not OXE_FLASH_REQUIRED <= set(value):
        raise ValueError('memlat_oxe_flash_override_requires_binary_sha256_args_cache_and_admission')
    _validate({k: value[k] for k in ENGINE_KEYS})
    if 'cache_pref' in value:
        flash_cache_pref.read_pref({}, value)
    if 'desktop_floor_gib' in value:
        flash_cache_pref.desktop_floor(value)
    if 'swap_tolerant_guard' in value:
        memory_policy.swap_tolerant_policy(value['swap_tolerant_guard'])
    if 'prefix_store' in value:
        flash_prefix_store(value['prefix_store'])
    args = value['oxe_args']
    if not isinstance(args, list) or not args or not all(isinstance(a, str) and not any(c in a for c in '\n\r\0') for a in args):
        raise ValueError('memlat_oxe_args_must_be_strings')
    if 'threads' in value and (type(value['threads']) is not int or not 1 <= value['threads'] <= 12):
        raise ValueError('invalid_memlat_integer')
    environment = value.get('environment', {})
    if not isinstance(environment, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in environment.items()):
        raise ValueError('memlat_environment_must_map_strings')
    if any(k.startswith(LEGACY_ENV_PREFIXES) for k in environment):
        raise ValueError('memlat_oxe_environment_excludes_classic_engine_keys')
    if not isinstance(value.get('validation', {}), dict):
        raise ValueError('memlat_oxe_validation_must_be_mapping')
    if type(value['cache_gib']) is not int or value['cache_gib'] not in (14, 16, 18, 20, 21):
        raise ValueError('memlat_oxe_flash_cache_gib_must_be_14_16_18_20_or_21')
    path = value['sidecar_path']
    if not isinstance(path, str) or not Path(path).is_absolute() or any(ord(c) < 32 for c in path):
        raise ValueError('memlat_oxe_flash_sidecar_path_must_be_absolute')
    if type(value['sidecar_bytes']) is not int or value['sidecar_bytes'] <= 0:
        raise ValueError('memlat_oxe_flash_sidecar_bytes_invalid')
    if 'idle_unload_seconds' in value and (type(value['idle_unload_seconds']) is not int
                                           or not 60 <= value['idle_unload_seconds'] <= 3600):
        raise ValueError('memlat_oxe_flash_idle_unload_invalid')
    if 'idle_qos' in value and (not isinstance(value['idle_qos'], dict)
                                or set(value['idle_qos']) != {'unit', 'cpu', 'latency_us'}
                                or value['idle_qos']['unit'] != 'local-ai-flash-qos@{pid}.service'
                                or type(value['idle_qos']['cpu']) is not int or value['idle_qos']['cpu'] != 8
                                or type(value['idle_qos']['latency_us']) is not int
                                or value['idle_qos']['latency_us'] != 100):
        raise ValueError('memlat_oxe_flash_idle_qos_invalid')
    _validate_flash_admission(value['admission'], value['cache_gib'])
    # Optional smaller cache the controller starts with when the primary admission fails.
    if 'session_reserve_bytes' in value and (type(value['session_reserve_bytes']) is not int
                                             or not 0 <= value['session_reserve_bytes'] <= 4 * 1024**3):
        raise ValueError('memlat_oxe_flash_session_reserve_must_be_0_to_4_gib')
    if 'fallback' in value:
        fallback = value['fallback']
        if (not isinstance(fallback, dict) or set(fallback) != {'cache_gib', 'admission'}
                or type(fallback['cache_gib']) is not int or fallback['cache_gib'] != 18 or value['cache_gib'] not in (20, 21)):
            raise ValueError('memlat_oxe_flash_fallback_must_be_18_under_20')
        _validate_flash_admission(fallback['admission'], 18)
        if fallback['admission']['measured_peak_bytes'] >= value['admission']['measured_peak_bytes']:
            raise ValueError('memlat_oxe_flash_fallback_must_need_less_memory')


# FLASH-PREFIX-1: the engine's persistent prefix store for Flash (system prompt + tools state on disk, reused after
# an idle unload or reload). Off unless the profile sets {'enabled': true}; the controller adds the flags.
FLASH_PREFIX_KEYS = {'enabled', 'max_bytes', 'min_tokens', 'ttl_hours'}
FLASH_PREFIX_DEFAULTS = {'enabled': False, 'max_bytes': 2 * 1024**3, 'min_tokens': 1024, 'ttl_hours': 168}


def flash_prefix_store(value):
    """The validated prefix-store setting with defaults filled in; ValueError when malformed."""
    if not isinstance(value, dict) or set(value) - FLASH_PREFIX_KEYS or type(value.get('enabled')) is not bool:
        raise ValueError('memlat_oxe_flash_prefix_store_invalid')
    out = {**FLASH_PREFIX_DEFAULTS, **value}
    # one entry is ~285 MB at a 5.5K-token system prompt (118 MB recurrent + ~30 KB/token + logits)
    if (type(out['max_bytes']) is not int or not 512 * 1024**2 <= out['max_bytes'] <= 16 * 1024**3
            or type(out['min_tokens']) is not int or not 256 <= out['min_tokens'] <= 65536
            or type(out['ttl_hours']) is not int or not 0 <= out['ttl_hours'] <= 24 * 90):
        raise ValueError('memlat_oxe_flash_prefix_store_invalid')
    return out


def _validate_flash_admission(admission, cache_gib):
    required = {'measured_peak_bytes', 'admitted_budget_bytes'}
    if not isinstance(admission, dict) or not required <= set(admission) <= required | {'minimum_available_bytes'}:
        raise ValueError('memlat_oxe_flash_admission_invalid')
    peak, budget = admission['measured_peak_bytes'], admission['admitted_budget_bytes']
    minimum = admission.get('minimum_available_bytes', 0)
    if type(peak) is not int or type(budget) is not int or peak <= 0 or budget <= 0:
        raise ValueError('memlat_oxe_flash_admission_invalid')
    if type(minimum) is not int or minimum < 0:
        raise ValueError('memlat_oxe_flash_admission_invalid')
    if not peak <= budget <= FLASH_CAP_BYTES or peak < cache_gib * 1024**3 + FLASH_DENSE_LOWER_BOUND:
        raise ValueError('memlat_oxe_flash_admission_invalid')


def flash_model_sha(shards):
    """Digest of the four Flash shard hashes, in shard order."""
    if not isinstance(shards, list) or len(shards) != 4 or not all(
            isinstance(row, dict) and isinstance(row.get('path'), str)
            and isinstance(row.get('sha256'), str) and re.fullmatch(r'[0-9a-f]{64}', row['sha256'])
            and type(row.get('bytes')) is int and row['bytes'] > 0 for row in shards):
        raise ValueError('memlat_oxe_flash_shards_required')
    return hashlib.sha256('\n'.join(row['sha256'] for row in shards).encode()).hexdigest()


def _apply_oxe_flash(result, flash):
    """Switch the Flash profile to the Rust engine; the model identity is recomputed from shards."""
    shards = (result.get('bigmoe') or {}).get('shards')
    model_sha = flash_model_sha(shards)
    # Without the profile's own flash_memory scope the engine lives under the 28 GiB unit limit.
    cap = FLASH_CAP_BYTES if isinstance(result.get('flash_memory'), dict) else 28 * 1024**3
    rows = [(flash['cache_gib'], flash['admission'])]
    if 'fallback' in flash:
        rows.append((flash['fallback']['cache_gib'], flash['fallback']['admission']))
    if any(admission['admitted_budget_bytes'] > cap or (cache == 21 and cap < FLASH_CAP_BYTES) for cache, admission in rows):
        raise ValueError('memlat_oxe_flash_admission_invalid')
    result['backend_kind'] = 'oxe'
    for key in ENGINE_KEYS:
        result[key] = flash[key]
    result['oxe_args'] = copy.deepcopy(flash['oxe_args'])
    result['extra_args'] = []
    result['threads'] = flash.get('threads', 8)
    environment = copy.deepcopy(flash.get('environment', {}))
    if environment.setdefault(OXE_MODEL_SHA_ENV, model_sha) != model_sha:
        raise ValueError('memlat_oxe_model_sha_must_match_profile')
    result['environment'] = environment
    result['sha256'] = model_sha
    cache = result.get('prefix_cache')
    result['prefix_cache'] = {**(cache if isinstance(cache, dict) else {}), 'enabled': False}
    result['validation'] = copy.deepcopy(flash.get('validation', {}))
    result['execution_policy_mode'] = 'memlat'
    result['oxe_flash'] = {'cache_gib': flash['cache_gib'], 'sidecar_path': flash['sidecar_path'],
                           'sidecar_bytes': flash['sidecar_bytes'], 'admission': copy.deepcopy(flash['admission']),
                           'idle_unload_seconds': flash.get('idle_unload_seconds', 600),
                           'shards': [{'path': row['path'], 'bytes': row['bytes'], 'sha256': row['sha256']}
                                      for row in shards]}
    for key in ('cache_pref', 'desktop_floor_gib'):
        if key in flash:
            result['oxe_flash'][key] = flash[key]
    if 'swap_tolerant_guard' in flash:
        result['oxe_flash']['swap_tolerant_guard'] = copy.deepcopy(flash['swap_tolerant_guard'])
    if 'session_reserve_bytes' in flash:
        result['oxe_flash']['session_reserve_bytes'] = flash['session_reserve_bytes']
    if 'prefix_store' in flash:
        result['oxe_flash']['prefix_store'] = flash_prefix_store(flash['prefix_store'])
    if 'idle_qos' in flash:
        result['oxe_flash']['idle_qos'] = copy.deepcopy(flash['idle_qos'])
    if 'fallback' in flash:
        result['oxe_flash']['fallback'] = copy.deepcopy(flash['fallback'])
    return result


def _validate_oxe(value):
    if not isinstance(value, dict) or set(value) - OXE_KEYS or not OXE_REQUIRED <= set(value):
        raise ValueError('memlat_oxe_override_requires_binary_sha256_and_args')
    _validate({k: value[k] for k in ENGINE_KEYS})
    args = value['oxe_args']
    if not isinstance(args, list) or not args or not all(isinstance(a, str) and not any(c in a for c in '\n\r\0') for a in args):
        raise ValueError('memlat_oxe_args_must_be_strings')
    if 'threads' in value and (type(value['threads']) is not int or not 1 <= value['threads'] <= 12):
        raise ValueError('invalid_memlat_integer')
    if 'rss_proof' in value:
        validate_rss_proof(value['rss_proof'])
    if 'context_size' in value and (type(value['context_size']) is not int or not 256 <= value['context_size'] <= LONG_CONTEXT_CAP):
        raise ValueError('invalid_memlat_context_size')
    # Above the standard cap only with an RSS proof for this engine and at least this context.
    if 'context_size' in value and value['context_size'] > BASE_CONTEXT_CAP and (
            'rss_proof' not in value or value['rss_proof']['engine_sha256'] != value['server_sha256']
            or value['rss_proof']['context'] < value['context_size']
            or value['rss_proof']['peak_rss_bytes'] > RSS_PROOF_PEAK_MAX):
        raise ValueError('oxe_context_above_131072_needs_rss_proof')
    environment = value.get('environment', {})
    if not isinstance(environment, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in environment.items()):
        raise ValueError('memlat_environment_must_map_strings')
    if any(k.startswith(LEGACY_ENV_PREFIXES) for k in environment):
        raise ValueError('memlat_oxe_environment_excludes_classic_engine_keys')
    if not isinstance(value.get('validation', {}), dict):
        raise ValueError('memlat_oxe_validation_must_be_mapping')


def _apply_oxe(result, oxe):
    """Switch a classic profile to the Rust engine; identity gates stay stock."""
    result['backend_kind'] = 'oxe'
    for key in ENGINE_KEYS:
        result[key] = oxe[key]
    result['oxe_args'] = copy.deepcopy(oxe['oxe_args'])
    # Classic llama.cpp flags and GGML_/LLAMA_ environment never reach Rust.
    result['extra_args'] = []
    if 'threads' in oxe:
        result['threads'] = oxe['threads']
    if 'context_size' in oxe:
        result['context_size'] = oxe['context_size']
    if 'rss_proof' in oxe:
        result['rss_proof'] = copy.deepcopy(oxe['rss_proof'])
    if result.get('context_size', 0) > BASE_CONTEXT_CAP and result.get('model_alias') not in ORNITH_CLASS:
        raise ValueError('oxe_context_above_131072_is_ornith_only')
    environment = copy.deepcopy(oxe.get('environment', {}))
    model_sha = result.get('sha256')
    if environment.setdefault(OXE_MODEL_SHA_ENV, model_sha) != model_sha or not isinstance(model_sha, str):
        raise ValueError('memlat_oxe_model_sha_must_match_profile')
    result['environment'] = environment
    # The external llama.cpp PrefixStore is not qualified with Rust.
    cache = result.get('prefix_cache')
    result['prefix_cache'] = {**(cache if isinstance(cache, dict) else {}), 'enabled': False}
    result['validation'] = copy.deepcopy(oxe.get('validation', {}))
    result['execution_policy_mode'] = 'memlat'
    return result


def _validate(overrides):
    if not isinstance(overrides, dict) or set(overrides) - KEYS:
        raise ValueError('unsupported_memlat_override')
    if overrides.get('spec_type') == 'none' and 'spec_draft_n_max' in overrides:
        raise ValueError('draft_count_with_disabled_speculation')
    # The engine binary and its pinned hash change together or not at all.
    if any(k in overrides for k in ENGINE_KEYS) and not all(k in overrides for k in ENGINE_KEYS):
        raise ValueError('memlat_engine_override_requires_binary_and_sha256')
    for key, value in overrides.items():
        if key == 'oxe':
            _validate_oxe(value)
        elif key == 'oxe_flash':
            _validate_oxe_flash(value)
        elif key == 'server_binary':
            if not isinstance(value, str) or not Path(value).is_absolute() or any(c in value for c in '\n\r\0'):
                raise ValueError('memlat_server_binary_must_be_absolute_path')
        elif key == 'server_sha256':
            if not isinstance(value, str) or not re.fullmatch(r'[0-9a-f]{64}', value):
                raise ValueError('memlat_server_sha256_must_be_lowercase_hex')
        elif key == 'extra_args':
            if not isinstance(value, list) or not all(isinstance(v, str) and not any(c in v for c in '\n\r\0') for v in value):
                raise ValueError('memlat_extra_args_must_be_strings')
        elif key == 'environment':
            if not isinstance(value, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in value.items()):
                raise ValueError('memlat_environment_must_map_strings')
        elif key == 'prefix_cache_enabled':
            if type(value) is not bool:
                raise ValueError('memlat_prefix_cache_enabled_must_be_boolean')
        elif key in ('cpu_strict', 'cpu_strict_batch'):
            if type(value) is not int or value not in (0, 1):
                raise ValueError('memlat_cpu_strict_must_be_zero_or_one')
        elif key in ('cpu_mask', 'cpu_mask_batch'):
            if not isinstance(value, str) or not re.fullmatch(r'0x[0-9a-fA-F]{1,64}', value) or int(value, 16) == 0:
                raise ValueError('invalid_memlat_cpu_mask')
        elif key == 'spec_type':
            if value not in ('none', 'draft-mtp'):
                raise ValueError('unsupported_memlat_speculation')
        elif type(value) is not int or not 1 <= value <= (32 if key == 'spec_draft_n_max' else 256):
            raise ValueError('invalid_memlat_integer')


def resolve(profile, *, available=None, oxe_enabled=None, oxe_flash_enabled=None):
    """Return a detached effective snapshot; never change identity/readiness gates.

    No overrides or no detected MEMLAT returns the stock profile exactly. Call
    once for launch and retain the result for that native child's lifecycle.
    A top-level ``rust_engine`` block switches the profile to the Rust engine whenever its
    switch is on (the default), with or without MEMLAT.
    A nested ``memlat_overrides.oxe`` (historical location) replaces the classic MEMLAT
    overrides on a MEMLAT boot unless LOCAL_AI_ENABLE_OXE=0; otherwise the classic MEMLAT
    overrides apply. A Flash profile (FLASH_KIND) accepts only ``oxe_flash`` and switches
    to the Rust Flash engine unless LOCAL_AI_ENABLE_OXE_FLASH=0; otherwise it stays stock.
    """
    result = copy.deepcopy(profile)
    if oxe_enabled is None:
        oxe_enabled = switch_on(OXE_ENABLE_ENV)
    if oxe_flash_enabled is None:
        oxe_flash_enabled = switch_on(OXE_FLASH_ENABLE_ENV)
    rust = result.get(RUST_ENGINE_KEY)
    if rust is not None:
        # M4: the top-level block is not MEMLAT-gated. Classic MEMLAT tuning may coexist, but the
        # same engine must not be configured in both places.
        if not isinstance(rust, dict) or not rust or set(rust) - set(RUST_KEYS):
            raise ValueError('rust_engine_requires_oxe_or_oxe_flash')
        nested = result.get('memlat_overrides')
        if isinstance(nested, dict) and set(rust) & set(nested):
            raise ValueError('rust_engine_also_in_memlat_overrides')
        _validate(rust)
        if result.get('backend_kind') == FLASH_KIND:
            if 'oxe' in rust:
                raise ValueError('rust_engine_oxe_requires_classic_backend')
            if oxe_flash_enabled:
                return _apply_oxe_flash(result, rust['oxe_flash'])
            return result
        if result.get('backend_kind') not in (None, 'llama'):
            raise ValueError('rust_engine_requires_classic_or_flash_backend')
        if 'oxe_flash' in rust:
            raise ValueError('memlat_oxe_flash_requires_flash_backend')
        if oxe_enabled:
            return _apply_oxe(result, rust['oxe'])
    overrides = result.get('memlat_overrides')
    if overrides is None:
        return result
    _validate(overrides)
    if not overrides or not (memlat_available() if available is None else available):
        return result
    if result.get('backend_kind') == FLASH_KIND:
        if set(overrides) - {'oxe_flash'}:
            raise ValueError('memlat_overrides_require_classic_backend')
        if oxe_flash_enabled:
            return _apply_oxe_flash(result, overrides['oxe_flash'])
        return result
    if result.get('backend_kind') not in (None, 'llama'):
        raise ValueError('memlat_overrides_require_classic_backend')
    if 'oxe_flash' in overrides:
        raise ValueError('memlat_oxe_flash_requires_flash_backend')
    if 'oxe' in overrides and oxe_enabled:
        return _apply_oxe(result, overrides['oxe'])
    if 'threads' in overrides:
        result['threads'] = overrides['threads']
    args = copy.deepcopy(overrides.get('extra_args', result.get('extra_args', [])))
    if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
        raise ValueError('invalid_profile_extra_args')
    replaced = {alias for key in overrides if key in FLAGS for alias in FLAGS[key]}
    if overrides.get('spec_type') == 'none':
        replaced.update(FLAGS['spec_draft_n_max'])
    # Top-level --threads is authoritative; remove aliases if overriding it.
    if 'threads' in overrides:
        replaced.update(('--threads', '-t'))
    normalized = []
    i = 0
    while i < len(args):
        name = args[i].split('=', 1)[0]
        if name in replaced:
            if '=' not in args[i]:
                if i + 1 >= len(args) or args[i + 1].startswith('--'):
                    raise ValueError('missing_profile_flag_value')
                i += 1
        else:
            normalized.append(args[i])
        i += 1
    for key, aliases in FLAGS.items():
        if key in overrides:
            normalized.extend((aliases[0], str(overrides[key])))
    result['extra_args'] = normalized
    for key in ENGINE_KEYS:
        if key in overrides:
            result[key] = overrides[key]
    if 'environment' in overrides:
        base = result.get('environment', {})
        if not isinstance(base, dict):
            raise ValueError('invalid_profile_environment')
        result['environment'] = {**base, **overrides['environment']}
    # small_batch_limit is applied after the environment merge and wins.
    if 'small_batch_limit' in overrides:
        result.setdefault('environment', {})['LLAMA_CPU_SMALL_BATCH_LIMIT'] = str(overrides['small_batch_limit'])
    if 'prefix_cache_enabled' in overrides:
        cache = result.get('prefix_cache')
        if not isinstance(cache, dict):
            raise ValueError('explicit_base_prefix_cache_required')
        cache['enabled'] = overrides['prefix_cache_enabled']
    if 'memlat_validation' in result:
        result['validation'] = copy.deepcopy(result['memlat_validation'])
    result['execution_policy_mode'] = 'memlat'
    return result
