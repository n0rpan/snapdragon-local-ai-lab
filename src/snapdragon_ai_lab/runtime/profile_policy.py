"""Read-only MEMLAT detection and explicit, per-profile execution overrides."""
import copy
import hashlib
import os
from pathlib import Path
import re

FLAGS = {'threads_batch': ('--threads-batch', '-tb'),
         'cpu_mask': ('--cpu-mask', '-C'), 'cpu_mask_batch': ('--cpu-mask-batch', '-Cb'),
         'cpu_strict': ('--cpu-strict',), 'cpu_strict_batch': ('--cpu-strict-batch',),
         'spec_type': ('--spec-type',), 'spec_draft_n_max': ('--spec-draft-n-max',)}
ENGINE_KEYS = ('server_binary', 'server_sha256')
KEYS = {'threads', 'small_batch_limit', 'prefix_cache_enabled', 'environment', 'extra_args', 'oxe', 'oxe_flash', *ENGINE_KEYS, *FLAGS}
# Nested MEMLAT-only Rust engine switch; applied only when this is set to '1'.
OXE_ENABLE_ENV = 'LOCAL_AI_ENABLE_OXE'
OXE_KEYS = {'server_binary', 'server_sha256', 'oxe_args', 'threads', 'context_size', 'environment', 'validation'}
OXE_REQUIRED = {'server_binary', 'server_sha256', 'oxe_args'}
OXE_MODEL_SHA_ENV = 'OXE_MODEL_SHA256'
# Separate MEMLAT-only switch for the Rust Flash engine; independent of LOCAL_AI_ENABLE_OXE.
OXE_FLASH_ENABLE_ENV = 'LOCAL_AI_ENABLE_OXE_FLASH'
FLASH_KIND = 'bigmoe-cpu-stream'
OXE_FLASH_KEYS = {'server_binary', 'server_sha256', 'oxe_args', 'threads', 'environment', 'validation',
                  'cache_gib', 'sidecar_path', 'sidecar_bytes', 'admission', 'idle_unload_seconds', 'fallback',
                  'session_reserve_bytes', 'idle_qos'}
OXE_FLASH_REQUIRED = {'server_binary', 'server_sha256', 'oxe_args', 'cache_gib', 'sidecar_path', 'sidecar_bytes', 'admission'}
FLASH_DENSE_LOWER_BOUND = 4522949120
# Absolute bound; the 28 GiB limit without a flash_memory policy is enforced at launch (oxe_runtime).
FLASH_CAP_BYTES = 29 * 1024**3
LEGACY_ENV_PREFIXES = ('GGML_', 'LLAMA_')


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
    if 'session_reserve_bytes' in flash:
        result['oxe_flash']['session_reserve_bytes'] = flash['session_reserve_bytes']
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
    if 'context_size' in value and (type(value['context_size']) is not int or not 256 <= value['context_size'] <= 131072):
        raise ValueError('invalid_memlat_context_size')
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
    A nested ``oxe`` override replaces the classic MEMLAT overrides only when
    LOCAL_AI_ENABLE_OXE=1; otherwise the classic MEMLAT overrides apply.
    A Flash profile (FLASH_KIND) accepts only ``oxe_flash`` and switches to the
    Rust Flash engine only when LOCAL_AI_ENABLE_OXE_FLASH=1; otherwise it stays stock.
    """
    result = copy.deepcopy(profile)
    overrides = result.get('memlat_overrides')
    if overrides is None:
        return result
    _validate(overrides)
    if not overrides or not (memlat_available() if available is None else available):
        return result
    if result.get('backend_kind') == FLASH_KIND:
        if set(overrides) - {'oxe_flash'}:
            raise ValueError('memlat_overrides_require_classic_backend')
        if oxe_flash_enabled is None:
            oxe_flash_enabled = os.environ.get(OXE_FLASH_ENABLE_ENV) == '1'
        if oxe_flash_enabled:
            return _apply_oxe_flash(result, overrides['oxe_flash'])
        return result
    if result.get('backend_kind') not in (None, 'llama'):
        raise ValueError('memlat_overrides_require_classic_backend')
    if 'oxe_flash' in overrides:
        raise ValueError('memlat_oxe_flash_requires_flash_backend')
    if oxe_enabled is None:
        oxe_enabled = os.environ.get(OXE_ENABLE_ENV) == '1'
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
