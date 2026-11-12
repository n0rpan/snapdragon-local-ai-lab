"""Rust engine (oxe) launch rules; no model loading during polling.

Production for the MoE profiles on MEMLAT since 2026-11-10 (enabled with
LOCAL_AI_ENABLE_OXE=1; see docs/RUST-DEPLOYMENT.md). The argument template is
explicit per profile. This module does not qualify an engine or change the
installed/default profile; without the flag the classic engines resolve.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import signal
import stat
import subprocess
import tempfile
import threading

if __package__:
    from . import flash_cache_pref, memory_policy, state_cache
else:
    import flash_cache_pref
    import memory_policy
    import state_cache

KIND = 'oxe'
ENABLE_ENV = 'LOCAL_AI_ENABLE_OXE'
# Flash (bigmoe-cpu-stream profile) switch; independent of ENABLE_ENV.
FLASH_ENABLE_ENV = 'LOCAL_AI_ENABLE_OXE_FLASH'
GIB = 1024**3
FLASH_HOST_FLOOR_BYTES = 1536 * 1024**2
FLASH_CAP_BYTES = 28 * GIB
FLASH_DENSE_LOWER_BOUND = 4522949120
# 4.87 GiB = 4.62 GiB measured fixed overhead + 0.25 GiB margin. With the 1 GiB host floor
# this is the measured rule: required available = cache GiB + 5.87 GiB.
FLASH_ADMISSION_OVERHEAD_BYTES = 5229122682
# Optional idle-QoS hook: a ROOT unit instance per engine pid sets the CPU idle knob. The values are fixed
# here and in the root helper; the profile only switches the hook on.
FLASH_IDLE_QOS_UNIT = 'local-ai-flash-qos@{pid}.service'
FLASH_IDLE_QOS_KEYS = frozenset({'unit', 'cpu', 'latency_us'})
FLASH_IDLE_QOS_CPU = 8
FLASH_IDLE_QOS_LATENCY_US = 100
# FLASH-PREFIX-1: the Flash prefix store lives in the user's home (O-21: prompts stay on the encrypted partition,
# never next to the expert sidecar on the plain weights partition). One directory per model (W4-4 c).
FLASH_PREFIX_DIR = ('.cache', 'local-ai', 'snapdragon-engine', 'prefix-flash')
FLASH_PREFIX_RAW_FLAGS = ('--prefix-cache', '--prefix-cache-max-bytes', '--prefix-cache-min-tokens', '--prefix-cache-ttl-h')
# CACHE-PERFECT P2: the disk state store comes only from runtime.json `cache` (state_cache.py), never raw argv.
STATE_RAW_FLAGS = ('--state-cache', '--state-budget-bytes', '--state-ttl-secs')


def _validate(config):
    if config.get('backend_kind') != KIND:
        raise ValueError('oxe_backend_required')
    alias = config.get('model_alias')
    if not isinstance(alias, str) or not alias or len(alias) > 128 or any(ord(c) < 32 for c in alias):
        raise ValueError('oxe_model_alias_required')
    for key in ('server_binary', 'model_path'):
        value = config.get(key)
        if not isinstance(value, str) or not Path(value).is_absolute() or any(ord(c) < 32 for c in value):
            raise ValueError('oxe_absolute_path_required')
    for key in ('server_sha256', 'sha256'):
        if not isinstance(config.get(key), str) or not re.fullmatch('[0-9a-f]{64}', config[key]):
            raise ValueError('oxe_pinned_identity_required')
    for key, maximum in (('threads', 12), ('context_size', 131072), ('expected_bytes', 64 * 1024**3)):
        if type(config.get(key)) is not int or not 1 <= config[key] <= maximum:
            raise ValueError('oxe_invalid_integer')
    if config.get('prefix_cache', {}).get('enabled'):
        raise ValueError('oxe_external_slot_restore_not_qualified')
    args = config.get('oxe_args')
    if not isinstance(args, list) or not args or len(args) > 64:
        raise ValueError('oxe_explicit_cli_template_required')
    for arg in args:
        if not isinstance(arg, str) or any(ord(c) < 32 for c in arg):
            raise ValueError('oxe_invalid_argument')
        if arg.split('=', 1)[0] in ('--host', '--port', '--listen', '--bind'):
            raise ValueError('oxe_controller_owns_loopback_bind')
        if arg.split('=', 1)[0] in STATE_RAW_FLAGS:
            raise ValueError('oxe_state_cache_flags_only_via_runtime_key')
        if '{' in arg or '}' in arg:
            if arg not in ('{model}', '{context}', '{threads}', '{alias}'):
                raise ValueError('oxe_unknown_argument_placeholder')
    if is_flash(config):
        _validate_flash(config)


def is_flash(config):
    return isinstance(config.get('oxe_flash'), dict)


def _adjacent(args, flag, value):
    return any(args[i] == flag and args[i + 1] == value for i in range(len(args) - 1))


def _flash_digest(shards):
    return hashlib.sha256('\n'.join(row['sha256'] for row in shards).encode()).hexdigest()


def flash_limits(config):
    """(host floor, memory cap) for a Flash config: the measured flash_memory pair when the
    profile carries it (own 29 GiB scope, 1 GiB floor; memory_policy validates it at launch),
    else the chat unit's 28 GiB with the default 1.5 GiB floor."""
    policy = config.get('flash_memory')
    if isinstance(policy, dict) and all(type(policy.get(key)) is int for key in ('host_floor_bytes', 'scope_max_bytes')):
        return policy['host_floor_bytes'], policy['scope_max_bytes']
    return FLASH_HOST_FLOOR_BYTES, FLASH_CAP_BYTES


def _validate_flash_admission(cache_gib, admission, cap=FLASH_CAP_BYTES):
    # A 21 GiB cache was only measured under the 29 GiB scope.
    if type(cache_gib) is not int or cache_gib not in (14, 16, 18, 20, 21) or (cache_gib == 21 and cap <= FLASH_CAP_BYTES):
        raise ValueError('oxe_flash_cache_gib_invalid')
    required = {'measured_peak_bytes', 'admitted_budget_bytes'}
    if not isinstance(admission, dict) or not required <= set(admission) <= required | {'minimum_available_bytes'}:
        raise ValueError('oxe_flash_admission_invalid')
    peak, budget = admission['measured_peak_bytes'], admission['admitted_budget_bytes']
    minimum = admission.get('minimum_available_bytes', 0)
    if type(peak) is not int or type(budget) is not int or peak <= 0 or budget <= 0:
        raise ValueError('oxe_flash_admission_invalid')
    if type(minimum) is not int or minimum < 0:
        raise ValueError('oxe_flash_admission_invalid')
    if not peak <= budget <= cap or peak < cache_gib * GIB + FLASH_DENSE_LOWER_BOUND:
        raise ValueError('oxe_flash_admission_invalid')


def _validate_flash_idle_qos(value):
    if (not isinstance(value, dict) or set(value) != FLASH_IDLE_QOS_KEYS or value['unit'] != FLASH_IDLE_QOS_UNIT
            or type(value['cpu']) is not int or value['cpu'] != FLASH_IDLE_QOS_CPU
            or type(value['latency_us']) is not int or value['latency_us'] != FLASH_IDLE_QOS_LATENCY_US):
        raise ValueError('oxe_flash_idle_qos_invalid')


def flash_idle_qos_unit(config, pid):
    """The idle-QoS unit for this engine pid, or None when the profile has no idle_qos."""
    if type(pid) is not int or pid <= 0:
        raise ValueError('oxe_flash_idle_qos_pid_invalid')
    flash = config.get('oxe_flash') if isinstance(config, dict) else None
    if not isinstance(flash, dict) or 'idle_qos' not in flash:
        return None
    return FLASH_IDLE_QOS_UNIT.format(pid=pid)


def flash_idle_qos_command(config, pid, verb):
    """systemctl argv for the idle-QoS unit (verb 'start' or 'stop'), or None without idle_qos."""
    if verb not in ('start', 'stop'):
        raise ValueError('oxe_flash_idle_qos_verb_invalid')
    unit = flash_idle_qos_unit(config, pid)
    return None if unit is None else ['systemctl', verb, '--no-block', unit]


def _validate_flash(config):
    """Admission and identity shape only; no file access."""
    flash = config['oxe_flash']
    cache_gib = flash.get('cache_gib')
    cap = flash_limits(config)[1]
    _validate_flash_admission(cache_gib, flash.get('admission'), cap)
    if 'session_reserve_bytes' in flash and (type(flash['session_reserve_bytes']) is not int
                                             or not 0 <= flash['session_reserve_bytes'] <= 4 * GIB):
        raise ValueError('oxe_flash_session_reserve_invalid')
    if 'fallback' in flash:
        fallback = flash['fallback']
        if (not isinstance(fallback, dict) or set(fallback) != {'cache_gib', 'admission'}
                or fallback['cache_gib'] != 18 or cache_gib not in (20, 21)):
            raise ValueError('oxe_flash_fallback_invalid')
        _validate_flash_admission(18, fallback['admission'], cap)
        if fallback['admission']['measured_peak_bytes'] >= flash['admission']['measured_peak_bytes']:
            raise ValueError('oxe_flash_fallback_invalid')
    sidecar = flash.get('sidecar_path')
    if not isinstance(sidecar, str) or not Path(sidecar).is_absolute() or any(ord(c) < 32 for c in sidecar):
        raise ValueError('oxe_flash_sidecar_invalid')
    if type(flash.get('sidecar_bytes')) is not int or flash['sidecar_bytes'] <= 0:
        raise ValueError('oxe_flash_sidecar_invalid')
    if 'idle_unload_seconds' in flash and (type(flash['idle_unload_seconds']) is not int
                                           or not 60 <= flash['idle_unload_seconds'] <= 3600):
        raise ValueError('oxe_flash_idle_unload_invalid')
    if 'idle_qos' in flash:
        _validate_flash_idle_qos(flash['idle_qos'])
    _validate_flash_prefix(config)
    shards = flash.get('shards')
    if not isinstance(shards, list) or len(shards) != 4 or not all(
            isinstance(row, dict) and isinstance(row.get('path'), str) and Path(row['path']).is_absolute()
            and type(row.get('bytes')) is int and row['bytes'] > 0
            and isinstance(row.get('sha256'), str) and re.fullmatch('[0-9a-f]{64}', row['sha256']) for row in shards):
        raise ValueError('oxe_flash_shards_invalid')
    if config.get('sha256') != _flash_digest(shards):
        raise ValueError('oxe_flash_model_sha_mismatch')
    args = config['oxe_args']
    cache_matches = _adjacent(args, '--flash-cache-gib', str(cache_gib))
    if 'swap_tolerant_guard' in flash:
        memory_policy.swap_tolerant_policy(flash['swap_tolerant_guard'])
    if flash_cache_pref.enabled({}, flash):
        flash_cache_pref.read_pref({}, flash)
        flash_cache_pref.desktop_floor(flash)
        cache_matches = cache_matches or _adjacent(args, '--flash-cache-gib', 'auto')
    if not _adjacent(args, '--engine', 'flash') or not cache_matches:
        raise ValueError('oxe_flash_args_must_match_admission')


def _validate_flash_prefix(config):
    """The prefix store comes only from oxe_flash.prefix_store (never raw argv flags), and not together with the
    unqualified warm-up reads (--flash-warmup 1, --flash-hotset, --session-cache)."""
    args = config['oxe_args']
    if any(a.split('=', 1)[0] in FLASH_PREFIX_RAW_FLAGS for a in args):
        raise ValueError('oxe_flash_prefix_flags_only_via_profile_key')
    setting = config['oxe_flash'].get('prefix_store')
    if setting is None:
        return
    if (not isinstance(setting, dict) or set(setting) != {'enabled', 'max_bytes', 'min_tokens', 'ttl_hours'}
            or type(setting['enabled']) is not bool
            or any(type(setting[k]) is not int or setting[k] < 0 for k in ('max_bytes', 'min_tokens', 'ttl_hours'))):
        raise ValueError('oxe_flash_prefix_store_invalid')
    if setting['enabled'] and (_adjacent(args, '--flash-warmup', '1') or '--flash-warmup=1' in args
                               or any(a.split('=', 1)[0] in ('--flash-hotset', '--session-cache') for a in args)):
        raise ValueError('oxe_flash_prefix_store_with_unqualified_warmup')


def flash_prefix_dir(config, home=None):
    """The prefix-store directory when the profile enables it, else None. Refused (ValueError) when it would lie
    inside the expert sidecar's directory (O-21)."""
    if not is_flash(config):
        return None
    setting = config['oxe_flash'].get('prefix_store')
    if not isinstance(setting, dict) or setting.get('enabled') is not True:
        return None
    if not Path(home if home is not None else Path.home()).is_absolute():
        raise ValueError('oxe_flash_prefix_dir_not_absolute')
    # FLASH_PREFIX_DIR by default; below LOCAL_AI_STATE_CACHE_ROOT / cache.root when overridden
    directory = state_cache.store_dir('prefix-flash', home, config)
    sidecar_dir = Path(os.path.realpath(Path(config['oxe_flash']['sidecar_path']).parent))
    resolved = Path(os.path.realpath(directory))
    if resolved == sidecar_dir or sidecar_dir in resolved.parents:
        raise ValueError('oxe_flash_prefix_dir_on_weights_partition')
    return directory


def flash_prefix_args(config, home=None):
    """oxe-server flags of the enabled prefix store (empty when off)."""
    directory = flash_prefix_dir(config, home)
    if directory is None:
        return []
    setting = config['oxe_flash']['prefix_store']
    return ['--prefix-cache', str(directory), '--prefix-cache-max-bytes', str(setting['max_bytes']),
            '--prefix-cache-min-tokens', str(setting['min_tokens']), '--prefix-cache-ttl-h', str(setting['ttl_hours'])]


def flash_files_ready(config):
    """Metadata only: shard and sidecar regular files with the admitted sizes."""
    try:
        flash = config['oxe_flash']
        rows = [(row['path'], row['bytes']) for row in flash['shards']]
        rows.append((flash['sidecar_path'], flash['sidecar_bytes']))
        for path, size in rows:
            info = os.stat(path)
            if not stat.S_ISREG(info.st_mode) or info.st_size != size:
                return False
        return True
    except (OSError, KeyError, TypeError, ValueError):
        return False


def flash_rule_required_available(config, cache_gib):
    return (cache_gib * GIB + FLASH_ADMISSION_OVERHEAD_BYTES + flash_limits(config)[0]
            + config['oxe_flash'].get('session_reserve_bytes', 0))


def flash_required_available(config):
    flash = config['oxe_flash']
    measured = flash['admission']['measured_peak_bytes'] + flash_limits(config)[0]
    return max(measured, flash_rule_required_available(config, flash['cache_gib']),
               flash['admission'].get('minimum_available_bytes', 0))


def flash_admit(config, available, memory_max=None):
    """Raise ValueError when the admitted Flash budget cannot start now."""
    admission = config['oxe_flash']['admission']
    if type(available) is not int or available < flash_required_available(config):
        raise ValueError('oxe_flash_insufficient_available_memory')
    if admission['admitted_budget_bytes'] > flash_limits(config)[1] or (
            type(memory_max) is int and admission['measured_peak_bytes'] > memory_max):
        raise ValueError('oxe_flash_budget_exceeds_cap')


def flash_swap_tolerant_guard(config, available, mem_total, env=None, swap_free=None):
    """F127 (O-30): the swap-tolerant MemoryGuard policy for this launch, else None (standard guard).

    Only when ALL hold: the profile opts in (``oxe_flash.swap_tolerant_guard``; public default off), the
    Flash flash_memory scope policy is set (own scope, swap.max 0), the preference is 20 or 18, and the
    engine's rule (mirrored) admits it on MemTotal (reason ``preferred-swap-tolerant``). auto/16/14 and
    a MemAvailable fallback keep the standard 1 GiB floor. Native /props is checked again after ready.
    """
    flash = config.get('oxe_flash')
    if not isinstance(flash, dict):
        return None
    policy = memory_policy.swap_tolerant_policy(flash.get('swap_tolerant_guard'))
    if policy is None or not isinstance(config.get('flash_memory'), dict) or type(mem_total) is not int:
        return None
    if not flash_cache_pref.enabled(env, flash) or type(available) is not int:
        return None
    pref = flash_cache_pref.read_pref(env, flash)
    if pref not in ('20', '18'):
        return None
    _, reason = flash_cache_pref.expected_tier(available, pref, config['context_size'], mem_total,
                                               flash_cache_pref.desktop_floor(flash), swap_free)
    return policy if reason == 'preferred-swap-tolerant' else None


def flash_select(config, available, memory_max=None, *, env=None, mem_total=None, swap_free=None):
    """The config to launch: as given when its admission passes, else its smaller-cache
    fallback (a detached copy with the cache argument and admission swapped) when that
    passes. Raises the primary admission's ValueError when neither can start now."""
    if flash_cache_pref.enabled(env, config['oxe_flash']):
        # This opt-in mirror admits the native decision; it never replaces measured proof
        # or changes the MemoryGuard's host floor/PSI policy. Native /props is authoritative.
        _validate(config)
        flash = config['oxe_flash']
        pref = flash_cache_pref.read_pref(env, flash)
        floor = flash_cache_pref.desktop_floor(flash)
        if type(available) is not int or available < flash_limits(config)[0]:
            raise ValueError('oxe_flash_insufficient_available_memory')
        tier, reason = flash_cache_pref.expected_tier(
            available, pref, config['context_size'], mem_total, floor, swap_free)
        if tier is None:
            raise ValueError('oxe_flash_insufficient_available_memory')
        # S2: a v0.1.0 engine admits an explicit 20/18 on MemTotal without reading swap. When this mirror
        # (which does read swap) did not admit it swap-tolerant, pass the MemAvailable-rule tier instead, so
        # no engine version can take the MemTotal path on a host whose swap cannot hold the overflow.
        engine_pref = pref
        if pref in flash_cache_pref.SWAP_TOLERANT_TIERS_GIB_STR and reason != 'preferred-swap-tolerant':
            engine_pref = str(tier)
        estimated = (flash_cache_pref.tier_need_gib(tier, config['context_size'])
                     - flash_cache_pref.HARNESS_GROWTH_GIB - flash_cache_pref.FLOOR_GIB) * GIB
        cap = flash_limits(config)[1]
        if (flash['admission']['admitted_budget_bytes'] > cap or estimated > cap
                or (type(memory_max) is int and estimated > memory_max)):
            raise ValueError('oxe_flash_budget_exceeds_cap')
        selected = copy.deepcopy(config)
        selected['oxe_flash']['cache_pref'] = pref
        selected['oxe_args'] = flash_cache_pref.apply_pref(config['oxe_args'], engine_pref)
        # A profile-specified desktop reserve is passed exactly; otherwise native default 3 applies.
        if 'desktop_floor_gib' in flash:
            args, original, i = [], selected['oxe_args'], 0
            while i < len(original):
                if original[i] == '--flash-desktop-floor-gib':
                    if i + 1 >= len(original):
                        raise ValueError('flash_desktop_floor_invalid')
                    i += 2
                elif original[i].startswith('--flash-desktop-floor-gib='):
                    i += 1
                else:
                    args.append(original[i]); i += 1
            selected['oxe_args'] = args + ['--flash-desktop-floor-gib', str(floor)]
        _validate(selected)
        return selected
    try:
        flash_admit(config, available, memory_max)
        return config
    except ValueError:
        fallback = config['oxe_flash'].get('fallback')
        if not isinstance(fallback, dict):
            raise
    smaller = copy.deepcopy(config)
    flash = smaller['oxe_flash']
    args = smaller['oxe_args']
    for i in range(len(args) - 1):
        if args[i] == '--flash-cache-gib':
            args[i + 1] = str(fallback['cache_gib'])
    flash['cache_gib'], flash['admission'] = fallback['cache_gib'], copy.deepcopy(fallback['admission'])
    del flash['fallback']
    flash['fallback_active'] = True
    _validate(smaller)
    flash_admit(smaller, available, memory_max)
    return smaller


def flash_smallest_required_available(config):
    flash = config['oxe_flash']
    smallest = flash.get('fallback', flash)
    measured = smallest['admission']['measured_peak_bytes'] + flash_limits(config)[0]
    return max(measured, flash_rule_required_available(config, smallest['cache_gib']),
               smallest['admission'].get('minimum_available_bytes', 0))


def _bigmoe_runtime():
    if __package__:
        from . import bigmoe_runtime
    else:
        import bigmoe_runtime
    return bigmoe_runtime


def flash_power_failure(snapshot, *, starting=False):
    """Same battery rule as the BigMoe Flash path (start >= 30%, stop <= 20%)."""
    return _bigmoe_runtime().power_failure(snapshot, starting=starting)


class FlashPowerGuard:
    """Power-only watch on our Flash child; memory is watched by the classic MemoryGuard."""
    def __init__(self, process, *, power=None, interval=1):
        self.process = process
        self.power_telemetry = _bigmoe_runtime().PowerTelemetry(power)
        self.power = self.power_telemetry.sample
        self.interval = interval
        self.failure = None
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self.watch, daemon=True)

    def check(self):
        return flash_power_failure(self.power())

    def start(self):
        self.thread.start()

    def watch(self):
        while not self.stopped.is_set() and self.process.poll() is None:
            try:
                failure = self.check()
            except (OSError, ValueError, TypeError):
                failure = 'flash_guard_observation_failed'
            if failure:
                self.failure = failure
                try:
                    os.killpg(self.process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(self.process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    self.process.wait()
                return
            self.stopped.wait(self.interval)

    def stop(self):
        self.stopped.set()
        if self.thread.ident is not None:
            self.thread.join(timeout=12)


def switch_on(name, env=None):
    """M4: Rust engines are the default; only an explicit 0/off/false/no switches one off."""
    value = (os.environ if env is None else env).get(name)
    return value is None or value.strip().lower() not in ('0', 'off', 'false', 'no')


def eligible(config):
    """Metadata only. Never hash a binary, read weights, or wake a model."""
    switch = FLASH_ENABLE_ENV if is_flash(config) else ENABLE_ENV
    if not switch_on(switch) or config.get('ready') is not True:
        return False
    try:
        _validate(config)
        return True
    except (ValueError, TypeError, AttributeError):
        return False


def ineligible_reason(config):
    """M4: why a Rust profile cannot start, as a message that names the fix; None when it can.

    Metadata and one stat of the engine binary; never hashes or starts anything.
    """
    switch = FLASH_ENABLE_ENV if is_flash(config) else ENABLE_ENV
    fallback = f'or set {switch}=0 to use the classic llama.cpp fallback'
    if not switch_on(switch):
        return f'The Rust engine is switched off ({switch}=0).'
    if config.get('ready') is not True:
        return 'This model profile is not marked ready; finish its setup and validation first.'
    try:
        _validate(config)
    except (ValueError, TypeError, AttributeError) as error:
        return f'The Rust engine profile is incomplete ({error}); fix runtime.json {fallback}.'
    binary = Path(config['server_binary'])
    if not binary.is_file():
        return (f'The Rust engine binary is missing: {binary}. Build or install it '
                f'(docs/ENGINE-SELECTION.md) {fallback}.')
    return None


def command(config, port):
    if not eligible(config):
        raise ValueError('oxe_not_enabled_or_ready')
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError('oxe_invalid_loopback_port')
    values = {'{model}': config['model_path'], '{context}': str(config['context_size']),
              '{threads}': str(config['threads']), '{alias}': config['model_alias']}
    args = conversation_store_args(config, [values.get(arg, arg) for arg in config['oxe_args']])
    tail = [*flash_prefix_args(config), *state_cache_args(config)]
    if state_cache.override(config) is not None:
        # a reviewed override root (gates): the stores this launch writes exist, 0700
        every = args + tail
        for i, arg in enumerate(every):
            name, eq, value = arg.partition('=')
            if name in ('--prefix-cache', '--session-cache', '--state-cache'):
                path = value if eq else (every[i + 1] if i + 1 < len(every) else None)
                if path:
                    state_cache.make_private(path)
    return [config['server_binary'], '--host', '127.0.0.1', '--port', str(port), *args, *tail]


# Legacy conversation stores an oxe profile may name in its raw args (each takes one value).
LEGACY_STORE_FLAGS = {'--prefix-cache': 'prefix', '--session-cache': 'sessions'}
LEGACY_STORE_OPTIONS = ('--prefix-cache-max-bytes', '--prefix-cache-min-tokens', '--prefix-cache-ttl-h')


def conversation_store_args(config, args, home=None):
    """The profile's raw args with its legacy conversation stores (--prefix-cache, --session-cache)
    brought under the state cache's rules (non-Flash; Flash keeps its own prefix_store key):
    - runtime.json cache.disk_budget_bytes == 0 (all conversation disk persistence off): dropped,
      with their option flags, as the engine's own budget-0 does while it runs;
    - a root override (LOCAL_AI_STATE_CACHE_ROOT / cache.root): their directory moves to
      <root>/prefix and <root>/sessions (the weight caches stay where they are).
    ValueError for an invalid override (no silent fallback to the default directories)."""
    overridden = state_cache.override(config) is not None
    if overridden:
        state_cache.base(home, config)  # validates; raises for a bad override
    off = not is_flash(config) and state_cache.settings(config)['disk_budget_bytes'] == 0
    out, i = [], 0
    while i < len(args):
        arg = args[i]
        name, eq, value = arg.partition('=')
        store = LEGACY_STORE_FLAGS.get(name)
        option = name in LEGACY_STORE_OPTIONS and not is_flash(config)
        if (store or option) and off:
            i += 1 if eq else 2
            continue
        if store and overridden:
            target = str(state_cache.store_dir(store, home, config))
            out += [name + '=' + target] if eq else [name, target]
            i += 1 if eq else 2
            continue
        out.append(arg)
        i += 1
    return out


_STATE_CACHE_PROBE = {}


def _binary_holds(binary, needles):
    """True when the engine file holds every byte string in `needles` (read, never executed).
    Read once per (binary identity, needles); False on any failure."""
    try:
        fd = os.open(os.fspath(binary), os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
        with os.fdopen(fd, 'rb') as stream:
            st = os.fstat(stream.fileno())
            key = (os.fspath(binary), st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, needles)
            if key not in _STATE_CACHE_PROBE:
                if not stat.S_ISREG(st.st_mode) or not 0 < st.st_size <= 256 * 1024**2:
                    return False
                want, tail = set(needles), b''
                while want and (block := stream.read(1024**2)):
                    data = tail + block
                    want = {w for w in want if w not in data}
                    tail = data[-64:]
                _STATE_CACHE_PROBE[key] = not want
            return _STATE_CACHE_PROBE[key]
    except (OSError, TypeError, ValueError):
        return False


def engine_has_state_cache(binary):
    """Capability probe (P2 review): an engine build from before CACHE-PERFECT P2 exits on the
    unknown flag --state-cache, so the flags go only to a binary that holds their names (its
    usage text and argument table). The lab may then ship before, with or after the engine.
    Nothing is executed: the file is read once per binary identity (device, inode, size, mtime),
    at launch, as verify_engine reads it; False on any failure."""
    return _binary_holds(binary, frozenset(flag.encode() + b' ' for flag in STATE_RAW_FLAGS))


STATE_PRELOAD_ROUTE = '/state-cache/preload'


# The engine's usage text carries this literal (engine main.rs USAGE, "capabilities:" line); its
# helper CI greps the compiled binary for it. The route string itself is not a reliable needle: an
# optimised build need not keep a match arm's string literal whole (the frozen 2e9ff52 binary
# serves the route but holds no '/state-cache/preload' bytes).
STATE_PRELOAD_MARKER = 'oxe-capability state-preload-v1'


def engine_has_state_preload(binary):
    """CACHE-PERFECT P2b: the engine serves POST /state-cache/preload: its usage text holds the
    capability marker. Same read-only probe as engine_has_state_cache (nothing executed)."""
    return _binary_holds(binary, frozenset({STATE_PRELOAD_MARKER.encode()}))


def state_cache_args(config, home=None, probe=None):
    """CACHE-PERFECT P2 disk state store flags: MoE (non-Flash) oxe only, only when
    runtime.json `cache.disk_budget_bytes` > 0, and only for an engine that knows the flags
    (`engine_has_state_cache`; `probe` replaces it in tests). Flash keeps its prefix store in v1."""
    if is_flash(config):
        return []
    args = state_cache.engine_args(config, home)
    if not args:
        return []
    return args if (engine_has_state_cache if probe is None else probe)(config.get('server_binary')) else []


# B1 (v0.1.1): the backend port is internal. Each launch gets a fresh random bearer token in a
# 0600 file under the user's runtime directory; the engine reads the file named by API_KEY_ENV
# and refuses requests without "Authorization: Bearer <token>". Only the controller holds it.
# An environment variable (not a flag) so an older engine that predates the check still starts.
API_KEY_ENV = 'OXE_API_KEY_FILE'
RUNTIME_DIR_ENV = 'LOCAL_AI_RUNTIME_DIR'


def _private_dir(path):
    """`path` as a 0700 directory owned by this user, or None. Never follows a symlink."""
    try:
        os.mkdir(path, 0o700)
    except FileExistsError:
        pass
    except OSError:
        return None
    try:
        info = os.lstat(path)
    except OSError:
        return None
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        return None
    return Path(path)


def api_key_directory(environ=None):
    """Where per-launch backend tokens live: $LOCAL_AI_RUNTIME_DIR, else $XDG_RUNTIME_DIR/local-ai,
    else a fresh private temporary directory."""
    environ = os.environ if environ is None else environ
    explicit = environ.get(RUNTIME_DIR_ENV)
    if explicit:
        return _private_dir(explicit)
    runtime = environ.get('XDG_RUNTIME_DIR')
    if runtime and os.path.isabs(runtime) and os.path.isdir(runtime):
        found = _private_dir(os.path.join(runtime, 'local-ai'))
        if found is not None:
            return found
    return Path(tempfile.mkdtemp(prefix='local-ai-'))


def new_api_key(port, environ=None):
    """Create one per-launch token file (0600, O_EXCL, no symlink). Returns (token, path).
    Raises OSError when no private directory is usable: the caller must not launch without it."""
    directory = api_key_directory(environ)
    if directory is None:
        raise OSError('no private runtime directory for the backend token')
    token = secrets.token_urlsafe(32)
    path = directory / ('oxe-backend-%d-%s.key' % (int(port), secrets.token_hex(4)))
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    try:
        os.fchmod(fd, 0o600)
        os.write(fd, token.encode('ascii'))
    finally:
        os.close(fd)
    return token, path


def remove_api_key(path):
    if path is None:
        return
    try:
        os.unlink(path)
    except OSError:
        pass


def backend_headers(token):
    """Headers every controller request to an oxe backend carries."""
    return {'Authorization': 'Bearer ' + token} if token else {}


def launch_environment(inherited):
    """Drop inherited classic llama.cpp/GGML tuning from the Rust child."""
    return {k: v for k, v in inherited.items() if not k.startswith(('GGML_', 'LLAMA_'))}


def verify_engine(config):
    """Startup only, inside the controller's inference lease."""
    _validate(config)
    fd = os.open(config['server_binary'], os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    with os.fdopen(fd, 'rb') as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_uid not in (0, os.getuid()) or not 0 < before.st_size <= 256 * 1024**2:
            raise ValueError('oxe_bounded_owned_binary_required')
        h = hashlib.sha256()
        while block := stream.read(1024**2):
            h.update(block)
        after = os.fstat(stream.fileno())
    fields = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
    if any(getattr(before, k) != getattr(after, k) for k in fields) or h.hexdigest() != config['server_sha256']:
        raise ValueError('oxe_engine_identity_changed')


def health_ready(raw):
    try:
        return json.loads(raw).get('status') == 'ok'
    except (ValueError, AttributeError, TypeError):
        return False


GENERATION_PATHS = frozenset({'/completion', '/completions', '/v1/completions',
                              '/chat/completions', '/v1/chat/completions'})
LENGTH_FIELDS = ('max_tokens', 'max_completion_tokens', 'n_predict')


def default_output_tokens(config):
    """Per-profile ``default_max_tokens`` (opt-in in runtime.json), bounded by context.

    oxe-server stops a reply with finish_reason "length" when prompt + output
    reaches n_ctx (no error), so no prompt-dependent margin is needed here.
    """
    if not isinstance(config, dict):
        return None
    value, context = config.get('default_max_tokens'), config.get('context_size')
    if type(value) is not int or value < 1:
        return None
    return min(value, context) if type(context) is int and context > 0 else value


def request_defaults(payload, path, config=None):
    """Defaults for a request to the Rust engine; values the client sent always win.

    CACHE-PERFECT (C1): `preserve_thinking=true` on every chat request, thinking off and on, so
    the Qwen3.x / Ornith template keeps the <think> block of every earlier assistant turn.
    Without it the template drops those blocks from all turns before the newest user message,
    so each new user message changes the rendering of the previous tool loop and the engine has
    to prefill that loop again. With thinking on the engine's ECHO puts the model's own
    reasoning back (the client returns none); it stays in context until compaction (O-96 d).
    An explicit client value still wins. `config['request_defaults']` is no longer needed.
    Only the MoE (non-Flash) profiles get it: Flash has no qualified render/prefix gate for it
    (Codex P1 review item 4), so a Flash chat keeps the template default."""
    result = copy.deepcopy(payload)
    # A JSON null is "absent" to the engine, which would then sample at its own 0.8 default.
    for key, value in (('temperature', 0), ('cache_prompt', True)):
        if result.get(key) is None:
            result[key] = value
    limit = default_output_tokens(config)
    # Only when the client sent no length field at all; never touch an explicit limit.
    if limit is not None and path in GENERATION_PATHS and not any(k in result for k in LENGTH_FIELDS):
        result['n_predict' if path == '/completion' else 'max_tokens'] = limit
    if path in ('/chat/completions', '/v1/chat/completions'):
        kwargs = result.setdefault('chat_template_kwargs', {})
        if isinstance(kwargs, dict):
            kwargs.setdefault('enable_thinking', False)
            if not (isinstance(config, dict) and is_flash(config)):
                kwargs.setdefault('preserve_thinking', True)
    return result
