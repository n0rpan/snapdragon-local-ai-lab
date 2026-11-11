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
import signal
import stat
import subprocess
import threading

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
    shards = flash.get('shards')
    if not isinstance(shards, list) or len(shards) != 4 or not all(
            isinstance(row, dict) and isinstance(row.get('path'), str) and Path(row['path']).is_absolute()
            and type(row.get('bytes')) is int and row['bytes'] > 0
            and isinstance(row.get('sha256'), str) and re.fullmatch('[0-9a-f]{64}', row['sha256']) for row in shards):
        raise ValueError('oxe_flash_shards_invalid')
    if config.get('sha256') != _flash_digest(shards):
        raise ValueError('oxe_flash_model_sha_mismatch')
    args = config['oxe_args']
    if not _adjacent(args, '--engine', 'flash') or not _adjacent(args, '--flash-cache-gib', str(cache_gib)):
        raise ValueError('oxe_flash_args_must_match_admission')


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


def flash_select(config, available, memory_max=None):
    """The config to launch: as given when its admission passes, else its smaller-cache
    fallback (a detached copy with the cache argument and admission swapped) when that
    passes. Raises the primary admission's ValueError when neither can start now."""
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


def eligible(config):
    """Metadata only. Never hash a binary, read weights, or wake a model."""
    switch = FLASH_ENABLE_ENV if is_flash(config) else ENABLE_ENV
    if os.environ.get(switch) != '1' or config.get('ready') is not True:
        return False
    try:
        _validate(config)
        return True
    except (ValueError, TypeError, AttributeError):
        return False


def command(config, port):
    if not eligible(config):
        raise ValueError('oxe_not_enabled_or_ready')
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError('oxe_invalid_loopback_port')
    values = {'{model}': config['model_path'], '{context}': str(config['context_size']),
              '{threads}': str(config['threads']), '{alias}': config['model_alias']}
    return [config['server_binary'], '--host', '127.0.0.1', '--port', str(port),
            *(values.get(arg, arg) for arg in config['oxe_args'])]


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

    `config['request_defaults']['preserve_thinking']` (off unless true) asks the Qwen3.6 /
    Ornith chat template to keep the (empty, when thinking is off) <think> block on every
    earlier assistant turn. Without it the template drops those blocks from all turns before
    the newest user message, so each new user message changes the rendering of the previous
    tool loop and the engine has to prefill that loop again."""
    profile = config
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
            wanted = profile.get('request_defaults') if isinstance(profile, dict) else None
            if isinstance(wanted, dict) and wanted.get('preserve_thinking') is True:
                kwargs.setdefault('preserve_thinking', True)
    return result
