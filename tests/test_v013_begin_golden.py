"""v0.1.3 split phase 2: golden traces of Manager._begin, frozen on the unsplit code.

For each engine kind (llama classic, oxe MoE, oxe Flash, bigmoe streamed Flash) this freezes the
resolved launch (Popen argv, environment delta, Popen options), the order of every side-effecting
collaborator call during the start, the Manager state afterwards, and for each refusal which
message wins and what cleanup follows. Model-free and process-free: every collaborator is a
recording fake, patched through the controller namespace or its own module object, so the frozen
values stay valid when _begin's body moves.
"""
import fcntl
import http.client
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from snapdragon_ai_lab.runtime import (bigmoe_runtime, classic_beta, controller, engine_ledger, explicit_route,
                                       memory_policy, oxe_runtime)

GIB = 1024**3
PORT, FLASH_PORT = 18080, 18090
PID = 4_000_123
TKN, KEY_FILE = 'TKN', '/run/key-1'
POLICY = {'scope_max_bytes': 29 * GIB, 'host_floor_bytes': GIB}
SHA_A, SHA_B = 'a' * 64, 'b' * 64
STDERR = object()


class FakeProcess:
    def __init__(self, h, code=None, immortal=False):
        self.h, self.pid, self.code, self.immortal, self.stderr = h, PID, code, immortal, STDERR

    def poll(self):
        return self.code

    def wait(self, timeout=None):
        if self.code is None:
            raise subprocess.TimeoutExpired('engine', timeout)
        return self.code


class FakeWriter:
    def __init__(self, h, path):
        self.h = h
        h.rec('EngineLogWriter', path)

    def write(self, text):
        self.h.rec('writer.write', text.strip())


class FakeGuard:
    def __init__(self, h, kwargs):
        self.h, self.fail = h, None
        h.rec('MemoryGuard', *('%s=%s' % item for item in sorted(kwargs.items())))

    def start(self, process):
        self.h.rec('guard.start')
        if self.h.guard_fail_on == 'start':
            self.fail = 'guard-oom'

    def failure(self):
        return self.fail

    def mark_ready(self):
        self.h.rec('guard.mark_ready')
        if self.h.guard_fail_on == 'mark_ready':
            self.fail = 'guard-oom'

    def mode(self):
        return self.h.guard_mode

    def leave_swap_tolerant(self):
        self.h.rec('guard.leave_swap_tolerant')

    def stop(self):
        self.h.rec('guard.stop')

    def metadata(self):
        return {'guard': 'metadata'}


class FakeRuntime:
    def __init__(self, h, config, port):
        self.h, self.fail = h, None
        h.rec('bigmoe.Runtime', config.get('tag'), port)
        if h.runtime_error is not None:
            raise h.runtime_error

    def require_memory(self, available, lease_owned):
        self.h.rec('runtime.require_memory', available, 'lease_owned=%s' % lease_owned)

    def prepare(self, lease_owned):
        self.h.rec('runtime.prepare', 'lease_owned=%s' % lease_owned)
        return ['bigmoe-server', '--port', str(FLASH_PORT)]

    def environment(self, environment):
        self.h.rec('runtime.environment')
        return {**environment, 'BIGMOE_ENV': '1'}

    def start_guard(self, process, probe, lease_owned, **kwargs):
        self.h.rec('runtime.start_guard', 'probe=%s' % (probe is self.h.available_probe),
                   'lease_owned=%s' % lease_owned, *('%s=%s' % item for item in sorted(kwargs.items())))
        if self.h.runtime_guard_fail:
            self.fail = 'runtime-guard'

    def guard_failure(self):
        return self.fail

    def readiness(self, lease_owned):
        self.h.rec('runtime.readiness', 'lease_owned=%s' % lease_owned)
        return SimpleNamespace(ready=self.h.runtime_ready)

    def verify_child(self, process):
        self.h.rec('runtime.verify_child')

    def stop_guard(self):
        self.h.rec('runtime.stop_guard')


class FakePowerGuard:
    def __init__(self, h, process):
        self.h, self.failure = h, None
        h.rec('FlashPowerGuard')

    def start(self):
        self.h.rec('FlashPowerGuard.start')
        if self.h.power_guard_fail:
            self.failure = 'battery'

    def stop(self):
        self.h.rec('FlashPowerGuard.stop')


class FakeLedger:
    def __init__(self, h, path, config):
        self.h, self.arguments, self.pass_fds = h, ['--ledger', 'fd:7'], (7,)
        h.rec('NativeLedger', path, config.get('tag'))
        if h.ledger_error is not None:
            raise h.ledger_error

    def start(self):
        self.h.rec('ledger.start')

    def stop(self, timeout=None):
        self.h.rec('ledger.stop')


class FakeReader:
    def __init__(self, h, name):
        self.h, self.name = h, name

    def start(self):
        self.h.rec(self.name + '.start')
        return self

    def join(self, timeout=None):
        self.h.rec(self.name + '.join')
        return True

    def stop(self, timeout=None):
        self.h.rec(self.name + '.stop')


class Harness:
    """All fakes and knobs for one Manager.begin; trace holds the recorded calls in order."""

    def __init__(self, root):
        self.root = root
        self.trace = []
        self.processes = []
        self.launch = None
        self.available = 64 * GIB
        self.start_seconds = 10.0
        self.port_error = None
        self.flock_error = None
        self.oxe_verify_error = None
        self.beta_verify_error = None
        self.command_error = None
        self.state_cache = False
        self.policy_error = None
        self.token_error = None
        self.guard_ctor_error = None
        self.popen_error = None
        self.proc_code = None
        self.guard_fail_on = None
        self.guard_mode = 'normal'
        self.power_guard_fail = False
        self.runtime_error = None
        self.runtime_guard_fail = False
        self.runtime_ready = True
        self.listener_in_group = True
        self.files_ready = True
        self.power_failure = None
        self.select_error = None
        self.pref_enabled = False
        self.ledger_error = None
        self.health_status = 200
        self.available_probe = None

    def clean(self, value):
        if isinstance(value, Path):
            value = str(value)
        if isinstance(value, str):
            return value.replace(str(self.root), '<TMP>')
        if isinstance(value, (list, tuple)):
            return type(value)(self.clean(item) for item in value)
        if isinstance(value, dict):
            return {self.clean(k): self.clean(v) for k, v in value.items()}
        return value

    def rec(self, name, *args):
        self.trace.append(' '.join([name, *(str(self.clean(arg)) for arg in args)]))

    def patches(self):
        h = self
        lease_path = self.root / 'state/generation.lock'

        def available():
            h.rec('available_memory_bytes')
            return h.available

        available_mock = Mock(side_effect=available)
        self.available_probe = available_mock

        def require_backend_port(port, setting):
            h.rec('require_backend_port', port, setting)
            if h.port_error is not None:
                raise h.port_error

        def open_lease_append():
            h.rec('open_lease_append')
            return open(h.root / 'appended.lock', 'a')

        def flock(lease, flags):
            h.rec('flock', lease.mode, 'EX|NB' if flags == fcntl.LOCK_EX | fcntl.LOCK_NB else flags)
            if h.flock_error is not None:
                raise h.flock_error

        def killpg(pid, sig):
            h.rec('killpg', pid, signal_name(sig))
            for process in h.processes:
                if process.pid == pid and not process.immortal and process.code is None:
                    process.code = -sig

        def disable_repack_cache(environment, alias):
            h.rec('disable_repack_cache', alias)
            for key in [key for key in environment if key.startswith(controller.REPACK_CACHE_ENV)]:
                del environment[key]

        class Connection:
            def __init__(self, host, port, timeout=None):
                self.host, self.port, self.timeout, self.path = host, port, timeout, None

            def request(self, method, path, headers=None):
                self.path = path
                auth = (headers or {}).get('Authorization', '')
                h.rec('http', method, path, self.host, self.port, 't=%s' % self.timeout,
                      'auth=' + (auth.split(' ', 1)[1] if auth else '-'))

            def getresponse(self):
                status, body = {'/health': (h.health_status, b'{"status":"ok"}'),
                                '/props': (200, b'{"cache":"props"}'),
                                '/v1/models': (200, b'{"data":[]}')}[self.path]
                return SimpleNamespace(status=status, read=lambda *size: body)

            def close(self):
                pass

        def popen(args, **kwargs):
            h.rec('Popen')
            if h.popen_error is not None:
                raise h.popen_error
            environment = kwargs.pop('env')
            base = dict(os.environ)
            names = {subprocess.DEVNULL: 'DEVNULL', subprocess.PIPE: 'PIPE'}
            h.launch = h.clean({
                'argv': list(args),
                'env_added': [(k, v) for k, v in environment.items() if k not in base],
                'env_changed': [(k, v) for k, v in environment.items() if k in base and base[k] != v],
                'env_removed': sorted(k for k in base if k not in environment),
                'options': {k: (names.get(v, v) if k in ('stdout', 'stderr') else v)
                            for k, v in sorted(kwargs.items())},
            })
            process = FakeProcess(h, code=h.proc_code)
            h.processes.append(process)
            return process

        def new_api_key(port):
            h.rec('new_api_key', port)
            if h.token_error is not None:
                raise h.token_error
            return TKN, Path(KEY_FILE)

        def command(config, port):
            h.rec('oxe.command', config.get('tag'), port)
            if h.command_error is not None:
                raise h.command_error
            return [str(config['server_binary']), '--port', str(port),
                    *(['--state-cache', '/sc'] if h.state_cache else [])]

        def verify_oxe(config):
            h.rec('oxe.verify_engine', config.get('tag'))
            if h.oxe_verify_error is not None:
                raise h.oxe_verify_error

        def verify_beta(config):
            h.rec('classic_beta.verify_engine', config.get('tag'))
            if h.beta_verify_error is not None:
                raise h.beta_verify_error

        original_launch_environment = oxe_runtime.launch_environment

        def launch_environment(environment):
            h.rec('launch_environment')
            return original_launch_environment(environment)

        def flash_select(config, available, limit, *, env, mem_total, swap_free):
            h.rec('flash_select', config.get('tag'), available, limit, mem_total, swap_free,
                  'env=%s' % (env is os.environ))
            if h.select_error is not None:
                raise h.select_error
            return {**config, 'tag': config.get('tag') + '-sel'}

        def flash_swap_tolerant_guard(config, available, mem_total, *, env, swap_free):
            h.rec('flash_swap_tolerant_guard', config.get('tag'), available, mem_total, swap_free)
            return None

        def flash_policy(config):
            h.rec('flash_policy', config.get('tag'))
            if h.policy_error is not None:
                raise h.policy_error
            return dict(POLICY) if config.get('flash_memory') is not None else None

        def scope_command(args, limit):
            h.rec('scope_command', limit)
            return ['exec-in-scope', str(limit), '--', *args]

        def memory_guard(**kwargs):
            if h.guard_ctor_error is not None:
                h.rec('MemoryGuard', 'raises')
                raise h.guard_ctor_error
            return FakeGuard(h, kwargs)

        original_ledger_enabled = engine_ledger.enabled

        def ledger_enabled(config, command):
            result = original_ledger_enabled(config, command)
            h.rec('ledger.enabled', result)
            return result

        def recorder(name, result=None, error=None):
            def call(*args, **kwargs):
                h.rec(name, *(h.clean(arg) for arg in args), *('%s=%s' % item for item in sorted(kwargs.items())))
                if error is not None:
                    raise error
                return result() if callable(result) else result
            return call

        return [
            patch.object(controller, 'BACKEND_PORT', PORT),
            patch.object(controller, 'FLASH_BACKEND_PORT', FLASH_PORT),
            patch.object(controller, 'START_SECONDS', self.start_seconds),
            patch.object(controller, 'TERM_WAIT_SECONDS', 0.01),
            patch.object(controller, 'KILL_WAIT_SECONDS', 0.01),
            patch.object(controller, 'LEASE', lease_path),
            patch.object(controller, 'ENGINE_DIAGNOSTIC_LOG', self.root / 'engine-stderr.log'),
            patch.object(controller, 'ENGINE_LEDGER_LOG', self.root / 'engine-ledger.log'),
            patch.object(controller, 'HARNESS_STATE', ''),
            patch.object(controller, 'require_backend_port', side_effect=require_backend_port),
            patch.object(controller, 'available_memory_bytes', available_mock),
            patch.object(controller, 'total_memory_bytes', side_effect=recorder('total_memory_bytes', 32 * GIB)),
            patch.object(controller, 'swap_bytes', side_effect=recorder('swap_bytes', (16 * GIB, 8 * GIB))),
            patch.object(controller, 'cgroup_memory_max', side_effect=recorder('cgroup_memory_max', 28 * GIB)),
            patch.object(controller, 'guard_production_path', side_effect=recorder('guard_production_path')),
            patch.object(controller, 'open_lease_append', side_effect=open_lease_append),
            patch.object(controller, 'EngineLogWriter', side_effect=lambda path: FakeWriter(h, path)),
            patch.object(controller, 'RepackErrorReader', side_effect=lambda stream, label, tee_writer: (
                h.rec('RepackErrorReader', 'label=' + label, 'tee=%s' % isinstance(tee_writer, FakeWriter),
                      'stream=%s' % (stream is STDERR)), FakeReader(h, 'RepackErrorReader'))[1]),
            patch.object(controller, 'EngineStderrReader', side_effect=lambda stream, writer, **hook: (
                h.rec('EngineStderrReader', 'hook=%s' % sorted(hook), 'stream=%s' % (stream is STDERR),
                      'writer=%s' % isinstance(writer, FakeWriter)), FakeReader(h, 'EngineStderrReader'))[1]),
            patch.object(controller, 'PrefixStore', side_effect=lambda directory, max_bytes: (
                h.rec('PrefixStore', directory, max_bytes), SimpleNamespace(root='/prefix/root'))[1]),
            patch.object(controller, 'disable_repack_cache', side_effect=disable_repack_cache),
            patch.object(controller, 'port_listener_in_group', side_effect=lambda port, pid: (
                h.rec('port_listener_in_group', port, pid), h.listener_in_group)[1]),
            patch.object(controller, 'listening_process', side_effect=lambda port: (
                h.rec('listening_process', port), 'other-app (pid 77)')[1]),
            patch.object(controller.fcntl, 'flock', side_effect=flock),
            patch.object(controller.os, 'killpg', side_effect=killpg),
            patch.object(http.client, 'HTTPConnection', Connection),
            patch.object(controller.subprocess, 'Popen', side_effect=popen),
            patch.object(oxe_runtime, 'verify_engine', side_effect=verify_oxe),
            patch.object(oxe_runtime, 'command', side_effect=command),
            patch.object(oxe_runtime, 'engine_has_state_preload', side_effect=recorder('engine_has_state_preload', True)),
            patch.object(oxe_runtime, 'launch_environment', side_effect=launch_environment),
            patch.object(oxe_runtime, 'new_api_key', side_effect=new_api_key),
            patch.object(oxe_runtime, 'remove_api_key', side_effect=recorder('remove_api_key')),
            patch.object(oxe_runtime, 'flash_files_ready', side_effect=lambda config: (
                h.rec('flash_files_ready', config.get('tag')), h.files_ready)[1]),
            patch.object(oxe_runtime, 'flash_power_failure', side_effect=lambda snapshot, starting: (
                h.rec('flash_power_failure', snapshot, 'starting=%s' % starting), h.power_failure)[1]),
            patch.object(oxe_runtime, 'flash_select', side_effect=flash_select),
            patch.object(oxe_runtime, 'flash_limits', side_effect=lambda config: (
                h.rec('flash_limits', config.get('tag')), (None, 29 * GIB))[1]),
            patch.object(oxe_runtime, 'flash_swap_tolerant_guard', side_effect=flash_swap_tolerant_guard),
            patch.object(oxe_runtime, 'flash_smallest_required_available', side_effect=lambda config: 20 * GIB),
            patch.object(oxe_runtime, 'FlashPowerGuard', side_effect=lambda process: FakePowerGuard(h, process)),
            patch.object(oxe_runtime, 'ineligible_reason', return_value=None),
            patch.object(oxe_runtime.flash_cache_pref, 'enabled', side_effect=lambda env, flash: (
                h.rec('flash_cache_pref.enabled', 'env=%s' % (env is os.environ)), h.pref_enabled)[1]),
            patch.object(oxe_runtime.flash_cache_pref, 'status_from_props', side_effect=lambda value: (
                h.rec('status_from_props', value), {'reason': 'fixed'})[1]),
            patch.object(bigmoe_runtime, 'Runtime', side_effect=lambda config, port: FakeRuntime(h, config, port)),
            patch.object(bigmoe_runtime, 'power_snapshot', side_effect=recorder('power_snapshot', 'snapshot')),
            patch.object(memory_policy, 'flash_policy', side_effect=flash_policy),
            patch.object(memory_policy, 'scope_command', side_effect=scope_command),
            patch.object(memory_policy, 'MemoryGuard', side_effect=memory_guard),
            patch.object(classic_beta, 'verify_engine', side_effect=verify_beta),
            patch.object(engine_ledger, 'enabled', side_effect=ledger_enabled),
            patch.object(engine_ledger, 'NativeLedger', side_effect=lambda path, config: FakeLedger(h, path, config)),
            patch.object(engine_ledger, 'record_transition', side_effect=lambda path, old, new, ms, success: h.rec(
                'record_transition', path, old, new, 'success=%s' % success)),
        ]


def signal_name(number):
    import signal
    return signal.Signals(number).name


def llama(model, **changes):
    return {'model_alias': 'llama-local', 'ready': True, 'model_path': str(model), 'expected_bytes': 100,
            'server_binary': '/opt/llama-server', 'context_size': 4096, 'threads': 6, 'gpu_layers': 0,
            'extra_args': ['--jinja'], 'environment': {'GGML_PROFILE_SET': '1'}, 'tag': 'llama', **changes}


def oxe(model, **changes):
    return {'model_alias': 'oxe-moe', 'backend_kind': oxe_runtime.KIND, 'ready': True, 'model_path': str(model),
            'expected_bytes': 100, 'server_binary': '/opt/oxe-server',
            'environment': {'OXE_PROFILE_ENV': '1', oxe_runtime.API_KEY_ENV: '/profile/key'}, 'tag': 'oxe', **changes}


def flash(model, **changes):
    return oxe(model, model_alias='oxe-flash', oxe_flash={'cache_gib': 20}, environment={}, tag='flash', **changes)


def bigmoe(model, **changes):
    return {'model_alias': 'flash-stream', 'backend_kind': bigmoe_runtime.KIND, 'ready': True,
            'model_path': str(model), 'expected_bytes': 100, 'tag': 'bigmoe', **changes}


KINDS = {'llama': llama, 'oxe': oxe, 'flash': flash, 'bigmoe': bigmoe}

LOCK = ['guard_production_path <TMP>/state/generation.lock', 'guard_production_path <TMP>/state/generation.lock',
        'open_lease_append', 'flock a EX|NB']
HEADER_LLAMA = ['require_backend_port 18080 LOCAL_LLM_BACKEND_PORT', 'available_memory_bytes', *LOCK]
READY_TAIL = ['guard.start', 'http GET /health 127.0.0.1 18080 t=2 auth=TKN']
OPTIONS = {'start_new_session': True, 'stderr': 'PIPE', 'stdout': 'DEVNULL', 'umask': 0o077}
LLAMA_ARGV = ['/opt/llama-server', '--model', '<TMP>/model.gguf', '--alias', 'llama-local', '--host', '127.0.0.1',
              '--port', '18080', '--ctx-size', '4096', '--threads', '6', '--n-gpu-layers', '0', '--jinja']
GUARD_PLAIN = 'MemoryGuard host_guard_version=GUARD-2 loading_timeout_seconds=10.0'
GUARD_SCOPED = ('MemoryGuard follow_process_cgroup=True host_guard_version=GUARD-2 '
                'loading_timeout_seconds=10.0 minimum_available_bytes=1073741824')


def llama_trace(*, after_lock=(), after_policy=(), after_token=(), reader=('EngineStderrReader hook=[] stream=True writer=True',
                'EngineStderrReader.start'), tag='llama', alias='llama-local'):
    return [*HEADER_LLAMA, *after_lock, 'flash_policy ' + tag, *after_policy, 'new_api_key 18080', *after_token,
            GUARD_PLAIN, 'EngineLogWriter <TMP>/engine-stderr.log', 'ledger.enabled False', 'Popen', *reader,
            *READY_TAIL, 'port_listener_in_group 18080 4000123', 'guard.mark_ready',
            f'record_transition <TMP>/engine-ledger.log None {alias} success=True',
            f'flash_idle_qos start {tag} 4000123', 'http GET /v1/models 127.0.0.1 18080 t=3 auth=TKN']


def oxe_trace(*, preload=(), after_policy=(), ledger=('ledger.enabled False',), after_popen=()):
    return [*HEADER_LLAMA, 'oxe.verify_engine oxe', 'oxe.command oxe 18080', *preload, 'flash_policy oxe',
            *after_policy, 'launch_environment', 'new_api_key 18080', GUARD_PLAIN,
            'EngineLogWriter <TMP>/engine-stderr.log', *ledger, 'Popen', *after_popen,
            'EngineStderrReader hook=[] stream=True writer=True', 'EngineStderrReader.start', *READY_TAIL,
            'guard.mark_ready', 'record_transition <TMP>/engine-ledger.log None oxe-moe success=True',
            'flash_idle_qos start oxe 4000123', 'http GET /v1/models 127.0.0.1 18080 t=3 auth=TKN']


def state(**changes):
    value = {'state': 'ready', 'launches': 1, 'active': 1, 'inferences': 1, 'state_engine': False,
             'state_preload': False, 'prefix_store': False, 'lease_mode': 'a', 'loaded_tag': 'llama',
             'process_attrs': ('llama', TKN, KEY_FILE), 'backend_runtime': False, 'classic_guard': True,
             'flash_power_guard': False, 'engine_reader': True, 'repack_reader': False, 'engine_ledger': False,
             'flash_cache_status': None, 'cache': ['/v1/models'], 'flash_idle_qos_pid': PID,
             'last_failure_info': None, 'last_memory_guard': None, 'stop_pending': None}
    value.update(changes)
    return value


def launch(argv, *, added=(), changed=(), removed=(), **options):
    return {'argv': list(argv), 'env_added': [tuple(item) for item in added], 'env_changed': list(changed),
            'env_removed': list(removed), 'options': {**OPTIONS, **options}}


OXE_ENV = [('OXE_PROFILE_ENV', '1'), (oxe_runtime.API_KEY_ENV, KEY_FILE)]
OXE_REMOVED = ['GGML_BASE_TUNING', 'LLAMA_BASE_TUNING']
OXE_STATE = state(loaded_tag='oxe', process_attrs=('oxe', TKN, KEY_FILE))

# name -> (kind, config changes, harness knobs, expected trace, expected launch, expected state)
SUCCESS = {
    'llama_plain': ('llama', {}, {}, llama_trace(),
                    launch([*LLAMA_ARGV, '--api-key-file', KEY_FILE], added=[('GGML_PROFILE_SET', '1')]),
                    state()),
    'llama_prefix_cache': (
        'llama', {'prefix_cache': {'enabled': True, 'directory': 'DIR', 'max_bytes': 1234}}, {},
        llama_trace(after_policy=['PrefixStore DIR 1234']),
        launch([*LLAMA_ARGV, '--slots', '--slot-save-path', '/prefix/root', '--api-key-file', KEY_FILE],
               added=[('GGML_PROFILE_SET', '1')]),
        state(prefix_store=True)),
    'llama_prefix_cache_default_budget': (
        'llama', {'prefix_cache': {'enabled': True, 'directory': 'DIR'}}, {},
        llama_trace(after_policy=[f'PrefixStore DIR {3 * GIB}']),
        launch([*LLAMA_ARGV, '--slots', '--slot-save-path', '/prefix/root', '--api-key-file', KEY_FILE],
               added=[('GGML_PROFILE_SET', '1')]),
        state(prefix_store=True)),
    'llama_repack_identity': (
        'llama', {'sha256': SHA_A, 'server_sha256': SHA_B}, {'repack': True},
        llama_trace(reader=('RepackErrorReader label=llama-local tee=True stream=True', 'RepackErrorReader.start')),
        launch([*LLAMA_ARGV, '--api-key-file', KEY_FILE],
               added=[('GGML_PROFILE_SET', '1'), ('GGML_CPU_REPACK_CACHE_MODEL_SHA256', SHA_A),
                      ('GGML_CPU_REPACK_CACHE_ENGINE_ID', SHA_B)]),
        state(engine_reader=False, repack_reader=True)),
    'llama_repack_no_identity': (
        'llama', {}, {'repack': True},
        llama_trace(after_token=['disable_repack_cache llama-local']),
        launch([*LLAMA_ARGV, '--api-key-file', KEY_FILE], added=[('GGML_PROFILE_SET', '1')],
               removed=['GGML_CPU_REPACK_CACHE']),
        state()),
    'classic_beta': (
        'llama', {'model_alias': classic_beta.ALIAS}, {},
        llama_trace(after_lock=['classic_beta.verify_engine llama'], alias=classic_beta.ALIAS),
        launch([*LLAMA_ARGV[:4], classic_beta.ALIAS, *LLAMA_ARGV[5:], '--api-key-file', KEY_FILE],
               added=[('GGML_PROFILE_SET', '1')]),
        state()),
    'oxe_plain': ('oxe', {}, {}, oxe_trace(),
                  launch(['/opt/oxe-server', '--port', '18080'], added=OXE_ENV, removed=OXE_REMOVED), OXE_STATE),
    'oxe_state_cache': (
        'oxe', {}, {'state_cache': True}, oxe_trace(preload=['engine_has_state_preload /opt/oxe-server']),
        launch(['/opt/oxe-server', '--port', '18080', '--state-cache', '/sc'], added=OXE_ENV, removed=OXE_REMOVED),
        {**OXE_STATE, 'state_engine': True, 'state_preload': True}),
    'oxe_prefix_cache': (
        'oxe', {'prefix_cache': {'enabled': True, 'directory': 'DIR', 'max_bytes': 99}}, {},
        oxe_trace(after_policy=['PrefixStore DIR 99']),
        launch(['/opt/oxe-server', '--port', '18080', '--slots', '--slot-save-path', '/prefix/root'],
               added=OXE_ENV, removed=OXE_REMOVED),
        {**OXE_STATE, 'prefix_store': True}),
    'oxe_ledger': (
        'oxe', {'environment': {engine_ledger.ENABLE_ENV: '1'}}, {},
        oxe_trace(ledger=('ledger.enabled True', 'NativeLedger <TMP>/engine-ledger.log oxe'),
                  after_popen=['ledger.start']),
        launch(['/opt/oxe-server', '--port', '18080', '--ledger', 'fd:7'],
               added=[(engine_ledger.ENABLE_ENV, '1'), (oxe_runtime.API_KEY_ENV, KEY_FILE)], removed=OXE_REMOVED,
               pass_fds=(7,)),
        {**OXE_STATE, 'engine_ledger': True}),
    'oxe_ledger_unavailable': (
        'oxe', {'environment': {engine_ledger.ENABLE_ENV: '1'}}, {'ledger_error': OSError('no log')},
        oxe_trace(ledger=('ledger.enabled True', 'NativeLedger <TMP>/engine-ledger.log oxe',
                          'writer.write {"event":"ledger_disabled","reason":"log_unavailable"}')),
        launch(['/opt/oxe-server', '--port', '18080'],
               added=[(engine_ledger.ENABLE_ENV, '1'), (oxe_runtime.API_KEY_ENV, KEY_FILE)], removed=OXE_REMOVED),
        OXE_STATE),
    'flash_plain': (
        'flash', {}, {},
        ['require_backend_port 18080 LOCAL_LLM_BACKEND_PORT', 'available_memory_bytes', 'flash_files_ready flash',
         'power_snapshot', 'flash_power_failure snapshot starting=True', 'flash_cache_pref.enabled env=True',
         'cgroup_memory_max', f'flash_select flash {64 * GIB} {28 * GIB} None None env=True',
         f'flash_swap_tolerant_guard flash-sel {64 * GIB} None None', *LOCK,
         'oxe.verify_engine flash-sel', 'oxe.command flash-sel 18080', 'flash_policy flash-sel',
         'launch_environment', 'new_api_key 18080', GUARD_PLAIN, 'EngineLogWriter <TMP>/engine-stderr.log',
         'ledger.enabled False', 'Popen', "EngineStderrReader hook=['on_line'] stream=True writer=True",
         'EngineStderrReader.start', 'guard.start', 'FlashPowerGuard', 'FlashPowerGuard.start',
         'http GET /health 127.0.0.1 18080 t=2 auth=TKN', 'guard.mark_ready',
         'record_transition <TMP>/engine-ledger.log None oxe-flash success=True',
         'flash_idle_qos start flash-sel 4000123', 'http GET /props 127.0.0.1 18080 t=3 auth=TKN',
         "status_from_props {'cache': 'props'}", 'http GET /v1/models 127.0.0.1 18080 t=3 auth=TKN'],
        launch(['/opt/oxe-server', '--port', '18080'], added=[(oxe_runtime.API_KEY_ENV, KEY_FILE)],
               removed=OXE_REMOVED),
        state(loaded_tag='flash-sel', process_attrs=('flash-sel', TKN, KEY_FILE), flash_power_guard=True,
              flash_cache_status={'reason': 'fixed'})),
    'flash_policy_pref': (
        'flash', {'flash_memory': dict(POLICY)}, {'pref_enabled': True, 'guard_mode': memory_policy.SWAP_TOLERANT_MODE},
        ['require_backend_port 18080 LOCAL_LLM_BACKEND_PORT', 'available_memory_bytes', 'flash_files_ready flash',
         'power_snapshot', 'flash_power_failure snapshot starting=True', 'flash_cache_pref.enabled env=True',
         'total_memory_bytes', 'swap_bytes', 'flash_limits flash',
         f'flash_select flash {64 * GIB} {29 * GIB} {32 * GIB} {8 * GIB} env=True',
         f'flash_swap_tolerant_guard flash-sel {64 * GIB} {32 * GIB} {8 * GIB}', *LOCK,
         'oxe.verify_engine flash-sel', 'oxe.command flash-sel 18080', 'flash_policy flash-sel',
         f'scope_command {29 * GIB}', 'launch_environment', 'new_api_key 18080', GUARD_SCOPED,
         'EngineLogWriter <TMP>/engine-stderr.log', 'ledger.enabled False', 'Popen',
         "EngineStderrReader hook=['on_line'] stream=True writer=True", 'EngineStderrReader.start', 'guard.start',
         'FlashPowerGuard', 'FlashPowerGuard.start', 'http GET /health 127.0.0.1 18080 t=2 auth=TKN',
         'guard.mark_ready', 'record_transition <TMP>/engine-ledger.log None oxe-flash success=True',
         'flash_idle_qos start flash-sel 4000123', 'http GET /props 127.0.0.1 18080 t=3 auth=TKN',
         "status_from_props {'cache': 'props'}", 'guard.leave_swap_tolerant',
         'http GET /v1/models 127.0.0.1 18080 t=3 auth=TKN'],
        launch(['exec-in-scope', str(29 * GIB), '--', '/opt/oxe-server', '--port', '18080'],
               added=[(oxe_runtime.API_KEY_ENV, KEY_FILE)], removed=OXE_REMOVED),
        state(loaded_tag='flash-sel', process_attrs=('flash-sel', TKN, KEY_FILE), flash_power_guard=True,
              flash_cache_status={'reason': 'fixed'})),
    'bigmoe_plain': (
        'bigmoe', {}, {},
        ['require_backend_port 18090 LOCAL_FLASH_BACKEND_PORT', 'available_memory_bytes', 'bigmoe.Runtime bigmoe 18090',
         'guard_production_path <TMP>/state/generation.lock', 'flock r+ EX|NB', 'flash_policy bigmoe',
         'available_memory_bytes', f'runtime.require_memory {64 * GIB} lease_owned=True',
         'runtime.prepare lease_owned=True', 'runtime.environment', 'EngineLogWriter <TMP>/engine-stderr.log',
         'ledger.enabled False', 'Popen', 'EngineStderrReader hook=[] stream=True writer=True',
         'EngineStderrReader.start', 'runtime.start_guard probe=True lease_owned=True',
         'runtime.readiness lease_owned=True', 'runtime.verify_child',
         'record_transition <TMP>/engine-ledger.log None flash-stream success=True',
         'flash_idle_qos start bigmoe 4000123'],
        launch(['bigmoe-server', '--port', '18090'], added=[('BIGMOE_ENV', '1')]),
        state(lease_mode='r+', loaded_tag='bigmoe', process_attrs=('bigmoe', None, None), backend_runtime=True,
              classic_guard=False, cache=[])),
    'bigmoe_policy': (
        'bigmoe', {'flash_memory': dict(POLICY)}, {},
        ['require_backend_port 18090 LOCAL_FLASH_BACKEND_PORT', 'available_memory_bytes', 'bigmoe.Runtime bigmoe 18090',
         'guard_production_path <TMP>/state/generation.lock', 'flock r+ EX|NB', 'flash_policy bigmoe',
         'available_memory_bytes', f'runtime.require_memory {64 * GIB} lease_owned=True',
         'runtime.prepare lease_owned=True', f'scope_command {29 * GIB}', 'runtime.environment',
         'EngineLogWriter <TMP>/engine-stderr.log', 'ledger.enabled False', 'Popen',
         'EngineStderrReader hook=[] stream=True writer=True', 'EngineStderrReader.start',
         f'runtime.start_guard probe=True lease_owned=True reserve_bytes={GIB}',
         'runtime.readiness lease_owned=True', 'runtime.verify_child',
         'record_transition <TMP>/engine-ledger.log None flash-stream success=True',
         'flash_idle_qos start bigmoe 4000123'],
        launch(['exec-in-scope', str(29 * GIB), '--', 'bigmoe-server', '--port', '18090'], added=[('BIGMOE_ENV', '1')]),
        state(lease_mode='r+', loaded_tag='bigmoe', process_attrs=('bigmoe', None, None), backend_runtime=True,
              classic_guard=False, cache=[])),
}

MSG_STOP_PENDING = ('The previous local engine has not exited yet; its memory and the inference lease stay '
                    'reserved. Retry shortly.')
MSG_LEASE = 'Another image, chat or setup job is running. Please try again when it finishes.'
MSG_GUARD = 'Local model stopped to preserve system memory.'
MSG_EXIT = 'Local model backend exited during loading. Setup diagnostics are in the LLM project.'
MSG_TIMEOUT = 'Local model loading timed out; no generation was started.'
MSG_TOKEN = 'Local model cannot start: no private runtime directory for its access token.'
MSG_RAM = ('This model needs about 3.0 GiB of available RAM; 1.0 GiB is available. '
           'Close other large apps and retry.')
MSG_NOT_FLASH = 'A Flash memory policy is set on a model that is not Flash.'

# name -> (kind, config changes, harness knobs, manager attributes, exception class, message)
FAILURES = {
    'image_running': ('llama', {}, {}, {'image_active': True}, 'Unavailable',
                      'An image is running; the resident chat model is retained.'),
    'stop_pending': ('llama', {}, {}, {'stop_pending': 'old'}, 'Unavailable', MSG_STOP_PENDING),
    'unknown_kind': ('llama', {'backend_kind': 'weird'}, {}, {}, 'Unavailable', 'Unknown local backend kind.'),
    'oxe_not_eligible': ('oxe', {}, {'eligible': False}, {}, 'Unavailable', 'The Rust engine is not enabled and ready.'),
    'beta_not_eligible': ('llama', {'model_alias': classic_beta.ALIAS}, {'eligible': False}, {}, 'Unavailable',
                          'Model proof or identity changed; requalification is required.'),
    'sleeping_llama': ('llama', {}, {'allow_start': False}, {}, 'Unavailable',
                       'Model is sleeping; submitting a message will load it.'),
    'sleeping_bigmoe': ('bigmoe', {}, {'allow_start': False}, {}, 'Unavailable',
                        'Model is sleeping; submitting a message will load it.'),
    'not_ready': ('oxe', {'ready': False}, {'port_error': controller.Unavailable('port busy')}, {}, 'Unavailable',
                  'Qwen setup and validation are still running. Check /_local/status.'),
    'port_busy': ('llama', {'model_path': '/nonexistent/model.gguf'},
                  {'port_error': controller.Unavailable('port busy')}, {}, 'Unavailable', 'port busy'),
    'port_busy_bigmoe': ('bigmoe', {}, {'port_error': controller.Unavailable('port busy')}, {}, 'Unavailable',
                         'port busy'),
    'model_missing': ('llama', {'model_path': '/nonexistent/model.gguf'}, {'available': GIB}, {}, 'Unavailable',
                      'The verified local model is not available yet.'),
    'size_mismatch': ('oxe', {'expected_bytes': 99}, {'available': GIB}, {}, 'Unavailable',
                      'The local model size does not match its verified profile.'),
    'ram_short_llama': ('llama', {}, {'available': GIB}, {}, 'Unavailable', MSG_RAM),
    'ram_short_oxe': ('oxe', {}, {'available': GIB}, {}, 'Unavailable', MSG_RAM),
    'ram_unknown_llama': ('llama', {}, {'available': None, 'flock_error': BlockingIOError()}, {}, 'Unavailable',
                          MSG_LEASE),
    'bigmoe_unqualified': ('bigmoe', {}, {'runtime_error': ValueError('bad')}, {}, 'Unavailable',
                           'Flash profile is unqualified or has insufficient admitted RAM.'),
    'flash_files': ('flash', {}, {'files_ready': False, 'power_failure': 'low'}, {}, 'Unavailable',
                    'Flash model shards or the expert sidecar are missing or changed.'),
    'flash_power': ('flash', {}, {'power_failure': 'low'}, {}, 'Unavailable',
                    'Flash needs at least 30% battery to start and stops at 20%.'),
    'flash_select_pref': ('flash', {}, {'select_error': ValueError('fit'), 'pref_enabled': True}, {}, 'Unavailable',
                          'Flash cache preference cannot fit the current native memory budget; the host floor and '
                          'runtime pressure guards remain active.'),
    'flash_select_fixed': ('flash', {}, {'select_error': ValueError('fit')}, {}, 'Unavailable',
                           'Flash (Rust) needs about 20.0 GiB of available RAM; close other large apps and retry.'),
    'lease_busy_llama': ('llama', {}, {'flock_error': BlockingIOError()}, {}, 'Unavailable', MSG_LEASE),
    'lease_busy_bigmoe': ('bigmoe', {}, {'flock_error': BlockingIOError()}, {}, 'Unavailable', MSG_LEASE),
    'oxe_binary_missing': ('oxe', {}, {'oxe_verify_error': FileNotFoundError('gone')}, {}, 'Unavailable',
                           'The Rust engine binary is missing.'),
    'oxe_identity_changed': ('oxe', {}, {'oxe_verify_error': ValueError('oxe_engine_identity_changed')}, {},
                             'ValueError', 'oxe_engine_identity_changed'),
    'beta_identity_changed': ('llama', {'model_alias': classic_beta.ALIAS},
                              {'beta_verify_error': ValueError('beta_engine_identity_changed')}, {},
                              'ValueError', 'beta_engine_identity_changed'),
    'state_cache_root': ('oxe', {}, {'command_error': ValueError('state_cache_root_override_invalid')}, {},
                         'Unavailable', 'LOCAL_AI_STATE_CACHE_ROOT (or runtime.json cache.root) must be an absolute '
                         'path inside your home; the model was not started.'),
    'command_other': ('flash', {}, {'command_error': ValueError('oxe_unknown_argument_placeholder')}, {},
                      'ValueError', 'oxe_unknown_argument_placeholder'),
    'policy_unqualified': ('flash', {}, {'policy_error': ValueError('flash_memory_policy_unqualified')}, {},
                           'Unavailable', 'The Flash memory policy in this profile is not a qualified one.'),
    'policy_unqualified_bigmoe': ('bigmoe', {}, {'policy_error': ValueError('flash_memory_policy_unqualified')}, {},
                                  'Unavailable', 'The Flash memory policy in this profile is not a qualified one.'),
    'policy_on_llama': ('llama', {'flash_memory': dict(POLICY)}, {}, {}, 'Unavailable', MSG_NOT_FLASH),
    'policy_on_oxe_moe': ('oxe', {'flash_memory': dict(POLICY)}, {}, {}, 'Unavailable', MSG_NOT_FLASH),
    'token_dir_llama': ('llama', {}, {'token_error': OSError('no dir')}, {}, 'Unavailable', MSG_TOKEN),
    'token_dir_oxe': ('oxe', {}, {'token_error': OSError('no dir')}, {}, 'Unavailable', MSG_TOKEN),
    'memory_policy_error': ('flash', {}, {'guard_ctor_error': memory_policy.MemoryPolicyError('host floor')}, {},
                            'Unavailable', 'Local model cannot load under the current memory conditions.'),
    'popen_error': ('oxe', {}, {'popen_error': OSError('exec failed')}, {}, 'OSError', 'exec failed'),
    'exit_llama': ('llama', {}, {'proc_code': 1}, {}, 'Unavailable', MSG_EXIT),
    'exit_bigmoe': ('bigmoe', {}, {'proc_code': 1}, {}, 'Unavailable', MSG_EXIT),
    'guard_start_llama': ('llama', {}, {'guard_fail_on': 'start', 'proc_code': 1}, {}, 'Unavailable', MSG_GUARD),
    'guard_ready_oxe': ('oxe', {}, {'guard_fail_on': 'mark_ready'}, {}, 'Unavailable', MSG_GUARD),
    'power_guard_flash': ('flash', {}, {'power_guard_fail': True, 'proc_code': 1}, {}, 'Unavailable',
                          'Flash stopped to preserve power.'),
    'runtime_guard_bigmoe': ('bigmoe', {}, {'runtime_guard_fail': True, 'proc_code': 1}, {}, 'Unavailable',
                             'Flash stopped to preserve power or available memory.'),
    'foreign_listener': ('llama', {}, {'listener_in_group': False}, {}, 'Unavailable',
                         'Local engine port 18080 is answered by other-app (pid 77), not the engine this controller '
                         'started. Close that app or set LOCAL_LLM_BACKEND_PORT to an unused port.'),
    'timeout_llama': ('llama', {}, {'start_seconds': 0.0}, {}, 'Unavailable', MSG_TIMEOUT),
    'timeout_oxe': ('oxe', {}, {'start_seconds': 0.0}, {}, 'Unavailable', MSG_TIMEOUT),
    'timeout_bigmoe': ('bigmoe', {}, {'start_seconds': 0.0}, {}, 'Unavailable', MSG_TIMEOUT),
}


def unloaded(**changes):
    """The Manager state after a refusal: nothing loaded unless a field says otherwise."""
    value = state(state='unloaded', launches=0, active=0, inferences=0, lease_mode=None, loaded_tag=None,
                  process_attrs=None, classic_guard=False, engine_reader=False, cache=[], flash_idle_qos_pid=None)
    value.update(changes)
    return value


# name -> (trace, state after) for FAILURES, taken from the unsplit code (commit 1) and read
# against its source: the trace includes the cleanup each refusal runs.
FAILURE_GOLDEN = {
    'image_running': (
        [],
        unloaded()),
    'stop_pending': (
        [
            'killpg 4000123 SIGKILL',
            'remove_api_key None',
            'killpg 4000123 SIGKILL',
        ],
        unloaded(
            state='stopping',
            process_attrs=(None, None, None),
            stop_pending="engine pid 4000123 has not exited: TimeoutExpired: Command 'engine' timed out after 0.01 seconds",
        )),
    'unknown_kind': (
        [],
        unloaded()),
    'oxe_not_eligible': (
        [],
        unloaded()),
    'beta_not_eligible': (
        [],
        unloaded()),
    'sleeping_llama': (
        [],
        unloaded()),
    'sleeping_bigmoe': (
        [],
        unloaded()),
    'not_ready': (
        [],
        unloaded(
            last_failure_info={'reason': 'Qwen setup and validation are still running. Check /_local/status.', 'kind': 'start', 'alias': 'oxe-moe', 'origin': 'organic'},
        )),
    'port_busy': (
        [
            'require_backend_port 18080 LOCAL_LLM_BACKEND_PORT',
        ],
        unloaded(
            last_failure_info={'reason': 'port busy', 'kind': 'start', 'alias': 'llama-local', 'origin': 'organic'},
        )),
    'port_busy_bigmoe': (
        [
            'require_backend_port 18090 LOCAL_FLASH_BACKEND_PORT',
        ],
        unloaded(
            last_failure_info={'reason': 'port busy', 'kind': 'start', 'alias': 'flash-stream', 'origin': 'organic'},
        )),
    'model_missing': (
        [
            'require_backend_port 18080 LOCAL_LLM_BACKEND_PORT',
        ],
        unloaded(
            last_failure_info={'reason': 'The verified local model is not available yet.', 'kind': 'start', 'alias': 'llama-local', 'origin': 'organic'},
        )),
    'size_mismatch': (
        [
            'require_backend_port 18080 LOCAL_LLM_BACKEND_PORT',
        ],
        unloaded(
            last_failure_info={'reason': 'The local model size does not match its verified profile.', 'kind': 'start', 'alias': 'oxe-moe', 'origin': 'organic'},
        )),
    'ram_short_llama': (
        [
            'require_backend_port 18080 LOCAL_LLM_BACKEND_PORT',
            'available_memory_bytes',
        ],
        unloaded(
            last_failure_info={'reason': 'This model needs about 3.0 GiB of available RAM; 1.0 GiB is available. Close other large apps and retry.', 'kind': 'start', 'alias': 'llama-local', 'origin': 'organic'},
        )),
    'ram_short_oxe': (
        [
            'require_backend_port 18080 LOCAL_LLM_BACKEND_PORT',
            'available_memory_bytes',
        ],
        unloaded(
            last_failure_info={'reason': 'This model needs about 3.0 GiB of available RAM; 1.0 GiB is available. Close other large apps and retry.', 'kind': 'start', 'alias': 'oxe-moe', 'origin': 'organic'},
        )),
    'ram_unknown_llama': (
        [
            'require_backend_port 18080 LOCAL_LLM_BACKEND_PORT',
            'available_memory_bytes',
            'guard_production_path <TMP>/state/generation.lock',
            'guard_production_path <TMP>/state/generation.lock',
            'open_lease_append',
            'flock a EX|NB',
        ],
        unloaded(
            last_failure_info={'reason': 'Another image, chat or setup job is running. Please try again when it finishes.', 'kind': 'start', 'alias': 'llama-local', 'origin': 'organic'},
        )),
    'bigmoe_unqualified': (
        [
            'require_backend_port 18090 LOCAL_FLASH_BACKEND_PORT',
            'available_memory_bytes',
            'bigmoe.Runtime bigmoe 18090',
        ],
        unloaded(
            last_failure_info={'reason': 'Flash profile is unqualified or has insufficient admitted RAM.', 'kind': 'start', 'alias': 'flash-stream', 'origin': 'organic'},
        )),
    'flash_files': (
        [
            'require_backend_port 18080 LOCAL_LLM_BACKEND_PORT',
            'available_memory_bytes',
            'flash_files_ready flash',
        ],
        unloaded(
            last_failure_info={'reason': 'Flash model shards or the expert sidecar are missing or changed.', 'kind': 'start', 'alias': 'oxe-flash', 'origin': 'organic'},
        )),
    'flash_power': (
        [
            'require_backend_port 18080 LOCAL_LLM_BACKEND_PORT',
            'available_memory_bytes',
            'flash_files_ready flash',
            'power_snapshot',
            'flash_power_failure snapshot starting=True',
        ],
        unloaded(
            last_failure_info={'reason': 'Flash needs at least 30% battery to start and stops at 20%.', 'kind': 'start', 'alias': 'oxe-flash', 'origin': 'organic'},
        )),
    'flash_select_pref': (
        [
            'require_backend_port 18080 LOCAL_LLM_BACKEND_PORT',
            'available_memory_bytes',
            'flash_files_ready flash',
            'power_snapshot',
            'flash_power_failure snapshot starting=True',
            'flash_cache_pref.enabled env=True',
            'total_memory_bytes',
            'swap_bytes',
            'cgroup_memory_max',
            'flash_select flash 68719476736 30064771072 34359738368 8589934592 env=True',
            'flash_cache_pref.enabled env=True',
        ],
        unloaded(
            last_failure_info={'reason': 'Flash cache preference cannot fit the current native memory budget; the host floor and runtime pressure guards remain active.', 'kind': 'start', 'alias': 'oxe-flash', 'origin': 'organic'},
        )),
    'flash_select_fixed': (
        [
            'require_backend_port 18080 LOCAL_LLM_BACKEND_PORT',
            'available_memory_bytes',
            'flash_files_ready flash',
            'power_snapshot',
            'flash_power_failure snapshot starting=True',
            'flash_cache_pref.enabled env=True',
            'cgroup_memory_max',
            'flash_select flash 68719476736 30064771072 None None env=True',
            'flash_cache_pref.enabled env=True',
        ],
        unloaded(
            last_failure_info={'reason': 'Flash (Rust) needs about 20.0 GiB of available RAM; close other large apps and retry.', 'kind': 'start', 'alias': 'oxe-flash', 'origin': 'organic'},
        )),
    'lease_busy_llama': (
        [
            'require_backend_port 18080 LOCAL_LLM_BACKEND_PORT',
            'available_memory_bytes',
            'guard_production_path <TMP>/state/generation.lock',
            'guard_production_path <TMP>/state/generation.lock',
            'open_lease_append',
            'flock a EX|NB',
        ],
        unloaded(
            last_failure_info={'reason': 'Another image, chat or setup job is running. Please try again when it finishes.', 'kind': 'start', 'alias': 'llama-local', 'origin': 'organic'},
        )),
    'lease_busy_bigmoe': (
        [
            'require_backend_port 18090 LOCAL_FLASH_BACKEND_PORT',
            'available_memory_bytes',
            'bigmoe.Runtime bigmoe 18090',
            'guard_production_path <TMP>/state/generation.lock',
            'flock r+ EX|NB',
        ],
        unloaded(
            last_failure_info={'reason': 'Another image, chat or setup job is running. Please try again when it finishes.', 'kind': 'start', 'alias': 'flash-stream', 'origin': 'organic'},
        )),
    'oxe_binary_missing': (
        [
            'require_backend_port 18080 LOCAL_LLM_BACKEND_PORT',
            'available_memory_bytes',
            'guard_production_path <TMP>/state/generation.lock',
            'guard_production_path <TMP>/state/generation.lock',
            'open_lease_append',
            'flock a EX|NB',
            'oxe.verify_engine oxe',
            'record_transition <TMP>/engine-ledger.log None oxe-moe success=False',
        ],
        unloaded(
            last_failure_info={'reason': 'The Rust engine binary is missing.', 'kind': 'start', 'alias': 'oxe-moe', 'origin': 'organic'},
        )),
    'oxe_identity_changed': (
        [
            'require_backend_port 18080 LOCAL_LLM_BACKEND_PORT',
            'available_memory_bytes',
            'guard_production_path <TMP>/state/generation.lock',
            'guard_production_path <TMP>/state/generation.lock',
            'open_lease_append',
            'flock a EX|NB',
            'oxe.verify_engine oxe',
            'record_transition <TMP>/engine-ledger.log None oxe-moe success=False',
        ],
        unloaded(
            last_failure_info={'reason': 'Local model could not be started.', 'kind': 'start', 'alias': 'oxe-moe', 'origin': 'organic', 'detail': 'ValueError: oxe_engine_identity_changed'},
        )),
    'beta_identity_changed': (
        [
            'require_backend_port 18080 LOCAL_LLM_BACKEND_PORT',
            'available_memory_bytes',
            'guard_production_path <TMP>/state/generation.lock',
            'guard_production_path <TMP>/state/generation.lock',
            'open_lease_append',
            'flock a EX|NB',
            'classic_beta.verify_engine llama',
            'record_transition <TMP>/engine-ledger.log None ornith-35b-a3b-local success=False',
        ],
        unloaded(
            last_failure_info={'reason': 'Local model could not be started.', 'kind': 'start', 'alias': 'ornith-35b-a3b-local', 'origin': 'organic', 'detail': 'ValueError: beta_engine_identity_changed'},
        )),
    'state_cache_root': (
        [
            'require_backend_port 18080 LOCAL_LLM_BACKEND_PORT',
            'available_memory_bytes',
            'guard_production_path <TMP>/state/generation.lock',
            'guard_production_path <TMP>/state/generation.lock',
            'open_lease_append',
            'flock a EX|NB',
            'oxe.verify_engine oxe',
            'oxe.command oxe 18080',
            'record_transition <TMP>/engine-ledger.log None oxe-moe success=False',
        ],
        unloaded(
            last_failure_info={'reason': 'LOCAL_AI_STATE_CACHE_ROOT (or runtime.json cache.root) must be an absolute path inside your home; the model was not started.', 'kind': 'start', 'alias': 'oxe-moe', 'origin': 'organic'},
        )),
    'command_other': (
        [
            'require_backend_port 18080 LOCAL_LLM_BACKEND_PORT',
            'available_memory_bytes',
            'flash_files_ready flash',
            'power_snapshot',
            'flash_power_failure snapshot starting=True',
            'flash_cache_pref.enabled env=True',
            'cgroup_memory_max',
            'flash_select flash 68719476736 30064771072 None None env=True',
            'flash_swap_tolerant_guard flash-sel 68719476736 None None',
            'guard_production_path <TMP>/state/generation.lock',
            'guard_production_path <TMP>/state/generation.lock',
            'open_lease_append',
            'flock a EX|NB',
            'oxe.verify_engine flash-sel',
            'oxe.command flash-sel 18080',
            'record_transition <TMP>/engine-ledger.log None oxe-flash success=False',
        ],
        unloaded(
            last_failure_info={'reason': 'Local model could not be started.', 'kind': 'start', 'alias': 'oxe-flash', 'origin': 'organic', 'detail': 'ValueError: oxe_unknown_argument_placeholder'},
        )),
    'policy_unqualified': (
        [
            'require_backend_port 18080 LOCAL_LLM_BACKEND_PORT',
            'available_memory_bytes',
            'flash_files_ready flash',
            'power_snapshot',
            'flash_power_failure snapshot starting=True',
            'flash_cache_pref.enabled env=True',
            'cgroup_memory_max',
            'flash_select flash 68719476736 30064771072 None None env=True',
            'flash_swap_tolerant_guard flash-sel 68719476736 None None',
            'guard_production_path <TMP>/state/generation.lock',
            'guard_production_path <TMP>/state/generation.lock',
            'open_lease_append',
            'flock a EX|NB',
            'oxe.verify_engine flash-sel',
            'oxe.command flash-sel 18080',
            'flash_policy flash-sel',
            'record_transition <TMP>/engine-ledger.log None oxe-flash success=False',
        ],
        unloaded(
            last_failure_info={'reason': 'The Flash memory policy in this profile is not a qualified one.', 'kind': 'start', 'alias': 'oxe-flash', 'origin': 'organic'},
        )),
    'policy_unqualified_bigmoe': (
        [
            'require_backend_port 18090 LOCAL_FLASH_BACKEND_PORT',
            'available_memory_bytes',
            'bigmoe.Runtime bigmoe 18090',
            'guard_production_path <TMP>/state/generation.lock',
            'flock r+ EX|NB',
            'flash_policy bigmoe',
            'runtime.stop_guard',
            'record_transition <TMP>/engine-ledger.log None flash-stream success=False',
        ],
        unloaded(
            last_failure_info={'reason': 'The Flash memory policy in this profile is not a qualified one.', 'kind': 'start', 'alias': 'flash-stream', 'origin': 'organic'},
        )),
    'policy_on_llama': (
        [
            'require_backend_port 18080 LOCAL_LLM_BACKEND_PORT',
            'available_memory_bytes',
            'guard_production_path <TMP>/state/generation.lock',
            'guard_production_path <TMP>/state/generation.lock',
            'open_lease_append',
            'flock a EX|NB',
            'flash_policy llama',
            'record_transition <TMP>/engine-ledger.log None llama-local success=False',
        ],
        unloaded(
            last_failure_info={'reason': 'A Flash memory policy is set on a model that is not Flash.', 'kind': 'start', 'alias': 'llama-local', 'origin': 'organic'},
        )),
    'policy_on_oxe_moe': (
        [
            'require_backend_port 18080 LOCAL_LLM_BACKEND_PORT',
            'available_memory_bytes',
            'guard_production_path <TMP>/state/generation.lock',
            'guard_production_path <TMP>/state/generation.lock',
            'open_lease_append',
            'flock a EX|NB',
            'oxe.verify_engine oxe',
            'oxe.command oxe 18080',
            'flash_policy oxe',
            'record_transition <TMP>/engine-ledger.log None oxe-moe success=False',
        ],
        unloaded(
            last_failure_info={'reason': 'A Flash memory policy is set on a model that is not Flash.', 'kind': 'start', 'alias': 'oxe-moe', 'origin': 'organic'},
        )),
    'token_dir_llama': (
        [
            'require_backend_port 18080 LOCAL_LLM_BACKEND_PORT',
            'available_memory_bytes',
            'guard_production_path <TMP>/state/generation.lock',
            'guard_production_path <TMP>/state/generation.lock',
            'open_lease_append',
            'flock a EX|NB',
            'flash_policy llama',
            'new_api_key 18080',
            'record_transition <TMP>/engine-ledger.log None llama-local success=False',
        ],
        unloaded(
            last_failure_info={'reason': 'Local model cannot start: no private runtime directory for its access token.', 'kind': 'start', 'alias': 'llama-local', 'origin': 'organic'},
        )),
    'token_dir_oxe': (
        [
            'require_backend_port 18080 LOCAL_LLM_BACKEND_PORT',
            'available_memory_bytes',
            'guard_production_path <TMP>/state/generation.lock',
            'guard_production_path <TMP>/state/generation.lock',
            'open_lease_append',
            'flock a EX|NB',
            'oxe.verify_engine oxe',
            'oxe.command oxe 18080',
            'flash_policy oxe',
            'launch_environment',
            'new_api_key 18080',
            'record_transition <TMP>/engine-ledger.log None oxe-moe success=False',
        ],
        unloaded(
            last_failure_info={'reason': 'Local model cannot start: no private runtime directory for its access token.', 'kind': 'start', 'alias': 'oxe-moe', 'origin': 'organic'},
        )),
    'memory_policy_error': (
        [
            'require_backend_port 18080 LOCAL_LLM_BACKEND_PORT',
            'available_memory_bytes',
            'flash_files_ready flash',
            'power_snapshot',
            'flash_power_failure snapshot starting=True',
            'flash_cache_pref.enabled env=True',
            'cgroup_memory_max',
            'flash_select flash 68719476736 30064771072 None None env=True',
            'flash_swap_tolerant_guard flash-sel 68719476736 None None',
            'guard_production_path <TMP>/state/generation.lock',
            'guard_production_path <TMP>/state/generation.lock',
            'open_lease_append',
            'flock a EX|NB',
            'oxe.verify_engine flash-sel',
            'oxe.command flash-sel 18080',
            'flash_policy flash-sel',
            'launch_environment',
            'new_api_key 18080',
            'MemoryGuard raises',
            'remove_api_key /run/key-1',
            'record_transition <TMP>/engine-ledger.log None oxe-flash success=False',
        ],
        unloaded(
            last_failure_info={'reason': 'Local model cannot load under the current memory conditions.', 'kind': 'start', 'alias': 'oxe-flash', 'origin': 'organic', 'detail': 'host floor'},
            last_memory_guard={'reason': 'host floor'},
        )),
    'popen_error': (
        [
            'require_backend_port 18080 LOCAL_LLM_BACKEND_PORT',
            'available_memory_bytes',
            'guard_production_path <TMP>/state/generation.lock',
            'guard_production_path <TMP>/state/generation.lock',
            'open_lease_append',
            'flock a EX|NB',
            'oxe.verify_engine oxe',
            'oxe.command oxe 18080',
            'flash_policy oxe',
            'launch_environment',
            'new_api_key 18080',
            'MemoryGuard host_guard_version=GUARD-2 loading_timeout_seconds=10.0',
            'EngineLogWriter <TMP>/engine-stderr.log',
            'ledger.enabled False',
            'Popen',
            'remove_api_key /run/key-1',
            'guard.stop',
            'record_transition <TMP>/engine-ledger.log None oxe-moe success=False',
        ],
        unloaded(
            last_failure_info={'reason': 'Local model could not be started.', 'kind': 'start', 'alias': 'oxe-moe', 'origin': 'organic', 'detail': 'OSError: exec failed'},
            last_memory_guard={'guard': 'metadata'},
        )),
    'exit_llama': (
        [
            'require_backend_port 18080 LOCAL_LLM_BACKEND_PORT',
            'available_memory_bytes',
            'guard_production_path <TMP>/state/generation.lock',
            'guard_production_path <TMP>/state/generation.lock',
            'open_lease_append',
            'flock a EX|NB',
            'flash_policy llama',
            'new_api_key 18080',
            'MemoryGuard host_guard_version=GUARD-2 loading_timeout_seconds=10.0',
            'EngineLogWriter <TMP>/engine-stderr.log',
            'ledger.enabled False',
            'Popen',
            'EngineStderrReader hook=[] stream=True writer=True',
            'EngineStderrReader.start',
            'guard.start',
            'writer.write {"engine_sha256": null, "event": "engine_died", "exit_code": 1, "model_alias": "llama-local", "pid": 4000123}',
            'remove_api_key /run/key-1',
            'EngineStderrReader.stop',
            'guard.stop',
            'record_transition <TMP>/engine-ledger.log None llama-local success=False',
        ],
        unloaded(
            last_failure_info={'reason': 'Local model backend exited during loading. Setup diagnostics are in the LLM project.', 'kind': 'start', 'alias': 'llama-local', 'origin': 'organic'},
            last_memory_guard={'guard': 'metadata'},
        )),
    'exit_bigmoe': (
        [
            'require_backend_port 18090 LOCAL_FLASH_BACKEND_PORT',
            'available_memory_bytes',
            'bigmoe.Runtime bigmoe 18090',
            'guard_production_path <TMP>/state/generation.lock',
            'flock r+ EX|NB',
            'flash_policy bigmoe',
            'available_memory_bytes',
            'runtime.require_memory 68719476736 lease_owned=True',
            'runtime.prepare lease_owned=True',
            'runtime.environment',
            'EngineLogWriter <TMP>/engine-stderr.log',
            'ledger.enabled False',
            'Popen',
            'EngineStderrReader hook=[] stream=True writer=True',
            'EngineStderrReader.start',
            'runtime.start_guard probe=True lease_owned=True',
            'writer.write {"engine_sha256": null, "event": "engine_died", "exit_code": 1, "model_alias": "flash-stream", "pid": 4000123}',
            'remove_api_key None',
            'EngineStderrReader.stop',
            'runtime.stop_guard',
            'record_transition <TMP>/engine-ledger.log None flash-stream success=False',
        ],
        unloaded(
            last_failure_info={'reason': 'Local model backend exited during loading. Setup diagnostics are in the LLM project.', 'kind': 'start', 'alias': 'flash-stream', 'origin': 'organic'},
        )),
    'guard_start_llama': (
        [
            'require_backend_port 18080 LOCAL_LLM_BACKEND_PORT',
            'available_memory_bytes',
            'guard_production_path <TMP>/state/generation.lock',
            'guard_production_path <TMP>/state/generation.lock',
            'open_lease_append',
            'flock a EX|NB',
            'flash_policy llama',
            'new_api_key 18080',
            'MemoryGuard host_guard_version=GUARD-2 loading_timeout_seconds=10.0',
            'EngineLogWriter <TMP>/engine-stderr.log',
            'ledger.enabled False',
            'Popen',
            'EngineStderrReader hook=[] stream=True writer=True',
            'EngineStderrReader.start',
            'guard.start',
            'writer.write {"engine_sha256": null, "event": "engine_died", "exit_code": 1, "guard_failure": "guard-oom", "model_alias": "llama-local", "pid": 4000123}',
            'remove_api_key /run/key-1',
            'EngineStderrReader.stop',
            'guard.stop',
            'record_transition <TMP>/engine-ledger.log None llama-local success=False',
        ],
        unloaded(
            last_failure_info={'reason': 'Local model stopped to preserve system memory.', 'kind': 'start', 'alias': 'llama-local', 'origin': 'organic', 'detail': 'guard-oom'},
            last_memory_guard={'guard': 'metadata'},
        )),
    'guard_ready_oxe': (
        [
            'require_backend_port 18080 LOCAL_LLM_BACKEND_PORT',
            'available_memory_bytes',
            'guard_production_path <TMP>/state/generation.lock',
            'guard_production_path <TMP>/state/generation.lock',
            'open_lease_append',
            'flock a EX|NB',
            'oxe.verify_engine oxe',
            'oxe.command oxe 18080',
            'flash_policy oxe',
            'launch_environment',
            'new_api_key 18080',
            'MemoryGuard host_guard_version=GUARD-2 loading_timeout_seconds=10.0',
            'EngineLogWriter <TMP>/engine-stderr.log',
            'ledger.enabled False',
            'Popen',
            'EngineStderrReader hook=[] stream=True writer=True',
            'EngineStderrReader.start',
            'guard.start',
            'http GET /health 127.0.0.1 18080 t=2 auth=TKN',
            'guard.mark_ready',
            'killpg 4000123 SIGTERM',
            'writer.write {"engine_sha256": null, "event": "engine_died", "exit_code": -15, "guard_failure": "guard-oom", "model_alias": "oxe-moe", "pid": 4000123}',
            'remove_api_key /run/key-1',
            'EngineStderrReader.stop',
            'guard.stop',
            'record_transition <TMP>/engine-ledger.log None oxe-moe success=False',
        ],
        unloaded(
            last_failure_info={'reason': 'Local model stopped to preserve system memory.', 'kind': 'start', 'alias': 'oxe-moe', 'origin': 'organic', 'detail': 'guard-oom'},
            last_memory_guard={'guard': 'metadata'},
        )),
    'power_guard_flash': (
        [
            'require_backend_port 18080 LOCAL_LLM_BACKEND_PORT',
            'available_memory_bytes',
            'flash_files_ready flash',
            'power_snapshot',
            'flash_power_failure snapshot starting=True',
            'flash_cache_pref.enabled env=True',
            'cgroup_memory_max',
            'flash_select flash 68719476736 30064771072 None None env=True',
            'flash_swap_tolerant_guard flash-sel 68719476736 None None',
            'guard_production_path <TMP>/state/generation.lock',
            'guard_production_path <TMP>/state/generation.lock',
            'open_lease_append',
            'flock a EX|NB',
            'oxe.verify_engine flash-sel',
            'oxe.command flash-sel 18080',
            'flash_policy flash-sel',
            'launch_environment',
            'new_api_key 18080',
            'MemoryGuard host_guard_version=GUARD-2 loading_timeout_seconds=10.0',
            'EngineLogWriter <TMP>/engine-stderr.log',
            'ledger.enabled False',
            'Popen',
            "EngineStderrReader hook=['on_line'] stream=True writer=True",
            'EngineStderrReader.start',
            'guard.start',
            'FlashPowerGuard',
            'FlashPowerGuard.start',
            'writer.write {"engine_sha256": null, "event": "engine_died", "exit_code": 1, "model_alias": "oxe-flash", "pid": 4000123}',
            'remove_api_key /run/key-1',
            'EngineStderrReader.stop',
            'guard.stop',
            'FlashPowerGuard.stop',
            'record_transition <TMP>/engine-ledger.log None oxe-flash success=False',
        ],
        unloaded(
            last_failure_info={'reason': 'Flash stopped to preserve power.', 'kind': 'start', 'alias': 'oxe-flash', 'origin': 'organic', 'detail': 'battery'},
            last_memory_guard={'guard': 'metadata'},
        )),
    'runtime_guard_bigmoe': (
        [
            'require_backend_port 18090 LOCAL_FLASH_BACKEND_PORT',
            'available_memory_bytes',
            'bigmoe.Runtime bigmoe 18090',
            'guard_production_path <TMP>/state/generation.lock',
            'flock r+ EX|NB',
            'flash_policy bigmoe',
            'available_memory_bytes',
            'runtime.require_memory 68719476736 lease_owned=True',
            'runtime.prepare lease_owned=True',
            'runtime.environment',
            'EngineLogWriter <TMP>/engine-stderr.log',
            'ledger.enabled False',
            'Popen',
            'EngineStderrReader hook=[] stream=True writer=True',
            'EngineStderrReader.start',
            'runtime.start_guard probe=True lease_owned=True',
            'writer.write {"engine_sha256": null, "event": "engine_died", "exit_code": 1, "guard_failure": "runtime-guard", "model_alias": "flash-stream", "pid": 4000123}',
            'remove_api_key None',
            'EngineStderrReader.stop',
            'runtime.stop_guard',
            'record_transition <TMP>/engine-ledger.log None flash-stream success=False',
        ],
        unloaded(
            last_failure_info={'reason': 'Flash stopped to preserve power or available memory.', 'kind': 'start', 'alias': 'flash-stream', 'origin': 'organic', 'detail': 'runtime-guard'},
        )),
    'foreign_listener': (
        [
            'require_backend_port 18080 LOCAL_LLM_BACKEND_PORT',
            'available_memory_bytes',
            'guard_production_path <TMP>/state/generation.lock',
            'guard_production_path <TMP>/state/generation.lock',
            'open_lease_append',
            'flock a EX|NB',
            'flash_policy llama',
            'new_api_key 18080',
            'MemoryGuard host_guard_version=GUARD-2 loading_timeout_seconds=10.0',
            'EngineLogWriter <TMP>/engine-stderr.log',
            'ledger.enabled False',
            'Popen',
            'EngineStderrReader hook=[] stream=True writer=True',
            'EngineStderrReader.start',
            'guard.start',
            'http GET /health 127.0.0.1 18080 t=2 auth=TKN',
            'port_listener_in_group 18080 4000123',
            'listening_process 18080',
            'killpg 4000123 SIGTERM',
            'remove_api_key /run/key-1',
            'EngineStderrReader.stop',
            'guard.stop',
            'record_transition <TMP>/engine-ledger.log None llama-local success=False',
        ],
        unloaded(
            last_failure_info={'reason': 'Local engine port 18080 is answered by other-app (pid 77), not the engine this controller started. Close that app or set LOCAL_LLM_BACKEND_PORT to an unused port.', 'kind': 'start', 'alias': 'llama-local', 'origin': 'organic'},
            last_memory_guard={'guard': 'metadata'},
        )),
    'timeout_llama': (
        [
            'require_backend_port 18080 LOCAL_LLM_BACKEND_PORT',
            'available_memory_bytes',
            'guard_production_path <TMP>/state/generation.lock',
            'guard_production_path <TMP>/state/generation.lock',
            'open_lease_append',
            'flock a EX|NB',
            'flash_policy llama',
            'new_api_key 18080',
            'MemoryGuard host_guard_version=GUARD-2 loading_timeout_seconds=0.0',
            'EngineLogWriter <TMP>/engine-stderr.log',
            'ledger.enabled False',
            'Popen',
            'EngineStderrReader hook=[] stream=True writer=True',
            'EngineStderrReader.start',
            'guard.start',
            'killpg 4000123 SIGTERM',
            'remove_api_key /run/key-1',
            'EngineStderrReader.stop',
            'guard.stop',
            'record_transition <TMP>/engine-ledger.log None llama-local success=False',
        ],
        unloaded(
            last_failure_info={'reason': 'Local model loading timed out; no generation was started.', 'kind': 'start', 'alias': 'llama-local', 'origin': 'organic'},
            last_memory_guard={'guard': 'metadata'},
        )),
    'timeout_oxe': (
        [
            'require_backend_port 18080 LOCAL_LLM_BACKEND_PORT',
            'available_memory_bytes',
            'guard_production_path <TMP>/state/generation.lock',
            'guard_production_path <TMP>/state/generation.lock',
            'open_lease_append',
            'flock a EX|NB',
            'oxe.verify_engine oxe',
            'oxe.command oxe 18080',
            'flash_policy oxe',
            'launch_environment',
            'new_api_key 18080',
            'MemoryGuard host_guard_version=GUARD-2 loading_timeout_seconds=0.0',
            'EngineLogWriter <TMP>/engine-stderr.log',
            'ledger.enabled False',
            'Popen',
            'EngineStderrReader hook=[] stream=True writer=True',
            'EngineStderrReader.start',
            'guard.start',
            'killpg 4000123 SIGTERM',
            'remove_api_key /run/key-1',
            'EngineStderrReader.stop',
            'guard.stop',
            'record_transition <TMP>/engine-ledger.log None oxe-moe success=False',
        ],
        unloaded(
            last_failure_info={'reason': 'Local model loading timed out; no generation was started.', 'kind': 'start', 'alias': 'oxe-moe', 'origin': 'organic'},
            last_memory_guard={'guard': 'metadata'},
        )),
    'timeout_bigmoe': (
        [
            'require_backend_port 18090 LOCAL_FLASH_BACKEND_PORT',
            'available_memory_bytes',
            'bigmoe.Runtime bigmoe 18090',
            'guard_production_path <TMP>/state/generation.lock',
            'flock r+ EX|NB',
            'flash_policy bigmoe',
            'available_memory_bytes',
            'runtime.require_memory 68719476736 lease_owned=True',
            'runtime.prepare lease_owned=True',
            'runtime.environment',
            'EngineLogWriter <TMP>/engine-stderr.log',
            'ledger.enabled False',
            'Popen',
            'EngineStderrReader hook=[] stream=True writer=True',
            'EngineStderrReader.start',
            'runtime.start_guard probe=True lease_owned=True',
            'killpg 4000123 SIGTERM',
            'remove_api_key None',
            'EngineStderrReader.stop',
            'runtime.stop_guard',
            'record_transition <TMP>/engine-ledger.log None flash-stream success=False',
        ],
        unloaded(
            last_failure_info={'reason': 'Local model loading timed out; no generation was started.', 'kind': 'start', 'alias': 'flash-stream', 'origin': 'organic'},
        )),
}


class BeginGolden(unittest.TestCase):
    maxDiff = None

    def run_begin(self, kind, changes, knobs, attributes=()):
        tmp = tempfile.TemporaryDirectory(dir='/tmp')
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        (root / 'state').mkdir()
        (root / 'state/generation.lock').write_bytes(b'')
        model = root / 'model.gguf'
        model.write_bytes(b'm' * 100)
        h = Harness(root)
        knobs = dict(knobs)
        allow_start = knobs.pop('allow_start', True)
        eligible = knobs.pop('eligible', True)
        repack = knobs.pop('repack', False)
        for key, value in knobs.items():
            self.assertTrue(hasattr(h, key), key)
            setattr(h, key, value)
        environment = {key: value for key, value in os.environ.items()
                       if not key.startswith(('GGML_', 'LLAMA_')) and key not in (
                           explicit_route.ENABLE_ENV, oxe_runtime.flash_cache_pref.ENV, oxe_runtime.API_KEY_ENV)}
        environment.update({explicit_route.ENABLE_ENV: '1', 'GGML_BASE_TUNING': '1', 'LLAMA_BASE_TUNING': '1'})
        if repack:
            environment[controller.REPACK_CACHE_ENV] = str(root / 'repack')
        config = KINDS[kind](model, **changes)
        patches = [patch.dict(os.environ, environment, clear=True), *h.patches()]
        for item in patches:
            item.start()
        try:
            manager = controller.Manager()
            manager.stop_event.set()
            manager.thread.join(timeout=5)
            manager.config = Mock(return_value=config)
            manager.eligible = Mock(return_value=eligible)
            manager._flash_idle_qos = lambda verb, cfg, pid: (
                h.rec('flash_idle_qos', verb, cfg.get('tag'), pid), (None, False, None))[1]
            for key, value in dict(attributes).items():
                setattr(manager, key, value)
            if 'stop_pending' in dict(attributes):
                manager.process = FakeProcess(h, immortal=True)
                h.processes.append(manager.process)
            error = None
            try:
                manager.begin(inference=True, allow_start=allow_start, model_alias=config['model_alias'])
            except Exception as caught:  # the golden data names the class and message
                error = caught
            trace, observed = list(h.trace), self.observe(manager, h)
            for process in h.processes:
                process.immortal = False
                if process.code is None:
                    process.code = 0
            with manager.mu:
                manager.image_active = False
                manager.cleanup()
        finally:
            for item in reversed(patches):
                item.stop()
        return error, trace, h.launch, observed

    def observe(self, manager, h):
        process = manager.process
        info = manager.last_failure_info
        return h.clean({
            'state': manager.state, 'launches': manager.launches, 'active': manager.active,
            'inferences': manager.inferences, 'state_engine': manager.state_engine,
            'state_preload': manager.state_preload, 'prefix_store': manager.prefix_store is not None,
            'lease_mode': getattr(manager.lease, 'mode', None),
            'loaded_tag': (manager.loaded_config or {}).get('tag'),
            'process_attrs': None if process is None else (
                vars(process).get('_local_ai_engine_config', {}).get('tag'),
                vars(process).get('_local_ai_api_key'), str(vars(process).get('_local_ai_api_key_file'))
                if vars(process).get('_local_ai_api_key_file') is not None else None),
            'backend_runtime': manager.backend_runtime is not None,
            'classic_guard': manager.classic_memory_guard is not None,
            'flash_power_guard': manager.flash_power_guard is not None,
            'engine_reader': manager.engine_reader is not None, 'repack_reader': manager.repack_reader is not None,
            'engine_ledger': manager.engine_ledger is not None, 'flash_cache_status': manager.flash_cache_status,
            'cache': sorted(manager.cache), 'flash_idle_qos_pid': manager.flash_idle_qos_pid,
            'last_failure_info': None if info is None else {k: v for k, v in info.items() if k != 't'},
            'last_memory_guard': manager.last_memory_guard, 'stop_pending': manager.stop_pending,
        })

    def test_successful_launches(self):
        for name, (kind, changes, knobs, trace, spec, after) in SUCCESS.items():
            with self.subTest(name):
                error, actual_trace, actual_launch, observed = self.run_begin(kind, changes, knobs)
                self.assertIsNone(error, msg=f'{name}: {error!r} after {actual_trace!r}')
                self.assertEqual(actual_trace, trace, msg=f'\nGOLDEN-TRACE {name} = {actual_trace!r}')
                self.assertEqual(actual_launch, spec, msg=f'\nGOLDEN-LAUNCH {name} = {actual_launch!r}')
                self.assertEqual(observed, after, msg=f'\nGOLDEN-STATE {name} = {observed!r}')

    def test_refusals_and_cleanup(self):
        for name, (kind, changes, knobs, attributes, kind_of_error, message) in FAILURES.items():
            with self.subTest(name):
                error, actual_trace, _, observed = self.run_begin(kind, changes, knobs, attributes)
                self.assertIsNotNone(error, msg=f'{name}: no error after {actual_trace!r}')
                self.assertEqual((type(error).__name__, str(error)), (kind_of_error, message),
                                 msg=f'{name}: {actual_trace!r}')
                golden = FAILURE_GOLDEN.get(name)
                self.assertEqual((actual_trace, observed), golden,
                                 msg=f'\nGOLDEN-FAILURE {name!r}: ({actual_trace!r}, {observed!r}),')
                if any(line.startswith('flock') for line in actual_trace) and MSG_LEASE not in str(error) \
                        and name != 'stop_pending':
                    # After the lease was taken: cleanup ran, the lease is released and the
                    # transition is recorded as failed, last.
                    self.assertEqual(actual_trace[-1],
                                     f"record_transition <TMP>/engine-ledger.log None "
                                     f"{KINDS[kind](Path('m'), **changes)['model_alias']} success=False")
                    self.assertEqual((observed['state'], observed['lease_mode'], observed['process_attrs']),
                                     ('unloaded', None, None))

    def test_unavailable_errors_drop_their_context(self):
        """`from None`: the refusals that translate a lower error do not chain it."""
        for name in ('oxe_binary_missing', 'state_cache_root', 'policy_unqualified', 'token_dir_oxe',
                     'memory_policy_error', 'flash_select_fixed', 'bigmoe_unqualified'):
            with self.subTest(name):
                kind, changes, knobs, attributes, _, _ = FAILURES[name]
                error, *_ = self.run_begin(kind, changes, knobs, attributes)
                self.assertIsNone(error.__cause__)
                self.assertTrue(error.__suppress_context__)


if __name__ == '__main__':
    unittest.main()
