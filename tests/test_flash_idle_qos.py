"""Flash idle-QoS hook: profile validation, unit/command helpers and the controller's best-effort wiring.

No model, no systemctl: subprocess.run and Popen are patched, every path is a temp directory.
"""
import copy
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

from snapdragon_ai_lab.runtime import controller, oxe_runtime, profile_policy
from test_flash_oxe_production import GIB, flash_profile

IDLE = {'unit': 'local-ai-flash-qos@{pid}.service', 'cpu': 8, 'latency_us': 100}
UNIT_4242 = 'local-ai-flash-qos@4242.service'


def resolved(idle=True):
    profile = flash_profile()
    if not idle:
        profile['memlat_overrides']['oxe_flash'].pop('idle_qos')
    return profile_policy.resolve(profile, available=True, oxe_flash_enabled=True)


def with_idle(value):
    profile = flash_profile()
    profile['memlat_overrides']['oxe_flash']['idle_qos'] = value
    return profile


class IdleQosValidation(unittest.TestCase):
    def test_valid_key_is_accepted_by_both_validators_and_carried_into_the_resolved_override(self):
        effective = resolved()
        self.assertEqual(effective['oxe_flash']['idle_qos'], IDLE)
        self.assertIsNone(oxe_runtime._validate(effective))
        profile = with_idle(copy.deepcopy(IDLE))
        self.assertEqual(profile_policy.resolve(profile, available=True, oxe_flash_enabled=False), profile)

    def test_resolved_copy_is_detached_from_the_profile(self):
        profile = with_idle(copy.deepcopy(IDLE))
        effective = profile_policy.resolve(profile, available=True, oxe_flash_enabled=True)
        effective['oxe_flash']['idle_qos']['cpu'] = 3
        self.assertEqual(profile['memlat_overrides']['oxe_flash']['idle_qos']['cpu'], 8)

    def test_wrong_values_extra_or_missing_keys_and_bools_are_rejected_in_both_validators(self):
        bad = {
            'wrong unit': {**IDLE, 'unit': 'local-ai-flash-qos@1.service'},
            'timer instead of service': {**IDLE, 'unit': 'local-ai-flash-qos@{pid}.timer'},
            'glob unit': {**IDLE, 'unit': 'local-ai-flash-qos@*.service'},
            'cpu 7': {**IDLE, 'cpu': 7},
            'cpu as string': {**IDLE, 'cpu': '8'},
            'cpu as float': {**IDLE, 'cpu': 8.0},
            'cpu bool True': {**IDLE, 'cpu': True},
            'latency 50': {**IDLE, 'latency_us': 50},
            'latency as string': {**IDLE, 'latency_us': '100'},
            'latency bool': {**IDLE, 'latency_us': True},
            'extra key': {**IDLE, 'extra': 1},
            'missing latency': {'unit': IDLE['unit'], 'cpu': 8},
            'not a dict': 'yes',
            'list': [],
            'null': None,
        }
        for name, value in bad.items():
            with self.subTest(name), self.assertRaisesRegex(ValueError, 'memlat_oxe_flash_idle_qos_invalid'):
                profile_policy.resolve(with_idle(value), available=True, oxe_flash_enabled=False)
            with self.subTest(name, validator='oxe_runtime'):
                effective = resolved()
                effective['oxe_flash']['idle_qos'] = value
                with self.assertRaisesRegex(ValueError, 'oxe_flash_idle_qos_invalid'):
                    oxe_runtime._validate(effective)


class IdleQosHelpers(unittest.TestCase):
    def test_unit_name_for_a_positive_pid(self):
        self.assertEqual(oxe_runtime.flash_idle_qos_unit(resolved(), 12345), 'local-ai-flash-qos@12345.service')

    def test_unit_is_none_without_idle_qos(self):
        self.assertIsNone(oxe_runtime.flash_idle_qos_unit(resolved(idle=False), 12345))
        self.assertIsNone(oxe_runtime.flash_idle_qos_unit({'backend_kind': 'oxe'}, 12345))

    def test_pid_must_be_a_positive_int_and_not_a_bool(self):
        for pid in (0, -1, True, False, '12', 1.0, None):
            with self.subTest(pid=pid), self.assertRaisesRegex(ValueError, 'idle_qos_pid_invalid'):
                oxe_runtime.flash_idle_qos_unit(resolved(), pid)
            with self.subTest(pid=pid), self.assertRaisesRegex(ValueError, 'idle_qos_pid_invalid'):
                oxe_runtime.flash_idle_qos_unit(resolved(idle=False), pid)

    def test_commands_for_start_and_stop(self):
        for verb in ('start', 'stop'):
            with self.subTest(verb=verb):
                self.assertEqual(oxe_runtime.flash_idle_qos_command(resolved(), 12345, verb),
                                 ['systemctl', verb, '--no-block', 'local-ai-flash-qos@12345.service'])
                self.assertIsNone(oxe_runtime.flash_idle_qos_command(resolved(idle=False), 12345, verb))

    def test_other_verbs_are_refused(self):
        for verb in ('restart', 'reset', '', None):
            with self.subTest(verb=verb), self.assertRaisesRegex(ValueError, 'verb_invalid'):
                oxe_runtime.flash_idle_qos_command(resolved(), 12345, verb)


class FakeEngine:
    """Patches that let Manager.begin() run its real Flash load path with a fake engine pid."""
    def __init__(self, root, manager, config, *, meta_fails=False, props_body=b'{}', guard=None):
        self.root = root
        self.manager = manager
        self.config = config
        self.meta_fails = meta_fails
        model = root / 'model.gguf'
        model.write_bytes(b'x')
        config['model_path'] = str(model)
        config['expected_bytes'] = 1
        # No ledger: the flash profile's environment would otherwise enable the engine ledger.
        config['environment'] = {}
        self.process = Mock(pid=4242)
        self.process.poll.return_value = None
        health = Mock()
        health.getresponse.return_value = Mock(status=200, read=Mock(return_value=b'{"status":"ok"}'))
        # Native /props snapshot read after readiness (flash-pref controller); an older engine has none.
        props = Mock()
        props.getresponse.return_value = Mock(status=200, read=Mock(return_value=props_body))
        meta = Mock()
        if meta_fails:
            meta.request.side_effect = RuntimeError('synthetic metadata failure')
        else:
            meta.getresponse.return_value = Mock(status=200, read=Mock(return_value=b'{"status":"ok"}'))
        self.patches = [
            patch.dict(os.environ, {oxe_runtime.FLASH_ENABLE_ENV: '1'}),
            patch.object(manager, 'config', return_value=config),
            patch.object(controller, 'LEASE', root / 'lease'),
            patch.object(controller, 'ENGINE_DIAGNOSTIC_LOG', root / 'private/stderr.log'),
            patch.object(controller, 'ENGINE_LEDGER_LOG', root / 'private/ledger.jsonl'),
            patch.object(controller.bigmoe_runtime, 'power_snapshot', return_value={}),
            patch.object(controller, 'available_memory_bytes', return_value=40 * GIB),
            patch.object(controller, 'cgroup_memory_max', return_value=None),
            patch.object(controller.oxe_runtime, 'flash_files_ready', return_value=True),
            patch.object(controller.oxe_runtime, 'flash_power_failure', return_value=None),
            patch.object(controller.oxe_runtime, 'FlashPowerGuard', return_value=Mock(failure=None)),
            patch.object(controller.oxe_runtime, 'verify_engine'),
            patch.object(controller.memory_policy, 'MemoryGuard',
                         return_value=guard or Mock(failure=Mock(return_value=None))),
            patch.object(controller.subprocess, 'Popen', return_value=self.process),
            patch.object(controller, 'EngineStderrReader', return_value=Mock()),
            patch.object(controller.http.client, 'HTTPConnection', side_effect=[health, props, meta]),
            patch.object(controller.os, 'killpg'),
        ]

    def __enter__(self):
        for patcher in self.patches:
            patcher.start()
        return self

    def __exit__(self, *exc):
        for patcher in reversed(self.patches):
            patcher.stop()
        return False


class IdleQosController(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix='flash-idle-qos-'))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.manager = controller.Manager()
        self.manager.stop_event.set()
        self.addCleanup(self.manager.close)

    def load(self, config, **kwargs):
        engine = FakeEngine(self.tmp, self.manager, config, **kwargs)
        return engine

    def test_start_runs_once_after_ready_and_stop_runs_once_on_unload(self):
        with patch.object(controller.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0)) as run:
            with self.load(resolved()):
                self.assertIs(self.manager.begin(allow_start=True), self.manager.process)
                run.assert_called_once_with(['systemctl', 'start', '--no-block', UNIT_4242],
                                            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                            stderr=subprocess.DEVNULL, timeout=5, check=False)
                status = self.manager.status()['flash_idle_qos']
                self.assertEqual(status, {'enabled': True, 'unit': UNIT_4242, 'started': True, 'error': None})
                self.manager.cleanup()
                self.assertEqual(run.call_count, 2)
                run.assert_called_with(['systemctl', 'stop', '--no-block', UNIT_4242],
                                       stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                       stderr=subprocess.DEVNULL, timeout=5, check=False)
                self.assertEqual(self.manager.status()['flash_idle_qos'],
                                 {'enabled': True, 'unit': None, 'started': False, 'error': None})

    def test_failed_load_after_the_hook_still_runs_the_stop(self):
        with patch.object(controller.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0)) as run:
            with self.load(resolved(), meta_fails=True):
                with self.assertRaisesRegex(RuntimeError, 'synthetic metadata failure'):
                    self.manager.begin(allow_start=True)
                self.assertEqual([c.args[0][1] for c in run.call_args_list], ['start', 'stop'])
                self.assertIsNone(self.manager.loaded_config)
                self.assertEqual(self.manager.state, 'unloaded')

    def test_errors_are_reported_in_status_and_never_propagate(self):
        failures = [OSError('no systemctl'), subprocess.TimeoutExpired('systemctl', 5)]
        with patch.object(controller.subprocess, 'run', side_effect=failures) as run:
            with self.load(resolved()):
                self.assertIs(self.manager.begin(allow_start=True), self.manager.process)
                self.assertEqual(self.manager.status()['flash_idle_qos'],
                                 {'enabled': True, 'unit': UNIT_4242, 'started': False, 'error': 'OSError'})
                self.manager.cleanup()
                self.assertEqual(run.call_count, 2)
                self.assertEqual(self.manager.status()['flash_idle_qos']['error'], 'TimeoutExpired')
                self.assertEqual(self.manager.state, 'unloaded')

    def test_nonzero_systemctl_exit_is_reported_as_not_started(self):
        with patch.object(controller.subprocess, 'run', return_value=subprocess.CompletedProcess([], 1)):
            with self.load(resolved()):
                self.manager.begin(allow_start=True)
                self.assertEqual(self.manager.status()['flash_idle_qos'],
                                 {'enabled': True, 'unit': UNIT_4242, 'started': False, 'error': 'CalledProcessError'})

    def test_status_never_runs_systemctl_and_reports_disabled_without_the_key(self):
        with patch.object(controller.subprocess, 'run') as run:
            manager = self.manager
            for config, enabled in ((resolved(idle=False), False), (resolved(), True)):
                with patch.object(manager, 'config', return_value=config):
                    self.assertEqual(manager.status()['flash_idle_qos'],
                                     {'enabled': enabled, 'unit': None, 'started': False, 'error': None})
            run.assert_not_called()

    def test_profile_without_idle_qos_never_calls_systemctl(self):
        with patch.object(controller.subprocess, 'run') as run:
            with self.load(resolved(idle=False)):
                self.manager.begin(allow_start=True)
                self.manager.cleanup()
                self.assertEqual(self.manager.status()['flash_idle_qos'],
                                 {'enabled': False, 'unit': None, 'started': False, 'error': None})
            run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
