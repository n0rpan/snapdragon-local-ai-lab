"""M1: last_error is cleared on a successful start and on a new explicit load, is set for real
start failures, and lab-origin failures never surface as the user's error.

The success path uses a real child process (tests/fixtures/stub_backend/llama_stub.py) on a
real free port; no model is loaded.
"""
import socket
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from snapdragon_ai_lab.runtime import controller

STUB = Path(__file__).resolve().parent / 'fixtures/stub_backend/llama_stub.py'


def free_port():
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        return probe.getsockname()[1]


class FakeHeaders(dict):
    def get(self, name, default=None):
        for key, value in self.items():
            if key.lower() == name.lower():
                return value
        return default


class LastErrorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir='/tmp')
        root = Path(self.tmp.name)
        self.model = root / 'stub.gguf'
        self.model.write_bytes(b'GGUF')
        self.port = free_port()
        guard = Mock()
        guard.failure.return_value = None
        guard.metadata.return_value = {}
        self.patches = [
            patch.object(controller, 'LEASE', root / 'generation.lock'),
            patch.object(controller, 'BACKEND_PORT', self.port),
            patch.object(controller, 'ENGINE_DIAGNOSTIC_LOG', root / 'engine-stderr.log'),
            patch.object(controller, 'available_memory_bytes', return_value=64 * 1024**3),
            patch.object(controller, 'resident_bytes', return_value=None),
            patch.object(controller.memory_policy, 'MemoryGuard', return_value=guard),
        ]
        for item in self.patches:
            item.start()
        self.manager = controller.Manager()
        self.manager.stop_event.set()
        self.manager.eligible = Mock(return_value=True)
        self.use(self.config())

    def tearDown(self):
        with self.manager.mu:
            self.manager.cleanup()
        for item in reversed(self.patches):
            item.stop()
        self.tmp.cleanup()

    def config(self, *, fail=False):
        return {'model_alias': 'stub-local', 'ready': True, 'model_path': str(self.model),
                'server_binary': sys.executable,
                'extra_args': [] if not fail else ['--fail']}

    def use(self, config):
        self.manager.config = Mock(return_value=config)

    def run_stub(self, config):
        # The controller launches [server_binary, --model, ...]; put the stub script first.
        original = controller.subprocess.Popen

        def popen(args, **kwargs):
            return original([args[0], str(STUB), *args[1:]], **kwargs)
        return patch.object(controller.subprocess, 'Popen', side_effect=popen)

    def test_fail_then_succeed_clears_last_error(self):
        self.use(self.config(fail=True))
        with self.run_stub(None), self.assertRaisesRegex(controller.Unavailable, 'exited during loading'):
            self.manager.begin(allow_start=True, model_alias='stub-local')
        status = self.manager.status()
        self.assertIn('exited during loading', status['last_error'])
        self.assertEqual(status['last_failure']['kind'], 'start')
        self.assertEqual(status['last_failure']['origin'], 'organic')
        self.assertEqual(status['last_failure']['alias'], 'stub-local')
        self.assertIsInstance(status['last_failure']['t'], int)

        self.use(self.config())
        with self.run_stub(None):
            self.manager.begin(allow_start=True, model_alias='stub-local')
        self.manager.end(touch=True)
        status = self.manager.status()
        self.assertEqual(status['state'], 'ready')
        self.assertIsNone(status['last_error'])
        self.assertIsNone(status['last_failure'])

    def test_real_start_failure_port_busy_sets_last_error(self):
        self.use(self.config())
        with patch.object(controller, 'require_backend_port',
                          side_effect=controller.Unavailable('Port 8093 is in use by example-app.')):
            with self.assertRaises(controller.Unavailable):
                self.manager.begin(allow_start=True, model_alias='stub-local')
        self.assertEqual(self.manager.status()['last_error'], 'Port 8093 is in use by example-app.')

    def test_sleeping_or_busy_admission_is_not_a_start_failure(self):
        self.use(self.config())
        with self.assertRaisesRegex(controller.Unavailable, 'sleeping'):
            self.manager.begin(allow_start=False, model_alias='stub-local')
        self.manager.image_active = True
        with self.assertRaisesRegex(controller.Unavailable, 'image is running'):
            self.manager.begin(allow_start=True, model_alias='stub-local')
        self.manager.image_active = False
        self.assertIsNone(self.manager.status()['last_error'])

    def test_new_explicit_load_clears_an_old_failure(self):
        self.manager.record_failure('old guard stop', kind='guard', alias='stub-local')
        self.assertEqual(self.manager.status()['last_error'], 'old guard stop')
        self.use(self.config())
        with patch.object(controller, 'require_backend_port', side_effect=controller.Unavailable('busy port')):
            with self.assertRaises(controller.Unavailable):
                self.manager.begin(allow_start=True, model_alias='stub-local', explicit_load=True)
        # The old guard stop is gone; only this load's own failure is shown.
        self.assertEqual(self.manager.status()['last_error'], 'busy port')
        self.assertNotIn('detail', self.manager.status()['last_failure'])

    def test_lab_origin_failures_do_not_surface_as_user_errors(self):
        self.use(self.config())
        with patch.object(controller, 'require_backend_port', side_effect=controller.Unavailable('busy port')):
            with self.assertRaises(controller.Unavailable):
                self.manager.begin(allow_start=True, model_alias='stub-local', origin='lab')
        status = self.manager.status()
        self.assertIsNone(status['last_error'])
        self.assertEqual(status['last_failure']['origin'], 'lab')

    def test_guard_stop_takes_the_origin_of_the_running_work(self):
        self.manager.activity_origin = 'lab'
        guard = Mock()
        guard.failure.return_value = 'host memory floor reached'
        guard.metadata.return_value = {}
        with self.manager.mu:
            self.manager.classic_memory_guard = guard
            self.manager.cleanup()
        self.assertIsNone(self.manager.status()['last_error'])
        self.assertEqual(self.manager.status()['last_failure']['kind'], 'guard')
        self.manager.activity_origin = 'organic'
        guard.failure.return_value = 'host memory floor reached'
        with self.manager.mu:
            self.manager.classic_memory_guard = guard
            self.manager.cleanup()
        self.assertEqual(self.manager.status()['last_error'], 'host memory floor reached')

    def test_request_origin_reads_the_real1_header(self):
        self.assertEqual(controller.request_origin(FakeHeaders({'X-Local-AI-Origin': ' Lab '})), 'lab')
        self.assertEqual(controller.request_origin(FakeHeaders({'X-Local-AI-Origin': 'other'})), 'organic')
        self.assertEqual(controller.request_origin(FakeHeaders()), 'organic')
        self.assertEqual(controller.request_origin(None), 'organic')


if __name__ == '__main__':
    unittest.main()
