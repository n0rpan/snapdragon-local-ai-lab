"""A real controller.Manager driving the model-free llama stub (tests/fixtures/stub_backend).

Not a test module (no test_ prefix): shared by behavioural tests that need a real child
process on a real loopback port, the real readiness loop and real cleanup, but never a model.
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


class StubManagerCase(unittest.TestCase):
    """setUp gives self.manager (reaper stopped), self.port (the backend port) and self.root."""
    preflight = True  # False: skip the port preflight (simulates the probe-to-bind race)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir='/tmp')
        self.root = Path(self.tmp.name)
        self.model = self.root / 'stub.gguf'
        self.model.write_bytes(b'GGUF')
        self.port = free_port()
        guard = Mock()
        guard.failure.return_value = None
        guard.metadata.return_value = {}
        self.patches = [
            patch.object(controller, 'LEASE', self.root / 'generation.lock'),
            patch.object(controller, 'BACKEND_PORT', self.port),
            patch.object(controller, 'ENGINE_DIAGNOSTIC_LOG', self.root / 'engine-stderr.log'),
            patch.object(controller, 'available_memory_bytes', return_value=64 * 1024**3),
            patch.object(controller, 'resident_bytes', return_value=None),
            patch.object(controller.memory_policy, 'MemoryGuard', return_value=guard),
            patch.object(controller, 'START_SECONDS', 10.0),
        ]
        if not self.preflight:
            self.patches.append(patch.object(controller, 'require_backend_port', return_value=None))
        for item in self.patches:
            item.start()
        self.manager = controller.Manager()
        self.manager.stop_event.set()
        self.manager.eligible = Mock(return_value=True)
        self.use()

    def tearDown(self):
        with self.manager.mu:
            self.manager.cleanup()
        for item in reversed(self.patches):
            item.stop()
        self.tmp.cleanup()

    def config(self, *stub_args):
        return {'model_alias': 'stub-local', 'ready': True, 'model_path': str(self.model),
                'server_binary': sys.executable, 'extra_args': list(stub_args)}

    def use(self, *stub_args):
        self.manager.config = Mock(return_value=self.config(*stub_args))

    def run_stub(self):
        """Patch Popen so [server_binary, ...] runs the stub script with the controller's argv."""
        original = controller.subprocess.Popen

        def popen(args, **kwargs):
            return original([args[0], str(STUB), *args[1:]], **kwargs)
        return patch.object(controller.subprocess, 'Popen', side_effect=popen)
