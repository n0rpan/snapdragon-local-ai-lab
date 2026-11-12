"""S6: classic readiness on the backend port must come from the engine this controller started."""
import socket
import subprocess
import sys
import unittest
from unittest.mock import patch

from snapdragon_ai_lab.runtime import controller

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
from _stub_manager import STUB, StubManagerCase  # noqa: E402


class PortOwnerTests(unittest.TestCase):
    def test_listener_in_our_group_is_ours(self):
        with socket.socket() as listener:
            listener.bind(('127.0.0.1', 0)); listener.listen()
            port = listener.getsockname()[1]
            self.assertIs(controller.port_listener_in_group(port, __import__('os').getpgid(0)), True)

    def test_listener_in_another_group_is_not_ours(self):
        code = ('import socket,sys,time; s=socket.socket(); s.bind(("127.0.0.1",0)); s.listen(); '
                'print(s.getsockname()[1], flush=True); time.sleep(30)')
        child = subprocess.Popen([sys.executable, '-c', code], stdout=subprocess.PIPE, text=True,
                                 start_new_session=True)
        try:
            port = int(child.stdout.readline())
            self.assertIs(controller.port_listener_in_group(port, __import__('os').getpgid(0)), False)
            self.assertIs(controller.port_listener_in_group(port, child.pid), True)
        finally:
            child.kill(); child.wait()

    def test_no_visible_listener_makes_no_claim(self):
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', 0))
            port = probe.getsockname()[1]
        self.assertIsNone(controller.port_listener_in_group(port, 1))


class ClassicReadinessOwnershipTests(StubManagerCase):
    """The readiness loop itself: a 200 on the backend port from a process that is not our child
    (it grabbed the port between the preflight probe and our engine's bind) must not be ready."""
    preflight = False  # the race: the probe saw a free port, a foreign server bound it afterwards

    def foreign_server(self):
        child = subprocess.Popen([sys.executable, str(STUB), '--port', str(self.port)],
                                 start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.addCleanup(lambda: (child.kill(), child.wait()))
        for _ in range(200):
            with socket.socket() as probe:
                if probe.connect_ex(('127.0.0.1', self.port)) == 0:
                    return child
            __import__('time').sleep(.02)
        self.fail('foreign stub never listened')

    def test_foreign_listener_is_not_marked_ready(self):
        self.foreign_server()
        self.use('--no-bind')  # our engine never binds; only the foreign server answers /health
        with self.run_stub(), self.assertRaisesRegex(controller.Unavailable, 'not the engine this controller started'):
            self.manager.begin(allow_start=True, model_alias='stub-local')
        self.assertNotEqual(self.manager.state, 'ready')
        self.assertIsNone(self.manager.process)
        self.assertIn('not the engine', self.manager.status()['last_error'])

    def test_our_own_listener_is_ready(self):
        self.use()
        with self.run_stub():
            self.manager.begin(allow_start=True, model_alias='stub-local')
        self.manager.end(touch=True)
        self.assertEqual(self.manager.state, 'ready')


if __name__ == '__main__':
    unittest.main()
