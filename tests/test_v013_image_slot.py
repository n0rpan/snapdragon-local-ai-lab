"""v0.1.3 lane H: image-slot broker stale-socket recovery (W1-569) and the bounded long-hold
warning (W1-572). Temp dirs only (short /tmp parents for AF_UNIX paths); no real state paths."""
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import time
import unittest

from snapdragon_ai_lab.runtime import image_slot

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _isolation import socket_tmpdir  # noqa: E402


def stale_socket(path, mode=0o600):
    """A socket inode with no listener, as a crashed controller leaves it."""
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    sock.bind(str(path))
    os.chmod(path, mode)
    sock.close()
    return os.lstat(path)


class BrokerCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=socket_tmpdir())
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.private = self.root / 'private'
        self.private.mkdir(mode=0o700)
        self.lease_path = self.root / 'inference.lock'
        self.released = threading.Event()
        self.logs = []

    def log(self, message):
        self.logs.append((time.monotonic(), message))

    @contextmanager
    def reserve(self, profile, pid):
        try:
            with self.lease_path.open('a+') as lease:
                fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                yield lease.fileno(), {'backend': 'npu'}
        finally:
            self.released.set()

    def broker(self, name='slot.sock', **options):
        broker = image_slot.Broker(self.private / name, self.reserve, log=self.log, **options).start()
        self.addCleanup(broker.close)
        return broker

    def assert_locked(self):
        with self.lease_path.open('a+') as probe:
            with self.assertRaises(BlockingIOError):
                fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)


class StaleSocketRecovery(BrokerCase):
    def test_stale_owner_only_socket_is_replaced_with_one_log_line(self):
        path = self.private / 'slot.sock'
        stale_socket(path)
        broker = self.broker()
        self.assertTrue(stat.S_ISSOCK(os.lstat(path).st_mode))
        self.assertEqual(len(self.logs), 1)
        self.assertIn('stale socket', self.logs[0][1])
        self.assertNotIn(str(self.root), self.logs[0][1])
        lease = image_slot.connect(broker.path, 'klein-q8-512-npu')
        lease.close()
        self.assertTrue(self.released.wait(5))

    def test_live_listener_is_refused_and_untouched(self):
        path = self.private / 'slot.sock'
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        self.addCleanup(listener.close)
        listener.bind(str(path))
        os.chmod(path, 0o600)
        listener.listen(4)
        inode = os.lstat(path).st_ino
        with self.assertRaises(OSError) as caught:
            image_slot.Broker(path, self.reserve, log=self.log).start()
        self.assertIn('listening', str(caught.exception))
        self.assertEqual(os.lstat(path).st_ino, inode)
        with socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as client:
            client.connect(str(path))  # the live server still works
        self.assertEqual(self.logs, [])

    def test_wider_mode_is_refused(self):
        path = self.private / 'slot.sock'
        for mode in (0o660, 0o604, 0o700 | 0o001):
            with self.subTest(mode=oct(mode)):
                info = stale_socket(path, mode)
                with self.assertRaises(OSError) as caught:
                    image_slot.Broker(path, self.reserve, log=self.log).start()
                self.assertIn('owner-only', str(caught.exception))
                self.assertEqual(os.lstat(path).st_ino, info.st_ino)
                path.unlink()

    def test_regular_file_and_symlink_are_refused(self):
        path = self.private / 'slot.sock'
        path.write_text('keep')
        with self.assertRaises(OSError):
            image_slot.Broker(path, self.reserve, log=self.log).start()
        self.assertEqual(path.read_text(), 'keep')
        path.unlink()
        target = self.root / 'elsewhere.sock'
        stale_socket(target)
        path.symlink_to(target)
        with self.assertRaises(OSError) as caught:
            image_slot.Broker(path, self.reserve, log=self.log).start()
        self.assertIn('not a socket', str(caught.exception))
        self.assertTrue(path.is_symlink())
        self.assertTrue(target.exists())

    def test_foreign_owner_is_refused(self):
        path = self.private / 'slot.sock'
        stale_socket(path)
        with self.assertRaises(OSError) as caught:
            image_slot.recover_stale_socket(path, uid=os.getuid() + 1, log=self.log)
        self.assertIn('another user', str(caught.exception))
        self.assertTrue(path.exists())

    def test_unknown_liveness_is_refused(self):
        path = self.private / 'slot.sock'
        stale_socket(path)
        with self.assertRaises(OSError):
            image_slot.recover_stale_socket(path, probe=lambda p: 'unknown', log=self.log)
        self.assertTrue(path.exists())

    def test_server_started_during_the_check_is_kept(self):
        # A new broker binds the path between the first probe and the unlink (tmpfs may even hand
        # it the old inode number): the re-check before the unlink must see it and refuse.
        path = self.private / 'slot.sock'
        stale_socket(path)
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        self.addCleanup(listener.close)
        calls = []

        def swap(p):
            calls.append(p)
            if len(calls) == 1:
                os.unlink(p)
                listener.bind(str(p))
                os.chmod(p, 0o600)
                listener.listen(4)
                return 'dead'
            return image_slot._probe_listener(p)
        with self.assertRaises(OSError) as caught:
            image_slot.recover_stale_socket(path, probe=swap, log=self.log)
        self.assertRegex(str(caught.exception), 'changed|listening')
        with socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as client:
            client.connect(str(path))  # the new server's socket is still there
        self.assertEqual(self.logs, [])

    def test_unchanged_dead_socket_is_probed_twice_before_the_unlink(self):
        path = self.private / 'slot.sock'
        stale_socket(path)
        calls = []
        self.assertTrue(image_slot.recover_stale_socket(path, probe=lambda p: calls.append(p) or 'dead', log=self.log))
        self.assertEqual(len(calls), 2)
        self.assertFalse(os.path.lexists(path))

    def test_absent_path_needs_no_recovery(self):
        self.assertFalse(image_slot.recover_stale_socket(self.private / 'none.sock', log=self.log))
        self.assertEqual(self.logs, [])


class LongHoldWarning(BrokerCase):
    def test_bounded_warnings_never_release_while_a_worker_holds_the_socket(self):
        broker = self.broker(warn_seconds=0.2, max_warnings=3, max_hold_seconds=600)
        lease = image_slot.connect(broker.path, 'klein-q8-512-npu')
        child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'], pass_fds=lease.pass_fds())
        self.addCleanup(lambda: child.poll() is None and (child.kill(), child.wait()))
        started = time.monotonic()
        lease.close()  # Studio's copies are gone; only the worker duplicate holds the permit
        self.assertFalse(self.released.wait(2.5), 'a warning released the permit')
        self.assert_locked()
        warnings = [(t, m) for t, m in self.logs if 'held the image permit' in m]
        self.assertEqual(len(warnings), 3, self.logs)
        self.assertTrue(all('warning' in m and 'stays reserved' in m for _, m in warnings))
        self.assertGreaterEqual(warnings[-1][0] - started, 0.6)          # 0.2, 0.4, 0.8 s: doubling
        self.assertEqual(broker.hold_warnings, 3)
        self.assertEqual(broker.stale, {})
        self.assertEqual(broker.forced_releases, 0)
        self.assertIsNone(child.poll(), 'the warning stopped the worker')
        child.kill()
        child.wait()
        self.assertTrue(self.released.wait(5), 'permit not released once every copy closed')
        self.assertEqual(broker.forced_releases, 0)

    def test_short_jobs_get_no_warning(self):
        broker = self.broker(warn_seconds=5, max_warnings=3)
        lease = image_slot.connect(broker.path, 'klein-q8-512-npu')
        lease.close()
        self.assertTrue(self.released.wait(5))
        self.assertEqual(self.logs, [])
        self.assertEqual(broker.hold_warnings, 0)

    def test_zero_disables_the_warning(self):
        broker = self.broker(warn_seconds=0, max_warnings=3)
        lease = image_slot.connect(broker.path, 'klein-q8-512-npu')
        time.sleep(0.5)
        lease.close()
        self.assertTrue(self.released.wait(5))
        self.assertEqual(broker.hold_warnings, 0)


if __name__ == '__main__':
    unittest.main()
