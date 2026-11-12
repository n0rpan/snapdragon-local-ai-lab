from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import unittest

from snapdragon_ai_lab.runtime import image_slot

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
from _isolation import socket_tmpdir  # noqa: E402  (N7: short AF_UNIX paths)


class ImageSlotTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=socket_tmpdir())
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.lease_path = self.root / 'inference.lock'
        self.entered = threading.Event()
        self.released = threading.Event()
        self.reservations = 0
        self.broker = image_slot.Broker(self.root/'private/slot.sock', self.reserve).start()
        self.addCleanup(self.broker.close)

    @contextmanager
    def reserve(self, profile, pid):
        self.assertEqual(profile, 'klein-q8-512-npu')
        self.assertEqual(pid, os.getpid())
        self.reservations += 1
        try:
            with self.lease_path.open('a+') as lease:
                fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.entered.set()
                yield lease.fileno(), {'backend': 'npu', 'chat_retained': True}
        finally:
            hook = getattr(self, 'on_release', None)
            if hook is not None:
                hook()
            self.released.set()

    def assert_locked(self):
        with self.lease_path.open('a+') as probe:
            with self.assertRaises(BlockingIOError):
                fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_descriptor_retains_real_canonical_lock(self):
        lease = image_slot.connect(self.broker.path, 'klein-q8-512-npu')
        try:
            self.assertEqual(lease.metadata, {'backend': 'npu', 'chat_retained': True})
            self.assertFalse(os.get_inheritable(lease.fileno()))
            self.assert_locked()
        finally:
            lease.close()
        self.assertTrue(self.released.wait(2))

    def test_child_holds_reservation_after_studio_descriptor_closes(self):
        lease = image_slot.connect(self.broker.path, 'klein-q8-512-npu')
        child = subprocess.Popen([sys.executable, '-c', 'import sys; sys.stdin.buffer.read(1)'],
                                 stdin=subprocess.PIPE, pass_fds=lease.pass_fds())
        try:
            lease.close()
            self.assertFalse(self.released.wait(0.1), 'controller resumed before inherited worker exit')
            self.assert_locked()
        finally:
            child.communicate(b'x', timeout=5)
            lease.close()
        self.assertTrue(self.released.wait(2))
        with self.lease_path.open('a+') as probe:
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def make_broker(self, name, **options):
        broker = image_slot.Broker(self.root/'private'/name, self.reserve, max_hold_seconds=0.3,
                                   retry_seconds=0.2, **options).start()
        self.addCleanup(broker.close)
        return broker

    def hung_worker(self, lease):
        child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'], pass_fds=lease.pass_fds())
        self.addCleanup(lambda: child.poll() is None and (child.kill(), child.wait()))
        return child

    @staticmethod
    def kill_quietly(pid):
        try:
            os.kill(pid, 9)
        except OSError:
            pass

    def record_fds_at_release(self, pid):
        seen = {}

        def at_release():
            try:
                seen['fds'] = len(os.listdir(f'/proc/{pid}/fd'))  # 0 once killed (zombie or gone)
            except OSError:
                seen['fds'] = 0
        self.on_release = at_release
        return seen

    def test_watchdog_never_releases_while_the_client_still_holds_the_socket(self):
        # S7, Codex review 2: past the maximum the job is stale, but a client that still holds the
        # handoff socket is never released under (the client itself is not killed by default).
        broker = self.make_broker('slot2.sock')
        lease = image_slot.connect(broker.path, 'klein-q8-512-npu')
        self.assertFalse(self.released.wait(1.0), 'released while the client still holds the socket')
        self.assertIn(os.getpid(), broker.stale)
        lease.close()
        self.assertTrue(self.released.wait(5))
        self.assertEqual(broker.forced_releases, 1)

    def test_watchdog_kills_a_hung_worker_and_releases_on_eof(self):
        broker = self.make_broker('slot3.sock')
        lease = image_slot.connect(broker.path, 'klein-q8-512-npu')
        child = self.hung_worker(lease)
        seen = self.record_fds_at_release(child.pid)
        lease.close()  # Studio's own copies close; only the hung worker holds the job now
        self.assertTrue(self.released.wait(15), 'slot was never taken back from a hung worker')
        self.assertEqual(seen['fds'], 0, 'the worker still held descriptors when the slot was released')
        self.assertEqual(child.wait(5), -9)

    def test_orphaned_worker_is_found_and_stopped_before_release(self):
        # Codex proof: Studio dies, its FD-holding worker is reparented away from Studio's tree.
        broker = self.make_broker('slot5.sock')
        lease = image_slot.connect(broker.path, 'klein-q8-512-npu')
        spawner = ('import subprocess,sys; fds=[int(x) for x in sys.argv[1:]]; '
                   'w=subprocess.Popen([sys.executable,"-c","import time; time.sleep(120)"],pass_fds=fds,'
                   'start_new_session=True); print(w.pid, flush=True)')
        parent = subprocess.Popen([sys.executable, '-c', spawner, *map(str, lease.pass_fds())],
                                  pass_fds=lease.pass_fds(), stdout=subprocess.PIPE, text=True)
        orphan = int(parent.stdout.readline())
        parent.wait(5); parent.stdout.close()
        self.addCleanup(self.kill_quietly, orphan)
        lease.close()
        self.assertTrue(os.path.exists(f'/proc/{orphan}'), 'orphan must outlive its parent')
        seen = self.record_fds_at_release(orphan)
        self.assertTrue(self.released.wait(15), 'orphaned worker was never stopped')
        self.assertEqual(seen['fds'], 0, 'released while the orphaned worker still held the job')

    def test_unidentifiable_or_unkillable_holders_keep_the_slot(self):
        for name, options in (('slot6.sock', {'holders': lambda connection: None}),
                              ('slot7.sock', {'kill': lambda pid, connection: (_ for _ in ()).throw(PermissionError(1, 'no'))})):
            with self.subTest(case=name):
                self.released.clear()
                broker = self.make_broker(name, **options)
                lease = image_slot.connect(broker.path, 'klein-q8-512-npu')
                child = self.hung_worker(lease)
                lease.close()
                self.assertFalse(self.released.wait(1.5), 'released while the worker is alive')
                self.assert_locked()
                child.kill(); child.wait()
                self.assertTrue(self.released.wait(5), 'normal release once the worker is gone')

    def test_socket_holders_follow_the_socket_not_the_tree_or_the_lease(self):
        # Codex's fake /proc: a worker reparented to PID 1 still holds the job socket.
        with tempfile.TemporaryDirectory(dir=socket_tmpdir()) as tmp:
            proc = Path(tmp)
            for pid, link in ((2000, 'socket:[4242]'), (2001, str(self.lease_path)), (2002, 'socket:[1]')):
                (proc / str(pid) / 'fd').mkdir(parents=True)
                (proc / str(pid) / 'stat').write_text(f'{pid} (x) S 1 {pid} {pid} 0')
                (proc / str(pid) / 'fd' / '3').symlink_to(link)
            self.assertEqual(image_slot.socket_holders(4242, proc_root=proc), {2000})

    def test_peer_inode_comes_from_the_kernel(self):
        a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        with a, b:
            peer = image_slot.socket_peer_inode(a)
            if peer is None:
                self.skipTest('kernel has no unix_diag; the watchdog fails closed there')
            self.assertEqual(peer, os.fstat(b.fileno()).st_ino)

    def test_unqualified_profile_never_reserves(self):
        with self.assertRaises(image_slot.SlotUnavailable):
            image_slot.connect(self.broker.path, 'arbitrary-command')
        self.assertEqual(self.reservations, 0)

    def test_oversized_message_rejected_before_reservation(self):
        with socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as connection:
            connection.connect(str(self.broker.path))
            connection.sendall(b'{' * (image_slot.MAX_MESSAGE+1))
            response = json.loads(connection.recv(image_slot.MAX_MESSAGE))
            self.assertFalse(response['ok'])
        self.assertEqual(self.reservations, 0)

    def test_existing_file_is_not_deleted(self):
        path = self.root/'private/existing'
        path.write_text('keep this preimage')
        with self.assertRaises(OSError):
            image_slot.Broker(path, self.reserve).start()
        self.assertEqual(path.read_text(), 'keep this preimage')

    def test_unsafe_socket_directory_refused(self):
        parent = self.root/'unsafe'
        parent.mkdir(mode=0o755)
        with self.assertRaises(image_slot.SlotUnavailable):
            image_slot.Broker(parent/'slot', self.reserve).start()


if __name__ == '__main__':
    unittest.main()
