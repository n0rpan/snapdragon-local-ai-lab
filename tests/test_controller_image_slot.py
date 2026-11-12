import sys
import fcntl
import inspect
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

from snapdragon_ai_lab.runtime import controller, image_npu, image_slot

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
from _isolation import socket_tmpdir  # noqa: E402  (N7: short AF_UNIX paths)


class ControllerImageSlots(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=socket_tmpdir())
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)/'inference.lock'
        # Start no listener in these unit tests. Exercise its reservation callback.
        with patch.dict(os.environ, {'LOCAL_AI_IMAGE_NPU': '0'}):
            self.manager = controller.Manager()
        self.manager.stop_event.set()
        self.manager.thread.join(2)
        self.patches = [patch.dict(os.environ, {'LOCAL_AI_IMAGE_NPU': '1'}),
                        patch.object(controller, 'LEASE', self.path),
                        patch.object(controller, 'available_memory_bytes', return_value=20*image_npu.GIB),
                        patch.object(controller, 'total_memory_bytes', return_value=31*image_npu.GIB),
                        patch.object(controller, 'swap_bytes', return_value=(16*image_npu.GIB, 12*image_npu.GIB)),
                        patch.object(controller, 'resident_bytes', return_value=14*image_npu.GIB)]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(self.close_manager)

    def close_manager(self):
        self.manager.process = None  # model-free mock, never a real child
        self.manager.close()

    def loaded(self):
        self.manager.process = Mock(pid=123456, poll=Mock(return_value=None))
        self.manager.lease = self.path.open('a+')
        fcntl.flock(self.manager.lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return self.manager.process, self.manager.lease

    def test_resident_process_and_lease_survive_image(self):
        process, original = self.loaded()
        with self.manager.reserve_image('klein-q8-512-npu', os.getpid()) as (fd, metadata):
            self.assertEqual(fd, original.fileno())
            self.assertTrue(metadata['chat_retained'])
            self.assertEqual(metadata['backend'], 'npu')
            self.assertIs(self.manager.process, process)
            self.assertTrue(self.manager.image_active)
            self.assertEqual((self.manager.active, self.manager.inferences), (1, 1))
            with self.assertRaises(controller.Unavailable):
                self.manager.begin(inference=False, allow_start=True)
        self.assertIs(self.manager.process, process)
        self.assertIs(self.manager.lease, original)
        self.assertFalse(original.closed)
        self.assertEqual((self.manager.active, self.manager.inferences), (0, 0))

    def test_no_model_takes_and_releases_canonical_lease(self):
        with self.manager.reserve_image('klein-q8-512-npu', os.getpid()):
            with self.path.open('a+') as probe:
                with self.assertRaises(BlockingIOError):
                    fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with self.path.open('a+') as probe:
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.assertIsNone(self.manager.process)

    def test_chat_has_priority_and_no_counter_leak(self):
        self.manager.inferences = 1
        self.manager.active = 1
        with self.assertRaises(controller.Unavailable):
            with self.manager.reserve_image('klein-q8-512-npu', os.getpid()):
                self.fail('busy chat was admitted')
        self.assertFalse(self.manager.queue_reserved)
        self.assertEqual((self.manager.active, self.manager.inferences), (1, 1))
        self.manager.inferences = self.manager.active = 0

    def test_failed_budget_selects_cpu_without_unloading_chat(self):
        process, _ = self.loaded()
        with patch.object(controller, 'resident_bytes', return_value=20*image_npu.GIB), \
             patch.object(self.manager, 'cleanup') as cleanup:
            with self.manager.reserve_image('klein-q8-512-npu', os.getpid()) as (_, metadata):
                self.assertEqual(metadata['backend'], 'cpu')
                self.assertIs(self.manager.process, process)
            cleanup.assert_not_called()

    def test_exception_releases_slot_and_preserves_loaded_model(self):
        process, _ = self.loaded()
        with self.assertRaisesRegex(RuntimeError, 'fixture'):
            with self.manager.reserve_image('klein-q8-512-npu', os.getpid()):
                raise RuntimeError('fixture')
        self.assertIs(self.manager.process, process)
        self.assertFalse(self.manager.image_active)
        self.assertFalse(self.manager.queue_reserved)
        self.assertEqual((self.manager.active, self.manager.inferences), (0, 0))

    def test_fifo_request_waits_for_image_to_finish(self):
        finished = threading.Event()
        def begin(**_):
            with self.manager.mu:
                self.assertFalse(self.manager.image_active)
                self.manager.active += 1
                self.manager.inferences += 1
        def request():
            self.manager.begin_queued(inference=True, wait_seconds=2)
            self.manager.end(inference=True)
            finished.set()
        with patch.object(self.manager, 'begin', side_effect=begin):
            with self.manager.reserve_image('klein-q8-512-npu', os.getpid()):
                thread = threading.Thread(target=request)
                thread.start()
                deadline = time.monotonic()+2
                while self.manager.queue_length() != 1 and time.monotonic() < deadline:
                    time.sleep(.005)
                self.assertEqual(self.manager.queue_length(), 1)
                self.assertFalse(finished.is_set())
            thread.join(3)
        self.assertTrue(finished.is_set())
        self.assertEqual((self.manager.active, self.manager.inferences), (0, 0))


class ImageWatchdogOptIn(unittest.TestCase):
    """LOCAL_AI_IMAGE_WATCHDOG_KILL_CLIENT: only the exact value '1' enables client kill."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=socket_tmpdir())
        self.addCleanup(self.tmp.cleanup)

    def start_manager(self, watchdog):
        key = 'LOCAL_AI_IMAGE_WATCHDOG_KILL_CLIENT'
        env = {'LOCAL_AI_IMAGE_NPU': '1', 'LOCAL_AI_IMAGE_SOCKET': str(Path(self.tmp.name)/'permit.sock')}
        if watchdog is not None:
            env[key] = watchdog
        broker_cls = Mock()
        broker_cls.return_value.start.return_value = broker_cls.return_value
        with patch.dict(os.environ, env):
            if watchdog is None:
                os.environ.pop(key, None)  # patch.dict restores the environment on exit
            with patch.object(controller.image_slot, 'Broker', broker_cls), \
                    patch.object(controller, 'LEASE', Path(self.tmp.name)/'inference.lock'):
                manager = controller.Manager()
        manager.stop_event.set()  # as in ControllerImageSlots: no reaper, no real state
        manager.thread.join(2)
        self.addCleanup(lambda: (setattr(manager, 'process', None), manager.close()))
        return broker_cls, manager

    def kill_client_passed(self, watchdog):
        broker_cls, _ = self.start_manager(watchdog)
        broker_cls.assert_called_once()
        return broker_cls.call_args.kwargs['kill_client']

    def test_exact_one_enables_kill_client(self):
        self.assertIs(self.kill_client_passed('1'), True)

    def test_other_values_stay_off(self):
        for value in ('0', '', 'true', 'yes', 'on', ' 1', '1 ', '01', '2'):
            with self.subTest(value=value):
                self.assertIs(self.kill_client_passed(value), False)

    def test_unset_stays_off(self):
        self.assertIs(self.kill_client_passed(None), False)

    def test_broker_default_is_off(self):
        self.assertIs(inspect.signature(image_slot.Broker.__init__).parameters['kill_client'].default, False)




if __name__ == '__main__':
    unittest.main()
