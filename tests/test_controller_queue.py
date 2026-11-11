"""FIFO admission of overlapping inference requests in the controller (no models, no backend)."""
import threading
import time
import unittest
from unittest.mock import patch

from snapdragon_ai_lab.runtime import controller


def fake_manager(fail_aliases=()):
    """A Manager whose begin() only does the bookkeeping the real one does on success."""
    m = controller.Manager()
    m.stop_event.set()

    def begin(*, inference=False, allow_start=False, model_alias=None):
        with m.mu:
            if inference and m.inferences:
                raise controller.Unavailable('A language-model request is already running; retry when it finishes.')
            if model_alias in fail_aliases:
                raise controller.Unavailable('synthetic load failure')
            time.sleep(0.02)  # a load or a health check
            m.active += 1
            if inference:
                m.inferences += 1
    m.begin = begin
    return m


class ControllerQueueTests(unittest.TestCase):
    def test_overlapping_requests_wait_in_fifo_order(self):
        m = fake_manager()
        m.begin_queued(inference=True)
        order, threads = [], []
        for i in range(3):
            def run(i=i):
                m.begin_queued(inference=True, wait_seconds=10)
                order.append(i)
                time.sleep(0.05)
                m.end(inference=True)
            t = threading.Thread(target=run)
            t.start()
            threads.append(t)
            while m.queue_length() < i + 1:  # enqueue strictly in order
                time.sleep(0.005)
        time.sleep(0.1)
        self.assertEqual(order, [], 'nobody runs while the first request is active')
        self.assertEqual(m.queue_length(), 3)
        m.end(inference=True)
        for t in threads:
            t.join(5)
        self.assertEqual(order, [0, 1, 2])
        self.assertEqual((m.inferences, m.active, m.queue_length()), (0, 0, 0))

    def test_never_two_inferences_at_once(self):
        m = fake_manager()
        peak, lock, running = [0], threading.Lock(), [0]

        def run():
            m.begin_queued(inference=True, wait_seconds=10)
            with lock:
                running[0] += 1
                peak[0] = max(peak[0], running[0])
            time.sleep(0.01)
            with lock:
                running[0] -= 1
            m.end(inference=True)
        ts = [threading.Thread(target=run) for _ in range(8)]
        for t in ts:
            t.start()
        for t in ts:
            t.join(10)
        self.assertEqual(peak[0], 1)
        self.assertEqual((m.inferences, m.queue_length()), (0, 0))

    def test_wait_is_bounded(self):
        m = fake_manager()
        m.begin_queued(inference=True)
        t0 = time.monotonic()
        with self.assertRaisesRegex(controller.Unavailable, 'exceeded'):
            m.begin_queued(inference=True, wait_seconds=0.2)
        self.assertLess(time.monotonic() - t0, 2)
        self.assertEqual(m.queue_length(), 0)
        m.end(inference=True)
        m.begin_queued(inference=True, wait_seconds=1)  # the slot is usable again
        m.end(inference=True)

    def test_default_wait_comes_from_the_environment_constant(self):
        m = fake_manager()
        m.begin_queued(inference=True)
        with patch.object(controller, 'QUEUE_WAIT_SECONDS', 0.1):
            with self.assertRaises(controller.Unavailable):
                m.begin_queued(inference=True)
        m.end(inference=True)

    def test_failed_begin_releases_the_turn(self):
        m = fake_manager(fail_aliases={'broken'})
        with self.assertRaisesRegex(controller.Unavailable, 'synthetic'):
            m.begin_queued(inference=True, model_alias='broken', wait_seconds=1)
        m.begin_queued(inference=True, wait_seconds=1)
        m.end(inference=True)
        self.assertEqual((m.inferences, m.active, m.queue_reserved), (0, 0, False))

    def test_non_inference_requests_are_not_queued(self):
        m = fake_manager()
        m.begin_queued(inference=True)
        m.begin_queued(inference=False)  # helper / status paths pass straight through
        self.assertEqual(m.active, 2)
        m.end()
        m.end(inference=True)


if __name__ == '__main__':
    unittest.main()
