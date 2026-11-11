import os
import sys
import threading
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from memo.cache import ComputeCache  # noqa: E402


class SingleThreadTest(unittest.TestCase):
    def test_hit_and_miss(self):
        cache = ComputeCache()
        calls = []

        def fn():
            calls.append(1)
            return 'v'

        self.assertEqual(cache.get_or_compute('k', fn), 'v')
        self.assertEqual(cache.get_or_compute('k', fn), 'v')
        self.assertEqual(len(calls), 1)
        self.assertEqual(cache.stats(), dict(hits=1, misses=1, computes=1))

    def test_invalidate(self):
        cache = ComputeCache()
        cache.get_or_compute('k', lambda: 1)
        cache.invalidate('k')
        self.assertEqual(cache.get_or_compute('k', lambda: 2), 2)


class SameKeyConcurrencyTest(unittest.TestCase):
    def test_same_key_computes_once(self):
        cache = ComputeCache()
        n = 8
        barrier = threading.Barrier(n)
        count_lock = threading.Lock()
        calls = []
        all_in = threading.Event()

        def fn():
            with count_lock:
                calls.append(1)
                if len(calls) == n:
                    all_in.set()
            all_in.wait(1.0)  # buggy code: every caller gets here; correct code: only one
            return object()

        results = [None] * n

        def worker(i):
            barrier.wait()
            results[i] = cache.get_or_compute('k', fn)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        self.assertFalse(any(t.is_alive() for t in threads))
        self.assertEqual(len(calls), 1)
        self.assertIsNotNone(results[0])
        self.assertTrue(all(r is results[0] for r in results))


if __name__ == '__main__':
    unittest.main()
