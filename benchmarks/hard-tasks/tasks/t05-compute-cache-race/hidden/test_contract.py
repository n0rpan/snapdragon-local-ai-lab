import os
import sys
import threading
import time
import unittest

sys.path.insert(0, os.environ["PROJECT"])
from memo.cache import ComputeCache  # noqa: E402

JOIN = 5


def start(target, *args):
    t = threading.Thread(target=target, args=args, daemon=True)
    t.start()
    return t


def finish(test, threads):
    for t in threads:
        t.join(timeout=JOIN)
    test.assertFalse(any(t.is_alive() for t in threads), "hang or deadlock")


class SameKeyTest(unittest.TestCase):
    def test_sixteen_threads_single_compute(self):
        cache = ComputeCache()
        n = 16
        barrier = threading.Barrier(n)
        arrived = threading.Event()
        count = [0]
        lock = threading.Lock()
        release = threading.Event()
        calls = []
        results = [None] * n

        def fn():
            with lock:
                calls.append(1)
            release.wait(JOIN)
            return object()

        def worker(i):
            barrier.wait(JOIN)
            with lock:
                count[0] += 1
                if count[0] == n:
                    arrived.set()
            results[i] = cache.get_or_compute('k', fn)

        threads = [start(worker, i) for i in range(n)]
        self.assertTrue(arrived.wait(JOIN))
        time.sleep(0.2)  # fallback only: let the workers reach their call
        release.set()
        finish(self, threads)
        self.assertEqual(len(calls), 1)
        self.assertIsNotNone(results[0])
        self.assertTrue(all(r is results[0] for r in results))
        self.assertEqual(cache.stats(), dict(hits=n - 1, misses=1, computes=1))


class DifferentKeyTest(unittest.TestCase):
    def _pair(self, first, second):
        cache = ComputeCache()
        other = {first: second, second: first}
        entered = {first: threading.Event(), second: threading.Event()}
        saw_other = {}
        out = {}

        def fn_for(me):
            def fn():
                entered[me].set()
                saw_other[me] = entered[other[me]].wait(3)
                return me
            return fn

        def run(key):
            out[key] = cache.get_or_compute(key, fn_for(key))

        t1 = start(run, first)
        self.assertTrue(entered[first].wait(JOIN))
        t2 = start(run, second)
        finish(self, [t1, t2])
        self.assertTrue(saw_other[first], 'computations for different keys were serialized')
        self.assertEqual(out, {first: first, second: second})

    def test_a_then_b(self):
        self._pair('a', 'b')

    def test_b_then_a(self):
        self._pair('b', 'a')


class ReentrantTest(unittest.TestCase):
    def test_fn_calls_other_key(self):
        cache = ComputeCache()
        out = {}

        def outer():
            out['v'] = cache.get_or_compute(
                'outer', lambda: cache.get_or_compute('inner', lambda: 41) + 1)

        finish(self, [start(outer)])
        self.assertEqual(out.get('v'), 42)
        self.assertEqual(cache.stats(), dict(hits=0, misses=2, computes=2))


class ErrorTest(unittest.TestCase):
    def test_error_shared_and_not_cached(self):
        cache = ComputeCache()
        exc = ValueError('boom')
        started = threading.Event()
        release = threading.Event()
        raised = []

        def failing():
            started.set()
            release.wait(JOIN)
            raise exc

        def run(fn):
            try:
                cache.get_or_compute('k', fn)
                raised.append(None)
            except Exception as e:  # noqa: BLE001
                raised.append(e)

        owner = start(run, failing)
        self.assertTrue(started.wait(JOIN))
        waiters = [start(run, lambda: 'late') for _ in range(4)]
        time.sleep(0.2)  # fallback only: let the waiters block on the computation
        release.set()
        finish(self, [owner] + waiters)
        self.assertEqual(len(raised), 5)
        self.assertTrue(all(e is exc for e in raised), raised)
        self.assertEqual(cache.get_or_compute('k', lambda: 'ok'), 'ok')
        self.assertEqual(cache.stats(), dict(hits=0, misses=2, computes=2))


class InvalidateTest(unittest.TestCase):
    def test_invalidate_inflight_result_not_stored(self):
        cache = ComputeCache()
        started = threading.Event()
        release = threading.Event()
        out = {}

        def slow():
            started.set()
            release.wait(JOIN)
            return 'v1'

        def owner():
            out['owner'] = cache.get_or_compute('k', slow)

        def waiter():
            out['waiter'] = cache.get_or_compute('k', lambda: 'never')

        t1 = start(owner)
        self.assertTrue(started.wait(JOIN))
        t2 = start(waiter)
        time.sleep(0.2)  # fallback only: let the waiter block on the computation
        cache.invalidate('k')
        release.set()
        finish(self, [t1, t2])
        self.assertEqual(out['owner'], 'v1')
        self.assertEqual(out['waiter'], 'v1')
        self.assertEqual(cache.get_or_compute('k', lambda: 'v2'), 'v2')
        self.assertEqual(cache.get_or_compute('k', lambda: 'v3'), 'v2')
        self.assertEqual(cache.stats(), dict(hits=2, misses=2, computes=2))

    def test_invalidate_cached_value_recomputes(self):
        cache = ComputeCache()
        self.assertEqual(cache.get_or_compute('k', lambda: 1), 1)
        cache.invalidate('k')
        self.assertEqual(cache.get_or_compute('k', lambda: 2), 2)
        self.assertEqual(cache.stats(), dict(hits=0, misses=2, computes=2))


class StatsTest(unittest.TestCase):
    def test_sequential_counts_exact(self):
        cache = ComputeCache()
        for i in range(1000):
            k = i % 10
            self.assertEqual(cache.get_or_compute(k, lambda k=k: ('v', k)), ('v', k))
        self.assertEqual(cache.stats(), dict(hits=990, misses=10, computes=10))

    def test_mixed_workload_20_threads_4_keys(self):
        cache = ComputeCache()
        n = 20
        barrier = threading.Barrier(n)
        results = [None] * n

        def fn_for(key):
            def fn():
                time.sleep(0.02)
                return ('val', key)
            return fn

        def worker(i):
            barrier.wait(JOIN)
            key = i % 4
            results[i] = cache.get_or_compute(key, fn_for(key))

        finish(self, [start(worker, i) for i in range(n)])
        self.assertEqual(cache.stats(), dict(hits=16, misses=4, computes=4))
        for k in range(4):
            group = [results[i] for i in range(n) if i % 4 == k]
            self.assertEqual(group[0], ('val', k))
            self.assertTrue(all(r is group[0] for r in group))

    def test_threaded_1000_calls_counts_exact(self):
        cache = ComputeCache()
        n, per = 8, 125
        barrier = threading.Barrier(n)
        bad = []

        def worker(_):
            barrier.wait(JOIN)
            for j in range(per):
                k = j % 5
                v = cache.get_or_compute(k, lambda k=k: ('v', k))
                if v != ('v', k):
                    bad.append((k, v))

        finish(self, [start(worker, t) for t in range(n)])
        self.assertEqual(bad, [])
        self.assertEqual(cache.stats(), dict(hits=995, misses=5, computes=5))


if __name__ == '__main__':
    unittest.main()
