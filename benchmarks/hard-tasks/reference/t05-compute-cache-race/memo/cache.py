"""Compute-on-miss cache shared between threads. Contract: see README.md."""
import threading


class _Flight:
    """One in-flight computation for a key."""

    def __init__(self):
        self.done = threading.Event()
        self.value = None
        self.error = None
        self.stale = False


class ComputeCache:
    """Memoizes fn() results per key, computing each key at most once at a time.

    get_or_compute(key, fn): return the cached value for key. On a miss the first
    caller runs fn() without holding the lock; concurrent callers for the same key
    wait for that computation and get the same object, or the same exception if
    fn raised (nothing is cached then).
    invalidate(key): drop the cached value. A computation in flight for key still
    delivers its result to the callers waiting on it, but the result is not stored;
    callers arriving meanwhile wait on it too; the next call after it finishes
    computes again.
    stats(): dict(hits=, misses=, computes=). The runner of fn counts a miss and a
    compute; every other call that returns a value counts a hit. Calls that raise
    count nothing. Counts are exact under concurrency.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._values = {}
        self._flights = {}
        self._hits = 0
        self._misses = 0
        self._computes = 0

    def get_or_compute(self, key, fn):
        with self._lock:
            if key in self._values:
                self._hits += 1
                return self._values[key]
            flight = self._flights.get(key)
            owner = flight is None
            if owner:
                flight = self._flights[key] = _Flight()
                self._misses += 1
                self._computes += 1
        if not owner:
            flight.done.wait()
            if flight.error is not None:
                raise flight.error
            with self._lock:
                self._hits += 1
            return flight.value
        try:
            value = fn()
        except BaseException as exc:
            with self._lock:
                flight.error = exc
                del self._flights[key]
            flight.done.set()
            raise
        with self._lock:
            flight.value = value
            del self._flights[key]
            if not flight.stale:
                self._values[key] = value
        flight.done.set()
        return value

    def invalidate(self, key):
        with self._lock:
            self._values.pop(key, None)
            flight = self._flights.get(key)
            if flight is not None:
                flight.stale = True

    def stats(self):
        with self._lock:
            return dict(hits=self._hits, misses=self._misses, computes=self._computes)
