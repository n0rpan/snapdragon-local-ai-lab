"""Compute-on-miss cache shared between threads. Contract: see README.md."""


class ComputeCache:
    """Memoizes fn() results per key.

    get_or_compute(key, fn) returns the cached value for key, or calls fn()
    and caches its result. invalidate(key) drops a cached value. stats()
    returns dict(hits=, misses=, computes=). The full concurrency contract
    (single flight per key, errors, stats, invalidate during a computation)
    is in README.md and must be honoured by every method.
    """

    def __init__(self):
        self._values = {}
        self._hits = 0
        self._misses = 0
        self._computes = 0

    def get_or_compute(self, key, fn):
        """Return the value for key, computing it with fn() on a miss."""
        if key in self._values:
            self._hits += 1
            return self._values[key]
        self._misses += 1
        self._computes += 1
        value = fn()
        self._values[key] = value
        return value

    def invalidate(self, key):
        """Drop the cached value for key, if any."""
        self._values.pop(key, None)

    def stats(self):
        """Return hits, misses and computes counts."""
        return dict(hits=self._hits, misses=self._misses, computes=self._computes)
