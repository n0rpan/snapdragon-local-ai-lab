# memo: ComputeCache

A small thread-safe cache of computed values, keyed by string or tuple keys.

## API

- `get_or_compute(key, fn) -> value`: return the cached value for `key`. On a miss, call `fn()`
  (no arguments), cache its result and return it.
- `invalidate(key) -> None`: drop the cached value for `key`, if any.
- `stats() -> dict`: `dict(hits=int, misses=int, computes=int)`.

## Concurrency contract

1. Single flight per key: for a given key, at most one `fn` call is in flight at a time.
   Concurrent callers for that key wait for the computation and return the same result object.
2. No global serialization: `fn` runs without holding any lock shared across keys. Computations
   for different keys run concurrently. `fn` may call `get_or_compute` for a different key
   without deadlock. (Calling `get_or_compute` for the same key from inside its own `fn` is
   not supported.)
3. Errors: if `fn` raises, the exception propagates to the caller that ran `fn` and to every
   caller waiting on that computation, as the same exception object. Nothing is cached, and the
   next call computes again.
4. Stats: the caller that runs `fn` counts one miss and one compute (even if `fn` raises). Every
   other call that returns a value without running `fn` (served from the cache, or after waiting
   for a computation) counts one hit. A call that raises because the computation it waited for
   raised counts nothing. Counts are exact under concurrency.
5. Invalidate: `invalidate(key)` drops the cached value. If it is called while a computation for
   `key` is in flight, that computation's result is still returned to the callers already waiting
   on it, but it is NOT stored. Callers that arrive while that computation is still running also
   wait on it (rule 1 still applies). The first call after it finishes computes again.
