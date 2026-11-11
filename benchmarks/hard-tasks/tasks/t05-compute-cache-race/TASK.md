Bug report: ComputeCache computes the same key several times under load

Under concurrent load, ComputeCache (memo/cache.py) calls fn() several times for the
same missing key, and the stats() counters come out wrong.

Fix memo/cache.py so ComputeCache meets the contract in README.md (the same contract is
in the class docstring). Keep the public API: get_or_compute(key, fn), invalidate(key),
stats(). Do not add dependencies.

Run the visible tests from the project root with:
    python3 -m unittest discover -s tests -v

The tests in tests/ are only part of the spec. README.md is the full spec.
