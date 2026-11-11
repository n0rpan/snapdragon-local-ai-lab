Implement `plan()` in `minibuild/planner.py`. It currently raises NotImplementedError.

The full specification is the docstring of `plan()`: traversal order, the staleness rule, the output order, and the exact errors (CycleError with its `.cycle` path, MissingSourceError with its `.name`) and when each is raised. Follow it exactly. Do not change the signature, the error classes in `minibuild/errors.py`, or the other modules.

Run the visible tests with:

    python3 -m unittest discover -s tests -t .

Hidden tests check every rule in the docstring, including edge cases and large inputs, so do not special-case the visible tests.
