`make test` fails, and users report that tail sometimes prints lines in the wrong order, drops the last line, and crashes on some inputs.

Fix the bugs in src/ so that tail behaves like `tail -n N` (see README.md). Keep the command-line interface and the function signatures in src/*.h unchanged.

Build with `make`, then run `make test`.
