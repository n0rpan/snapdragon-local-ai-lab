# tail

A small clone of `tail -n N`.

    tail [-n N]

- Reads stdin and splits it into lines on `\n`.
- Prints the last N lines (default 10), oldest first, each followed by `\n`.
- A final line without a trailing newline is still a line; it is printed with `\n` added.
- N = 0 prints nothing. N larger than the number of lines prints all lines.
- Empty input prints nothing. Empty lines are lines.
- N must be a non-negative decimal integer. Otherwise (negative, non-numeric, empty)
  it prints `tail: invalid line count` to stderr and exits with status 2.
- Otherwise exits 0.

Layout: `src/strbuf.*` (growable byte buffer, NUL-terminated view),
`src/ring.*` (ring of the last N lines, copies), `src/tail.c` (main).
