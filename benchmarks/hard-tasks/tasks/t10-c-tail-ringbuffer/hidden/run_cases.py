#!/usr/bin/env python3
"""Hidden differential check: tail binaries versus a model of `tail -n N` (stdlib only)."""
import os
import random
import subprocess
import sys

USAGE_ERR = b'tail: invalid line count'
SAN_MARKERS = (b'ERROR: AddressSanitizer', b'ERROR: LeakSanitizer', b'runtime error', b'LeakSanitizer')
DEFAULT_N = 10


def split_lines(data):
    if not data:
        return []
    lines = data.split(b'\n')
    if data.endswith(b'\n'):
        lines.pop()
    return lines


def model(data, n):
    if n <= 0:
        return b''
    return b''.join(line + b'\n' for line in split_lines(data)[-n:])


def rand_line(rng, length):
    alphabet = b'abcdefghijklmnopqrstuvwxyz0123456789 '
    return bytes(rng.choice(alphabet) for _ in range(length))


def cases():
    """Yields (name, args, stdin, expected_stdout, expected_exit)."""
    rng = random.Random(20240611)
    out = []

    def add(name, data, n=None):
        args = [] if n is None else ['-n', str(n)]
        out.append((name, args, data, model(data, DEFAULT_N if n is None else n), 0))

    def add_invalid(name, args):
        out.append((name, args, b'a\nb\n', b'', 2))

    for length in [0, 1, 15, 16, 17, 31, 32, 33, 64, 255, 256, 4096]:
        add(f'len{length}-x5-n3', b''.join(rand_line(rng, length) + b'\n' for _ in range(5)), 3)
    for length in [16, 17, 32, 33, 256, 4096]:
        add(f'len{length}-nonl-n2', b'\n'.join(rand_line(rng, length) for _ in range(3)), 2)

    add('empty-default', b'')
    add('empty-n0', b'', 0)
    add('single-nonl-default', b'x')
    add('newline-only', b'\n')
    add('blank-lines-n2', b'\n\n\n', 2)
    add('n0-nonempty', b'a\nb\n', 0)
    add('n-larger', b'a\nb\nc\n', 100)
    add('nonl-n5', b'a\nb', 5)
    add('default-25', b''.join(b'%d\n' % i for i in range(1, 26)))
    add('n1-wrap', b''.join(b'%d\n' % i for i in range(1, 10)), 1)
    big = b''.join(rand_line(rng, rng.randint(0, 40)) + b'\n' for _ in range(10000))
    add('10000-lines-n7', big, 7)

    for i in range(8):
        nlines = rng.randint(0, 60)
        lines = [rand_line(rng, rng.randint(0, 70)) for _ in range(nlines)]
        data = b'\n'.join(lines) + (b'\n' if lines and rng.random() < 0.5 else b'')
        n = rng.choice([None, rng.randint(0, 20)])
        add(f'random-{i}', data, n)

    add_invalid('invalid-negative', ['-n', '-1'])
    add_invalid('invalid-alpha', ['-n', 'abc'])
    add_invalid('invalid-empty', ['-n', ''])
    return out


def check(binary, args, data, expected, exit_code):
    problems = []
    try:
        proc = subprocess.run([binary, *args], input=data, capture_output=True, timeout=30)
    except subprocess.TimeoutExpired:
        return ['timeout']
    if any(marker in proc.stderr for marker in SAN_MARKERS):
        problems.append('sanitizer report')
    if proc.returncode != exit_code:
        problems.append(f'exit {proc.returncode}, want {exit_code}')
    if exit_code == 0 and proc.stdout != expected:
        problems.append(f'stdout differs ({len(proc.stdout)} bytes, want {len(expected)})')
    if exit_code == 2:
        if proc.stdout != b'' or proc.stderr.strip() != USAGE_ERR:
            problems.append('bad usage error output')
    return problems


def main(argv):
    if len(argv) < 2:
        print('usage: run_cases.py BINARY...', file=sys.stderr)
        return 2
    total = failed = 0
    for binary in argv[1:]:
        for name, args, data, expected, exit_code in cases():
            total += 1
            problems = check(binary, args, data, expected, exit_code)
            if problems:
                failed += 1
                print(f'FAIL {os.path.basename(binary)} {name}: {"; ".join(problems)}')
    print(f'{total - failed}/{total} cases passed')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
