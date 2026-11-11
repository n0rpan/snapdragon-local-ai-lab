#!/usr/bin/env bash
set -u
cd "$PROJECT" || exit 1
T="${TMPDIR:-/tmp}"
gcc -std=c11 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer src/*.c -o "$T/tail_asan" || { echo "asan build failed"; exit 1; }
gcc -std=c11 -O2 src/*.c -o "$T/tail_o2" || { echo "O2 build failed"; exit 1; }
export ASAN_OPTIONS=detect_leaks=1:abort_on_error=0
export UBSAN_OPTIONS=halt_on_error=1
python3 -I -B "$HIDDEN/run_cases.py" "$T/tail_asan" "$T/tail_o2"
