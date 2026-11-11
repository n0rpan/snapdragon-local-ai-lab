#!/bin/sh
# Byte-for-byte comparisons of ./tail against expected `tail -n` output.
cd "$(dirname "$0")/.." || exit 1
t=$(mktemp -d) || exit 1
fail=0

# usage: run NAME EXPECTED_FORMAT INPUT_FORMAT [tail args...]
run() {
	name=$1; exp=$2; inp=$3; shift 3
	printf "$inp" | ./tail "$@" > "$t/out" 2>/dev/null
	printf "$exp" > "$t/exp"
	if cmp -s "$t/out" "$t/exp"; then echo "ok: $name"; else echo "FAIL: $name"; fail=1; fi
}

run "last 3 of 5" 'c\nd\ne\n' 'a\nb\nc\nd\ne\n' -n 3
run "last line without newline" 'b\nc\n' 'a\nb\nc' -n 2
run "N larger than line count" 'a\nb\n' 'a\nb\n' -n 5
run "N zero" '' 'a\nb\n' -n 0
run "empty lines" '\n\n' '\n\n\n' -n 2
run "empty input" '' ''
run "default count is 10" '3\n4\n5\n6\n7\n8\n9\n10\n11\n12\n' '1\n2\n3\n4\n5\n6\n7\n8\n9\n10\n11\n12\n'

./tail -n -1 < /dev/null > "$t/out" 2> "$t/err"
if [ $? -eq 2 ] && [ ! -s "$t/out" ] && [ "$(cat "$t/err")" = "tail: invalid line count" ]; then
	echo "ok: invalid count"
else
	echo "FAIL: invalid count"; fail=1
fi

rm -rf "$t"
exit $fail
