#!/usr/bin/env bash
# Hidden checker: runs scripts/build_report.sh on the fixture data, then on hidden data.
set -u
cd "$PROJECT" || exit 1

run() {  # $1 = label, $2 = expected file
  rm -rf out
  if ! timeout 60 bash scripts/build_report.sh; then
    echo "FAIL $1: build_report.sh exited non-zero"
    return 1
  fi
  if ! cmp -s "$2" out/summary.csv; then
    echo "FAIL $1: out/summary.csv differs from $(basename "$2")"
    diff "$2" out/summary.csv | head -60
    return 1
  fi
  echo "PASS $1"
  return 0
}

fail=0
run fixture "$HIDDEN/expected_fixture_summary.csv" || fail=1

rm -rf data/exports
mkdir -p data/exports
cp "$HIDDEN"/data/*.csv data/exports/
run hidden "$HIDDEN/expected_summary.csv" || fail=1

exit $fail
