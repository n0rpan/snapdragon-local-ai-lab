#!/usr/bin/env bash
set -u
cd "$PROJECT"
python3 -B -m unittest discover -s "$HIDDEN" -p 'test_*.py' -t "$HIDDEN" -v 2>&1
