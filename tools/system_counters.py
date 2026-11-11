#!/usr/bin/env python3
"""Snapshot/diff read-only batch counters. No model, lease or system writes."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from snapdragon_ai_lab.system_counters import snapshot, difference  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    snap = sub.add_parser('snapshot')
    snap.add_argument('--cgroup', action='append', default=[])
    snap.add_argument('--out', type=Path, required=True)
    diff = sub.add_parser('diff')
    diff.add_argument('before', type=Path); diff.add_argument('after', type=Path)
    diff.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    data = (snapshot(cgroups=args.cgroup) if args.action == 'snapshot' else
            difference(json.loads(args.before.read_text()), json.loads(args.after.read_text())))
    args.out.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(args.out, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'w') as stream:
        json.dump(data, stream, indent=2, sort_keys=True); stream.write('\n')


if __name__ == '__main__':
    main()
