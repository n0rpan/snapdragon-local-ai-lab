#!/usr/bin/env python3
"""v0.1.3 hardware batch manifest (V013-PLAN sec. 4): validate and dry-run only in lane E.

Default action is a dry run: it validates benchmarks/v013-batch.json and prints the ordered plan with
each arm's private state root resolved under --state-home. It touches no hardware and writes nothing.

--execute needs --lease PATH: a JSON file {"batch": "<id>", "expires_unix": <int>} issued by the
orchestrator for one batch. Without a valid, unexpired lease for that batch the tool refuses (exit 2).
Even with a valid lease lane E does not run arms: the Codex runner does (exit 4 says so).
Exit codes: 0 ok, 1 invalid manifest, 2 refused, 4 execute requested but the runner is not in this lane.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / 'benchmarks' / 'v013-batch.json'
SCHEMA = 'v013-batch/1'
REQUIRED_ARM = ('id', 'gate', 'pass', 'kill')


def load_manifest(path):
    with open(path, encoding='utf-8') as stream:
        return json.load(stream)


def validate(m):
    """List of error strings; empty when the manifest is well formed."""
    errs = []
    if m.get('schema') != SCHEMA:
        errs.append(f'schema must be {SCHEMA}')
    real1 = m.get('real1') or {}
    for key in ('origin_header', 'lab_rows', 'organic_rows', 'print_policy'):
        if not real1.get(key):
            errs.append(f'real1.{key} missing')
    template = (m.get('state_roots') or {}).get('template', '')
    if '{batch}' not in template or '{arm}' not in template or '$STATE' not in template:
        errs.append('state_roots.template must hold $STATE, {batch} and {arm}')
    batches = m.get('batches')
    if not isinstance(batches, list) or not batches:
        return errs + ['batches must be a non-empty list']
    seen_batches, seen_arms = set(), set()
    for b in batches:
        bid = b.get('id')
        if not bid or bid in seen_batches:
            errs.append(f'batch id missing or duplicated: {bid!r}')
        seen_batches.add(bid)
        arms = b.get('arms') or []
        ids = [a.get('id') for a in arms]
        if sorted(ids) != sorted(set(ids)):
            errs.append(f'batch {bid}: duplicate arm ids')
        if list(b.get('order') or []) != ids:
            errs.append(f'batch {bid}: order must list exactly the arm ids in order')
        if not isinstance(b.get('hours_est'), (int, float)):
            errs.append(f'batch {bid}: hours_est must be a number')
        for a in arms:
            for key in REQUIRED_ARM:
                if not isinstance(a.get(key), str) or not a.get(key).strip():
                    errs.append(f'arm {a.get("id")!r}: {key} missing')
            if a.get('id') in seen_arms:
                errs.append(f'arm id {a.get("id")!r} appears in more than one batch')
            seen_arms.add(a.get('id'))
    return errs


def state_root(m, batch, arm, state_home):
    template = m['state_roots']['template']
    return template.replace('$STATE', str(state_home)).replace('{batch}', batch).replace('{arm}', arm)


def plan(m, state_home, only=None):
    lines = [f'{m["schema"]}: {len(m["batches"])} batches; dry run, nothing executed']
    for b in m['batches']:
        if only and b['id'] != only:
            continue
        lines.append(f'batch {b["id"]} (~{b["hours_est"]} h, {len(b["arms"])} arms):')
        by_id = {a['id']: a for a in b['arms']}
        for n, arm_id in enumerate(b['order'], 1):
            arm = by_id[arm_id]
            lines.append(f'  {n:2d}. {arm_id}: {arm["gate"]}')
            lines.append(f'      pass: {arm["pass"]}')
            lines.append(f'      kill: {arm["kill"]}')
            lines.append(f'      state root: {state_root(m, b["id"], arm_id, state_home)}')
    return '\n'.join(lines)


def check_lease(path, batch, now):
    """None when the lease is valid for `batch` and unexpired, else the reason."""
    try:
        with open(path, encoding='utf-8') as stream:
            lease = json.load(stream)
    except (OSError, ValueError):
        return 'lease unreadable or not JSON'
    if not isinstance(lease, dict):
        return 'lease must be a JSON object'
    if lease.get('batch') != batch:
        return f'lease is for batch {lease.get("batch")!r}, not {batch!r}'
    expires = lease.get('expires_unix')
    if type(expires) is not int or expires <= now:
        return 'lease missing expires_unix or expired'
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--manifest', default=str(DEFAULT_MANIFEST))
    ap.add_argument('--state-home', default=os.environ.get('XDG_STATE_HOME', str(Path.home() / '.local/state')),
                    help='where $STATE resolves in the plan (strings only; nothing is created)')
    ap.add_argument('--batch', help='show one batch only')
    ap.add_argument('--execute', action='store_true', help='run the batch (needs --lease; runner is Codex)')
    ap.add_argument('--lease', help='JSON lease for the batch: {"batch": id, "expires_unix": int}')
    ap.add_argument('--now', type=int, default=None, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    try:
        m = load_manifest(args.manifest)
    except (OSError, ValueError) as exc:
        print(f'manifest unreadable: {exc.__class__.__name__}', file=sys.stderr)
        return 1
    errs = validate(m)
    if errs:
        for e in errs:
            print('invalid manifest: ' + e, file=sys.stderr)
        return 1
    if args.batch and args.batch not in {b['id'] for b in m['batches']}:
        print(f'unknown batch {args.batch!r}', file=sys.stderr)
        return 2

    if not args.execute:
        print(plan(m, args.state_home, args.batch))
        return 0
    if not args.batch:
        print('refusing: --execute needs --batch', file=sys.stderr)
        return 2
    if not args.lease:
        print('refusing: --execute needs --lease', file=sys.stderr)
        return 2
    reason = check_lease(args.lease, args.batch, args.now if args.now is not None else int(time.time()))
    if reason:
        print(f'refusing: {reason}', file=sys.stderr)
        return 2
    print(f'lease ok for batch {args.batch}; lane E does not run arms (the Codex runner does)', file=sys.stderr)
    return 4


if __name__ == '__main__':
    sys.exit(main())
