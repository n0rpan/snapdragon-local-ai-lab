#!/usr/bin/env python3
"""Offline analyses over existing data (V013 D-2 and the Flash stall share). Numbers only, no hardware.

ledger   queue-wait distribution and heat carry-over from request-ledger files. Heat carry-over is read
         as decode ms per output token for requests that follow a short gap (under 60 s) versus a long
         gap (600 s or more) in the same session; a slower decode after a short gap would point at heat
         or memory carried over from the previous request. Gaps come from the ledger's gap_ms field.
receipt  Flash stall share from one receipt: time under PSI-full above 2 s in the loaded arm, divided
         by the arm's wall time. The receipt holds sample counts but not the sampling interval, so the
         interval must be given (--sample-s); without it the share is not computed.

Prints counts and percentiles only; never session hashes, prompt text, paths or file names.
Tags: [M] measured in the given files, [D] derived with a stated assumption.
Usage: offline_analyses.py ledger PATH...   |   offline_analyses.py receipt RECEIPT.json --sample-s SECONDS
"""
import argparse
import glob
import json
import os
import sys

SHORT_GAP_MS = 60_000
LONG_GAP_MS = 600_000
PCTS = (50, 90)


def is_num(v):
    return type(v) in (int, float) and v == v and v not in (float('inf'), float('-inf'))


def percentile(values, p):
    values = sorted(values)
    return values[min(len(values) - 1, int(len(values) * p / 100))]


def pct_stats(values):
    if not values:
        return {'n': 0}
    return {'n': len(values), **{f'p{p}': round(percentile(values, p), 2) for p in PCTS},
            'max': round(max(values), 2)}


def load_rows(paths):
    rows = []
    for path in paths:
        with open(path, encoding='utf-8') as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if isinstance(row, dict):
                    rows.append(row)
    return rows


def ledger_files(args):
    files = []
    for given in args:
        if os.path.isdir(given):
            files.extend(sorted(glob.glob(os.path.join(given, '**', 'requests-*.jsonl'), recursive=True)))
            files.extend(sorted(glob.glob(os.path.join(given, '**', 'native-ledger.jsonl'), recursive=True)))
        else:
            files.append(given)
    return files


def ledger_report(rows):
    queue = [r['queue_wait_ms'] for r in rows if is_num(r.get('queue_wait_ms'))]
    short, long_ = [], []
    for r in rows:
        if not (is_num(r.get('decode_ms')) and is_num(r.get('output_tokens')) and r['output_tokens'] > 0):
            continue
        gap = r.get('gap_ms')
        if not is_num(gap):
            continue
        per_token = r['decode_ms'] / r['output_tokens']
        if gap < SHORT_GAP_MS:
            short.append(per_token)
        elif gap >= LONG_GAP_MS:
            long_.append(per_token)
    return {
        'rows': len(rows),
        'queue_wait_ms [M]': pct_stats(queue),
        'decode ms per token after gap < 60 s [M]': pct_stats(short),
        'decode ms per token after gap >= 600 s [M]': pct_stats(long_),
        'heat carry-over ratio (short / long p50) [D]': (
            round(percentile(short, 50) / percentile(long_, 50), 3) if short and long_ else None),
    }


def receipt_report(receipt, sample_s):
    loaded = (((receipt.get('qualification_arm') or {}).get('desktop') or {}).get('loaded')) or {}
    out = {'decode18_over14 [M]': (receipt.get('decode') or {}).get('ratio18_over14')}
    stall = loaded.get('psi_full_above_2_seconds')
    samples = loaded.get('samples')
    if not is_num(stall) or not is_num(samples) or samples <= 0:
        out['stall share'] = 'not measurable: loaded-arm fields missing'
        return out
    out['loaded psi_full_above_2_seconds [M]'] = round(stall, 2)
    out['loaded samples [M]'] = samples
    if sample_s is None:
        out['stall share'] = 'not computed: give --sample-s (the receipt has no sampling interval)'
        return out
    wall = samples * sample_s
    out['loaded wall seconds [D: samples x sample-s]'] = round(wall, 1)
    out['stall share of loaded wall [D]'] = round(stall / wall, 4)
    return out


def render(d):
    return '\n'.join(f'{k}: {v}' for k, v in d.items())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    lg = sub.add_parser('ledger')
    lg.add_argument('paths', nargs='+')
    rc = sub.add_parser('receipt')
    rc.add_argument('receipt')
    rc.add_argument('--sample-s', type=float, default=None, help='seconds between PSI samples in the receipt run')
    args = ap.parse_args(argv)
    if args.cmd == 'ledger':
        files = ledger_files(args.paths)
        if not files:
            print('no ledger files', file=sys.stderr)
            return 1
        print(render(ledger_report(load_rows(files))))
        return 0
    with open(args.receipt, encoding='utf-8') as stream:
        receipt = json.load(stream)
    if not isinstance(receipt, dict):
        print('receipt is not a JSON object', file=sys.stderr)
        return 1
    if args.sample_s is not None and not 0 < args.sample_s <= 3600:
        print('--sample-s must be in (0, 3600]', file=sys.stderr)
        return 2
    print(render(receipt_report(receipt, args.sample_s)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
