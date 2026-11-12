"""Evaluate the cache-miss classifier on existing ledger files (CACHE-PERFECT P0). Read-only.

A row's predecessor is, strongest first: the exact `previous_request_id` when the row names one;
else the last complete chat row of the same (session, model, agent, origin). Only rows that
predate the session field fall back to the HEURISTIC chain by (client, model, agent): a row
continues it when the message count did not shrink and the gap since the previous row's end is
under --gap-s; its basis is reported as `heuristic-<basis>`. Error, incomplete and non-chat rows
are never a predecessor. Output is counts per miss_class, basis and chain only; no row content
leaves this script.
"""
import argparse
import collections
import json
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(TOOLS.parent / 'src'))
from snapdragon_ai_lab.runtime.cache_miss import classify, summarize  # noqa: E402


def load_rows(paths):
    rows = []
    for p in paths:
        for line in Path(p).read_text(encoding='utf-8', errors='replace').splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    rows.sort(key=lambda r: r.get('t') or 0)
    return rows


def _end(row):
    return (row.get('t') or 0) + (row.get('wall_ms') or 0) / 1000.0


def evaluate(rows, gap_s=1200.0):
    by_id, by_session, heuristic = {}, {}, {}
    classes, bases, chains = collections.Counter(), collections.Counter(), collections.Counter()
    judged = unjudged = invalid = 0
    for row in rows:
        cur = summarize(row)
        if cur is None:
            invalid += 1  # non-chat, error, incomplete or count-less: neither judged nor a predecessor
            continue
        if cur['prev_id'] is not None:
            chain, prev = 'previous_request_id', by_id.get(cur['prev_id'])
        elif cur['session'] is not None:
            chain = 'session'
            prev = by_session.get((cur['session'], cur['model'], cur['agent'], cur['origin']))
        else:
            chain, key = 'heuristic', (row.get('client'), cur['model'], cur['agent'])
            prev, prev_end = heuristic.get(key, (None, 0))
            shrunk = prev is not None and None not in (cur['n_messages'], prev['n_messages']) and cur['n_messages'] < prev['n_messages']
            if prev is not None and ((row.get('t') or 0) - prev_end > gap_s or shrunk):
                prev = None
            heuristic[key] = (cur, _end(row))
        if cur['id'] is not None:
            by_id[cur['id']] = cur
        if cur['session'] is not None:
            by_session[(cur['session'], cur['model'], cur['agent'], cur['origin'])] = cur
        cls, _miss, basis = classify(prev, cur)
        if cls is None:
            unjudged += 1
            continue
        judged += 1
        classes[cls] += 1
        chains[chain] += 1
        bases[('heuristic-' if chain == 'heuristic' else '') + basis] += 1
    return {'rows': len(rows), 'judged': judged, 'unjudged_first_or_unlabelled': unjudged,
            'not_chat_or_incomplete': invalid, 'miss_class': dict(classes.most_common()),
            'basis': dict(bases), 'chain': dict(chains)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('ledger', nargs='+', help='requests-YYYYMMDD.jsonl files (read-only)')
    ap.add_argument('--gap-s', type=float, default=1200.0)
    args = ap.parse_args(argv)
    print(json.dumps(evaluate(load_rows(args.ledger), args.gap_s), indent=1, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
