#!/usr/bin/env python3
"""REAL-1 organic report: numbers only, over request-ledger rows with origin=organic.

Lab and replay rows (origin=lab) and rows written before REAL-1 (no origin field) are never counted.
Refuses to print any metric when fewer than MIN_SESSIONS distinct organic sessions exist; it then says
"not enough organic data" and exits 2. Session hashes, row ids, client tokens, paths and message
content are never printed. Every figure is tagged [M] (measured from ledger fields) or [E] (estimate
from a stated formula). Percentiles use sorted[min(n-1, int(n*p/100))], as in ledger-summary.

Usage: organic_report.py [--json] [LEDGER_FILE_OR_DIR ...]   (default: LOCAL_AI_LEDGER_DIR or
~/.local/state/local-ai/ledger). Read-only.
"""
import argparse
from collections import Counter
import glob
import json
import os
from pathlib import Path
import sys

SCHEMA = 'organic-report/1'
MIN_SESSIONS = 20
PCTS = (50, 90)
# context bands in tokens, upper bounds (last band is open)
BANDS = ((8 * 1024, '<8K'), (32 * 1024, '8K-32K'), (64 * 1024, '32K-64K'),
         (128 * 1024, '64K-128K'), (256 * 1024, '128K-256K'), (None, '>256K'))
MISS_CLASSES = frozenset({'append-ok', 'retry', 'history-edit', 'reasoning-drop', 'tool-rerender',
                          'compaction', 'template-drift', 'unknown'})
C3_OK = frozenset({'append-ok', 'retry'})
RESUME_SRC = frozenset({'live', 'ram', 'disk', 'cold'})
MTP_MIN_DRAFTS = 30  # below this a band's acceptance is not reported


def default_dir():
    return Path(os.environ.get('LOCAL_AI_LEDGER_DIR', str(Path.home() / '.local/state/local-ai/ledger')))


def is_num(v):
    return type(v) in (int, float) and v == v and v not in (float('inf'), float('-inf'))


def percentile(values, p):
    values = sorted(values)
    return values[min(len(values) - 1, int(len(values) * p / 100))]


def stats(values):
    if not values:
        return {'n': 0, 'p50': None, 'p90': None, 'max': None}
    return {'n': len(values), **{f'p{p}': percentile(values, p) for p in PCTS}, 'max': max(values)}


def band_of(ctx):
    for upper, label in BANDS:
        if upper is None or ctx <= upper:
            return label
    return BANDS[-1][1]


def context_of(row):
    if is_num(row.get('ctx_tokens')):
        return row['ctx_tokens']
    cached, new = row.get('cached_tokens'), row.get('new_tokens')
    return cached + new if is_num(cached) and is_num(new) else None


def load(paths):
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


def expand(args):
    paths = []
    for given in args or [str(default_dir())]:
        if os.path.isdir(given):
            paths.extend(sorted(glob.glob(os.path.join(given, 'requests-*.jsonl'))))
        else:
            paths.append(given)
    return paths


def build(rows):
    """The report dict. status is 'insufficient' (no metrics) or 'ok'."""
    organic = [r for r in rows if r.get('origin') == 'organic']
    by_session = {}
    for r in organic:
        if isinstance(r.get('session'), str):
            by_session.setdefault(r['session'], []).append(r)
    out = {'schema': SCHEMA, 'organic_rows': len(organic), 'sessions': len(by_session),
           'min_sessions': MIN_SESSIONS}
    if len(by_session) < MIN_SESSIONS:
        out['status'] = 'insufficient'
        return out
    out['status'] = 'ok'
    for r in by_session.values():
        r.sort(key=lambda x: x.get('t') if is_num(x.get('t')) else 0)

    out['turns_per_session'] = stats([len(v) for v in by_session.values()])
    out['context_tokens'] = stats([c for c in (context_of(r) for r in organic) if c is not None])
    out['new_tokens_per_turn'] = stats([r['new_tokens'] for r in organic if is_num(r.get('new_tokens'))])
    out['output_tokens_per_turn'] = stats([r['output_tokens'] for r in organic if is_num(r.get('output_tokens'))])

    cold, warm = [], []
    for rows_s in by_session.values():
        first = rows_s[0]
        if not is_num(first.get('ttfb_ms')) or not is_num(first.get('cached_tokens')):
            continue
        (cold if first['cached_tokens'] == 0 else warm).append(first['ttfb_ms'])
    out['turn1_ttfb_ms'] = {'cold': stats(cold), 'warm': stats(warm)}

    prefill = sum(r['prefill_ms'] for r in organic if is_num(r.get('prefill_ms')))
    decode = sum(r['decode_ms'] for r in organic if is_num(r.get('decode_ms')))
    out['prefill_share_of_engine_ms'] = round(prefill / (prefill + decode), 3) if prefill + decode > 0 else None

    judged = [r for r in organic if r.get('miss_class') in MISS_CLASSES]
    out['c3'] = {'judged': len(judged),
                 'violations_by_class': dict(sorted(Counter(
                     r['miss_class'] for r in judged if r['miss_class'] not in C3_OK).items()))}

    resume = Counter(r['resume_src'] if r.get('resume_src') in RESUME_SRC else 'absent' for r in organic)
    out['resume_src_mix'] = dict(sorted(resume.items()))

    # [E] cache tokens priced at the measured prefill rate of this traffic (tokens per ms over rows with both)
    tok = sum(r['new_tokens'] for r in organic if is_num(r.get('new_tokens')) and is_num(r.get('prefill_ms')))
    ms = sum(r['prefill_ms'] for r in organic if is_num(r.get('new_tokens')) and is_num(r.get('prefill_ms')))
    cached = sum(r['cached_tokens'] for r in organic if is_num(r.get('cached_tokens')))
    out['saved_s_estimate'] = round(cached / (tok / ms) / 1000, 1) if tok > 0 and ms > 0 else None

    comp = [r for r in organic if r.get('turn_kind') == 'compaction']
    hits = [r for r in comp if r.get('miss_class') in ('append-ok', 'retry')]
    out['compactions'] = {'count': len(comp), 'cache_hit': len(hits),
                          'miss_classes': dict(sorted(Counter(r.get('miss_class') for r in comp
                                                              if r.get('miss_class')).items()))}

    wall_total, band_wall, queue, wall_both = 0.0, Counter(), 0.0, 0.0
    pf_wall, dc_wall = 0.0, 0.0
    drafts, accepted = Counter(), Counter()
    for r in organic:
        ctx = context_of(r)
        wall = r.get('wall_ms')
        if ctx is None or not is_num(wall):
            continue
        wall_total += wall
        band_wall[band_of(ctx)] += wall
        if is_num(r.get('queue_wait_ms')):
            queue += r['queue_wait_ms']
            wall_both += wall
        if is_num(r.get('prefill_ms')):
            pf_wall += r['prefill_ms']
        if is_num(r.get('decode_ms')):
            dc_wall += r['decode_ms']
        if is_num(r.get('draft_n')) and r['draft_n'] > 0 and is_num(r.get('draft_n_accepted')):
            drafts[band_of(ctx)] += r['draft_n']
            accepted[band_of(ctx)] += r['draft_n_accepted']
    out['wall_share_by_context_band'] = ({lbl: round(band_wall[lbl] / wall_total, 3)
                                          for _, lbl in BANDS if band_wall[lbl]} if wall_total > 0 else {})
    out['wall_shares'] = {
        'queue_wait': round(queue / wall_both, 3) if wall_both > 0 else None,
        'prefill': round(pf_wall / wall_total, 3) if wall_total > 0 else None,
        'decode': round(dc_wall / wall_total, 3) if wall_total > 0 else None,
    }
    out['mtp_acceptance_by_band'] = {lbl: {'drafts': drafts[lbl], 'acceptance': round(accepted[lbl] / drafts[lbl], 3)}
                                     for _, lbl in BANDS if drafts[lbl] >= MTP_MIN_DRAFTS}
    return out


def _f(v):
    if v is None:
        return '-'
    return f'{v:.3f}' if isinstance(v, float) else str(v)


def render(rep):
    if rep['status'] != 'ok':
        return f"not enough organic data: {rep['sessions']} sessions, need {MIN_SESSIONS}"
    lines = [f"organic rows {rep['organic_rows']}  sessions {rep['sessions']}  [M]"]
    for key, label in (('turns_per_session', 'turns per session'), ('context_tokens', 'context tokens / request'),
                       ('new_tokens_per_turn', 'new tokens / request'),
                       ('output_tokens_per_turn', 'output tokens / request')):
        s = rep[key]
        lines.append(f"{label}: p50 {_f(s['p50'])} p90 {_f(s['p90'])} max {_f(s['max'])} (n {s['n']}) [M]")
    t1 = rep['turn1_ttfb_ms']
    for kind in ('cold', 'warm'):
        s = t1[kind]
        lines.append(f"turn-1 TTFT {kind} ms: p50 {_f(s['p50'])} p90 {_f(s['p90'])} (n {s['n']}) [M]")
    lines.append(f"prefill share of engine ms: {_f(rep['prefill_share_of_engine_ms'])} [M]")
    c3 = rep['c3']
    lines.append(f"C3 judged {c3['judged']}; violations by class: {c3['violations_by_class'] or 'none'} [M]")
    lines.append(f"resume_src mix: {rep['resume_src_mix']} [M]")
    lines.append(f"saved s (cached tokens at measured prefill rate): {_f(rep['saved_s_estimate'])} [E]")
    c = rep['compactions']
    lines.append(f"compactions {c['count']}, cache hits {c['cache_hit']}, miss classes {c['miss_classes'] or 'none'} [M]")
    lines.append(f"share of wall by context band: {rep['wall_share_by_context_band']} [M]")
    lines.append(f"wall shares (queue, prefill, decode): {rep['wall_shares']} [M]")
    lines.append(f"MTP acceptance by band (bands with >= {MTP_MIN_DRAFTS} drafts): "
                 f"{rep['mtp_acceptance_by_band'] or 'none'} [M]")
    return '\n'.join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('paths', nargs='*', help='ledger files or dirs (default: the ledger dir)')
    ap.add_argument('--json', action='store_true', help='print JSON instead of the table')
    args = ap.parse_args(argv)
    paths = expand(args.paths)
    if not paths:
        print('no ledger files', file=sys.stderr)
        return 1
    rep = build(load(paths))
    if rep['status'] != 'ok':
        print(render(rep))
        return 2
    print(json.dumps(rep, indent=2, sort_keys=True) if args.json else render(rep))
    return 0


if __name__ == '__main__':
    sys.exit(main())
