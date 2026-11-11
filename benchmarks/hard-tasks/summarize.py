#!/usr/bin/env python3
"""Summarise results.jsonl into a Markdown table plus the O-1 verdict.

O-1 (the owner, 2026-11-10): Flash keeps full priority only if it solves >= 1 task
Ornith fails, or >= 2 more tasks of the set; otherwise it is escalation-only.
Runs with harness_ok == false are infrastructure failures: they stay in
results.jsonl (append-only), are listed in their own section and are re-run by
simply re-running run_set.py (it treats them as not done); never counted as
model failures. Per (task, alias, rep) only the LAST harness_ok attempt counts,
so retries are not double-counted.

    python3 summarize.py --root /abs/results/dir [--flash ALIAS] [--ornith ALIAS] [--out HARD-TASKS.md]
"""
import argparse
import json
from pathlib import Path
import re


def load(root):
    rows = []
    for line in (root / 'results.jsonl').read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def last_valid(rows):
    """The last harness_ok row per (task, alias, rep), in first-seen key order."""
    latest = {}
    for row in rows:
        if row.get('harness_ok') is True:
            latest[(row['task'], row['model_alias'], row.get('rep', 0))] = row
    return list(latest.values())


def partial(row):
    """Hidden unittest pass fraction from the checker transcript, when available."""
    path = Path(row.get('output', '')) / 'check-output.txt'
    try:
        text = path.read_text()
    except OSError:
        return None
    ran = re.findall(r'Ran (\d+) tests?', text)
    if not ran:
        return None
    total = int(ran[-1])
    bad = sum(int(n) for n in re.findall(r'(?:failures|errors)=(\d+)', text.split('Ran ' + ran[-1])[-1]))
    return f'{total - bad}/{total}'


def verdict(table, tasks, flash, ornith):
    def solved(alias, task):
        return table.get((task, alias), {}).get('solved_majority')
    if not any((t, flash) in table for t in tasks) or not any((t, ornith) in table for t in tasks):
        return 'incomplete: both Flash and Ornith rows are needed', [], 0
    missing = [t for t in tasks if (t, flash) not in table or (t, ornith) not in table]
    flash_only = [t for t in tasks if solved(flash, t) and not solved(ornith, t)]
    diff = sum(bool(solved(flash, t)) for t in tasks) - sum(bool(solved(ornith, t)) for t in tasks)
    keep = len(flash_only) >= 1 or diff >= 2
    text = ('FULL PRIORITY' if keep else 'ESCALATION-ONLY') + f' (Flash-only solves: {len(flash_only)}, Flash minus Ornith: {diff:+d})'
    if missing:
        text += f'; INCOMPLETE, missing rows for {missing}'
    return text, flash_only, diff


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--flash', default='qwen3.8-flash-next-local')
    parser.add_argument('--ornith', default='ornith-35b-a3b-local')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    rows = load(args.root)
    infra = [r for r in rows if r.get('harness_ok') is not True]
    good = last_valid(rows)
    aliases = list(dict.fromkeys(r['model_alias'] for r in rows))
    tasks = sorted({r['task'] for r in rows})
    table = {}
    for task in tasks:
        for alias in aliases:
            runs = [r for r in good if r['task'] == task and r['model_alias'] == alias]
            if not runs:
                continue
            wins = sum(bool(r.get('solved')) for r in runs)
            table[(task, alias)] = dict(
                runs=len(runs), wins=wins, solved_majority=wins * 2 > len(runs),
                timed_out=sum(bool(r.get('timed_out')) for r in runs),
                seconds=sum(r.get('task_seconds') or 0 for r in runs) / len(runs),
                tokens=sum(r.get('tokens_total') or 0 for r in runs) / len(runs),
                output_tokens=sum((r.get('tokens') or {}).get('output', 0) for r in runs) / len(runs),
                turns=sum(r.get('turns') or 0 for r in runs) / len(runs),
                partial=partial(runs[-1]))
    lines = ['# Hard-task set results (WP-M3)', '',
             f'Source: `{args.root}/results.jsonl`. Solved = independent hidden checker passed on the final repo state.',
             'Cell: solved mark, hidden tests passed, agent wall seconds, total tokens (input+output+cache read), '
             'output tokens, turns (LLM requests). T = hit the wall-clock cap.', '']
    lines.append('| Task | ' + ' | '.join(aliases) + ' |')
    lines.append('|---|' + '---|' * len(aliases))
    for task in tasks:
        cells = []
        for alias in aliases:
            cell = table.get((task, alias))
            if not cell:
                cells.append('-')
                continue
            mark = 'PASS' if cell['solved_majority'] else 'fail'
            if cell['runs'] > 1:
                mark += f" {cell['wins']}/{cell['runs']}"
            extra = (' T' if cell['timed_out'] else '')
            cells.append(f"{mark}{extra} ({cell['partial'] or '-'}) {cell['seconds']:.0f}s "
                         f"{cell['tokens']:.0f}tok/{cell['output_tokens']:.0f}out {cell['turns']:.0f}t")
        lines.append(f'| {task} | ' + ' | '.join(cells) + ' |')
    totals = []
    for alias in aliases:
        cells = [table[(t, alias)] for t in tasks if (t, alias) in table]
        totals.append(f"{sum(c['solved_majority'] for c in cells)}/{len(cells)} solved, "
                      f"{sum(c['seconds'] for c in cells):.0f}s, {sum(c['tokens'] for c in cells):.0f}tok")
    lines.append('| **Total** | ' + ' | '.join(totals) + ' |')
    text, flash_only, diff = verdict(table, tasks, args.flash, args.ornith)
    lines += ['', f'**O-1 verdict:** {text}.', f'Flash-only solves: {flash_only or "none"}.']
    if infra:
        valid_keys = {(r['task'], r['model_alias'], r.get('rep', 0)) for r in good}
        lines += ['', '## Infrastructure failures (harness_ok false; not counted)', '']
        for r in infra:
            key = (r['task'], r['model_alias'], r.get('rep', 0))
            state = 'superseded by a valid retry' if key in valid_keys else 'NOT YET RE-RUN'
            lines.append(f"- {r['task']}@{r['model_alias']} r{key[2]} ({state}): "
                         f"{r.get('error') or r.get('inner_error')} [{r.get('output', '-')}]")
    report = '\n'.join(lines) + '\n'
    if args.out:
        args.out.write_text(report)
    print(report)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
