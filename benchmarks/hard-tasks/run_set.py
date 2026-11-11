#!/usr/bin/env python3
"""Run the hard-task set: every task x every alias, one fresh fixture copy per run.

Appends one JSON line per run to <root>/results.jsonl (the per-run summary.json),
skips (task, alias, rep) rows already completed (resume after a window ends), and
sends one tiny warm-up chat to each alias first so model load is not billed to
the first task. Runs are sequential; the controller serialises model use.

Resume rule: a (task, alias, rep) is completed only by a row with harness_ok
true (a model failure with harness_ok true counts as completed). Rows with
harness_ok false are infrastructure failures: they stay in results.jsonl
(append-only, with their per-run directory) and are simply re-run on resume.

    python3 run_set.py --root /abs/results/hard-tasks-YYYYMMDD \
        --alias ornith-35b-a3b-local=900 --alias qwen3.6-35b-a3b-local=900 \
        --alias qwen3.8-flash-next-local=2400 [--task ID ...] [--repeat 1]
"""
import argparse
import json
from pathlib import Path
import sys
import time
import urllib.request

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_hard_task  # noqa: E402
import validate  # noqa: E402


def warm(controller_url, alias, timeout=900):
    body = json.dumps(dict(model=alias, messages=[{'role': 'user', 'content': 'Reply with OK.'}], max_tokens=4,
                           temperature=0, stream=False, chat_template_kwargs={'enable_thinking': False})).encode()
    request = urllib.request.Request(controller_url.rstrip('/') + '/v1/chat/completions', data=body,
                                     headers={'Content-Type': 'application/json'})
    began = time.monotonic()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        response.read(1024 * 1024)
    return time.monotonic() - began


def done_rows(results):
    """(task, alias, rep) keys completed by a harness_ok row; infrastructure failures are not done."""
    rows = set()
    if results.is_file():
        for line in results.read_text().splitlines():
            try:
                row = json.loads(line)
                if row.get('harness_ok') is True:
                    rows.add((row['task'], row['model_alias'], row.get('rep', 0)))
            except (ValueError, KeyError, AttributeError):
                pass
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--alias', action='append', required=True, help='ALIAS=CAP_SECONDS, in run order')
    parser.add_argument('--task', action='append')
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--controller', default='http://127.0.0.1:8092')
    parser.add_argument('--no-warm', action='store_true')
    args = parser.parse_args()
    if not args.root.is_absolute():
        raise SystemExit('absolute --root required')
    args.root.mkdir(parents=True, exist_ok=True)
    results = args.root / 'results.jsonl'
    tasks = args.task or validate.all_tasks()
    plan = []
    for spec in args.alias:
        alias, _, cap = spec.partition('=')
        plan.append((alias, int(cap or 900)))
    done = done_rows(results)
    for alias, cap in plan:
        pending = [(t, r) for r in range(args.repeat) for t in tasks if (t, alias, r) not in done]
        if not pending:
            continue
        if not args.no_warm:
            try:
                print(json.dumps(dict(event='warm', alias=alias, seconds=warm(args.controller, alias))), flush=True)
            except OSError as error:
                print(json.dumps(dict(event='warm_failed', alias=alias, error=str(error))), flush=True)
        for task, rep in pending:
            output = args.root / alias / f'{task}-r{rep}-{time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())}'
            output.parent.mkdir(parents=True, exist_ok=True)
            try:
                run_hard_task.run(task, alias, output, controller_url=args.controller, cap_seconds=cap)
                row = json.loads((output / 'summary.json').read_text())
            except Exception as error:  # a broken run is recorded, never silently skipped
                row = dict(task=task, model_alias=alias, solved=False, harness_ok=False,
                           error=type(error).__name__ + ':' + str(error))
            row.update(rep=rep, output=str(output))
            with results.open('a') as stream:
                stream.write(json.dumps(row) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
