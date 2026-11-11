#!/usr/bin/env python3
"""Model-free checker validation (WP-M3 gate).

For every task: the checker must FAIL on the untouched fixture and PASS on the
fixture with the reference solution overlaid (reference/<id>/, same relative
paths; an optional reference/<id>/.delete lists project paths to remove).
Also reports fixture size (the model sees it inside a 12K-token input window).

    python3 validate.py [--task ID ...] [--no-sandbox] [--out validation.json]
"""
import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import check_task  # noqa: E402

REFERENCE = HERE / 'reference'


def all_tasks():
    return sorted(p.name for p in check_task.TASKS.iterdir() if (p / 'task.json').is_file())


def overlay(project, task):
    ref = REFERENCE / task
    if not ref.is_dir():
        raise ValueError('reference_missing:' + task)
    delete = ref / '.delete'
    if delete.is_file():
        for line in delete.read_text().split('\n'):
            line = line.strip()
            if line:
                target = project / line
                if target.is_dir():
                    shutil.rmtree(target)
                elif target.exists():
                    target.unlink()
    for source in ref.rglob('*'):
        if source.is_file() and source.name != '.delete' and '__pycache__' not in source.parts:
            target = project / source.relative_to(ref)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)


def fixture_bytes(project):
    return sum(p.stat().st_size for p in project.rglob('*') if p.is_file() and '__pycache__' not in p.parts)


def validate(task, sandbox=True):
    base = check_task.task_dir(task)
    meta = check_task.load_task(task)
    project = base / 'project'
    row = dict(task=task, category=meta.get('category'), fixture_bytes=fixture_bytes(project),
               prompt_bytes=len((base / 'TASK.md').read_bytes()))
    with tempfile.TemporaryDirectory(prefix='hard-validate-') as scratch:
        broken = Path(scratch) / 'untouched'
        shutil.copytree(project, broken)
        untouched = check_task.check(task, broken, sandbox=sandbox)
        solved = Path(scratch) / 'reference'
        shutil.copytree(project, solved)
        overlay(solved, task)
        reference = check_task.check(task, solved, sandbox=sandbox)
    row.update(untouched_failed=not untouched['passed'], reference_passed=reference['passed'],
               untouched_seconds=untouched['seconds'], reference_seconds=reference['seconds'],
               untouched_tail=untouched['output_tail'][-1500:], reference_tail=reference['output_tail'][-1500:])
    row['ok'] = row['untouched_failed'] and row['reference_passed']
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--task', action='append')
    parser.add_argument('--no-sandbox', action='store_true')
    parser.add_argument('--out', type=Path)
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    rows = [validate(t, sandbox=not args.no_sandbox) for t in (args.task or all_tasks())]
    for row in rows:
        print(f"{'OK  ' if row['ok'] else 'BAD '} {row['task']:<32} untouched={'FAIL' if row['untouched_failed'] else 'PASS'}"
              f" reference={'PASS' if row['reference_passed'] else 'FAIL'} fixture={row['fixture_bytes']}B"
              f" check={row['reference_seconds']}s")
        if args.verbose or not row['ok']:
            print('  --- untouched tail ---\n' + row['untouched_tail'][-600:])
            print('  --- reference tail ---\n' + row['reference_tail'][-600:])
    if args.out:
        args.out.write_text(json.dumps(dict(sandbox=not args.no_sandbox, tasks=rows), indent=2) + '\n')
    return 0 if rows and all(r['ok'] for r in rows) else 1


if __name__ == '__main__':
    raise SystemExit(main())
