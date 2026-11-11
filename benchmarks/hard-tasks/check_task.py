#!/usr/bin/env python3
"""Independent hard-task checker: judges a final repository state, never model text.

The project directory is copied to a private scratch directory, the task's hidden
checker is mounted read-only beside it, and `hidden/check.sh` runs in an offline
bwrap namespace (no network, no home, clean environment) with a wall-clock limit.
Exit 0 from check.sh means solved. The judged project directory is never modified.

    python3 check_task.py --task t01-... --project /path/to/final/project [--json]
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile
import time

HERE = Path(__file__).resolve().parent
TASKS = HERE / 'tasks'
OUTPUT_LIMIT = 64 * 1024
IGNORED = shutil.ignore_patterns('__pycache__', '*.pyc', '.git')


def task_dir(task):
    path = TASKS / task
    if '/' in task or task.startswith('.') or not (path / 'task.json').is_file():
        raise ValueError('unknown_task:' + task)
    return path


def load_task(task):
    meta = json.loads((task_dir(task) / 'task.json').read_text())
    if meta.get('id') != task:
        raise ValueError('task_json_id_mismatch:' + task)
    return meta


def sandbox_command(project, hidden):
    command = ['/usr/bin/bwrap', '--unshare-all', '--die-with-parent', '--new-session', '--cap-drop', 'ALL',
               '--clearenv', '--ro-bind', '/usr', '/usr', '--symlink', 'usr/lib', '/lib',
               '--symlink', 'usr/bin', '/bin', '--symlink', 'usr/lib', '/lib64', '--proc', '/proc', '--dev', '/dev',
               '--tmpfs', '/tmp', '--bind', str(project), '/project', '--ro-bind', str(hidden), '/hidden']
    environment = dict(PATH='/usr/bin', HOME='/tmp', LANG='C.UTF-8', LC_ALL='C.UTF-8', TMPDIR='/tmp',
                       PROJECT='/project', HIDDEN='/hidden', PYTHONDONTWRITEBYTECODE='1', PYTHONHASHSEED='0')
    for key, value in environment.items():
        command += ['--setenv', key, value]
    return command + ['--chdir', '/project', '/usr/bin/bash', '/hidden/check.sh']


def host_command(project, hidden):
    # Only for model-free validation on machines without bwrap. Never for model output.
    environment = dict(PATH='/usr/bin:/bin', HOME=str(project.parent), LANG='C.UTF-8', LC_ALL='C.UTF-8',
                       TMPDIR=str(project.parent), PROJECT=str(project), HIDDEN=str(hidden),
                       PYTHONDONTWRITEBYTECODE='1', PYTHONHASHSEED='0')
    return ['/usr/bin/bash', str(hidden / 'check.sh')], environment


def check(task, project, sandbox=True):
    meta = load_task(task)
    hidden = task_dir(task) / 'hidden'
    project = Path(project).resolve()
    if not project.is_dir():
        raise ValueError('project_dir_required')
    timeout = int(meta.get('check_timeout_s', 120))
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix='hard-check-') as scratch:
        copy = Path(scratch) / 'project'
        shutil.copytree(project, copy, symlinks=True, ignore=IGNORED)
        if sandbox:
            command, environment = sandbox_command(copy, hidden), None
        else:
            command, environment = host_command(copy, hidden)
        child = subprocess.Popen(command, cwd=copy, env=environment, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        timed_out = False
        try:
            output, _ = child.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            output, _ = child.communicate()
        text = output.decode(errors='replace')
    return dict(task=task, passed=(not timed_out and child.returncode == 0), exit=child.returncode,
                timed_out=timed_out, check_timeout_s=timeout, seconds=round(time.monotonic() - started, 3),
                sandbox=sandbox, output_tail=text[-OUTPUT_LIMIT:])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--task', required=True)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--no-sandbox', action='store_true', help='model-free validation only')
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()
    result = check(args.task, args.project, sandbox=not args.no_sandbox)
    if args.json:
        print(json.dumps(result))
    else:
        print(result['output_tail'])
        print('PASS' if result['passed'] else 'FAIL', args.task, f"{result['seconds']}s")
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
