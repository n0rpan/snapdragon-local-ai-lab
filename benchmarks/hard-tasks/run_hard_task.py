#!/usr/bin/env python3
"""Run ONE hard task through real OpenCode against an already-running controller.

Reuses the protected normal13 controller-client harness (benchmarks/classic):
the same pinned OpenCode binary and SHA check, the same offline bwrap layout
(--unshare-all, clearenv, private HOME/XDG, OpenCode config with the local
provider only), the same owner-only Unix-socket loopback bridge and inside
caller (inside.py). Only the fixture, prompt and wall-clock cap differ, and the
independent checker runs afterwards on the host side in its own offline bwrap
against the final project state (check_task.py).

Like controller_normal13.py this driver does NOT start a native server and does
NOT take the inference lease: the controller owns model load, lease and memory.

    python3 run_hard_task.py --task t03-fair-allocation --alias ornith-35b-a3b-local \
        --controller http://127.0.0.1:8092 --output /abs/fresh/dir --cap-seconds 900
"""
import argparse
import copy
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time

HERE = Path(__file__).resolve().parent
CLASSIC = HERE.parent / 'classic'
sys.path.insert(0, str(CLASSIC))
sys.path.insert(0, str(HERE))
import normal13 as common  # noqa: E402
import controller_normal13 as controller  # noqa: E402
import check_task  # noqa: E402

USER_OPENCODE = Path.home() / '.config/opencode/opencode.json'
DEFAULT_LIMIT = dict(context=16384, input=12288, output=2048)


def tree_sha(path):
    digest = hashlib.sha256()
    for item in sorted(p for p in Path(path).rglob('*') if p.is_file() and '__pycache__' not in p.parts):
        digest.update(str(item.relative_to(path)).encode() + b'\0' + common.sha(item).encode() + b'\n')
    return digest.hexdigest()


def alias_limit(alias, config_path=USER_OPENCODE):
    """Context/input/output limits of the alias in the user's OpenCode config (read-only)."""
    try:
        config = json.loads(Path(config_path).read_text())
        providers = config.get('providers') or config.get('provider') or {}
        for provider in providers.values():
            limit = provider.get('models', {}).get(alias, {}).get('limit')
            if isinstance(limit, dict) and all(isinstance(limit.get(k), int) for k in DEFAULT_LIMIT):
                return {k: limit[k] for k in DEFAULT_LIMIT}, str(config_path)
    except (OSError, ValueError, AttributeError):
        pass
    return dict(DEFAULT_LIMIT), 'default'


def prepare(output, task, alias, cap_seconds, limit):
    """Fresh private copy of the fixture plus the unchanged normal13 inside files."""
    base = check_task.task_dir(task)
    for name in ('home', 'config', 'data', 'cache', 'state'):
        (output / name).mkdir(mode=0o700)
    shutil.copytree(base / 'project', output / 'project', ignore=check_task.IGNORED)
    for name in ('inside.py', 'oracle.py', 'bounded_log.py'):
        (output / name).write_bytes((common.ASSETS / name).read_bytes())
    (output / 'loopback_bridge.py').write_bytes((CLASSIC / 'loopback_bridge.py').read_bytes())
    (output / 'hard_inside.py').write_bytes((HERE / 'hard_inside.py').read_bytes())
    fixture = dict(task=task, model_alias=alias, cap_seconds=int(cap_seconds),
                   prompt=(base / 'TASK.md').read_text().strip(), protected={})
    common.write_json(output / 'fixture.json', fixture)
    # OpenCode config: same transformation as normal13.prepare, alias-named model.
    config = json.loads((common.ASSETS / 'opencode-base.synthetic.json').read_text())
    provider = config['providers']['local-qwen']
    provider['settings'].update(baseURL='http://127.0.0.1:18096/v1', apiKey='local-only',
                                headerTimeout=1800000, chunkTimeout=1800000)
    metadata = copy.deepcopy(provider['models']['qwen3.6-35b-a3b-local'])
    metadata.update(name='Private hard task ' + task, limit=limit)
    provider['models'] = {alias: metadata}
    provider.setdefault('body', {}).update(temperature=0, cache_prompt=True,
                                           chat_template_kwargs={'enable_thinking': False})
    config.update(model='local-qwen/' + alias, providers={'local-qwen': provider}, update='disable', warming=False)
    config.setdefault('agents', {})['title'] = {'disabled': True}
    config['policies'] = [{'action': 'provider.use', 'resource': '*', 'effect': 'deny'},
                          {'action': 'provider.use', 'resource': 'local-qwen', 'effect': 'allow'}]
    common.write_json(output / 'opencode.json', config)
    return fixture


def sandbox_command(output, fixture, opencode, socket_path):
    # The controller client's exact namespace (engine/model binds already stripped),
    # with hard_inside.py as the entry point. No protected file binds: the checker is not inside.
    stub = dict(server_binary='/nonexistent/engine/server', model_path='/nonexistent/model.gguf')
    command = controller.sandbox_command(output, stub, fixture, opencode, socket_path)
    if command[-1] != '/work/inside_client.py':
        raise ValueError('unsupported_controller_sandbox_command')
    command[-1] = '/work/hard_inside.py'
    if any(part.startswith('/nonexistent') for part in command):
        raise ValueError('stub_bind_leaked')
    return command


def run(task, alias, output, *, controller_url, cap_seconds, opencode=None, check=True):
    meta = check_task.load_task(task)
    port = controller.controller_port(controller_url)
    if not output.is_absolute() or '..' in output.parts or any(p.is_symlink() for p in (output, *output.parents)):
        raise ValueError('fresh_absolute_output_required')
    if not isinstance(alias, str) or not alias or any(c in alias for c in '\r\n\0/'):
        raise ValueError('explicit_controller_model_alias_required')
    pinned = json.loads((common.ASSETS / 'opencode-binary.expected.json').read_text())
    opencode = common.opencode_path(opencode, pinned)
    opencode_identity = common.identity(opencode)
    if common.sha(opencode) != pinned['sha256']:
        raise ValueError('qualified_opencode_changed')
    limit, limit_source = alias_limit(alias)
    output.mkdir(mode=0o700, exist_ok=False)
    fixture = prepare(output, task, alias, cap_seconds, limit)
    report = dict(kind='hard-task-opencode', task=task, category=meta.get('category'), model_alias=alias,
                  solved=False, harness_ok=False, cap_seconds=int(cap_seconds), opencode_limit=limit,
                  opencode_limit_source=limit_source, opencode_sha256=pinned['sha256'],
                  task_tree_sha256=tree_sha(check_task.task_dir(task)),
                  native_server_started=False, canonical_lease_acquired=False,
                  started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    child = reader = host = host_reader = None
    socket_directory = None
    began = time.monotonic()

    def interrupted(number, _):
        raise RuntimeError('owned_client_signal_' + str(number))
    previous = {n: signal.signal(n, interrupted) for n in (signal.SIGTERM, signal.SIGINT)}
    try:
        if ctypes.CDLL(None).prctl(36, 1, 0, 0, 0):
            raise ValueError('subreaper_required')
        socket_directory = tempfile.TemporaryDirectory(prefix='local-ai-hard-')
        socket_path = Path(socket_directory.name) / 'controller.sock'
        host = subprocess.Popen([sys.executable, '-B', str(CLASSIC / 'loopback_bridge.py'), 'host', '--socket',
                                 str(socket_path), '--target-host', '127.0.0.1', '--target-port', str(port)],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
        host_reader = threading.Thread(target=common.capture, args=(host.stdout, output / 'host-bridge.log'), daemon=True)
        host_reader.start()
        until = time.monotonic() + 5
        while not socket_path.exists():
            if host.poll() is not None:
                raise RuntimeError('host_bridge_exited')
            if time.monotonic() > until:
                raise TimeoutError('host_bridge_deadline')
            time.sleep(.05)
        before_socket = controller.socket_identity(socket_path)
        child = subprocess.Popen(sandbox_command(output, fixture, opencode, socket_path), stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        reader = threading.Thread(target=common.capture, args=(child.stdout, output / 'sandbox.log'), daemon=True)
        reader.start()
        child.wait(timeout=int(cap_seconds) + 120)
        path = output / 'hard-result.json'
        if path.stat().st_size > 8 * 1024 ** 2:
            raise ValueError('bounded_inner_result_required')
        inner = json.loads(path.read_text())
        agent = inner.get('agent', {})
        report.update(inner_error=inner.get('error'), agent_error=agent.get('error'),
                      timed_out=agent.get('timed_out'), task_seconds=agent.get('seconds'),
                      turns=agent.get('turns'), tokens=agent.get('tokens'), tokens_total=agent.get('tokens_total'),
                      tool_calls=agent.get('tool_calls'), completed_tools=agent.get('completed_tools'),
                      opencode_exit=agent.get('opencode_exit'), complete=agent.get('complete'),
                      terminal_reason=agent.get('terminal_reason'), error_events=agent.get('error_events'),
                      malformed_events=agent.get('malformed_events'), final_text=agent.get('final_text'),
                      bridge_ready_seconds=inner.get('bridge_ready_seconds'))
        report['socket_identity_unchanged'] = controller.socket_identity(socket_path) == before_socket
        report['harness_ok'] = inner.get('passed') is True and 'error' not in inner
    except Exception as error:
        report['error'] = type(error).__name__ + ':' + str(error)
    finally:
        controller.stop(child)
        controller.stop(host)
        for worker in (reader, host_reader):
            if worker:
                worker.join(5)
        report['client_namespace_reaped'] = child is not None and child.poll() is not None
        report['all_client_children_reaped'] = controller.drain()
        if socket_directory:
            socket_directory.cleanup()
        for number, handler in previous.items():
            signal.signal(number, handler)
        report['opencode_identity_unchanged'] = common.identity(opencode) == opencode_identity
        report['harness_ok'] = report['harness_ok'] and all(report.get(k) is True for k in (
            'client_namespace_reaped', 'all_client_children_reaped', 'opencode_identity_unchanged',
            'socket_identity_unchanged'))
        report['agent_wall_seconds'] = time.monotonic() - began
    # Independent judgement of the final repo state (never the model's text).
    if check and (output / 'project').is_dir():
        verdict = check_task.check(task, output / 'project', sandbox=True)
        (output / 'check-output.txt').write_text(verdict.pop('output_tail'))
        report['check'] = verdict
        report['solved'] = verdict['passed'] is True and report['harness_ok']
        report['solved_within_cap'] = report['solved'] and report.get('timed_out') is False
    report['wall_seconds'] = time.monotonic() - began
    common.write_json(output / 'summary.json', report)
    print(json.dumps({k: report.get(k) for k in ('task', 'model_alias', 'solved', 'harness_ok', 'timed_out',
                                                 'task_seconds', 'turns', 'tokens_total', 'error')}), flush=True)
    return 0 if report['harness_ok'] else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--task', required=True)
    parser.add_argument('--alias', required=True)
    parser.add_argument('--controller', default='http://127.0.0.1:8092')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cap-seconds', type=int, default=900)
    parser.add_argument('--opencode', type=Path)
    args = parser.parse_args()
    return run(args.task, args.alias, args.output, controller_url=args.controller,
               cap_seconds=args.cap_seconds, opencode=args.opencode)


if __name__ == '__main__':
    raise SystemExit(main())
