"""Hard-task OpenCode caller inside the offline bwrap namespace (controller via Unix bridge).

Same pattern as classic/controller-normal13/inside_client.py: start the inside
loopback bridge, confirm the alias in the nonwaking /v1/models catalog, run the
pinned OpenCode once on the task prompt in /work/project, record events. The
independent checker is NOT here: it runs on the host side after this namespace
exits, against the final project state, with tests this namespace never sees.
"""
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
import inside as original

ROOT = Path('/work')


class Tracker(original.Completion):
    """Original completion tracker plus token, turn and tool accounting."""

    def __init__(self):
        super().__init__()
        self.turns = 0
        self.tokens = dict(input=0, output=0, reasoning=0, cache_read=0, cache_write=0)
        self.tool_calls = {}
        self.first_event = self.last_event = None

    def event(self, value):
        super().event(value)
        now = value.get('timestamp')
        if isinstance(now, (int, float)):
            self.first_event = now if self.first_event is None else self.first_event
            self.last_event = now
        part = value.get('part', {}) if isinstance(value.get('part'), dict) else {}
        if value.get('type') == 'step_finish':
            self.turns += 1
            tokens = part.get('tokens') or {}
            cache = tokens.get('cache') or {}
            for key, raw in (('input', tokens.get('input')), ('output', tokens.get('output')),
                             ('reasoning', tokens.get('reasoning')), ('cache_read', cache.get('read')),
                             ('cache_write', cache.get('write'))):
                if isinstance(raw, int):
                    self.tokens[key] += raw
        elif value.get('type') == 'tool_use':
            name = str(part.get('tool'))
            self.tool_calls[name] = self.tool_calls.get(name, 0) + 1


def run_agent(fixture):
    tracker = Tracker()
    started = time.monotonic()
    error = None
    logs = {}
    cap = int(fixture['cap_seconds'])
    child = subprocess.Popen(['/opencode', '--standalone', 'run', '--auto', '--format', 'json', '--archived',
                              fixture['prompt']], cwd=ROOT / 'project', stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, start_new_session=True)
    readers = []
    for key, stream in (('stdout', child.stdout), ('stderr', child.stderr)):
        logs[key] = {}
        thread = threading.Thread(target=original.capture, args=(stream, ROOT / f'agent-{key}.log',
                                  tracker if key == 'stdout' else None, logs[key]), daemon=True)
        thread.start()
        readers.append(thread)
    timed_out = False
    try:
        child.wait(timeout=cap)
    except subprocess.TimeoutExpired:
        timed_out = True
        error = 'agent_wall_cap'
    finally:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=5)
        for thread in readers:
            thread.join(5)
    result = tracker.result()
    result.pop('test_commands_completed', None)  # normal13-specific
    result.update(opencode_exit=child.returncode, timed_out=timed_out, cap_seconds=cap, error=error,
                  seconds=time.monotonic() - started, turns=tracker.turns, tokens=tracker.tokens,
                  tokens_total=tracker.tokens['input'] + tracker.tokens['output'] + tracker.tokens['cache_read'],
                  tool_calls=tracker.tool_calls, logs=logs,
                  readers_joined=not any(t.is_alive() for t in readers))
    result['final_text'] = result['final_text'][-4000:]
    return result


def main():
    fixture = json.loads((ROOT / 'fixture.json').read_text())
    report = dict(passed=False, native_server_started=False, canonical_lease_acquired=False,
                  task=fixture['task'], model_alias=fixture['model_alias'])
    bridge = reader = None
    began = time.monotonic()
    try:
        bridge = subprocess.Popen(['/usr/bin/python3', '-B', '/work/loopback_bridge.py', 'inside',
                                   '--socket', '/work/controller.sock', '--port', str(original.PORT)],
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
        reader = threading.Thread(target=original.capture, args=(bridge.stdout, ROOT / 'inside-bridge.log'), daemon=True)
        reader.start()
        until = time.monotonic() + 15
        while True:
            if bridge.poll() is not None:
                raise RuntimeError('namespace_bridge_exited')
            try:
                catalog, _ = original.api('/v1/models', timeout=2)
                aliases = {row.get('id') for row in catalog.get('data', []) if isinstance(row, dict)}
                if fixture['model_alias'] not in aliases:
                    raise RuntimeError('requested_controller_model_not_listed')
                break
            except (OSError, ValueError):
                if time.monotonic() > until:
                    raise TimeoutError('namespace_bridge_deadline')
                time.sleep(.1)
        report['bridge_ready_seconds'] = time.monotonic() - began
        report['agent'] = run_agent(fixture)
        report['passed'] = report['agent']['readers_joined'] is True
    except Exception as error:
        report['error'] = type(error).__name__ + ':' + str(error)
    finally:
        if bridge is not None and bridge.poll() is None:
            bridge.terminate()
            try:
                bridge.wait(timeout=3)
            except subprocess.TimeoutExpired:
                bridge.kill()
                bridge.wait(timeout=3)
        if reader:
            reader.join(5)
        report.update(inside_bridge_reaped=bridge is not None and bridge.poll() is not None,
                      inside_bridge_reader_joined=reader is not None and not reader.is_alive(),
                      wall_seconds=time.monotonic() - began)
        report['passed'] = report['passed'] and report['inside_bridge_reaped'] and report['inside_bridge_reader_joined']
        (ROOT / 'hard-result.json').write_text(json.dumps(report, indent=2) + '\n')
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
