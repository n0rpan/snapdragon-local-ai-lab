#!/usr/bin/env python3
"""Five-minute local controller smoke with explicit, immutable synthetic references.

Run `capture` on the qualified candidate before installation, then `verify` on
the installed controller. Neither mode installs anything or blesses a changed
reference. Caller owns the quiet window; controller owns inference and guards.
"""
import argparse
from datetime import datetime
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import time
from urllib.parse import urlsplit


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def identity(response):
    """Hash generated IDs where exposed; otherwise the assembled chat message.

    Tool IDs are transport-generated. Compare names and parsed arguments, never
    random call IDs or SSE chunk boundaries. No n_probs: exercise the real fast
    path and MTP configuration rather than disabling them to obtain logprobs.
    """
    tokens = response.get('tokens')
    if isinstance(tokens, list) and tokens and all(type(x) is int for x in tokens):
        return {'generated_ids_sha256': digest(tokens), 'generated_n': len(tokens),
                'text_sha256': digest(response.get('content'))}
    message = response['choices'][0]['message']
    canonical = {'content': message.get('content'), 'tool_calls': []}
    for call in message.get('tool_calls', []):
        function = call['function']
        canonical['tool_calls'].append({'name': function['name'], 'arguments': json.loads(function['arguments'])})
    return {'assembled_message_sha256': digest(canonical),
            'generated_n': response.get('usage', {}).get('completion_tokens')}


def safe_timings(response):
    return {k: v for k, v in response.get('timings', {}).items()
            if k in {'cache_n', 'prompt_n', 'prompt_ms', 'predicted_n', 'predicted_ms', 'draft_n', 'draft_n_accepted'}
            and type(v) in (int, float) and math.isfinite(v)}


def save_new(path, value):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'w') as f:
        json.dump(value, f, indent=2, allow_nan=False); f.write('\n')


def host_oom_kills():
    return int(next(line.split()[1] for line in Path('/proc/vmstat').read_text().splitlines()
                    if line.startswith('oom_kill ')))


class Suite:
    def __init__(self, url, alias, engine_sha, deadline, *, session_resume=False, session_api="completion"):
        parsed = urlsplit(url)
        if (parsed.scheme != 'http' or parsed.hostname != '127.0.0.1' or parsed.path not in ('', '/')
                or parsed.username or parsed.password or parsed.query or parsed.fragment):
            raise ValueError('literal_loopback_controller_url_required')
        self.port = parsed.port or 80
        self.alias, self.engine_sha = alias, engine_sha
        self.deadline = deadline
        self.session_resume = session_resume
        if session_api not in ("chat", "completion"): raise ValueError("unknown_session_api")
        self.session_api = session_api
        self.rows, self.guards = {}, []

    def remaining(self):
        n = self.deadline - time.monotonic()
        if n <= 0:
            raise TimeoutError('five_minute_suite_budget_exhausted')
        return n

    def api(self, path, body=None, timeout=90):
        client = http.client.HTTPConnection('127.0.0.1', self.port, timeout=min(timeout, self.remaining()))
        try:
            client.request('GET' if body is None else 'POST', path,
                           None if body is None else json.dumps(body),
                           {'Content-Type': 'application/json', 'X-Local-AI-Origin': 'lab'})
            response = client.getresponse()
            raw = response.read(2**20 + 1)
            if response.status != 200 or len(raw) > 2**20:
                raise RuntimeError('controller_http_' + str(response.status))
            return json.loads(raw)
        finally:
            client.close()

    def guard(self):
        # The response bytes can arrive before the handler releases its slot.
        # Wait briefly for that one response, and fail on queued work.
        idle_deadline = time.monotonic() + 0.25
        while True:
            status = self.api('/_local/status', timeout=2)
            active = status.get('active_requests', 0)
            if status.get('queued_requests', 0) or active > 1:
                raise RuntimeError('controller_not_exclusively_idle_between_smokes')
            if not active:
                break
            if time.monotonic() >= idle_deadline:
                raise RuntimeError('controller_not_exclusively_idle_between_smokes')
            time.sleep(0.025)
        if status.get('state') != 'ready' or status.get('model') != self.alias:
            raise RuntimeError('wrong_loaded_model')
        if status.get('validation', {}).get('server_sha256') != self.engine_sha:
            raise RuntimeError('wrong_engine_sha256')
        g = status.get('memory_policy')
        if (not isinstance(g, dict) or g.get('reason') is not None
                or type(g.get('swap_current_bytes')) is not int or g['swap_current_bytes'] != 0
                or any(g.get('memory_events', {}).get(k) != 0 for k in ('oom', 'oom_kill'))):
            raise RuntimeError('missing_or_failed_memory_guard')
        self.guards.append(g)

    def completion(self, prompt, count=8, timeout=90):
        return self.api('/completion', {'model': self.alias, 'prompt': prompt, 'n_predict': count,
                                       'return_tokens': True, 'temperature': 0, 'seed': 42}, timeout=timeout)

    def chat(self, messages, **extra):
        return self.api('/v1/chat/completions', {'model': self.alias, 'messages': messages,
                         'temperature': 0, 'seed': 42, 'max_tokens': 32,
                         'chat_template_kwargs': {'enable_thinking': False}, **extra})

    def add(self, name, response, **extra):
        timings = safe_timings(response)
        if not {'prompt_ms', 'predicted_ms'} <= timings.keys():
            raise RuntimeError('missing_native_timings')
        self.rows[name] = {'identity': identity(response), 'native_timings': timings, **extra}
        self.guard()

    def chat_session_resume(self):
        """Chat-only release persistence: a completed turn, follow-up, unload/reload.

        Raw completion has no post-reply persistence contract in v0.1.2. Keep the
        legacy raw fixture separate, and require the full prior chat C3 boundary.
        """
        messages = [{'role': 'system', 'content': 'Synthetic session fixture. ' +
                     'alpha beta gamma delta. ' * 400},
                    {'role': 'user', 'content': 'Reply with just OK.'}]
        first = self.chat(messages, max_tokens=16)
        usage = first.get('usage', {})
        if type(usage.get('prompt_tokens')) is not int or usage['prompt_tokens'] < 1024:
            raise RuntimeError('session_fixture_below_disk_store_minimum')
        if type(usage.get('completion_tokens')) is not int or usage['completion_tokens'] < 1:
            raise RuntimeError('session_fixture_missing_completion_count')
        required = usage['prompt_tokens'] + usage['completion_tokens'] - 1
        follow = messages + [first['choices'][0]['message'], {'role': 'user', 'content': 'Reply with just READY.'}]
        uninterrupted = self.chat(follow, max_tokens=16)
        self.guard()
        self.api('/_local/unload', {}, timeout=30)
        resumed = self.chat(follow, max_tokens=16)
        if identity(resumed) != identity(uninterrupted):
            raise RuntimeError('session_resume_output_drift')
        cached = safe_timings(resumed).get('cache_n', 0)
        if cached < required:
            raise RuntimeError('session_resume_did_not_reuse_C3_prefix')
        if resumed.get('timings', {}).get('resume_src') not in ('disk', 'ram'):
            raise RuntimeError('session_resume_not_restored_after_reload')
        self.add('session_resume', resumed, cached_tokens=cached, C3_required_tokens=required, session_api='chat')

    def run(self):
        initial = self.api('/_local/status', timeout=2)
        if initial.get('state') == 'loading' or initial.get('active_requests', 0) or initial.get('queued_requests', 0):
            raise RuntimeError('controller_busy_before_smoke')
        self.add('chat', self.chat([{'role': 'user', 'content': 'Reply with just OK.'}]))
        tool = {'type': 'function', 'function': {'name': 'report_result', 'description': 'Report the requested value.',
                'parameters': {'type': 'object', 'properties': {'value': {'type': 'string'}}, 'required': ['value']}}}
        messages = [{'role': 'user', 'content': 'Call report_result with value ready. Use the tool.'}]
        first = self.chat(messages, tools=[tool], max_tokens=128)
        calls = first['choices'][0]['message'].get('tool_calls', [])
        if len(calls) != 1 or calls[0]['function']['name'] != 'report_result':
            raise RuntimeError('tool_call_missing_or_wrong_name')
        if json.loads(calls[0]['function']['arguments']) != {'value': 'ready'}:
            raise RuntimeError('tool_call_invalid_arguments')
        self.add('tool_call', first)
        messages += [first['choices'][0]['message'], {'role': 'tool', 'tool_call_id': calls[0]['id'], 'content': 'Recorded ready.'}]
        followup = self.chat(messages, tools=[tool])
        message = followup['choices'][0]['message']
        if message.get('tool_calls') or not isinstance(message.get('content'), str) or not message['content'].strip():
            raise RuntimeError('tool_roundtrip_missing_final_reply')
        self.add('tool_followup', followup)
        # A representative 4K context smoke, not the full-window 027–029 tests.
        self.add('context_4k', self.completion([1000 + (i * 31) % 5000 for i in range(4096)]))
        payload = json.dumps({'model': self.alias, 'prompt': [12000 + i % 1000 for i in range(8192)],
                              'n_predict': 2048, 'temperature': 0, 'stream': False}).encode()
        client = socket.create_connection(('127.0.0.1', self.port), timeout=min(3, self.remaining()))
        try:
            client.settimeout(min(3, self.remaining()))
            client.sendall(f'POST /completion HTTP/1.1\r\nHost: 127.0.0.1:{self.port}\r\nX-Local-AI-Origin: lab\r\nContent-Type: application/json\r\nContent-Length: '.encode()
                           + str(len(payload)).encode() + b'\r\n\r\n' + payload)
            time.sleep(min(1, self.remaining()))
        finally:
            client.close()
        began = time.monotonic()
        # FIFO may still contain the just-cancelled request; do not poll guard
        # until the short follow-up has proven that cancellation freed it.
        response = self.completion('Reply OK.', 4, timeout=5)
        elapsed = time.monotonic() - began
        if elapsed >= 5:
            raise RuntimeError('cancel_recovery_at_least_5seconds')
        self.add('cancel', response, recovery_s=elapsed)
        if self.session_resume and self.session_api == 'chat':
            self.chat_session_resume()
        elif self.session_resume:
            prompt = [1000 + (i * 13) % 5000 for i in range(2304)]
            first = self.completion(prompt)
            if not first.get('tokens'):
                raise RuntimeError('session_fixture_needs_generated_token_ids')
            follow = prompt + first['tokens'] + [1001, 1002, 1003]
            uninterrupted = self.completion(follow)
            repeat = self.completion(prompt)
            if identity(repeat) != identity(first):
                raise RuntimeError('session_fixture_not_deterministic')
            self.chat([{'role': 'user', 'content': 'Reply with just OK.'}])
            self.guard()
            self.api('/_local/unload', {}, timeout=30)
            resumed = self.completion(follow)
            if identity(resumed) != identity(uninterrupted):
                raise RuntimeError('session_resume_output_drift')
            cached = safe_timings(resumed).get('cache_n', 0)
            if cached < 512:
                raise RuntimeError('session_resume_did_not_reuse_saved_prefix')
            self.add('session_resume', resumed, cached_tokens=cached)
        return self.rows


def compare(reference, candidate, ratio=1.25):
    failures = []
    if set(reference) != set(candidate):
        return ['case_set_changed']
    for name, row in candidate.items():
        if row['identity'] != reference[name]['identity']:
            failures.append(name + ':output_identity_changed')
        for field in ('prompt_ms', 'predicted_ms'):
            if field == 'prompt_ms' and any(row['native_timings'].get(k) != reference[name]['native_timings'].get(k)
                                            for k in ('prompt_n', 'cache_n')):
                # Cache coverage is reported; unequal prompt work is not a
                # matched prefill speed comparison.
                continue
            base = reference[name]['native_timings'].get(field)
            new = row['native_timings'].get(field)
            if base is not None and new is not None and new > max(base * ratio, base + 50):
                failures.append(name + ':' + field + '_regressed')
    return failures


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('capture', 'verify'))
    p.add_argument('--url', required=True)
    p.add_argument('--alias', required=True)
    p.add_argument('--engine-sha256', required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--qualified-receipt', type=Path, help='capture requires an independently passed candidate receipt')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--session-resume', action='store_true', help='required for a release shipping disk sessions')
    p.add_argument('--session-api', choices=('completion', 'chat'), default='completion', help='keep raw and chat persistence references separate')
    p.add_argument('--timeout-s', type=float, default=300)
    args = p.parse_args()
    if not re.fullmatch('[a-f0-9]{64}', args.engine_sha256) or not 0 < args.timeout_s <= 300:
        p.error('pinned engine SHA and a <=300s timeout required')
    if args.out.exists() or (args.mode == 'capture' and args.reference.exists()):
        p.error('output/reference must be fresh; refusing to overwrite evidence')
    if args.mode == 'capture':
        proof = json.loads(args.qualified_receipt.read_text()) if args.qualified_receipt else {}
        if proof.get('passed') is not True or proof.get('engine_sha256') != args.engine_sha256:
            p.error('capture requires passed independent qualification bound to this engine_sha256')
        reference = None
    else:
        reference = json.loads(args.reference.read_text())
        if (reference.get('alias') != args.alias or reference.get('engine_sha256') != args.engine_sha256
                or reference.get('session_resume') is not args.session_resume or reference.get('session_api', 'completion') != args.session_api or reference.get('passed') is not True):
            p.error('reference does not match this alias/engine/features')
    began = time.monotonic()
    oom_before = host_oom_kills()
    report = {'passed': False, 'verdict': 'FAIL', 'alias': args.alias, 'engine_sha256': args.engine_sha256,
              'session_resume': args.session_resume, 'session_api': args.session_api, 'started': datetime.now().astimezone().isoformat(), 'guards': []}
    suite = Suite(args.url, args.alias, args.engine_sha256, began + args.timeout_s, session_resume=args.session_resume, session_api=args.session_api)
    def deadline(_signal, _frame):
        raise TimeoutError('five_minute_suite_budget_exhausted')
    previous_alarm = signal.signal(signal.SIGALRM, deadline)
    signal.setitimer(signal.ITIMER_REAL, args.timeout_s)
    try:
        report['cases'] = suite.run()
        report['failures'] = compare(reference['cases'], report['cases']) if reference else []
        if host_oom_kills() != oom_before:
            report['failures'].append('host_oom_kill')
        report['passed'] = not report['failures']
        report['verdict'] = 'PASS' if report['passed'] else 'FAIL'
    except TimeoutError as e:
        report.update(verdict='BLOCKED', error=str(e))
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, IndexError, http.client.HTTPException) as e:
        report['error'] = type(e).__name__ + ':' + str(e)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_alarm)
        report.update(wall_s=time.monotonic() - began, guards=suite.guards, cases=suite.rows,
                      host_oom_kills=host_oom_kills() - oom_before)
        if report['host_oom_kills']:
            report.update(passed=False, verdict='FAIL')
            if 'host_oom_kill' not in report.setdefault('failures', []):
                report['failures'].append('host_oom_kill')
        args.out.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        save_new(args.out, report)
    if args.mode == 'capture' and report['passed']:
        args.reference.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        save_new(args.reference, report)
    print(json.dumps({'verdict': report['verdict'], 'wall_s': report['wall_s'], 'out': str(args.out)}))
    return 0 if report['passed'] else 2 if report['verdict'] == 'BLOCKED' else 1


if __name__ == '__main__':
    raise SystemExit(main())
