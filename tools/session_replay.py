#!/usr/bin/env python3
"""Record a live coding-agent session and replay it for equal-work speed comparisons.

record   logging HTTP proxy in front of a llama-server-compatible server
replay   re-send the recorded request bodies in order with fixed sampling (equal work)
compare  compare two replay results made from the same recording (determinism gate)
info     summarise a recording (agents, sampling parameters, token totals)

Python 3.10+ standard library only. The tool connects only to URLs given on its
command line.
"""
import argparse
import copy
import gzip
import datetime as dt
import hashlib
import http.client
import json
import os
from pathlib import Path
import secrets
import signal
import statistics
import subprocess
import sys
import threading
import time
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

VERSION = 1
TOOL = 'session_replay'
RECORD_PATHS = ('/v1/chat/completions', '/chat/completions', '/completion', '/completions')
CHAT_PATHS = ('/v1/chat/completions', '/chat/completions')
HOP_BY_HOP = frozenset({'connection', 'keep-alive', 'proxy-connection', 'proxy-authenticate',
                        'proxy-authorization', 'te', 'trailer', 'trailers', 'transfer-encoding',
                        'upgrade'})
# Module-level roots so tests can point the environment sampler at a temp tree.
SYS_ROOT = Path('/sys')
PROC_ROOT = Path('/proc')
INFO_KEYS = ('temperature', 'top_p', 'top_k', 'min_p', 'max_tokens', 'max_completion_tokens',
             'n_predict', 'seed', 'stop', 'ignore_eos')


class UsageError(Exception):
    """Bad arguments, unreadable input, or a privacy refusal: reported on stderr, exit 2."""


def utc_iso():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')


def parse_url(url):
    parts = urlsplit(url)
    if parts.scheme not in ('http', 'https') or not parts.hostname:
        raise UsageError(f'not an http(s) URL: {url}')
    return parts


def make_connection(parts, timeout):
    port = parts.port or (443 if parts.scheme == 'https' else 80)
    cls = http.client.HTTPSConnection if parts.scheme == 'https' else http.client.HTTPConnection
    return cls(parts.hostname, port, timeout=timeout)


def join_path(parts, path):
    return parts.path.rstrip('/') + path


def get_json(parts, path, timeout=10.0):
    """GET a JSON document; None on any failure (the endpoint is optional metadata)."""
    conn = None
    try:
        conn = make_connection(parts, timeout)
        conn.request('GET', join_path(parts, path))
        resp = conn.getresponse()
        data = resp.read()
        if resp.status != 200:
            return None
        return json.loads(data)
    except (OSError, ValueError, http.client.HTTPException):
        return None
    finally:
        if conn is not None:
            conn.close()


def write_atomic_json(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    with open(tmp, 'w', encoding='utf-8') as handle:
        handle.write(json.dumps(obj, indent=2) + '\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(tmp, 0o644)
    os.replace(tmp, path)


def median_or_none(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def pct_delta(a, b):
    if a is None or b is None or a == 0:
        return None
    return (b - a) / a * 100.0


def fmt(value, spec):
    return 'n/a' if value is None else format(value, spec)


# ---------------------------------------------------------------- recording

def in_git_worktree(path):
    probe = Path(path).resolve()
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        done = subprocess.run(['git', '-C', str(probe), 'rev-parse', '--is-inside-work-tree'],
                              capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return False
    return done.returncode == 0 and done.stdout.strip() == 'true'


def prepare_out_dir(out_dir, allow_git):
    out = Path(out_dir)
    if not allow_git and in_git_worktree(out):
        raise UsageError(f'refusing: {out} is inside a git work tree '
                         '(recordings hold private prompts; pass --allow-git to override)')
    if not out.exists():
        out.mkdir(mode=0o700, parents=True)
        os.chmod(out, 0o700)
    recording = out / 'recording.jsonl'
    if not recording.exists():
        fd = os.open(recording, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.close(fd)
    os.chmod(recording, 0o600)
    return recording


class TurnCapture:
    """Parses one proxied request/response as the bytes pass through; never alters them."""

    def __init__(self, path, body_bytes):
        self.path = path
        self.t_start = time.time()
        self.t_first_byte = None
        self.status = None
        self.sse = False
        self.done = False
        self.error = None
        self.aborted = False
        self.usage = None
        self.timings = None
        self.finish_reason = None
        # reconstructed reply: content, reasoning, tool calls (streamed deltas or message)
        self.content = []
        self.reasoning = []
        self.tool_calls = {}
        self._buf = b''
        self._raw = bytearray()
        try:
            if body_bytes and body_bytes[:2] == b'\x1f\x8b':  # gzip request body
                body_bytes = gzip.decompress(body_bytes)
            self.body = json.loads(body_bytes) if body_bytes else None
        except ValueError:
            self.body = None

    def start(self, status, content_type):
        self.status = status
        self.sse = 'text/event-stream' in (content_type or '').lower()

    def feed(self, data):
        if self.t_first_byte is None:
            self.t_first_byte = time.time()
        if not self.sse:
            self._raw.extend(data)
            return
        self._buf += data
        while b'\n' in self._buf:
            line, self._buf = self._buf.split(b'\n', 1)
            self._line(line)

    def _line(self, line):
        line = line.rstrip(b'\r')
        if not line.startswith(b'data:'):
            return
        payload = line[5:].strip()
        if payload == b'[DONE]':
            self.done = True
            return
        try:
            obj = json.loads(payload)
        except ValueError:
            self.error = 'bad SSE frame'
            self.aborted = True
            return
        self._absorb(obj)

    def _absorb(self, obj):
        if not isinstance(obj, dict):
            return
        if 'error' in obj:
            self.error = json.dumps(obj['error'])[:500]
            self.aborted = True
            return
        for choice in obj.get('choices') or []:
            if isinstance(choice, dict) and choice.get('finish_reason') is not None:
                self.finish_reason = choice['finish_reason']
            if isinstance(choice, dict):
                self._absorb_text(choice.get('delta') or choice.get('message') or {})
        if isinstance(obj.get('content'), str) and 'choices' not in obj:  # native /completion
            self.content.append(obj['content'])
        if obj.get('stop') is True and self.finish_reason is None:  # native /completion
            self.finish_reason = 'stop'
        if isinstance(obj.get('usage'), dict):
            self.usage = obj['usage']
        if isinstance(obj.get('timings'), dict):
            self.timings = obj['timings']

    def _absorb_text(self, d):
        if not isinstance(d, dict):
            return
        if isinstance(d.get('content'), str):
            self.content.append(d['content'])
        if isinstance(d.get('reasoning_content'), str):
            self.reasoning.append(d['reasoning_content'])
        for i, tc in enumerate(d.get('tool_calls') or []):
            if not isinstance(tc, dict):
                continue
            slot = self.tool_calls.setdefault(tc.get('index', i), {'name': '', 'arguments': ''})
            fn = tc.get('function') or {}
            slot['name'] += fn.get('name') or ''
            slot['arguments'] += fn.get('arguments') or ''

    def response(self):
        return {'content': ''.join(self.content), 'reasoning': ''.join(self.reasoning),
                'tool_calls': [self.tool_calls[k] for k in sorted(self.tool_calls)]}

    def finish(self, error=None):
        """Return the turn fields (without kind/seq) once the exchange has ended."""
        if self.sse and self._buf:
            self._line(self._buf)
            self._buf = b''
        aborted = self.aborted
        error_text = self.error
        if not self.sse and self.status is not None and self.status < 400 and self._raw:
            try:
                obj = json.loads(bytes(self._raw))
            except ValueError:
                obj = None
                aborted = True
                error_text = error_text or 'unparseable response body'
            if obj is not None:
                self._absorb(obj)
                error_text = error_text or self.error
                aborted = aborted or self.aborted
        if error is not None:
            aborted = True
            error_text = error
        elif self.sse and not self.done and self.status is not None and self.status < 400:
            aborted = True
            error_text = error_text or 'stream ended without [DONE]'
        if self.status is not None and self.status >= 400 and error_text is None:
            error_text = f'HTTP {self.status}'
        if self.status is None and error_text is None:
            error_text = 'no response'
        if self.body is None and error_text is None:
            error_text = 'request body is not JSON'
        ct, source = None, None
        if self.usage and self.usage.get('completion_tokens') is not None:
            ct, source = int(self.usage['completion_tokens']), 'usage'
        elif self.timings and self.timings.get('predicted_n') is not None:
            ct, source = int(self.timings['predicted_n']), 'timings'
        return {'path': self.path, 't_start': self.t_start, 't_first_byte': self.t_first_byte,
                't_end': time.time(), 'body': self.body, 'status': self.status,
                'completion_tokens': ct, 'completion_tokens_source': source,
                'finish_reason': self.finish_reason, 'timings': self.timings,
                'usage': self.usage, 'aborted': bool(aborted), 'error': error_text,
                'response': self.response()}


class _RecordHandler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    server_version = 'session-replay-proxy'
    sys_version = ''

    def log_message(self, fmt_, *args):
        return

    def do_GET(self):
        self._proxy()

    do_POST = do_PUT = do_DELETE = do_PATCH = do_GET

    def _read_body(self):
        if 'chunked' in self.headers.get('Transfer-Encoding', '').lower():
            parts = []
            while True:
                size_line = self.rfile.readline(65537)
                if not size_line:
                    raise ValueError('truncated chunked request')
                size = int(size_line.split(b';', 1)[0].strip() or b'0', 16)
                if size == 0:
                    while self.rfile.readline(65537) not in (b'\r\n', b'\n', b''):
                        pass
                    return b''.join(parts)
                parts.append(self.rfile.read(size))
                self.rfile.readline(65537)
        length = int(self.headers.get('Content-Length') or 0)
        return self.rfile.read(length) if length else b''

    def _write(self, data, chunked):
        if chunked:
            self.wfile.write(b'%x\r\n' % len(data) + data + b'\r\n')
        else:
            self.wfile.write(data)
        self.wfile.flush()

    def _send_simple(self, status, message):
        data = message.encode('utf-8')
        self.send_response_only(status)
        self.send_header('Content-Type', 'text/plain; charset=utf-8')
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Connection', 'close')
        self.end_headers()
        self.wfile.write(data)

    def _proxy(self):
        self.close_connection = True
        method, path = self.command, self.path
        try:
            body = self._read_body()
        except (ValueError, OSError) as exc:
            self._send_simple(400, f'bad request body: {exc}')
            return
        recordable = method == 'POST' and urlsplit(path).path in RECORD_PATHS
        capture = TurnCapture(path, body) if recordable else None
        headers = {k: v for k, v in self.headers.items()
                   if k.lower() not in HOP_BY_HOP | {'host', 'content-length', 'accept-encoding'}}
        upstream = self.server.upstream
        conn = make_connection(upstream, self.server.timeout)
        try:
            conn.request(method, join_path(upstream, path), body=body, headers=headers)
            resp = conn.getresponse()
        except (OSError, http.client.HTTPException) as exc:
            conn.close()
            self._send_simple(502, f'upstream unreachable: {exc}')
            if recordable:
                self.server.record_turn(capture.finish(error=f'upstream unreachable: {exc}'))
            return

        if capture is not None:
            capture.start(resp.status, resp.getheader('Content-Type'))
        chunked = resp.getheader('Content-Length') is None
        client_error = upstream_error = None
        try:
            self.send_response_only(resp.status, resp.reason)
            for name, value in resp.getheaders():
                if name.lower() in HOP_BY_HOP or name.lower() in ('content-length', 'date', 'server'):
                    continue
                self.send_header(name, value)
            if chunked:
                self.send_header('Transfer-Encoding', 'chunked')
            else:
                self.send_header('Content-Length', resp.getheader('Content-Length'))
            self.send_header('Connection', 'close')
            self.end_headers()
            self.wfile.flush()
        except OSError as exc:
            client_error = f'client disconnected: {exc}'
        try:
            while client_error is None:
                data = resp.read1(65536)
                if not data:
                    break
                if capture is not None:
                    capture.feed(data)
                try:
                    self._write(data, chunked)
                except OSError as exc:
                    client_error = f'client disconnected: {exc}'
        except (OSError, http.client.HTTPException) as exc:
            upstream_error = f'upstream error mid-response: {exc}'
        finally:
            conn.close()
        complete = client_error is None and upstream_error is None and (
            capture is None or not capture.sse or capture.done)
        if chunked and complete:
            try:
                self.wfile.write(b'0\r\n\r\n')
                self.wfile.flush()
            except OSError:
                pass
        if capture is not None:
            self.server.record_turn(capture.finish(error=client_error or upstream_error))


class RecorderServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, listen, upstream, out_dir, allow_git=False, timeout=600.0):
        self.upstream = parse_url(upstream)
        self.upstream_url = upstream
        self.timeout = timeout
        self.recording = prepare_out_dir(out_dir, allow_git)
        self.lock = threading.Lock()
        self.turn_count = 0
        if self.recording.stat().st_size == 0:
            self._append({'kind': 'header', 'version': VERSION, 'tool': TOOL,
                          'created': utc_iso(), 'upstream': upstream,
                          'props': get_json(self.upstream, '/props')})
        else:
            _, turns, _ = load_recording(self.recording)
            self.turn_count = max((int(t.get('seq', 0)) for t in turns), default=0)
            with open(self.recording, 'rb') as handle:
                handle.seek(-1, os.SEEK_END)
                if handle.read(1) != b'\n':  # previous writer died mid-line
                    self._append_raw(b'\n')
        super().__init__(listen, _RecordHandler)

    def _append_raw(self, data):
        with open(self.recording, 'ab') as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())

    def _append(self, obj):
        self._append_raw((json.dumps(obj, ensure_ascii=False) + '\n').encode('utf-8'))

    def record_turn(self, fields):
        with self.lock:
            self.turn_count += 1
            self._append({'kind': 'turn', 'seq': self.turn_count, **fields})


def parse_listen(text):
    host, sep, port = text.rpartition(':')
    if not sep or not port.isdigit():
        raise UsageError(f'--listen must be HOST:PORT, got {text!r}')
    return host or '127.0.0.1', int(port)


def cmd_record(args):
    host, port = parse_listen(args.listen)
    server = RecorderServer((host, port), args.upstream, args.out,
                            allow_git=args.allow_git, timeout=args.timeout)
    print(f'recording on http://{host}:{server.server_address[1]} -> {server.recording} '
          f'(upstream {args.upstream}); Ctrl-C to stop', flush=True)

    def _stop(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _stop)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    print(f'recorded {server.turn_count} turns in {server.recording}', flush=True)
    return 0


# ---------------------------------------------------------------- recording files

def load_recording(path):
    """Return (header, turn records, sha256 of the file bytes)."""
    path = Path(path)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise UsageError(f'cannot read recording {path}: {exc}')
    lines = [(n, t) for n, t in enumerate(raw.decode('utf-8', 'replace').splitlines(), 1) if t.strip()]
    records = []
    for pos, (lineno, text) in enumerate(lines):
        try:
            records.append(json.loads(text))
        except ValueError:
            if pos == len(lines) - 1:
                print(f'warning: ignoring truncated last line {lineno} of {path}', file=sys.stderr)
                break
            raise UsageError(f'{path}:{lineno}: not valid JSON')
    if not records or not isinstance(records[0], dict) or records[0].get('kind') != 'header':
        raise UsageError(f'{path}: first line is not a session_replay header')
    turns = [r for r in records[1:] if isinstance(r, dict) and r.get('kind') == 'turn']
    return records[0], turns, hashlib.sha256(raw).hexdigest()


def content_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return ''.join(p.get('text', '') for p in content
                       if isinstance(p, dict) and isinstance(p.get('text'), str))
    return ''


def agent_info(body):
    """(12-hex hash of the first system message, its character count)."""
    messages = body.get('messages') if isinstance(body, dict) else None
    if not isinstance(messages, list):
        return 'none', 0
    for message in messages:
        if isinstance(message, dict) and message.get('role') == 'system':
            text = content_text(message.get('content'))
            return hashlib.sha256(text.encode('utf-8')).hexdigest()[:12], len(text)
    return 'none', 0


def skip_reason(turn):
    if turn.get('aborted'):
        return f"aborted: {turn.get('error') or 'unknown error'}"
    completion = turn.get('completion_tokens')
    if not isinstance(completion, int) or completion < 1:
        return 'no completion tokens recorded'
    if not isinstance(turn.get('body'), dict):
        return 'no request body recorded'
    return None


# ---------------------------------------------------------------- environment sampling

def sample_env(sys_root=None, proc_root=None):
    sys_root = Path(SYS_ROOT if sys_root is None else sys_root)
    proc_root = Path(PROC_ROOT if proc_root is None else proc_root)
    thermal = {}
    for zone in sorted((sys_root / 'class/thermal').glob('thermal_zone*')):
        try:
            ztype = (zone / 'type').read_text().strip()
            temp = int((zone / 'temp').read_text().strip()) / 1000.0
        except (OSError, ValueError):
            continue
        key, n = ztype, 2
        while key in thermal:
            key = f'{ztype}#{n}'
            n += 1
        thermal[key] = temp
    cpufreq = {}
    for policy in sorted((sys_root / 'devices/system/cpu/cpufreq').glob('policy*')):
        try:
            cpufreq[policy.name] = int((policy / 'scaling_cur_freq').read_text().strip()) / 1000.0
        except (OSError, ValueError):
            continue
    try:
        loadavg = os.getloadavg()[0]
    except (OSError, AttributeError):
        loadavg = None
    mem_available = None
    try:
        for line in (proc_root / 'meminfo').read_text().splitlines():
            if line.startswith('MemAvailable:'):
                mem_available = round(int(line.split()[1]) / 1024.0, 1)
                break
    except (OSError, ValueError, IndexError):
        pass
    return {'t': time.time(), 'thermal': thermal, 'cpufreq_mhz': cpufreq,
            'loadavg': loadavg, 'mem_available_mib': mem_available}


def start_sampler(samples, interval):
    if not interval or interval <= 0:
        return None
    stop = threading.Event()

    def _run():
        while not stop.wait(interval):
            samples.append(sample_env())

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return stop, thread


# ---------------------------------------------------------------- replay

def parse_template_kwargs(items):
    """['KEY=JSON', ...] -> dict (the replay arm's chat_template_kwargs overrides)."""
    out = {}
    for item in items or []:
        key, sep, value = item.partition('=')
        try:
            if not key or not sep:
                raise ValueError(item)
            out[key] = json.loads(value)
        except ValueError:
            raise UsageError(f'--template-kwarg wants KEY=JSON, got {item!r}')
    return out


def rewrite_body(body, target, path, oxe_defaults=True, template_kwargs=None):
    out = copy.deepcopy(body)
    if oxe_defaults and path in CHAT_PATHS:
        # what the Local AI controller adds for oxe (oxe_runtime.request_defaults), so a replay
        # against a bare server renders the same prompt as production
        kwargs = out.setdefault('chat_template_kwargs', {})
        if isinstance(kwargs, dict):
            kwargs.setdefault('enable_thinking', False)
    if template_kwargs and path in CHAT_PATHS:
        # an A/B arm: force these template kwargs (e.g. preserve_thinking=true)
        kwargs = out.get('chat_template_kwargs')
        out['chat_template_kwargs'] = {**(kwargs if isinstance(kwargs, dict) else {}), **template_kwargs}
    out['cache_prompt'] = True
    out['temperature'] = 0
    out['n_predict'] = target
    out['max_tokens'] = target
    out.pop('max_completion_tokens', None)
    out['ignore_eos'] = True
    for key in ('stop', 'seed', 'top_p', 'top_k', 'min_p'):
        out.pop(key, None)
    out['stream'] = True
    if path in CHAT_PATHS:
        out['stream_options'] = {'include_usage': True}
    else:
        out.pop('stream_options', None)
    return out


def _has_output(obj):
    for choice in obj.get('choices') or []:
        delta = (choice or {}).get('delta') or {}
        if delta.get('content') or delta.get('reasoning_content') or delta.get('tool_calls'):
            return True
    return bool(obj.get('content'))


def post_stream(parts, path, body, timeout):
    """POST a streamed request, read it to the end, and return what the frames said."""
    result = {'status': None, 'ttft_ms': None, 'usage': None, 'timings': None,
              'finish_reason': None, 'done': False, 'error': None}
    conn = make_connection(parts, timeout)
    t0 = time.perf_counter()
    try:
        conn.request('POST', join_path(parts, path), body=json.dumps(body).encode('utf-8'),
                     headers={'Content-Type': 'application/json', 'Accept': 'text/event-stream'})
        resp = conn.getresponse()
        result['status'] = resp.status
        if resp.status != 200:
            result['error'] = f'HTTP {resp.status}: ' + resp.read(300).decode('utf-8', 'replace')
        else:
            while True:
                line = resp.readline()
                if not line:
                    break
                line = line.strip()
                if not line.startswith(b'data:'):
                    continue
                payload = line[5:].strip()
                if payload == b'[DONE]':
                    result['done'] = True
                    break
                try:
                    obj = json.loads(payload)
                except ValueError:
                    result['error'] = 'bad SSE frame'
                    break
                if not isinstance(obj, dict):
                    continue
                if 'error' in obj:
                    result['error'] = json.dumps(obj['error'])[:500]
                    break
                if result['ttft_ms'] is None and _has_output(obj):
                    result['ttft_ms'] = (time.perf_counter() - t0) * 1000.0
                for choice in obj.get('choices') or []:
                    if isinstance(choice, dict) and choice.get('finish_reason') is not None:
                        result['finish_reason'] = choice['finish_reason']
                if obj.get('stop') is True and result['finish_reason'] is None:
                    result['finish_reason'] = 'stop'
                if isinstance(obj.get('usage'), dict):
                    result['usage'] = obj['usage']
                if isinstance(obj.get('timings'), dict):
                    result['timings'] = obj['timings']
            if not result['done'] and result['error'] is None:
                result['error'] = 'stream ended without [DONE]'
    except (OSError, http.client.HTTPException, ValueError) as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        conn.close()
    result['wall_ms'] = (time.perf_counter() - t0) * 1000.0
    return result


def post_json(parts, path, body, timeout):
    result = {'status': None, 'usage': None, 'timings': None, 'error': None}
    conn = make_connection(parts, timeout)
    t0 = time.perf_counter()
    try:
        conn.request('POST', join_path(parts, path), body=json.dumps(body).encode('utf-8'),
                     headers={'Content-Type': 'application/json'})
        resp = conn.getresponse()
        result['status'] = resp.status
        data = resp.read()
        if resp.status != 200:
            result['error'] = f'HTTP {resp.status}'
        else:
            obj = json.loads(data)
            result['usage'] = obj.get('usage')
            result['timings'] = obj.get('timings')
    except (OSError, http.client.HTTPException, ValueError) as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        conn.close()
    result['wall_ms'] = (time.perf_counter() - t0) * 1000.0
    return result


def run_scrub(parts, timeout):
    body = {'messages': [{'role': 'system', 'content': 'scrub ' + secrets.token_hex(16)},
                         {'role': 'user', 'content': 'hi'}],
            'temperature': 0, 'n_predict': 1, 'cache_prompt': True, 'stream': False}
    return post_json(parts, '/v1/chat/completions', body, timeout)


def run_turn(parts, turn, timeout, oxe_defaults=True, template_kwargs=None):
    path = urlsplit(turn['path']).path
    target = turn['completion_tokens']
    original = turn['body']
    agent, system_chars = agent_info(original)
    messages = original.get('messages') if isinstance(original.get('messages'), list) else []
    body = rewrite_body(original, target, path, oxe_defaults, template_kwargs)
    t_send = time.time()
    reply = post_stream(parts, path, body, timeout)
    timings = reply['timings'] or {}
    usage = reply['usage'] or {}
    cache_n = timings.get('cache_n')
    prompt_n = timings.get('prompt_n')
    prompt_ms = timings.get('prompt_ms')
    predicted_n = timings.get('predicted_n')
    predicted_ms = timings.get('predicted_ms')
    prompt_tokens = usage.get('prompt_tokens')
    if prompt_tokens is None and cache_n is not None and prompt_n is not None:
        prompt_tokens = cache_n + prompt_n
    wall = reply['wall_ms']
    other = None
    if prompt_ms is not None and predicted_ms is not None:
        other = wall - prompt_ms - predicted_ms
    error = reply['error']
    return {
        'seq': turn['seq'], 'path': path, 'agent': agent, 'system_chars': system_chars,
        'n_messages': len(messages),
        'last_role': messages[-1].get('role') if messages and isinstance(messages[-1], dict) else None,
        'target_n': target, 'status': reply['status'], 't_send': t_send,
        'prompt_tokens': prompt_tokens, 'cache_n': cache_n, 'prompt_n': prompt_n,
        'prompt_ms': prompt_ms, 'predicted_n': predicted_n, 'predicted_ms': predicted_ms,
        'decode_tok_s': (predicted_n / predicted_ms * 1000.0
                         if predicted_n is not None and predicted_ms else None),
        'prefill_tok_s': (prompt_n / prompt_ms * 1000.0
                          if prompt_n is not None and prompt_ms else None),
        'ttft_ms': reply['ttft_ms'], 'wall_ms': wall, 'other_ms': other,
        'draft_n': timings.get('draft_n'), 'draft_n_accepted': timings.get('draft_n_accepted'),
        'equal_work': predicted_n == target and error is None, 'error': error,
    }


def compute_totals(rows):
    def total(key):
        return sum((r.get(key) or 0) for r in rows)

    return {
        'wall_s': total('wall_ms') / 1000.0, 'prompt_s': total('prompt_ms') / 1000.0,
        'decode_s': total('predicted_ms') / 1000.0, 'other_s': total('other_ms') / 1000.0,
        'prompt_tokens_new': int(total('prompt_n')), 'cache_tokens': int(total('cache_n')),
        'predicted_tokens': int(total('predicted_n')), 'draft_n': int(total('draft_n')),
        'draft_n_accepted': int(total('draft_n_accepted')),
        'equal_work_all': bool(rows) and all(r['equal_work'] for r in rows),
        'n_turns': len(rows), 'n_errors': sum(1 for r in rows if r['error']),
        'ttft_first_ms': rows[0]['ttft_ms'] if rows else None,
    }


def run_repeat(parts, turns, args, index):
    samples = [sample_env()]
    scrub = None if args.no_scrub else run_scrub(parts, args.timeout)
    sampler = start_sampler(samples, args.sample_interval)
    rows, skipped, eligible = [], [], 0
    try:
        for turn in turns:
            if args.limit and eligible >= args.limit:
                break
            reason = skip_reason(turn)
            if reason:
                skipped.append({'seq': turn.get('seq'), 'reason': reason})
                continue
            eligible += 1
            row = run_turn(parts, turn, args.timeout, not getattr(args, "no_oxe_defaults", False),
                           parse_template_kwargs(getattr(args, 'template_kwarg', None)))
            sample = sample_env()
            samples.append(sample)
            row['pkg_temp_max_c'] = max(sample['thermal'].values()) if sample['thermal'] else None
            row['freq_mhz'] = sample['cpufreq_mhz']
            rows.append(row)
    finally:
        if sampler is not None:
            sampler[0].set()
            sampler[1].join(5)
    samples.append(sample_env())
    return {'repeat': index + 1, 'scrub': scrub, 'turns': rows, 'skipped': skipped,
            'totals': compute_totals(rows), 'env': samples}


def summarize(repeats):
    walls = [r['totals']['wall_s'] for r in repeats]
    median = statistics.median(walls)
    spread = (max(walls) - min(walls)) / median * 100.0 if median > 0 else 0.0
    return {'wall_s_median': median, 'wall_s_all': walls,
            'decode_s_median': statistics.median(r['totals']['decode_s'] for r in repeats),
            'prompt_s_median': statistics.median(r['totals']['prompt_s'] for r in repeats),
            'spread_pct': spread}


def print_replay_table(repeats):
    print(f"{'repeat':>6} {'wall_s':>8} {'prefill_s':>9} {'decode_s':>8} {'other_s':>8} "
          f"{'new_prompt':>10} {'errors':>6}  equal_work")
    for rep in repeats:
        t = rep['totals']
        print(f"{rep['repeat']:>6} {t['wall_s']:>8.3f} {t['prompt_s']:>9.3f} {t['decode_s']:>8.3f} "
              f"{t['other_s']:>8.3f} {t['prompt_tokens_new']:>10} {t['n_errors']:>6}  "
              f"{'yes' if t['equal_work_all'] else 'NO'}")


def cmd_replay(args):
    if args.repeat < 1:
        raise UsageError('--repeat must be >= 1')
    _, turns, sha = load_recording(args.recording)
    parts = parse_url(args.server)
    props = get_json(parts, '/props')
    models = get_json(parts, '/v1/models')
    started = utc_iso()
    repeats = [run_repeat(parts, turns, args, i) for i in range(args.repeat)]
    result = {'tool': TOOL, 'version': VERSION, 'label': args.label,
              'recording': str(Path(args.recording).resolve()), 'recording_sha256': sha,
              'template_kwargs': parse_template_kwargs(getattr(args, 'template_kwarg', None)),
              'server': args.server, 'props': props, 'models': models, 'started': started,
              'repeats': repeats, 'summary': summarize(repeats)}
    write_atomic_json(args.out, result)
    print_replay_table(repeats)
    print(f"summary: wall median {result['summary']['wall_s_median']:.3f} s, "
          f"spread {result['summary']['spread_pct']:.2f}% over {args.repeat} repeat(s); "
          f"result -> {args.out}")
    equal = all(rep['totals']['equal_work_all'] for rep in repeats)
    return 0 if equal else 3


# ---------------------------------------------------------------- compare / info

def load_result(path):
    try:
        with open(path, encoding='utf-8') as handle:
            result = json.load(handle)
    except (OSError, ValueError) as exc:
        raise UsageError(f'cannot read result {path}: {exc}')
    if result.get('tool') != TOOL or 'repeats' not in result:
        raise UsageError(f'{path} is not a session_replay result')
    return result


def per_turn_medians(result):
    grouped = {}
    for rep in result['repeats']:
        for row in rep['turns']:
            grouped.setdefault(row['seq'], []).append(row)
    return {seq: {'wall': median_or_none([r['wall_ms'] for r in rows]),
                  'prompt': median_or_none([r['prompt_ms'] for r in rows]),
                  'decode': median_or_none([r['predicted_ms'] for r in rows])}
            for seq, rows in grouped.items()}


def cmd_compare(args):
    a, b = load_result(args.a), load_result(args.b)
    if a.get('recording_sha256') != b.get('recording_sha256'):
        msg = 'results come from different recordings (recording_sha256 differs)'
        if not args.allow_different_recording:
            print(f'refusing to compare: {msg}; pass --allow-different-recording to override',
                  file=sys.stderr)
            return 2
        print(f'warning: {msg}', file=sys.stderr)
    ta, tb = per_turn_medians(a), per_turn_medians(b)
    common = sorted(set(ta) & set(tb))
    print(f"{'seq':>5} {'wall A ms':>10} {'wall B ms':>10} {'d%':>7} {'prefill A':>10} "
          f"{'prefill B':>10} {'d%':>7} {'decode A':>10} {'decode B':>10} {'d%':>7}")
    for seq in common:
        x, y = ta[seq], tb[seq]
        print(f"{seq:>5} {fmt(x['wall'], '10.1f')} {fmt(y['wall'], '10.1f')} "
              f"{fmt(pct_delta(x['wall'], y['wall']), '+7.2f')} {fmt(x['prompt'], '10.1f')} "
              f"{fmt(y['prompt'], '10.1f')} {fmt(pct_delta(x['prompt'], y['prompt']), '+7.2f')} "
              f"{fmt(x['decode'], '10.1f')} {fmt(y['decode'], '10.1f')} "
              f"{fmt(pct_delta(x['decode'], y['decode']), '+7.2f')}")
    if not common:
        print('no turns in common')
    wall_a = median_or_none([r['totals']['wall_s'] for r in a['repeats']])
    wall_b = median_or_none([r['totals']['wall_s'] for r in b['repeats']])
    delta = pct_delta(wall_a, wall_b)
    print(f"totals wall (median over repeats): A {fmt(wall_a, '.3f')} s  B {fmt(wall_b, '.3f')} s  "
          f"delta {fmt(delta, '+.2f')}%")
    within = delta is not None and abs(delta) <= args.gate_pct
    print(f"gate {args.gate_pct:g}%: {'WITHIN' if within else 'OUTSIDE'}")
    return 0 if within else 1


def _value_key(value):
    if value is None:
        return 'null'
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, (int, float, str)):
        return str(value)
    return json.dumps(value, sort_keys=True)


def render_reply(resp):
    """The reply as the model emitted it (Qwen3.x/Ornith XML tool calls), for tokenizing."""
    parts = []
    if resp.get('reasoning'):
        parts.append('<think>\n' + resp['reasoning'] + '\n</think>\n\n')
    text = resp.get('content') or ''
    calls = []
    for tc in resp.get('tool_calls') or []:
        try:
            args = json.loads(tc.get('arguments') or '{}')
        except ValueError:
            args = {}
        params = ''.join(f'<parameter={k}>\n{v if isinstance(v, str) else json.dumps(v)}\n</parameter>\n'
                         for k, v in (args.items() if isinstance(args, dict) else []))
        calls.append(f"<tool_call>\n<function={tc.get('name', '')}>\n{params}</function>\n</tool_call>")
    if calls:
        text = (text + '\n\n' if text else '') + '\n'.join(calls)
    parts.append(text)
    return ''.join(parts)


def post_plain(parts, path, body, timeout=60.0):
    conn = make_connection(parts, timeout)
    try:
        conn.request('POST', join_path(parts, path), body=json.dumps(body).encode('utf-8'),
                     headers={'Content-Type': 'application/json'})
        resp = conn.getresponse()
        data = resp.read()
        if resp.status != 200:
            raise UsageError(f'{path}: HTTP {resp.status}: {data[:200]!r}')
        return json.loads(data)
    finally:
        conn.close()


def cmd_tokens(args):
    """Token ids per turn (prompt exact; completion reconstructed) via a tokenizer-only
    server: `oxe-server --engine stub --model <gguf>` (loads only the tokenizer + template)."""
    parts = parse_url(args.server)
    _header, turns, _sha = load_recording(args.recording)
    out = Path(args.out)
    n_exact = n_total = 0
    rows = []
    for turn in turns:
        body = turn.get('body') or {}
        path = urlsplit(turn.get('path', '')).path
        if not isinstance(body, dict) or turn.get('aborted') or turn.get('completion_tokens') is None:
            continue
        if path in CHAT_PATHS:
            req = {k: body[k] for k in ('messages', 'tools', 'tool_choice', 'chat_template_kwargs') if k in body}
            if not args.no_oxe_defaults:
                kw = req.setdefault('chat_template_kwargs', {})
                if isinstance(kw, dict):
                    kw.setdefault('enable_thinking', False)
            prompt = post_plain(parts, '/apply-template', req)['prompt']
        else:
            prompt = body.get('prompt') if isinstance(body.get('prompt'), str) else ''
        prompt_ids = post_plain(parts, '/tokenize', {'content': prompt, 'add_special': True, 'parse_special': True})['tokens']
        reply = render_reply(turn.get('response') or {})
        completion_ids = post_plain(parts, '/tokenize', {'content': reply, 'add_special': False, 'parse_special': True})['tokens'] if reply else []
        recorded = turn['completion_tokens']
        # the recorded count includes the end-of-generation token when the reply stopped on it
        exact = len(completion_ids) in (recorded, recorded - 1)
        n_total += 1
        n_exact += exact
        timings = turn.get('timings') or {}
        rows.append({'seq': turn['seq'], 'agent': agent_info(body)[0], 'prompt_ids': prompt_ids,
                     'completion_ids': completion_ids, 'completion_tokens_recorded': recorded,
                     'completion_len_matches': exact, 'finish_reason': turn.get('finish_reason'),
                     'cache_n': timings.get('cache_n'), 'prompt_n': timings.get('prompt_n'),
                     'prompt_ms': timings.get('prompt_ms'), 'predicted_n': timings.get('predicted_n'),
                     'predicted_ms': timings.get('predicted_ms')})
    tmp = out.with_name(out.name + '.tmp')
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    os.replace(tmp, out)
    print(f'{n_total} turns -> {out}; completion length matches the recorded count in {n_exact}/{n_total}')
    return 0 if n_exact == n_total else 4


def cmd_info(args):
    header, turns, sha = load_recording(args.recording)
    aborted = [t for t in turns if t.get('aborted')]
    usable = [t for t in turns if not skip_reason(t)]
    total_completion = sum(t['completion_tokens'] for t in turns
                           if not t.get('aborted') and isinstance(t.get('completion_tokens'), int))
    print(f'recording: {args.recording}')
    print(f'header: version {header.get("version")}, created {header.get("created")}, '
          f'upstream {header.get("upstream")}')
    print(f'turns: {len(turns)}  replayable: {len(usable)}  aborted: {len(aborted)}  '
          f'completion tokens (non-aborted): {total_completion}')
    agents = {}
    for turn in turns:
        key, chars = agent_info(turn.get('body'))
        count, _ = agents.get(key, (0, chars))
        agents[key] = (count + 1, chars)
    print(f'agents: {len(agents)}')
    for key, (count, chars) in sorted(agents.items(), key=lambda kv: -kv[1][0]):
        print(f'  {key}: {count} turns, system prompt {chars} chars')
    print('sampling parameters (turns per value):')
    for key in INFO_KEYS:
        counts = Counter()
        for turn in turns:
            body = turn.get('body') if isinstance(turn.get('body'), dict) else {}
            counts[_value_key(body[key]) if key in body else 'absent'] += 1
        rendered = ', '.join(f'{v}={n}' for v, n in counts.most_common())
        print(f'  {key}: {rendered or "no turns"}')
    ov = concurrency(turns)
    print(f"concurrency (WP-D6): {ov['overlapping_requests']}/{ov['requests']} requests overlap another; "
          f"decode {ov['decode_s']:.1f} s, with >=2 in flight {ov['overlap_decode_s']:.1f} s "
          f"({ov['overlap_share_pct']:.1f}%)")
    return 0


def concurrency(turns):
    """Client-side overlap from the recorded times: a request is in flight from t_start to
    t_end; its decode phase is t_first_byte..t_end (streamed) or the whole request."""
    iv = [(t['t_start'], t.get('t_first_byte') or t['t_start'], t['t_end']) for t in turns
          if t.get('t_start') is not None and t.get('t_end') is not None]
    overlapping = sum(1 for i, (a, _, b) in enumerate(iv)
                      if any(j != i and a < d and c < b for j, (c, _, d) in enumerate(iv)))
    # sweep: seconds of each request's decode phase during which another request is in flight
    decode = overlap = 0.0
    for i, (_, f, b) in enumerate(iv):
        decode += max(0.0, b - f)
        others = sorted((max(c, f), min(d, b)) for j, (c, _, d) in enumerate(iv) if j != i and c < b and d > f)
        end = f
        for lo, hi in others:
            lo = max(lo, end)
            if hi > lo:
                overlap += hi - lo
                end = hi
    return {'requests': len(iv), 'overlapping_requests': overlapping, 'decode_s': decode,
            'overlap_decode_s': overlap, 'overlap_share_pct': 100.0 * overlap / decode if decode else 0.0}


# ---------------------------------------------------------------- CLI

def build_parser():
    parser = argparse.ArgumentParser(prog='session_replay.py',
                                     description='Record and replay coding-agent sessions.')
    sub = parser.add_subparsers(dest='cmd', required=True)

    rec = sub.add_parser('record', help='logging proxy in front of a server')
    rec.add_argument('--listen', default='127.0.0.1:18200', help='HOST:PORT to listen on')
    rec.add_argument('--upstream', required=True, help='upstream base URL, e.g. http://127.0.0.1:8080')
    rec.add_argument('--out', required=True, help='directory for recording.jsonl (created 0700)')
    rec.add_argument('--allow-git', action='store_true', help='allow --out inside a git work tree')
    rec.add_argument('--timeout', type=float, default=600.0, help='upstream socket timeout (s)')

    rep = sub.add_parser('replay', help='replay a recording with equal work')
    rep.add_argument('recording')
    rep.add_argument('--server', required=True, help='base URL of the server under test')
    rep.add_argument('--out', required=True, help='result JSON path')
    rep.add_argument('--label', default='', help='free-text label stored in the result')
    rep.add_argument('--repeat', type=int, default=1)
    rep.add_argument('--no-scrub', action='store_true', help='skip the prefix-state scrub request')
    rep.add_argument('--no-oxe-defaults', action='store_true', help='do not add the controller default chat_template_kwargs.enable_thinking=false')
    rep.add_argument('--template-kwarg', action='append', metavar='KEY=JSON',
                     help='force a chat_template_kwargs value on every chat turn (A/B arm), e.g. preserve_thinking=true')
    rep.add_argument('--limit', type=int, default=None, help='replay only the first N eligible turns')
    rep.add_argument('--timeout', type=float, default=1800.0)
    rep.add_argument('--sample-interval', type=float, default=0.0,
                     help='background environment sample period in seconds (0 = off)')

    cmp_ = sub.add_parser('compare', help='compare two replay results')
    cmp_.add_argument('a')
    cmp_.add_argument('b')
    cmp_.add_argument('--gate-pct', type=float, default=2.0)
    cmp_.add_argument('--allow-different-recording', action='store_true')

    info = sub.add_parser('info', help='summarise a recording')
    info.add_argument('recording')

    tk = sub.add_parser('tokens', help='prompt/completion token ids per turn (via a tokenizer-only oxe-server --engine stub)')
    tk.add_argument('recording')
    tk.add_argument('--server', required=True)
    tk.add_argument('--out', required=True)
    tk.add_argument('--no-oxe-defaults', action='store_true')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.cmd == 'record':
            return cmd_record(args)
        if args.cmd == 'replay':
            return cmd_replay(args)
        if args.cmd == 'compare':
            return cmd_compare(args)
        if args.cmd == 'tokens':
            return cmd_tokens(args)
        return cmd_info(args)
    except UsageError as exc:
        print(f'session_replay: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
