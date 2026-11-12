"""Deterministic llama-server-shaped mock for session_replay tests. Binds 127.0.0.1 port 0 only.

No model is involved: token counts are fake (4-character units of the request JSON), generation
length is scripted, and each generated token is one "w " content delta.
"""
import os as _os, tempfile as _tempfile
_os.environ.setdefault('LOCAL_AI_LEDGER_DIR', _tempfile.mkdtemp(prefix='local-ai-test-ledger-'))  # never the real request ledger
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

NATURAL_LEN = 7
TOKEN_SLEEP_S = 0.001
CHAT_PATHS = ('/v1/chat/completions', '/chat/completions')
COMPLETION_PATHS = ('/completion', '/completions')
DEFAULT_PROPS = {'model_path': '/models/synthetic.gguf', 'build_info': 'mock-build',
                 'total_slots': 1, 'default_generation_settings': {'n_ctx': 16384}}


def units_of(body):
    if isinstance(body, dict) and 'messages' in body:
        text = json.dumps(body.get('messages')) + json.dumps(body.get('tools'))
    else:
        text = json.dumps(body.get('prompt', '') if isinstance(body, dict) else '')
    return [text[i:i + 4] for i in range(0, len(text) - len(text) % 4, 4)]


def generation_len(body, ignore_n_predict):
    """Return (tokens to generate, limit or None)."""
    if ignore_n_predict:
        return 3, None
    limit = None
    for key in ('n_predict', 'max_tokens', 'max_completion_tokens'):
        value = body.get(key)
        if isinstance(value, int) and value > 0:
            limit = value
            break
    if limit is None:
        return NATURAL_LEN, None
    if body.get('ignore_eos') is True:
        return limit, limit
    return min(limit, NATURAL_LEN), limit


class MockServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, fail_at=(), ignore_n_predict=False, props=None):
        super().__init__(('127.0.0.1', 0), MockHandler)
        self.fail_at = set(fail_at)            # 0-based index among POST requests
        self.ignore_n_predict = ignore_n_predict
        self.props = dict(DEFAULT_PROPS if props is None else props)
        self.lock = threading.Lock()
        self.received = []                      # [{'path', 'body'}] for every POST
        self.get_paths = []
        self.prev_units = []
        self.url = f'http://127.0.0.1:{self.server_address[1]}'
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self.shutdown()
        self.server_close()
        if self._thread is not None:
            self._thread.join(5)


def _timings(cache_n, prompt_n, gen):
    prompt_ms = prompt_n * 0.5
    predicted_ms = gen * 2.0
    return {'cache_n': cache_n, 'prompt_n': prompt_n, 'prompt_ms': prompt_ms,
            'prompt_per_token_ms': prompt_ms / prompt_n if prompt_n else 0.0,
            'prompt_per_second': 2000.0 if prompt_n else 0.0,
            'predicted_n': gen, 'predicted_ms': predicted_ms,
            'predicted_per_token_ms': 2.0 if gen else 0.0,
            'predicted_per_second': 500.0 if gen else 0.0}


class MockHandler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    server_version = 'mock-oxe'
    sys_version = ''

    def log_message(self, fmt, *args):
        return

    def _json(self, obj, status=200):
        data = json.dumps(obj).encode('utf-8')
        self.close_connection = True
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Connection', 'close')
        self.end_headers()
        self.wfile.write(data)

    def _start_stream(self):
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Transfer-Encoding', 'chunked')
        self.send_header('Connection', 'close')
        self.end_headers()

    def _chunk(self, data):
        self.wfile.write(b'%x\r\n' % len(data) + data + b'\r\n')
        self.wfile.flush()

    def _frame(self, obj):
        self._chunk(b'data: ' + json.dumps(obj).encode('utf-8') + b'\n\n')

    def do_GET(self):
        path = urlsplit(self.path).path
        with self.server.lock:
            self.server.get_paths.append(path)
        if path == '/props':
            self._json(self.server.props)
        elif path == '/v1/models':
            self._json({'object': 'list', 'data': [{'id': 'mock-model', 'object': 'model'}]})
        elif path == '/health':
            self._json({'status': 'ok'})
        else:
            self._json({'error': 'not found'}, 404)

    def do_POST(self):
        self.close_connection = True
        length = int(self.headers.get('Content-Length') or 0)
        body = json.loads(self.rfile.read(length) or b'{}')
        path = urlsplit(self.path).path
        if path == '/apply-template':
            self._json({'prompt': '<|sys|>' + json.dumps(body.get('messages')) + json.dumps(body.get('chat_template_kwargs'))})
            return
        if path == '/tokenize':
            text = body.get('content') or ''
            self._json({'tokens': [len(w) for w in text.split()]})
            return
        if path not in CHAT_PATHS + COMPLETION_PATHS:
            self._json({'error': 'not found'}, 404)
            return
        server = self.server
        with server.lock:
            index = len(server.received)
            server.received.append({'path': path, 'body': body, 'origin': self.headers.get('X-Local-AI-Origin')})
            fail = index in server.fail_at
            units = units_of(body)
            prev = server.prev_units
            common = 0
            while common < min(len(units), len(prev)) and units[common] == prev[common]:
                common += 1
            total = len(units)
            cache_n = min(common, max(total - 1, 0))
            prompt_n = total - cache_n
            server.prev_units = units
        gen, limit = generation_len(body, server.ignore_n_predict)
        finish = 'length' if limit is not None and gen >= limit else 'stop'
        timings = _timings(cache_n, prompt_n, gen)
        usage = {'prompt_tokens': total, 'completion_tokens': gen, 'total_tokens': total + gen,
                 'prompt_tokens_details': {'cached_tokens': cache_n}}
        stream = body.get('stream') is True
        if path in CHAT_PATHS:
            if stream:
                include_usage = bool((body.get('stream_options') or {}).get('include_usage'))
                self._stream_chat(gen, finish, timings, usage if include_usage else None, fail)
            else:
                time.sleep(gen * TOKEN_SLEEP_S)
                self._json({'id': 'chatcmpl-mock', 'object': 'chat.completion',
                            'model': 'mock-model',
                            'choices': [{'index': 0, 'finish_reason': finish,
                                         'message': {'role': 'assistant',
                                                     'content': 'w ' * gen}}],
                            'usage': usage, 'timings': timings})
        else:
            if stream:
                self._stream_completion(gen, timings, fail)
            else:
                time.sleep(gen * TOKEN_SLEEP_S)
                self._json({'content': 'w ' * gen, 'stop': True, 'tokens_predicted': gen,
                            'timings': timings})

    def _stream_chat(self, gen, finish, timings, usage, fail):
        self._start_stream()
        self._frame({'object': 'chat.completion.chunk',
                     'choices': [{'index': 0, 'delta': {'role': 'assistant'},
                                  'finish_reason': None}]})
        for i in range(gen):
            if fail and i == 2:
                return  # close mid-stream: no terminating chunk, no [DONE]
            time.sleep(TOKEN_SLEEP_S)
            self._frame({'object': 'chat.completion.chunk',
                         'choices': [{'index': 0, 'delta': {'content': 'w '},
                                      'finish_reason': None}]})
        last = {'object': 'chat.completion.chunk',
                'choices': [{'index': 0, 'delta': {}, 'finish_reason': finish}]}
        if usage is None:
            last['timings'] = timings
            self._frame(last)
        else:
            self._frame(last)
            self._frame({'object': 'chat.completion.chunk', 'choices': [],
                         'usage': usage, 'timings': timings})
        self._chunk(b'data: [DONE]\n\n')
        self._chunk(b'')

    def _stream_completion(self, gen, timings, fail):
        self._start_stream()
        for i in range(gen):
            if fail and i == 2:
                return
            time.sleep(TOKEN_SLEEP_S)
            self._frame({'content': 'w ', 'stop': False})
        self._frame({'content': '', 'stop': True, 'tokens_predicted': gen, 'timings': timings})
        self._chunk(b'data: [DONE]\n\n')
        self._chunk(b'')
