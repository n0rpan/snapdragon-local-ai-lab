"""A model-free controller stack for the OpenAI-surface tests.

`Stack` serves the real controller `Handler` and `Manager.config/models/...` over a
synthetic runtime.json, with model start replaced by a marker (no process, no lease, no
memory check). The backend is either `FakeEngine` (pure Python, shapes copied from
oxe-server) or a real `oxe-server --engine stub` given by port.

Run it by hand to point a real client at it:

    oxe-server --engine stub --model <gguf> --alias ornith-35b-a3b-local --port 18193 &
    PYTHONPATH=src python3 tests/openai_api_stack.py --backend-port 18193 --port 18292
"""
import os as _os, tempfile as _tempfile
_os.environ.setdefault('LOCAL_AI_LEDGER_DIR', _tempfile.mkdtemp(prefix='local-ai-test-ledger-'))  # never the real request ledger
import argparse
import contextlib
import json
import os
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from snapdragon_ai_lab.runtime import bigmoe_runtime, controller
try:
    from snapdragon_ai_lab.runtime.harness_registry import HarnessRegistry
except ImportError:  # a controller from before the harness registry (for before/after audits)
    HarnessRegistry = None

ORNITH, QWEN, FLASH = 'ornith-35b-a3b-local', 'qwen3.6-35b-a3b-local', 'qwen3.8-flash-next-local'
REPLY = 'Hello from the fake engine.'


def oxe_profile(name, display):
    return {'ready': True, 'backend_kind': 'oxe', 'display_name': display, 'model_path': '/synthetic/%s.gguf' % name,
            'sha256': 'a' * 64, 'expected_bytes': 100, 'server_binary': '/synthetic/oxe-server',
            'server_sha256': 'b' * 64, 'context_size': 32768, 'threads': 8,
            'oxe_args': ['--model', '{model}', '--alias', '{alias}', '--ctx-size', '{context}']}


def runtime_value(**extra):
    return {'ready': True, 'model_alias': ORNITH, 'selected_model': ORNITH, **oxe_profile(ORNITH, 'Ornith 1.5 35B A3B'),
            'profiles': {ORNITH: oxe_profile(ORNITH, 'Ornith 1.5 35B A3B'), QWEN: oxe_profile(QWEN, 'Qwen 3.6 35B MoE'),
                         FLASH: {'ready': True, 'backend_kind': bigmoe_runtime.KIND, 'display_name': 'Flash-Next'}},
            **extra}


class FakeEngine:
    """oxe-server's routes and reply shapes, with a character-level stand-in for the
    prompt cache (4 characters per "token", reuse = common prefix with the last prompt)."""

    def __init__(self):
        engine = self
        self.requests = []
        self.last_prompt = ''

        class Handler(BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'

            def log_message(self, *args):
                pass

            def _json(self, code, value):
                data = json.dumps(value).encode()
                self.send_response(code)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _sse(self, frames):
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Connection', 'close')
                self.end_headers()
                for frame in frames:
                    self.wfile.write(('data: %s\n\n' % (frame if isinstance(frame, str) else json.dumps(frame))).encode())
                    self.wfile.flush()
                self.close_connection = True

            def do_GET(self):
                if self.path == '/health':
                    return self._json(200, {'status': 'ok'})
                self._json(404, {'error': {'message': 'File Not Found', 'type': 'not_found_error', 'code': 404}})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get('Content-Length', '0'))) or b'{}')
                engine.requests.append({'path': self.path, 'body': body, 'headers': dict(self.headers.items())})
                if self.path == '/v1/chat/completions':
                    return self.chat(body)
                if self.path == '/completion':
                    return self.completion(body)
                self._json(404, {'error': {'message': 'File Not Found', 'type': 'not_found_error', 'code': 404}})

            def counts(self, prompt):
                shared = 0
                for a, b in zip(prompt, engine.last_prompt):
                    if a != b:
                        break
                    shared += 1
                engine.last_prompt = prompt
                total = max(1, len(prompt) // 4)
                cached = min(shared // 4, total - 1)
                return cached, total - cached

            def chat(self, body):
                fmt = body.get('response_format')
                if fmt is not None and not (isinstance(fmt, dict) and fmt.get('type') == 'text'):
                    return self._json(400, {'error': {'code': 400, 'type': 'invalid_request_error',
                                                      'message': '"response_format" is not supported by oxe-server'}})
                prompt = json.dumps([body.get('tools', []), body.get('messages')], ensure_ascii=False)
                cached, evaluated = self.counts(prompt)
                words = REPLY.split(' ')[:body.get('max_tokens') or body.get('max_completion_tokens') or 99]
                finish = 'stop' if len(words) == len(REPLY.split(' ')) else 'length'
                usage = {'completion_tokens': len(words), 'prompt_tokens': cached + evaluated,
                         'total_tokens': cached + evaluated + len(words), 'prompt_tokens_details': {'cached_tokens': cached}}
                timings = {'cache_n': cached, 'prompt_n': evaluated, 'predicted_n': len(words)}
                meta = {'created': 1, 'id': 'chatcmpl-fake', 'model': body.get('model'), 'system_fingerprint': 'fake'}
                if body.get('stream') is not True:
                    return self._json(200, {'choices': [{'finish_reason': finish, 'index': 0,
                                                         'message': {'role': 'assistant', 'content': ' '.join(words)}}],
                                            **meta, 'object': 'chat.completion', 'usage': usage, 'timings': timings})
                chunk = lambda delta, fin=None: {'choices': [{'finish_reason': fin, 'index': 0, 'delta': delta}],
                                                 **meta, 'object': 'chat.completion.chunk'}
                frames = [chunk({'role': 'assistant', 'content': None})]
                frames += [chunk({'content': (' ' if i else '') + w}) for i, w in enumerate(words)]
                frames.append(chunk({}, finish))
                if isinstance(body.get('stream_options'), dict) and body['stream_options'].get('include_usage'):
                    frames.append({'choices': [], **meta, 'object': 'chat.completion.chunk', 'usage': usage})
                frames[-1]['timings'] = timings
                self._sse(frames + ['[DONE]'])

            def completion(self, body):
                words = REPLY.split(' ')[:body.get('n_predict') or 99]
                stop_type = 'eos' if len(words) == len(REPLY.split(' ')) else 'limit'
                prompt = body.get('prompt') if isinstance(body.get('prompt'), str) else ''
                cached, evaluated = self.counts(prompt)
                final = {'index': 0, 'content': '', 'tokens': [], 'id_slot': 0, 'stop': True, 'model': ORNITH,
                         'tokens_predicted': len(words), 'tokens_evaluated': cached + evaluated, 'stop_type': stop_type,
                         'timings': {'cache_n': cached, 'prompt_n': evaluated, 'predicted_n': len(words)}}
                prob = lambda w: {'id': 1, 'token': w, 'bytes': list(w.encode()), 'logprob': -0.5,
                                  'top_logprobs': [{'id': 1, 'token': w, 'bytes': [], 'logprob': -0.5}]}
                pieces = [(' ' if i else '') + w for i, w in enumerate(words)]
                if body.get('stream') is True:
                    frames = [{'index': 0, 'content': p, 'tokens': [], 'stop': False, 'id_slot': 0,
                               **({'completion_probabilities': [prob(p)]} if body.get('n_probs') else {})} for p in pieces]
                    return self._sse(frames + [final])
                final['content'] = ''.join(pieces)
                if body.get('n_probs'):
                    final['completion_probabilities'] = [prob(p) for p in pieces]
                self._json(200, final)

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.server.daemon_threads = True
        self.port = self.server.server_port
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)


class StackManager(controller.Manager):
    """The real Manager with model start and stop replaced by markers. Nothing is launched
    and nothing is signalled."""

    loads = 0

    def begin(self, *, inference=False, allow_start=False, model_alias=None, origin=None, explicit_load=False):
        with self.mu:
            if inference and self.inferences:
                raise controller.Unavailable('A language-model request is already running; retry when it finishes.')
            config = self.config(model_alias) if model_alias is not None or self.process is None else self.loaded_config
            if self.process is not None and config['model_alias'] != self.loaded_config['model_alias']:
                self.cleanup()
            if self.process is None:
                if not allow_start:
                    raise controller.Unavailable('Model is sleeping; submitting a message will load it.')
                self.process = SimpleNamespace(pid=None, poll=lambda: None)
                self.loaded_config, self.state = config, 'ready'
                self.loads += 1
            self.active += 1
            if inference:
                self.inferences += 1
            return self.process

    def cleanup(self):
        self.process, self.loaded_config, self.state = None, None, 'unloaded'


class Stack(contextlib.ExitStack):
    def __init__(self, backend_port, port=0, state=None, flash_listed=True, environ=None, **config):
        super().__init__()
        self.tmp = Path(self.enter_context(tempfile.TemporaryDirectory()))
        self.config_path = self.tmp / 'runtime.json'
        self.config_path.write_text(json.dumps(runtime_value(**config)))
        self.state = Path(state) if state else self.tmp / 'harness-prefixes'
        self.enter_context(patch.object(controller, 'CONFIG', self.config_path))
        self.enter_context(patch.object(controller, 'BACKEND_PORT', backend_port))
        if hasattr(controller, 'HARNESS_STATE'):
            self.enter_context(patch.object(controller, 'HARNESS_STATE', ''))
        self.enter_context(patch.dict(os.environ, {'LOCAL_AI_ENABLE_OXE': '1', **(environ or {})}))
        if 'LOCAL_AI_API_KEY' not in (environ or {}):
            os.environ.pop('LOCAL_AI_API_KEY', None)  # restored by patch.dict on exit
        if flash_listed:  # listing only; Flash requests are not served by this stack
            self.enter_context(patch.object(bigmoe_runtime, 'eligible', lambda config: True))
        self.manager = StackManager()
        if HarnessRegistry is not None:
            self.manager.harness_registry = HarnessRegistry(self.state)
        self.server = controller.Server(('127.0.0.1', port), controller.Handler)
        self.server.manager = self.manager
        self.port = self.server.server_port
        self.base = 'http://127.0.0.1:%d' % self.port
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.callback(self._stop)

    def _stop(self):
        self.server.shutdown()
        self.server.server_close()
        self.manager.close()
        self.thread.join(timeout=2)

    def set_config(self, **changes):
        value = json.loads(self.config_path.read_text())
        value.update(changes)
        self.config_path.write_text(json.dumps(value))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--backend-port', type=int, help='a running oxe-server --engine stub (default: the Python fake engine)')
    parser.add_argument('--port', type=int, default=18292)
    parser.add_argument('--learning', action='store_true', help='start with harness learning on')
    parser.add_argument('--cors-origin', action='append', default=[])
    args = parser.parse_args()
    fake = None if args.backend_port else FakeEngine()
    extra = {'harness_learning': {'enabled': True}} if args.learning else {}
    if args.cors_origin:
        extra['openai_api'] = {'cors_origins': args.cors_origin}
    environ = {k: os.environ[k] for k in ('LOCAL_AI_API_KEY',) if os.environ.get(k)}
    with Stack(args.backend_port or fake.port, port=args.port, environ=environ, **extra) as stack:
        print('controller stack on %s/v1 (backend port %s)' % (stack.base, args.backend_port or fake.port), flush=True)
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            pass
    if fake:
        fake.close()


if __name__ == '__main__':
    main()
