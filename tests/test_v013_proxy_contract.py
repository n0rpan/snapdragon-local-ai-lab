"""v0.1.3 contract for the proxy split: `ProxyRoutes.proxy_request` byte for byte.

Taken on the unsplit proxy (claude/v013-split2 at 878a492), before proxy_request is cut into
request-prep, forwarding/streaming and ledger/finish. Each test drives the real controller.Handler
over loopback and records one transcript:
- `engine`: every request the engine got, as raw bytes (request line, every header in the order
  sent, body), the engine port masked as ENGINE;
- `client`: the exact reply bytes the client got (only the Server and Date values masked) and
  whether the controller then closed the connection;
- `events`: every call the proxy made, in order, on the Manager (config, eligible, begin_queued,
  prepare_prefix, current_prefix_key, cache and session-hint writes, harness lock, registry and
  compat counters, record_engine_death, end), on the ledger entry (request_ledger.Entry: label,
  label_turn, row writes, admitted_now, observe, finish), on the disconnect watcher (start, stop),
  and the engine log lines.
A transcript must equal GOLDEN[test name]; a mismatch prints the whole actual transcript.
Model-free and loopback only: a raw-socket fake engine and a recording fake Manager. Names are
patched through the controller namespace and module objects, as the other controller tests do.
"""
import os as _os, tempfile as _tempfile
_os.environ.setdefault('LOCAL_AI_LEDGER_DIR', _tempfile.mkdtemp(prefix='local-ai-test-ledger-'))  # never the real request ledger
import base64
import gzip
import json
import os
from pathlib import Path
import re
import socket
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zlib

from snapdragon_ai_lab.runtime import (bigmoe_runtime, controller, explicit_route, lan_share, openai_api,
                                       oxe_runtime, request_ledger, tool_output_cap)
from snapdragon_ai_lab.runtime.client_disconnect import ClientDisconnect

TIMEOUT = 10
CHAT = '/v1/chat/completions'
LLAMA = {'model_alias': 'synthetic-loopback', 'backend_kind': 'llama'}
OXE = {'model_alias': 'synthetic-oxe', 'backend_kind': oxe_runtime.KIND}
FLASH = {'model_alias': 'synthetic-flash&1', 'backend_kind': bigmoe_runtime.KIND}
BETA = {'model_alias': 'ornith-35b-a3b-local', 'backend_kind': 'llama'}  # classic beta + context mode model
OPENCODE_SYSTEM = 'You are an AI agent running in OpenCode, a test fixture.'


def plain(value):
    """JSON-safe copy of a recorded argument (tuples become lists, objects their type name)."""
    return json.loads(json.dumps(value, default=lambda o: '<%s>' % type(o).__name__))


def gz(data):
    """gzip bytes that do not depend on the Python version (the OS byte of the header is fixed)."""
    out = bytearray(gzip.compress(data, mtime=0))
    out[9] = 3
    return bytes(out)


def head(status, *headers):
    return ('HTTP/1.1 %s\r\n' % status + ''.join('%s: %s\r\n' % pair for pair in headers) + '\r\n').encode()


class RawEngine:
    """A loopback engine on a raw socket: records each request byte for byte, then plays `script`
    (bytes are sent, callables get the connection) and closes."""

    def __init__(self):
        self.requests = []
        self.script = []
        self.sock = socket.socket()
        self.sock.bind(('127.0.0.1', 0))
        self.sock.listen(16)
        self.sock.settimeout(0.1)
        self.port = self.sock.getsockname()[1]
        self.stopped = threading.Event()
        threading.Thread(target=self._accept, daemon=True).start()

    def close(self):
        self.stopped.set()
        self.sock.close()

    def _accept(self):
        while not self.stopped.is_set():
            try:
                conn, _ = self.sock.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            threading.Thread(target=self._serve, args=(conn,), daemon=True).start()

    def _serve(self, conn):
        conn.settimeout(TIMEOUT)
        try:
            data = b''
            while b'\r\n\r\n' not in data:
                piece = conn.recv(65536)
                if not piece:
                    return  # connected, no request (a cancelled shared request)
                data += piece
            raw_head, _, body = data.partition(b'\r\n\r\n')
            found = re.search(rb'(?im)^content-length:\s*(\d+)\s*$', raw_head)
            length = int(found.group(1)) if found else 0
            while len(body) < length:
                piece = conn.recv(65536)
                if not piece:
                    break
                body += piece
            self.requests.append([raw_head.replace(str(self.port).encode(), b'ENGINE').decode('latin-1'),
                                  body.decode('latin-1')])
            for step in self.script:
                if callable(step):
                    step(conn)
                else:
                    conn.sendall(step)
        except OSError:
            pass
        finally:
            try:
                conn.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            conn.close()


class ProxyContract(unittest.TestCase):
    maxDiff = None

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config_file = self.root / 'runtime.json'
        self.config_file.write_text('{}')
        self.events = []
        self.exit_code = None
        self.finish_error = None
        self.admitted_error = None
        self.observed = threading.Condition()
        self.observed_count = 0
        self.handled = threading.Condition()
        self.handled_count = 0
        self.engine = RawEngine()
        self.addCleanup(self.engine.close)
        case = self

        class Process:
            pid = 4242
            _local_ai_api_key = None

            def poll(self):
                return case.exit_code

        self.process = Process()
        self.manager = FakeManager(self, dict(LLAMA))
        self.process._local_ai_engine_config = self.manager.loaded_config

        class Entry:
            session = 'ses-hint-1'

            def __init__(self, path, payload, user_agent=None, model_alias=None, headers=None, origin=None, client=None):
                case.ev('Entry', path, plain(payload), user_agent, model_alias, headers is not None, origin, client)
                self.row = Row()

            def label(self, harness=None, prefix_key=None):
                case.ev('led.label', harness, prefix_key)

            def label_turn(self, turn):
                case.ev('led.label_turn', plain(turn))

            def admitted_now(self, *times):
                case.ev('led.admitted_now', list(times))
                if case.admitted_error is not None:
                    raise case.admitted_error

            def observe(self, chunk):
                case.ev('led.observe', chunk.decode('latin-1'))
                with case.observed:
                    case.observed_count += 1
                    case.observed.notify_all()

            def finish(self, status, error=None):
                case.ev('led.finish', status, error)
                if case.finish_error is not None:
                    raise case.finish_error

        class Row(dict):
            def __setitem__(self, key, value):
                case.ev('led.row', key, plain(value))
                super().__setitem__(key, value)

            def update(self, other):
                case.ev('led.row.update', plain(other))
                super().update(other)

        class Watcher(ClientDisconnect):
            def start(self):
                case.ev('watcher.start')
                return super().start()

            def stop(self):
                case.ev('watcher.stop')
                super().stop()

        class CountingHandler(controller.Handler):
            def handle_one_request(self):
                try:
                    super().handle_one_request()
                finally:
                    with case.handled:
                        case.handled_count += 1
                        case.handled.notify_all()

        class QuietServer(controller.Server):
            def handle_error(self, request, client_address):
                pass  # the F80 tests raise out of the handler on purpose

        self.proxy = QuietServer(('127.0.0.1', 0), CountingHandler)
        self.proxy.manager = self.manager
        threading.Thread(target=self.proxy.serve_forever, daemon=True).start()
        self.addCleanup(self.proxy.server_close)
        self.addCleanup(self.proxy.shutdown)

        real_completion_request = openai_api.completion_request

        def fixed_completion_request(payload):
            native, meta = real_completion_request(payload)
            meta['id'], meta['created'] = 'cmpl-fixed', 1700000000  # the only clock/random fields
            return native, meta

        env = patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        for name in (openai_api.KEY_ENV, explicit_route.ENABLE_ENV, tool_output_cap.ENV):
            os.environ.pop(name, None)
        for item in (patch.object(controller, 'CONFIG', self.config_file),
                     patch.object(controller, 'BACKEND_PORT', self.engine.port),
                     patch.object(controller, 'FLASH_BACKEND_PORT', 47999),
                     patch.object(controller, 'ClientDisconnect', Watcher),
                     patch.object(controller, 'thermal_sample', return_value={'thermal_c': 41.5}),
                     patch.object(request_ledger, 'Entry', Entry),
                     patch.object(openai_api, 'completion_request', fixed_completion_request)):
            item.start()
            self.addCleanup(item.stop)

    # -------------------------------------------------------------- recording helpers
    def use(self, config):
        """The loaded (and default) model profile for this test."""
        self.manager.target = self.manager.loaded_config = dict(config)
        self.process._local_ai_engine_config = self.manager.loaded_config

    def ev(self, *event):
        self.events.append(plain(list(event)))

    def wait_observed(self, count):
        def step(_conn):
            with self.observed:
                self.assertTrue(self.observed.wait_for(lambda: self.observed_count >= count, TIMEOUT))
        return step

    def set_exit(self, code):
        def step(_conn):
            self.exit_code = code
        return step

    def share(self, max_tokens=64):
        """Requests on this server now arrive as a LAN-shared request (the gate itself is lan_share's)."""
        class Share:
            opts = {'max_tokens': max_tokens}

            def handle(self, handler, forward):
                handler.share_client = 'peer-1'
                return forward()
        self.proxy.share_listener = SimpleNamespace(controller=Share())

    def exchange(self, method, path, body=b'', headers=(), length=True, until=None):
        """One raw request; the reply bytes and whether the controller closed the connection."""
        with self.handled:
            wanted = self.handled_count + 1
        sock = socket.create_connection(('127.0.0.1', self.proxy.server_port), timeout=TIMEOUT)
        self.addCleanup(sock.close)
        lines = ['%s %s HTTP/1.1' % (method, path)]
        if not any(name.lower() == 'host' for name, _ in headers):
            lines.append('Host: 127.0.0.1:%d' % self.proxy.server_port)
        lines += ['%s: %s' % pair for pair in headers]
        if length and (body or method == 'POST'):
            lines.append('Content-Length: %d' % len(body))
        sock.sendall(('\r\n'.join(lines) + '\r\n\r\n').encode('latin-1') + body)
        data = b''
        if until is not None:  # read up to `until`, then go away (client disconnect)
            while until not in data:
                piece = sock.recv(65536)
                if not piece:
                    break
                data += piece
            sock.close()
            closed = 'client'
        with self.handled:
            self.assertTrue(self.handled.wait_for(lambda: self.handled_count >= wanted, TIMEOUT), 'handler hung')
        if until is None:
            sock.settimeout(1.0)
            closed = False
            try:
                while True:
                    piece = sock.recv(65536)
                    if not piece:
                        closed = True
                        break
                    data += piece
            except socket.timeout:
                pass
        text = re.sub(r'(?m)^(Server|Date): [^\r\n]*', r'\1: *', data.decode('latin-1'))
        # the reason phrase comes from http.server's table, which differs between Python versions
        text = re.sub(r'^(HTTP/1\.[01] \d{3}) [^\r\n]*', r'\1 *', text)
        return text, closed

    def run_case(self, method, path, body=b'', headers=(), **kwargs):
        client, closed = self.exchange(method, path, body, headers, **kwargs)
        return {'engine': list(self.engine.requests), 'client': client, 'closed': closed, 'events': list(self.events)}

    def check(self, transcript, name=None):
        name = name or self._testMethodName
        actual = plain(transcript)
        for part in ([actual] if 'events' in actual else actual.values()):
            names = [event[0] for event in part['events']]
            admitted = 'begin_queued' in names and 'begin_queued.raised' not in names
            # manager.end: exactly once after admission, never without it (its place is in GOLDEN)
            self.assertEqual(names.count('end'), 1 if admitted else 0, names)
        if GOLDEN.get(name) != actual:
            # readable, and once more as base64 (CI log masking can rewrite header values in the readable copy)
            dump = json.dumps(actual, sort_keys=True)
            self.fail('GOLDEN-ACTUAL %s %s\nGOLDEN-B64 %s %s' % (
                name, dump, name, base64.b64encode(zlib.compress(dump.encode())).decode()))

    def chat(self, **extra):
        return json.dumps({'messages': [{'role': 'user', 'content': 'hi'}], **extra}).encode()

    def json_engine(self, status='200 OK', body=b'{"ok":true}', *extra):
        self.engine.script = [head(status, ('Content-Type', 'application/json'),
                                   ('Content-Length', len(body)), *extra) + body]

    def sse_engine(self, *pieces):
        """An SSE reply, one piece per read on the controller side (each waits for its observe)."""
        script = [head('200 OK', ('Content-Type', 'text/event-stream'), ('Connection', 'close')) + pieces[0]]
        for index, piece in enumerate(pieces[1:], 1):
            script += [self.wait_observed(index), piece]
        self.engine.script = script

    # ============================================================== request prep
    def test_passthrough_keeps_the_body_bytes_and_filters_headers(self):
        self.json_engine('200 OK', b'{"content":"x"}', ('Access-Control-Allow-Origin', '*'),
                         ('Keep-Alive', 'timeout=5'), ('X-Engine', 'e1'), ('Set-Cookie', 's=1'))
        body = b'{"prompt":  "hi",\n "n_predict": 4}'
        self.check(self.run_case('POST', '/completion', body, (
            ('Content-Type', 'application/json'), ('User-Agent', 'contract/1'),
            ('Origin', 'http://127.0.0.1:%d' % self.proxy.server_port), ('Sec-Fetch-Site', 'same-origin'),
            ('Sec-Fetch-Mode', 'cors'), ('X-Oxe-Session', 'forged'), ('X-Oxe-Turn-Kind', 'compaction'),
            ('X-Oxe-State-Policy', 'keep'), ('Keep-Alive', '5'), ('Connection', 'keep-alive'), ('TE', 'trailers'),
            ('Proxy-Authorization', 'Basic eDp5'), ('Accept-Encoding', 'br'), ('Authorization', 'Bearer cli'),
            ('X-Api-Key', 'ck'), ('Cookie', 'a=b'), ('X-Custom', '1'), ('X-Local-AI-Origin', 'lab'))))

    def test_local_api_key_is_checked_and_never_forwarded(self):
        key_file = self.root / 'key'
        key_file.write_text('lk-1\n')
        self.config_file.write_text(json.dumps({'openai_api': {'api_key_file': str(key_file)}}))
        self.json_engine()
        self.check(self.run_case('POST', '/completion', b'{"prompt":"k"}', (
            ('Authorization', 'Bearer lk-1'), ('X-Api-Key', 'lk-1'), ('Api-Key', 'lk-1'), ('X-Other', 'o'))))

    def test_model_alias_is_rewritten(self):
        self.json_engine()
        self.check(self.run_case('POST', '/completion', b'{"model": "qwen-local", "prompt": "a"}',
                                 (('Content-Type', 'application/json'),)))

    def test_gzip_body_passes_through_when_nothing_rewrites_it(self):
        self.json_engine()
        body = gz(b'{"prompt":"gz"}')
        self.check(self.run_case('POST', '/completion', body, (('Content-Encoding', 'gzip'),)))

    def test_gzip_body_rewritten_drops_content_encoding(self):
        self.json_engine()
        body = gz(b'{"model":"qwen-local","prompt":"gz"}')
        self.check(self.run_case('POST', '/completion', body, (('Content-Encoding', 'GZIP'),)))

    def test_corrupt_gzip_body_is_400(self):
        self.check(self.run_case('POST', '/completion', b'not gzip', (('Content-Encoding', 'gzip'),)))

    def test_gzip_bomb_is_400(self):
        body = gz(b' ' * (16 * 1024**2 + 1))
        self.check(self.run_case('POST', '/completion', body, (('Content-Encoding', 'gzip'),)))

    def test_invalid_json_is_400(self):
        self.check(self.run_case('POST', '/completion', b'{"prompt":'))

    def test_non_string_model_is_400(self):
        self.check(self.run_case('POST', '/completion', b'{"model":7,"prompt":"a"}'))

    def test_unknown_model_is_404_model_not_found(self):
        self.check(self.run_case('POST', CHAT, self.chat(model='missing-model')))

    def test_no_model_configured_is_503(self):
        self.manager.loaded_config = None
        self.manager.config_error = 'No model is selected yet.'
        self.check(self.run_case('POST', CHAT, self.chat()))

    def test_bad_content_length_is_400(self):
        self.check(self.run_case('POST', '/completion', b'', (('Content-Length', 'abc'),), length=False))

    def test_negative_content_length_is_413(self):
        self.check(self.run_case('POST', '/completion', b'', (('Content-Length', '-1'),), length=False))

    def test_huge_content_length_is_413(self):
        self.check(self.run_case('POST', '/completion', b'', (('Content-Length', str(16 * 1024 * 1024 + 1)),),
                                 length=False))

    def test_transfer_encoding_is_411(self):
        self.check(self.run_case('POST', '/completion', b'0\r\n\r\n', (('Transfer-Encoding', 'chunked'),),
                                 length=False))

    def test_explicit_route_rewrites_to_the_routed_model(self):
        os.environ[explicit_route.ENABLE_ENV] = '1'
        self.json_engine()
        self.check(self.run_case('POST', CHAT, self.chat(model=explicit_route.ALIAS, local_route='code')))

    def test_explicit_route_disabled_is_400(self):
        self.check(self.run_case('POST', CHAT, self.chat(model=explicit_route.ALIAS, local_route='code')))

    def test_local_route_without_auto_is_dropped(self):
        self.json_engine()
        self.check(self.run_case('POST', CHAT, self.chat(local_route='code')))

    def test_classic_beta_defaults_and_context_mode_row(self):
        self.use(BETA)
        self.manager.runtime_data = {'context_modes': {'models': {BETA['model_alias']: 'long'}}}
        self.json_engine()
        self.check(self.run_case('POST', CHAT, self.chat(model=BETA['model_alias'])))

    def test_request_defaults_seam_is_read_at_call_time(self):
        def defaults(target, payload, path):
            self.ev('request_defaults', target['model_alias'], path)
            return {**payload, 'seam': 1}
        self.json_engine()
        with patch.object(controller, 'request_defaults', defaults):
            self.check(self.run_case('POST', '/completion', b'{"prompt":"s"}'))

    def test_tool_output_cap_off_forwards_identical_bytes(self):
        self.json_engine()
        body = (b'{"messages": [{"role": "user", "content": "x"}, {"role": "tool", "tool_call_id": "c1", '
                b'"content": "' + b'line\\n' * 400 + b'"}],  "tools": []}')
        self.check(self.run_case('POST', CHAT, body))

    def test_tool_output_cap_on_without_changes_adds_no_row(self):
        os.environ[tool_output_cap.ENV] = 'on'
        os.environ[tool_output_cap.ENV + '_STORE'] = str(self.root / 'cap')
        self.json_engine()
        self.check(self.run_case('POST', CHAT, b'{"messages": [{"role": "user", "content": "x"}]}'))

    def test_tool_output_cap_rewrite_and_row(self):
        def cap(payload, headers):
            self.ev('cap_request', plain(payload), headers.get('X-Session-Id'))
            return {**payload, 'capped': True}, {'capped': 1, 'reused': 0, 'hidden_tokens': 900, 'reads': 0}
        self.json_engine()
        with patch.object(tool_output_cap, 'cap_request', cap):
            self.check(self.run_case('POST', CHAT, self.chat(), (('X-Session-Id', 'ses-a'),)))

    def test_tool_output_cap_failure_forwards_untouched(self):
        def cap(payload, headers):
            raise RuntimeError('store broke')
        self.json_engine()
        with patch.object(tool_output_cap, 'cap_request', cap):
            self.check(self.run_case('POST', CHAT, b'{"messages":  [{"role": "user", "content": "x"}]}'))

    def test_helper_request_is_not_inference(self):
        self.json_engine('200 OK', b'{"tokens":[1]}')
        self.check(self.run_case('POST', '/tokenize', b'{"model":"qwen-local","content":"a"}'))

    def test_oxe_chat_normalised_with_harness_turn_and_launch_token(self):
        self.use(OXE)
        self.process._local_ai_api_key = 'lt-1'
        self.sse_engine(b'data: {"choices":[{"delta":{"content":"a"}}]}\n\n',
                        b'data: {"timings":{"cache_n":3,"prompt_n":5}}\n\ndata: [DONE]\n\n')
        body = json.dumps({'model': 'synthetic-oxe', 'n': 2, 'temperature': None, 'stream': True, 'messages': [
            {'role': 'system', 'content': OPENCODE_SYSTEM}, {'role': 'user', 'content': 'hi'}]}).encode()
        self.check(self.run_case('POST', CHAT, body, (
            ('User-Agent', 'opencode/2'), ('Authorization', 'Bearer cli'), ('X-Api-Key', 'ck'), ('Cookie', 'a=b'),
            ('Origin', 'http://127.0.0.1:%d' % self.proxy.server_port), ('X-Session-Id', 'ses-b'))))

    def test_oxe_completions_non_stream_shim(self):
        self.use(OXE)
        self.json_engine('200 OK', b'{"content":"lo","stop_type":"eos","tokens_evaluated":2,"tokens_predicted":1}')
        self.check(self.run_case('POST', '/v1/completions', b'{"prompt":"hel","max_tokens":3}'))

    def test_oxe_completions_stream_shim(self):
        self.use(OXE)
        self.sse_engine(b'data: {"content":"a","stop":false}\n\n', b'data: {"content":"b","st',
                        b'op":true,"timings":{"cache_n":1}}\n\n', b': bye')
        self.check(self.run_case('POST', '/v1/completions', b'{"prompt":"p","stream":true,"echo":true}'))

    def test_oxe_completions_unsupported_is_400(self):
        self.use(OXE)
        self.check(self.run_case('POST', '/v1/completions', b'{"prompt":"p","suffix":"s"}'))

    def test_oxe_unsupported_endpoint_is_404(self):
        self.use(OXE)
        self.check(self.run_case('POST', '/v1/embeddings', b'{"input":"a"}'))

    def test_oxe_backend_error_clears_prefix_and_harness_key(self):
        self.use(OXE)
        self.json_engine('500 Internal Server Error', b'{"error":"boom"}')
        body = json.dumps({'messages': [{'role': 'system', 'content': OPENCODE_SYSTEM},
                                        {'role': 'user', 'content': 'x'}]}).encode()
        self.check(self.run_case('POST', CHAT, body))

    # -------------------------------------------------------------- shared (LAN) requests
    def test_remote_llama_payload_ephemeral_and_no_prefix(self):
        self.share()
        self.json_engine()
        body = json.dumps({'messages': [{'role': 'user', 'content': 'hi'}], 'max_tokens': 999, 'slot_id': 3,
                           'cache_prompt': True, 'stream': False}).encode()
        self.check(self.run_case('POST', CHAT, body, (('Host', 'peer.example:8092'), ('X-Oxe-Session', 'forged'))))

    def test_remote_oxe_has_no_harness_or_session_headers(self):
        self.share()
        self.use(OXE)
        self.process._local_ai_api_key = 'lt-1'
        self.json_engine()
        body = json.dumps({'messages': [{'role': 'system', 'content': OPENCODE_SYSTEM},
                                        {'role': 'user', 'content': 'hi'}], 'max_tokens': 8}).encode()
        self.check(self.run_case('POST', CHAT, body))

    def test_remote_without_body_is_400(self):
        self.share()
        self.check(self.run_case('POST', CHAT, b''))

    def test_remote_start_failure_hides_the_cause(self):
        self.share()
        self.manager.begin_error = RuntimeError('engine exploded at a private path')
        self.check(self.run_case('POST', CHAT, self.chat()))

    def test_remote_preempted_before_the_response(self):
        self.share()
        self.manager.on_begin = lambda kwargs: kwargs['preempt'].cancel()
        self.json_engine()
        self.check(self.run_case('POST', CHAT, self.chat()))

    def test_remote_preempted_in_a_stream(self):
        self.share()
        self.engine.script = [head('200 OK', ('Content-Type', 'text/event-stream'), ('Connection', 'close'))
                              + b'data: {"n":1}\n\n', self.wait_observed(1),
                              lambda _conn: self.manager.preempt.cancel()]
        self.check(self.run_case('POST', CHAT, self.chat(stream=True)))

    def test_remote_preempted_in_a_json_reply(self):
        self.share()
        self.engine.script = [head('200 OK', ('Content-Type', 'application/json'), ('Connection', 'close'))
                              + b'{"partial":', self.wait_observed(1),
                              lambda _conn: self.manager.preempt.cancel()]
        self.check(self.run_case('POST', CHAT, self.chat()))

    # -------------------------------------------------------------- Flash (bigmoe)
    def flash(self, runtime=True, chat_body=None):
        self.use(FLASH)

        def normalize_chat(target, payload, port):
            self.ev('normalize_chat', target['model_alias'], plain(payload), port)
            if payload.get('messages') == 'bad':
                raise ValueError('bad')
            return {'model': payload.get('native', 'flash-native'), 'messages': payload['messages'],
                    'max_tokens': payload.get('max_tokens')}
        item = patch.object(bigmoe_runtime, 'normalize_chat', normalize_chat)
        item.start()
        self.addCleanup(item.stop)
        if runtime:
            def shaped(payload):
                self.ev('chat_body', plain(payload))
                if chat_body is not None:
                    return chat_body(payload)
                return {**payload, 'model': 'flash-native', 'shaped': True}
            self.manager.backend_runtime = SimpleNamespace(
                spec=SimpleNamespace(port=self.engine.port, backend_model_id='flash-native'),
                chat_body=shaped, headers=lambda: {'Authorization': 'Bearer rk'})

    def test_flash_page(self):
        self.flash()
        self.check(self.run_case('GET', '/'))

    def test_flash_page_unqualified_is_503(self):
        self.flash()
        self.manager.eligible_value = False
        self.check(self.run_case('GET', '/'))

    def test_flash_other_route_is_403(self):
        self.flash()
        self.check(self.run_case('GET', '/props'))

    def test_flash_normalize_failure_is_400(self):
        self.flash()
        self.check(self.run_case('POST', CHAT, b'{"messages":"bad"}'))

    def test_flash_chat_headers_and_body(self):
        self.flash()
        self.json_engine('200 OK', b'{"ok":1}', ('Set-Cookie', 's=1'), ('Authorization', 'x'), ('X-Api-Key', 'y'),
                         ('X-Kept', 'k'))
        body = json.dumps({'messages': [{'role': 'user', 'content': 'hi'}], 'max_tokens': 4096,
                           'max_completion_tokens': 5}).encode()
        self.check(self.run_case('POST', CHAT, body, (
            ('Origin', 'http://127.0.0.1:%d' % self.proxy.server_port), ('Authorization', 'Bearer cli'),
            ('Cookie', 'a=b'), ('Content-Type', 'text/plain'), ('X-Oxe-Session', 'kept-for-flash'))))

    def test_flash_chat_route_with_a_query_is_403(self):
        self.flash()
        self.json_engine()
        self.check(self.run_case('POST', CHAT + '?trace=1', self.chat()))

    def test_flash_chat_body_reshaped_by_the_runtime(self):
        self.flash()
        self.json_engine()
        self.check(self.run_case('POST', CHAT, self.chat(native='other-model')))

    def test_flash_bad_chat_body_is_400_after_admission(self):
        def broken(payload):
            raise KeyError('x')
        self.flash(chat_body=broken)
        self.check(self.run_case('POST', CHAT, self.chat(native='other-model')))

    # -------------------------------------------------------------- admission replies
    def test_cached_get_is_answered_without_admission(self):
        self.manager.cache.data['/props'] = ('application/json', b'{"cached":1}')
        self.check(self.run_case('GET', '/props'))

    def test_get_with_a_query_is_neither_served_from_nor_stored_in_the_cache(self):
        self.manager.cache.data['/props'] = ('application/json', b'{"cached":1}')
        self.json_engine('200 OK', b'{"props":"q"}')
        self.check(self.run_case('GET', '/props?x=1'))

    def test_model_busy_is_409(self):
        self.manager.begin_error = controller.ModelBusy('Another model is loaded.')
        self.check(self.run_case('POST', CHAT, self.chat()))

    def test_busy_is_503(self):
        self.manager.begin_error = controller.Unavailable('A request is running; try again.')
        self.check(self.run_case('POST', CHAT, self.chat()))

    def test_unavailable_is_503(self):
        self.manager.begin_error = controller.Unavailable('Model files are missing.')
        self.check(self.run_case('POST', CHAT, self.chat()))

    def test_unavailable_root_page(self):
        self.manager.begin_error = controller.Unavailable('Setup is <pending>.')
        self.check(self.run_case('GET', '/'))

    def test_local_start_failure_names_the_cause(self):
        self.manager.begin_error = RuntimeError('engine exploded\nsecond line')
        self.check(self.run_case('POST', CHAT, self.chat()))

    def test_controller_error_after_admission_is_500(self):
        self.manager.prepare_error = RuntimeError('prefix store broke')
        self.check(self.run_case('POST', CHAT, self.chat()))

    # ============================================================== forwarding / streaming
    def test_sse_stream_in_chunks(self):
        self.sse_engine(b'data: {"n":1}\n\n', b'data: {"n":2}\n\n', b'data: [DONE]\n\n')
        self.check(self.run_case('POST', CHAT, self.chat(stream=True)))

    def test_cors_reply_headers_replace_the_engines(self):
        self.config_file.write_text(json.dumps({'openai_api': {'cors_origins': ['http://app.example']}}))
        self.json_engine('200 OK', b'{"ok":1}', ('Access-Control-Allow-Origin', '*'),
                         ('access-control-allow-headers', 'x'), ('Transfer-Encoding', 'identity'))
        self.check(self.run_case('POST', CHAT, self.chat(), (('Origin', 'http://app.example'),)))

    def test_gzip_ui_asset_is_decoded_cached_and_served_from_cache(self):
        data = gz(b'<html>ui</html>')
        self.engine.script = [head('200 OK', ('Content-Type', 'text/html'), ('Content-Encoding', 'gzip'),
                                   ('Content-Length', len(data))) + data]
        first = self.run_case('GET', '/index.html', headers=(('Accept-Encoding', 'br'),))
        self.events.clear()
        second = self.run_case('GET', '/index.html')
        self.check({'first': first, 'second': second})

    def test_corrupt_gzip_reply_is_502(self):
        self.engine.script = [head('200 OK', ('Content-Type', 'text/javascript'), ('Content-Encoding', 'gzip'),
                                   ('Content-Length', 4)) + b'nope']
        self.check(self.run_case('GET', '/app.js'))

    def test_cacheable_get_is_cached(self):
        self.json_engine('200 OK', b'{"props":1}')
        self.check(self.run_case('GET', '/props'))

    def test_uncached_get_streams(self):
        self.json_engine('200 OK', b'{"slots":[]}')
        self.check(self.run_case('GET', '/slots'))

    def test_head(self):
        self.engine.script = [head('200 OK', ('Content-Type', 'application/json'), ('Content-Length', 11))]
        self.check(self.run_case('HEAD', '/props'))

    def test_backend_4xx_passes_through(self):
        self.json_engine('400 Bad Request', b'{"error":"no"}')
        self.check(self.run_case('POST', CHAT, self.chat()))

    def test_backend_dies_before_headers(self):
        self.engine.script = [lambda _conn: None]
        self.check(self.run_case('POST', CHAT, self.chat()))

    def test_backend_process_dies_before_headers(self):
        self.engine.script = [self.set_exit(1)]
        self.check(self.run_case('POST', CHAT, self.chat()))

    def test_backend_dies_mid_stream(self):
        self.engine.script = [head('200 OK', ('Content-Type', 'text/event-stream'), ('Connection', 'close'))
                              + b'data: {"n":1}\n\n', self.wait_observed(1), self.set_exit(-9)]
        self.check(self.run_case('POST', CHAT, self.chat(stream=True)))

    def test_backend_connection_refused_is_503(self):
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', 0))
            port = probe.getsockname()[1]
        with patch.object(controller, 'BACKEND_PORT', port):
            self.check(self.run_case('POST', CHAT, self.chat()))

    def test_client_disconnect_mid_stream(self):
        def until_upstream_closed(conn):
            while conn.recv(65536):
                pass
        self.engine.script = [head('200 OK', ('Content-Type', 'text/event-stream'), ('Connection', 'close'))
                              + b'data: {"n":1}\n\n', self.wait_observed(1), until_upstream_closed]
        self.check(self.run_case('POST', CHAT, self.chat(stream=True), until=b'data: {"n":1}\n\n\r\n'))

    # ============================================================== ledger / finish (F80)
    def test_ledger_finish_raising_still_ends(self):
        self.finish_error = RuntimeError('ledger disk full')
        self.json_engine()
        self.check(self.run_case('POST', CHAT, self.chat()))

    def test_ledger_admitted_now_raising_still_serves_and_ends(self):
        # the slot was taken by begin_queued: a ledger error after it must not leak the slot
        self.admitted_error = RuntimeError('ledger clock broke')
        self.json_engine()
        self.check(self.run_case('POST', CHAT, self.chat()))

    def test_engine_log_raising_still_ends(self):
        self.manager.log_error = OSError('log disk full')
        self.engine.script = [lambda _conn: None]
        self.check(self.run_case('POST', CHAT, self.chat()))


class FakeManager:
    """The Manager surface proxy_request uses; every call and write is recorded in order."""

    def __init__(self, case, target):
        self._case = case
        self.target = target
        self.loaded_config = target
        self.config_error = None
        self.backend_runtime = None
        self.eligible_value = True
        self.begin_error = None
        self.on_begin = None
        self.prepare_error = None
        self.log_error = None
        self.preempt = None
        self.runtime_data = {}
        self.admission_times = (1.5, 2.5)
        self._prefix_key = 'pk-0'
        self.cache = RecordingDict(case, 'cache')
        self.session_hints = RecordingDict(case, 'session_hints')
        self.harness_mu = RecordingLock(case)
        self.compat_stats = SimpleNamespace(update=lambda notes: case.ev('compat_stats.update', list(notes)))
        self.harness_registry = SimpleNamespace(
            observe=lambda payload, harness, target, learning=None, user_agent=None: (
                case.ev('harness.observe', plain(payload), harness, target['model_alias'], plain(learning), user_agent)
                or 'hk-1'),
            complete=lambda key, target, *timings: case.ev('harness.complete', key, target['model_alias'], list(timings)))
        manager = self

        class Log:
            def write(self, line):
                case.ev('engine_log', line)
                if manager.log_error is not None:
                    raise manager.log_error
        self.engine_log_writer = Log()

    @property
    def current_prefix_key(self):
        return self._prefix_key

    @current_prefix_key.setter
    def current_prefix_key(self, value):
        self._case.ev('current_prefix_key', value)
        self._prefix_key = value

    def config(self, alias=None):
        self._case.ev('config', alias)
        if alias is None:
            if self.config_error is not None:
                raise controller.Unavailable(self.config_error)
            return self.target
        known = {'qwen-local': self.target, self.target['model_alias']: self.target,
                 'ornith-35b-a3b-local': dict(BETA)}
        if alias not in known:
            raise controller.Unavailable('Unknown model: %s.' % alias)
        return known[alias]

    def eligible(self, target):
        self._case.ev('eligible', target['model_alias'])
        return self.eligible_value

    def runtime_value(self):
        self._case.ev('runtime_value')
        return self.runtime_data

    def begin_queued(self, **kwargs):
        self.preempt = kwargs.get('preempt')
        self._case.ev('begin_queued', {k: ('<Preempt>' if isinstance(v, lan_share.Preempt) else v)
                                       for k, v in kwargs.items()})
        if self.on_begin is not None:
            self.on_begin(kwargs)
        if self.begin_error is not None:
            self._case.ev('begin_queued.raised', type(self.begin_error).__name__)
            raise self.begin_error
        return self._case.process

    def prepare_prefix(self, path, payload):
        self._case.ev('prepare_prefix', path, plain(payload))
        if self.prepare_error is not None:
            raise self.prepare_error

    def record_engine_death(self, process, config):
        self._case.ev('record_engine_death', process is self._case.process, plain(config))

    def end(self, **kwargs):
        self._case.ev('end', kwargs)


class RecordingDict:
    def __init__(self, case, name):
        self.case, self.name, self.data = case, name, {}

    def __contains__(self, key):
        return key in self.data

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.case.ev(self.name + '.set', key, [part.decode('latin-1') if isinstance(part, bytes) else part
                                               for part in (value if isinstance(value, tuple) else (value,))])
        self.data[key] = value


class RecordingLock:
    def __init__(self, case):
        self.case, self.lock = case, threading.Lock()

    def __enter__(self):
        self.lock.acquire()
        self.case.ev('harness_mu.enter')
        return self

    def __exit__(self, *exc):
        self.case.ev('harness_mu.exit')
        self.lock.release()
        return False


# Recorded on the unsplit proxy (helper CI, two runs identical); see the module docstring.
GOLDEN = {'test_backend_4xx_passes_through': {'client': 'HTTP/1.1 400 *\r\n'
                                               'Server: *\r\n'
                                               'Date: *\r\n'
                                               'Content-Type: application/json\r\n'
                                               'Cache-Control: no-store\r\n'
                                               'Transfer-Encoding: chunked\r\n'
                                               '\r\n'
                                               'e\r\n'
                                               '{"error":"no"}\r\n'
                                               '0\r\n'
                                               '\r\n',
                                     'closed': False,
                                     'engine': [['POST /v1/chat/completions HTTP/1.1\r\n'
                                                 'Accept-Encoding: identity\r\n'
                                                 'Content-Length: 49\r\n'
                                                 'Host: 127.0.0.1:ENGINE',
                                                 '{"messages": [{"role": "user", "content": "hi"}]}']],
                                     'events': [['Entry',
                                                 '/v1/chat/completions',
                                                 {'messages': [{'content': 'hi', 'role': 'user'}]},
                                                 None,
                                                 'synthetic-loopback',
                                                 True,
                                                 None,
                                                 None],
                                                ['led.label', None, None],
                                                ['led.label_turn', None],
                                                ['begin_queued', {'allow_start': True, 'inference': True, 'model_alias': None, 'origin': 'organic'}],
                                                ['led.admitted_now', [1.5, 2.5]],
                                                ['prepare_prefix', '/v1/chat/completions', {'messages': [{'content': 'hi', 'role': 'user'}]}],
                                                ['watcher.start'],
                                                ['current_prefix_key', None],
                                                ['led.observe', '{"error":"no"}'],
                                                ['watcher.stop'],
                                                ['watcher.stop'],
                                                ['led.row.update', {'thermal_c': 41.5}],
                                                ['led.finish', 400, None],
                                                ['end', {'inference': True, 'touch': False}]]},
 'test_backend_connection_refused_is_503': {'client': 'HTTP/1.1 503 *\r\n'
                                                      'Server: *\r\n'
                                                      'Date: *\r\n'
                                                      'Content-Type: application/json\r\n'
                                                      'Content-Length: 118\r\n'
                                                      'Cache-Control: no-store\r\n'
                                                      'Retry-After: 5\r\n'
                                                      '\r\n'
                                                      '{"error": {"message": "Local backend connection failed.", "type": "local_backend_error", "param": null, '
                                                      '"code": null}}',
                                            'closed': False,
                                            'engine': [],
                                            'events': [['Entry',
                                                        '/v1/chat/completions',
                                                        {'messages': [{'content': 'hi', 'role': 'user'}]},
                                                        None,
                                                        'synthetic-loopback',
                                                        True,
                                                        None,
                                                        None],
                                                       ['led.label', None, None],
                                                       ['led.label_turn', None],
                                                       ['begin_queued', {'allow_start': True, 'inference': True, 'model_alias': None, 'origin': 'organic'}],
                                                       ['led.admitted_now', [1.5, 2.5]],
                                                       ['prepare_prefix', '/v1/chat/completions', {'messages': [{'content': 'hi', 'role': 'user'}]}],
                                                       ['current_prefix_key', None],
                                                       ['led.row.update', {'thermal_c': 41.5}],
                                                       ['led.finish', 503, None],
                                                       ['end', {'inference': True, 'touch': False}]]},
 'test_backend_dies_before_headers': {'client': 'HTTP/1.1 502 *\r\n'
                                                'Server: *\r\n'
                                                'Date: *\r\n'
                                                'Content-Type: application/json\r\n'
                                                'Content-Length: 111\r\n'
                                                'Cache-Control: no-store\r\n'
                                                '\r\n'
                                                '{"error": {"message": "Local backend connection closed.", "type": "backend_died", "param": null, "code": '
                                                'null}}',
                                      'closed': False,
                                      'engine': [['POST /v1/chat/completions HTTP/1.1\r\n'
                                                  'Accept-Encoding: identity\r\n'
                                                  'Content-Length: 49\r\n'
                                                  'Host: 127.0.0.1:ENGINE',
                                                  '{"messages": [{"role": "user", "content": "hi"}]}']],
                                      'events': [['Entry',
                                                  '/v1/chat/completions',
                                                  {'messages': [{'content': 'hi', 'role': 'user'}]},
                                                  None,
                                                  'synthetic-loopback',
                                                  True,
                                                  None,
                                                  None],
                                                 ['led.label', None, None],
                                                 ['led.label_turn', None],
                                                 ['begin_queued', {'allow_start': True, 'inference': True, 'model_alias': None, 'origin': 'organic'}],
                                                 ['led.admitted_now', [1.5, 2.5]],
                                                 ['prepare_prefix', '/v1/chat/completions', {'messages': [{'content': 'hi', 'role': 'user'}]}],
                                                 ['watcher.start'],
                                                 ['watcher.stop'],
                                                 ['led.row.update', {'thermal_c': 41.5}],
                                                 ['led.finish', 503, 'backend_died'],
                                                 ['engine_log',
                                                  '{"error": "backend_died", "event": "request_end", "model_alias": "synthetic-loopback", "status": null}\n'],
                                                 ['end', {'inference': True, 'touch': False}]]},
 'test_backend_dies_mid_stream': {'client': 'HTTP/1.1 200 *\r\n'
                                            'Server: *\r\n'
                                            'Date: *\r\n'
                                            'Content-Type: text/event-stream\r\n'
                                            'Cache-Control: no-store\r\n'
                                            'Transfer-Encoding: chunked\r\n'
                                            '\r\n'
                                            'f\r\n'
                                            'data: {"n":1}\n'
                                            '\n'
                                            '\r\n',
                                  'closed': True,
                                  'engine': [['POST /v1/chat/completions HTTP/1.1\r\nAccept-Encoding: identity\r\nContent-Length: 65\r\nHost: 127.0.0.1:ENGINE',
                                              '{"messages": [{"role": "user", "content": "hi"}], "stream": true}']],
                                  'events': [['Entry',
                                              '/v1/chat/completions',
                                              {'messages': [{'content': 'hi', 'role': 'user'}], 'stream': True},
                                              None,
                                              'synthetic-loopback',
                                              True,
                                              None,
                                              None],
                                             ['led.label', None, None],
                                             ['led.label_turn', None],
                                             ['begin_queued', {'allow_start': True, 'inference': True, 'model_alias': None, 'origin': 'organic'}],
                                             ['led.admitted_now', [1.5, 2.5]],
                                             ['prepare_prefix', '/v1/chat/completions', {'messages': [{'content': 'hi', 'role': 'user'}], 'stream': True}],
                                             ['watcher.start'],
                                             ['led.observe', 'data: {"n":1}\n\n'],
                                             ['watcher.stop'],
                                             ['watcher.stop'],
                                             ['record_engine_death', True, {'backend_kind': 'llama', 'model_alias': 'synthetic-loopback'}],
                                             ['led.row.update', {'thermal_c': 41.5}],
                                             ['led.finish', 200, 'backend_died'],
                                             ['engine_log',
                                              '{"error": "backend_died", "event": "request_end", "model_alias": "synthetic-loopback", "status": 200}\n'],
                                             ['end', {'inference': True, 'touch': False}]]},
 'test_backend_process_dies_before_headers': {'client': 'HTTP/1.1 502 *\r\n'
                                                        'Server: *\r\n'
                                                        'Date: *\r\n'
                                                        'Content-Type: application/json\r\n'
                                                        'Content-Length: 111\r\n'
                                                        'Cache-Control: no-store\r\n'
                                                        '\r\n'
                                                        '{"error": {"message": "Local backend connection closed.", "type": "backend_died", "param": null, '
                                                        '"code": null}}',
                                              'closed': False,
                                              'engine': [['POST /v1/chat/completions HTTP/1.1\r\n'
                                                          'Accept-Encoding: identity\r\n'
                                                          'Content-Length: 49\r\n'
                                                          'Host: 127.0.0.1:ENGINE',
                                                          '{"messages": [{"role": "user", "content": "hi"}]}']],
                                              'events': [['Entry',
                                                          '/v1/chat/completions',
                                                          {'messages': [{'content': 'hi', 'role': 'user'}]},
                                                          None,
                                                          'synthetic-loopback',
                                                          True,
                                                          None,
                                                          None],
                                                         ['led.label', None, None],
                                                         ['led.label_turn', None],
                                                         ['begin_queued', {'allow_start': True, 'inference': True, 'model_alias': None, 'origin': 'organic'}],
                                                         ['led.admitted_now', [1.5, 2.5]],
                                                         ['prepare_prefix', '/v1/chat/completions', {'messages': [{'content': 'hi', 'role': 'user'}]}],
                                                         ['watcher.start'],
                                                         ['record_engine_death', True, {'backend_kind': 'llama', 'model_alias': 'synthetic-loopback'}],
                                                         ['watcher.stop'],
                                                         ['record_engine_death', True, {'backend_kind': 'llama', 'model_alias': 'synthetic-loopback'}],
                                                         ['led.row.update', {'thermal_c': 41.5}],
                                                         ['led.finish', 503, 'backend_died'],
                                                         ['engine_log',
                                                          '{"error": "backend_died", "event": "request_end", "model_alias": "synthetic-loopback", "status": '
                                                          'null}\n'],
                                                         ['end', {'inference': True, 'touch': False}]]},
 'test_bad_content_length_is_400': {'client': 'HTTP/1.1 400 *\r\n'
                                              'Server: *\r\n'
                                              'Date: *\r\n'
                                              'Content-Type: application/json\r\n'
                                              'Content-Length: 111\r\n'
                                              'Cache-Control: no-store\r\n'
                                              '\r\n'
                                              '{"error": {"message": "Invalid content length.", "type": "invalid_request_error", "param": null, "code": null}}',
                                    'closed': True,
                                    'engine': [],
                                    'events': []},
 'test_busy_is_503': {'client': 'HTTP/1.1 503 *\r\n'
                                'Server: *\r\n'
                                'Date: *\r\n'
                                'Content-Type: application/json\r\n'
                                'Content-Length: 117\r\n'
                                'Cache-Control: no-store\r\n'
                                'Retry-After: 5\r\n'
                                '\r\n'
                                '{"error": {"message": "A request is running; try again.", "type": "local_backend_busy", "param": null, "code": null}}',
                      'closed': False,
                      'engine': [],
                      'events': [['Entry',
                                  '/v1/chat/completions',
                                  {'messages': [{'content': 'hi', 'role': 'user'}]},
                                  None,
                                  'synthetic-loopback',
                                  True,
                                  None,
                                  None],
                                 ['led.label', None, None],
                                 ['led.label_turn', None],
                                 ['begin_queued', {'allow_start': True, 'inference': True, 'model_alias': None, 'origin': 'organic'}],
                                 ['begin_queued.raised', 'Unavailable'],
                                 ['led.finish', 503, 'busy']]},
 'test_cacheable_get_is_cached': {'client': 'HTTP/1.1 200 *\r\n'
                                            'Server: *\r\n'
                                            'Date: *\r\n'
                                            'Content-Type: application/json\r\n'
                                            'Content-Length: 11\r\n'
                                            'Cache-Control: no-store\r\n'
                                            '\r\n'
                                            '{"props":1}',
                                  'closed': False,
                                  'engine': [['GET /props HTTP/1.1\r\nAccept-Encoding: identity\r\nHost: 127.0.0.1:ENGINE', '']],
                                  'events': [['begin_queued', {'allow_start': False, 'inference': False, 'model_alias': None, 'origin': 'organic'}],
                                             ['cache.set', '/props', ['application/json', '{"props":1}']],
                                             ['end', {'inference': False, 'touch': False}]]},
 'test_cached_get_is_answered_without_admission': {'client': 'HTTP/1.1 200 *\r\n'
                                                             'Server: *\r\n'
                                                             'Date: *\r\n'
                                                             'Content-Type: application/json\r\n'
                                                             'Content-Length: 12\r\n'
                                                             'Cache-Control: no-store\r\n'
                                                             '\r\n'
                                                             '{"cached":1}',
                                                   'closed': False,
                                                   'engine': [],
                                                   'events': []},
 'test_classic_beta_defaults_and_context_mode_row': {'client': 'HTTP/1.1 200 *\r\n'
                                                               'Server: *\r\n'
                                                               'Date: *\r\n'
                                                               'Content-Type: application/json\r\n'
                                                               'Cache-Control: no-store\r\n'
                                                               'Transfer-Encoding: chunked\r\n'
                                                               '\r\n'
                                                               'b\r\n'
                                                               '{"ok":true}\r\n'
                                                               '0\r\n'
                                                               '\r\n',
                                                     'closed': False,
                                                     'engine': [['POST /v1/chat/completions HTTP/1.1\r\n'
                                                                 'Accept-Encoding: identity\r\n'
                                                                 'Content-Length: 194\r\n'
                                                                 'Host: 127.0.0.1:ENGINE',
                                                                 '{"messages": [{"role": "user", "content": "hi"}], "model": "ornith-35b-a3b-local", '
                                                                 '"temperature": 0, "cache_prompt": true, "max_tokens": 2048, "chat_template_kwargs": '
                                                                 '{"enable_thinking": false}}']],
                                                     'events': [['config', 'ornith-35b-a3b-local'],
                                                                ['config', 'ornith-35b-a3b-local'],
                                                                ['Entry',
                                                                 '/v1/chat/completions',
                                                                 {'cache_prompt': True,
                                                                  'chat_template_kwargs': {'enable_thinking': False},
                                                                  'max_tokens': 2048,
                                                                  'messages': [{'content': 'hi', 'role': 'user'}],
                                                                  'model': 'ornith-35b-a3b-local',
                                                                  'temperature': 0},
                                                                 None,
                                                                 'ornith-35b-a3b-local',
                                                                 True,
                                                                 None,
                                                                 None],
                                                                ['led.label', None, None],
                                                                ['led.label_turn', None],
                                                                ['runtime_value'],
                                                                ['led.row', 'mode', 'fast'],
                                                                ['begin_queued',
                                                                 {'allow_start': True,
                                                                  'inference': True,
                                                                  'model_alias': 'ornith-35b-a3b-local',
                                                                  'origin': 'organic'}],
                                                                ['led.admitted_now', [1.5, 2.5]],
                                                                ['prepare_prefix',
                                                                 '/v1/chat/completions',
                                                                 {'cache_prompt': True,
                                                                  'chat_template_kwargs': {'enable_thinking': False},
                                                                  'max_tokens': 2048,
                                                                  'messages': [{'content': 'hi', 'role': 'user'}],
                                                                  'model': 'ornith-35b-a3b-local',
                                                                  'temperature': 0}],
                                                                ['watcher.start'],
                                                                ['led.observe', '{"ok":true}'],
                                                                ['watcher.stop'],
                                                                ['watcher.stop'],
                                                                ['led.row.update', {'thermal_c': 41.5}],
                                                                ['led.finish', 200, None],
                                                                ['end', {'inference': True, 'touch': False}]]},
 'test_client_disconnect_mid_stream': {'client': 'HTTP/1.1 200 *\r\n'
                                                 'Server: *\r\n'
                                                 'Date: *\r\n'
                                                 'Content-Type: text/event-stream\r\n'
                                                 'Cache-Control: no-store\r\n'
                                                 'Transfer-Encoding: chunked\r\n'
                                                 '\r\n'
                                                 'f\r\n'
                                                 'data: {"n":1}\n'
                                                 '\n'
                                                 '\r\n',
                                       'closed': 'client',
                                       'engine': [['POST /v1/chat/completions HTTP/1.1\r\n'
                                                   'Accept-Encoding: identity\r\n'
                                                   'Content-Length: 65\r\n'
                                                   'Host: 127.0.0.1:ENGINE',
                                                   '{"messages": [{"role": "user", "content": "hi"}], "stream": true}']],
                                       'events': [['Entry',
                                                   '/v1/chat/completions',
                                                   {'messages': [{'content': 'hi', 'role': 'user'}], 'stream': True},
                                                   None,
                                                   'synthetic-loopback',
                                                   True,
                                                   None,
                                                   None],
                                                  ['led.label', None, None],
                                                  ['led.label_turn', None],
                                                  ['begin_queued', {'allow_start': True, 'inference': True, 'model_alias': None, 'origin': 'organic'}],
                                                  ['led.admitted_now', [1.5, 2.5]],
                                                  ['prepare_prefix', '/v1/chat/completions', {'messages': [{'content': 'hi', 'role': 'user'}], 'stream': True}],
                                                  ['watcher.start'],
                                                  ['led.observe', 'data: {"n":1}\n\n'],
                                                  ['watcher.stop'],
                                                  ['watcher.stop'],
                                                  ['led.row.update', {'thermal_c': 41.5}],
                                                  ['led.finish', 200, 'client_disconnected'],
                                                  ['engine_log',
                                                   '{"error": "client_disconnected", "event": "request_end", "model_alias": "synthetic-loopback", "status": '
                                                   '200}\n'],
                                                  ['end', {'inference': True, 'touch': False}]]},
 'test_controller_error_after_admission_is_500': {'client': 'HTTP/1.1 500 *\r\n'
                                                            'Server: *\r\n'
                                                            'Date: *\r\n'
                                                            'Content-Type: application/json\r\n'
                                                            'Content-Length: 150\r\n'
                                                            'Cache-Control: no-store\r\n'
                                                            '\r\n'
                                                            '{"error": {"message": "Local controller error before the request reached the engine.", "type": '
                                                            '"local_controller_error", "param": null, "code": null}}',
                                                  'closed': True,
                                                  'engine': [],
                                                  'events': [['Entry',
                                                              '/v1/chat/completions',
                                                              {'messages': [{'content': 'hi', 'role': 'user'}]},
                                                              None,
                                                              'synthetic-loopback',
                                                              True,
                                                              None,
                                                              None],
                                                             ['led.label', None, None],
                                                             ['led.label_turn', None],
                                                             ['begin_queued',
                                                              {'allow_start': True, 'inference': True, 'model_alias': None, 'origin': 'organic'}],
                                                             ['led.admitted_now', [1.5, 2.5]],
                                                             ['prepare_prefix', '/v1/chat/completions', {'messages': [{'content': 'hi', 'role': 'user'}]}],
                                                             ['end', {'inference': True, 'touch': False}],
                                                             ['led.finish', 500, 'local_controller_error']]},
 'test_corrupt_gzip_body_is_400': {'client': 'HTTP/1.1 400 *\r\n'
                                             'Server: *\r\n'
                                             'Date: *\r\n'
                                             'Content-Type: application/json\r\n'
                                             'Content-Length: 110\r\n'
                                             'Cache-Control: no-store\r\n'
                                             '\r\n'
                                             '{"error": {"message": "Invalid model request.", "type": "invalid_request_error", "param": null, "code": null}}',
                                   'closed': False,
                                   'engine': [],
                                   'events': []},
 'test_corrupt_gzip_reply_is_502': {'client': 'HTTP/1.1 502 *\r\n'
                                              'Server: *\r\n'
                                              'Date: *\r\n'
                                              'Content-Type: application/json\r\n'
                                              'Content-Length: 110\r\n'
                                              'Cache-Control: no-store\r\n'
                                              '\r\n'
                                              '{"error": {"message": "The local backend sent a corrupt compressed response.", "type": "local_backend_error"}}',
                                    'closed': False,
                                    'engine': [['GET /app.js HTTP/1.1\r\nHost: 127.0.0.1:ENGINE\r\nAccept-Encoding: gzip', '']],
                                    'events': [['begin_queued', {'allow_start': False, 'inference': False, 'model_alias': None, 'origin': 'organic'}],
                                               ['end', {'inference': False, 'touch': False}]]},
 'test_cors_reply_headers_replace_the_engines': {'client': 'HTTP/1.1 200 *\r\n'
                                                           'Server: *\r\n'
                                                           'Date: *\r\n'
                                                           'Content-Type: application/json\r\n'
                                                           'Access-Control-Allow-Origin: http://app.example\r\n'
                                                           'Vary: Origin\r\n'
                                                           'Cache-Control: no-store\r\n'
                                                           'Transfer-Encoding: chunked\r\n'
                                                           '\r\n'
                                                           '8\r\n'
                                                           '{"ok":1}\r\n'
                                                           '0\r\n'
                                                           '\r\n',
                                                 'closed': False,
                                                 'engine': [['POST /v1/chat/completions HTTP/1.1\r\n'
                                                             'Accept-Encoding: identity\r\n'
                                                             'Content-Length: 49\r\n'
                                                             'Host: 127.0.0.1:ENGINE',
                                                             '{"messages": [{"role": "user", "content": "hi"}]}']],
                                                 'events': [['Entry',
                                                             '/v1/chat/completions',
                                                             {'messages': [{'content': 'hi', 'role': 'user'}]},
                                                             None,
                                                             'synthetic-loopback',
                                                             True,
                                                             None,
                                                             None],
                                                            ['led.label', None, None],
                                                            ['led.label_turn', None],
                                                            ['begin_queued',
                                                             {'allow_start': True, 'inference': True, 'model_alias': None, 'origin': 'organic'}],
                                                            ['led.admitted_now', [1.5, 2.5]],
                                                            ['prepare_prefix', '/v1/chat/completions', {'messages': [{'content': 'hi', 'role': 'user'}]}],
                                                            ['watcher.start'],
                                                            ['led.observe', '{"ok":1}'],
                                                            ['watcher.stop'],
                                                            ['watcher.stop'],
                                                            ['led.row.update', {'thermal_c': 41.5}],
                                                            ['led.finish', 200, None],
                                                            ['end', {'inference': True, 'touch': False}]]},
 'test_engine_log_raising_still_ends': {'client': 'HTTP/1.1 502 *\r\n'
                                                  'Server: *\r\n'
                                                  'Date: *\r\n'
                                                  'Content-Type: application/json\r\n'
                                                  'Content-Length: 111\r\n'
                                                  'Cache-Control: no-store\r\n'
                                                  '\r\n'
                                                  '{"error": {"message": "Local backend connection closed.", "type": "backend_died", "param": null, "code": '
                                                  'null}}',
                                        'closed': True,
                                        'engine': [['POST /v1/chat/completions HTTP/1.1\r\n'
                                                    'Accept-Encoding: identity\r\n'
                                                    'Content-Length: 49\r\n'
                                                    'Host: 127.0.0.1:ENGINE',
                                                    '{"messages": [{"role": "user", "content": "hi"}]}']],
                                        'events': [['Entry',
                                                    '/v1/chat/completions',
                                                    {'messages': [{'content': 'hi', 'role': 'user'}]},
                                                    None,
                                                    'synthetic-loopback',
                                                    True,
                                                    None,
                                                    None],
                                                   ['led.label', None, None],
                                                   ['led.label_turn', None],
                                                   ['begin_queued', {'allow_start': True, 'inference': True, 'model_alias': None, 'origin': 'organic'}],
                                                   ['led.admitted_now', [1.5, 2.5]],
                                                   ['prepare_prefix', '/v1/chat/completions', {'messages': [{'content': 'hi', 'role': 'user'}]}],
                                                   ['watcher.start'],
                                                   ['watcher.stop'],
                                                   ['led.row.update', {'thermal_c': 41.5}],
                                                   ['led.finish', 503, 'backend_died'],
                                                   ['engine_log',
                                                    '{"error": "backend_died", "event": "request_end", "model_alias": "synthetic-loopback", "status": null}\n'],
                                                   ['end', {'inference': True, 'touch': False}]]},
 'test_explicit_route_disabled_is_400': {'client': 'HTTP/1.1 400 *\r\n'
                                                   'Server: *\r\n'
                                                   'Date: *\r\n'
                                                   'Content-Type: application/json\r\n'
                                                   'Content-Length: 110\r\n'
                                                   'Cache-Control: no-store\r\n'
                                                   '\r\n'
                                                   '{"error": {"message": "Invalid model request.", "type": "invalid_request_error", "param": null, "code": '
                                                   'null}}',
                                         'closed': False,
                                         'engine': [],
                                         'events': []},
 'test_explicit_route_rewrites_to_the_routed_model': {'client': 'HTTP/1.1 200 *\r\n'
                                                                'Server: *\r\n'
                                                                'Date: *\r\n'
                                                                'Content-Type: application/json\r\n'
                                                                'Cache-Control: no-store\r\n'
                                                                'Transfer-Encoding: chunked\r\n'
                                                                '\r\n'
                                                                'b\r\n'
                                                                '{"ok":true}\r\n'
                                                                '0\r\n'
                                                                '\r\n',
                                                      'closed': False,
                                                      'engine': [['POST /v1/chat/completions HTTP/1.1\r\n'
                                                                  'Accept-Encoding: identity\r\n'
                                                                  'Content-Length: 194\r\n'
                                                                  'Host: 127.0.0.1:ENGINE',
                                                                  '{"messages": [{"role": "user", "content": "hi"}], "model": "ornith-35b-a3b-local", '
                                                                  '"temperature": 0, "cache_prompt": true, "max_tokens": 2048, "chat_template_kwargs": '
                                                                  '{"enable_thinking": false}}']],
                                                      'events': [['config', 'ornith-35b-a3b-local'],
                                                                 ['config', 'ornith-35b-a3b-local'],
                                                                 ['Entry',
                                                                  '/v1/chat/completions',
                                                                  {'cache_prompt': True,
                                                                   'chat_template_kwargs': {'enable_thinking': False},
                                                                   'max_tokens': 2048,
                                                                   'messages': [{'content': 'hi', 'role': 'user'}],
                                                                   'model': 'ornith-35b-a3b-local',
                                                                   'temperature': 0},
                                                                  None,
                                                                  'ornith-35b-a3b-local',
                                                                  True,
                                                                  None,
                                                                  None],
                                                                 ['led.label', None, None],
                                                                 ['led.label_turn', None],
                                                                 ['runtime_value'],
                                                                 ['led.row', 'mode', 'fast'],
                                                                 ['begin_queued',
                                                                  {'allow_start': True,
                                                                   'inference': True,
                                                                   'model_alias': 'ornith-35b-a3b-local',
                                                                   'origin': 'organic'}],
                                                                 ['led.admitted_now', [1.5, 2.5]],
                                                                 ['prepare_prefix',
                                                                  '/v1/chat/completions',
                                                                  {'cache_prompt': True,
                                                                   'chat_template_kwargs': {'enable_thinking': False},
                                                                   'max_tokens': 2048,
                                                                   'messages': [{'content': 'hi', 'role': 'user'}],
                                                                   'model': 'ornith-35b-a3b-local',
                                                                   'temperature': 0}],
                                                                 ['watcher.start'],
                                                                 ['led.observe', '{"ok":true}'],
                                                                 ['watcher.stop'],
                                                                 ['watcher.stop'],
                                                                 ['led.row.update', {'thermal_c': 41.5}],
                                                                 ['led.finish', 200, None],
                                                                 ['end', {'inference': True, 'touch': False}]]},
 'test_flash_bad_chat_body_is_400_after_admission': {'client': 'HTTP/1.1 400 *\r\n'
                                                               'Server: *\r\n'
                                                               'Date: *\r\n'
                                                               'Content-Type: application/json\r\n'
                                                               'Content-Length: 149\r\n'
                                                               'Cache-Control: no-store\r\n'
                                                               '\r\n'
                                                               '{"error": {"message": "Invalid Flash chat request or unqualified reasoning settings.", "type": '
                                                               '"invalid_request_error", "param": null, "code": null}}',
                                                     'closed': False,
                                                     'engine': [],
                                                     'events': [['normalize_chat',
                                                                 'synthetic-flash&1',
                                                                 {'messages': [{'content': 'hi', 'role': 'user'}], 'native': 'other-model'},
                                                                 47999],
                                                                ['Entry',
                                                                 '/v1/chat/completions',
                                                                 {'max_tokens': None, 'messages': [{'content': 'hi', 'role': 'user'}], 'model': 'other-model'},
                                                                 None,
                                                                 'synthetic-flash&1',
                                                                 True,
                                                                 None,
                                                                 None],
                                                                ['led.label', None, None],
                                                                ['led.label_turn', None],
                                                                ['begin_queued',
                                                                 {'allow_start': True, 'inference': True, 'model_alias': None, 'origin': 'organic'}],
                                                                ['led.admitted_now', [1.5, 2.5]],
                                                                ['chat_body',
                                                                 {'max_tokens': None, 'messages': [{'content': 'hi', 'role': 'user'}], 'model': 'other-model'}],
                                                                ['end', {'inference': True, 'touch': False}],
                                                                ['led.finish', 400, 'bad_request']]},
 'test_flash_chat_body_reshaped_by_the_runtime': {'client': 'HTTP/1.1 200 *\r\n'
                                                            'Server: *\r\n'
                                                            'Date: *\r\n'
                                                            'Content-Type: application/json\r\n'
                                                            'Cache-Control: no-store\r\n'
                                                            'Transfer-Encoding: chunked\r\n'
                                                            '\r\n'
                                                            'b\r\n'
                                                            '{"ok":true}\r\n'
                                                            '0\r\n'
                                                            '\r\n',
                                                  'closed': False,
                                                  'engine': [['POST /v1/chat/completions HTTP/1.1\r\n'
                                                              'Accept-Encoding: identity\r\n'
                                                              'Content-Length: 110\r\n'
                                                              'Host: 127.0.0.1:ENGINE\r\n'
                                                              'Authorization: Bearer rk\r\n'
                                                              'Content-Type: application/json',
                                                              '{"model": "flash-native", "messages": [{"role": "user", "content": "hi"}], "max_tokens": null, '
                                                              '"shaped": true}']],
                                                  'events': [['normalize_chat',
                                                              'synthetic-flash&1',
                                                              {'messages': [{'content': 'hi', 'role': 'user'}], 'native': 'other-model'},
                                                              47999],
                                                             ['Entry',
                                                              '/v1/chat/completions',
                                                              {'max_tokens': None, 'messages': [{'content': 'hi', 'role': 'user'}], 'model': 'other-model'},
                                                              None,
                                                              'synthetic-flash&1',
                                                              True,
                                                              None,
                                                              None],
                                                             ['led.label', None, None],
                                                             ['led.label_turn', None],
                                                             ['begin_queued',
                                                              {'allow_start': True, 'inference': True, 'model_alias': None, 'origin': 'organic'}],
                                                             ['led.admitted_now', [1.5, 2.5]],
                                                             ['chat_body',
                                                              {'max_tokens': None, 'messages': [{'content': 'hi', 'role': 'user'}], 'model': 'other-model'}],
                                                             ['watcher.start'],
                                                             ['led.observe', '{"ok":true}'],
                                                             ['watcher.stop'],
                                                             ['watcher.stop'],
                                                             ['led.row.update', {'thermal_c': 41.5}],
                                                             ['led.finish', 200, None],
                                                             ['end', {'inference': True, 'touch': False}]]},
 'test_flash_chat_headers_and_body': {'client': 'HTTP/1.1 200 *\r\n'
                                                'Server: *\r\n'
                                                'Date: *\r\n'
                                                'Content-Type: application/json\r\n'
                                                'X-Kept: k\r\n'
                                                'Cache-Control: no-store\r\n'
                                                'Transfer-Encoding: chunked\r\n'
                                                '\r\n'
                                                '8\r\n'
                                                '{"ok":1}\r\n'
                                                '0\r\n'
                                                '\r\n',
                                      'closed': False,
                                      'engine': [['POST /v1/chat/completions HTTP/1.1\r\n'
                                                  'Accept-Encoding: identity\r\n'
                                                  'Content-Length: 94\r\n'
                                                  'Content-Type: application/json\r\n'
                                                  'Host: 127.0.0.1:ENGINE\r\n'
                                                  'Origin: http://127.0.0.1:ENGINE\r\n'
                                                  'Authorization: Bearer rk',
                                                  '{"model": "flash-native", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 2048}']],
                                      'events': [['harness_mu.enter'],
                                                 ['compat_stats.update', ['flash_max_completion_tokens_renamed', 'flash_max_tokens_capped']],
                                                 ['harness_mu.exit'],
                                                 ['normalize_chat',
                                                  'synthetic-flash&1',
                                                  {'max_tokens': 2048, 'messages': [{'content': 'hi', 'role': 'user'}]},
                                                  47999],
                                                 ['Entry',
                                                  '/v1/chat/completions',
                                                  {'max_tokens': 2048, 'messages': [{'content': 'hi', 'role': 'user'}], 'model': 'flash-native'},
                                                  None,
                                                  'synthetic-flash&1',
                                                  True,
                                                  None,
                                                  None],
                                                 ['led.label', None, None],
                                                 ['led.label_turn', None],
                                                 ['begin_queued', {'allow_start': True, 'inference': True, 'model_alias': None, 'origin': 'organic'}],
                                                 ['led.admitted_now', [1.5, 2.5]],
                                                 ['watcher.start'],
                                                 ['led.observe', '{"ok":1}'],
                                                 ['watcher.stop'],
                                                 ['watcher.stop'],
                                                 ['led.row.update', {'thermal_c': 41.5}],
                                                 ['led.finish', 200, None],
                                                 ['end', {'inference': True, 'touch': False}]]},
 'test_flash_chat_route_with_a_query_is_403': {'client': 'HTTP/1.1 403 *\r\n'
                                                         'Server: *\r\n'
                                                         'Date: *\r\n'
                                                         'Content-Type: application/json\r\n'
                                                         'Content-Length: 127\r\n'
                                                         'Cache-Control: no-store\r\n'
                                                         '\r\n'
                                                         '{"error": {"message": "This Flash route is not exposed by Local AI.", "type": "permission_error", '
                                                         '"param": null, "code": null}}',
                                               'closed': True,
                                               'engine': [],
                                               'events': []},
 'test_flash_normalize_failure_is_400': {'client': 'HTTP/1.1 400 *\r\n'
                                                   'Server: *\r\n'
                                                   'Date: *\r\n'
                                                   'Content-Type: application/json\r\n'
                                                   'Content-Length: 149\r\n'
                                                   'Cache-Control: no-store\r\n'
                                                   '\r\n'
                                                   '{"error": {"message": "Invalid Flash chat request or unqualified reasoning settings.", "type": '
                                                   '"invalid_request_error", "param": null, "code": null}}',
                                         'closed': False,
                                         'engine': [],
                                         'events': [['normalize_chat', 'synthetic-flash&1', {'messages': 'bad'}, 47999]]},
 'test_flash_other_route_is_403': {'client': 'HTTP/1.1 403 *\r\n'
                                             'Server: *\r\n'
                                             'Date: *\r\n'
                                             'Content-Type: application/json\r\n'
                                             'Content-Length: 127\r\n'
                                             'Cache-Control: no-store\r\n'
                                             '\r\n'
                                             '{"error": {"message": "This Flash route is not exposed by Local AI.", "type": "permission_error", "param": null, '
                                             '"code": null}}',
                                   'closed': True,
                                   'engine': [],
                                   'events': []},
 'test_flash_page': {'client': 'HTTP/1.1 200 *\r\n'
                               'Server: *\r\n'
                               'Date: *\r\n'
                               'Content-Type: text/html; charset=utf-8\r\n'
                               'Content-Length: 1712\r\n'
                               'Cache-Control: no-store\r\n'
                               '\r\n'
                               '<!doctype html><html><meta charset="utf-8"><title>Local Flash chat</title>\n'
                               '<style>body{font:16px system-ui;max-width:760px;margin:5vh '
                               'auto;padding:16px;background:#171717;color:#eee}textarea{width:100%;min-height:90px}pre{white-space:pre-wrap}button{margin-top:8px}</style>\n'
                               '<body data-model="synthetic-flash&amp;1"><h1>Local Flash chat</h1><p>This page keeps messages in memory until it closes. The '
                               'model loads when you send a message.</p>\n'
                               '<pre id="transcript"></pre><textarea id="prompt" aria-label="Message"></textarea><button id="send">Send</button><p '
                               'id="status"></p>\n'
                               "<script>const messages=[];const transcript=document.getElementById('transcript');const "
                               "status=document.getElementById('status');const button=document.getElementById('send');const "
                               "input=document.getElementById('prompt');\n"
                               "button.onclick=async()=>{const text=input.value.trim();if(!text)return;button.disabled=true;status.textContent='Loading or "
                               "answering...';messages.push({role:'user',content:text});transcript.textContent+='You: '+text+'\\n\\n';input.value='';try{const "
                               'reply=await '
                               "fetch('/v1/chat/completions',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({model:document.body.dataset.model,messages,stream:false,max_tokens:512,think:false})});const "
                               "data=await reply.json();if(!reply.ok)throw new Error('The local backend could not answer.');const "
                               "answer=data.choices?.[0]?.message?.content;if(typeof answer!=='string')throw new Error('Invalid local "
                               "answer.');messages.push({role:'assistant',content:answer});transcript.textContent+='Assistant: "
                               "'+answer+'\\n\\n';status.textContent='Ready';}catch(error){messages.pop();status.textContent=error.message;}finally{button.disabled=false;}};\n"
                               '</script></body></html>',
                     'closed': False,
                     'engine': [],
                     'events': [['eligible', 'synthetic-flash&1']]},
 'test_flash_page_unqualified_is_503': {'client': 'HTTP/1.1 503 *\r\n'
                                                  'Server: *\r\n'
                                                  'Date: *\r\n'
                                                  'Content-Type: application/json\r\n'
                                                  'Content-Length: 115\r\n'
                                                  'Cache-Control: no-store\r\n'
                                                  'Retry-After: 5\r\n'
                                                  '\r\n'
                                                  '{"error": {"message": "Flash profile is not qualified.", "type": "unavailable_error", "param": null, '
                                                  '"code": null}}',
                                        'closed': False,
                                        'engine': [],
                                        'events': [['eligible', 'synthetic-flash&1']]},
 'test_get_with_a_query_is_neither_served_from_nor_stored_in_the_cache': {'client': 'HTTP/1.1 200 *\r\n'
                                                                                    'Server: *\r\n'
                                                                                    'Date: *\r\n'
                                                                                    'Content-Type: application/json\r\n'
                                                                                    'Cache-Control: no-store\r\n'
                                                                                    'Transfer-Encoding: chunked\r\n'
                                                                                    '\r\n'
                                                                                    'd\r\n'
                                                                                    '{"props":"q"}\r\n'
                                                                                    '0\r\n'
                                                                                    '\r\n',
                                                                          'closed': False,
                                                                          'engine': [['GET /props?x=1 HTTP/1.1\r\n'
                                                                                      'Accept-Encoding: identity\r\n'
                                                                                      'Host: 127.0.0.1:ENGINE',
                                                                                      '']],
                                                                          'events': [['begin_queued',
                                                                                      {'allow_start': False,
                                                                                       'inference': False,
                                                                                       'model_alias': None,
                                                                                       'origin': 'organic'}],
                                                                                     ['end', {'inference': False, 'touch': False}]]},
 'test_gzip_body_passes_through_when_nothing_rewrites_it': {'client': 'HTTP/1.1 200 *\r\n'
                                                                      'Server: *\r\n'
                                                                      'Date: *\r\n'
                                                                      'Content-Type: application/json\r\n'
                                                                      'Cache-Control: no-store\r\n'
                                                                      'Transfer-Encoding: chunked\r\n'
                                                                      '\r\n'
                                                                      'b\r\n'
                                                                      '{"ok":true}\r\n'
                                                                      '0\r\n'
                                                                      '\r\n',
                                                            'closed': False,
                                                            'engine': [['POST /completion HTTP/1.1\r\n'
                                                                        'Accept-Encoding: identity\r\n'
                                                                        'Content-Length: 35\r\n'
                                                                        'Content-Encoding: gzip\r\n'
                                                                        'Host: 127.0.0.1:ENGINE',
                                                                        '\x1f'
                                                                        '\x8b\x08\x00\x00\x00\x00\x00\x02\x03«V*(ÊÏ-(Q²RJ¯Rª\x05\x00)\x08+^\x0f\x00\x00\x00']],
                                                            'events': [['Entry', '/completion', {'prompt': 'gz'}, None, 'synthetic-loopback', True, None, None],
                                                                       ['led.label', None, None],
                                                                       ['led.label_turn', None],
                                                                       ['begin_queued',
                                                                        {'allow_start': True, 'inference': True, 'model_alias': None, 'origin': 'organic'}],
                                                                       ['led.admitted_now', [1.5, 2.5]],
                                                                       ['prepare_prefix', '/completion', {'prompt': 'gz'}],
                                                                       ['watcher.start'],
                                                                       ['led.observe', '{"ok":true}'],
                                                                       ['watcher.stop'],
                                                                       ['watcher.stop'],
                                                                       ['led.row.update', {'thermal_c': 41.5}],
                                                                       ['led.finish', 200, None],
                                                                       ['end', {'inference': True, 'touch': False}]]},
 'test_gzip_body_rewritten_drops_content_encoding': {'client': 'HTTP/1.1 200 *\r\n'
                                                               'Server: *\r\n'
                                                               'Date: *\r\n'
                                                               'Content-Type: application/json\r\n'
                                                               'Cache-Control: no-store\r\n'
                                                               'Transfer-Encoding: chunked\r\n'
                                                               '\r\n'
                                                               'b\r\n'
                                                               '{"ok":true}\r\n'
                                                               '0\r\n'
                                                               '\r\n',
                                                     'closed': False,
                                                     'engine': [['POST /completion HTTP/1.1\r\n'
                                                                 'Accept-Encoding: identity\r\n'
                                                                 'Content-Length: 47\r\n'
                                                                 'Host: 127.0.0.1:ENGINE',
                                                                 '{"model": "synthetic-loopback", "prompt": "gz"}']],
                                                     'events': [['config', 'qwen-local'],
                                                                ['config', 'synthetic-loopback'],
                                                                ['Entry',
                                                                 '/completion',
                                                                 {'model': 'synthetic-loopback', 'prompt': 'gz'},
                                                                 None,
                                                                 'synthetic-loopback',
                                                                 True,
                                                                 None,
                                                                 None],
                                                                ['led.label', None, None],
                                                                ['led.label_turn', None],
                                                                ['begin_queued',
                                                                 {'allow_start': True,
                                                                  'inference': True,
                                                                  'model_alias': 'synthetic-loopback',
                                                                  'origin': 'organic'}],
                                                                ['led.admitted_now', [1.5, 2.5]],
                                                                ['prepare_prefix', '/completion', {'model': 'synthetic-loopback', 'prompt': 'gz'}],
                                                                ['watcher.start'],
                                                                ['led.observe', '{"ok":true}'],
                                                                ['watcher.stop'],
                                                                ['watcher.stop'],
                                                                ['led.row.update', {'thermal_c': 41.5}],
                                                                ['led.finish', 200, None],
                                                                ['end', {'inference': True, 'touch': False}]]},
 'test_gzip_bomb_is_400': {'client': 'HTTP/1.1 400 *\r\n'
                                     'Server: *\r\n'
                                     'Date: *\r\n'
                                     'Content-Type: application/json\r\n'
                                     'Content-Length: 110\r\n'
                                     'Cache-Control: no-store\r\n'
                                     '\r\n'
                                     '{"error": {"message": "Invalid model request.", "type": "invalid_request_error", "param": null, "code": null}}',
                           'closed': False,
                           'engine': [],
                           'events': []},
 'test_gzip_ui_asset_is_decoded_cached_and_served_from_cache': {'first': {'client': 'HTTP/1.1 200 *\r\n'
                                                                                    'Server: *\r\n'
                                                                                    'Date: *\r\n'
                                                                                    'Content-Type: text/html\r\n'
                                                                                    'Content-Length: 15\r\n'
                                                                                    'Cache-Control: no-store\r\n'
                                                                                    '\r\n'
                                                                                    '<html>ui</html>',
                                                                          'closed': False,
                                                                          'engine': [['GET /index.html HTTP/1.1\r\n'
                                                                                      'Host: 127.0.0.1:ENGINE\r\n'
                                                                                      'Accept-Encoding: gzip',
                                                                                      '']],
                                                                          'events': [['begin_queued',
                                                                                      {'allow_start': False,
                                                                                       'inference': False,
                                                                                       'model_alias': None,
                                                                                       'origin': 'organic'}],
                                                                                     ['cache.set', '/index.html', ['text/html', '<html>ui</html>']],
                                                                                     ['end', {'inference': False, 'touch': False}]]},
                                                                'second': {'client': 'HTTP/1.1 200 *\r\n'
                                                                                     'Server: *\r\n'
                                                                                     'Date: *\r\n'
                                                                                     'Content-Type: text/html\r\n'
                                                                                     'Content-Length: 15\r\n'
                                                                                     'Cache-Control: no-store\r\n'
                                                                                     '\r\n'
                                                                                     '<html>ui</html>',
                                                                           'closed': False,
                                                                           'engine': [['GET /index.html HTTP/1.1\r\n'
                                                                                       'Host: 127.0.0.1:ENGINE\r\n'
                                                                                       'Accept-Encoding: gzip',
                                                                                       '']],
                                                                           'events': []}},
 'test_head': {'client': 'HTTP/1.1 200 *\r\n'
                         'Server: *\r\n'
                         'Date: *\r\n'
                         'Content-Type: application/json\r\n'
                         'Cache-Control: no-store\r\n'
                         'Transfer-Encoding: chunked\r\n'
                         '\r\n',
               'closed': False,
               'engine': [['HEAD /props HTTP/1.1\r\nAccept-Encoding: identity\r\nHost: 127.0.0.1:ENGINE', '']],
               'events': [['begin_queued', {'allow_start': False, 'inference': False, 'model_alias': None, 'origin': 'organic'}],
                          ['end', {'inference': False, 'touch': False}]]},
 'test_helper_request_is_not_inference': {'client': 'HTTP/1.1 200 *\r\n'
                                                    'Server: *\r\n'
                                                    'Date: *\r\n'
                                                    'Content-Type: application/json\r\n'
                                                    'Cache-Control: no-store\r\n'
                                                    'Transfer-Encoding: chunked\r\n'
                                                    '\r\n'
                                                    'e\r\n'
                                                    '{"tokens":[1]}\r\n'
                                                    '0\r\n'
                                                    '\r\n',
                                          'closed': False,
                                          'engine': [['POST /tokenize HTTP/1.1\r\nAccept-Encoding: identity\r\nContent-Length: 47\r\nHost: 127.0.0.1:ENGINE',
                                                      '{"model": "synthetic-loopback", "content": "a"}']],
                                          'events': [['config', 'qwen-local'],
                                                     ['config', 'synthetic-loopback'],
                                                     ['begin_queued',
                                                      {'allow_start': True, 'inference': False, 'model_alias': 'synthetic-loopback', 'origin': 'organic'}],
                                                     ['end', {'inference': False, 'touch': True}]]},
 'test_huge_content_length_is_413': {'client': 'HTTP/1.1 413 *\r\n'
                                               'Server: *\r\n'
                                               'Date: *\r\n'
                                               'Content-Type: application/json\r\n'
                                               'Content-Length: 109\r\n'
                                               'Cache-Control: no-store\r\n'
                                               '\r\n'
                                               '{"error": {"message": "Request is too large.", "type": "invalid_request_error", "param": null, "code": null}}',
                                     'closed': True,
                                     'engine': [],
                                     'events': []},
 'test_invalid_json_is_400': {'client': 'HTTP/1.1 400 *\r\n'
                                        'Server: *\r\n'
                                        'Date: *\r\n'
                                        'Content-Type: application/json\r\n'
                                        'Content-Length: 110\r\n'
                                        'Cache-Control: no-store\r\n'
                                        '\r\n'
                                        '{"error": {"message": "Invalid model request.", "type": "invalid_request_error", "param": null, "code": null}}',
                              'closed': False,
                              'engine': [],
                              'events': []},
 'test_ledger_admitted_now_raising_still_serves_and_ends': {'client': 'HTTP/1.1 200 *\r\n'
                                                                      'Server: *\r\n'
                                                                      'Date: *\r\n'
                                                                      'Content-Type: application/json\r\n'
                                                                      'Cache-Control: no-store\r\n'
                                                                      'Transfer-Encoding: chunked\r\n'
                                                                      '\r\n'
                                                                      'b\r\n'
                                                                      '{"ok":true}\r\n'
                                                                      '0\r\n'
                                                                      '\r\n',
                                                            'closed': False,
                                                            'engine': [['POST /v1/chat/completions HTTP/1.1\r\n'
                                                                        'Accept-Encoding: identity\r\n'
                                                                        'Content-Length: 49\r\n'
                                                                        'Host: 127.0.0.1:ENGINE',
                                                                        '{"messages": [{"role": "user", "content": "hi"}]}']],
                                                            'events': [['Entry',
                                                                        '/v1/chat/completions',
                                                                        {'messages': [{'content': 'hi', 'role': 'user'}]},
                                                                        None,
                                                                        'synthetic-loopback',
                                                                        True,
                                                                        None,
                                                                        None],
                                                                       ['led.label', None, None],
                                                                       ['led.label_turn', None],
                                                                       ['begin_queued',
                                                                        {'allow_start': True, 'inference': True, 'model_alias': None, 'origin': 'organic'}],
                                                                       ['led.admitted_now', [1.5, 2.5]],
                                                                       ['prepare_prefix',
                                                                        '/v1/chat/completions',
                                                                        {'messages': [{'content': 'hi', 'role': 'user'}]}],
                                                                       ['watcher.start'],
                                                                       ['led.observe', '{"ok":true}'],
                                                                       ['watcher.stop'],
                                                                       ['watcher.stop'],
                                                                       ['led.row.update', {'thermal_c': 41.5}],
                                                                       ['led.finish', 200, None],
                                                                       ['end', {'inference': True, 'touch': False}]]},
 'test_ledger_finish_raising_still_ends': {'client': 'HTTP/1.1 200 *\r\n'
                                                     'Server: *\r\n'
                                                     'Date: *\r\n'
                                                     'Content-Type: application/json\r\n'
                                                     'Cache-Control: no-store\r\n'
                                                     'Transfer-Encoding: chunked\r\n'
                                                     '\r\n'
                                                     'b\r\n'
                                                     '{"ok":true}\r\n'
                                                     '0\r\n'
                                                     '\r\n',
                                           'closed': True,
                                           'engine': [['POST /v1/chat/completions HTTP/1.1\r\n'
                                                       'Accept-Encoding: identity\r\n'
                                                       'Content-Length: 49\r\n'
                                                       'Host: 127.0.0.1:ENGINE',
                                                       '{"messages": [{"role": "user", "content": "hi"}]}']],
                                           'events': [['Entry',
                                                       '/v1/chat/completions',
                                                       {'messages': [{'content': 'hi', 'role': 'user'}]},
                                                       None,
                                                       'synthetic-loopback',
                                                       True,
                                                       None,
                                                       None],
                                                      ['led.label', None, None],
                                                      ['led.label_turn', None],
                                                      ['begin_queued', {'allow_start': True, 'inference': True, 'model_alias': None, 'origin': 'organic'}],
                                                      ['led.admitted_now', [1.5, 2.5]],
                                                      ['prepare_prefix', '/v1/chat/completions', {'messages': [{'content': 'hi', 'role': 'user'}]}],
                                                      ['watcher.start'],
                                                      ['led.observe', '{"ok":true}'],
                                                      ['watcher.stop'],
                                                      ['watcher.stop'],
                                                      ['led.row.update', {'thermal_c': 41.5}],
                                                      ['led.finish', 200, None],
                                                      ['end', {'inference': True, 'touch': False}]]},
 'test_local_api_key_is_checked_and_never_forwarded': {'client': 'HTTP/1.1 200 *\r\n'
                                                                 'Server: *\r\n'
                                                                 'Date: *\r\n'
                                                                 'Content-Type: application/json\r\n'
                                                                 'Cache-Control: no-store\r\n'
                                                                 'Transfer-Encoding: chunked\r\n'
                                                                 '\r\n'
                                                                 'b\r\n'
                                                                 '{"ok":true}\r\n'
                                                                 '0\r\n'
                                                                 '\r\n',
                                                       'closed': False,
                                                       'engine': [['POST /completion HTTP/1.1\r\n'
                                                                   'Accept-Encoding: identity\r\n'
                                                                   'Content-Length: 14\r\n'
                                                                   'X-Other: o\r\n'
                                                                   'Host: 127.0.0.1:ENGINE',
                                                                   '{"prompt":"k"}']],
                                                       'events': [['Entry', '/completion', {'prompt': 'k'}, None, 'synthetic-loopback', True, None, None],
                                                                  ['led.label', None, None],
                                                                  ['led.label_turn', None],
                                                                  ['begin_queued',
                                                                   {'allow_start': True, 'inference': True, 'model_alias': None, 'origin': 'organic'}],
                                                                  ['led.admitted_now', [1.5, 2.5]],
                                                                  ['prepare_prefix', '/completion', {'prompt': 'k'}],
                                                                  ['watcher.start'],
                                                                  ['led.observe', '{"ok":true}'],
                                                                  ['watcher.stop'],
                                                                  ['watcher.stop'],
                                                                  ['led.row.update', {'thermal_c': 41.5}],
                                                                  ['led.finish', 200, None],
                                                                  ['end', {'inference': True, 'touch': False}]]},
 'test_local_route_without_auto_is_dropped': {'client': 'HTTP/1.1 200 *\r\n'
                                                        'Server: *\r\n'
                                                        'Date: *\r\n'
                                                        'Content-Type: application/json\r\n'
                                                        'Cache-Control: no-store\r\n'
                                                        'Transfer-Encoding: chunked\r\n'
                                                        '\r\n'
                                                        'b\r\n'
                                                        '{"ok":true}\r\n'
                                                        '0\r\n'
                                                        '\r\n',
                                              'closed': False,
                                              'engine': [['POST /v1/chat/completions HTTP/1.1\r\n'
                                                          'Accept-Encoding: identity\r\n'
                                                          'Content-Length: 49\r\n'
                                                          'Host: 127.0.0.1:ENGINE',
                                                          '{"messages": [{"role": "user", "content": "hi"}]}']],
                                              'events': [['Entry',
                                                          '/v1/chat/completions',
                                                          {'messages': [{'content': 'hi', 'role': 'user'}]},
                                                          None,
                                                          'synthetic-loopback',
                                                          True,
                                                          None,
                                                          None],
                                                         ['led.label', None, None],
                                                         ['led.label_turn', None],
                                                         ['begin_queued', {'allow_start': True, 'inference': True, 'model_alias': None, 'origin': 'organic'}],
                                                         ['led.admitted_now', [1.5, 2.5]],
                                                         ['prepare_prefix', '/v1/chat/completions', {'messages': [{'content': 'hi', 'role': 'user'}]}],
                                                         ['watcher.start'],
                                                         ['led.observe', '{"ok":true}'],
                                                         ['watcher.stop'],
                                                         ['watcher.stop'],
                                                         ['led.row.update', {'thermal_c': 41.5}],
                                                         ['led.finish', 200, None],
                                                         ['end', {'inference': True, 'touch': False}]]},
 'test_local_start_failure_names_the_cause': {'client': 'HTTP/1.1 503 *\r\n'
                                                        'Server: *\r\n'
                                                        'Date: *\r\n'
                                                        'Content-Type: application/json\r\n'
                                                        'Content-Length: 198\r\n'
                                                        'Cache-Control: no-store\r\n'
                                                        'Retry-After: 5\r\n'
                                                        '\r\n'
                                                        '{"error": {"message": "Local backend could not be started (RuntimeError: engine exploded). Details: '
                                                        'journalctl --user -u local-ai-chat.", "type": "local_backend_error", "param": null, "code": null}}',
                                              'closed': False,
                                              'engine': [],
                                              'events': [['Entry',
                                                          '/v1/chat/completions',
                                                          {'messages': [{'content': 'hi', 'role': 'user'}]},
                                                          None,
                                                          'synthetic-loopback',
                                                          True,
                                                          None,
                                                          None],
                                                         ['led.label', None, None],
                                                         ['led.label_turn', None],
                                                         ['begin_queued', {'allow_start': True, 'inference': True, 'model_alias': None, 'origin': 'organic'}],
                                                         ['begin_queued.raised', 'RuntimeError'],
                                                         ['led.finish', 503, 'start_failed']]},
 'test_model_alias_is_rewritten': {'client': 'HTTP/1.1 200 *\r\n'
                                             'Server: *\r\n'
                                             'Date: *\r\n'
                                             'Content-Type: application/json\r\n'
                                             'Cache-Control: no-store\r\n'
                                             'Transfer-Encoding: chunked\r\n'
                                             '\r\n'
                                             'b\r\n'
                                             '{"ok":true}\r\n'
                                             '0\r\n'
                                             '\r\n',
                                   'closed': False,
                                   'engine': [['POST /completion HTTP/1.1\r\n'
                                               'Accept-Encoding: identity\r\n'
                                               'Content-Length: 46\r\n'
                                               'Content-Type: application/json\r\n'
                                               'Host: 127.0.0.1:ENGINE',
                                               '{"model": "synthetic-loopback", "prompt": "a"}']],
                                   'events': [['config', 'qwen-local'],
                                              ['config', 'synthetic-loopback'],
                                              ['Entry',
                                               '/completion',
                                               {'model': 'synthetic-loopback', 'prompt': 'a'},
                                               None,
                                               'synthetic-loopback',
                                               True,
                                               None,
                                               None],
                                              ['led.label', None, None],
                                              ['led.label_turn', None],
                                              ['begin_queued',
                                               {'allow_start': True, 'inference': True, 'model_alias': 'synthetic-loopback', 'origin': 'organic'}],
                                              ['led.admitted_now', [1.5, 2.5]],
                                              ['prepare_prefix', '/completion', {'model': 'synthetic-loopback', 'prompt': 'a'}],
                                              ['watcher.start'],
                                              ['led.observe', '{"ok":true}'],
                                              ['watcher.stop'],
                                              ['watcher.stop'],
                                              ['led.row.update', {'thermal_c': 41.5}],
                                              ['led.finish', 200, None],
                                              ['end', {'inference': True, 'touch': False}]]},
 'test_model_busy_is_409': {'client': 'HTTP/1.1 409 *\r\n'
                                      'Server: *\r\n'
                                      'Date: *\r\n'
                                      'Content-Type: application/json\r\n'
                                      'Content-Length: 113\r\n'
                                      'Cache-Control: no-store\r\n'
                                      '\r\n'
                                      '{"error": {"message": "Another model is loaded.", "type": "conflict_error", "code": "model_busy", "param": null}}',
                            'closed': False,
                            'engine': [],
                            'events': [['Entry',
                                        '/v1/chat/completions',
                                        {'messages': [{'content': 'hi', 'role': 'user'}]},
                                        None,
                                        'synthetic-loopback',
                                        True,
                                        None,
                                        None],
                                       ['led.label', None, None],
                                       ['led.label_turn', None],
                                       ['begin_queued', {'allow_start': True, 'inference': True, 'model_alias': None, 'origin': 'organic'}],
                                       ['begin_queued.raised', 'ModelBusy'],
                                       ['led.finish', 409, 'model_busy']]},
 'test_negative_content_length_is_413': {'client': 'HTTP/1.1 413 *\r\n'
                                                   'Server: *\r\n'
                                                   'Date: *\r\n'
                                                   'Content-Type: application/json\r\n'
                                                   'Content-Length: 109\r\n'
                                                   'Cache-Control: no-store\r\n'
                                                   '\r\n'
                                                   '{"error": {"message": "Request is too large.", "type": "invalid_request_error", "param": null, "code": '
                                                   'null}}',
                                         'closed': True,
                                         'engine': [],
                                         'events': []},
 'test_no_model_configured_is_503': {'client': 'HTTP/1.1 503 *\r\n'
                                               'Server: *\r\n'
                                               'Date: *\r\n'
                                               'Content-Type: application/json\r\n'
                                               'Content-Length: 110\r\n'
                                               'Cache-Control: no-store\r\n'
                                               'Retry-After: 5\r\n'
                                               '\r\n'
                                               '{"error": {"message": "No model is selected yet.", "type": "local_backend_busy", "param": null, "code": null}}',
                                     'closed': False,
                                     'engine': [],
                                     'events': [['config', None]]},
 'test_non_string_model_is_400': {'client': 'HTTP/1.1 400 *\r\n'
                                            'Server: *\r\n'
                                            'Date: *\r\n'
                                            'Content-Type: application/json\r\n'
                                            'Content-Length: 110\r\n'
                                            'Cache-Control: no-store\r\n'
                                            '\r\n'
                                            '{"error": {"message": "Invalid model request.", "type": "invalid_request_error", "param": null, "code": null}}',
                                  'closed': False,
                                  'engine': [],
                                  'events': []},
 'test_oxe_backend_error_clears_prefix_and_harness_key': {'client': 'HTTP/1.1 500 *\r\n'
                                                                    'Server: *\r\n'
                                                                    'Date: *\r\n'
                                                                    'Content-Type: application/json\r\n'
                                                                    'Cache-Control: no-store\r\n'
                                                                    'Transfer-Encoding: chunked\r\n'
                                                                    '\r\n'
                                                                    '10\r\n'
                                                                    '{"error":"boom"}\r\n'
                                                                    '0\r\n'
                                                                    '\r\n',
                                                          'closed': False,
                                                          'engine': [['POST /v1/chat/completions HTTP/1.1\r\n'
                                                                      'Accept-Encoding: identity\r\n'
                                                                      'Content-Length: 258\r\n'
                                                                      'Host: 127.0.0.1:ENGINE\r\n'
                                                                      'X-Oxe-Session: ses-hint-1\r\n'
                                                                      'X-Oxe-Turn-Kind: chat',
                                                                      '{"messages": [{"role": "system", "content": "You are an AI agent running in OpenCode, a '
                                                                      'test fixture."}, {"role": "user", "content": "x"}], "temperature": 0, "cache_prompt": '
                                                                      'true, "chat_template_kwargs": {"enable_thinking": false, "preserve_thinking": true}}']],
                                                          'events': [['harness_mu.enter'],
                                                                     ['compat_stats.update', []],
                                                                     ['harness.observe',
                                                                      {'cache_prompt': True,
                                                                       'chat_template_kwargs': {'enable_thinking': False, 'preserve_thinking': True},
                                                                       'messages': [{'content': 'You are an AI agent running in OpenCode, a test fixture.',
                                                                                     'role': 'system'},
                                                                                    {'content': 'x', 'role': 'user'}],
                                                                       'temperature': 0},
                                                                      'opencode',
                                                                      'synthetic-oxe',
                                                                      {'enabled': False, 'min_prefix_chars': 3000, 'min_requests': 3},
                                                                      None],
                                                                     ['harness_mu.exit'],
                                                                     ['Entry',
                                                                      '/v1/chat/completions',
                                                                      {'cache_prompt': True,
                                                                       'chat_template_kwargs': {'enable_thinking': False, 'preserve_thinking': True},
                                                                       'messages': [{'content': 'You are an AI agent running in OpenCode, a test fixture.',
                                                                                     'role': 'system'},
                                                                                    {'content': 'x', 'role': 'user'}],
                                                                       'temperature': 0},
                                                                      None,
                                                                      'synthetic-oxe',
                                                                      True,
                                                                      None,
                                                                      None],
                                                                     ['led.label', 'opencode', 'hk-1'],
                                                                     ['led.label_turn',
                                                                      {'compact_layout': None,
                                                                       'instruction_hash': None,
                                                                       'tools_hash': None,
                                                                       'turn_kind': 'chat'}],
                                                                     ['begin_queued',
                                                                      {'allow_start': True, 'inference': True, 'model_alias': None, 'origin': 'organic'}],
                                                                     ['led.admitted_now', [1.5, 2.5]],
                                                                     ['prepare_prefix',
                                                                      '/v1/chat/completions',
                                                                      {'cache_prompt': True,
                                                                       'chat_template_kwargs': {'enable_thinking': False, 'preserve_thinking': True},
                                                                       'messages': [{'content': 'You are an AI agent running in OpenCode, a test fixture.',
                                                                                     'role': 'system'},
                                                                                    {'content': 'x', 'role': 'user'}],
                                                                       'temperature': 0}],
                                                                     ['watcher.start'],
                                                                     ['current_prefix_key', None],
                                                                     ['led.observe', '{"error":"boom"}'],
                                                                     ['watcher.stop'],
                                                                     ['watcher.stop'],
                                                                     ['led.row.update', {'thermal_c': 41.5}],
                                                                     ['led.finish', 500, None],
                                                                     ['end', {'inference': True, 'touch': False}]]},
 'test_oxe_chat_normalised_with_harness_turn_and_launch_token': {'client': 'HTTP/1.1 200 *\r\n'
                                                                           'Server: *\r\n'
                                                                           'Date: *\r\n'
                                                                           'Content-Type: text/event-stream\r\n'
                                                                           'Cache-Control: no-store\r\n'
                                                                           'Transfer-Encoding: chunked\r\n'
                                                                           '\r\n'
                                                                           '2f\r\n'
                                                                           'data: {"choices":[{"delta":{"content":"a"}}]}\n'
                                                                           '\n'
                                                                           '\r\n'
                                                                           '3c\r\n'
                                                                           'data: {"timings":{"cache_n":3,"prompt_n":5}}\n'
                                                                           '\n'
                                                                           'data: [DONE]\n'
                                                                           '\n'
                                                                           '\r\n'
                                                                           '0\r\n'
                                                                           '\r\n',
                                                                 'closed': False,
                                                                 'engine': [['POST /v1/chat/completions HTTP/1.1\r\n'
                                                                             'Accept-Encoding: identity\r\n'
                                                                             'Content-Length: 301\r\n'
                                                                             'User-Agent: opencode/2\r\n'
                                                                             'X-Session-Id: ses-b\r\n'
                                                                             'Host: 127.0.0.1:ENGINE\r\n'
                                                                             'X-Oxe-Session: ses-hint-1\r\n'
                                                                             'X-Oxe-Turn-Kind: chat\r\n'
                                                                             'Authorization: Bearer lt-1',
                                                                             '{"model": "synthetic-oxe", "stream": true, "messages": [{"role": "system", '
                                                                             '"content": "You are an AI agent running in OpenCode, a test fixture."}, {"role": '
                                                                             '"user", "content": "hi"}], "temperature": 0, "cache_prompt": true, '
                                                                             '"chat_template_kwargs": {"enable_thinking": false, "preserve_thinking": true}}']],
                                                                 'events': [['config', 'synthetic-oxe'],
                                                                            ['config', 'synthetic-oxe'],
                                                                            ['harness_mu.enter'],
                                                                            ['compat_stats.update', ['null_temperature_dropped', 'n_reduced_to_1']],
                                                                            ['harness.observe',
                                                                             {'cache_prompt': True,
                                                                              'chat_template_kwargs': {'enable_thinking': False, 'preserve_thinking': True},
                                                                              'messages': [{'content': 'You are an AI agent running in OpenCode, a test '
                                                                                                       'fixture.',
                                                                                            'role': 'system'},
                                                                                           {'content': 'hi', 'role': 'user'}],
                                                                              'model': 'synthetic-oxe',
                                                                              'stream': True,
                                                                              'temperature': 0},
                                                                             'opencode',
                                                                             'synthetic-oxe',
                                                                             {'enabled': False, 'min_prefix_chars': 3000, 'min_requests': 3},
                                                                             'opencode/2'],
                                                                            ['harness_mu.exit'],
                                                                            ['Entry',
                                                                             '/v1/chat/completions',
                                                                             {'cache_prompt': True,
                                                                              'chat_template_kwargs': {'enable_thinking': False, 'preserve_thinking': True},
                                                                              'messages': [{'content': 'You are an AI agent running in OpenCode, a test '
                                                                                                       'fixture.',
                                                                                            'role': 'system'},
                                                                                           {'content': 'hi', 'role': 'user'}],
                                                                              'model': 'synthetic-oxe',
                                                                              'stream': True,
                                                                              'temperature': 0},
                                                                             'opencode/2',
                                                                             'synthetic-oxe',
                                                                             True,
                                                                             None,
                                                                             None],
                                                                            ['led.label', 'opencode', 'hk-1'],
                                                                            ['led.label_turn',
                                                                             {'compact_layout': None,
                                                                              'instruction_hash': None,
                                                                              'tools_hash': None,
                                                                              'turn_kind': 'chat'}],
                                                                            ['begin_queued',
                                                                             {'allow_start': True,
                                                                              'inference': True,
                                                                              'model_alias': 'synthetic-oxe',
                                                                              'origin': 'organic'}],
                                                                            ['led.admitted_now', [1.5, 2.5]],
                                                                            ['prepare_prefix',
                                                                             '/v1/chat/completions',
                                                                             {'cache_prompt': True,
                                                                              'chat_template_kwargs': {'enable_thinking': False, 'preserve_thinking': True},
                                                                              'messages': [{'content': 'You are an AI agent running in OpenCode, a test '
                                                                                                       'fixture.',
                                                                                            'role': 'system'},
                                                                                           {'content': 'hi', 'role': 'user'}],
                                                                              'model': 'synthetic-oxe',
                                                                              'stream': True,
                                                                              'temperature': 0}],
                                                                            ['watcher.start'],
                                                                            ['led.observe', 'data: {"choices":[{"delta":{"content":"a"}}]}\n\n'],
                                                                            ['led.observe', 'data: {"timings":{"cache_n":3,"prompt_n":5}}\n\ndata: [DONE]\n\n'],
                                                                            ['watcher.stop'],
                                                                            ['harness_mu.enter'],
                                                                            ['harness.complete', 'hk-1', 'synthetic-oxe', [3, 5]],
                                                                            ['harness_mu.exit'],
                                                                            ['watcher.stop'],
                                                                            ['led.row.update', {'thermal_c': 41.5}],
                                                                            ['led.finish', 200, None],
                                                                            ['end', {'inference': True, 'touch': False}]]},
 'test_oxe_completions_non_stream_shim': {'client': 'HTTP/1.1 200 *\r\n'
                                                    'Server: *\r\n'
                                                    'Date: *\r\n'
                                                    'Content-Type: application/json\r\n'
                                                    'Content-Length: 245\r\n'
                                                    'Cache-Control: no-store\r\n'
                                                    '\r\n'
                                                    '{"id": "cmpl-fixed", "object": "text_completion", "created": 1700000000, "model": null, "choices": '
                                                    '[{"text": "lo", "index": 0, "logprobs": null, "finish_reason": "stop"}], "usage": {"prompt_tokens": 2, '
                                                    '"completion_tokens": 1, "total_tokens": 3}}',
                                          'closed': False,
                                          'engine': [['POST /completion HTTP/1.1\r\nAccept-Encoding: identity\r\nContent-Length: 73\r\nHost: 127.0.0.1:ENGINE',
                                                      '{"temperature": 0, "cache_prompt": true, "prompt": "hel", "n_predict": 3}']],
                                          'events': [['harness_mu.enter'],
                                                     ['compat_stats.update', []],
                                                     ['harness_mu.exit'],
                                                     ['Entry',
                                                      '/v1/completions',
                                                      {'cache_prompt': True, 'n_predict': 3, 'prompt': 'hel', 'temperature': 0},
                                                      None,
                                                      'synthetic-oxe',
                                                      True,
                                                      None,
                                                      None],
                                                     ['led.label', None, None],
                                                     ['led.label_turn', None],
                                                     ['begin_queued', {'allow_start': True, 'inference': True, 'model_alias': None, 'origin': 'organic'}],
                                                     ['led.admitted_now', [1.5, 2.5]],
                                                     ['prepare_prefix',
                                                      '/v1/completions',
                                                      {'cache_prompt': True, 'n_predict': 3, 'prompt': 'hel', 'temperature': 0}],
                                                     ['watcher.start'],
                                                     ['led.observe', '{"content":"lo","stop_type":"eos","tokens_evaluated":2,"tokens_predicted":1}'],
                                                     ['watcher.stop'],
                                                     ['watcher.stop'],
                                                     ['led.row.update', {'thermal_c': 41.5}],
                                                     ['led.finish', 200, None],
                                                     ['end', {'inference': True, 'touch': False}]]},
 'test_oxe_completions_stream_shim': {'client': 'HTTP/1.1 200 *\r\n'
                                                'Server: *\r\n'
                                                'Date: *\r\n'
                                                'Content-Type: text/event-stream\r\n'
                                                'Cache-Control: no-store\r\n'
                                                'Transfer-Encoding: chunked\r\n'
                                                '\r\n'
                                                'b1\r\n'
                                                'data: {"id": "cmpl-fixed", "object": "text_completion", "created": 1700000000, "model": null, "choices": '
                                                '[{"text": "pa", "index": 0, "logprobs": null, "finish_reason": null}]}\n'
                                                '\n'
                                                '\r\n'
                                                'dd\r\n'
                                                'data: {"id": "cmpl-fixed", "object": "text_completion", "created": 1700000000, "model": null, "choices": '
                                                '[{"text": "b", "index": 0, "logprobs": null, "finish_reason": "length"}], "timings": {"cache_n": 1}}\n'
                                                '\n'
                                                'data: [DONE]\n'
                                                '\n'
                                                '\r\n'
                                                '7\r\n'
                                                ': bye\n'
                                                '\n'
                                                '\r\n'
                                                '0\r\n'
                                                '\r\n',
                                      'closed': False,
                                      'engine': [['POST /completion HTTP/1.1\r\nAccept-Encoding: identity\r\nContent-Length: 71\r\nHost: 127.0.0.1:ENGINE',
                                                  '{"stream": true, "temperature": 0, "cache_prompt": true, "prompt": "p"}']],
                                      'events': [['harness_mu.enter'],
                                                 ['compat_stats.update', []],
                                                 ['harness_mu.exit'],
                                                 ['Entry',
                                                  '/v1/completions',
                                                  {'cache_prompt': True, 'prompt': 'p', 'stream': True, 'temperature': 0},
                                                  None,
                                                  'synthetic-oxe',
                                                  True,
                                                  None,
                                                  None],
                                                 ['led.label', None, None],
                                                 ['led.label_turn', None],
                                                 ['begin_queued', {'allow_start': True, 'inference': True, 'model_alias': None, 'origin': 'organic'}],
                                                 ['led.admitted_now', [1.5, 2.5]],
                                                 ['prepare_prefix', '/v1/completions', {'cache_prompt': True, 'prompt': 'p', 'stream': True, 'temperature': 0}],
                                                 ['watcher.start'],
                                                 ['led.observe', 'data: {"content":"a","stop":false}\n\n'],
                                                 ['led.observe', 'data: {"content":"b","st'],
                                                 ['led.observe', 'op":true,"timings":{"cache_n":1}}\n\n'],
                                                 ['led.observe', ': bye'],
                                                 ['watcher.stop'],
                                                 ['watcher.stop'],
                                                 ['led.row.update', {'thermal_c': 41.5}],
                                                 ['led.finish', 200, None],
                                                 ['end', {'inference': True, 'touch': False}]]},
 'test_oxe_completions_unsupported_is_400': {'client': 'HTTP/1.1 400 *\r\n'
                                                       'Server: *\r\n'
                                                       'Date: *\r\n'
                                                       'Content-Type: application/json\r\n'
                                                       'Content-Length: 151\r\n'
                                                       'Cache-Control: no-store\r\n'
                                                       '\r\n'
                                                       '{"error": {"message": "\\"suffix\\" (insert mode) is not supported by this model server.", "type": '
                                                       '"invalid_request_error", "param": null, "code": null}}',
                                             'closed': False,
                                             'engine': [],
                                             'events': []},
 'test_oxe_unsupported_endpoint_is_404': {'client': 'HTTP/1.1 404 *\r\n'
                                                    'Server: *\r\n'
                                                    'Date: *\r\n'
                                                    'Content-Type: application/json\r\n'
                                                    'Content-Length: 175\r\n'
                                                    'Cache-Control: no-store\r\n'
                                                    '\r\n'
                                                    '{"error": {"message": "This local model server does not provide embeddings (/v1/embeddings).", "type": '
                                                    '"invalid_request_error", "code": "unsupported_endpoint", "param": null}}',
                                          'closed': False,
                                          'engine': [],
                                          'events': []},
 'test_passthrough_keeps_the_body_bytes_and_filters_headers': {'client': 'HTTP/1.1 200 *\r\n'
                                                                         'Server: *\r\n'
                                                                         'Date: *\r\n'
                                                                         'Content-Type: application/json\r\n'
                                                                         'X-Engine: e1\r\n'
                                                                         'Set-Cookie: s=1\r\n'
                                                                         'Cache-Control: no-store\r\n'
                                                                         'Transfer-Encoding: chunked\r\n'
                                                                         '\r\n'
                                                                         'f\r\n'
                                                                         '{"content":"x"}\r\n'
                                                                         '0\r\n'
                                                                         '\r\n',
                                                               'closed': False,
                                                               'engine': [['POST /completion HTTP/1.1\r\n'
                                                                           'Accept-Encoding: identity\r\n'
                                                                           'Content-Length: 34\r\n'
                                                                           'Content-Type: application/json\r\n'
                                                                           'User-Agent: contract/1\r\n'
                                                                           'X-Custom: 1\r\n'
                                                                           'X-Local-AI-Origin: lab\r\n'
                                                                           'Host: 127.0.0.1:ENGINE',
                                                                           '{"prompt":  "hi",\n "n_predict": 4}']],
                                                               'events': [['Entry',
                                                                           '/completion',
                                                                           {'n_predict': 4, 'prompt': 'hi'},
                                                                           'contract/1',
                                                                           'synthetic-loopback',
                                                                           True,
                                                                           None,
                                                                           None],
                                                                          ['led.label', None, None],
                                                                          ['led.label_turn', None],
                                                                          ['begin_queued',
                                                                           {'allow_start': True, 'inference': True, 'model_alias': None, 'origin': 'lab'}],
                                                                          ['led.admitted_now', [1.5, 2.5]],
                                                                          ['prepare_prefix', '/completion', {'n_predict': 4, 'prompt': 'hi'}],
                                                                          ['watcher.start'],
                                                                          ['led.observe', '{"content":"x"}'],
                                                                          ['watcher.stop'],
                                                                          ['watcher.stop'],
                                                                          ['led.row.update', {'thermal_c': 41.5}],
                                                                          ['led.finish', 200, None],
                                                                          ['end', {'inference': True, 'touch': False}]]},
 'test_remote_llama_payload_ephemeral_and_no_prefix': {'client': 'HTTP/1.1 200 *\r\n'
                                                                 'Server: *\r\n'
                                                                 'Date: *\r\n'
                                                                 'Content-Type: application/json\r\n'
                                                                 'Cache-Control: no-store\r\n'
                                                                 'Transfer-Encoding: chunked\r\n'
                                                                 '\r\n'
                                                                 'b\r\n'
                                                                 '{"ok":true}\r\n'
                                                                 '0\r\n'
                                                                 '\r\n',
                                                       'closed': False,
                                                       'engine': [['POST /v1/chat/completions HTTP/1.1\r\n'
                                                                   'Accept-Encoding: identity\r\n'
                                                                   'Content-Length: 107\r\n'
                                                                   'Host: 127.0.0.1:ENGINE\r\n'
                                                                   'X-Oxe-State-Policy: ephemeral',
                                                                   '{"messages": [{"role": "user", "content": "hi"}], "stream": false, "max_tokens": 64, '
                                                                   '"cache_prompt": false}']],
                                                       'events': [['Entry',
                                                                   '/v1/chat/completions',
                                                                   {'cache_prompt': False,
                                                                    'max_tokens': 64,
                                                                    'messages': [{'content': 'hi', 'role': 'user'}],
                                                                    'stream': False},
                                                                   None,
                                                                   'synthetic-loopback',
                                                                   True,
                                                                   'remote',
                                                                   'peer-1'],
                                                                  ['led.label', None, None],
                                                                  ['led.label_turn', None],
                                                                  ['begin_queued',
                                                                   {'allow_start': False,
                                                                    'inference': True,
                                                                    'model_alias': None,
                                                                    'origin': 'remote',
                                                                    'preempt': '<Preempt>',
                                                                    'remote': True,
                                                                    'wait_seconds': 2.0}],
                                                                  ['led.admitted_now', [1.5, 2.5]],
                                                                  ['current_prefix_key', None],
                                                                  ['watcher.start'],
                                                                  ['led.observe', '{"ok":true}'],
                                                                  ['watcher.stop'],
                                                                  ['watcher.stop'],
                                                                  ['led.row.update', {'thermal_c': 41.5}],
                                                                  ['led.finish', 200, None],
                                                                  ['end', {'inference': True, 'touch': False}]]},
 'test_remote_oxe_has_no_harness_or_session_headers': {'client': 'HTTP/1.1 200 *\r\n'
                                                                 'Server: *\r\n'
                                                                 'Date: *\r\n'
                                                                 'Content-Type: application/json\r\n'
                                                                 'Cache-Control: no-store\r\n'
                                                                 'Transfer-Encoding: chunked\r\n'
                                                                 '\r\n'
                                                                 'b\r\n'
                                                                 '{"ok":true}\r\n'
                                                                 '0\r\n'
                                                                 '\r\n',
                                                       'closed': False,
                                                       'engine': [['POST /v1/chat/completions HTTP/1.1\r\n'
                                                                   'Accept-Encoding: identity\r\n'
                                                                   'Content-Length: 276\r\n'
                                                                   'Host: 127.0.0.1:ENGINE\r\n'
                                                                   'X-Oxe-State-Policy: ephemeral\r\n'
                                                                   'Authorization: Bearer lt-1',
                                                                   '{"messages": [{"role": "system", "content": "You are an AI agent running in OpenCode, a '
                                                                   'test fixture."}, {"role": "user", "content": "hi"}], "max_tokens": 8, "temperature": 0, '
                                                                   '"cache_prompt": true, "chat_template_kwargs": {"enable_thinking": false, '
                                                                   '"preserve_thinking": true}}']],
                                                       'events': [['harness_mu.enter'],
                                                                  ['compat_stats.update', []],
                                                                  ['harness_mu.exit'],
                                                                  ['Entry',
                                                                   '/v1/chat/completions',
                                                                   {'cache_prompt': True,
                                                                    'chat_template_kwargs': {'enable_thinking': False, 'preserve_thinking': True},
                                                                    'max_tokens': 8,
                                                                    'messages': [{'content': 'You are an AI agent running in OpenCode, a test fixture.',
                                                                                  'role': 'system'},
                                                                                 {'content': 'hi', 'role': 'user'}],
                                                                    'temperature': 0},
                                                                   None,
                                                                   'synthetic-oxe',
                                                                   True,
                                                                   'remote',
                                                                   'peer-1'],
                                                                  ['led.label', 'opencode', None],
                                                                  ['led.label_turn',
                                                                   {'compact_layout': None, 'instruction_hash': None, 'tools_hash': None, 'turn_kind': 'chat'}],
                                                                  ['begin_queued',
                                                                   {'allow_start': False,
                                                                    'inference': True,
                                                                    'model_alias': None,
                                                                    'origin': 'remote',
                                                                    'preempt': '<Preempt>',
                                                                    'remote': True,
                                                                    'wait_seconds': 2.0}],
                                                                  ['led.admitted_now', [1.5, 2.5]],
                                                                  ['current_prefix_key', None],
                                                                  ['watcher.start'],
                                                                  ['led.observe', '{"ok":true}'],
                                                                  ['watcher.stop'],
                                                                  ['watcher.stop'],
                                                                  ['led.row.update', {'thermal_c': 41.5}],
                                                                  ['led.finish', 200, None],
                                                                  ['end', {'inference': True, 'touch': False}]]},
 'test_remote_preempted_before_the_response': {'client': 'HTTP/1.1 503 *\r\n'
                                                         'Server: *\r\n'
                                                         'Date: *\r\n'
                                                         'Content-Type: application/json\r\n'
                                                         'Content-Length: 144\r\n'
                                                         'Cache-Control: no-store\r\n'
                                                         'Retry-After: 5\r\n'
                                                         '\r\n'
                                                         '{"error": {"message": "Preempted by local use on the host; retry later.", "type": "preempted", '
                                                         '"code": "preempted_by_local_use", "param": null}}',
                                               'closed': True,
                                               'engine': [],
                                               'events': [['Entry',
                                                           '/v1/chat/completions',
                                                           {'cache_prompt': False, 'messages': [{'content': 'hi', 'role': 'user'}]},
                                                           None,
                                                           'synthetic-loopback',
                                                           True,
                                                           'remote',
                                                           'peer-1'],
                                                          ['led.label', None, None],
                                                          ['led.label_turn', None],
                                                          ['begin_queued',
                                                           {'allow_start': False,
                                                            'inference': True,
                                                            'model_alias': None,
                                                            'origin': 'remote',
                                                            'preempt': '<Preempt>',
                                                            'remote': True,
                                                            'wait_seconds': 2.0}],
                                                          ['led.admitted_now', [1.5, 2.5]],
                                                          ['current_prefix_key', None],
                                                          ['led.row.update', {'thermal_c': 41.5}],
                                                          ['led.finish', 503, 'preempted'],
                                                          ['engine_log',
                                                           '{"error": "preempted", "event": "request_end", "model_alias": "synthetic-loopback", "status": '
                                                           'null}\n'],
                                                          ['end', {'inference': True, 'touch': False}]]},
 'test_remote_preempted_in_a_json_reply': {'client': 'HTTP/1.1 200 *\r\n'
                                                     'Server: *\r\n'
                                                     'Date: *\r\n'
                                                     'Content-Type: application/json\r\n'
                                                     'Cache-Control: no-store\r\n'
                                                     'Transfer-Encoding: chunked\r\n'
                                                     '\r\n'
                                                     'b\r\n'
                                                     '{"partial":\r\n',
                                           'closed': True,
                                           'engine': [['POST /v1/chat/completions HTTP/1.1\r\n'
                                                       'Accept-Encoding: identity\r\n'
                                                       'Content-Length: 72\r\n'
                                                       'Host: 127.0.0.1:ENGINE\r\n'
                                                       'X-Oxe-State-Policy: ephemeral',
                                                       '{"messages": [{"role": "user", "content": "hi"}], "cache_prompt": false}']],
                                           'events': [['Entry',
                                                       '/v1/chat/completions',
                                                       {'cache_prompt': False, 'messages': [{'content': 'hi', 'role': 'user'}]},
                                                       None,
                                                       'synthetic-loopback',
                                                       True,
                                                       'remote',
                                                       'peer-1'],
                                                      ['led.label', None, None],
                                                      ['led.label_turn', None],
                                                      ['begin_queued',
                                                       {'allow_start': False,
                                                        'inference': True,
                                                        'model_alias': None,
                                                        'origin': 'remote',
                                                        'preempt': '<Preempt>',
                                                        'remote': True,
                                                        'wait_seconds': 2.0}],
                                                      ['led.admitted_now', [1.5, 2.5]],
                                                      ['current_prefix_key', None],
                                                      ['watcher.start'],
                                                      ['led.observe', '{"partial":'],
                                                      ['watcher.stop'],
                                                      ['watcher.stop'],
                                                      ['led.row.update', {'thermal_c': 41.5}],
                                                      ['led.finish', 200, 'preempted'],
                                                      ['engine_log',
                                                       '{"error": "preempted", "event": "request_end", "model_alias": "synthetic-loopback", "status": 200}\n'],
                                                      ['end', {'inference': True, 'touch': False}]]},
 'test_remote_preempted_in_a_stream': {'client': 'HTTP/1.1 200 *\r\n'
                                                 'Server: *\r\n'
                                                 'Date: *\r\n'
                                                 'Content-Type: text/event-stream\r\n'
                                                 'Cache-Control: no-store\r\n'
                                                 'Transfer-Encoding: chunked\r\n'
                                                 '\r\n'
                                                 'f\r\n'
                                                 'data: {"n":1}\n'
                                                 '\n'
                                                 '\r\n'
                                                 '89\r\n'
                                                 'data: {"error": {"message": "Preempted by local use on the host; retry later.", "type": "preempted", "code": '
                                                 '"preempted_by_local_use"}}\n'
                                                 '\n'
                                                 '\r\n'
                                                 '0\r\n'
                                                 '\r\n',
                                       'closed': True,
                                       'engine': [['POST /v1/chat/completions HTTP/1.1\r\n'
                                                   'Accept-Encoding: identity\r\n'
                                                   'Content-Length: 88\r\n'
                                                   'Host: 127.0.0.1:ENGINE\r\n'
                                                   'X-Oxe-State-Policy: ephemeral',
                                                   '{"messages": [{"role": "user", "content": "hi"}], "stream": true, "cache_prompt": false}']],
                                       'events': [['Entry',
                                                   '/v1/chat/completions',
                                                   {'cache_prompt': False, 'messages': [{'content': 'hi', 'role': 'user'}], 'stream': True},
                                                   None,
                                                   'synthetic-loopback',
                                                   True,
                                                   'remote',
                                                   'peer-1'],
                                                  ['led.label', None, None],
                                                  ['led.label_turn', None],
                                                  ['begin_queued',
                                                   {'allow_start': False,
                                                    'inference': True,
                                                    'model_alias': None,
                                                    'origin': 'remote',
                                                    'preempt': '<Preempt>',
                                                    'remote': True,
                                                    'wait_seconds': 2.0}],
                                                  ['led.admitted_now', [1.5, 2.5]],
                                                  ['current_prefix_key', None],
                                                  ['watcher.start'],
                                                  ['led.observe', 'data: {"n":1}\n\n'],
                                                  ['watcher.stop'],
                                                  ['watcher.stop'],
                                                  ['led.row.update', {'thermal_c': 41.5}],
                                                  ['led.finish', 200, 'preempted'],
                                                  ['engine_log',
                                                   '{"error": "preempted", "event": "request_end", "model_alias": "synthetic-loopback", "status": 200}\n'],
                                                  ['end', {'inference': True, 'touch': False}]]},
 'test_remote_start_failure_hides_the_cause': {'client': 'HTTP/1.1 503 *\r\n'
                                                         'Server: *\r\n'
                                                         'Date: *\r\n'
                                                         'Content-Type: application/json\r\n'
                                                         'Content-Length: 127\r\n'
                                                         'Cache-Control: no-store\r\n'
                                                         'Retry-After: 5\r\n'
                                                         '\r\n'
                                                         '{"error": {"message": "The host could not start its local model.", "type": "local_backend_error", '
                                                         '"param": null, "code": null}}',
                                               'closed': False,
                                               'engine': [],
                                               'events': [['Entry',
                                                           '/v1/chat/completions',
                                                           {'cache_prompt': False, 'messages': [{'content': 'hi', 'role': 'user'}]},
                                                           None,
                                                           'synthetic-loopback',
                                                           True,
                                                           'remote',
                                                           'peer-1'],
                                                          ['led.label', None, None],
                                                          ['led.label_turn', None],
                                                          ['begin_queued',
                                                           {'allow_start': False,
                                                            'inference': True,
                                                            'model_alias': None,
                                                            'origin': 'remote',
                                                            'preempt': '<Preempt>',
                                                            'remote': True,
                                                            'wait_seconds': 2.0}],
                                                          ['begin_queued.raised', 'RuntimeError'],
                                                          ['led.finish', 503, 'start_failed']]},
 'test_remote_without_body_is_400': {'client': 'HTTP/1.1 400 *\r\n'
                                               'Server: *\r\n'
                                               'Date: *\r\n'
                                               'Content-Type: application/json\r\n'
                                               'Content-Length: 110\r\n'
                                               'Cache-Control: no-store\r\n'
                                               '\r\n'
                                               '{"error": {"message": "Invalid model request.", "type": "invalid_request_error", "param": null, "code": null}}',
                                     'closed': False,
                                     'engine': [],
                                     'events': []},
 'test_request_defaults_seam_is_read_at_call_time': {'client': 'HTTP/1.1 200 *\r\n'
                                                               'Server: *\r\n'
                                                               'Date: *\r\n'
                                                               'Content-Type: application/json\r\n'
                                                               'Cache-Control: no-store\r\n'
                                                               'Transfer-Encoding: chunked\r\n'
                                                               '\r\n'
                                                               'b\r\n'
                                                               '{"ok":true}\r\n'
                                                               '0\r\n'
                                                               '\r\n',
                                                     'closed': False,
                                                     'engine': [['POST /completion HTTP/1.1\r\n'
                                                                 'Accept-Encoding: identity\r\n'
                                                                 'Content-Length: 26\r\n'
                                                                 'Host: 127.0.0.1:ENGINE',
                                                                 '{"prompt": "s", "seam": 1}']],
                                                     'events': [['request_defaults', 'synthetic-loopback', '/completion'],
                                                                ['Entry',
                                                                 '/completion',
                                                                 {'prompt': 's', 'seam': 1},
                                                                 None,
                                                                 'synthetic-loopback',
                                                                 True,
                                                                 None,
                                                                 None],
                                                                ['led.label', None, None],
                                                                ['led.label_turn', None],
                                                                ['begin_queued',
                                                                 {'allow_start': True, 'inference': True, 'model_alias': None, 'origin': 'organic'}],
                                                                ['led.admitted_now', [1.5, 2.5]],
                                                                ['prepare_prefix', '/completion', {'prompt': 's', 'seam': 1}],
                                                                ['watcher.start'],
                                                                ['led.observe', '{"ok":true}'],
                                                                ['watcher.stop'],
                                                                ['watcher.stop'],
                                                                ['led.row.update', {'thermal_c': 41.5}],
                                                                ['led.finish', 200, None],
                                                                ['end', {'inference': True, 'touch': False}]]},
 'test_sse_stream_in_chunks': {'client': 'HTTP/1.1 200 *\r\n'
                                         'Server: *\r\n'
                                         'Date: *\r\n'
                                         'Content-Type: text/event-stream\r\n'
                                         'Cache-Control: no-store\r\n'
                                         'Transfer-Encoding: chunked\r\n'
                                         '\r\n'
                                         'f\r\n'
                                         'data: {"n":1}\n'
                                         '\n'
                                         '\r\n'
                                         'f\r\n'
                                         'data: {"n":2}\n'
                                         '\n'
                                         '\r\n'
                                         'e\r\n'
                                         'data: [DONE]\n'
                                         '\n'
                                         '\r\n'
                                         '0\r\n'
                                         '\r\n',
                               'closed': False,
                               'engine': [['POST /v1/chat/completions HTTP/1.1\r\nAccept-Encoding: identity\r\nContent-Length: 65\r\nHost: 127.0.0.1:ENGINE',
                                           '{"messages": [{"role": "user", "content": "hi"}], "stream": true}']],
                               'events': [['Entry',
                                           '/v1/chat/completions',
                                           {'messages': [{'content': 'hi', 'role': 'user'}], 'stream': True},
                                           None,
                                           'synthetic-loopback',
                                           True,
                                           None,
                                           None],
                                          ['led.label', None, None],
                                          ['led.label_turn', None],
                                          ['begin_queued', {'allow_start': True, 'inference': True, 'model_alias': None, 'origin': 'organic'}],
                                          ['led.admitted_now', [1.5, 2.5]],
                                          ['prepare_prefix', '/v1/chat/completions', {'messages': [{'content': 'hi', 'role': 'user'}], 'stream': True}],
                                          ['watcher.start'],
                                          ['led.observe', 'data: {"n":1}\n\n'],
                                          ['led.observe', 'data: {"n":2}\n\n'],
                                          ['led.observe', 'data: [DONE]\n\n'],
                                          ['watcher.stop'],
                                          ['watcher.stop'],
                                          ['led.row.update', {'thermal_c': 41.5}],
                                          ['led.finish', 200, None],
                                          ['end', {'inference': True, 'touch': False}]]},
 'test_tool_output_cap_failure_forwards_untouched': {'client': 'HTTP/1.1 200 *\r\n'
                                                               'Server: *\r\n'
                                                               'Date: *\r\n'
                                                               'Content-Type: application/json\r\n'
                                                               'Cache-Control: no-store\r\n'
                                                               'Transfer-Encoding: chunked\r\n'
                                                               '\r\n'
                                                               'b\r\n'
                                                               '{"ok":true}\r\n'
                                                               '0\r\n'
                                                               '\r\n',
                                                     'closed': False,
                                                     'engine': [['POST /v1/chat/completions HTTP/1.1\r\n'
                                                                 'Accept-Encoding: identity\r\n'
                                                                 'Content-Length: 49\r\n'
                                                                 'Host: 127.0.0.1:ENGINE',
                                                                 '{"messages":  [{"role": "user", "content": "x"}]}']],
                                                     'events': [['Entry',
                                                                 '/v1/chat/completions',
                                                                 {'messages': [{'content': 'x', 'role': 'user'}]},
                                                                 None,
                                                                 'synthetic-loopback',
                                                                 True,
                                                                 None,
                                                                 None],
                                                                ['led.label', None, None],
                                                                ['led.label_turn', None],
                                                                ['begin_queued',
                                                                 {'allow_start': True, 'inference': True, 'model_alias': None, 'origin': 'organic'}],
                                                                ['led.admitted_now', [1.5, 2.5]],
                                                                ['prepare_prefix', '/v1/chat/completions', {'messages': [{'content': 'x', 'role': 'user'}]}],
                                                                ['watcher.start'],
                                                                ['led.observe', '{"ok":true}'],
                                                                ['watcher.stop'],
                                                                ['watcher.stop'],
                                                                ['led.row.update', {'thermal_c': 41.5}],
                                                                ['led.finish', 200, None],
                                                                ['end', {'inference': True, 'touch': False}]]},
 'test_tool_output_cap_off_forwards_identical_bytes': {'client': 'HTTP/1.1 200 *\r\n'
                                                                 'Server: *\r\n'
                                                                 'Date: *\r\n'
                                                                 'Content-Type: application/json\r\n'
                                                                 'Cache-Control: no-store\r\n'
                                                                 'Transfer-Encoding: chunked\r\n'
                                                                 '\r\n'
                                                                 'b\r\n'
                                                                 '{"ok":true}\r\n'
                                                                 '0\r\n'
                                                                 '\r\n',
                                                       'closed': False,
                                                       'engine': [['POST /v1/chat/completions HTTP/1.1\r\n'
                                                                   'Accept-Encoding: identity\r\n'
                                                                   'Content-Length: 2517\r\n'
                                                                   'Host: 127.0.0.1:ENGINE',
                                                                   '{"messages": [{"role": "user", "content": "x"}, {"role": "tool", "tool_call_id": "c1", '
                                                                   '"content": '
                                                                   '"line\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\nline\\n"}],  '
                                                                   '"tools": []}']],
                                                       'events': [['Entry',
                                                                   '/v1/chat/completions',
                                                                   {'messages': [{'content': 'x', 'role': 'user'},
                                                                                 {'content': 'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n',
                                                                                  'role': 'tool',
                                                                                  'tool_call_id': 'c1'}],
                                                                    'tools': []},
                                                                   None,
                                                                   'synthetic-loopback',
                                                                   True,
                                                                   None,
                                                                   None],
                                                                  ['led.label', None, None],
                                                                  ['led.label_turn', None],
                                                                  ['begin_queued',
                                                                   {'allow_start': True, 'inference': True, 'model_alias': None, 'origin': 'organic'}],
                                                                  ['led.admitted_now', [1.5, 2.5]],
                                                                  ['prepare_prefix',
                                                                   '/v1/chat/completions',
                                                                   {'messages': [{'content': 'x', 'role': 'user'},
                                                                                 {'content': 'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n'
                                                                                             'line\n',
                                                                                  'role': 'tool',
                                                                                  'tool_call_id': 'c1'}],
                                                                    'tools': []}],
                                                                  ['watcher.start'],
                                                                  ['led.observe', '{"ok":true}'],
                                                                  ['watcher.stop'],
                                                                  ['watcher.stop'],
                                                                  ['led.row.update', {'thermal_c': 41.5}],
                                                                  ['led.finish', 200, None],
                                                                  ['end', {'inference': True, 'touch': False}]]},
 'test_tool_output_cap_on_without_changes_adds_no_row': {'client': 'HTTP/1.1 200 *\r\n'
                                                                   'Server: *\r\n'
                                                                   'Date: *\r\n'
                                                                   'Content-Type: application/json\r\n'
                                                                   'Cache-Control: no-store\r\n'
                                                                   'Transfer-Encoding: chunked\r\n'
                                                                   '\r\n'
                                                                   'b\r\n'
                                                                   '{"ok":true}\r\n'
                                                                   '0\r\n'
                                                                   '\r\n',
                                                         'closed': False,
                                                         'engine': [['POST /v1/chat/completions HTTP/1.1\r\n'
                                                                     'Accept-Encoding: identity\r\n'
                                                                     'Content-Length: 48\r\n'
                                                                     'Host: 127.0.0.1:ENGINE',
                                                                     '{"messages": [{"role": "user", "content": "x"}]}']],
                                                         'events': [['Entry',
                                                                     '/v1/chat/completions',
                                                                     {'messages': [{'content': 'x', 'role': 'user'}]},
                                                                     None,
                                                                     'synthetic-loopback',
                                                                     True,
                                                                     None,
                                                                     None],
                                                                    ['led.label', None, None],
                                                                    ['led.label_turn', None],
                                                                    ['begin_queued',
                                                                     {'allow_start': True, 'inference': True, 'model_alias': None, 'origin': 'organic'}],
                                                                    ['led.admitted_now', [1.5, 2.5]],
                                                                    ['prepare_prefix',
                                                                     '/v1/chat/completions',
                                                                     {'messages': [{'content': 'x', 'role': 'user'}]}],
                                                                    ['watcher.start'],
                                                                    ['led.observe', '{"ok":true}'],
                                                                    ['watcher.stop'],
                                                                    ['watcher.stop'],
                                                                    ['led.row.update', {'thermal_c': 41.5}],
                                                                    ['led.finish', 200, None],
                                                                    ['end', {'inference': True, 'touch': False}]]},
 'test_tool_output_cap_rewrite_and_row': {'client': 'HTTP/1.1 200 *\r\n'
                                                    'Server: *\r\n'
                                                    'Date: *\r\n'
                                                    'Content-Type: application/json\r\n'
                                                    'Cache-Control: no-store\r\n'
                                                    'Transfer-Encoding: chunked\r\n'
                                                    '\r\n'
                                                    'b\r\n'
                                                    '{"ok":true}\r\n'
                                                    '0\r\n'
                                                    '\r\n',
                                          'closed': False,
                                          'engine': [['POST /v1/chat/completions HTTP/1.1\r\n'
                                                      'Accept-Encoding: identity\r\n'
                                                      'Content-Length: 65\r\n'
                                                      'X-Session-Id: ses-a\r\n'
                                                      'Host: 127.0.0.1:ENGINE',
                                                      '{"messages": [{"role": "user", "content": "hi"}], "capped": true}']],
                                          'events': [['cap_request', {'messages': [{'content': 'hi', 'role': 'user'}]}, 'ses-a'],
                                                     ['Entry',
                                                      '/v1/chat/completions',
                                                      {'capped': True, 'messages': [{'content': 'hi', 'role': 'user'}]},
                                                      None,
                                                      'synthetic-loopback',
                                                      True,
                                                      None,
                                                      None],
                                                     ['led.label', None, None],
                                                     ['led.label_turn', None],
                                                     ['led.row', 'tool_cap', {'capped': 1, 'hidden_tokens': 900, 'reads': 0, 'reused': 0}],
                                                     ['begin_queued', {'allow_start': True, 'inference': True, 'model_alias': None, 'origin': 'organic'}],
                                                     ['led.admitted_now', [1.5, 2.5]],
                                                     ['prepare_prefix',
                                                      '/v1/chat/completions',
                                                      {'capped': True, 'messages': [{'content': 'hi', 'role': 'user'}]}],
                                                     ['watcher.start'],
                                                     ['led.observe', '{"ok":true}'],
                                                     ['watcher.stop'],
                                                     ['watcher.stop'],
                                                     ['led.row.update', {'thermal_c': 41.5}],
                                                     ['led.finish', 200, None],
                                                     ['end', {'inference': True, 'touch': False}]]},
 'test_transfer_encoding_is_411': {'client': 'HTTP/1.1 411 *\r\n'
                                             'Server: *\r\n'
                                             'Date: *\r\n'
                                             'Content-Type: application/json\r\n'
                                             'Content-Length: 128\r\n'
                                             'Cache-Control: no-store\r\n'
                                             '\r\n'
                                             '{"error": {"message": "Use a content length for request bodies.", "type": "invalid_request_error", "param": '
                                             'null, "code": null}}',
                                   'closed': True,
                                   'engine': [],
                                   'events': []},
 'test_unavailable_is_503': {'client': 'HTTP/1.1 503 *\r\n'
                                       'Server: *\r\n'
                                       'Date: *\r\n'
                                       'Content-Type: application/json\r\n'
                                       'Content-Length: 109\r\n'
                                       'Cache-Control: no-store\r\n'
                                       'Retry-After: 5\r\n'
                                       '\r\n'
                                       '{"error": {"message": "Model files are missing.", "type": "local_backend_busy", "param": null, "code": null}}',
                             'closed': False,
                             'engine': [],
                             'events': [['Entry',
                                         '/v1/chat/completions',
                                         {'messages': [{'content': 'hi', 'role': 'user'}]},
                                         None,
                                         'synthetic-loopback',
                                         True,
                                         None,
                                         None],
                                        ['led.label', None, None],
                                        ['led.label_turn', None],
                                        ['begin_queued', {'allow_start': True, 'inference': True, 'model_alias': None, 'origin': 'organic'}],
                                        ['begin_queued.raised', 'Unavailable'],
                                        ['led.finish', 503, 'unavailable']]},
 'test_unavailable_root_page': {'client': 'HTTP/1.1 503 *\r\n'
                                          'Server: *\r\n'
                                          'Date: *\r\n'
                                          'Content-Type: text/html; charset=utf-8\r\n'
                                          'Content-Length: 402\r\n'
                                          'Cache-Control: no-store\r\n'
                                          'Retry-After: 5\r\n'
                                          '\r\n'
                                          '<!doctype html><meta charset="utf-8"><title>Local Qwen</title>\n'
                                          '<style>body{font:17px system-ui;max-width:650px;margin:12vh '
                                          'auto;padding:20px;background:#171717;color:#eee}a{color:#9dcaff}</style>\n'
                                          '<h1>Local Qwen</h1><p>Setup is &lt;pending&gt;.</p>\n'
                                          '<p><a href="/">Try opening chat again</a> &middot; <a href="/_local/status">Setup status</a> &middot; <a '
                                          'href="http://127.0.0.1:8090">Image studio</a></p>',
                                'closed': False,
                                'engine': [],
                                'events': [['begin_queued', {'allow_start': True, 'inference': False, 'model_alias': None, 'origin': 'organic'}],
                                           ['begin_queued.raised', 'Unavailable']]},
 'test_uncached_get_streams': {'client': 'HTTP/1.1 200 *\r\n'
                                         'Server: *\r\n'
                                         'Date: *\r\n'
                                         'Content-Type: application/json\r\n'
                                         'Cache-Control: no-store\r\n'
                                         'Transfer-Encoding: chunked\r\n'
                                         '\r\n'
                                         'c\r\n'
                                         '{"slots":[]}\r\n'
                                         '0\r\n'
                                         '\r\n',
                               'closed': False,
                               'engine': [['GET /slots HTTP/1.1\r\nAccept-Encoding: identity\r\nHost: 127.0.0.1:ENGINE', '']],
                               'events': [['begin_queued', {'allow_start': False, 'inference': False, 'model_alias': None, 'origin': 'organic'}],
                                          ['end', {'inference': False, 'touch': False}]]},
 'test_unknown_model_is_404_model_not_found': {'client': 'HTTP/1.1 404 *\r\n'
                                                         'Server: *\r\n'
                                                         'Date: *\r\n'
                                                         'Content-Type: application/json\r\n'
                                                         'Content-Length: 133\r\n'
                                                         'Cache-Control: no-store\r\n'
                                                         '\r\n'
                                                         '{"error": {"message": "Unknown model: missing-model.", "type": "invalid_request_error", "param": '
                                                         '"model", "code": "model_not_found"}}',
                                               'closed': False,
                                               'engine': [],
                                               'events': [['config', 'missing-model']]}}

if __name__ == '__main__':
    unittest.main()
