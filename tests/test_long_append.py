"""v0.1.3 LONG-APPEND (steering 11-12 evening #7): long prefills never silently fail or retry.

Model-free, loopback only: a fake slow engine streams its HTTP head at once, then the oxe
keepalive comments (`: prefill done/total Ns`) while it "prefills", then the reply. The controller
must forward the comments and keep the stream alive past its socket read timeout (it is an idle
timeout), the ledger's TTFB must not count a comment, and the OpenCode overlay must carry waits at
least as long as the controller's (the controller never gives up first).
"""
import os as _os, tempfile as _tempfile
_os.environ.setdefault('LOCAL_AI_LEDGER_DIR', _tempfile.mkdtemp(prefix='local-ai-test-ledger-'))  # never the real request ledger
import http.client
import json
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from snapdragon_ai_lab.runtime import controller, request_ledger
from snapdragon_ai_lab.runtime import launch_local_plan as launcher
from snapdragon_ai_lab.runtime.engine_diagnostics import EngineLogWriter

CHUNK = {'id': 'c', 'object': 'chat.completion.chunk', 'created': 1, 'model': 'm',
         'choices': [{'index': 0, 'delta': {'content': 'ok'}, 'finish_reason': None}]}
LAST = {'id': 'c', 'object': 'chat.completion.chunk', 'created': 1, 'model': 'm',
        'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]}


class Backend(ThreadingHTTPServer):
    daemon_threads = True


class SlowEngineThroughController(unittest.TestCase):
    """A fake oxe engine whose 'prefill' takes `prefill_s`, with or without keepalive comments."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.prefill_s, self.every_s, self.keepalive = 1.2, 0.1, True
        test_case = self

        class SlowEngine(BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'

            def log_message(self, *_args):
                pass

            def do_POST(self):
                self.rfile.read(int(self.headers['Content-Length']))
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.send_header('Connection', 'close')
                self.end_headers()
                self.wfile.flush()
                self.close_connection = True
                t0 = time.monotonic()
                total = 28500
                try:
                    while time.monotonic() - t0 < test_case.prefill_s:
                        time.sleep(test_case.every_s)
                        if test_case.keepalive:
                            done = int(total * min(1.0, (time.monotonic() - t0) / test_case.prefill_s))
                            self.wfile.write(f': prefill {done}/{total} {time.monotonic() - t0:.0f}s\n\n'.encode())
                            self.wfile.flush()
                    for frame in (CHUNK, LAST):
                        self.wfile.write(b'data: ' + json.dumps(frame).encode() + b'\n\n')
                    self.wfile.write(b'data: [DONE]\n\n')
                    self.wfile.flush()
                except OSError:
                    pass

        self.backend = Backend(('127.0.0.1', 0), SlowEngine)
        threading.Thread(target=self.backend.serve_forever, daemon=True).start()
        self.addCleanup(self.backend.server_close)
        self.addCleanup(self.backend.shutdown)

        process = SimpleNamespace(pid=9999, poll=lambda: None)
        config = {'model_alias': 'synthetic-loopback', 'backend_kind': 'llama', 'server_sha256': 'a' * 64}
        process._local_ai_engine_config = config
        self.ended = threading.Event()

        class FakeManager:
            pass

        manager = FakeManager()
        manager.backend_runtime = None
        manager.current_prefix_key = None
        manager.cache = {}
        manager.loaded_config = config
        manager.process = process
        manager.engine_log_writer = EngineLogWriter(Path(self.tmp.name) / 'engine-stderr.log')
        manager.config = lambda _alias=None: config
        manager.begin_queued = lambda **_kwargs: process
        manager.prepare_prefix = lambda *_args: None
        manager.end = lambda **_kwargs: self.ended.set()
        manager.record_engine_death = lambda *_args: None
        self.proxy = controller.Server(('127.0.0.1', 0), controller.Handler)
        self.proxy.manager = manager
        threading.Thread(target=self.proxy.serve_forever, daemon=True).start()
        self.addCleanup(self.proxy.server_close)
        self.addCleanup(self.proxy.shutdown)
        for item in (patch.object(controller, 'BACKEND_PORT', self.backend.server_port),
                     # the controller's read timeout, scaled down: 0.5 s against a 1.2 s prefill
                     patch.object(controller, 'BACKEND_READ_TIMEOUT_SECONDS', 0.5)):
            item.start()
            self.addCleanup(item.stop)

    def stream(self):
        client = http.client.HTTPConnection('127.0.0.1', self.proxy.server_port, timeout=10)
        self.addCleanup(client.close)
        body = json.dumps({'prompt': 'x', 'stream': True}).encode()
        t0 = time.monotonic()
        client.request('POST', '/completion', body=body, headers={'Content-Type': 'application/json'})
        response = client.getresponse()
        t_head = time.monotonic() - t0
        lines, data = [], b''
        try:
            while chunk := response.read1(65536):
                data += chunk
        except (OSError, http.client.HTTPException):
            pass
        lines = [line for line in data.decode().split('\n') if line.strip()]
        return response.status, t_head, lines, time.monotonic() - t0

    def test_keepalive_comments_pass_through_and_keep_a_long_prefill_alive(self):
        status, t_head, lines, total = self.stream()
        self.assertEqual(status, 200)
        self.assertGreater(total, self.prefill_s)
        self.assertLess(t_head, 0.5, 'the head must reach the client before the prefill ends')
        comments = [line for line in lines if line.startswith(': prefill ')]
        self.assertGreaterEqual(len(comments), 5, lines)
        first_data = next(i for i, line in enumerate(lines) if line.startswith('data: '))
        self.assertTrue(all(line.startswith(':') for line in lines[:first_data]), lines)
        self.assertEqual(lines[-1], 'data: [DONE]', 'the stream outlived the 0.5 s read timeout')
        self.assertTrue(self.ended.wait(2))

    def test_without_keepalive_the_same_prefill_hits_the_read_timeout(self):
        # the failure this ticket removes: an idle stream longer than the controller's read timeout
        self.keepalive = False
        status, _, lines, _ = self.stream()
        self.assertEqual(status, 200)
        self.assertNotIn('data: [DONE]', lines)
        self.assertTrue(self.ended.wait(2))


class LedgerAndTimeouts(unittest.TestCase):
    def test_comment_chunks_are_not_the_first_byte(self):
        self.assertTrue(request_ledger.is_comment_only(b': prefill 4096/28500 120s\n\n'))
        self.assertTrue(request_ledger.is_comment_only(b': queued 3s\n\n: prefill 0/10 15s\n\n'))
        self.assertFalse(request_ledger.is_comment_only(b': prefill 1/2 1s\n\ndata: {}\n\n'))
        self.assertFalse(request_ledger.is_comment_only(b'data: {"x":1}\n\n'))
        entry = request_ledger.Entry('/v1/chat/completions', {'messages': []})
        entry.observe(b': prefill 4096/28500 120s\n\n')
        self.assertIsNone(entry.first_byte)
        entry.observe(b'data: ' + json.dumps(CHUNK).encode() + b'\n\n')
        self.assertIsNotNone(entry.first_byte)

    def test_opencode_waits_cover_the_longest_prefill_and_the_controller_never_gives_up_first(self):
        # measured: a 28.5K append at ~83K depth ran > 600 s (24-34 tok/s: ~950-1200 s); OpenCode
        # 2.2.7 defaults are headerTimeout 300000 and chunkTimeout 900000 (only data events reset it)
        for model in launcher.MOE_PRESET_MODELS:
            for mode in launcher.OPENCODE_MODE_INPUT:
                with self.subTest(model=model, mode=mode):
                    value = json.loads(launcher.environment_for_classic_model({}, model, mode)['OPENCODE_CONFIG_CONTENT'])
                    settings = value['providers']['local-qwen']['settings']
                    self.assertGreaterEqual(settings['chunkTimeout'], 1_800_000)
                    self.assertGreaterEqual(settings['headerTimeout'], 1_800_000)
                    self.assertNotIn('timeout', settings)
                    self.assertGreaterEqual(controller.BACKEND_READ_TIMEOUT_SECONDS * 1000, settings['chunkTimeout'])
        # a longer (or disabled) wait the user already set is kept; a shorter one is raised
        for given, want in ((7_200_000, 7_200_000), (False, False), (900_000, launcher.MOE_CHUNK_TIMEOUT_MS), ('x', launcher.MOE_CHUNK_TIMEOUT_MS)):
            incoming = {'providers': {'local-qwen': {'settings': {'baseURL': 'http://127.0.0.1:8092/v1', 'chunkTimeout': given}}}}
            value = json.loads(launcher.environment_for_classic_model({'OPENCODE_CONFIG_CONTENT': json.dumps(incoming)}, 'ornith', 'fast')['OPENCODE_CONFIG_CONTENT'])
            self.assertEqual(value['providers']['local-qwen']['settings']['chunkTimeout'], want)
            self.assertEqual(value['providers']['local-qwen']['settings']['baseURL'], 'http://127.0.0.1:8092/v1')
        # Flash / dense launches are unchanged
        flash = json.loads(launcher.environment_for_classic_model({}, 'flash')['OPENCODE_CONFIG_CONTENT'])
        self.assertNotIn('settings', flash.get('providers', {}).get('local-qwen', {}))


if __name__ == '__main__':
    unittest.main()
