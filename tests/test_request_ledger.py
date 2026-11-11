"""Request ledger: numbers only, private files, backend timings parsed from the response tail."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import http.client
import json
import os
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from snapdragon_ai_lab.runtime import controller, request_ledger

MARKER = 'prompt-text-marker-q7'
SSE = (b'data: {"choices":[{"delta":{"role":"assistant","content":null}}]}\n\n'
       b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n'
       b'data: {"choices":[{"finish_reason":"tool_calls","index":0,"delta":{}}],"timings":{"cache_n":5120,"prompt_n":312,'
       b'"prompt_ms":1712.5,"predicted_n":64,"predicted_ms":1490.0,"draft_n":10,"draft_n_accepted":7}}\n\n'
       b'data: [DONE]\n\n')


class LedgerUnitTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.env = patch.dict(os.environ, {'LOCAL_AI_LEDGER_DIR': str(self.tmp / 'ledger'), 'LOCAL_AI_LEDGER': '1'})
        self.env.start()

    def tearDown(self):
        self.env.stop()

    def test_parse_tail_sse_and_json(self):
        self.assertEqual(request_ledger.parse_tail(SSE)['prompt_n'], 312)
        self.assertEqual(request_ledger.parse_tail(SSE)['finish_reason'], 'tool_calls')
        body = b'{"choices":[{"finish_reason":"stop","message":{"content":"x"}}],"usage":{},"timings":{"cache_n":0,"prompt_n":9,"prompt_ms":5.0,"predicted_n":3,"predicted_ms":4.0}}'
        t = request_ledger.parse_tail(body)
        self.assertEqual((t['prompt_n'], t['predicted_n'], t['finish_reason']), (9, 3, 'stop'))
        self.assertEqual(request_ledger.parse_tail(b'garbage'), {})

    def test_entry_has_no_content_and_files_are_private(self):
        payload = {'messages': [{'role': 'system', 'content': MARKER}, {'role': 'user', 'content': MARKER}],
                   'tools': [{'type': 'function'}] * 3, 'stream': True, 'max_tokens': 2048, 'temperature': 0}
        e = request_ledger.Entry('/v1/chat/completions', payload, 'opencode/2.2.6 (linux; ' + MARKER + ')', 'ornith')
        e.admitted_now(0.25, 0.5)
        e.observe(SSE[:40])
        e.observe(SSE[40:])
        row = e.finish(200)
        files = list((self.tmp / 'ledger').glob('requests-*.jsonl'))
        self.assertEqual(len(files), 1)
        text = files[0].read_text()
        self.assertNotIn(MARKER, text)
        self.assertEqual(oct(files[0].stat().st_mode & 0o777), '0o600')
        self.assertEqual(oct((self.tmp / 'ledger').stat().st_mode & 0o777), '0o700')
        got = json.loads(text.splitlines()[0])
        self.assertEqual(got['client'], 'opencode/2.2.6')
        self.assertEqual((got['n_messages'], got['n_tools'], got['last_role'], got['max_tokens']), (2, 3, 'user', 2048))
        self.assertEqual((got['queue_wait_ms'], got['admit_ms'], got['status']), (250.0, 500.0, 200))
        self.assertEqual((got['cache_n'], got['prompt_n'], got['draft_n_accepted'], got['finish_reason']), (5120, 312, 7, 'tool_calls'))
        self.assertEqual(len(got['agent']), 12)
        self.assertEqual(row, got)

    def test_harness_labels(self):
        e = request_ledger.Entry('/v1/chat/completions', {}, None)
        e.label(harness='opencode', prefix_key='ab' * 32)
        e.label(harness='bad name with spaces')
        row = e.finish(200)
        self.assertEqual((row['harness'], row['prefix_key']), ('opencode', 'ab' * 8))

    def test_disabled(self):
        with patch.dict(os.environ, {'LOCAL_AI_LEDGER': '0'}):
            request_ledger.Entry('/completion', {}, None).finish(200)
        self.assertFalse((self.tmp / 'ledger').exists())


class LedgerThroughProxyTest(unittest.TestCase):
    """One streamed chat request through the controller's forward(): the ledger line has the
    queue/admit times from begin_queued and the backend's timings, and the client gets the
    stream unchanged."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.env = patch.dict(os.environ, {'LOCAL_AI_LEDGER_DIR': str(self.tmp), 'LOCAL_AI_LEDGER': '1'})
        self.env.start()

        class Backend(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                self.rfile.read(int(self.headers['Content-Length']))
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.end_headers()
                self.wfile.write(SSE)
                self.close_connection = True
        self.backend = ThreadingHTTPServer(('127.0.0.1', 0), Backend)
        threading.Thread(target=self.backend.serve_forever, daemon=True).start()
        self.ports = patch.object(controller, 'BACKEND_PORT', self.backend.server_port)
        self.ports.start()
        profile = {'model_alias': 'synthetic-moe', 'backend_kind': 'oxe'}
        self.manager = SimpleNamespace(config=lambda *a: profile, loaded_config=profile, backend_runtime=None,
                                       begin_queued=lambda **k: None, end=lambda **k: None,
                                       admission_times=(0.125, 0.0625),
                                       harness_mu=threading.Lock(), compat_stats={},
                                       harness_registry=SimpleNamespace(observe=lambda *a, **k: None),
                                       eligible=lambda c: True, prepare_prefix=lambda *a: None,
                                       current_prefix_key=None, cache={})
        self.proxy = controller.Server(('127.0.0.1', 0), controller.Handler)
        self.proxy.manager = self.manager
        threading.Thread(target=self.proxy.serve_forever, daemon=True).start()

    def tearDown(self):
        self.proxy.shutdown(); self.proxy.server_close()
        self.backend.shutdown(); self.backend.server_close()
        self.ports.stop(); self.env.stop()

    def test_stream_is_logged(self):
        c = http.client.HTTPConnection('127.0.0.1', self.proxy.server_port, timeout=5)
        body = {'model': 'synthetic-moe', 'stream': True, 'messages': [{'role': 'system', 'content': MARKER}, {'role': 'user', 'content': MARKER}]}
        c.request('POST', '/v1/chat/completions', json.dumps(body), {'Content-Type': 'application/json', 'User-Agent': 'opencode/2.2.6'})
        r = c.getresponse()
        data = r.read()
        c.close()
        self.assertEqual(r.status, 200)
        self.assertEqual(data, SSE)
        import time
        deadline = time.time() + 3
        while True:  # the line is written after the last chunk reaches the client
            lines = [json.loads(l) for f in self.tmp.glob('requests-*.jsonl') for l in f.read_text().splitlines()]
            if lines or time.time() > deadline:
                break
            time.sleep(0.01)
        self.assertEqual(len(lines), 1)
        row = lines[0]
        self.assertEqual((row['status'], row['queue_wait_ms'], row['admit_ms'], row['prompt_n'], row['finish_reason']), (200, 125.0, 62.5, 312, 'tool_calls'))
        self.assertIn('ttfb_ms', row)
        self.assertNotIn(MARKER, json.dumps(row))


if __name__ == '__main__':
    unittest.main()
