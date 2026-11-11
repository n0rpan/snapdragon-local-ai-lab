"""Loopback HTTP cancellation tests with a slow, model-free backend."""
import os as _os, tempfile as _tempfile
_os.environ.setdefault('LOCAL_AI_LEDGER_DIR', _tempfile.mkdtemp(prefix='local-ai-test-ledger-'))  # never the real request ledger
import http.client
import json
import select
import socket
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from snapdragon_ai_lab.runtime import controller
from snapdragon_ai_lab.runtime.client_disconnect import ClientDisconnect
from snapdragon_ai_lab.runtime.engine_diagnostics import EngineLogWriter


_HTTPConnection = http.client.HTTPConnection


class Backend(ThreadingHTTPServer):
    daemon_threads = True


class ControllerDisconnectHTTPTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.backend_done = threading.Event()
        self.backend_started = threading.Event()
        self.headers_sent = threading.Event()
        self.received_body = None
        self.mode = 'before_headers'
        test_case = self

        class SlowHandler(BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'

            def log_message(self, *_args):
                pass

            def do_POST(self):
                test_case.received_body = self.rfile.read(int(self.headers['Content-Length']))
                test_case.backend_started.set()
                if test_case.mode == 'stream':
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/event-stream')
                    self.send_header('Connection', 'close')
                    self.end_headers()
                    self.wfile.flush()
                    self.close_connection = True
                    test_case.headers_sent.set()
                deadline = time.monotonic() + 3
                while time.monotonic() < deadline:
                    readable, _, _ = select.select([self.connection], [], [], 0.02)
                    if not readable:
                        continue
                    try:
                        data = self.connection.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT)
                    except OSError:
                        test_case.backend_done.set()
                        return
                    if not data:
                        test_case.backend_done.set()
                        return
                test_case.backend_done.set()

        self.backend = Backend(('127.0.0.1', 0), SlowHandler)
        self.backend_thread = threading.Thread(target=self.backend.serve_forever, daemon=True)
        self.backend_thread.start()
        self.addCleanup(self.backend.server_close)
        self.addCleanup(self.backend.shutdown)

        self.watchers = []
        self.connection_wrappers = []

        class TrackingDisconnect(ClientDisconnect):
            def __init__(inner_self, client, upstream):
                super().__init__(client, upstream)
                inner_self.joined = threading.Event()
                test_case.watchers.append(inner_self)

            def stop(inner_self):
                super().stop()
                inner_self.joined.set()

        class TrackingHTTPConnection(_HTTPConnection):
            def __init__(inner_self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                test_case.connection_wrappers.append(inner_self)

            def getresponse(inner_self):
                response = super().getresponse()
                inner_self.sock_after_getresponse = inner_self.sock
                return response

        log = Path(self.tmp.name) / 'engine-stderr.log'
        process = SimpleNamespace(pid=9999, poll=lambda: None)
        config = {'model_alias': 'synthetic-loopback', 'backend_kind': 'llama',
                  'server_sha256': 'a' * 64}
        process._local_ai_engine_config = config
        self.end_calls = []
        self.ended = threading.Event()

        class FakeManager:
            pass

        self.manager = FakeManager()
        self.manager.backend_runtime = None
        self.manager.current_prefix_key = None
        self.manager.cache = {}
        self.manager.loaded_config = config
        self.manager.process = process
        self.manager.engine_log_writer = EngineLogWriter(log)
        self.manager.config = lambda _alias=None: config
        self.manager.begin_queued = lambda **_kwargs: process
        self.manager.prepare_prefix = lambda *_args: None

        def end_request(**kwargs):
            watcher = self.watchers[-1] if self.watchers else None
            self.end_calls.append((kwargs, watcher is not None
                and watcher.joined.is_set() and not watcher.thread.is_alive()))
            self.ended.set()

        self.manager.end = end_request
        self.manager.record_engine_death = lambda *_args: (_ for _ in ()).throw(
            AssertionError('synthetic backend process should stay alive'))
        self.proxy = controller.Server(('127.0.0.1', 0), controller.Handler)
        self.proxy.manager = self.manager
        self.proxy_thread = threading.Thread(target=self.proxy.serve_forever, daemon=True)
        self.proxy_thread.start()
        self.addCleanup(self.proxy.server_close)
        self.addCleanup(self.proxy.shutdown)

        self.patches = [
            patch.object(controller, 'BACKEND_PORT', self.backend.server_port),
            patch.object(controller, 'ClientDisconnect', TrackingDisconnect),
            patch.object(controller.http.client, 'HTTPConnection', TrackingHTTPConnection),
        ]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)

    def client_request(self):
        client = _HTTPConnection('127.0.0.1', self.proxy.server_port, timeout=3)
        self.addCleanup(client.close)
        body = b'{"prompt":"PRIVATE_HTTP_BODY"}'
        client.request('POST', '/completion', body=body,
                       headers={'Content-Type': 'application/json'})
        return client

    def assert_request_end_is_private_and_joined(self, expected_status):
        self.assertTrue(self.ended.wait(2), 'proxy handler did not finish promptly')
        self.assertEqual(len(self.end_calls), 1)
        kwargs, joined_before_end = self.end_calls[0]
        self.assertTrue(kwargs.get('inference'))
        self.assertTrue(joined_before_end, 'manager.end ran before the watcher joined')
        log = Path(self.tmp.name) / 'engine-stderr.log'
        events = [json.loads(line) for line in log.read_text().splitlines()]
        self.assertEqual(events[-1]['event'], 'request_end')
        self.assertEqual(events[-1]['error'], 'client_disconnected')
        self.assertEqual(events[-1]['status'], expected_status)
        self.assertNotIn(b'PRIVATE_HTTP_BODY', log.read_bytes())

    def test_client_eof_closes_backend_while_waiting_before_headers(self):
        self.mode = 'before_headers'
        client = self.client_request()
        self.assertTrue(self.backend_started.wait(1))
        self.assertFalse(self.backend_done.is_set())
        self.assertEqual(self.received_body, b'{"prompt":"PRIVATE_HTTP_BODY"}')
        client.close()
        self.assertTrue(self.backend_done.wait(2), 'client EOF did not close the upstream connection')
        self.assert_request_end_is_private_and_joined(None)

    def test_client_eof_closes_connection_close_stream_while_waiting_for_token(self):
        self.mode = 'stream'
        client = self.client_request()
        response = client.getresponse()
        self.assertEqual(response.status, 200)
        self.assertTrue(self.headers_sent.wait(1))
        self.assertTrue(self.connection_wrappers)
        upstream = self.connection_wrappers[0]
        self.assertIsNone(upstream.sock_after_getresponse,
                          'Connection: close should clear HTTPConnection.sock after headers')
        watcher = self.watchers[0]
        self.assertIsNotNone(watcher.upstream)
        client.close()
        self.assertTrue(self.backend_done.wait(2), 'idle stream EOF did not close the captured upstream socket')
        self.assert_request_end_is_private_and_joined(200)


if __name__ == '__main__':
    unittest.main()
