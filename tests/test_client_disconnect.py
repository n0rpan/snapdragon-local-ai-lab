"""Model-free socket and mocked-proxy tests for client cancellation."""
import os as _os, tempfile as _tempfile
_os.environ.setdefault('LOCAL_AI_LEDGER_DIR', _tempfile.mkdtemp(prefix='local-ai-test-ledger-'))  # never the real request ledger
import io
import json
import socket
import struct
import tempfile
import threading
import time
import unittest
from email.message import Message
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from snapdragon_ai_lab.runtime import controller
from snapdragon_ai_lab.runtime.client_disconnect import ClientDisconnect
from snapdragon_ai_lab.runtime.engine_diagnostics import EngineLogWriter


class ClientDisconnectSocketTests(unittest.TestCase):
    def make_pair(self):
        client, peer = socket.socketpair()
        upstream, upstream_peer = socket.socketpair()
        self.addCleanup(client.close)
        self.addCleanup(peer.close)
        self.addCleanup(upstream.close)
        self.addCleanup(upstream_peer.close)
        return client, peer, upstream, upstream_peer

    def make_tcp_client_pair(self):
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(('127.0.0.1', 0))
        listener.listen(1)
        peer = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        peer.connect(listener.getsockname())
        client, _ = listener.accept()
        listener.close()
        self.addCleanup(client.close)
        self.addCleanup(peer.close)
        return client, peer

    def test_eof_before_headers_unblocks_blocked_upstream_read(self):
        client, peer, upstream, upstream_peer = self.make_pair()
        reader_done = threading.Event()

        def read_upstream():
            upstream.recv(1)
            reader_done.set()

        reader_thread = threading.Thread(target=read_upstream, daemon=True)
        reader_thread.start()
        watcher = ClientDisconnect(client, upstream, interval=0.01).start()
        peer.shutdown(socket.SHUT_WR)
        self.assertTrue(watcher.disconnected.wait(1))
        self.assertTrue(reader_done.wait(1))
        watcher.stop()
        reader_thread.join(1)
        self.assertFalse(reader_thread.is_alive())

    def test_eof_during_idle_stream_unblocks_upstream_read(self):
        client, peer, upstream, upstream_peer = self.make_pair()
        received = []
        done = threading.Event()

        def stream_read():
            received.append(upstream.recv(1))
            done.set()

        reader_thread = threading.Thread(target=stream_read, daemon=True)
        reader_thread.start()
        watcher = ClientDisconnect(client, upstream, interval=0.01).start()
        peer.shutdown(socket.SHUT_WR)
        self.assertTrue(watcher.disconnected.wait(1))
        self.assertTrue(done.wait(1))
        self.assertEqual(received, [b''])
        watcher.stop()
        reader_thread.join(1)

    def test_healthy_client_does_not_cancel_upstream(self):
        client, peer, upstream, upstream_peer = self.make_pair()
        watcher = ClientDisconnect(client, upstream, interval=0.01).start()
        upstream_peer.sendall(b'R')
        self.assertEqual(upstream.recv(1), b'R')
        self.assertFalse(watcher.disconnected.wait(0.1))
        watcher.stop()

    def test_pipelined_bytes_are_peeked_but_not_consumed(self):
        client, peer, upstream, upstream_peer = self.make_pair()
        watcher = ClientDisconnect(client, upstream, interval=0.01).start()
        payload = b'GET /next HTTP/1.1\r\nHost: localhost\r\n\r\n'
        peer.sendall(payload)
        deadline = time.monotonic() + 1
        available = b''
        while time.monotonic() < deadline:
            try:
                available = client.recv(len(payload), socket.MSG_PEEK | socket.MSG_DONTWAIT)
                if available:
                    break
            except BlockingIOError:
                pass
            time.sleep(0.01)
        self.assertEqual(available, payload)
        self.assertFalse(watcher.disconnected.is_set())
        watcher.stop()

    def test_stop_join_prevents_old_watcher_cancelling_next_request(self):
        client, peer, upstream, upstream_peer = self.make_pair()
        watcher = ClientDisconnect(client, upstream, interval=0.01).start()
        peer.sendall(b'next request is buffered')
        watcher.stop()
        self.assertFalse(watcher.thread.is_alive())
        peer.shutdown(socket.SHUT_WR)
        upstream_peer.sendall(b'K')
        self.assertEqual(upstream.recv(1), b'K')
        self.assertFalse(watcher.disconnected.is_set())

    def test_client_reset_cancels_upstream(self):
        client, peer = self.make_tcp_client_pair()
        upstream, upstream_peer = socket.socketpair()
        self.addCleanup(upstream.close)
        self.addCleanup(upstream_peer.close)
        watcher = ClientDisconnect(client, upstream, interval=0.01).start()
        peer.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack('ii', 1, 0))
        peer.close()
        self.assertTrue(watcher.disconnected.wait(1))
        watcher.stop()
        upstream.settimeout(1)
        self.assertEqual(upstream.recv(1), b'')


class HandlerDisconnectClassificationTests(unittest.TestCase):
    def run_proxy_failure(self, watcher_disconnected, process_exit, upstream_exception):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        log = Path(tmp.name) / 'engine-stderr.log'
        writer = EngineLogWriter(log)
        config = {'model_alias': 'test-model', 'server_sha256': 'a' * 64,
                  'backend_kind': 'llama'}
        process = SimpleNamespace(pid=4321, poll=lambda: process_exit,
                                   _local_ai_engine_config=config)

        with patch.object(controller.threading.Thread, 'start'):
            manager = controller.Manager()
        manager.engine_log_writer = writer
        manager.process = process
        manager.loaded_config = config
        manager.config = lambda alias=None: config
        manager.begin_queued = lambda **kwargs: process
        manager.prepare_prefix = lambda *args: None
        manager.end_calls = []
        watcher_stopped = [False]

        def end_request(**kwargs):
            self.assertTrue(watcher_stopped[0], 'watcher must join before the FIFO slot is released')
            manager.end_calls.append(kwargs)

        manager.end = end_request

        class WatcherStub:
            def __init__(self, client, upstream):
                self.disconnected = threading.Event()
                if watcher_disconnected:
                    self.disconnected.set()
            def start(self): return self
            def stop(self): watcher_stopped[0] = True

        class BackendStub:
            sock = object()
            def __init__(self, *args, **kwargs): pass
            def request(self, *args, **kwargs): pass
            def getresponse(self): raise upstream_exception
            def close(self): pass

        body = json.dumps({'prompt': 'PRIVATE_HTTP_BODY'}).encode()
        handler = object.__new__(controller.Handler)
        handler.server = SimpleNamespace(manager=manager, server_port=8092)
        handler.connection = object()
        handler.headers = Message()
        handler.headers['Content-Length'] = str(len(body))
        handler.headers['Host'] = '127.0.0.1:8092'  # F80 Host allowlist
        handler.command = 'POST'
        handler.path = '/completion'
        handler.rfile = io.BytesIO(body)
        handler.close_connection = False
        replies = []
        handler.json_reply = lambda code, value: replies.append((code, value))

        with patch.object(controller, 'ClientDisconnect', WatcherStub), \
                patch.object(controller.http.client, 'HTTPConnection', BackendStub):
            handler.forward()
        events = [json.loads(line) for line in log.read_text().splitlines()]
        return replies, events, handler, manager

    def test_client_eof_is_client_disconnected_not_backend_died(self):
        replies, events, handler, manager = self.run_proxy_failure(
            True, None, ConnectionResetError('upstream was shut down by client watcher'))
        self.assertEqual(replies, [])
        self.assertTrue(handler.close_connection)
        self.assertEqual([event['error'] for event in events], ['client_disconnected'])
        self.assertEqual(len(manager.end_calls), 1)
        self.assertNotIn('PRIVATE_HTTP_BODY', json.dumps(events))

    def test_remote_disconnected_with_dead_native_returns_backend_died_502(self):
        replies, events, _handler, _manager = self.run_proxy_failure(
            False, -9, controller.http.client.RemoteDisconnected('backend closed before response'))
        self.assertEqual(replies[0][0], 502)
        self.assertEqual(replies[0][1]['error']['type'], 'backend_died')
        self.assertEqual([event['event'] for event in events], ['engine_died', 'request_end'])
        self.assertEqual(events[-1]['error'], 'backend_died')
        self.assertNotIn('PRIVATE_HTTP_BODY', json.dumps(events))

    def test_native_exit_after_stream_body_omits_success_terminator(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        log = Path(tmp.name) / 'engine-stderr.log'
        writer = EngineLogWriter(log)
        config = {'model_alias': 'test-model', 'server_sha256': 'b' * 64,
                  'backend_kind': 'llama'}
        process = SimpleNamespace(pid=5432, poll=lambda: -9,
                                   _local_ai_engine_config=config)
        with patch.object(controller.threading.Thread, 'start'):
            manager = controller.Manager()
        manager.engine_log_writer = writer
        manager.process = process
        manager.loaded_config = config
        manager.config = lambda alias=None: config
        manager.begin_queued = lambda **kwargs: process
        manager.prepare_prefix = lambda *args: None
        manager.end = lambda **kwargs: None

        class WatcherStub:
            def __init__(self, client, upstream):
                self.disconnected = threading.Event()
            def start(self): return self
            def stop(self): pass

        class StreamResponse:
            status = 200
            def __init__(self): self.chunks = iter((b'answer chunk', b''))
            def getheader(self, name, default=None): return default
            def getheaders(self): return [('Content-Type', 'text/plain')]
            def read1(self, _size): return next(self.chunks)

        class BackendStub:
            sock = object()
            def __init__(self, *args, **kwargs): pass
            def request(self, *args, **kwargs): pass
            def getresponse(self): return StreamResponse()
            def close(self): pass

        body = b'{"prompt":"PRIVATE_HTTP_BODY"}'
        handler = object.__new__(controller.Handler)
        handler.server = SimpleNamespace(manager=manager, server_port=8092)
        handler.connection = object()
        handler.headers = Message()
        handler.headers['Content-Length'] = str(len(body))
        handler.headers['Host'] = '127.0.0.1:8092'  # F80 Host allowlist
        handler.command = 'POST'
        handler.path = '/completion'
        handler.rfile = io.BytesIO(body)
        handler.wfile = io.BytesIO()
        handler.close_connection = False
        handler.send_response = lambda *_args, **_kwargs: None
        handler.send_header = lambda *_args, **_kwargs: None
        handler.end_headers = lambda: None
        replies = []
        handler.json_reply = lambda code, value: replies.append((code, value))

        with patch.object(controller, 'ClientDisconnect', WatcherStub), \
                patch.object(controller.http.client, 'HTTPConnection', BackendStub):
            handler.forward()
        self.assertEqual(replies, [])
        self.assertTrue(handler.close_connection)
        self.assertEqual(handler.wfile.getvalue(), b'c\r\nanswer chunk\r\n')
        events = [json.loads(line) for line in log.read_text().splitlines()]
        self.assertEqual([event['event'] for event in events], ['engine_died', 'request_end'])
        self.assertEqual(events[-1]['error'], 'backend_died')
        self.assertNotIn(b'PRIVATE_HTTP_BODY', log.read_bytes())


if __name__ == '__main__':
    unittest.main()
