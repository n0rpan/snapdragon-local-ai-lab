"""F80 controller security: Host allowlist, browser Origin rule, and queue release on error.

Real controller.Server on 127.0.0.1 port 0 with a fake manager (SimpleNamespace) and a tiny
synthetic backend. No model, no engine, no real runtime.json (CONFIG is patched to a temp file).
"""
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from snapdragon_ai_lab.runtime import controller

TIMEOUT = 5
CHAT = '/v1/chat/completions'


class SecurityBase(unittest.TestCase):
    def setUp(self):
        self.seen = []
        seen = self.seen

        class Native(BaseHTTPRequestHandler):
            def log_message(self, *args): pass
            def do_POST(self):
                self.rfile.read(int(self.headers['Content-Length']))
                seen.append(self.headers.get('Host'))
                data = b'{"choices":[{"message":{"content":"READY"}}]}'
                self.send_response(200); self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(data))); self.end_headers(); self.wfile.write(data)

        self.backend = ThreadingHTTPServer(('127.0.0.1', 0), Native)
        self.backend_thread = threading.Thread(target=self.backend.serve_forever, daemon=True)
        self.backend_thread.start()
        self.addCleanup(self._stop_backend)

        # The API key comes from the environment or runtime.json; keep both empty.
        env = patch.dict(os.environ)
        env.start(); self.addCleanup(env.stop)
        os.environ.pop(controller.openai_api.KEY_ENV, None)

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        config = Path(tmp.name) / 'runtime.json'
        config.write_text('{}')
        self.config_patch = patch.object(controller, 'CONFIG', config); self.config_patch.start()
        self.addCleanup(self.config_patch.stop)
        self.port_patch = patch.object(controller, 'BACKEND_PORT', self.backend.server_port); self.port_patch.start()
        self.addCleanup(self.port_patch.stop)

        self.manager = self.make_manager()
        self.proxy = controller.Server(('127.0.0.1', 0), controller.Handler)
        self.proxy.manager = self.manager
        self.proxy_thread = threading.Thread(target=self.proxy.serve_forever, daemon=True)
        self.proxy_thread.start()
        self.addCleanup(self._stop_proxy)
        self.port = self.proxy.server_address[1]

    def _stop_backend(self):
        self.backend.shutdown(); self.backend.server_close(); self.backend_thread.join(TIMEOUT)

    def _stop_proxy(self):
        self.proxy.shutdown(); self.proxy.server_close(); self.proxy_thread.join(TIMEOUT)

    def make_manager(self, **overrides):
        # Only what forward() touches before and around admission for a classic (non-Flash) model.
        profile = {'model_alias': 'synthetic-classic'}
        fields = dict(
            config=Mock(return_value=profile),
            loaded_config=None,
            backend_runtime=None,
            status=Mock(return_value={'state': 'unloaded', 'validated': False, 'model': None}),
            models=Mock(return_value=[]),
            begin=Mock(return_value=None),
            begin_queued=Mock(return_value=None),
            end=Mock(return_value=None),
            prepare_prefix=Mock(return_value=None),
            eligible=Mock(return_value=True),
            current_prefix_key=None,
            cache={},
        )
        fields.update(overrides)
        return SimpleNamespace(**fields)

    def send(self, method, path, host=None, hosts=None, body=None, headers=None):
        """One request. `host` sets a single Host header; `hosts` sends each listed value as its
        own Host header (no automatic one)."""
        connection = http.client.HTTPConnection('127.0.0.1', self.port, timeout=TIMEOUT)
        try:
            if hosts is not None:
                connection.putrequest(method, path, skip_host=True)
                for value in hosts:
                    connection.putheader('Host', value)
                for name, value in (headers or {}).items():
                    connection.putheader(name, value)
                if body is not None:
                    connection.putheader('Content-Length', str(len(body)))
                connection.endheaders(message_body=body)
            else:
                sent = {} if host is None else {'Host': host}
                sent.update(headers or {})
                connection.request(method, path, body=body, headers=sent)
            response = connection.getresponse()
            return response.status, response.read()
        finally:
            connection.close()

    def inference_body(self):
        return json.dumps({'messages': []}).encode()


class HostAllowlistTests(SecurityBase):
    def test_foreign_or_malformed_host_is_refused_before_any_work(self):
        bad_hosts = [
            {'host': f'evil.example:{self.port}'},
            {'host': '127.0.0.1:1'},
            {'host': 'localhost'},
            {'hosts': [f'127.0.0.1:{self.port}', f'127.0.0.1:{self.port}']},
            {'hosts': []},                                   # no Host at all (HTTP/1.0 style)
            {'host': f'localhost.:{self.port}'},              # trailing dot
            {'host': f'evil.localhost:{self.port}'},
        ]
        requests = [('GET', '/_local/status', None), ('GET', '/v1/models', None),
                    ('POST', CHAT, b'{"messages":[]}')]
        for method, path, body in requests:
            for case in bad_hosts:
                with self.subTest(method=method, path=path, case=case):
                    with patch.object(controller.Handler, 'api_settings') as settings:
                        status, data = self.send(method, path, body=body, **case)
                    self.assertEqual(status, 403)
                    self.assertIn(b'Host not allowed', data)
                    settings.assert_not_called()
        self.assertEqual(self.manager.begin_queued.call_count, 0)
        self.assertEqual(self.manager.begin.call_count, 0)
        self.assertEqual(self.manager.end.call_count, 0)
        self.assertEqual(self.seen, [])

    def test_loopback_names_with_the_bound_port_pass_the_host_check(self):
        port = self.port
        for host in (f'127.0.0.1:{port}', f'localhost:{port}', f'[::1]:{port}',
                     f'LOCALHOST:{port}'):
            with self.subTest(host=host):
                status, data = self.send('GET', '/v1/models', host=host)
                self.assertNotIn(b'Host not allowed', data)
                self.assertEqual(status, 503)  # the fake manager's model is not validated
        self.assertEqual(self.manager.begin_queued.call_count, 0)


class OriginTests(SecurityBase):
    def test_ipv6_loopback_origin_with_same_origin_fetch_is_not_refused(self):
        status, data = self.send('POST', CHAT, body=self.inference_body(),
                                 headers={'Origin': f'http://[::1]:{self.port}',
                                          'Sec-Fetch-Site': 'same-origin'})
        self.assertNotIn(b'Cross-origin', data)
        self.assertEqual(status, 200)
        self.assertEqual(self.manager.begin_queued.call_count, 1)

    def test_foreign_origin_is_refused_as_cross_origin(self):
        status, data = self.send('POST', CHAT, body=self.inference_body(),
                                 headers={'Origin': 'http://evil.example'})
        self.assertEqual(status, 403)
        self.assertIn(b'Cross-origin', data)
        self.assertEqual(self.manager.begin_queued.call_count, 0)


class QueueReleaseTests(SecurityBase):
    def assert_controller_error(self, status, data):
        self.assertEqual(status, 500)
        self.assertEqual(json.loads(data)['error']['type'], 'local_controller_error')

    def test_prepare_prefix_exception_releases_the_slot_once(self):
        self.manager.prepare_prefix = Mock(side_effect=RuntimeError('synthetic prefix failure'))
        status, data = self.send('POST', CHAT, body=self.inference_body())
        self.assert_controller_error(status, data)
        self.manager.prepare_prefix.assert_called_once()
        self.assertEqual(self.manager.end.call_count, 1)
        self.assertIs(self.manager.end.call_args.kwargs['inference'], True)

    def test_runtime_headers_exception_releases_the_slot_once(self):
        def broken_headers():
            raise RuntimeError('synthetic runtime failure')
        self.manager.backend_runtime = SimpleNamespace(
            spec=SimpleNamespace(port=self.backend.server_port, backend_model_id='x'),
            chat_body=lambda payload: payload,
            headers=broken_headers)
        status, data = self.send('POST', CHAT, body=self.inference_body())
        self.assert_controller_error(status, data)
        self.assertEqual(self.manager.end.call_count, 1)
        self.assertIs(self.manager.end.call_args.kwargs['inference'], True)

    def test_flash_bad_chat_body_releases_the_slot_once(self):
        # the runtime cannot convert the body: 400, and end() exactly once (it used to run twice
        # when the 400 reply itself failed, leaving inferences at -1)
        def bad_body(payload):
            raise ValueError('synthetic bad flash body')
        self.manager.backend_runtime = SimpleNamespace(
            spec=SimpleNamespace(port=self.backend.server_port, backend_model_id='x'),
            chat_body=bad_body, headers=lambda: {})
        status, data = self.send('POST', CHAT, body=json.dumps({'model': 'other', 'messages': []}).encode())
        self.assertEqual(status, 400)
        self.assertEqual(self.manager.end.call_count, 1)

    def test_normal_path_releases_the_slot_once(self):
        released = threading.Event()
        self.manager.end.side_effect = lambda **_: released.set()
        status, data = self.send('POST', CHAT, body=self.inference_body())
        self.assertEqual(status, 200)
        self.assertIn(b'READY', data)
        self.assertTrue(released.wait(timeout=2))
        self.assertEqual(self.manager.begin_queued.call_count, 1)
        self.assertEqual(self.manager.end.call_count, 1)
        self.assertIs(self.manager.end.call_args.kwargs['inference'], True)


if __name__ == '__main__':
    unittest.main()
