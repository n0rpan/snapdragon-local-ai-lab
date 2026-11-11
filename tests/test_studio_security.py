"""F80 for the Image Studio server: Host allowlist, no cross-site POSTs, bounded bodies. No model, owned server."""
import http.client
import importlib.machinery
import importlib.util
from http.server import ThreadingHTTPServer
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / 'src/snapdragon_ai_lab/runtime/studio'


class StudioSecurity(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        env = {'IMG_OUTDIR': str(root / 'output'), 'LOCAL_IMAGE_CONFIG_DIR': str(root / 'config'),
               'LOCAL_AI_LEASE': str(root / 'inference.lock'), 'SD_HOME': str(root / 'stack')}
        with patch.dict(os.environ, env):
            loader = importlib.machinery.SourceFileLoader('security_studio', str(SOURCE))
            spec = importlib.util.spec_from_loader(loader.name, loader)
            self.studio = importlib.util.module_from_spec(spec)
            loader.exec_module(self.studio)
        self.srv = ThreadingHTTPServer(('127.0.0.1', 0), self.studio.H)
        self.port = self.srv.server_address[1]
        self.thread = threading.Thread(target=self.srv.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.srv.shutdown(); self.srv.server_close(); self.thread.join(5); self.tmp.cleanup()

    def send(self, method, path, headers, body=None):
        c = http.client.HTTPConnection('127.0.0.1', self.port, timeout=5)
        try:
            c.putrequest(method, path, skip_host=True, skip_accept_encoding=True)
            for k, v in headers:
                c.putheader(k, v)
            if body is not None and not any(k == 'Content-Length' for k, _ in headers):
                c.putheader('Content-Length', str(len(body)))
            c.endheaders(message_body=body)
            r = c.getresponse()
            return r.status, r.read()
        finally:
            c.close()

    def test_foreign_host_is_refused(self):
        for host in (f'evil.example:{self.port}', 'localhost', f'localhost.:{self.port}'):
            with self.subTest(host=host):
                status, data = self.send('GET', '/activity', [('Host', host)])
                self.assertEqual(status, 403)
                self.assertIn(b'Host not allowed', data)
        status, _ = self.send('GET', '/activity', [])
        self.assertEqual(status, 403)

    def test_loopback_host_passes(self):
        status, data = self.send('GET', '/activity', [('Host', f'127.0.0.1:{self.port}')])
        self.assertNotEqual(status, 403, data)

    def test_cross_site_post_is_refused(self):
        h = [('Host', f'127.0.0.1:{self.port}')]
        for extra in ([('Origin', 'http://evil.example')], [('Sec-Fetch-Site', 'cross-site')]):
            with self.subTest(extra=extra):
                status, data = self.send('POST', '/generate', h + extra, b'{}')
                self.assertEqual(status, 403)
                self.assertIn(b'Cross-origin', data)

    def test_oversized_bodies_are_refused_before_reading(self):
        h = [('Host', f'127.0.0.1:{self.port}')]
        status, _ = self.send('POST', '/generate', h + [('Content-Length', str(2 * 1024 * 1024))], b'{}')
        self.assertEqual(status, 413)
        status, _ = self.send('POST', '/upload', h + [('Content-Length', str(26 * 1024 * 1024))], b'x')
        self.assertEqual(status, 413)

    def test_upload_extension_allowlist(self):
        h = [('Host', f'127.0.0.1:{self.port}'), ('Content-Type', 'multipart/form-data; boundary=XB')]
        body = b'--XB\r\nContent-Disposition: form-data; name="f"; filename="a.sh"\r\n\r\necho hi\r\n--XB--\r\n'
        status, data = self.send('POST', '/upload', h, body)
        self.assertEqual(status, 400)
        self.assertIn(b'unsupported image type', data)


if __name__ == '__main__':
    unittest.main()
