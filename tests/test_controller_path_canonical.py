"""S4 (v0.1.1): the controller routes and forwards one canonical path. A trailing slash on an
inference route must not skip admission, the ledger, defaults or the disconnect watcher (the
Rust engine trims it and runs the request anyway). Model-free: synthetic backend only."""
import json
import tempfile
import unittest
from unittest.mock import Mock, patch

from snapdragon_ai_lab.runtime import controller
from test_controller_security import SecurityBase


class CanonicalTarget(unittest.TestCase):
    def test_rules(self):
        c = controller.canonical_target
        self.assertEqual(c('/v1/chat/completions/'), '/v1/chat/completions')
        self.assertEqual(c('/v1/chat/completions///'), '/v1/chat/completions')
        self.assertEqual(c('/_local/ledger/?limit=5'), '/_local/ledger?limit=5')
        self.assertEqual(c('/'), '/'); self.assertEqual(c('/?x=1'), '/?x=1')
        self.assertEqual(c('/props#frag'), '/props')
        for bad in ('//v1/chat/completions', '///', 'http://127.0.0.1/v1/models', '*', '', None):
            self.assertIsNone(c(bad), bad)


class TrailingSlashIsInference(SecurityBase):
    def setUp(self):
        super().setUp()
        self.paths = []
        paths = self.paths
        handler = self.backend.RequestHandlerClass
        original = handler.do_POST

        def do_POST(inner):
            paths.append(inner.path)
            return original(inner)
        handler.do_POST = do_POST
        self.addCleanup(setattr, handler, 'do_POST', original)

    def post(self, path, body):
        return self.send('POST', path, host=f'127.0.0.1:{self.port}', body=body,
                         headers={'Content-Type': 'application/json'})

    def test_every_inference_route_with_trailing_slash_is_admitted_as_inference(self):
        entries = []
        real_entry = controller.request_ledger.Entry

        def entry(path, *args, **kwargs):
            entries.append(path)
            return real_entry(path, *args, **kwargs)
        with patch.object(controller.request_ledger, 'Entry', side_effect=entry), \
                patch.dict(controller.os.environ, {'LOCAL_AI_LEDGER_DIR': self.tmp_dir()}), \
                patch.object(controller, 'ClientDisconnect') as watcher:
            watcher.return_value.start.return_value = watcher.return_value
            watcher.return_value.disconnected.is_set.return_value = False
            for route in sorted(controller.INFERENCE_PATHS):
                with self.subTest(route=route):
                    self.manager.begin_queued.reset_mock()
                    status, _ = self.post(route + '/', json.dumps({'messages': [], 'prompt': 'x'}).encode())
                    self.assertEqual(status, 200)
                    self.assertTrue(self.manager.begin_queued.call_args.kwargs['inference'], route)
                    self.assertEqual(self.paths[-1], route, 'forwarded path must be the canonical one')
                    self.assertEqual(entries[-1], route)
            self.assertEqual(watcher.call_count, len(controller.INFERENCE_PATHS))

    def tmp_dir(self):
        # the handler thread may still append its ledger row after the client has its reply
        d = tempfile.TemporaryDirectory(ignore_cleanup_errors=True); self.addCleanup(d.cleanup)
        return d.name

    def test_request_defaults_apply_with_trailing_slash(self):
        seen = []
        with patch.object(controller, 'request_defaults', side_effect=lambda t, p, path: seen.append(path) or p):
            status, _ = self.post('/v1/chat/completions/', json.dumps({'messages': []}).encode())
        self.assertEqual(status, 200)
        self.assertEqual(seen, ['/v1/chat/completions'])

    def test_double_leading_slash_is_never_forwarded_raw(self):
        # Python's http.server already folds a leading '//' to '/' (gh-87389); canonical_target
        # refuses it in case a server class does not. Either way the engine never sees '//'.
        status, _ = self.post('//v1/chat/completions/', b'{"messages": []}')
        if status == 404:
            self.manager.begin_queued.assert_not_called()
            self.assertEqual(self.paths, [])
        else:
            self.assertEqual(status, 200)
            self.assertTrue(self.manager.begin_queued.call_args.kwargs['inference'])
            self.assertEqual(self.paths, ['/v1/chat/completions'])

if __name__ == '__main__':
    unittest.main()
