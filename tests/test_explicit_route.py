"""Explicit routing stays opt-in and uses normal controller lifecycle gates."""
import os as _os, tempfile as _tempfile
_os.environ.setdefault('LOCAL_AI_LEDGER_DIR', _tempfile.mkdtemp(prefix='local-ai-test-ledger-'))  # never the real request ledger
import json
import http.client
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

from snapdragon_ai_lab.runtime import controller, engine_ledger, explicit_route as route


class ExplicitRouteTests(unittest.TestCase):
    def test_default_is_off_and_never_examines_prompt_text(self):
        request = {'model': 'qwen-local', 'messages': [{'content': 'plan a huge architecture'}]}
        self.assertIs(route.rewrite(request), request)
        with self.assertRaisesRegex(ValueError, 'disabled'):
            route.rewrite({'model': 'local-auto', 'local_route': 'think-hard'})

    def test_manual_or_panel_model_always_wins_and_extension_is_consumed(self):
        for alias in ('qwen-local', 'qwen3.6-35b-a3b-local', 'qwen3.8-flash-next-local'):
            request = {'model': alias, 'local_route': 'code', 'messages': []}
            for enabled in (False, True):
                self.assertEqual(route.rewrite(request, enabled=enabled), {'model': alias, 'messages': []})
            self.assertIn('local_route', request)
        self.assertEqual(route.rewrite({'local_route': 'think-hard'}), {})

    def test_local_auto_requires_a_concrete_explicit_choice(self):
        for choice in (None, True, [], {}, 'auto', 'guess'):
            with self.assertRaisesRegex(ValueError, 'requires'):
                route.rewrite({'model': 'local-auto', 'local_route': choice}, enabled=True)
        for choice, alias in route.TARGETS.items():
            request = {'model': 'local-auto', 'local_route': choice, 'max_tokens': 99}
            self.assertEqual(route.rewrite(request, enabled=True), {'model': alias, 'max_tokens': 99})
            self.assertEqual(request['model'], 'local-auto')

    def test_model_listing_is_nonwaking_and_flagged(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / 'runtime.json'
            config.write_text(json.dumps({'model_alias': 'qwen-local', 'selected_model': route.TARGETS['code'],
                                          'profiles': {v: {} for v in route.TARGETS.values()}}))
            manager = controller.Manager(); manager.stop_event.set()
            try:
                with patch.object(controller, 'CONFIG', config), \
                        patch.object(manager, 'config', side_effect=lambda alias=None: {'model_alias': alias or route.TARGETS['code']}), \
                        patch.object(manager, 'eligible', return_value=True), \
                        patch.object(manager, 'begin', side_effect=AssertionError('poll woke model')):
                    with patch.dict(os.environ, {route.ENABLE_ENV: '0'}):
                        self.assertNotIn('local-auto', {x['id'] for x in manager.models()})
                    with patch.dict(os.environ, {route.ENABLE_ENV: '1'}):
                        self.assertIn('local-auto', {x['id'] for x in manager.models()})
                self.assertIsNone(manager.process)
            finally:
                manager.close()

    def test_switch_ledger_is_numeric_private_and_failure_tolerant(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'private/ledger.jsonl'
            self.assertTrue(engine_ledger.record_transition(path, None, route.TARGETS['code'], 45.5, success=True))
            self.assertTrue(engine_ledger.record_transition(path, route.TARGETS['code'], route.TARGETS['think-hard'], 123.0, success=False))
            rows = [json.loads(x) for x in path.read_text().splitlines()]
            self.assertEqual([x['event'] for x in rows], ['model_load', 'model_switch'])
            self.assertEqual(rows[1]['switch_ms'], 123.0)
            self.assertFalse(rows[1]['success'])
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            self.assertFalse(engine_ledger.record_transition(path, None, 'x', float('nan'), success=True))
            bad = Path(tmp) / 'symlink'; bad.symlink_to(path)
            self.assertFalse(engine_ledger.record_transition(bad, None, 'x', 1.0, success=True))
            self.assertEqual(len(path.read_text().splitlines()), 2)

    def test_http_route_stays_off_until_enabled_and_passes_concrete_alias_to_queue(self):
        manager = Mock()
        manager.config.side_effect = lambda alias=None: {'model_alias': alias, 'backend_kind': 'llama'}
        manager.begin_queued.side_effect = controller.Unavailable('synthetic stop before model load')
        server = controller.Server(('127.0.0.1', 0), controller.Handler)
        server.manager = manager
        thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
        def request(payload):
            client = http.client.HTTPConnection('127.0.0.1', server.server_port, timeout=2)
            try:
                client.request('POST', '/v1/chat/completions', json.dumps(payload), {'Content-Type': 'application/json'})
                response = client.getresponse(); response.read()
                return response.status
            finally:
                client.close()
        try:
            with patch.dict(os.environ, {route.ENABLE_ENV: '0'}):
                self.assertEqual(request({'model': 'local-auto', 'local_route': 'code', 'messages': []}), 400)
                manager.begin_queued.assert_not_called()
            with patch.dict(os.environ, {route.ENABLE_ENV: '1'}):
                for choice, alias in route.TARGETS.items():
                    self.assertEqual(request({'model': 'local-auto', 'local_route': choice, 'messages': []}), 503)
                    self.assertEqual(manager.begin_queued.call_args.kwargs['model_alias'], alias)
            with patch.dict(os.environ, {route.ENABLE_ENV: '0'}):
                self.assertEqual(request({'model': route.TARGETS['code'], 'local_route': 'think-hard', 'messages': []}), 503)
                self.assertEqual(manager.begin_queued.call_args.kwargs['model_alias'], route.TARGETS['code'])
        finally:
            server.shutdown(); server.server_close(); thread.join(2)

    def test_switch_unloads_old_child_before_launch_and_records_failure_without_request_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); model = root / 'fixture.gguf'; model.write_bytes(b'synthetic')
            previous = {'model_alias': 'fixture-code', 'backend_kind': 'llama'}
            target = {'model_alias': 'fixture-hard', 'backend_kind': 'llama', 'ready': True,
                      'model_path': str(model), 'server_binary': '/synthetic-server', 'prefix_cache': {'enabled': False}}
            manager = controller.Manager(); manager.stop_event.set()
            manager.process = Mock(); manager.process.poll.return_value = None
            manager.loaded_config = previous
            calls = []
            original_cleanup = manager.cleanup
            def cleanup():
                calls.append('unload')
                manager.process = None
                original_cleanup()
            def launch(*args, **kwargs):
                calls.append('launch')
                self.assertIsNone(manager.process)
                raise RuntimeError('synthetic load failure')
            try:
                with patch.dict(os.environ, {route.ENABLE_ENV: '1'}), \
                        patch.object(manager, 'config', return_value=target), \
                        patch.object(manager, 'cleanup', side_effect=cleanup), \
                        patch.object(controller, 'LEASE', root / 'lease'), \
                        patch.object(controller, 'ENGINE_DIAGNOSTIC_LOG', root / 'private/stderr.log'), \
                        patch.object(controller, 'available_memory_bytes', return_value=None), \
                        patch.object(controller.memory_policy, 'MemoryGuard', return_value=Mock()), \
                        patch.object(controller.subprocess, 'Popen', side_effect=launch), \
                        patch.object(engine_ledger, 'record_transition', return_value=True) as record:
                    with self.assertRaisesRegex(RuntimeError, 'synthetic load failure'):
                        manager.begin(allow_start=True, model_alias='fixture-hard')
                    self.assertEqual(calls[:2], ['unload', 'launch'])
                    record.assert_called_once()
                    self.assertEqual(record.call_args.args[1:3], ('fixture-code', 'fixture-hard'))
                    self.assertFalse(record.call_args.kwargs['success'])
                self.assertIsNone(manager.process)
                self.assertIsNone(manager.lease)
            finally:
                manager.close()


if __name__ == '__main__':
    unittest.main()
