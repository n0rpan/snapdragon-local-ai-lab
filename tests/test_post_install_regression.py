import importlib.util
import io
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

spec = importlib.util.spec_from_file_location('regression', Path(__file__).resolve().parents[1] / 'tools/post_install_regression.py')
r = importlib.util.module_from_spec(spec); spec.loader.exec_module(r)


class RegressionTests(unittest.TestCase):
    def test_live_suite_marks_lab_traffic(self):
        client = Mock()
        client.getresponse.return_value.status = 200
        client.getresponse.return_value.read.return_value = b'{}'
        with patch.object(r.http.client, 'HTTPConnection', return_value=client):
            suite = r.Suite('http://127.0.0.1:8092', 'fixture', 'a' * 64, time.monotonic() + 10)
            suite.api('/v1/chat/completions', {'messages': []})
        self.assertEqual(client.request.call_args.args[3]['X-Local-AI-Origin'], 'lab')

    def test_generated_ids_and_assembled_calls_have_distinct_honest_hashes(self):
        generated = r.identity({'tokens': [1, 2], 'content': 'ok'})
        self.assertIn('generated_ids_sha256', generated)
        def tool(call_id, args):
            return {'choices': [{'message': {'content': None, 'tool_calls': [
                {'id': call_id, 'function': {'name': 'report_result', 'arguments': args}}]}}],
                'usage': {'completion_tokens': 12}}
        one = r.identity(tool('random-one', '{"a":1,"b":2}'))
        two = r.identity(tool('random-two', '{"b":2,"a":1}'))
        self.assertEqual(one, two)
        self.assertIn('assembled_message_sha256', one)
        self.assertNotIn('generated_ids_sha256', one)

    def test_changed_output_never_auto_blessed_and_timings_require_matched_prompt_work(self):
        def row(hash='same', prompt=100, cached=0, ms=100):
            return {'identity': {'generated_ids_sha256': hash},
                    'native_timings': {'prompt_n': prompt, 'cache_n': cached, 'prompt_ms': ms, 'predicted_ms': 100}}
        self.assertTrue(r.compare({'chat': row()}, {'chat': row(hash='changed', ms=1)}))
        self.assertTrue(r.compare({'chat': row()}, {'chat': row(ms=300)}))
        self.assertFalse(r.compare({'chat': row()}, {'chat': row(prompt=1000, ms=300)}))
        self.assertEqual(r.compare({'chat': row()}, {}), ['case_set_changed'])

    def test_reference_write_is_private_and_cannot_replace_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'reference.json'
            r.save_new(path, {'passed': True})
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            with self.assertRaises(FileExistsError):
                r.save_new(path, {'passed': False})
            self.assertEqual(json.loads(path.read_text()), {'passed': True})

    def test_only_literal_loopback_and_bounded_budget(self):
        for url in ('https://127.0.0.1:8092', 'http://example.com', 'http://127.0.0.1:8092/private', 'http://user:secret@127.0.0.1:8092'):
            with self.assertRaisesRegex(ValueError, 'loopback'):
                r.Suite(url, 'fixture', 'a' * 64, time.monotonic() + 1)
        suite = r.Suite('http://127.0.0.1:8092', 'fixture', 'a' * 64, time.monotonic() - 1)
        with self.assertRaises(TimeoutError):
            suite.remaining()

    def test_actual_guard_and_pinned_engine_are_required(self):
        suite = r.Suite('http://127.0.0.1:8092', 'fixture', 'a' * 64, time.monotonic() + 5)
        status = {'state': 'ready', 'model': 'fixture', 'validation': {'server_sha256': 'a' * 64},
                  'memory_policy': {'reason': None, 'swap_current_bytes': 0, 'memory_events': {'oom': 0, 'oom_kill': 0}}}
        suite.api = Mock(return_value=status)
        suite.guard(); self.assertEqual(len(suite.guards), 1)
        status['memory_policy']['swap_current_bytes'] = 4096
        with self.assertRaisesRegex(RuntimeError, 'memory_guard'):
            suite.guard()
        status['validation']['server_sha256'] = 'b' * 64
        with self.assertRaisesRegex(RuntimeError, 'wrong_engine'):
            suite.guard()

    def test_guard_waits_for_own_response_but_rejects_queued_work(self):
        suite = r.Suite('http://127.0.0.1:8092', 'fixture', 'a' * 64, time.monotonic() + 5)
        ready = {'state': 'ready', 'model': 'fixture', 'validation': {'server_sha256': 'a' * 64},
                 'memory_policy': {'reason': None, 'swap_current_bytes': 0, 'memory_events': {'oom': 0, 'oom_kill': 0}}}
        suite.api = Mock(side_effect=[dict(ready, active_requests=1), ready,
                                      dict(ready, active_requests=1, queued_requests=1)])
        suite.guard()
        self.assertEqual(suite.api.call_count, 2)
        with self.assertRaisesRegex(RuntimeError, 'not_exclusively_idle'):
            suite.guard()

    def test_capture_requires_independent_proof_and_verify_rejects_wrong_engine_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); ref = root / 'ref.json'
            common = ['--url', 'http://127.0.0.1:8092', '--alias', 'fixture', '--engine-sha256', 'a' * 64,
                      '--reference', str(ref), '--out', str(root / 'result.json')]
            with patch('sys.argv', ['regression', 'capture', *common]), patch('sys.stderr', io.StringIO()), patch.object(r, 'Suite') as suite:
                with self.assertRaises(SystemExit):
                    r.main()
                suite.assert_not_called()
            ref.write_text(json.dumps({'passed': True, 'engine_sha256': 'b' * 64, 'alias': 'fixture', 'session_resume': False}))
            with patch('sys.argv', ['regression', 'verify', *common]), patch('sys.stderr', io.StringIO()), patch.object(r, 'Suite') as suite:
                with self.assertRaises(SystemExit):
                    r.main()
                suite.assert_not_called()

    def test_full_synthetic_sequence_and_failed_disk_reuse_are_distinguished(self):
        class FakeSuite(r.Suite):
            def __init__(self, fail_resume=False):
                super().__init__('http://127.0.0.1:8092', 'fixture', 'a' * 64,
                                 time.monotonic() + 10, session_resume=True)
                self.unloaded = False
                self.fail_resume = fail_resume

            def api(self, path, body=None, timeout=90):
                timings = {'cache_n': 0 if self.unloaded and self.fail_resume else 2304,
                           'prompt_n': 1, 'prompt_ms': 1., 'predicted_ms': 1.}
                if path == '/_local/status':
                    return {'state': 'ready', 'model': self.alias, 'validation': {'server_sha256': self.engine_sha},
                            'memory_policy': {'reason': None, 'swap_current_bytes': 0,
                                              'memory_events': {'oom': 0, 'oom_kill': 0}}}
                if path == '/_local/unload':
                    self.unloaded = True
                    return {'state': 'unloaded'}
                if path == '/completion':
                    return {'tokens': [42, 43], 'content': 'synthetic', 'timings': timings}
                if 'tools' in body and len(body['messages']) == 1:
                    message = {'content': None, 'tool_calls': [{'id': 'transport-id', 'function': {
                        'name': 'report_result', 'arguments': '{"value":"ready"}'}}]}
                else:
                    message = {'content': 'OK'}
                return {'choices': [{'message': message}], 'usage': {'completion_tokens': 2}, 'timings': timings}

        with patch.object(r.socket, 'create_connection', return_value=Mock()), patch.object(r.time, 'sleep'):
            suite = FakeSuite()
            rows = suite.run()
            self.assertEqual(set(rows), {'chat', 'tool_call', 'tool_followup', 'context_4k', 'cancel', 'session_resume'})
            self.assertEqual(rows['session_resume']['cached_tokens'], 2304)
            self.assertEqual(len(suite.guards), 7)
            with self.assertRaisesRegex(RuntimeError, 'did_not_reuse'):
                FakeSuite(fail_resume=True).run()

    def test_chat_resume_requires_exact_prior_C3_and_identity(self):
        class ChatSuite(r.Suite):
            def __init__(self, cached=1401, drift=False):
                super().__init__('http://127.0.0.1:8092', 'fixture', 'a' * 64,
                                 time.monotonic() + 5, session_resume=True, session_api='chat')
                self.calls = 0
                self.cached, self.drift = cached, drift
                self.guard = Mock()
                self.api = Mock(return_value={'state': 'unloaded'})
            def chat(self, messages, **kw):
                self.calls += 1
                return {'choices': [{'message': {'role': 'assistant', 'content': 'changed' if self.drift and self.calls == 3 else 'OK'}}],
                        'usage': {'prompt_tokens': 1400, 'completion_tokens': 2},
                        'timings': {'cache_n': self.cached, 'prompt_n': 1, 'prompt_ms': 1., 'predicted_ms': 1., 'resume_src': 'disk'}}
        suite = ChatSuite(); suite.chat_session_resume()
        self.assertEqual(suite.rows['session_resume']['C3_required_tokens'], 1401)
        self.assertEqual(suite.rows['session_resume']['session_api'], 'chat')
        suite.api.assert_called_once_with('/_local/unload', {}, timeout=30)
        with self.assertRaisesRegex(RuntimeError, 'reuse_C3'):
            ChatSuite(cached=1400).chat_session_resume()
        with self.assertRaisesRegex(RuntimeError, 'output_drift'):
            ChatSuite(drift=True).chat_session_resume()

    def test_timeout_is_blocked_and_cannot_create_a_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); proof = root / 'proof.json'; reference = root / 'reference.json'; out = root / 'out.json'
            proof.write_text(json.dumps({'passed': True, 'engine_sha256': 'a' * 64}))
            args = ['regression', 'capture', '--url', 'http://127.0.0.1:8092', '--alias', 'fixture',
                    '--engine-sha256', 'a' * 64, '--reference', str(reference), '--qualified-receipt', str(proof), '--out', str(out)]
            fake = Mock(rows={}, guards=[])
            fake.run.side_effect = TimeoutError('synthetic timeout')
            with patch('sys.argv', args), patch.object(r, 'Suite', return_value=fake), patch('sys.stdout', io.StringIO()):
                self.assertEqual(r.main(), 2)
            self.assertFalse(reference.exists())
            report = json.loads(out.read_text())
            self.assertFalse(report['passed'])
            self.assertEqual(report['verdict'], 'BLOCKED')


if __name__ == '__main__':
    unittest.main()
