"""Model-free tests for tools/long_session.py. Only local fake servers on 127.0.0.1 ephemeral ports."""
import contextlib
import http.client
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import random
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location('long_session', ROOT / 'tools/long_session.py')
ls = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ls)


class FakeServer(ThreadingHTTPServer):
    """OpenAI-shaped streaming server with fake timings. Records every request body."""
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, fail_at=(), json_mode=False):
        super().__init__(('127.0.0.1', 0), _Handler)
        self.fail_at = set(fail_at)
        self.json_mode = json_mode
        self.bodies = []
        self.paths = []
        self.lock = threading.Lock()
        self.url = 'http://127.0.0.1:{}/v1'.format(self.server_address[1])
        self._thread = threading.Thread(target=self.serve_forever, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self.shutdown()
        self.server_close()


class _Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.0'

    def log_message(self, *args):
        pass

    def do_POST(self):
        length = int(self.headers.get('Content-Length', '0'))
        body = json.loads(self.rfile.read(length))
        srv = self.server
        with srv.lock:
            idx = len(srv.bodies)
            srv.bodies.append(body)
            srv.paths.append(self.path)
        if idx in srv.fail_at:
            self.send_response(500)
            self.send_header('Content-Length', '2')
            self.end_headers()
            self.wfile.write(b'{}')
            return
        prompt_chars = len(json.dumps(body['messages']))
        prompt = prompt_chars // 4
        prev = srv.bodies[idx - 1] if idx > 0 else None
        cached = (len(json.dumps(prev['messages'])) // 4) if prev else 0
        usage = {'prompt_tokens': prompt, 'completion_tokens': 3,
                 'prompt_tokens_details': {'cached_tokens': cached}}
        timings = {'prompt_ms': 12.5, 'predicted_per_second': 40.0,
                   'prompt_n': prompt - cached, 'cache_n': cached, 'predicted_n': 3}
        if srv.json_mode:
            payload = json.dumps({'choices': [{'message': {'content': 'w w w'},
                                               'finish_reason': 'length'}],
                                  'usage': usage, 'timings': timings}).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        chunks = [
            {'choices': [{'delta': {'role': 'assistant', 'content': 'w '}, 'index': 0}]},
            {'choices': [{'delta': {'content': 'w w'}, 'index': 0}]},
            {'choices': [{'delta': {}, 'index': 0, 'finish_reason': 'length'}]},
            {'choices': [], 'usage': usage, 'timings': timings},
        ]
        lines = [b'data: ' + json.dumps(c).encode() + b'\n\n' for c in chunks]
        lines.append(b'data: [DONE]\n\n')
        payload = b''.join(lines)
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def _run(argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = ls.main(argv)
    return code, out.getvalue(), err.getvalue()


def _gen(tmp, name, seed=7, turns=20, target='16K', preserve=True):
    path = os.path.join(tmp, name)
    code, _, _ = _run(['generate', '--out', path, '--seed', str(seed), '--turns', str(turns),
                       '--target-context', target,
                       *(['--preserve-thinking'] if preserve else ['--no-preserve-thinking'])])
    assert code == 0
    return path


class TestSampler(unittest.TestCase):
    def test_lognormal_hits_p50_and_p90(self):
        rng = random.Random(3)
        n = 40000
        users = sorted(ls.lognormal(rng, 600.0, 5000.0, 16, 10 ** 9) for _ in range(n))
        self.assertAlmostEqual(users[n // 2] / 600.0, 1.0, delta=0.06)
        self.assertAlmostEqual(users[int(n * 0.9)] / 5000.0, 1.0, delta=0.08)

    def test_tool_output_shape_targets(self):
        rng = random.Random(11)
        n = 40000
        sizes = [ls.lognormal(rng, ls.TOOL_BYTES_P50, ls.TOOL_BYTES_P90,
                              ls.TOOL_BYTES_MIN, ls.TOOL_BYTES_MAX) for _ in range(n)]
        p50 = ls.percentile(sizes, 50)
        p90 = ls.percentile(sizes, 90)
        over = sum(1 for s in sizes if s > 16384) / n
        self.assertTrue(1024 <= p50 <= 2048, p50)
        self.assertTrue(10240 <= p90 <= 15360, p90)
        self.assertTrue(0.05 <= over <= 0.09, over)

    def test_percentile_interpolates(self):
        self.assertEqual(ls.percentile([1, 2, 3, 4], 50), 2.5)
        self.assertIsNone(ls.percentile([], 50))


class TestGenerator(unittest.TestCase):
    def test_same_seed_gives_identical_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = _gen(tmp, 'a.json', seed=42, turns=30, target='32K')
            b = _gen(tmp, 'b.json', seed=42, turns=30, target='32K')
            c = _gen(tmp, 'c.json', seed=43, turns=30, target='32K')
            self.assertEqual(Path(a).read_bytes(), Path(b).read_bytes())
            self.assertNotEqual(Path(a).read_bytes(), Path(c).read_bytes())

    def test_message_structure(self):
        session = ls.generate_session(5, 12, 16384, True)
        for turn in session['turns']:
            u, a, t = turn['user'], turn['assistant'], turn['tool']
            self.assertEqual(u['role'], 'user')
            self.assertEqual(a['role'], 'assistant')
            self.assertTrue(a['content'].startswith(ls.THINK_BLOCK))
            call = a['tool_calls'][0]
            self.assertEqual(call['type'], 'function')
            self.assertIn(call['function']['name'], {'read', 'grep', 'glob', 'bash'})
            json.loads(call['function']['arguments'])
            self.assertEqual(t['role'], 'tool')
            self.assertEqual(t['tool_call_id'], call['id'])
            self.assertTrue(t['content'].isascii())
            self.assertLessEqual(len(t['content']), ls.TOOL_BYTES_MAX)

    def test_no_think_block_when_preserve_off(self):
        session = ls.generate_session(5, 3, 16384, False)
        for turn in session['turns']:
            self.assertEqual(turn['assistant']['content'], '')

    def test_tools_are_openai_shaped(self):
        for tool in ls.TOOLS:
            self.assertEqual(tool['type'], 'function')
            self.assertIn('parameters', tool['function'])

    def test_stops_at_target_context(self):
        session = ls.generate_session(9, 200, 16384, True)
        summary = session['summary']
        self.assertEqual(summary['stopped_by'], 'target')
        self.assertEqual(summary['final_estimated_context_tokens'], 16384)
        total = session['prefix']['estimated_tokens'] + sum(
            t['estimated_tokens']['turn'] for t in session['turns'])
        self.assertEqual(total, 16384)

    def test_stops_at_turn_cap_when_target_is_far(self):
        session = ls.generate_session(9, 5, 10 ** 7, True)
        self.assertEqual(session['summary']['stopped_by'], 'turns')
        self.assertEqual(len(session['turns']), 5)

    def test_file_shape_matches_targets(self):
        # 200 turns at a 1M-token target: the turn cap binds, so the sample is the full 200 turns.
        session = ls.generate_session(2026, 200, 1024 * 1024, True)
        s = session['summary']
        self.assertEqual(s['turns'], 200)
        self.assertTrue(450 <= s['user_tokens_p50'] <= 800, s['user_tokens_p50'])
        self.assertTrue(3000 <= s['user_tokens_p90'] <= 8000, s['user_tokens_p90'])
        self.assertTrue(1000 <= s['tool_bytes_p50'] <= 2200, s['tool_bytes_p50'])
        self.assertTrue(8000 <= s['tool_bytes_p90'] <= 18000, s['tool_bytes_p90'])
        self.assertTrue(0.03 <= s['tool_bytes_over_16k_fraction'] <= 0.12,
                        s['tool_bytes_over_16k_fraction'])

    def test_text_contains_no_obvious_private_markers(self):
        session = ls.generate_session(1, 10, 16384, True)
        blob = json.dumps(session).lower()
        for marker in ('@', 'http://', 'password', 'api_key', '/home/'):
            self.assertNotIn(marker, blob)


class TestEndpoint(unittest.TestCase):
    def test_accepts_loopback_only(self):
        self.assertEqual(ls.parse_endpoint('http://127.0.0.1:8092/v1'),
                         ('127.0.0.1', 8092, '/v1/chat/completions'))
        self.assertEqual(ls.parse_endpoint('http://127.0.0.1:9000')[2], '/v1/chat/completions')
        for bad in ('http://example.com:80/v1', 'https://127.0.0.1:8092/v1',
                    'http://127.0.0.1/v1', 'http://127.0.0.1:8092/v1?x=1'):
            with self.assertRaises(ls.UsageError):
                ls.parse_endpoint(bad)


class TestReplay(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.session = _gen(self.tmp.name, 's.json', seed=3, turns=6, target='16K')

    def tearDown(self):
        self.tmp.cleanup()

    def test_receipt_schema_and_request_shape(self):
        receipt_path = os.path.join(self.tmp.name, 'r.json')
        with FakeServer() as srv:
            code, out, _ = _run(['replay', self.session, '--base-url', srv.url, '--model', 'fake',
                                 '--receipt', receipt_path, '--curve-bin', '2048'])
        self.assertEqual(code, 0)
        receipt = json.loads(Path(receipt_path).read_text())
        for key in ('schema', 'tool', 'status', 'error', 'started_utc', 'finished_utc', 'endpoint',
                    'model', 'session', 'request', 'turns_planned', 'turns_run', 'summary',
                    'curve', 'turns'):
            self.assertIn(key, receipt)
        self.assertEqual(receipt['schema'], ls.RECEIPT_SCHEMA)
        self.assertEqual(receipt['status'], 'complete')
        self.assertEqual(receipt['turns_run'], receipt['turns_planned'])
        self.assertEqual(len(receipt['turns']), receipt['turns_run'])
        self.assertEqual(receipt['request']['max_tokens'], 64)
        self.assertEqual(receipt['request']['temperature'], 0)
        for turn in receipt['turns']:
            for key in ('turn', 'estimated_prompt_tokens', 'prompt_tokens', 'cached_tokens',
                        'ttft_ms', 'prefill_ms', 'decode_tok_s', 'wall_ms'):
                self.assertIn(key, turn)
            self.assertEqual(turn['decode_source'], 'timings')
        self.assertIsNotNone(receipt['summary']['ttft_ms']['p50'])
        self.assertTrue(receipt['curve'])
        self.assertIsNotNone(receipt['summary']['cache_hit_ratio'])
        self.assertIsNotNone(receipt['summary']['prompt_tokens_total'])
        # The receipt must not carry message text.
        text = Path(receipt_path).read_text()
        first_user = json.loads(Path(self.session).read_text())['turns'][0]['user']['content']
        self.assertNotIn(first_user[:80], text)

    def test_request_bodies(self):
        receipt_path = os.path.join(self.tmp.name, 'r2.json')
        with FakeServer() as srv:
            _run(['replay', self.session, '--base-url', srv.url, '--model', 'fake',
                  '--receipt', receipt_path])
            bodies, paths = srv.bodies, srv.paths
        self.assertEqual(set(paths), {'/v1/chat/completions'})
        self.assertEqual(len(bodies), 6)
        for body in bodies:
            self.assertEqual(body['temperature'], 0)
            self.assertEqual(body['max_tokens'], 64)
            self.assertTrue(body['stream'])
            self.assertEqual(body['chat_template_kwargs'],
                             {'enable_thinking': False, 'preserve_thinking': True})
            self.assertEqual(body['tools'], ls.TOOLS)
        # Each request is the previous one plus that turn's history and a new user tail.
        self.assertEqual(len(bodies[0]['messages']), 2)
        self.assertEqual(len(bodies[1]['messages']), 5)
        self.assertEqual(bodies[1]['messages'][-1]['role'], 'user')
        self.assertTrue(bodies[1]['messages'][2]['content'].startswith(ls.THINK_BLOCK))

    def test_json_response_mode(self):
        receipt_path = os.path.join(self.tmp.name, 'r3.json')
        with FakeServer(json_mode=True) as srv:
            code, _, _ = _run(['replay', self.session, '--base-url', srv.url, '--model', 'fake',
                               '--receipt', receipt_path])
        self.assertEqual(code, 0)
        receipt = json.loads(Path(receipt_path).read_text())
        self.assertEqual(receipt['status'], 'complete')
        self.assertEqual(receipt['turns'][0]['decode_source'], 'timings')

    def test_aborts_on_http_error_and_writes_partial_receipt(self):
        receipt_path = os.path.join(self.tmp.name, 'r4.json')
        with FakeServer(fail_at={2}) as srv:
            code, _, _ = _run(['replay', self.session, '--base-url', srv.url, '--model', 'fake',
                               '--receipt', receipt_path])
            self.assertEqual(len(srv.bodies), 3)
        self.assertEqual(code, 3)
        receipt = json.loads(Path(receipt_path).read_text())
        self.assertEqual(receipt['status'], 'aborted_http_error')
        self.assertEqual(receipt['turns_run'], 2)
        self.assertIn('500', receipt['error'])

    def test_max_context_stops_before_sending(self):
        big = _gen(self.tmp.name, 'big.json', seed=4, turns=40, target='32K')
        receipt_path = os.path.join(self.tmp.name, 'r5.json')
        with FakeServer() as srv:
            code, _, _ = _run(['replay', big, '--base-url', srv.url, '--model', 'fake',
                               '--receipt', receipt_path, '--max-context', '8000'])
            sent = len(srv.bodies)
        self.assertEqual(code, 0)
        receipt = json.loads(Path(receipt_path).read_text())
        self.assertEqual(receipt['status'], 'stopped_max_context')
        self.assertEqual(sent, receipt['turns_run'])
        self.assertLess(receipt['turns_run'], receipt['turns_planned'])
        self.assertLessEqual(receipt['turns'][-1]['estimated_prompt_tokens'], 8000)

    def test_dry_run_opens_no_connection(self):
        receipt_path = os.path.join(self.tmp.name, 'dry.json')
        real = http.client.HTTPConnection

        def forbidden(*a, **k):
            raise AssertionError('dry-run must not open a connection')

        http.client.HTTPConnection = forbidden
        try:
            code, out, _ = _run(['replay', self.session, '--base-url', 'http://127.0.0.1:1/v1',
                                 '--model', 'fake', '--receipt', receipt_path, '--dry-run'])
        finally:
            http.client.HTTPConnection = real
        self.assertEqual(code, 0)
        plan = json.loads(out)
        self.assertEqual(plan['status'], 'dry_run')
        self.assertEqual(plan['turns_planned'], 6)
        self.assertTrue(plan['endpoint_not_contacted'])
        receipt = json.loads(Path(receipt_path).read_text())
        self.assertEqual(receipt['status'], 'dry_run')
        self.assertEqual(receipt['schema'], ls.RECEIPT_SCHEMA)

    def test_live_replay_requires_receipt(self):
        code, _, err = _run(['replay', self.session, '--base-url', 'http://127.0.0.1:1/v1',
                             '--model', 'fake'])
        self.assertEqual(code, 2)
        self.assertIn('--receipt', err)

    def test_key_env_is_read_from_environment_only(self):
        receipt_path = os.path.join(self.tmp.name, 'key.json')
        os.environ.pop('LONG_SESSION_TEST_KEY', None)
        code, _, err = _run(['replay', self.session, '--base-url', 'http://127.0.0.1:1/v1',
                             '--model', 'fake', '--receipt', receipt_path,
                             '--key-env', 'LONG_SESSION_TEST_KEY'])
        self.assertEqual(code, 2)
        self.assertIn('LONG_SESSION_TEST_KEY', err)


if __name__ == '__main__':
    unittest.main()
