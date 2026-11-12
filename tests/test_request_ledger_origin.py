"""REAL-1: origin tag (organic vs lab), session hash and gap, token/timing names, tool-output
sizes, owner-only files, retention, and the ledger-summary script on a fixture. Numbers and
hashes only: no prompt or reply text is ever written."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from snapdragon_ai_lab.runtime import request_ledger as rl

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / 'tests' / 'fixtures' / 'ledger' / 'requests-20261112.jsonl'
MARKER = 'real1-secret-text-marker-x9'


class Headers(dict):
    """Case-insensitive .get like http.server's message headers."""
    def get(self, key, default=None):
        for k, v in self.items():
            if k.lower() == key.lower():
                return v
        return default


def payload(first='hello ' + MARKER, n_tool=0):
    msgs = [{'role': 'system', 'content': 'sys'}, {'role': 'user', 'content': first}]
    for i in range(n_tool):
        msgs.append({'role': 'assistant', 'content': ''})
        msgs.append({'role': 'tool', 'content': 'x' * (1000 * (i + 1)) + MARKER})
    return {'messages': msgs, 'stream': True}


class OriginTests(unittest.TestCase):
    def test_marker_detection(self):
        self.assertEqual(rl.origin_of('lab'), 'lab')
        self.assertEqual(rl.origin_of(' LAB '), 'lab')
        self.assertEqual(rl.origin_of('lab,organic'), 'organic')
        self.assertEqual(rl.origin_of('organic'), 'organic')
        self.assertEqual(rl.origin_of(''), 'organic')
        self.assertEqual(rl.origin_of(None), 'organic')

    def test_entry_default_is_organic_and_header_marks_lab(self):
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {'LOCAL_AI_LEDGER_DIR': d}):
            self.assertEqual(rl.Entry('/x', payload(), None, 'm', headers=Headers()).row['origin'], 'organic')
            self.assertEqual(rl.Entry('/x', payload(), None, 'm', headers=None).row['origin'], 'organic')
            lab = rl.Entry('/x', payload(), None, 'm', headers=Headers({'X-Local-AI-Origin': 'lab'}))
            self.assertEqual(lab.row['origin'], 'lab')

    def test_bad_header_value_type_never_fails_the_request(self):
        class Broken(dict):
            def get(self, *a, **k):
                raise RuntimeError('boom')
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {'LOCAL_AI_LEDGER_DIR': d}):
            row = rl.Entry('/x', payload(), None, 'm', headers=Broken()).finish(200)
        self.assertEqual(row['origin'], 'organic')
        self.assertIsNone(row['session'])


class SessionAndGapTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.env = patch.dict(os.environ, {'LOCAL_AI_LEDGER_DIR': self.tmp, 'LOCAL_AI_LEDGER': '1'})
        self.env.start()
        rl._salt[0] = None
        rl._last_end.clear()

    def tearDown(self):
        self.env.stop()

    def test_session_hash_is_keyed_stable_and_not_text(self):
        a = rl.session_key(Headers(), payload('first message ' + MARKER))
        b = rl.session_key(Headers(), payload('first message ' + MARKER, n_tool=3))
        c = rl.session_key(Headers(), payload('another opener'))
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertRegex(a, r'^[0-9a-f]{16}$')
        self.assertIsNone(rl.session_key(Headers(), {'messages': [{'role': 'system', 'content': 'only'}]}))
        h = rl.session_key(Headers({'X-Session-Id': 'abc'}), payload('zzz'))
        self.assertEqual(h, rl.session_key(Headers({'x-session-id': 'abc'}), payload('other')))

    def test_salt_is_owner_only_and_not_recomputed_per_row(self):
        rl.session_key(Headers(), payload())
        salt = Path(self.tmp) / '.salt'
        self.assertEqual(oct(salt.stat().st_mode & 0o777), '0o600')
        self.assertEqual(len(salt.read_bytes()), 32)
        before = salt.read_bytes()
        rl.session_key(Headers(), payload('again'))
        self.assertEqual(salt.read_bytes(), before)

    def test_gap_since_previous_request_in_same_session(self):
        first = rl.Entry('/x', payload(), None, 'm', headers=Headers())
        self.assertIsNone(first.row['gap_ms'])
        first.finish(200)
        time.sleep(0.05)
        second = rl.Entry('/x', payload(), None, 'm', headers=Headers())
        self.assertGreaterEqual(second.row['gap_ms'], 40.0)
        other = rl.Entry('/x', payload('different opener'), None, 'm', headers=Headers())
        self.assertIsNone(other.row['gap_ms'])

    def test_gap_table_is_bounded(self):
        with patch.object(rl, 'GAP_SESSIONS', 3):
            for i in range(6):
                rl.Entry('/x', payload(f'opener {i}'), None, 'm', headers=Headers()).finish(200)
        self.assertLessEqual(len(rl._last_end), 3)


class RowSchemaTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.env = patch.dict(os.environ, {'LOCAL_AI_LEDGER_DIR': str(self.tmp), 'LOCAL_AI_LEDGER': '1'})
        self.env.start()
        rl._salt[0] = None
        rl._last_end.clear()

    def tearDown(self):
        self.env.stop()

    def test_token_and_timing_names_come_from_backend_timings(self):
        t = {'cache_n': 5000, 'prompt_n': 312, 'prompt_ms': 1712.5, 'predicted_n': 64, 'predicted_ms': 1490.0}
        row = rl.derive_counts(dict(t))
        self.assertEqual((row['ctx_tokens'], row['cached_tokens'], row['new_tokens']), (5312, 5000, 312))
        self.assertEqual((row['output_tokens'], row['prefill_ms'], row['decode_ms']), (64, 1712.5, 1490.0))
        self.assertEqual(rl.derive_counts({'predicted_n': 3}), {'predicted_n': 3, 'output_tokens': 3})

    def test_tool_output_sizes_only(self):
        row = rl.Entry('/x', payload(n_tool=3), None, 'm', headers=Headers()).finish(200)
        self.assertEqual(row['tool_bytes'], [1000 + len(MARKER), 2000 + len(MARKER), 3000 + len(MARKER)])
        self.assertEqual(row['n_tool_msgs'], 3)

    def test_no_text_fields_are_written(self):
        e = rl.Entry('/v1/chat/completions', payload(n_tool=2), 'opencode/2.2.6 ' + MARKER, 'ornith',
                     headers=Headers({'X-Local-AI-Origin': 'lab', 'X-Session-Id': MARKER}))
        e.admitted_now(0.1, 0.2)
        e.observe(b'data: {"timings":{"cache_n":1,"prompt_n":2,"predicted_n":3}}\n\n')
        e.finish(200)
        text = next(self.tmp.glob('requests-*.jsonl')).read_text()
        self.assertNotIn(MARKER, text)
        row = json.loads(text)
        for key, value in row.items():
            if isinstance(value, str):
                self.assertLessEqual(len(value), 64, key)
            self.assertNotIn(key, ('messages', 'content', 'text', 'prompt', 'reply', 'system'))
        self.assertRegex(row['session'], r'^[0-9a-f]{16}$')
        self.assertNotEqual(row['session'], MARKER)

    def test_files_are_owner_only(self):
        rl.Entry('/x', payload(), None, 'm', headers=Headers()).finish(200)
        f = next(self.tmp.glob('requests-*.jsonl'))
        self.assertEqual(oct(f.stat().st_mode & 0o777), '0o600')
        self.assertEqual(oct(self.tmp.stat().st_mode & 0o777), '0o700')

    def test_retention_removes_old_files(self):
        old = self.tmp / 'requests-20260101.jsonl'
        old.write_text('{}\n')
        past = time.time() - 30 * 86400
        os.utime(old, (past, past))
        recent_file = self.tmp / 'requests-20261111.jsonl'
        recent_file.write_text('{}\n')
        with patch.dict(os.environ, {'LOCAL_AI_LEDGER_DAYS': '14'}):
            rl._pruned_day[0] = None
            rl.Entry('/x', payload(), None, 'm', headers=Headers()).finish(200)
        self.assertFalse(old.exists())
        self.assertTrue(recent_file.exists())


class SummaryScriptTests(unittest.TestCase):
    def run_summary(self, *args):
        return subprocess.run([sys.executable, '-I', str(ROOT / 'tools' / 'ledger-summary'), *args],
                              capture_output=True, text=True, check=True)

    def test_organic_only_percentiles_on_fixture(self):
        got = json.loads(self.run_summary('--json', str(FIXTURE)).stdout)
        self.assertEqual(got['rows'], 7)
        self.assertEqual(got['origins'], {'organic': 5, 'lab': 1, 'unlabelled': 1})
        self.assertEqual(got['organic_requests'], 5)
        ctx = got['metrics']['ctx_tokens']
        self.assertEqual((ctx['n'], ctx['p50'], ctx['p90'], ctx['max']), (5, 42000, 70000, 70000))
        self.assertNotIn(999999, [ctx['max']])
        self.assertEqual(got['metrics']['tool_bytes']['n'], 6)
        self.assertEqual(got['metrics']['tool_bytes']['max'], 30000)
        self.assertEqual(got['share_ctx_over_32k'], 0.8)
        self.assertEqual(got['share_tool_outputs_over_16k'], 0.333)
        self.assertEqual(got['sessions'], 2)
        self.assertEqual(got['turns_per_session'], {'p50': 4, 'p90': 4, 'max': 4})

    def test_table_output_and_empty_dir(self):
        out = self.run_summary(str(FIXTURE)).stdout
        self.assertIn('organic requests 5', out)
        self.assertIn('context tokens / request', out)
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / 'requests-20261112.jsonl').write_text(json.dumps({'origin': 'lab', 'ctx_tokens': 5}) + '\n')
            self.assertIn('no organic rows', self.run_summary(d).stdout)


if __name__ == '__main__':
    unittest.main()
