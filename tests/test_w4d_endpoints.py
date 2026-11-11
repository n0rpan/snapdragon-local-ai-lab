"""W4d endpoints: /_local/status capabilities, /_local/ledger rows and labels, the rule that
control routes never load a model, the LOCAL_AI_LEDGER=0 switch, request_ledger.recent and
the reserved alias name. Model-free: the controller runs over the Python FakeEngine."""
import os as _os, tempfile as _tempfile
_os.environ.setdefault('LOCAL_AI_LEDGER_DIR', _tempfile.mkdtemp(prefix='local-ai-test-ledger-'))  # never the real request ledger
import importlib.util
import json
import os
import re
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from snapdragon_ai_lab.runtime import openai_api, request_ledger

import openai_api_stack as stack_module
from openai_api_stack import FakeEngine, ORNITH, QWEN, Stack

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('openai_api_check', ROOT / 'tools/openai_api_check.py')
check = importlib.util.module_from_spec(spec); spec.loader.exec_module(check)

PROBE = 'ledger probe text 7431'


def chat(base, text=PROBE, system=None, user_agent=None):
    messages = ([{'role': 'system', 'content': system}] if system else []) + [{'role': 'user', 'content': text}]
    headers = {'User-Agent': user_agent} if user_agent else None
    return check.call(base, 'POST', '/v1/chat/completions', {'model': 'local-default', 'messages': messages}, headers=headers)


def ledger(base, limit=50):
    return check.call(base, 'GET', '/_local/ledger?limit=%s' % limit)


def wait_for_rows(base, count, limit=50, seconds=5.0):
    """The ledger row is written in the handler's `finally`, after the reply has been sent,
    so a client may ask for it a moment early."""
    end = time.monotonic() + seconds
    while True:
        value = ledger(base, limit)[2]
        if len(value['rows']) >= count or time.monotonic() > end:
            return value
        time.sleep(0.05)


class ServerEndpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = FakeEngine()
        cls.addClassCleanup(cls.engine.close)

    def stack(self, **config):
        stack = Stack(self.engine.port, **config)
        self.addCleanup(stack.close)
        return stack

    def ledger_dir(self, **extra):
        """A fresh ledger directory, patched in for the rest of the test."""
        tmp = tempfile.TemporaryDirectory(prefix='w4d-ledger-')
        self.addCleanup(tmp.cleanup)
        patcher = patch.dict(os.environ, {'LOCAL_AI_LEDGER_DIR': tmp.name, 'LOCAL_AI_LEDGER': '1', **extra})
        patcher.start()
        self.addCleanup(patcher.stop)
        return Path(tmp.name)

    def test_status_capabilities_and_idle_active_block(self):
        stack = self.stack()
        status, _, value = check.call(stack.base, 'GET', '/_local/status')
        self.assertEqual(status, 200)
        for name in ('models', 'storage', 'harness', 'ledger'):
            self.assertIn(name, value['capabilities'])
        self.assertIsInstance(value['capabilities'], list)
        active = value['active']
        self.assertIsInstance(active, dict)
        self.assertEqual(active['phase'], 'idle')
        self.assertEqual(active['queued_requests'], 0)
        for key in ('prompt_tokens_done', 'prompt_tokens_total', 'tokens_out', 'tokens_per_second'):
            self.assertIn(key, active)
            self.assertIsNone(active[key])

    def test_ledger_rows_newest_first_and_no_content(self):
        self.ledger_dir()
        stack = self.stack()
        status, _, value = ledger(stack.base)
        self.assertEqual(status, 200)
        self.assertTrue(value['available'] is False or value['rows'] == [])

        self.assertEqual(chat(stack.base, user_agent='TestClient/1.0')[0], 200)
        self.assertEqual(chat(stack.base, user_agent='TestClient/1.0')[0], 200)
        status, _, _ = check.call(stack.base, 'POST', '/v1/completions', {'model': 'local-default', 'prompt': 'hi'})
        self.assertEqual(status, 200)
        wait_for_rows(stack.base, 3)

        status, _, value = ledger(stack.base, 2)
        self.assertEqual(status, 200)
        rows = value['rows']
        self.assertEqual(len(rows), 2)
        times = [row['t'] for row in rows]
        self.assertEqual(times, sorted(times, reverse=True))  # newest first
        for row in rows:
            self.assertEqual(row['status'], 200)
            self.assertEqual(row['model'], ORNITH)
            self.assertIsInstance(row['wall_ms'], (int, float))
            self.assertIn('queue_wait_ms', row)
            self.assertIn('admit_ms', row)
        # rows[0] is the /v1/completions request (no User-Agent sent), rows[1] the second chat request
        self.assertEqual(rows[1]['client'], 'TestClient/1.0')
        self.assertIsInstance(value['medians'], dict)
        self.assertIn('wall_ms', value['medians'])

        for bad in ('abc', '99999'):
            status, _, _ = ledger(stack.base, bad)
            self.assertEqual(status, 200, bad)  # NOTE: the controller falls back to limit 50 for a non-numeric or 5-digit limit; it does not clamp '99999' to 200.

        full = ledger(stack.base)[2]['rows']
        self.assertNotIn(PROBE, json.dumps(full))

    def test_ledger_rows_carry_harness_labels(self):
        self.ledger_dir()
        stack = self.stack()
        status, _, _ = chat(stack.base, system='You are a coding assistant running inside pi 1.5.1. Use the repo rules.')
        self.assertEqual(status, 200)
        value = wait_for_rows(stack.base, 1, limit=1)
        row = value['rows'][0]
        self.assertEqual(row['harness'], 'pi')
        self.assertIsInstance(row['prefix_key'], str)
        self.assertRegex(row['prefix_key'], r'^[0-9a-f]{8,64}$')

    def test_control_routes_never_load_a_model(self):
        stack = self.stack()
        before = len(self.engine.requests)
        for method, path, body, expected in (('GET', '/_local/ledger', None, 200),
                                             ('GET', '/_local/harnesses', None, 200),
                                             ('GET', '/_local/nothing', None, 404),
                                             ('POST', '/_local/nothing', {}, 404)):
            status, _, _ = check.call(stack.base, method, path, body)
            self.assertEqual(status, expected, method + ' ' + path)
        self.assertEqual(stack.manager.loads, 0)
        self.assertEqual(len(self.engine.requests), before)

    def test_ledger_off_writes_nothing(self):
        tmp = tempfile.TemporaryDirectory(prefix='w4d-ledger-off-')
        self.addCleanup(tmp.cleanup)
        with patch.dict(os.environ, {'LOCAL_AI_LEDGER': '0', 'LOCAL_AI_LEDGER_DIR': tmp.name}):
            stack = self.stack()
            status, _, _ = chat(stack.base)
            self.assertEqual(status, 200)
            time.sleep(0.2)  # the row would be written after the reply, if it were written at all
            self.assertEqual(os.listdir(tmp.name), [])


class LedgerUnits(unittest.TestCase):
    def ledger_dir(self):
        tmp = tempfile.TemporaryDirectory(prefix='w4d-recent-')
        self.addCleanup(tmp.cleanup)
        patcher = patch.dict(os.environ, {'LOCAL_AI_LEDGER_DIR': tmp.name, 'LOCAL_AI_LEDGER': '1'})
        patcher.start()
        self.addCleanup(patcher.stop)
        return Path(tmp.name)

    def test_recent_order_and_medians(self):
        self.ledger_dir()
        request_ledger.write({'status': 200, 'wall_ms': 100, 'prompt_n': 100, 'prompt_ms': 500,
                              'predicted_n': 50, 'predicted_ms': 1000})
        request_ledger.write({'status': 200, 'wall_ms': 300, 'prompt_n': 200, 'prompt_ms': 1000,
                              'predicted_n': 100, 'predicted_ms': 2000})
        request_ledger.write({'status': 503, 'wall_ms': 50})
        value = request_ledger.recent(10)
        self.assertTrue(value['available'])
        self.assertEqual(len(value['rows']), 3)
        self.assertEqual([row['status'] for row in value['rows']], [503, 200, 200])  # newest first
        self.assertEqual(value['answered'], 2)
        self.assertEqual(value['medians']['wall_ms'], 200)
        self.assertEqual(value['medians']['prompt_tokens_per_second'], 200.0)
        self.assertEqual(value['medians']['predicted_tokens_per_second'], 50.0)

    def test_garbage_line_is_skipped(self):
        directory = self.ledger_dir()
        day = time.strftime('%Y%m%d')
        lines = [json.dumps({'status': 200, 'wall_ms': 10}), 'this is not json', json.dumps({'status': 503, 'wall_ms': 20})]
        (directory / ('requests-%s.jsonl' % day)).write_text('\n'.join(lines) + '\n')
        value = request_ledger.recent(10)
        self.assertTrue(value['available'])
        self.assertEqual(len(value['rows']), 2)
        self.assertEqual(value['answered'], 1)

    def test_missing_directory_is_unavailable(self):
        directory = self.ledger_dir()
        with patch.dict(os.environ, {'LOCAL_AI_LEDGER_DIR': str(directory / 'does-not-exist')}):
            self.assertFalse(request_ledger.recent(10)['available'])


class ReservedAlias(unittest.TestCase):
    def test_reserved_name_is_not_an_alias(self):
        table = openai_api.alias_table({'profiles': {}, 'openai_api': {'model_aliases': {'local-auto': ORNITH, 'local-fast': QWEN}}})
        self.assertIn('local-fast', table)
        self.assertNotIn('local-auto', table)


if __name__ == '__main__':
    unittest.main()
