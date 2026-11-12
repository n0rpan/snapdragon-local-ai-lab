"""Distinguish edited history from cache loss with identical appended messages."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from snapdragon_ai_lab.runtime import request_ledger as rl


class LedgerHistoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        env = patch.dict(os.environ, {'LOCAL_AI_LEDGER_DIR': self.tmp.name})
        env.start()
        self.addCleanup(env.stop)
        rl._salt[0] = None
        rl._history.clear()
        self.headers = {'X-Session-Id': 'synthetic-session', 'X-Local-AI-Origin': 'lab'}
        self.messages = [{'role': 'system', 'content': 'private-system'}, {'role': 'user', 'content': 'private-user'}]

    def entry(self, messages=None, model='ornith', headers=None):
        return rl.Entry('/v1/chat/completions', {'messages': self.messages if messages is None else messages},
                        model_alias=model, headers=self.headers if headers is None else headers)

    def complete(self, entry, cache=100, new=20, generated=10):
        entry.observe(json.dumps({'timings': {'cache_n': cache, 'prompt_n': new, 'predicted_n': generated}}).encode())
        return entry.finish(200)

    def test_append_with_lost_completion_is_not_a_history_rewrite(self):
        first = self.entry()
        self.complete(first)
        second = self.entry(self.messages + [{'role': 'assistant', 'content': 'private-answer'},
                                            {'role': 'tool', 'content': 'private-tool-result'}])
        row = self.complete(second, cache=120)
        self.assertEqual(row['history_change'], 'append')
        self.assertEqual(row['previous_request_id'], first.row['id'])
        self.assertEqual(row['history_common_messages'], 2)
        self.assertEqual(row['cache_shortfall_tokens'], 10)
        self.assertNotIn('first_divergent_message_hash', row)

    def test_edit_names_first_changed_message_with_keyed_hashes_only(self):
        self.complete(self.entry())
        edited = [self.messages[0], {'role': 'user', 'content': 'private-edited'}]
        row = self.complete(self.entry(edited))
        self.assertEqual((row['history_change'], row['first_divergent_message']), ('edited', 1))
        self.assertRegex(row['first_divergent_message_hash'], r'^[0-9a-f]{32}$')
        self.assertNotEqual(row['first_divergent_message_hash'], row['previous_divergent_message_hash'])
        disk = ''.join(p.read_text() for p in Path(self.tmp.name).glob('requests-*.jsonl'))
        self.assertNotIn('private-', disk)
        self.assertNotIn('private-', repr(rl._history))

    def test_truncation_has_no_invented_current_message(self):
        self.complete(self.entry())
        row = self.complete(self.entry(self.messages[:1]))
        self.assertEqual(row['history_change'], 'truncated')
        self.assertEqual(row['first_divergent_message'], 1)
        self.assertIsNone(row['first_divergent_message_hash'])

    def test_model_agent_and_origin_do_not_cross_contaminate(self):
        self.complete(self.entry())
        for entry in (self.entry(model='qwen'), self.entry([{'role': 'system', 'content': 'other'}, self.messages[1]]),
                      self.entry(headers={'X-Session-Id': 'synthetic-session'})):
            self.assertEqual(entry.row['history_change'], 'initial')

    def test_out_of_order_completion_cannot_replace_current_request(self):
        first = self.entry()
        second = self.entry(self.messages + [{'role': 'assistant', 'content': 'new'}])
        self.complete(first)
        # First was in flight at second's entry; do not pretend its counts were available.
        self.assertNotIn('previous_expected_cache', second.row)
        self.complete(second, cache=120, new=30, generated=15)
        third = self.entry(self.messages + [{'role': 'assistant', 'content': 'new'}])
        self.assertEqual(third.row['previous_expected_cache'], 165)
        self.assertEqual(third.row['previous_request_id'], second.row['id'])

    def test_bounded_storage_and_over_limit_resets_the_chain(self):
        with patch.object(rl, 'HISTORY_HASH_BUDGET', 4):
            for i in range(10):
                self.entry(headers={'X-Session-Id': f'session-{i}'})
            self.assertLessEqual(sum(len(x['hashes']) for x in rl._history.values()), 4)
        self.complete(self.entry())
        with patch.object(rl, 'HISTORY_MESSAGES', 1):
            self.assertEqual(self.entry().row['history_change'], 'over_limit')
        self.assertEqual(self.entry().row['history_change'], 'initial')

    def test_hash_failure_cannot_break_request_or_emit_content(self):
        with patch.object(rl, '_message_hash', side_effect=ValueError('private-failure')):
            row = self.complete(self.entry())
        self.assertEqual(row['status'], 200)
        self.assertNotIn('private-failure', json.dumps(row))

    def test_empty_or_nonprivate_salt_never_produces_unkeyed_history_hashes(self):
        salt = Path(self.tmp.name) / '.salt'
        for content, mode in ((b'', 0o600), (b'x' * 32, 0o644)):
            rl._salt[0] = None
            salt.write_bytes(content)
            salt.chmod(mode)
            row = self.complete(self.entry())
            self.assertIsNone(row['session'])
            self.assertNotIn('history_change', row)
            self.assertEqual(row['status'], 200)


if __name__ == '__main__':
    unittest.main()
