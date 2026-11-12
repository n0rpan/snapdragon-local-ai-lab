"""Model-free tests for the per-turn cache-miss classifier and its ledger wiring (CACHE-PERFECT P0)."""
import os
import shutil
import tempfile
import unittest

from snapdragon_ai_lab.runtime import cache_miss as cm
from snapdragon_ai_lab.runtime import request_ledger as rl


def row(cache, new, predicted=0, n=4, last='user', agent='a1', model='m1', finish='stop', **extra):
    base = {'cache_n': cache, 'prompt_n': new, 'predicted_n': predicted, 'n_messages': n,
            'last_role': last, 'agent': agent, 'model': model, 'finish_reason': finish}
    base.update(extra)
    return cm.summarize(base)


class ClassifyTests(unittest.TestCase):
    def test_append_ok_when_previous_prompt_and_completion_are_reused(self):
        prev = row(100, 50, predicted=20, n=3, last='user')
        cur = row(169, 30, n=5, last='user')          # expected 150 + 20 - 1 = 169 reused
        self.assertEqual(cm.classify(prev, cur), ('append-ok', 0, 'counts'))

    def test_exact_c3_boundary_is_not_a_miss(self):
        prev = row(100, 50, predicted=20, n=3, last='user')
        self.assertEqual(cm.classify(prev, row(169, 1, n=5))[0], 'append-ok')  # 169 == 170 - 1
        self.assertEqual(cm.classify(prev, row(168, 1, n=5))[0], 'unknown')    # 2 tokens short

    def test_whole_previous_completion_rerendered_after_tool_call_is_tool_rerender(self):
        prev = row(100, 50, predicted=71, n=3, last='user', finish='tool_calls')
        cur = row(148, 80, n=5, last='tool')          # expected 169+... shortfall 72 ~ predicted 71 (Breaker A)
        self.assertEqual(cm.classify(prev, cur), ('tool-rerender', 72, 'counts'))

    def test_whole_completion_after_plain_stop_is_reasoning_drop(self):
        prev = row(100, 50, predicted=71, n=3, last='user', finish='stop')
        cur = row(150, 80, n=5, last='user')
        self.assertEqual(cm.classify(prev, cur)[0], 'reasoning-drop')

    def test_history_shrink_is_compaction(self):
        prev = row(100, 50, predicted=20, n=12, last='tool')
        cur = row(20, 3000, n=4, last='user')
        self.assertEqual(cm.classify(prev, cur)[0], 'compaction')

    def test_user_turn_adding_one_message_with_partial_loss_is_reasoning_drop(self):
        prev = row(1000, 100, predicted=50, n=6, last='tool')   # expected 1149
        cur = row(250, 900, n=7, last='user')                    # 899 short, one user message added
        self.assertEqual(cm.classify(prev, cur), ('reasoning-drop', 899, 'counts'))

    def test_user_turn_with_same_message_count_is_history_edit(self):
        prev = row(1000, 100, predicted=50, n=6, last='assistant')
        cur = row(400, 900, n=6, last='user')
        self.assertEqual(cm.classify(prev, cur)[0], 'history-edit')

    def test_tool_result_shortfall_is_tool_rerender(self):
        prev = row(1000, 100, predicted=50, n=6, last='user')
        cur = row(600, 500, n=8, last='tool')
        self.assertEqual(cm.classify(prev, cur)[0], 'tool-rerender')

    def test_unexplained_shortfall_is_unknown(self):
        prev = row(1000, 100, predicted=50, n=6, last='user')
        cur = row(600, 500, n=7, last='assistant')
        self.assertEqual(cm.classify(prev, cur)[0], 'unknown')

    def test_divergence_fields_override_counts(self):
        prev = row(100, 50, predicted=20, n=5, last='tool', finish='tool_calls')
        last_assistant = row(120, 900, n=7, last='user', first_divergent_role='assistant', first_divergent_index=4)
        # an assistant hash change is no proof of the cause: basis inferred (Codex P0 item 1)
        self.assertEqual(cm.classify(prev, last_assistant), ('tool-rerender', 49, 'inferred'))
        older = row(120, 900, n=7, last='user', first_divergent_role='assistant', first_divergent_index=1)
        self.assertEqual(cm.classify(prev, older)[0], 'reasoning-drop')
        tool = row(120, 900, n=7, last='user', first_divergent_role='tool', first_divergent_index=3)
        self.assertEqual(cm.classify(prev, tool)[0], 'tool-rerender')
        user = row(120, 900, n=7, last='user', first_divergent_role='user', first_divergent_index=2)
        self.assertEqual(cm.classify(prev, user)[0], 'history-edit')

    def test_template_change_is_template_drift(self):
        prev = row(1000, 100, predicted=50, n=6, last='user', template_sha='aa')
        cur = row(400, 900, n=7, last='user', template_sha='bb')
        self.assertEqual(cm.classify(prev, cur)[0], 'template-drift')

    def test_no_predecessor_or_other_agent_or_model_is_not_judged(self):
        cur = row(10, 10, n=2)
        self.assertEqual(cm.classify(None, cur), (None, None, None))
        self.assertEqual(cm.classify(row(100, 50, predicted=5, agent='x'), cur), (None, None, None))
        self.assertEqual(cm.classify(row(100, 50, predicted=5, model='other'), cur), (None, None, None))

    def test_rows_without_backend_counts_are_not_summarised(self):
        self.assertIsNone(cm.summarize({'status': 500, 'n_messages': 3}))
        self.assertIsNone(cm.summarize({'cache_n': 'x', 'prompt_n': 3}))

    def test_summary_keeps_numbers_and_enums_only(self):
        s = cm.summarize({'cache_n': 5, 'prompt_n': 6, 'predicted_n': 2, 'n_messages': 3, 'last_role': 'user',
                          'agent': 'h', 'model': 'm', 'messages': [{'content': 'secret'}],
                          'first_divergent_role': 'system-prompt-text', 'first_divergent_index': 'x'})
        self.assertNotIn('secret', repr(s))
        self.assertIsNone(s['div_role'])
        self.assertIsNone(s['div_index'])


class LedgerWiringTests(unittest.TestCase):
    def setUp(self):
        self.home = tempfile.mkdtemp(prefix='cache-miss-home-')
        self.ledger = os.path.join(self.home, 'ledger')
        self.old_env = os.environ.get('LOCAL_AI_LEDGER_DIR')
        os.environ['LOCAL_AI_LEDGER_DIR'] = self.ledger
        self.session = 'test-session-%s' % os.path.basename(self.home)
        rl._history.clear()
        rl._salt[0] = None

    def tearDown(self):
        if self.old_env is None:
            os.environ.pop('LOCAL_AI_LEDGER_DIR', None)
        else:
            os.environ['LOCAL_AI_LEDGER_DIR'] = self.old_env
        shutil.rmtree(self.home, ignore_errors=True)

    def finish(self, messages, timings):
        payload = {'messages': messages, 'tools': []}
        entry = rl.Entry('/v1/chat/completions', payload, user_agent='opencode/2.2.7',
                         model_alias='m1', headers={'X-Session-Id': self.session})
        entry.observe(('data: {"timings": ' + repr(timings).replace("'", '"') + '}\n').encode())
        return entry.finish(200)

    def test_second_turn_gets_a_class_and_first_turn_gets_none(self):
        sys_msg = {'role': 'system', 'content': 'sys'}
        first = self.finish([sys_msg, {'role': 'user', 'content': 'hi'}],
                            {'cache_n': 10, 'prompt_n': 40, 'predicted_n': 5})
        self.assertNotIn('miss_class', first)
        second = self.finish([sys_msg, {'role': 'user', 'content': 'hi'}, {'role': 'assistant', 'content': 'yo'},
                              {'role': 'user', 'content': 'again'}],
                             {'cache_n': 54, 'prompt_n': 10, 'predicted_n': 5})
        self.assertEqual(second['miss_class'], 'append-ok')
        self.assertEqual(second['miss_tokens'], 0)

    def test_unknown_miss_logs_a_warning_without_content(self):
        sys_msg = {'role': 'system', 'content': 'sys'}
        self.finish([sys_msg, {'role': 'user', 'content': 'hi'}], {'cache_n': 10, 'prompt_n': 40, 'predicted_n': 5})
        with self.assertLogs('local_ai.ledger', level='WARNING') as logs:
            # history unchanged and appended, yet the cached prefix is gone: nothing explains it
            bad = self.finish([sys_msg, {'role': 'user', 'content': 'hi'}, {'role': 'assistant', 'content': 'x'},
                               {'role': 'tool', 'content': 'y'}, {'role': 'assistant', 'content': 'SECRET-PROMPT-TEXT'}],
                              {'cache_n': 5, 'prompt_n': 300, 'predicted_n': 5})
        self.assertEqual(bad['miss_class'], 'unknown')
        self.assertNotIn('SECRET', ' '.join(logs.output))
        self.assertNotIn('SECRET', repr(bad))

    def test_classifier_fault_never_costs_the_row(self):
        sys_msg = {'role': 'system', 'content': 'sys'}
        original = cm.classify
        def boom(*_args):
            raise RuntimeError('boom')
        cm.classify = boom
        try:
            r = self.finish([sys_msg, {'role': 'user', 'content': 'hi'}], {'cache_n': 1, 'prompt_n': 2, 'predicted_n': 1})
        finally:
            cm.classify = original
        self.assertEqual(r['status'], 200)


if __name__ == '__main__':
    unittest.main()
