"""CACHE-PERFECT M2 (Codex P0 review items 2, 6, 7): predecessor keyed by (session, model, agent,
origin), error/non-chat rows skipped, exact retries and edited replies are `retry`, completeness."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from snapdragon_ai_lab.runtime import cache_miss as cm
from snapdragon_ai_lab.runtime import request_ledger as rl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import cache_miss_eval  # noqa: E402

CHAT = '/v1/chat/completions'
SYS = {'role': 'system', 'content': 'agent-a'}
TITLE = {'role': 'system', 'content': 'title-agent'}


def sse(text=None, calls=None, **timings):
    out = b''
    if text is not None:
        out += b'data: ' + json.dumps({'choices': [{'delta': {'content': text}}]}).encode() + b'\n\n'
    for i, (name, args) in enumerate(calls or ()):
        out += b'data: ' + json.dumps({'choices': [{'delta': {'tool_calls': [
            {'index': i, 'function': {'name': name, 'arguments': args}}]}}]}).encode() + b'\n\n'
    out += b'data: ' + json.dumps({'choices': [{'delta': {}, 'finish_reason': 'stop'}], 'timings': timings}).encode()
    return out + b'\n\ndata: [DONE]\n\n'


class LedgerM2Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        env = patch.dict(os.environ, {'LOCAL_AI_LEDGER_DIR': self.tmp.name})
        env.start()
        self.addCleanup(env.stop)
        rl._salt[0] = None
        rl._history.clear()

    def turn(self, messages, body, model='ornith', origin=None, path=CHAT, status=200, error=None, payload=None):
        headers = {'X-Session-Id': 's-1'}
        if origin:
            headers['X-Local-AI-Origin'] = origin
        entry = rl.Entry(path, payload or {'messages': messages}, 'opencode/2.2.7', model, headers=headers)
        if body:
            for i in range(0, len(body), 7):  # split SSE events across chunks
                entry.observe(body[i:i + 7])
        return entry.finish(status, error)

    def test_title_agent_model_and_origin_between_turns_keep_a_predecessor(self):
        a1 = self.turn([SYS, {'role': 'user', 'content': 'q'}], sse('ok', cache_n=0, prompt_n=100, predicted_n=20))
        for msgs, model, origin in (([TITLE, {'role': 'user', 'content': 'q'}], 'ornith', None),
                                    ([SYS, {'role': 'user', 'content': 'q'}], 'qwen', None),
                                    ([SYS, {'role': 'user', 'content': 'q'}], 'ornith', 'lab')):
            other = self.turn(msgs, sse('t', cache_n=0, prompt_n=30, predicted_n=3), model, origin)
            self.assertEqual(other['history_change'], 'initial')
            self.assertNotIn('miss_class', other)
        a2 = self.turn([SYS, {'role': 'user', 'content': 'q'}, {'role': 'assistant', 'content': 'ok'},
                        {'role': 'user', 'content': 'next'}], sse('ok2', cache_n=40, prompt_n=100, predicted_n=5))
        self.assertEqual(a2['previous_request_id'], a1['id'])
        self.assertEqual(a2['previous_reply'], 'present')
        self.assertEqual((a2['miss_class'], a2['miss_tokens']), ('unknown', 79))  # A's loss is observed

    def test_completion_path_and_http_500_rows_are_not_predecessors(self):
        base = [SYS, {'role': 'user', 'content': 'q'}]
        a1 = self.turn(base, sse('ok', cache_n=0, prompt_n=100, predicted_n=20))
        raw = self.turn(None, b'{"content":"x","timings":{"cache_n":0,"prompt_n":7,"predicted_n":1}}',
                        path='/completion', payload={'prompt': 'x'})
        self.assertNotIn('history_change', raw)
        follow = base + [{'role': 'assistant', 'content': 'ok'}, {'role': 'user', 'content': 'next'}]
        failed = self.turn(follow, b'', status=500, error='local_controller_error')
        self.assertEqual(failed['completeness'], 'errored')
        self.assertNotIn('miss_class', failed)
        a2 = self.turn(follow, sse('ok2', cache_n=119, prompt_n=10, predicted_n=5))
        self.assertEqual((a2['previous_request_id'], a2['history_change']), (a1['id'], 'append'))
        self.assertEqual(a2['miss_class'], 'append-ok')

    def test_repeated_prompt_is_a_retry_not_a_rerender_miss(self):
        base = [SYS, {'role': 'user', 'content': 'q'}]
        self.turn(base, sse('ok', cache_n=0, prompt_n=100, predicted_n=50))
        again = self.turn(base, sse('ok', cache_n=99, prompt_n=1, predicted_n=50))
        self.assertEqual((again['history_change'], again['previous_reply']), ('same', 'absent'))
        self.assertEqual((again['miss_class'], again['miss_tokens']), ('retry', 0))
        # a retry that lost the prompt itself is still the alarm
        lost = self.turn(base, sse('ok', cache_n=10, prompt_n=90, predicted_n=50))
        self.assertEqual((lost['miss_class'], lost['miss_tokens']), ('unknown', 89))

    def test_edited_prior_output_is_a_retry_and_the_unedited_reply_is_eligible(self):
        base = [SYS, {'role': 'user', 'content': 'q'}]
        self.turn(base, sse('the answer', cache_n=0, prompt_n=100, predicted_n=40))
        edited = self.turn(base + [{'role': 'assistant', 'content': 'an edited answer'}, {'role': 'user', 'content': 'n'}],
                           sse('x', cache_n=99, prompt_n=60, predicted_n=4))
        self.assertEqual((edited['previous_reply'], edited['miss_class'], edited['miss_tokens']), ('edited', 'retry', 0))
        rl._history.clear()
        self.turn(base, sse('the answer', cache_n=0, prompt_n=100, predicted_n=40))
        kept = self.turn(base + [{'role': 'assistant', 'content': ' the answer\n'}, {'role': 'user', 'content': 'n'}],
                         sse('x', cache_n=99, prompt_n=60, predicted_n=4))
        self.assertEqual(kept['previous_reply'], 'present')  # trimmed text compares equal
        self.assertEqual((kept['miss_class'], kept['miss_tokens']), ('reasoning-drop', 40))  # whole completion lost

    def test_tool_call_reply_digest_ignores_argument_key_order(self):
        base = [SYS, {'role': 'user', 'content': 'q'}]
        self.turn(base, sse(calls=[('read', '{"b": 1, "a": "x"}')], cache_n=0, prompt_n=100, predicted_n=30))
        call = {'role': 'assistant', 'content': '', 'tool_calls': [
            {'id': 'c1', 'type': 'function', 'function': {'name': 'read', 'arguments': '{"a":"x","b":1}'}}]}
        row = self.turn(base + [call, {'role': 'tool', 'content': 'r'}], sse('d', cache_n=129, prompt_n=9, predicted_n=2))
        self.assertEqual((row['previous_reply'], row['miss_class']), ('present', 'append-ok'))

    def test_non_stream_json_reply_is_digested(self):
        base = [SYS, {'role': 'user', 'content': 'q'}]
        body = json.dumps({'choices': [{'message': {'role': 'assistant', 'content': 'plain'}, 'finish_reason': 'stop'}],
                           'timings': {'cache_n': 0, 'prompt_n': 50, 'predicted_n': 3}}).encode()
        self.turn(base, body)
        row = self.turn(base + [{'role': 'assistant', 'content': 'plain'}, {'role': 'user', 'content': 'n'}],
                        sse('x', cache_n=52, prompt_n=9, predicted_n=2))
        self.assertEqual(row['previous_reply'], 'present')

    def test_completeness_closed_after_done_is_valid_and_partial_is_not(self):
        base = [SYS, {'role': 'user', 'content': 'q'}]
        closed = self.turn(base, sse('ok', cache_n=0, prompt_n=100, predicted_n=20), error='client_disconnected')
        self.assertEqual(closed['completeness'], 'closed-after-done')
        follow = base + [{'role': 'assistant', 'content': 'ok'}, {'role': 'user', 'content': 'n'}]
        partial = self.turn(follow, b'data: {"choices":[{"delta":{"content":"he"}}]}\n\n', error='client_disconnected')
        self.assertEqual(partial['completeness'], 'incomplete')
        self.assertNotIn('miss_class', partial)
        nxt = self.turn(follow, sse('x', cache_n=119, prompt_n=9, predicted_n=2))
        self.assertEqual((nxt['previous_request_id'], nxt['miss_class']), (closed['id'], 'append-ok'))
        died = self.turn(follow, sse('x', cache_n=0, prompt_n=9, predicted_n=2), error='backend_died')
        self.assertEqual(died['completeness'], 'errored')

    def test_digest_and_history_hold_no_text(self):
        self.turn([SYS, {'role': 'user', 'content': 'q'}], sse('PRIVATE-REPLY', cache_n=0, prompt_n=10, predicted_n=2))
        self.assertNotIn('PRIVATE', repr(rl._history))
        disk = ''.join(p.read_text() for p in Path(self.tmp.name).glob('requests-*.jsonl'))
        self.assertNotIn('PRIVATE', disk)


class SummarizeTests(unittest.TestCase):
    def test_summarize_checks_path_status_error_and_completeness(self):
        ok = {'cache_n': 1, 'prompt_n': 2, 'predicted_n': 1, 'path': CHAT, 'status': 200, 'error': None}
        self.assertIsNotNone(cm.summarize(ok))
        for bad in ({'path': '/completion'}, {'status': 500}, {'error': 'backend_died'}, {'completeness': 'incomplete'},
                    {'completeness': 'errored'}):
            self.assertIsNone(cm.summarize(dict(ok, **bad)), bad)
        self.assertIsNotNone(cm.summarize(dict(ok, error='client_disconnected', completeness='closed-after-done')))

    def test_other_session_or_origin_is_not_judged(self):
        prev = cm.summarize({'cache_n': 0, 'prompt_n': 100, 'predicted_n': 5, 'agent': 'a', 'model': 'm', 'session': 's1'})
        cur = cm.summarize({'cache_n': 0, 'prompt_n': 100, 'agent': 'a', 'model': 'm', 'session': 's2'})
        self.assertEqual(cm.classify(prev, cur), (None, None, None))
        lab = cm.summarize({'cache_n': 0, 'prompt_n': 100, 'agent': 'a', 'model': 'm', 'session': 's1', 'origin': 'lab'})
        self.assertEqual(cm.classify(dict(prev, origin='organic'), lab), (None, None, None))


def erow(i, session, cache, new, t, n=4, **extra):
    base = {'id': 'r%d' % i, 't': t, 'path': CHAT, 'status': 200, 'error': None, 'cache_n': cache, 'prompt_n': new,
            'predicted_n': 10, 'n_messages': n, 'last_role': 'user', 'agent': 'a', 'model': 'm', 'session': session,
            'origin': 'organic', 'client': 'opencode/2.2.7'}
    base.update(extra)
    return base


class EvaluateTests(unittest.TestCase):
    def test_new_sessions_of_one_agent_with_equal_counts_are_not_misses(self):
        rows = [erow(1, 's1', 0, 500, 1.0), erow(2, 's2', 0, 400, 2.0), erow(3, 's3', 0, 450, 3.0)]
        got = cache_miss_eval.evaluate(rows)
        self.assertEqual(got['judged'], 0)
        self.assertEqual(got['miss_class'], {})

    def test_previous_request_id_is_exact_and_old_rows_are_heuristic(self):
        rows = [erow(1, 's1', 0, 100, 1.0), erow(2, 's1', 0, 300, 2.0, n=2, agent='title'),
                erow(3, 's1', 109, 20, 3.0, n=6, previous_request_id='r1'),
                erow(4, None, 0, 100, 4.0), erow(5, None, 109, 5, 5.0, n=6)]
        got = cache_miss_eval.evaluate(rows)
        self.assertEqual(got['chain'], {'previous_request_id': 1, 'heuristic': 1})
        self.assertEqual(got['basis'], {'counts': 1, 'heuristic-counts': 1})
        self.assertEqual(got['miss_class'], {'append-ok': 2})

    def test_error_rows_are_skipped_as_predecessors(self):
        rows = [erow(1, 's1', 0, 100, 1.0), erow(2, 's1', 0, 5, 2.0, status=500, error='x'),
                erow(3, 's1', 109, 20, 3.0, n=6)]
        got = cache_miss_eval.evaluate(rows)
        self.assertEqual((got['miss_class'], got['not_chat_or_incomplete']), ({'append-ok': 1}, 1))


if __name__ == '__main__':
    unittest.main()
