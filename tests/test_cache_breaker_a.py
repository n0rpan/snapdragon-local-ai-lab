"""Model-free tests for the Breaker A offline root-cause tool on synthetic recordings."""
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
sys.path.insert(0, str(TOOLS))
import cache_breaker_a as bA  # noqa: E402

IM, NL, OPENER = 900, 10, 905
ROLE = {'system': 901, 'user': 902, 'assistant': 903, 'tool': 904}
SYSTEM = {'role': 'system', 'content': 'you are a test agent'}


def enc(text):
    return [b + 1000 for b in text.encode()]


def render(messages, opener=True):
    ids = []
    for m in messages:
        ids += [IM, ROLE[m['role']], NL]
        if m['role'] == 'assistant' and opener:
            ids.append(OPENER)
        ids += enc(m.get('content') or '')
    return ids


def generation_prompt():
    return [IM, ROLE['assistant'], NL, OPENER]


def turn(seq, messages, response, path='/v1/chat/completions'):
    return {'kind': 'turn', 'seq': seq, 'status': 200, 'aborted': False, 'path': path,
            'body': {'messages': messages, 'tools': []}, 'response': response,
            'completion_tokens': 1, 'finish_reason': 'stop'}


def row(seq, prompt_ids, completion_text, agent='ag', matches=True):
    return {'seq': seq, 'agent': agent, 'prompt_ids': prompt_ids,
            'completion_ids': enc(completion_text), 'completion_len_matches': matches}


def write_case(turns, rows):
    d = tempfile.mkdtemp(prefix='breaker-a-')
    rec = os.path.join(d, 'recording.jsonl')
    tok = os.path.join(d, 'tokens.jsonl')
    with open(rec, 'w') as f:
        f.write(json.dumps({'kind': 'header', 'version': 1, 'tool': 'session_replay'}) + '\n')
        for t in turns:
            f.write(json.dumps(t) + '\n')
    with open(tok, 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    return d, rec, tok


class BreakerATests(unittest.TestCase):
    def setUp(self):
        self.dirs = []

    def tearDown(self):
        for d in self.dirs:
            shutil.rmtree(d, ignore_errors=True)

    def run_case(self, turns, rows):
        d, rec, tok = write_case(turns, rows)
        self.dirs.append(d)
        from session_replay import load_recording
        _h, loaded, _sha = load_recording(rec)
        token_rows = [json.loads(line) for line in Path(tok).read_text().splitlines() if line.strip()]
        return bA.analyse(loaded, token_rows, 900)

    def two_turns(self, first_gen_text, second_messages, opener_second):
        """Turn 1 answers `first_gen_text`; turn 2 sends that answer back and adds a user message."""
        hist1 = [SYSTEM, {'role': 'user', 'content': 'hello'}]
        prompt1 = render(hist1) + generation_prompt()
        hist2 = hist1 + [{'role': 'assistant', 'content': first_gen_text}, {'role': 'user', 'content': 'more'}]
        prompt2 = render(hist2, opener=opener_second) + generation_prompt()
        turns = [turn(1, hist1, {'content': first_gen_text}), turn(2, hist2, {'content': 'x'})]
        rows = [row(1, prompt1, first_gen_text), row(2, prompt2, 'x')]
        return turns, rows

    def test_append_only_history_with_the_same_opener_is_clean(self):
        turns, rows = self.two_turns('answer one', None, opener_second=True)
        res = self.run_case(turns, rows)
        self.assertEqual(res['by_result'], {'clean': 1})
        self.assertEqual(res.get('divergent_cause', {}), {})

    def test_previous_generation_prompt_opener_missing_from_rendering_is_attributed(self):
        turns, rows = self.two_turns('answer one', None, opener_second=False)
        res = self.run_case(turns, rows)
        self.assertEqual(res['by_result'].get('divergent'), 1)
        self.assertEqual(res['divergent_region'], {'generation-prompt': 1})
        self.assertEqual(res['divergent_cause'], {'prev-generation-prompt:opener-differs': 1})
        # the divergence is the opener token; everything from there (the completion) is re-prefilled
        self.assertEqual(res['lost_tokens_total'], len(enc('answer one')) + 1)

    def test_client_rewrite_of_earlier_history_is_client_history_changed(self):
        prev_msgs = [SYSTEM, {'role': 'user', 'content': 'hello'},
                     {'role': 'assistant', 'content': 'first answer'}, {'role': 'user', 'content': 'hi2'}]
        cur_msgs = [SYSTEM, {'role': 'user', 'content': 'hello'},
                    {'role': 'assistant', 'content': 'first answer, edited by the client'},
                    {'role': 'user', 'content': 'hi2'}, {'role': 'assistant', 'content': 'answer'},
                    {'role': 'user', 'content': 'more'}]
        turns = [turn(1, prev_msgs, {'content': 'answer'}), turn(2, cur_msgs, {'content': 'ok'})]
        rows = [row(1, render(prev_msgs) + generation_prompt(), 'answer'),
                row(2, render(cur_msgs) + generation_prompt(), 'ok')]
        res = self.run_case(turns, rows)
        self.assertEqual(res['divergent_cause'], {'client-history-changed': 1})
        self.assertEqual(res['divergent_region'], {'history': 1})

    def test_other_conversation_under_the_same_agent_is_not_counted_as_a_divergence(self):
        turns, rows = self.two_turns('answer one', None, opener_second=True)
        turns[1]['body']['messages'][1] = {'role': 'user', 'content': 'a different conversation'}
        turns[1]['body']['messages'] = [SYSTEM, {'role': 'user', 'content': 'a different conversation'}]
        rows[1]['prompt_ids'] = render(turns[1]['body']['messages']) + generation_prompt()
        res = self.run_case(turns, rows)
        self.assertEqual(res['by_result'], {'other-conversation': 1})

    def test_completion_argument_key_order_is_attributed_to_arguments(self):
        gen_args = '{"path": "a.txt", "limit": 5}'
        client_args = '{"limit": 5, "path": "a.txt"}'
        hist1 = [SYSTEM, {'role': 'user', 'content': 'read it'}]
        prompt1 = render(hist1) + generation_prompt()
        response1 = {'content': '', 'tool_calls': [{'name': 'read', 'arguments': gen_args}]}
        client_msg = {'role': 'assistant', 'content': '', 'tool_calls': [
            {'id': 'c1', 'type': 'function', 'function': {'name': 'read', 'arguments': client_args}}]}
        hist2 = hist1 + [client_msg, {'role': 'tool', 'content': 'data'}]
        turns = [turn(1, hist1, response1), turn(2, hist2, {'content': 'ok'})]
        # the engine's own completion is what it generated; the client's copy of it is re-rendered
        prompt2 = render(hist2, opener=True) + generation_prompt()
        rows = [row(1, prompt1, gen_args), row(2, prompt2, 'ok')]
        rows[0]['completion_ids'] = enc(gen_args)
        res = self.run_case(turns, rows)
        self.assertEqual(res['divergent_region'].get('completion'), 1)
        self.assertEqual(res['divergent_cause'], {'arguments-key-order': 1})

    def test_reconstruction_suspect_is_reported_before_field_causes(self):
        hist1 = [SYSTEM, {'role': 'user', 'content': 'read it'}]
        prompt1 = render(hist1) + generation_prompt()
        gen = 'plain answer'
        hist2 = hist1 + [{'role': 'assistant', 'content': 'plain answr'}, {'role': 'user', 'content': 'next'}]
        turns = [turn(1, hist1, {'content': gen}), turn(2, hist2, {'content': 'ok'})]
        rows = [row(1, prompt1, gen, matches=False), row(2, render(hist2) + generation_prompt(), 'ok')]
        # the client's copy of the answer differs inside the completion, and the completion's token
        # reconstruction is flagged as unreliable
        res = self.run_case(turns, rows)
        self.assertEqual(res['divergent_cause'], {'unresolved': 1})

    def test_counts_only_output_has_no_content(self):
        turns, rows = self.two_turns('secret answer text', None, opener_second=False)
        res = self.run_case(turns, rows)
        blob = json.dumps(res)
        self.assertNotIn('secret', blob)
        self.assertNotIn('"prompt_ids"', blob)

    def test_span_roles_group_consecutive_tool_results(self):
        msgs = [SYSTEM, {'role': 'user', 'content': 'x'}, {'role': 'assistant', 'content': 'a'},
                {'role': 'tool', 'content': 't1'}, {'role': 'tool', 'content': 't2'}]
        self.assertEqual(bA.span_roles(msgs), ['system', 'user', 'assistant', 'tool'])

    def test_history_category_maps_spans_to_source_messages_after_coalesced_tool_results(self):
        call = {'role': 'assistant', 'content': 'c', 'tool_calls': [{'function': {'name': 'f', 'arguments': '{}'}}]}
        old = [SYSTEM, {'role': 'user', 'content': 'x'}, call, {'role': 'tool', 'content': 't1'},
               {'role': 'tool', 'content': 't2'}, dict(call, content='d')]
        prompt = []
        for role, src in bA.span_sources(old):
            prompt += [IM, ROLE[role], NL] + enc(old[src]['content'])
        last = len(prompt) - 1               # inside the assistant tool-call span (message 5, span 4)
        prompt += generation_prompt()
        new = old + [{'role': 'tool', 'content': 't3'}]
        cat = bA.history_category({'body': {'messages': old}}, {'body': {'messages': new}}, prompt, prompt,
                                  last, IM, len(prompt) - 4)
        self.assertEqual(cat, 'history-render-changed-assistant-tool-call')
        self.assertEqual(bA.span_sources(old)[3:], [('tool', 3), ('assistant', 5)])

    def test_argument_category_rules(self):
        self.assertIsNone(bA.argument_category('{"a":1}', '{"a":1}'))
        self.assertEqual(bA.argument_category('{"a":1,"b":2}', '{"b":2,"a":1}'), 'arguments-key-order')
        self.assertEqual(bA.argument_category('{"a":1}', '{"a": 1}'), 'arguments-whitespace-or-escape')
        self.assertEqual(bA.argument_category('{"a":1}', '{"a":2}'), 'arguments-value')
        self.assertEqual(bA.argument_category('{"a":1}', '{"a":'), 'arguments-not-json')


if __name__ == '__main__':
    unittest.main()
