"""Model-free tests for tools/closed_loop_session.py. Only the built-in fake server on 127.0.0.1."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location('closed_loop_session', ROOT / 'tools/closed_loop_session.py')
cl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cl)
sys.path.insert(0, str(ROOT / 'src'))
from snapdragon_ai_lab.runtime import cache_turn  # noqa: E402


def _run(argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cl.main(argv)
    return code, out.getvalue(), err.getvalue()


class ClosedLoopTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.ledger = os.path.join(self.tmp.name, 'ledger')

    def live(self, srv, *extra, target='16K', name='r.json'):
        receipt = os.path.join(self.tmp.name, name)
        code, out, err = _run(['--model', 'fake', '--base-url', srv.url, '--receipt', receipt,
                               '--target-context', target, '--ledger-dir', self.ledger, *extra])
        with open(receipt) as f:
            return code, json.load(f), out, err

    def test_thinking_on_closed_loop_passes_and_echoes_actual_replies(self):
        with cl.FakeServer(ledger_dir=self.ledger) as srv:
            code, rec, _, _ = self.live(srv, '--thinking')
            bodies, headers = srv.requests, srv.headers_seen
        self.assertEqual(code, 0, json.dumps(rec['verdict']['gates']))
        self.assertTrue(rec['verdict']['passed'])
        self.assertTrue(all(h.get('X-Local-AI-Origin') == 'lab' for h in headers))
        self.assertTrue(all(b['chat_template_kwargs'] == {'enable_thinking': True, 'preserve_thinking': True}
                            for b in bodies))
        # every non-compaction request after the first carries the previous ACTUAL reply
        echoed = 0
        for prev, cur in zip(bodies, bodies[1:]):
            n = len(prev['messages'])
            if cur['messages'][:n] != prev['messages']:
                continue  # compaction / post-compaction: a new layout
            reply = cur['messages'][n]
            self.assertEqual(reply['role'], 'assistant')
            self.assertTrue(reply.get('reasoning_content', '').startswith('Considering'))
            for call in reply.get('tool_calls') or []:
                self.assertTrue(call['function']['arguments'].startswith('{"filePath": "src/fake_'))
                results = [m for m in cur['messages'][n + 1:] if m['role'] == 'tool']
                self.assertEqual(results[0]['tool_call_id'], call['id'])
                self.assertGreater(len(results[0]['content']), 0)
            echoed += 1
        self.assertGreater(echoed, 5)
        kinds = [r['kind'] for r in rec['requests']]
        self.assertIn('compaction', kinds)
        self.assertEqual(kinds.count('post-compaction'), 1)
        self.assertGreater(rec['verdict']['reasoning_requests'], 0)
        self.assertEqual(rec['verdict']['gates']['unknown_miss_class']['ledger_rows'], len(rec['requests']))

    def test_compaction_layout_is_what_the_controller_labels(self):
        with cl.FakeServer(ledger_dir=self.ledger) as srv:
            code, rec, _, _ = self.live(srv)
            bodies = srv.requests
        self.assertEqual(code, 0)
        labels = [cache_turn.label(b)['turn_kind'] for b in bodies]
        i = labels.index('compaction')
        self.assertEqual(labels[i + 1], 'post-compaction')
        post = cache_turn.label(bodies[i + 1])
        self.assertEqual(post['compact_layout'], 'ok')
        self.assertEqual(bodies[i]['tools'], bodies[i - 1]['tools'])
        # the summary request is a prefix cut of the history plus the instruction
        self.assertEqual(bodies[i]['messages'][0], bodies[i - 1]['messages'][0])
        self.assertIn('Summary:', bodies[i + 1]['messages'][1]['content'])
        self.assertTrue(all(b['chat_template_kwargs']['enable_thinking'] is False for b in bodies))
        comp = rec['verdict']['gates']['compaction']
        self.assertTrue(comp['pass'] and comp['post_compaction_cached'][0] > 0)

    def test_reasoning_drop_sends_no_reasoning(self):
        with cl.FakeServer(ledger_dir=self.ledger) as srv:
            code, rec, _, _ = self.live(srv, '--thinking', '--reasoning-echo', 'drop', '--no-compaction')
            bodies = srv.requests
        self.assertEqual(code, 0, json.dumps(rec['verdict']['gates']))
        self.assertFalse(any('reasoning_content' in m for b in bodies for m in b['messages']))
        self.assertFalse(any(r['kind'] == 'compaction' for r in rec['requests']))

    def test_broken_echo_fails_hit_and_unknown_gates(self):
        with cl.FakeServer(ledger_dir=self.ledger, break_echo=True) as srv:
            code, rec, _, _ = self.live(srv, '--no-compaction')
        self.assertEqual(code, 1)
        g = rec['verdict']['gates']
        # a lost reply is small next to the history: the aggregate can stay >= 99%, the
        # per-request C3 gate and the ledger's unknown class catch it
        self.assertFalse(g['c3_per_request']['pass'])
        self.assertEqual(g['c3_per_request']['short_requests'], g['c3_per_request']['append_requests'])
        self.assertFalse(g['unknown_miss_class']['pass'])
        self.assertGreater(g['unknown_miss_class']['value'], 0)
        self.assertTrue(rec['verdict']['shortfalls'])
        self.assertEqual(rec['verdict']['shortfalls'][0]['echo_miss_class'], 'content-mismatch')

    def test_shrunk_rerender_fails_c3_even_when_the_prompt_shrinks(self):
        # mutation: the previous reply comes back without its reasoning; the new prompt is SHORTER
        # than previous prompt + completion, so a target clamped to the new prompt would hide it
        prev = {'prompt_tokens': 1000, 'completion_tokens': 200}
        resp = {'usage': {'prompt_tokens': 1100, 'completion_tokens': 10,
                          'prompt_tokens_details': {'cached_tokens': 1099}},
                'timings': {}, 'ttft_ms': 1.0, 'wall_ms': 2.0, 'finish_reason': 'stop',
                'content': 'x', 'reasoning': '', 'tool_calls': []}
        row = cl._turn_row(1, 'chat', 2, 0, [{'role': 'user', 'content': 'u'}], resp, prev, 'c3')
        self.assertEqual(row['eligible_tokens'], 1199)
        self.assertEqual(row['eligible_hit_tokens'], 1099)
        v = cl.verdict([row], None, 0, False, 1000, 'complete')
        self.assertFalse(v['gates']['c3_per_request']['pass'])
        self.assertEqual(v['shortfalls'][0]['short'], 100)

    def test_lost_reasoning_end_to_end_fails_c3(self):
        with cl.FakeServer(ledger_dir=self.ledger, shrink_rerender=True, reasoning_chars=4000) as srv:
            code, rec, _, _ = self.live(srv, '--thinking', '--no-compaction')
        self.assertEqual(code, 1)
        g = rec['verdict']['gates']['c3_per_request']
        self.assertFalse(g['pass'])
        self.assertEqual(g['short_requests'], g['append_requests'])

    def test_invalid_tool_json_fails_and_is_answered_with_an_error(self):
        with cl.FakeServer(ledger_dir=self.ledger, bad_json_every=2) as srv:
            code, rec, _, _ = self.live(srv, '--no-compaction')
            bodies = srv.requests
        self.assertEqual(code, 1)
        g = rec['verdict']['gates']['tool_json_valid']
        self.assertFalse(g['pass'])
        self.assertLess(g['valid'], g['calls'])
        self.assertTrue(any('not a valid JSON object' in (m.get('content') or '')
                            for b in bodies for m in b['messages'] if m['role'] == 'tool'))

    def test_without_ledger_the_unknown_gate_cannot_pass(self):
        with cl.FakeServer() as srv:
            receipt = os.path.join(self.tmp.name, 'n.json')
            code, _, _ = _run(['--model', 'fake', '--base-url', srv.url, '--receipt', receipt,
                               '--target-context', '8K', '--no-compaction'])
        with open(receipt) as f:
            rec = json.load(f)
        self.assertEqual(code, 1)
        self.assertFalse(rec['verdict']['gates']['unknown_miss_class']['ledger_read'])
        self.assertTrue(rec['verdict']['gates']['hit_ratio_eligible']['pass'])

    def test_deterministic_requests_and_no_text_in_receipt(self):
        bodies = []
        for k in range(2):
            with cl.FakeServer(ledger_dir=os.path.join(self.tmp.name, 'l{}'.format(k))) as srv:
                receipt = os.path.join(self.tmp.name, 'd{}.json'.format(k))
                _run(['--model', 'fake', '--base-url', srv.url, '--receipt', receipt, '--seed', '5',
                      '--target-context', '8K', '--no-compaction'])
                bodies.append(srv.requests)
        self.assertEqual(bodies[0], bodies[1])
        text = Path(receipt).read_text()
        user = bodies[0][0]['messages'][1]['content']
        self.assertNotIn(user[:60], text)
        self.assertNotIn(cl.ls.SYSTEM_PROMPT[:60], text)

    def test_dry_run_needs_no_endpoint_and_loopback_only(self):
        code, out, _ = _run(['--model', 'fake', '--dry-run', '--target-context', '8K'])
        self.assertEqual(code, 0, out)
        self.assertTrue(json.loads(out)['dry_run'])
        code, _, err = _run(['--model', 'm', '--base-url', 'http://localhost:8092/v1', '--receipt',
                             os.path.join(self.tmp.name, 'x.json')])
        self.assertEqual(code, 2)
        self.assertIn('127.0.0.1', err)

    def test_split_tail_keeps_whole_user_turns(self):
        h = [{'role': 'system', 'content': 's'}]
        for i in range(6):
            h += [{'role': 'user', 'content': 'u' * 400}, {'role': 'assistant', 'content': 'a' * 400}]
        cut = cl.split_tail(h, 450)
        self.assertEqual(h[cut]['role'], 'user')
        self.assertEqual(len(h) - cut, 4)
        self.assertEqual(cl.split_tail(h, 1), len(h))


class ThinkingMetricsTest(unittest.TestCase):
    """O86: numbers only, computed from the engine's tokenizer endpoints, preceding context only."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def live(self, srv, *extra):
        receipt = os.path.join(self.tmp.name, 'o86.json')
        code, _, _ = _run(['--model', 'fake', '--base-url', srv.url, '--receipt', receipt,
                           '--target-context', '16K', '--thinking', '--ledger-dir',
                           os.path.join(self.tmp.name, 'ledger'), *extra])
        return code, json.loads(Path(receipt).read_text()), receipt

    def test_ngram_copy_share_is_exact_and_bounded(self):
        ctx = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        same = cl.ngram_copy_shares(list(ctx), ctx)
        self.assertEqual({same['copy_share_n{}'.format(n)] for n in cl.O86_NS}, {1.0})
        disjoint = cl.ngram_copy_shares([100 + i for i in range(20)], ctx)
        self.assertEqual({disjoint['copy_share_n{}'.format(n)] for n in cl.O86_NS}, {0.0})
        short = cl.ngram_copy_shares([1, 2], ctx)
        self.assertIsNone(short['copy_share_n3'])
        self.assertEqual(short['copy_windows_n3'], 0)
        # only the first of the three 4-gram windows occurs in the context
        mixed = cl.ngram_copy_shares([1, 2, 3, 4, 50, 51], ctx)
        self.assertEqual((mixed['copy_hits_n4'], mixed['copy_windows_n4']), (1, 3))

    def test_context_is_the_request_prompt_never_the_current_reply(self):
        with cl.FakeServer(ledger_dir=os.path.join(self.tmp.name, 'ledger')) as srv:
            code, rec, _ = self.live(srv)
            chat = srv.requests
            helpers = srv.helper_requests
        self.assertEqual(code, 0, json.dumps(rec['verdict']['gates']))
        templates = [h['body'] for h in helpers if h['path'] == '/apply-template']
        self.assertEqual(len(templates), sum(1 for r in rec['requests'] if r.get('thinking', {}).get('status') == 'ok'))
        # each rendered context is exactly a request body the client sent (the reply is never in it)
        for t in templates:
            self.assertTrue(any(b['messages'] == t['messages'] for b in chat))
        ok = [r['thinking'] for r in rec['requests'] if r.get('thinking', {}).get('status') == 'ok']
        self.assertTrue(ok)
        for m in ok:
            self.assertGreater(m['reasoning_tokens'], 0)
            for n in cl.O86_NS:
                share = m['copy_share_n{}'.format(n)]
                self.assertTrue(0.0 <= share <= 1.0)
        self.assertEqual(rec['thinking_summary']['measured_turns'], len(ok))

    def test_metrics_use_echo_tokens_instead_of_raw_render_or_current_reply(self):
        body = {'messages': [{'role': 'user', 'content': 'synthetic context'}]}
        with patch.object(cl, 'helper_post', side_effect=[
            {'prompt': 'raw render', 'tokens': [9, 9, 9, 9], 'echo_tokens': [1, 2, 3, 4]},
            {'tokens': [1, 2, 3, 4]},
        ]) as helper:
            metrics = cl.thinking_metrics('127.0.0.1', 1, body, 'current reasoning')
        self.assertEqual(metrics['copy_share_n4'], 1.0)
        self.assertEqual(helper.call_count, 2)
        template = helper.call_args_list[0].args[3]
        self.assertTrue(template['echo'])
        self.assertTrue(template['return_tokens'])
        self.assertEqual(template['messages'], body['messages'])
        with patch.object(cl, 'helper_post', return_value={'prompt': 'raw only', 'tokens': [1, 2, 3]}):
            self.assertEqual(cl.thinking_metrics('127.0.0.1', 1, body, 'reasoning')['status'], 'malformed')

    def test_only_numbers_are_stored(self):
        with cl.FakeServer(ledger_dir=os.path.join(self.tmp.name, 'ledger')) as srv:
            code, rec, path = self.live(srv)
        text = Path(path).read_text()
        self.assertNotIn('Considering the next step', text)
        self.assertNotIn('<|assistant|>', text)
        self.assertNotIn('Done with step', text)
        def leaves(v):
            if isinstance(v, dict):
                for x in v.values():
                    yield from leaves(x)
            elif isinstance(v, list):
                for x in v:
                    yield from leaves(x)
            else:
                yield v
        thinking = [r['thinking'] for r in rec['requests'] if 'thinking' in r]
        self.assertTrue(thinking)
        for t in thinking:
            self.assertIn(t['status'], ('ok', 'no_reasoning'))
            for v in leaves(t):
                self.assertTrue(v is None or isinstance(v, (int, float, str)))
            self.assertTrue(all(isinstance(v, (int, float, type(None))) for k, v in t.items() if k != 'status'))

    def test_missing_helper_endpoint_is_recorded_and_does_not_fail_the_run(self):
        with cl.FakeServer(ledger_dir=os.path.join(self.tmp.name, 'ledger'), no_tokenize=True) as srv:
            code, rec, _ = self.live(srv)
        self.assertEqual(code, 0, json.dumps(rec['verdict']['gates']))
        statuses = {r['thinking']['status'] for r in rec['requests'] if 'thinking' in r}
        self.assertEqual(statuses, {'http_404'})
        self.assertEqual(rec['thinking_summary']['measured_turns'], 0)

    def test_metrics_off_sends_no_helper_requests(self):
        with cl.FakeServer(ledger_dir=os.path.join(self.tmp.name, 'ledger')) as srv:
            code, rec, _ = self.live(srv, '--no-thinking-metrics')
            helpers = srv.helper_requests
        self.assertEqual(code, 0)
        self.assertEqual(helpers, [])
        self.assertFalse(any('thinking' in r for r in rec['requests']))
        self.assertFalse(rec['params']['thinking_metrics'])


if __name__ == '__main__':
    unittest.main()
