"""The OpenAI-compatible surface and opt-in harness learning. Model-free.

The stack tests run the real controller handler over a Python fake of oxe-server. The
`Live` class repeats the audit and the learning flow against a real
`oxe-server --engine stub` when OXE_STUB_SERVER (binary) and OXE_STUB_GGUF (any model
file with a tokenizer and chat template) are set; without them it is skipped.
"""
import importlib.util
import json
import os
import shutil
import stat
import tempfile
import unittest
from pathlib import Path

from snapdragon_ai_lab.runtime import openai_api
from snapdragon_ai_lab.runtime.harness_registry import HarnessRegistry, agent_label, learned_id, learning_settings

import openai_api_stack as stack_module
from openai_api_stack import FLASH, ORNITH, QWEN, FakeEngine, Stack

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('openai_api_check', ROOT / 'tools/openai_api_check.py')
check = importlib.util.module_from_spec(spec); spec.loader.exec_module(check)

VALUE = stack_module.runtime_value()


def harness_body(name='Gamma', question='Fix the test.', model='local-default', **extra):
    system, tools = check.synthetic_harness(name, tools=6, system_words=600)
    return {'model': model, 'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': question}],
            'tools': tools, 'max_tokens': 4, **extra}


class Names(unittest.TestCase):
    def test_default_and_heavy_aliases(self):
        for name in (None, '', 'local-default', 'qwen-local', 'openai/local-default'):
            self.assertIsNone(openai_api.resolve_model(name, VALUE))
        self.assertEqual(openai_api.resolve_model('local-heavy', VALUE), FLASH)
        self.assertEqual(openai_api.resolve_model('openai/local-heavy', VALUE), FLASH)
        self.assertEqual(openai_api.resolve_model(QWEN, VALUE), QWEN)
        self.assertEqual(openai_api.resolve_model('openai/' + QWEN, VALUE), QWEN)
        self.assertEqual(openai_api.resolve_model('gpt-4o', VALUE), 'gpt-4o')  # reported as unknown by the caller

    def test_configured_aliases_and_fallback(self):
        value = {**VALUE, 'openai_api': {'model_aliases': {'local-fast': QWEN, 'local-heavy': None, 'bad name': QWEN,
                                                           ORNITH: QWEN}, 'unknown_model': 'default'}}
        self.assertEqual(openai_api.alias_table(value), {'local-fast': QWEN})  # no shadowing of a profile
        self.assertEqual(openai_api.resolve_model('local-fast', value), QWEN)
        self.assertIsNone(openai_api.resolve_model('gpt-4o', value))
        self.assertEqual(openai_api.resolve_model(ORNITH, value), ORNITH)

    def test_model_rows_order_and_hidden_alias(self):
        rows = [{'id': n, 'object': 'model', 'created': 0, 'owned_by': 'local', 'name': n.upper()}
                for n in ('qwen-local', ORNITH, QWEN, FLASH)]
        out = openai_api.model_rows(VALUE, rows, ORNITH)
        self.assertEqual([r['id'] for r in out], ['local-default', 'local-heavy', ORNITH, QWEN, FLASH, 'qwen-local'])
        self.assertEqual(out[0]['alias_of'], ORNITH)
        self.assertEqual(out[1]['alias_of'], FLASH)
        self.assertTrue(all(r['object'] == 'model' for r in out))
        without_flash = openai_api.model_rows(VALUE, rows[:3], ORNITH)
        self.assertNotIn('local-heavy', [r['id'] for r in without_flash])  # an unqualified target is not offered


class Errors(unittest.TestCase):
    def test_shapes(self):
        self.assertEqual(openai_api.normalize_error(400, {'error': 'Invalid model request.'}),
                         {'error': {'message': 'Invalid model request.', 'type': 'invalid_request_error', 'param': None, 'code': None}})
        kept = openai_api.normalize_error(503, {'error': {'message': 'busy', 'type': 'local_backend_busy'}})
        self.assertEqual(kept['error'], {'message': 'busy', 'type': 'local_backend_busy', 'param': None, 'code': None})
        engine = openai_api.normalize_error(400, {'error': {'code': 400, 'message': 'm', 'type': 't', 'n_ctx': 5}})
        self.assertEqual((engine['error']['code'], engine['error']['n_ctx']), (400, 5))
        self.assertEqual(openai_api.normalize_error(200, {'error': 'x'}), {'error': 'x'})
        self.assertEqual(openai_api.normalize_error(404, {'error': 'Unknown local model.'})['error']['code'], 'model_not_found')


class KeyAndCors(unittest.TestCase):
    def test_key_sources_and_check(self):
        self.assertIsNone(openai_api.api_key({}, {}))
        self.assertEqual(openai_api.api_key({}, {'LOCAL_AI_API_KEY': ' k1 '}), 'k1')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'key'
            path.write_text('k2\n')
            value = {'openai_api': {'api_key_file': str(path)}}
            self.assertEqual(openai_api.api_key(value, {}), 'k2')
            path.unlink()
            closed = openai_api.api_key(value, {})  # a configured but missing key file never opens the API
            self.assertTrue(closed and closed != 'k2')
        self.assertTrue(openai_api.authorized({'Authorization': 'Bearer k1'}, 'k1'))
        self.assertTrue(openai_api.authorized({'x-api-key': 'k1'}, 'k1'))
        self.assertFalse(openai_api.authorized({'Authorization': 'Bearer k2'}, 'k1'))
        self.assertFalse(openai_api.authorized({}, 'k1'))

    def test_keyed_routes(self):
        for method, path in (('POST', '/v1/chat/completions'), ('GET', '/v1/models'), ('POST', '/completion'),
                             ('POST', '/tokenize'), ('GET', '/props'), ('GET', '/v1/models/x.js'),
                             ('POST', '/index.js')):
            self.assertTrue(openai_api.key_required(method, path), path)
        for method, path in (('GET', '/'), ('GET', '/index.js'), ('OPTIONS', '/v1/chat/completions')):
            self.assertTrue(openai_api.key_required(method, path), path)
        for method, path in (('GET', '/_local/status'), ('POST', '/_local/unload'), ('GET', '/health')):
            self.assertFalse(openai_api.key_required(method, path), path)
        for bad in ('~/key', 'relative/key', 5):  # configured but unusable: closed, never open
            self.assertTrue(openai_api.api_key({'openai_api': {'api_key_file': bad}}, {}))

    def test_cors_is_opt_in(self):
        value = {'openai_api': {'cors_origins': ['http://localhost:3000']}}
        self.assertIsNone(openai_api.cors_origin(['http://x.test'], {}, False))
        self.assertEqual(openai_api.cors_origin(['http://localhost:3000'], value, False), 'http://localhost:3000')
        self.assertIsNone(openai_api.cors_origin(['http://x.test'], value, False))
        self.assertIsNone(openai_api.cors_origin(['http://localhost:3000', 'http://x.test'], value, False))
        star = {'openai_api': {'cors_origins': ['*']}}
        self.assertIsNone(openai_api.cors_origin(['http://x.test'], star, False))  # "*" needs an API key
        self.assertEqual(openai_api.cors_origin(['http://x.test'], star, True), 'http://x.test')


class RequestFixes(unittest.TestCase):
    def test_flash_range(self):
        notes = []
        out = openai_api.flash_request({'messages': [], 'max_completion_tokens': 9000, 'temperature': 0.7,
                                        'reasoning_effort': 'high', 'n': 3}, notes)
        self.assertEqual(out, {'messages': [], 'max_tokens': 2048, 'temperature': 0})
        self.assertEqual(sorted(notes), ['flash_max_completion_tokens_renamed', 'flash_max_tokens_capped',
                                         'flash_reasoning_effort_dropped', 'flash_temperature_forced_0', 'n_reduced_to_1'])
        notes = []
        body = {'messages': [], 'max_tokens': 512, 'temperature': 0}
        self.assertEqual(openai_api.flash_request(body, notes), body)  # a smaller limit is kept
        self.assertEqual(notes, [])

    def test_completion_translation(self):
        native, meta = openai_api.completion_request({'model': ORNITH, 'prompt': ['Hello'], 'max_tokens': 5, 'logprobs': 3,
                                                      'echo': True, 'stop': ['\n'], 'temperature': 0, 'n': 1, 'user': 'u'})
        self.assertEqual(native, {'model': ORNITH, 'stop': ['\n'], 'temperature': 0, 'prompt': 'Hello', 'n_predict': 5, 'n_probs': 3})
        reply = openai_api.completion_response({'content': ' world', 'stop_type': 'eos', 'tokens_predicted': 2,
                                                'tokens_evaluated': 3, 'model': ORNITH, 'timings': {'cache_n': 1, 'prompt_n': 2},
                                                'completion_probabilities': [{'token': ' world', 'logprob': -0.1, 'top_logprobs': [
                                                    {'token': ' world', 'logprob': -0.1}]}]}, meta)
        self.assertEqual(reply['object'], 'text_completion')
        self.assertEqual(reply['choices'][0]['text'], 'Hello world')
        self.assertEqual(reply['choices'][0]['finish_reason'], 'stop')
        self.assertEqual(reply['choices'][0]['logprobs'], {'tokens': [' world'], 'token_logprobs': [-0.1],
                                                           'top_logprobs': [{' world': -0.1}], 'text_offset': [5]})
        self.assertEqual(reply['usage'], {'prompt_tokens': 3, 'completion_tokens': 2, 'total_tokens': 5,
                                          'prompt_tokens_details': {'cached_tokens': 1}})
        for bad in ({'prompt': ['a', 'b']}, {'prompt': 'a', 'suffix': 'b'}, {'prompt': 5}, []):
            with self.assertRaises(openai_api.Unsupported):
                openai_api.completion_request(bad)
        error = {'error': {'message': 'x', 'type': 't'}}
        self.assertIs(openai_api.completion_response(error, meta), error)

    def test_completion_stream_split_frames(self):
        _, meta = openai_api.completion_request({'model': ORNITH, 'prompt': 'a', 'stream': True,
                                                 'stream_options': {'include_usage': True}})
        stream = openai_api.CompletionStream(meta)
        raw = (b'data: {"content":"Hi","stop":false}\n\n'
               b'data: {"content":"","stop":true,"stop_type":"limit","tokens_predicted":1,"tokens_evaluated":2,"timings":{"cache_n":0}}\n\n')
        out = b''.join(stream.feed(raw[i:i + 7]) for i in range(0, len(raw), 7)) + stream.finish()
        events = check.sse_events(out.decode())
        self.assertEqual(events[-1], '[DONE]')
        self.assertEqual([e['choices'][0]['text'] for e in events[:2]], ['Hi', ''])
        self.assertEqual(events[1]['choices'][0]['finish_reason'], 'length')
        self.assertEqual((events[2]['choices'], events[2]['usage']['total_tokens']), ([], 3))
        self.assertTrue(stream.done)
        odd = openai_api.CompletionStream(meta)  # malformed engine data passes through, never raises
        raw = b'data: {"content":null,"stop":true,"tokens_predicted":"x","timings":5}\n\ndata: {"error":{"message":"m"}}\n\ndata: [1]\n\n'
        out = odd.feed(raw) + odd.finish()
        self.assertIn(b'"error"', out)
        self.assertIn(b'[DONE]', out)


class Learning(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name) / 'state'
        self.registry = HarnessRegistry(self.dir)
        self.config = {'model_alias': ORNITH, 'sha256': 'a', 'server_sha256': 'b', 'context_size': 1}
        self.on = learning_settings({'harness_learning': {'enabled': True}})

    def primers(self):
        return sorted((self.dir / 'primers').glob('*.json')) if (self.dir / 'primers').is_dir() else []

    def test_off_by_default_keeps_nothing_on_disk(self):
        self.assertEqual(learning_settings({}), {'enabled': False, 'min_requests': 3, 'min_prefix_chars': 3000})
        for _ in range(10):
            key = self.registry.observe(harness_body(), 'other', self.config, learning=learning_settings({}),
                                        user_agent='Aider/0.99 python')
        self.registry.complete(key, self.config, 900, 40)
        self.assertEqual(self.registry.entries, {})
        self.assertEqual(self.primers(), [])
        self.assertFalse((self.dir / 'index.json').exists())
        row = self.registry.learning_status()['candidates'][0]
        self.assertEqual((row['requests'], row['consistent_requests'], row['stable'], row['agent']), (10, 10, True, 'Aider'))
        self.assertEqual((row['last_cache_n'], row['last_prompt_n']), (900, 40))
        self.assertNotIn('Gamma', json.dumps(self.registry.learning_status()))  # numbers only, no prompt text

    def test_learned_after_n_identical_requests(self):
        body = harness_body()
        name = learned_id(body['messages'][:1], body['tools'])
        for count in (1, 2):
            self.registry.observe(harness_body(question='q%d' % count), 'other', self.config, learning=self.on, user_agent='Zed/0.2')
            self.assertEqual(self.registry.entries, {}, count)
        key = self.registry.observe(harness_body(question='q3'), 'other', self.config, learning=self.on, user_agent='Zed/0.2')
        entry = self.registry.entries[key]
        self.assertEqual((entry['harness'], entry['requests'], entry['learned']), (name, 3, True))
        self.assertEqual(self.registry.candidates, {})
        status = self.registry.status(self.config)[name]
        self.assertEqual((status['learned'], status['label']), (True, 'Zed'))
        # the primer: system prompt and tools only, private file
        primer_path = self.primers()[0]
        primer = json.loads(primer_path.read_text())
        self.assertEqual([m['role'] for m in primer['messages']], ['system', 'user'])
        self.assertNotIn('q3', primer_path.read_text())
        self.assertEqual(primer['tools'], body['tools'])
        self.assertEqual(stat.S_IMODE(primer_path.stat().st_mode), 0o600)
        self.assertEqual(stat.S_IMODE((self.dir / 'index.json').stat().st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(self.dir.stat().st_mode), 0o700)
        self.assertEqual(len(self.registry.primers(name, ORNITH)), 1)
        # survives a controller restart, keeps its name with learning switched off again
        again = HarnessRegistry(self.dir)
        key2 = again.observe(harness_body(question='later'), 'other', self.config, learning=learning_settings({}))
        self.assertEqual((key2, again.entries[key]['requests'], again.entries[key]['harness']), (key, 4, name))

    def test_changing_prompt_is_never_learned_and_small_prompts_are_skipped(self):
        for i in range(6):
            body = harness_body()
            body['messages'][0]['content'] += '\nNow: 12:0%d' % i
            self.registry.observe(body, 'other', self.config, learning=self.on)
        row = self.registry.learning_status(self.on)['candidates'][0]
        self.assertEqual((row['distinct_prefixes'], row['consistent_requests'], row['stable']), (6, 1, False))
        self.assertEqual(self.registry.entries, {})
        small = {'messages': [{'role': 'system', 'content': 'You are helpful.'}, {'role': 'user', 'content': 'hi'}]}
        for _ in range(5):
            self.registry.observe(small, 'other', self.config, learning=self.on)
        self.assertEqual(self.registry.entries, {})
        self.assertFalse(self.registry.learning_status(self.on)['candidates'][0]['large_enough'])
        self.assertEqual(self.primers(), [])

    def test_approve_one_client_forget_and_ignore(self):
        off = learning_settings({})
        body = harness_body()
        name = learned_id(body['messages'][:1], body['tools'])
        self.assertFalse(self.registry.approve(name))  # never seen
        self.assertFalse(self.registry.approve('pi'))
        self.registry.observe(body, 'other', self.config, learning=off)
        self.assertTrue(self.registry.approve(name))
        key = self.registry.observe(body, 'other', self.config, learning=off)  # learned at the next request
        self.assertEqual(self.registry.entries[key]['harness'], name)
        self.assertTrue(self.registry.set_label(name, 'My <b>agent</b>'))
        self.assertEqual(self.registry.status()[name]['label'], 'My bagentb')
        self.assertFalse(self.registry.set_label('pi', 'x'))
        self.registry.forget(name, ignore=True)
        self.assertEqual((self.registry.entries, self.primers()), ({}, []))
        for _ in range(5):
            self.registry.observe(body, 'other', self.config, learning=self.on)
        self.assertEqual(self.registry.entries, {})  # ignored: not learned again, even with learning on
        self.assertTrue(self.registry.learning_status()['candidates'][0]['ignored'])
        self.assertTrue(self.registry.approve(name))
        self.registry.observe(body, 'other', self.config, learning=off)
        self.assertEqual(len(self.registry.entries), 1)

    def test_known_harness_unchanged_and_learned_prefix_cap(self):
        key = self.registry.observe(harness_body(), 'pi', self.config)  # O-9: a known harness is stored at once
        self.assertEqual(self.registry.entries[key]['harness'], 'pi')
        self.assertEqual(len(self.primers()), 1)
        clock = [0]
        registry = HarnessRegistry(self.dir / 'b', clock=lambda: clock[0])
        for i in range(7):  # one learned client in seven project directories
            clock[0] += 1
            body = harness_body('Delta')
            body['messages'][0]['content'] += '\ncwd: /work/project-%d' % i
            for _ in range(3):
                registry.observe(body, 'other', self.config, learning=self.on)
        self.assertEqual(len(registry.entries), 4)
        self.assertEqual(len(set(e['harness'] for e in registry.entries.values())), 1)
        self.assertEqual(len(list((self.dir / 'b' / 'primers').glob('*.json'))), 4)

    def test_unidentified_request_with_a_known_prefix_keeps_the_known_row(self):
        key = self.registry.observe(harness_body(), 'pi', self.config)
        for _ in range(4):
            self.assertEqual(self.registry.observe(harness_body(), 'other', self.config, learning=self.on), key)
        self.assertEqual((self.registry.entries[key]['harness'], self.registry.entries[key]['requests']), ('pi', 5))
        self.assertEqual(self.registry.candidates, {})

    def test_old_unknown_entries_are_dropped_on_load(self):
        self.registry.observe(harness_body(), 'pi', self.config)
        index = json.loads((self.dir / 'index.json').read_text())
        key = next(iter(index['entries']))
        index['entries'][key]['harness'] = 'other'
        (self.dir / 'index.json').write_text(json.dumps(index))
        self.assertEqual(HarnessRegistry(self.dir).entries, {})
        self.assertEqual(self.primers(), [])

    def test_labels(self):
        self.assertEqual(agent_label('OpenAI/Python 1.2.3'), 'OpenAI')
        self.assertEqual(agent_label('<script>x</script>/1'), 'scriptx')
        self.assertEqual(agent_label(None), '')


class ControllerStack(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = FakeEngine()
        cls.addClassCleanup(cls.engine.close)

    def stack(self, **config):
        stack = Stack(self.engine.port, **config)
        self.addCleanup(stack.close)
        return stack

    def test_audit_table(self):
        stack = self.stack()
        rows = {(e, f): v for e, f, v, _ in check.audit(stack.base, 'local-default')}
        missing = {k for k, v in rows.items() if v == 'missing'}
        self.assertEqual(missing, {('/v1/chat/completions', 'response_format json_object'),
                                   ('/v1/chat/completions', 'response_format json_schema'),
                                   ('/v1/chat/completions', 'logprobs / top_logprobs'),  # the fake engine has none
                                   ('/v1/completions', 'suffix (insert)'), ('/v1/embeddings', 'any'), ('/v1/responses', 'any'),
                                   ('CORS', 'preflight from another origin')})
        for key in (('/v1/models', 'list'), ('/v1/models/{id}', 'retrieve'), ('/v1/chat/completions', 'non-streaming'),
                    ('/v1/chat/completions', 'streaming (SSE, [DONE])'), ('/v1/chat/completions', 'stream_options.include_usage'),
                    ('/v1/completions', 'non-streaming'), ('/v1/completions', 'streaming (SSE, [DONE])'),
                    ('/v1/completions', 'logprobs'), ('errors', 'unknown model'), ('errors', 'malformed JSON body'),
                    ('auth', 'API key')):
            self.assertEqual(rows[key], 'supported', key)
        self.assertEqual(rows[('/v1/chat/completions', 'n > 1')], 'partial')

    def test_models_and_aliases_reach_the_right_profile(self):
        stack = self.stack()
        status, _, value = check.call(stack.base, 'GET', '/v1/models')
        self.assertEqual([r['id'] for r in value['data']], ['local-default', 'local-heavy', ORNITH, QWEN, FLASH, 'qwen-local'])
        self.assertEqual(check.call(stack.base, 'GET', '/v1/models/local-heavy')[2]['alias_of'], FLASH)
        self.assertEqual(check.call(stack.base, 'GET', '/v1/models/openai/' + QWEN)[2]['id'], QWEN)
        self.assertEqual(check.call(stack.base, 'GET', '/v1/models/gpt-4o')[0], 404)
        self.assertEqual(stack.manager.loads, 0)  # listing never loads a model
        for name, expected in (('local-default', ORNITH), ('openai/local-default', ORNITH), (QWEN, QWEN), ('openai/' + QWEN, QWEN)):
            status, _, value = check.call(stack.base, 'POST', '/v1/chat/completions',
                                          {'model': name, 'messages': [{'role': 'user', 'content': 'hi'}]})
            self.assertEqual((status, value['model'], self.engine.requests[-1]['body']['model']), (200, expected, expected), name)
        self.assertEqual(stack.manager.config('local-heavy')['model_alias'], FLASH)
        stack.set_config(selected_model=QWEN)
        self.assertEqual(check.call(stack.base, 'GET', '/v1/models/local-default')[2]['alias_of'], QWEN)

    def test_unsupported_endpoints_do_not_load_a_model(self):
        stack = self.stack()
        for path in ('/v1/embeddings', '/v1/responses', '/v1/rerank'):
            status, _, value = check.call(stack.base, 'POST', path, {'model': 'local-default', 'input': 'x'})
            self.assertEqual((status, value['error']['code']), (404, 'unsupported_endpoint'), path)
        self.assertEqual(stack.manager.loads, 0)
        status, _, value = check.call(stack.base, 'POST', '/v1/chat/completions', {'model': 'gpt-4o', 'messages': []})
        self.assertEqual((status, value['error']['code'], value['error']['param']), (404, 'model_not_found', 'model'))
        self.assertEqual(stack.manager.loads, 0)

    def test_unknown_model_fallback_is_opt_in(self):
        stack = self.stack(openai_api={'unknown_model': 'default'})
        status, _, value = check.call(stack.base, 'POST', '/v1/chat/completions',
                                      {'model': 'gpt-4o-mini', 'messages': [{'role': 'user', 'content': 'hi'}]})
        self.assertEqual((status, value['model']), (200, ORNITH))

    def test_completions_shim_reaches_native_route(self):
        stack = self.stack()
        status, _, value = check.call(stack.base, 'POST', '/v1/completions',
                                      {'model': 'local-default', 'prompt': 'Hello', 'max_tokens': 2, 'n': 4})
        sent = self.engine.requests[-1]
        self.assertEqual((sent['path'], sent['body']['n_predict'], sent['body']['prompt']), ('/completion', 2, 'Hello'))
        self.assertNotIn('max_tokens', sent['body'])
        self.assertEqual((status, value['choices'][0]['text'], value['choices'][0]['finish_reason']), (200, 'Hello from', 'length'))
        self.assertEqual(stack.manager.compat_stats['n_reduced_to_1'], 1)
        status, _, value = check.call(stack.base, 'POST', '/v1/completions', {'model': 'local-default', 'prompt': ['a', 'b']})
        self.assertEqual((status, value['error']['type']), (400, 'invalid_request_error'))

    def test_api_key(self):
        stack = self.stack(environ={'LOCAL_AI_API_KEY': 'key-for-test'})
        body = {'model': 'local-default', 'messages': [{'role': 'user', 'content': 'hi'}]}
        for key in (None, 'wrong'):
            for method, path, payload in (('GET', '/v1/models', None), ('POST', '/v1/chat/completions', body),
                                          ('POST', '/chat/completions', body), ('POST', '/completion', {'prompt': 'x'})):
                status, _, value = check.call(stack.base, method, path, payload, key=key)
                self.assertEqual((status, value['error']['code']), (401, 'invalid_api_key'), (key, path))
        self.assertEqual(stack.manager.loads, 0)
        self.assertEqual(check.call(stack.base, 'GET', '/v1/models', key='key-for-test')[0], 200)
        self.assertEqual(check.call(stack.base, 'POST', '/v1/chat/completions', body, key='key-for-test')[0], 200)
        self.assertNotIn('Authorization', self.engine.requests[-1]['headers'])  # the key stays in the controller
        self.assertEqual(check.call(stack.base, 'GET', '/_local/status', key=None)[0], 200)
        self.assertEqual(check.call(stack.base, 'GET', '/health', key=None)[0], 200)
        before = len(self.engine.requests)
        for method, path in (('GET', '/'), ('GET', '/x.js'), ('OPTIONS', '/v1/chat/completions'), ('GET', '/_local/nothing'),
                             ('POST', '/_local/nothing')):
            self.assertIn(check.call(stack.base, method, path, {} if method == 'POST' else None, key=None)[0], (401, 404), path)
        self.assertEqual((len(self.engine.requests), stack.manager.loads), (before, 1))  # none reached the engine
        stack.config_path.write_text('{broken')  # an unreadable runtime.json does not drop the key
        self.assertEqual(check.call(stack.base, 'POST', '/completion', {'prompt': 'x'}, key=None)[0], 401)

    def test_cors_only_for_configured_origins(self):
        origin = 'http://localhost:3000'
        stack = self.stack(openai_api={'cors_origins': [origin]})
        preflight = {'Origin': origin, 'Access-Control-Request-Method': 'POST', 'Access-Control-Request-Headers': 'authorization',
                     'Access-Control-Request-Private-Network': 'true'}
        status, headers, _ = check.call(stack.base, 'OPTIONS', '/v1/chat/completions', headers=preflight)
        self.assertEqual((status, headers['Access-Control-Allow-Origin'], headers['Access-Control-Allow-Private-Network']), (204, origin, 'true'))
        self.assertEqual(stack.manager.loads, 0)
        body = {'model': 'local-default', 'messages': [{'role': 'user', 'content': 'hi'}]}
        for extra in ({}, {'stream': True}):
            status, headers, _ = check.call(stack.base, 'POST', '/v1/chat/completions', {**body, **extra},
                                            headers={'Origin': origin, 'Sec-Fetch-Site': 'cross-site'})
            self.assertEqual((status, headers.get('Access-Control-Allow-Origin')), (200, origin))  # not the engine's "*"
        status, headers, _ = check.call(stack.base, 'POST', '/v1/chat/completions', body, headers={'Origin': 'http://evil.test'})
        self.assertEqual((status, headers.get('Access-Control-Allow-Origin')), (403, None))
        # the control routes stay same-origin even for a configured origin
        self.assertEqual(check.call(stack.base, 'POST', '/_local/unload', {}, headers={'Origin': origin})[0], 403)
        status, headers, _ = check.call(stack.base, 'POST', '/v1/chat/completions', body)  # no Origin: no CORS headers
        self.assertEqual((status, headers.get('Access-Control-Allow-Origin')), (200, None))

    def chat(self, stack, agent='Gamma-Agent/1.0', **kw):
        return check.call(stack.base, 'POST', '/v1/chat/completions', harness_body(**kw), headers={'User-Agent': agent})

    def test_learning_flow_through_the_controller(self):
        stack = self.stack()
        primers = stack.state / 'primers'
        for i in range(4):
            self.assertEqual(self.chat(stack, question='q%d' % i)[0], 200)
        status, _, listed = check.call(stack.base, 'GET', '/_local/harnesses')
        self.assertEqual((status, listed['harnesses'], listed['harness_learning']['enabled']), (200, {}, False))
        candidate = listed['harness_learning']['candidates'][0]
        self.assertEqual((candidate['requests'], candidate['agent'], candidate['stable']), (4, 'Gamma-Agent', True))
        self.assertGreater(candidate['last_cache_n'], 0)  # the engine reused the prefix without any learning
        self.assertFalse(primers.exists())
        self.assertFalse((stack.state / 'index.json').exists())
        # opt in (persisted in runtime.json), then N more identical requests
        status, _, value = check.call(stack.base, 'POST', '/_local/harness/learning', {'enabled': True, 'min_requests': 2})
        self.assertEqual((status, value['enabled'], value['min_requests']), (200, True, 2))
        self.assertEqual(json.loads(stack.config_path.read_text())['harness_learning'], {'enabled': True, 'min_requests': 2})
        self.assertEqual(check.call(stack.base, 'POST', '/_local/harness/learning', {'enabled': 'yes'})[0], 400)
        # the setting applies at once, also while a model loaded before the change is still up,
        # and for a request that names no model
        self.assertIsNotNone(stack.manager.loaded_config)
        self.assertTrue(check.call(stack.base, 'GET', '/_local/harnesses')[2]['harness_learning']['enabled'])
        body = harness_body(question='secret user question'); del body['model']
        self.assertEqual(check.call(stack.base, 'POST', '/v1/chat/completions', body, headers={'User-Agent': 'Gamma-Agent/1.0'})[0], 200)
        _, _, status_page = check.call(stack.base, 'GET', '/_local/status')
        name, row = next(iter(status_page['harnesses'].items()))
        self.assertTrue(name.startswith('learned-'))
        self.assertEqual((row['learned'], row['label'], row['warm'], row['prefixes'][0]['requests']), (True, 'Gamma-Agent', 1, 5))
        files = list(primers.glob('*.json'))
        self.assertEqual(len(files), 1)
        self.assertNotIn('secret user question', files[0].read_text())
        self.assertEqual(stat.S_IMODE(files[0].stat().st_mode), 0o600)
        # label, prepare (one primer through the queue), prepare all, forget
        self.assertEqual(check.call(stack.base, 'POST', '/_local/harness/label', {'harness': name, 'label': 'Gamma'})[0], 200)
        before = len(self.engine.requests)
        status, _, prepared = check.call(stack.base, 'POST', '/_local/harness/prepare', {'harness': name})
        self.assertEqual((status, len(prepared['prepared'])), (200, 1))
        primer = self.engine.requests[-1]['body']
        self.assertEqual((len(self.engine.requests) - before, primer['max_tokens'], primer['messages'][-1]['content'][:10]),
                         (1, 1, 'Initialize'))
        stack.manager.harness_registry.cleared()
        status, _, prepared = check.call(stack.base, 'POST', '/_local/harness/prepare', {'harness': '*'})
        self.assertEqual((status, [h['harness'] for h in prepared['harnesses']]), (200, [name]))
        status, _, prepared = check.call(stack.base, 'POST', '/_local/harness/prepare', {'harness': '*'})
        self.assertEqual(prepared['harnesses'], [])  # everything is warm: nothing to do
        status, _, value = check.call(stack.base, 'POST', '/_local/harness/forget', {'harness': name, 'ignore': True})
        self.assertEqual((status, value['ignored']), (200, True))
        self.assertEqual(list(primers.glob('*.json')), [])
        self.chat(stack); self.chat(stack); self.chat(stack)
        self.assertEqual(check.call(stack.base, 'GET', '/_local/harnesses')[2]['harnesses'], {})
        # approve that one client again
        self.assertEqual(check.call(stack.base, 'POST', '/_local/harness/learn', {'harness': name})[0], 200)
        self.assertEqual(check.call(stack.base, 'POST', '/_local/harness/learn', {'harness': 'learned-00000000'})[0], 404)
        self.chat(stack)
        self.assertIn(name, check.call(stack.base, 'GET', '/_local/harnesses')[2]['harnesses'])

    def test_known_harness_is_not_a_candidate(self):
        stack = self.stack()
        body = harness_body()
        body['messages'][0]['content'] = 'You are a coding assistant running inside pi. ' + body['messages'][0]['content']
        check.call(stack.base, 'POST', '/v1/chat/completions', body)
        listed = check.call(stack.base, 'GET', '/_local/harnesses')[2]
        self.assertEqual((list(listed['harnesses']), listed['harness_learning']['candidates']), (['pi'], []))


@unittest.skipUnless(os.environ.get('OXE_STUB_SERVER') and os.environ.get('OXE_STUB_GGUF'),
                     'set OXE_STUB_SERVER and OXE_STUB_GGUF to run against a real oxe-server --engine stub')
class Live(unittest.TestCase):
    def engine(self, store):
        engine = check.StubEngine(os.environ['OXE_STUB_SERVER'], os.environ['OXE_STUB_GGUF'], store).start()
        self.addCleanup(engine.stop)
        return engine

    def test_audit_through_controller_and_stub(self):
        with tempfile.TemporaryDirectory() as store:
            engine = self.engine(store)
            with Stack(engine.port) as stack:
                rows = check.audit(stack.base, 'local-default')
        missing = {(e, f) for e, f, v, _ in rows if v == 'missing'}
        self.assertEqual(missing, {('/v1/chat/completions', 'response_format json_object'),
                                   ('/v1/chat/completions', 'response_format json_schema'),
                                   ('/v1/completions', 'suffix (insert)'), ('/v1/embeddings', 'any'), ('/v1/responses', 'any'),
                                   ('CORS', 'preflight from another origin')})

    def test_prepare_moves_the_cold_prefill_off_the_first_request(self):
        """After an engine update (empty prefix store) a learned harness is prepared, and its
        next first request reuses the prefix; without Prepare that request evaluates it all."""
        def first_request(prepare):
            with tempfile.TemporaryDirectory() as tmp:
                store = Path(tmp) / 'store'
                engine = self.engine(str(store))
                with Stack(engine.port, harness_learning={'enabled': True}) as stack:
                    for i in range(3):
                        check.call(stack.base, 'POST', '/v1/chat/completions', big_body('q%d' % i))
                    name = next(iter(check.call(stack.base, 'GET', '/_local/harnesses')[2]['harnesses']))
                    engine.stop(); shutil.rmtree(store); engine.start()   # an engine update: the store is empty
                    stack.manager.harness_registry.cleared()
                    spent = None
                    if prepare:
                        spent = check.call(stack.base, 'POST', '/_local/harness/prepare', {'harness': name})[2]['prepared'][0]
                    value = check.call(stack.base, 'POST', '/v1/chat/completions', big_body('a new session'))[2]
                    return spent, value['timings']['cache_n'], value['timings']['prompt_n']

        def big_body(question):
            system, tools = check.synthetic_harness('Alpha')
            return {'model': 'local-default', 'max_tokens': 4, 'tools': tools,
                    'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': question}]}

        _, cold_cache, cold_prompt = first_request(prepare=False)
        spent, warm_cache, warm_prompt = first_request(prepare=True)
        self.assertEqual(cold_cache, 0)
        self.assertGreater(cold_prompt, 3000)
        self.assertGreater(spent['prompt_n'], 3000)      # Prepare paid the prefill
        self.assertGreater(warm_cache, 3000)             # so the first real request did not
        self.assertLess(warm_prompt, 100)
        print('\nfirst request after an engine update: without Prepare reused %d / evaluated %d; '
              'with Prepare reused %d / evaluated %d (Prepare itself evaluated %d)'
              % (cold_cache, cold_prompt, warm_cache, warm_prompt, spent['prompt_n']))


if __name__ == '__main__':
    unittest.main()
