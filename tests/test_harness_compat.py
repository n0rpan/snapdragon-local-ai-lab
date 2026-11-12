"""Request compatibility and the harness prefix registry (no model, no backend)."""
import copy
import json
import os
import stat
import tempfile
import unittest
from unittest.mock import patch

from snapdragon_ai_lab.runtime import controller, harness_compat as compat, oxe_runtime
from snapdragon_ai_lab.runtime.harness_registry import (
    HarnessRegistry, engine_identity, prefix_key, timings_from_tail)

CHAT = '/v1/chat/completions'
TOOLS = [{'type': 'function', 'function': {'name': 'bash', 'parameters': {'type': 'object'}}},
         {'type': 'function', 'function': {'name': 'read', 'parameters': {'type': 'object'}}}]


def body(**extra):
    return {'model': 'ornith-35b-a3b-local',
            'messages': [{'role': 'system', 'content': 'You are a coding assistant running inside pi, a harness.'},
                         {'role': 'user', 'content': 'hello'}],
            'tools': copy.deepcopy(TOOLS), **extra}


def profile(**extra):
    return {'backend_kind': 'oxe', 'model_alias': 'ornith-35b-a3b-local', 'sha256': 'a' * 64,
            'server_sha256': 'b' * 64, 'context_size': 32768, **extra}


class NormalizeTests(unittest.TestCase):
    def test_ordinary_request_is_returned_untouched(self):
        payload = body(temperature=0, stream=True)
        result, notes = compat.normalize(payload, CHAT)
        self.assertIs(result, payload)
        self.assertEqual(notes, [])

    def test_null_temperature_becomes_the_greedy_default(self):
        payload = body(temperature=None, cache_prompt=None)
        before = copy.deepcopy(payload)
        result, notes = compat.normalize(payload, CHAT)
        self.assertEqual(payload, before, 'input is not modified')
        self.assertNotIn('temperature', result)
        self.assertEqual(sorted(notes), ['null_cache_prompt_dropped', 'null_temperature_dropped'])
        final = oxe_runtime.request_defaults(result, CHAT)
        self.assertEqual((final['temperature'], final['cache_prompt']), (0, True))

    def test_request_defaults_alone_also_replaces_nulls(self):
        final = oxe_runtime.request_defaults({'temperature': None, 'cache_prompt': None, 'messages': []}, CHAT)
        self.assertEqual((final['temperature'], final['cache_prompt']), (0, True))

    def test_preserve_thinking_is_always_on_and_client_wins(self):
        # CACHE-PERFECT P1 (C1): on for every chat request, profile opt-in no longer needed
        on = {'request_defaults': {'preserve_thinking': True}}
        for profile in (None, {}, {'request_defaults': {'preserve_thinking': 'yes'}}, on):
            self.assertEqual(oxe_runtime.request_defaults(body(), CHAT, profile)['chat_template_kwargs'],
                             {'enable_thinking': False, 'preserve_thinking': True})
        client = body(chat_template_kwargs={'preserve_thinking': False})
        self.assertIs(oxe_runtime.request_defaults(client, CHAT, on)['chat_template_kwargs']['preserve_thinking'], False)
        self.assertNotIn('chat_template_kwargs', oxe_runtime.request_defaults({'prompt': 'x'}, '/completion', on))

    def test_explicit_client_temperature_is_kept(self):
        result, notes = compat.normalize(body(temperature=0.7), CHAT)
        self.assertEqual(oxe_runtime.request_defaults(result, CHAT)['temperature'], 0.7)
        self.assertEqual(notes, [])

    def test_late_system_message_becomes_a_user_reminder(self):
        payload = body()
        payload['messages'] += [{'role': 'system', 'content': 'runtime: plan mode'},
                                {'role': 'developer', 'content': [{'type': 'text', 'text': 'be brief'}]}]
        result, notes = compat.normalize(payload, CHAT)
        self.assertEqual([m['role'] for m in result['messages']], ['system', 'user', 'user', 'user'])
        self.assertEqual(result['messages'][2]['content'], '<system-reminder>\nruntime: plan mode\n</system-reminder>')
        self.assertEqual(result['messages'][3]['content'], '<system-reminder>\nbe brief\n</system-reminder>')
        self.assertEqual(result['messages'][0], payload['messages'][0], 'leading system prompt is untouched')
        self.assertEqual(notes.count('late_system_message_to_user'), 2)

    def test_several_leading_system_messages_stay(self):
        payload = body()
        payload['messages'].insert(1, {'role': 'developer', 'content': 'second'})
        result, notes = compat.normalize(payload, CHAT)
        self.assertIs(result, payload)

    def test_rewrite_is_stable_when_the_conversation_grows(self):
        """Turn N+1 must render turn N's messages identically (prefix reuse)."""
        first = body()
        first['messages'] += [{'role': 'system', 'content': 'note'}, {'role': 'user', 'content': 'q'}]
        second = copy.deepcopy(first)
        second['messages'] += [{'role': 'assistant', 'content': 'a'}, {'role': 'user', 'content': 'q2'}]
        a, _ = compat.normalize(first, CHAT)
        b, _ = compat.normalize(second, CHAT)
        self.assertEqual(b['messages'][:len(a['messages'])], a['messages'])

    def test_named_tool_choice_keeps_only_that_tool(self):
        payload = body(tool_choice={'type': 'function', 'function': {'name': 'read'}})
        result, notes = compat.normalize(payload, CHAT)
        self.assertEqual(result['tool_choice'], 'required')
        self.assertEqual([t['function']['name'] for t in result['tools']], ['read'])
        self.assertEqual(len(payload['tools']), 2)
        self.assertEqual(notes, ['named_tool_choice_to_required'])

    def test_unknown_named_tool_choice_falls_back_to_auto(self):
        result, notes = compat.normalize(body(tool_choice={'type': 'function', 'function': {'name': 'nope'}}), CHAT)
        self.assertEqual(result['tool_choice'], 'auto')
        self.assertEqual(len(result['tools']), 2)
        self.assertEqual(notes, ['unknown_tool_choice_to_auto'])

    def test_string_tool_choice_is_untouched(self):
        payload = body(tool_choice='none')
        self.assertIs(compat.normalize(payload, CHAT)[0], payload)

    def test_image_part_is_replaced_by_a_text_note(self):
        payload = body()
        payload['messages'][1]['content'] = [{'type': 'text', 'text': 'see'},
                                             {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,AAAA'}}]
        result, notes = compat.normalize(payload, CHAT)
        parts = result['messages'][1]['content']
        self.assertEqual(parts[0], {'type': 'text', 'text': 'see'})
        self.assertEqual(parts[1], {'type': 'text', 'text': '[image_url omitted: this local model reads text only]'})
        self.assertNotIn('AAAA', json.dumps(result))
        self.assertEqual(notes, ['non_text_part_replaced'])

    def test_completion_path_only_gets_the_null_fix(self):
        payload = {'prompt': 'x', 'temperature': None, 'tool_choice': {'function': {'name': 'a'}}}
        result, notes = compat.normalize(payload, '/completion')
        self.assertEqual(notes, ['null_temperature_dropped'])
        self.assertIn('tool_choice', result)

    def test_non_dict_payload_passes_through(self):
        self.assertEqual(compat.normalize([1], CHAT), ([1], []))


class IdentifyTests(unittest.TestCase):
    def test_by_system_prompt(self):
        self.assertEqual(compat.identify(body()), 'pi')
        other = body()
        other['messages'][0]['content'] = 'You are an AI agent running in OpenCode, ...'
        self.assertEqual(compat.identify(other), 'opencode')

    def test_dsh_by_system_prompt(self):
        payload = body()
        payload['messages'][0]['content'] = 'You are an AI agent powered by DeepSeek Harness.\n\nYou are a coding agent'
        self.assertEqual(compat.identify(payload), 'dsh')

    def test_user_agent_is_only_a_fallback(self):
        title = {'messages': [{'role': 'system', 'content': 'You name conversations.'}, {'role': 'user', 'content': 'x'}]}
        self.assertEqual(compat.identify(title, 'opencode/2.2.6 ai-sdk'), 'opencode')
        self.assertEqual(compat.identify(body(), 'opencode/2.2.6'), 'pi', 'a known system prompt wins')
        self.assertEqual(compat.identify(title, 'some-api/1.0'), 'other')

    def test_unknown_and_malformed(self):
        self.assertEqual(compat.identify({'messages': [{'role': 'user', 'content': 'x'}]}), 'other')
        self.assertEqual(compat.identify(None), 'other')
        self.assertEqual(compat.identify({'messages': 'x'}, 5), 'other')


class RegistryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.now = [1000.0]
        self.dir = os.path.join(self.tmp.name, 'harness')

    def registry(self, **kw):
        return HarnessRegistry(self.dir, clock=lambda: self.now[0], **kw)

    def test_key_covers_system_tools_and_template_only(self):
        a = body()
        b = body()
        b['messages'][1]['content'] = 'another question'
        b['temperature'] = 0
        self.assertEqual(prefix_key(a), prefix_key(b))
        c = body()
        c['tools'] = c['tools'][::-1]
        self.assertNotEqual(prefix_key(a), prefix_key(c), 'tool order changes the rendered prompt')
        d = body(chat_template_kwargs={'enable_thinking': False})
        self.assertNotEqual(prefix_key(a), prefix_key(d))
        self.assertIsNone(prefix_key({'messages': [{'role': 'user', 'content': 'x'}]}))

    def test_observe_complete_and_warm_state(self):
        reg = self.registry()
        key = reg.observe(body(), 'pi', profile())
        self.assertFalse(reg.status(profile())['pi']['prefixes'][0]['warm'])
        reg.complete(key, profile(), 1280, 249)
        row = reg.status(profile())['pi']
        self.assertEqual(row['warm'], 1)
        self.assertEqual((row['prefixes'][0]['last_cache_n'], row['prefixes'][0]['last_prompt_n']), (1280, 249))
        self.assertEqual(row['prefixes'][0]['tools'], 2)
        # a new engine binary or model invalidates the stored prefix
        self.assertEqual(reg.status(profile(server_sha256='c' * 64))['pi']['warm'], 0)
        self.assertEqual(reg.status(profile(context_size=16384))['pi']['warm'], 0)
        reg.cleared()
        self.assertEqual(reg.status(profile())['pi']['warm'], 0)

    def test_persists_without_user_turns_and_with_private_modes(self):
        reg = self.registry()
        payload = body()
        payload['messages'][1]['content'] = 'SECRET-USER-TEXT'
        key = reg.observe(payload, 'pi', profile())
        reg.complete(key, profile(), 0, 1500)
        again = self.registry()
        self.assertEqual(again.status(profile())['pi']['warm'], 1)
        for root, _dirs, files in os.walk(self.dir):
            self.assertEqual(stat.S_IMODE(os.stat(root).st_mode), 0o700)
            for name in files:
                path = os.path.join(root, name)
                self.assertEqual(stat.S_IMODE(os.stat(path).st_mode), 0o600)
                with open(path) as stream:
                    self.assertNotIn('SECRET-USER-TEXT', stream.read())
        with open(os.path.join(self.dir, 'index.json')) as stream:
            index = json.load(stream)
        self.assertNotIn('You are a coding assistant', json.dumps(index), 'the index holds no prompt text')

    def test_primer_warms_exactly_the_recorded_prefix(self):
        reg = self.registry()
        payload = body(chat_template_kwargs={'enable_thinking': False})
        key = reg.observe(payload, 'pi', profile())
        (got_key, primer), = reg.primers('pi', 'ornith-35b-a3b-local')
        self.assertEqual(got_key, key)
        self.assertEqual(prefix_key(primer), key)
        self.assertEqual(primer['messages'][0], payload['messages'][0])
        self.assertEqual(primer['tools'], payload['tools'])
        self.assertEqual((primer['max_tokens'], primer['temperature'], primer['stream'], primer['tool_choice']),
                         (1, 0, False, 'none'))
        self.assertEqual(len(primer['messages']), 2)
        self.assertEqual(reg.primers('dsh', 'x'), [])

    def test_damaged_primer_is_not_used(self):
        reg = self.registry()
        key = reg.observe(body(), 'pi', profile())
        path = os.path.join(self.dir, 'primers', key + '.json')
        with open(path) as stream:
            primer = json.load(stream)
        primer['messages'][0]['content'] = 'tampered'
        with open(path, 'w') as stream:
            json.dump(primer, stream)
        self.assertEqual(reg.primers('pi', 'm'), [])

    def test_lru_bound_and_forget(self):
        reg = self.registry(max_entries=2)
        keys = []
        for i in range(3):
            self.now[0] += 1
            payload = body()
            payload['messages'][0]['content'] += ' cwd %d' % i
            keys.append(reg.observe(payload, 'pi' if i else 'dsh', profile()))
        self.assertEqual(set(reg.entries), set(keys[1:]))
        self.assertFalse(os.path.exists(os.path.join(self.dir, 'primers', keys[0] + '.json')))
        reg.forget('pi')
        self.assertEqual(reg.entries, {})
        self.assertEqual(os.listdir(os.path.join(self.dir, 'primers')), [])

    def test_memory_only_registry_writes_nothing(self):
        reg = HarnessRegistry(None)
        key = reg.observe(body(), 'pi', profile())
        reg.complete(key, profile(), 1, 2)
        self.assertEqual(reg.status(profile())['pi']['warm'], 1)
        self.assertEqual(reg.primers('pi', 'm'), [])

    def test_timings_from_tail(self):
        sse = b'data: {"choices":[],"timings":{"cache_n":5376,"prompt_n": 212,"predicted_n":9}}\n\ndata: [DONE]\n\n'
        self.assertEqual(timings_from_tail(sse), (5376, 212))
        self.assertEqual(timings_from_tail(b'{"usage":{}}'), (None, None))
        self.assertEqual(timings_from_tail(b''), (None, None))

    def test_engine_identity(self):
        self.assertNotEqual(engine_identity(profile()), engine_identity(profile(sha256='d' * 64)))
        self.assertEqual(engine_identity(None), '')


class ControllerPrepareTests(unittest.TestCase):
    def manager(self, directory):
        m = controller.Manager()
        m.stop_event.set()
        m.harness_registry = HarnessRegistry(directory)
        m.begin = lambda **kw: self.events.append(('begin', kw)) or self._enter(m)
        self.addCleanup(m.close)
        return m

    @staticmethod
    def _enter(m):
        with m.mu:
            m.active += 1
            m.inferences += 1

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.events = []

    def test_prepare_sends_one_token_primers_through_the_queue(self):
        m = self.manager(self.tmp.name)
        m.harness_registry.observe(body(), 'pi', profile())
        sent = []

        def backend_json(path, payload):
            sent.append((path, payload))
            return {'timings': {'cache_n': 0, 'prompt_n': 1500}}
        with patch.object(m, 'config', return_value=profile()), patch.object(m, 'backend_json', backend_json):
            result = m.prepare_harness('pi')
        self.assertEqual(result['prepared'][0]['prompt_n'], 1500)
        (path, payload), = sent
        self.assertEqual(path, '/v1/chat/completions')
        self.assertEqual((payload['max_tokens'], payload['temperature'], payload['cache_prompt']), (1, 0, True))
        self.assertEqual(payload['chat_template_kwargs'], {'enable_thinking': False, 'preserve_thinking': True})
        self.assertEqual(self.events[0][1]['inference'], True)
        self.assertEqual((m.inferences, m.active), (0, 0), 'the slot is released')
        self.assertEqual(m.harness_registry.status(profile())['pi']['warm'], 1)

    def test_prepare_releases_the_slot_when_the_backend_fails(self):
        m = self.manager(self.tmp.name)
        m.harness_registry.observe(body(), 'pi', profile())
        with patch.object(m, 'config', return_value=profile()), \
                patch.object(m, 'backend_json', side_effect=ValueError('boom')):
            with self.assertRaises(ValueError):
                m.prepare_harness('pi')
        self.assertEqual((m.inferences, m.active), (0, 0))
        self.assertEqual(m.harness_registry.status(profile())['pi']['warm'], 0)

    def test_prepare_refuses_unknown_harness_and_non_rust_models(self):
        m = self.manager(self.tmp.name)
        with patch.object(m, 'config', return_value=profile()):
            with self.assertRaises(controller.Unavailable):
                m.prepare_harness('pi')
            with self.assertRaises(controller.Unavailable):
                m.prepare_harness(None)
        m.harness_registry.observe(body(), 'pi', profile())
        with patch.object(m, 'config', return_value=profile(backend_kind='llama')):
            with self.assertRaises(controller.Unavailable):
                m.prepare_harness('pi')
        self.assertEqual(self.events, [], 'no model was started')


if __name__ == '__main__':
    unittest.main()
