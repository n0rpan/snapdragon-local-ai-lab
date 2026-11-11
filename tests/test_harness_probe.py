"""Tests for tools/harness_probe.py. Only local mock servers on 127.0.0.1 ephemeral ports."""
import contextlib
import hashlib
import http.client
import importlib.util
import io
import json
import shutil
import sys
import tempfile
import threading
import time
import unittest
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = ROOT / 'tools'
sys.path.insert(0, str(TOOLS_DIR))
_spec = importlib.util.spec_from_file_location('harness_probe', TOOLS_DIR / 'harness_probe.py')
hp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hp)

CHAT = '/v1/chat/completions'
SYS = 'You are the synthetic probe agent. ' * 3
USER = 'list the files'
BASH = {'type': 'function', 'function': {'name': 'bash', 'description': 'run a command',
                                          'parameters': {'type': 'object',
                                                         'properties': {'command': {'type': 'string'}},
                                                         'required': ['command']}}}
READ = {'type': 'function', 'function': {'name': 'read', 'description': 'read a file',
                                         'parameters': {'type': 'object',
                                                        'properties': {'path': {'type': 'string'}},
                                                        'required': ['path']}}}
TOOLS = [BASH, READ]


def tool(name, properties=None, required=None, **extra):
    params = {'type': 'object', 'properties': properties if properties is not None else {}}
    if required is not None:
        params['required'] = required
    function = {'name': name, 'parameters': params}
    function.update(extra)
    return {'type': 'function', 'function': function}


def request_json(port, method, path, body=None, headers=None):
    conn = http.client.HTTPConnection('127.0.0.1', port, timeout=20)
    try:
        payload = None if body is None else json.dumps(body)
        head = {'Content-Type': 'application/json'} if body is not None else {}
        head.update(headers or {})
        conn.request(method, path, body=payload, headers=head)
        resp = conn.getresponse()
        return resp.status, dict(resp.getheaders()), resp.read()
    finally:
        conn.close()


def sse_events(raw):
    lines = [line[len('data: '):] for line in raw.decode('utf-8').split('\n') if line.startswith('data: ')]
    assert lines and lines[-1] == '[DONE]', 'stream must end with data: [DONE]'
    return [json.loads(line) for line in lines[:-1]]


class FakeTokenizer:
    """Duck-typed stand-in for hp.Tokenizer: one token per character, no server.

    Each message is rendered on its own, so a verbatim history is a token prefix of the next
    prompt (a whole-array json.dumps is not: its closing bracket moves when a message is added).
    """

    def prompt_ids(self, body):
        text = ''.join(json.dumps(m, sort_keys=True, ensure_ascii=False) + '\n'
                       for m in body.get('messages') or [])
        return [ord(c) for c in text]

    def ids(self, text, add_special=False):
        return [ord(c) for c in (text or '')]

    def anchor_tokens(self, body):
        systems = []
        for message in body.get('messages') or []:
            if message.get('role') != 'system':
                break
            systems.append(message)
        return len(self.prompt_ids({'messages': systems}))


class HarnessProbeTestCase(unittest.TestCase):
    def start_upstream(self, **kwargs):
        server = hp.MockUpstream(0, **kwargs)
        thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': 0.02}, daemon=True)
        thread.start()

        def _stop():
            server.shutdown()
            server.server_close()
            thread.join(5)

        self.addCleanup(_stop)
        return server

    def tmpdir(self, prefix):
        path = Path(tempfile.mkdtemp(prefix=prefix))
        self.addCleanup(shutil.rmtree, path, True)
        return path


class ToolRoundsTests(unittest.TestCase):
    def test_counts_tool_messages_after_last_user_only(self):
        messages = [
            {'role': 'system', 'content': SYS},
            {'role': 'user', 'content': USER},
            {'role': 'assistant', 'content': None, 'tool_calls': []},
            {'role': 'tool', 'content': 'a'},
            {'role': 'assistant', 'content': None, 'tool_calls': []},
            {'role': 'tool', 'content': 'b'},
        ]
        self.assertEqual(hp.tool_rounds(messages), 2)

    def test_tool_messages_before_a_newer_user_message_do_not_count(self):
        messages = [
            {'role': 'user', 'content': 'first'},
            {'role': 'tool', 'content': 'old'},
            {'role': 'assistant', 'content': 'done'},
            {'role': 'user', 'content': 'second'},
            {'role': 'tool', 'content': 'new'},
        ]
        self.assertEqual(hp.tool_rounds(messages), 1)

    def test_assistant_messages_between_tools_are_skipped(self):
        messages = [{'role': 'user', 'content': 'x'}, {'role': 'tool', 'content': 'a'},
                    {'role': 'assistant', 'content': 'thinking'}, {'role': 'tool', 'content': 'b'}]
        self.assertEqual(hp.tool_rounds(messages), 2)

    def test_no_tool_messages_or_no_messages(self):
        self.assertEqual(hp.tool_rounds([{'role': 'user', 'content': 'x'}]), 0)
        self.assertEqual(hp.tool_rounds([]), 0)
        self.assertEqual(hp.tool_rounds(None), 0)


class PickToolTests(unittest.TestCase):
    def test_prefers_bash_over_other_names(self):
        tools = [tool('write_file'), tool('grep'), tool('bash'), tool('read')]
        self.assertEqual(hp.pick_tool(tools, 0)['function']['name'], 'bash')

    def test_round_one_picks_a_different_preferred_tool(self):
        tools = [tool('write_file'), tool('read'), tool('bash')]
        self.assertEqual(hp.pick_tool(tools, 0)['function']['name'], 'bash')
        self.assertEqual(hp.pick_tool(tools, 1)['function']['name'], 'read')
        self.assertEqual(hp.pick_tool(tools, 2)['function']['name'], 'bash')  # wraps around

    def test_preference_match_is_case_insensitive(self):
        self.assertEqual(hp.pick_tool([tool('write'), tool('Bash')], 0)['function']['name'], 'Bash')

    def test_falls_back_to_first_tool_when_no_preferred_name(self):
        tools = [tool('frobnicate'), tool('zap')]
        self.assertEqual(hp.pick_tool(tools, 0)['function']['name'], 'frobnicate')
        self.assertEqual(hp.pick_tool(tools, 1)['function']['name'], 'zap')

    def test_none_for_no_tools(self):
        self.assertIsNone(hp.pick_tool([], 0))
        self.assertIsNone(hp.pick_tool(None, 0))


class ScriptedArgumentsTests(unittest.TestCase):
    def test_string_properties_are_filled_by_name(self):
        args = hp.scripted_arguments(tool('bash', {'command': {'type': 'string'}}, ['command']))
        self.assertEqual(args, {'command': 'ls'})
        for name in ('path', 'file_path'):
            args = hp.scripted_arguments(tool('read', {name: {'type': 'string'}}, [name]))
            self.assertEqual(args, {name: 'README.md'})

    def test_other_string_names(self):
        args = hp.scripted_arguments(tool('grep', {'pattern': {'type': 'string'},
                                                   'query': {'type': 'string'}},
                                              ['pattern', 'query']))
        self.assertEqual(args, {'pattern': '*', 'query': 'probe'})

    def test_enum_first_value_wins(self):
        args = hp.scripted_arguments(tool('mode', {'mode': {'type': 'string', 'enum': ['fast', 'slow']}},
                                          ['mode']))
        self.assertEqual(args, {'mode': 'fast'})

    def test_integer_boolean_array_and_object_types(self):
        props = {'count': {'type': 'integer'}, 'ratio': {'type': 'number'}, 'force': {'type': 'boolean'},
                 'items': {'type': 'array'}, 'opts': {'type': 'object'}}
        args = hp.scripted_arguments(tool('misc', props, list(props)))
        self.assertEqual(args, {'count': 1, 'ratio': 1, 'force': False, 'items': [], 'opts': {}})

    def test_nullable_type_list_uses_the_non_null_type(self):
        args = hp.scripted_arguments(tool('nullable', {'path': {'type': ['null', 'string']}}, ['path']))
        self.assertEqual(args, {'path': 'README.md'})

    def test_only_required_keys_are_filled(self):
        props = {'command': {'type': 'string'}, 'timeout': {'type': 'integer'}}
        args = hp.scripted_arguments(tool('bash', props, ['command']))
        self.assertEqual(args, {'command': 'ls'})

    def test_missing_required_list_fills_first_property_only(self):
        props = {'alpha': {'type': 'integer'}, 'beta': {'type': 'integer'}}
        self.assertEqual(hp.scripted_arguments(tool('t', props)), {'alpha': 1})

    def test_required_name_without_property_is_skipped(self):
        args = hp.scripted_arguments(tool('t', {'command': {'type': 'string'}}, ['command', 'ghost']))
        self.assertEqual(args, {'command': 'ls'})

    def test_missing_or_empty_parameters(self):
        self.assertEqual(hp.scripted_arguments({'type': 'function', 'function': {'name': 'x'}}), {})
        self.assertEqual(hp.scripted_arguments({'function': {'name': 'x', 'parameters': {}}}), {})
        self.assertEqual(hp.scripted_arguments(tool('x', {}, [])), {})
        self.assertEqual(hp.scripted_arguments({'function': {'name': 'x', 'parameters': None}}), {})


class ScriptedReplyTests(unittest.TestCase):
    def body(self, messages, tools=TOOLS, **extra):
        body = {'model': 'probe-local', 'messages': messages, **extra}
        if tools is not None:
            body['tools'] = tools
        return body

    def test_first_round_is_a_tool_call_to_the_preferred_tool(self):
        content, call = hp.scripted_reply(self.body([{'role': 'user', 'content': USER}]), 2)
        self.assertIsNone(content)
        self.assertEqual(call['name'], 'bash')
        self.assertEqual(call['id'], 'call_probe0000')
        self.assertEqual(json.loads(call['arguments']), {'command': 'ls'})

    def test_second_round_is_a_different_tool_call(self):
        messages = [{'role': 'user', 'content': USER}, {'role': 'tool', 'content': 'out'}]
        content, call = hp.scripted_reply(self.body(messages), 2)
        self.assertIsNone(content)
        self.assertEqual(call['name'], 'read')
        self.assertEqual(call['id'], 'call_probe0001')
        self.assertEqual(json.loads(call['arguments']), {'path': 'README.md'})

    def test_plain_text_after_rounds_tool_results(self):
        messages = [{'role': 'user', 'content': USER}, {'role': 'tool', 'content': 'a'},
                    {'role': 'tool', 'content': 'b'}]
        self.assertEqual(hp.scripted_reply(self.body(messages), 2), ('Probe complete.', None))

    def test_zero_rounds_is_plain_text_immediately(self):
        messages = [{'role': 'user', 'content': USER}]
        self.assertEqual(hp.scripted_reply(self.body(messages), 0), ('Probe complete.', None))

    def test_plain_text_without_tools(self):
        messages = [{'role': 'user', 'content': USER}]
        self.assertEqual(hp.scripted_reply(self.body(messages, tools=None), 2), ('Probe complete.', None))
        self.assertEqual(hp.scripted_reply(self.body(messages, tools=[]), 2), ('Probe complete.', None))

    def test_plain_text_when_tool_choice_is_none(self):
        messages = [{'role': 'user', 'content': USER}]
        body = self.body(messages, tool_choice='none')
        self.assertEqual(hp.scripted_reply(body, 2), ('Probe complete.', None))


class MockUpstreamHttpTests(HarnessProbeTestCase):
    def setUp(self):
        self.server = self.start_upstream(rounds=1)
        self.port = int(self.server.url.rsplit(':', 1)[1])

    def chat(self, **extra):
        body = {'model': 'probe-local', 'messages': [{'role': 'system', 'content': SYS},
                                                     {'role': 'user', 'content': USER}],
                'tools': TOOLS, **extra}
        return request_json(self.port, 'POST', CHAT, body)

    def test_non_streaming_tool_call_reply(self):
        status, _, raw = self.chat()
        self.assertEqual(status, 200)
        choice = json.loads(raw)['choices'][0]
        self.assertEqual(choice['finish_reason'], 'tool_calls')
        message = choice['message']
        self.assertEqual(message['role'], 'assistant')
        self.assertIsNone(message['content'])
        self.assertEqual(len(message['tool_calls']), 1)
        call = message['tool_calls'][0]
        self.assertEqual(call['type'], 'function')
        self.assertEqual(call['function']['name'], 'bash')
        self.assertEqual(json.loads(call['function']['arguments']), {'command': 'ls'})

    def test_non_streaming_plain_reply_after_rounds(self):
        messages = [{'role': 'user', 'content': USER}, {'role': 'tool', 'content': 'x'}]
        status, _, raw = request_json(self.port, 'POST', CHAT, {'messages': messages, 'tools': TOOLS})
        choice = json.loads(raw)['choices'][0]
        self.assertEqual(status, 200)
        self.assertEqual(choice['finish_reason'], 'stop')
        self.assertEqual(choice['message']['content'], 'Probe complete.')
        self.assertNotIn('tool_calls', choice['message'])

    def test_streaming_tool_call_is_sse_ending_in_done(self):
        status, headers, raw = self.chat(stream=True)
        self.assertEqual(status, 200)
        self.assertIn('text/event-stream', headers['Content-Type'])
        events = sse_events(raw)
        self.assertTrue(all(e['object'] == 'chat.completion.chunk' for e in events))
        deltas = [c['delta'] for e in events for c in e['choices'] if c.get('delta')]
        names = [tc['function']['name'] for d in deltas for tc in d.get('tool_calls', []) if 'name' in tc['function']]
        self.assertEqual(names, ['bash'])
        self.assertEqual(events[-1]['choices'][0]['finish_reason'], 'tool_calls')

    def test_streaming_usage_chunk_only_with_include_usage(self):
        _, _, raw = self.chat(stream=True, stream_options={'include_usage': True})
        usage_chunks = [e for e in sse_events(raw) if 'usage' in e]
        self.assertEqual(len(usage_chunks), 1)
        self.assertEqual(usage_chunks[0]['choices'], [])
        self.assertEqual(usage_chunks[0]['usage']['total_tokens'],
                         usage_chunks[0]['usage']['prompt_tokens'] + usage_chunks[0]['usage']['completion_tokens'])

        _, _, raw = self.chat(stream=True)
        self.assertFalse([e for e in sse_events(raw) if 'usage' in e])

        _, _, raw = self.chat(stream=True, stream_options={'include_usage': False})
        self.assertFalse([e for e in sse_events(raw) if 'usage' in e])

    def test_streaming_plain_reply_ends_in_done(self):
        messages = [{'role': 'user', 'content': USER}, {'role': 'tool', 'content': 'x'}]
        _, _, raw = request_json(self.port, 'POST', CHAT, {'messages': messages, 'tools': TOOLS, 'stream': True})
        events = sse_events(raw)
        text = ''.join(c['delta'].get('content') or '' for e in events for c in e['choices'])
        self.assertEqual(text, 'Probe complete.')
        self.assertEqual(events[-1]['choices'][0]['finish_reason'], 'stop')

    def test_models_endpoint(self):
        status, _, raw = request_json(self.port, 'GET', '/v1/models')
        self.assertEqual(status, 200)
        data = json.loads(raw)
        self.assertEqual(data['object'], 'list')
        self.assertEqual(data['data'][0]['id'], 'probe-local')

    def test_request_log_never_keeps_credentials(self):
        secret = 'sk-probe-secret-value'
        body = {'messages': [{'role': 'user', 'content': USER}], 'tools': TOOLS}
        status, _, _ = request_json(self.port, 'POST', CHAT, body,
                                    headers={'Authorization': 'Bearer ' + secret})
        self.assertEqual(status, 200)
        status, _, _ = request_json(self.port, 'POST', CHAT, body,
                                    headers={'Authorization': 'Bearer ' + secret,
                                             'X-Api-Key': secret, 'Api-Key': secret,
                                             'Cookie': 'session=' + secret,
                                             'X-Probe': 'kept'})
        self.assertEqual(status, 200)
        with self.server.lock:
            logged = list(self.server.requests)
        self.assertEqual(len(logged), 2)
        for entry in logged:
            for name in ('authorization', 'x-api-key', 'api-key', 'cookie'):
                self.assertNotIn(name, entry['headers'])
        self.assertNotIn(secret, json.dumps(logged))
        self.assertEqual(logged[-1]['headers'].get('x-probe'), 'kept')
        self.assertEqual(logged[-1]['path'], CHAT)


class DescribeTurnTests(unittest.TestCase):
    def turn(self, body, **extra):
        return {'seq': 3, 'path': CHAT, 'body': body, **extra}

    def test_describes_system_tools_sampling_and_shapes(self):
        body = {
            'model': 'm', 'stream': True, 'stream_options': {'include_usage': True},
            'temperature': 0.2, 'top_p': 0.9, 'stop': ['x'], 'max_tokens': 100,
            'tool_choice': 'auto', 'store': True, 'reasoning_effort': 'low',
            'messages': [
                {'role': 'developer', 'content': 'dev text'},
                {'role': 'system', 'content': 'sys text'},
                {'role': 'user', 'content': [{'type': 'text', 'text': 'hi'}]},
                {'role': 'assistant', 'content': None,
                 'tool_calls': [{'id': 'c', 'type': 'function', 'function': {'name': 'bash', 'arguments': '{}'}}]},
                {'role': 'tool', 'content': 'res'},
                {'role': 'system', 'content': 'later note'},
            ],
            'tools': [BASH],
        }
        row = hp.describe_turn(self.turn(body))
        system_text = 'dev text\nsys text'
        self.assertEqual(row['seq'], 3)
        self.assertEqual(row['path'], CHAT)
        self.assertEqual(row['system_chars'], len(system_text))
        self.assertEqual(row['system_roles'], ['developer', 'system'])
        self.assertEqual(row['tools'], 1)
        self.assertEqual(row['tool_names'], ['bash'])
        self.assertEqual(row['sampling'], {'temperature': 0.2, 'top_p': 0.9, 'stop': ['x']})
        self.assertEqual(row['length'], {'max_tokens': 100})
        self.assertEqual(row['tool_choice'], 'auto')
        self.assertTrue(row['stream'])
        self.assertTrue(row['include_usage'])
        self.assertEqual(row['messages'], 6)
        self.assertIn('assistant:null', row['message_shapes'])
        self.assertIn('user:parts', row['message_shapes'])
        self.assertIn('tool:string', row['message_shapes'])
        self.assertIn('developer:string', row['message_shapes'])
        self.assertEqual(row['later_system_messages'], 1)

    def test_other_keys_versus_keys_ignored_by_oxe(self):
        body = {'messages': [{'role': 'user', 'content': USER}], 'store': True, 'reasoning_effort': 'low',
                'cache_prompt': True, 'chat_template_kwargs': {'enable_thinking': False}}
        row = hp.describe_turn(self.turn(body))
        self.assertEqual(row['other_keys'], ['cache_prompt', 'chat_template_kwargs', 'reasoning_effort', 'store'])
        self.assertEqual(row['keys_ignored_by_oxe'], ['reasoning_effort', 'store'])
        self.assertNotIn('cache_prompt', row['keys_ignored_by_oxe'])

    def test_stream_and_usage_defaults(self):
        row = hp.describe_turn(self.turn({'messages': [{'role': 'user', 'content': USER}]}))
        self.assertFalse(row['stream'])
        self.assertFalse(row['include_usage'])
        self.assertEqual(row['sampling'], {})
        self.assertEqual(row['length'], {})
        self.assertEqual(row['tool_choice'], 'absent')

    def test_missing_body_and_messages_do_not_crash(self):
        row = hp.describe_turn({'seq': 1, 'path': CHAT})
        self.assertEqual(row['messages'], 0)
        self.assertEqual(row['tools'], 0)
        self.assertIsNone(row['tools_sha'])
        self.assertEqual(row['system_chars'], 0)
        self.assertEqual(row['message_shapes'], [])


class HistoryCheckTests(unittest.TestCase):
    RESPONSE = {'content': '', 'reasoning': '',
                'tool_calls': [{'name': 'bash', 'arguments': '{"command": "ls", "b": 1}'}]}

    def prev(self):
        return {'body': {'messages': [{'role': 'system', 'content': SYS},
                                      {'role': 'user', 'content': USER}]},
                'response': self.RESPONSE}

    def echo(self, arguments='{"b":1,"command":"ls"}', content=None):
        return {'role': 'assistant', 'content': content,
                'tool_calls': [{'id': 'call_probe0000', 'type': 'function',
                                'function': {'name': 'bash', 'arguments': arguments}}]}

    def turn(self, messages):
        return {'body': {'messages': messages}}

    def base_messages(self):
        return [{'role': 'system', 'content': SYS}, {'role': 'user', 'content': USER}]

    def test_verbatim_prefix_with_matching_echo_ignores_key_order_and_whitespace(self):
        messages = self.base_messages() + [self.echo(), {'role': 'tool', 'tool_call_id': 'call_probe0000',
                                                         'content': 'out'}]
        result = hp.history_check(self.prev(), self.turn(messages))
        self.assertTrue(result['verbatim_prefix'])
        self.assertTrue(result['assistant_echo_matches'])
        self.assertEqual(result['assistant_content_kind'], 'null')
        self.assertEqual(result['assistant_extra_keys'], [])

    def test_echo_with_different_arguments_does_not_match(self):
        messages = self.base_messages() + [self.echo(arguments='{"command": "pwd"}')]
        result = hp.history_check(self.prev(), self.turn(messages))
        self.assertTrue(result['verbatim_prefix'])
        self.assertFalse(result['assistant_echo_matches'])

    def test_detects_changed_earlier_message(self):
        messages = [{'role': 'system', 'content': SYS}, {'role': 'user', 'content': 'changed'}, self.echo()]
        result = hp.history_check(self.prev(), self.turn(messages))
        self.assertFalse(result['verbatim_prefix'])
        self.assertIn('message 1 (user) changed', result['reason'])

    def test_detects_changed_system_prompt(self):
        messages = [{'role': 'system', 'content': SYS + ' edited'}, {'role': 'user', 'content': USER}, self.echo()]
        result = hp.history_check(self.prev(), self.turn(messages))
        self.assertFalse(result['verbatim_prefix'])
        self.assertIn('message 0 (system) changed', result['reason'])

    def test_detects_shrunk_history(self):
        result = hp.history_check(self.prev(), self.turn(self.base_messages()))
        self.assertFalse(result['verbatim_prefix'])
        self.assertIn('shrank', result['reason'])
        self.assertNotIn('assistant_echo_matches', result)

    def test_next_message_not_an_assistant_reply(self):
        messages = self.base_messages() + [{'role': 'user', 'content': 'and then?'}]
        result = hp.history_check(self.prev(), self.turn(messages))
        self.assertTrue(result['verbatim_prefix'])
        self.assertFalse(result['assistant_echo_matches'])
        self.assertIn("'user'", result['reason'])

    def test_reports_string_and_parts_content_kinds(self):
        messages = self.base_messages() + [self.echo(content='thinking out loud')]
        self.assertEqual(hp.history_check(self.prev(), self.turn(messages))['assistant_content_kind'], 'string')
        parts = [{'type': 'text', 'text': 'hi'}]
        messages = self.base_messages() + [self.echo(content=parts)]
        self.assertEqual(hp.history_check(self.prev(), self.turn(messages))['assistant_content_kind'], 'parts')


class CommonPrefixTests(unittest.TestCase):
    def test_lengths(self):
        self.assertEqual(hp.common_prefix([1, 2, 3], [1, 2, 4]), 2)
        self.assertEqual(hp.common_prefix([1, 2], [1, 2, 3]), 2)
        self.assertEqual(hp.common_prefix([1, 2, 3], [1, 2, 3]), 3)
        self.assertEqual(hp.common_prefix([], [1]), 0)
        self.assertEqual(hp.common_prefix([9], []), 0)
        self.assertEqual(hp.common_prefix('abcx', 'abdx'), 2)


class RecordedToolLoopTests(HarnessProbeTestCase):
    """Three requests through the real recorder in front of MockUpstream: a tool loop."""

    def record_tool_loop(self, alter_second_request=False):
        sr = hp.sr  # the same session_replay module harness_probe imports

        upstream = self.start_upstream(rounds=2)
        out = self.tmpdir('hp-rec-') / 'rec'
        recorder = sr.RecorderServer(('127.0.0.1', 0), upstream.url, out)
        thread = threading.Thread(target=recorder.serve_forever, kwargs={'poll_interval': 0.02}, daemon=True)
        thread.start()

        def _stop():
            recorder.shutdown()
            recorder.server_close()
            thread.join(5)

        self.addCleanup(_stop)
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        url = 'http://127.0.0.1:%d%s' % (recorder.server_address[1], CHAT)

        def call(messages):
            data = json.dumps({'model': 'probe-local', 'messages': messages, 'tools': TOOLS}).encode()
            req = urllib.request.Request(url, data=data, method='POST',
                                         headers={'Content-Type': 'application/json'})
            with opener.open(req, timeout=20) as resp:
                return json.loads(resp.read())

        system = {'role': 'system', 'content': SYS}
        first = call([system, {'role': 'user', 'content': USER}])
        call_one = first['choices'][0]['message']
        user_two = 'list the files, please' if alter_second_request else USER
        messages = [system, {'role': 'user', 'content': user_two}, call_one,
                    {'role': 'tool', 'tool_call_id': call_one['tool_calls'][0]['id'], 'content': 'README.md'}]
        second = call(messages)
        call_two = second['choices'][0]['message']
        messages = messages + [call_two,
                               {'role': 'tool', 'tool_call_id': call_two['tool_calls'][0]['id'], 'content': 'src'}]
        third = call(messages)
        self.assertEqual(first['choices'][0]['finish_reason'], 'tool_calls')
        self.assertEqual(second['choices'][0]['finish_reason'], 'tool_calls')
        self.assertEqual(third['choices'][0]['message']['content'], 'Probe complete.')

        deadline = time.time() + 5
        while time.time() < deadline:
            lines = [x for x in Path(recorder.recording).read_text().splitlines() if x.strip()]
            if len(lines) >= 4:  # header + three turns
                break
            time.sleep(0.01)
        return Path(recorder.recording)


class AnalyzeRecordingTests(RecordedToolLoopTests):
    def test_single_agent_verbatim_history_without_tokenizer(self):
        recording = self.record_tool_loop()
        result = hp.analyze_recording(recording)
        self.assertEqual(result['recording_sha256'], hashlib.sha256(recording.read_bytes()).hexdigest())
        rows = result['requests']
        self.assertEqual([r['seq'] for r in rows], [1, 2, 3])
        self.assertNotIn('history', rows[0])
        self.assertNotIn('reuse', rows[1])
        self.assertNotIn('prompt_tokens', rows[1])
        self.assertTrue(rows[1]['history']['verbatim_prefix'])
        self.assertTrue(rows[2]['history']['verbatim_prefix'])
        self.assertTrue(rows[1]['history']['assistant_echo_matches'])
        self.assertTrue(rows[2]['history']['assistant_echo_matches'])
        self.assertEqual(rows[1]['history']['assistant_content_kind'], 'null')

        summary = hp.summarize(result)
        self.assertEqual(summary['requests'], 3)
        self.assertEqual(len(summary['agents']), 1)
        agent = summary['agents'][0]
        self.assertEqual(agent['requests'], 3)
        self.assertTrue(agent['history_verbatim'])
        self.assertTrue(agent['assistant_echo_matches'])
        self.assertIsNone(agent['prompt_extends_previous_prompt'])
        self.assertEqual(agent['tools'], 2)
        self.assertEqual(agent['stream'], [False])
        self.assertEqual(agent['sampling'], ['{}'])
        self.assertIsNone(agent['anchor_tokens'])

    def test_summarize_flags_a_changed_history(self):
        recording = self.record_tool_loop(alter_second_request=True)
        summary = hp.summarize(hp.analyze_recording(recording))
        agent = summary['agents'][0]
        self.assertFalse(agent['history_verbatim'])


class TokenReuseTests(RecordedToolLoopTests):
    def test_verbatim_history_extends_previous_prompt(self):
        recording = self.record_tool_loop()
        result = hp.analyze_recording(recording, FakeTokenizer())
        rows = result['requests']
        self.assertGreater(rows[0]['prompt_tokens'], 0)
        self.assertGreater(rows[0]['anchor_tokens'], 0)
        for row in rows[1:]:
            reuse = row['reuse']
            self.assertEqual(set(reuse), {'previous_prompt_tokens', 'previous_total_tokens', 'reusable_tokens',
                                          'new_tokens', 'extends_previous_prompt', 'extends_previous_reply'})
            self.assertTrue(reuse['extends_previous_prompt'])
            self.assertGreaterEqual(reuse['reusable_tokens'], reuse['previous_prompt_tokens'])
            self.assertGreaterEqual(reuse['previous_total_tokens'], reuse['previous_prompt_tokens'])
        summary = hp.summarize(result)['agents'][0]
        self.assertTrue(summary['prompt_extends_previous_prompt'])
        self.assertIsNotNone(summary['min_reuse_share'])
        self.assertGreater(summary['min_reuse_share'], 0)

    def test_altered_earlier_message_breaks_extension(self):
        recording = self.record_tool_loop(alter_second_request=True)
        result = hp.analyze_recording(recording, FakeTokenizer())
        rows = result['requests']
        self.assertFalse(rows[1]['history']['verbatim_prefix'])
        self.assertIn('message 1 (user) changed', rows[1]['history']['reason'])
        self.assertFalse(rows[1]['reuse']['extends_previous_prompt'])
        self.assertLess(rows[1]['reuse']['reusable_tokens'], rows[1]['reuse']['previous_prompt_tokens'])
        # the third request is verbatim against the second, so it extends that prompt again
        self.assertTrue(rows[2]['reuse']['extends_previous_prompt'])
        summary = hp.summarize(result)['agents'][0]
        self.assertFalse(summary['prompt_extends_previous_prompt'])
        self.assertFalse(summary['history_verbatim'])


class CmdAnalyzeTests(RecordedToolLoopTests):
    def run_main(self, argv):
        out = io.StringIO()
        err = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = hp.main(argv)
        return code, out.getvalue(), err.getvalue()

    def test_analyze_writes_json_and_exits_zero(self):
        recording = self.record_tool_loop()
        out_path = self.tmpdir('hp-out-') / 'report.json'
        code, stdout, _ = self.run_main(['analyze', str(recording), '--out', str(out_path)])
        self.assertEqual(code, 0)
        self.assertTrue(out_path.exists())
        report = json.loads(out_path.read_text())
        self.assertEqual(list(report), ['recordings'])
        self.assertEqual(len(report['recordings']), 1)
        result = report['recordings'][0]
        self.assertEqual(result['recording'], str(recording))
        self.assertEqual(len(result['requests']), 3)
        self.assertNotIn('_first_ids', result)
        agent = result['summary']['agents'][0]
        self.assertTrue(agent['history_verbatim'])
        self.assertTrue(agent['assistant_echo_matches'])
        self.assertIn('history verbatim True', stdout)

    def test_analyze_missing_recording_is_a_usage_error(self):
        missing = self.tmpdir('hp-missing-') / 'nope.jsonl'
        code, _, _ = self.run_main(['analyze', str(missing)])
        self.assertEqual(code, 2)


if __name__ == '__main__':
    unittest.main()


def _write_recording(directory, name, turns):
    path = Path(directory) / name
    path.mkdir()
    lines = [{'kind': 'header', 'version': 1, 'tool': 'session_replay', 'created': 'x', 'upstream': 'mock'}]
    lines += [{'kind': 'turn', 'seq': i + 1, 'path': '/v1/chat/completions', 'status': 200, 'aborted': False,
               'completion_tokens': 3, 'body': body, 'response': {'content': 'ok', 'reasoning': '', 'tool_calls': []}}
              for i, body in enumerate(turns)]
    (path / 'recording.jsonl').write_text(''.join(json.dumps(line) + '\n' for line in lines))
    return str(path / 'recording.jsonl')


def _conversation(system, prompts, steps=2, side_system=None):
    """Bodies of a session: per prompt `steps` requests (prompt, then one tool round)."""
    messages, bodies = [{'role': 'system', 'content': system}], []
    for index, prompt in enumerate(prompts):
        messages = messages + [{'role': 'user', 'content': prompt}]
        bodies.append({'messages': list(messages)})
        if side_system and index == 0:
            bodies.append({'messages': [{'role': 'system', 'content': side_system},
                                        {'role': 'user', 'content': 'title for: ' + prompt}]})
        for _ in range(steps - 1):
            messages = messages + [{'role': 'assistant', 'content': 'a'}, {'role': 'tool', 'content': 'r',
                                                                           'tool_call_id': 'c'}]
            bodies.append({'messages': list(messages)})
        messages = messages + [{'role': 'assistant', 'content': 'done'}]
    return bodies


class InterleaveTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix='harness-interleave-')
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.a = _write_recording(self.tmp, 'a', _conversation('harness A', ['a1', 'a2', 'a3']))
        self.b = _write_recording(self.tmp, 'b', _conversation('harness B', ['b1', 'b2'], side_system='titles'))

    def test_prompt_blocks_keep_side_requests_with_their_prompt(self):
        _h, turns, _sha = hp.sr.load_recording(self.b)
        blocks = hp.prompt_blocks(turns)
        self.assertEqual([len(block) for block in blocks], [3, 2])
        self.assertEqual(blocks[0][1]['body']['messages'][0]['content'], 'titles')

    def test_round_robin_order_and_renumbering(self):
        header, turns = hp.interleave([self.a, self.b])
        self.assertEqual([(t['source'], t['block']) for t in turns],
                         [(0, 0)] * 2 + [(1, 0)] * 3 + [(0, 1)] * 2 + [(1, 1)] * 2 + [(0, 2)] * 2)
        self.assertEqual([t['seq'] for t in turns], list(range(1, 12)))
        self.assertEqual([item['blocks'] for item in header['interleaved_from']], [3, 2])
        self.assertEqual(turns[2]['source_seq'], 1)

    def test_command_writes_a_loadable_recording_and_refuses_to_overwrite(self):
        out = str(Path(self.tmp) / 'mixed')
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(hp.main(['interleave', self.a, self.b, '--out', out, '--allow-git']), 0)
        header, turns, _sha = hp.sr.load_recording(str(Path(out) / 'recording.jsonl'))
        self.assertEqual(len(turns), 11)
        self.assertEqual(header['kind'], 'header')
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(hp.main(['interleave', self.a, self.b, '--out', out, '--allow-git']), 2)
