"""Replay proof must ignore SSE chunking and never expose argument values."""
import importlib.util
import json
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('replay_proof', Path(__file__).resolve().parents[1] / 'tools/session_replay.py')
sr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sr)


class ReplayProofTests(unittest.TestCase):
    body = {'tools': [{'type': 'function', 'function': {'name': 'write', 'parameters': {
        'type': 'object', 'properties': {'path': {'type': 'string'}, 'ok': {'type': 'boolean'}},
        'required': ['path', 'ok'], 'additionalProperties': False}}}]}

    def response(self, arguments, name='write'):
        return {'content': '', 'reasoning': '', 'tool_calls': [{'name': name, 'arguments': arguments}]}

    def test_chunking_and_argument_key_order_do_not_change_hash(self):
        capture = sr.TurnCapture('/v1/chat/completions', b'')
        for function in ({'name': 'write', 'arguments': '{"ok":true,'}, {'arguments': '"path":"private-value"}'}):
            capture._absorb({'choices': [{'delta': {'tool_calls': [{'index': 0, 'function': function}]}}]})
        a = sr.response_proof(capture.response(), self.body)
        b = sr.response_proof(self.response('{"path":"private-value","ok":true}'), self.body)
        self.assertEqual(a['output_sha256'], b['output_sha256'])
        self.assertEqual(a['tool_json_calls'], 1)
        self.assertNotIn('private-value', json.dumps(a))

    def test_invalid_and_unknown_calls_are_counted(self):
        for argument in ('{"path":', '{"ok":NaN}', '{"ok":1e999}'):
            self.assertEqual(sr.response_proof(self.response(argument), self.body)['tool_json_errors'][0]['kind'], 'invalid-json')
        self.assertEqual(sr.response_proof(self.response('{}', 'unknown'), self.body)['tool_json_errors'][0]['kind'], 'unknown-tool')
        empty = sr.response_proof({'content': 'OK', 'reasoning': '', 'tool_calls': []}, {'tools': None})
        self.assertEqual(empty['tool_json_calls'], 0)

    @unittest.skipUnless(importlib.util.find_spec('jsonschema'), 'optional schema validator unavailable')
    def test_schema_required_type_and_external_ref(self):
        for argument in ('{"path":"secret"}', '{"path":"secret","ok":"True"}'):
            proof = sr.response_proof(self.response(argument), self.body, True)
            self.assertEqual(proof['tool_json_errors'], [{'call': 0, 'kind': 'schema-mismatch'}])
            self.assertNotIn('secret', json.dumps(proof))
        valid = sr.response_proof(self.response('{"path":"x","ok":true}'), self.body, True)
        self.assertEqual(valid['tool_json_errors'], [])
        self.assertEqual(valid['tool_json_schema_checked'], 1)
        remote = {'tools': [{'function': {'name': 'write', 'parameters': {'$ref': 'https://example.invalid/schema'}}}]}
        self.assertEqual(sr.response_proof(self.response('{}'), remote, True)['tool_json_errors'][0]['kind'], 'external-schema-ref')
        broken = {'tools': [{'function': {'name': 'write', 'parameters': {'$ref': '#/$defs/missing'}}}]}
        self.assertEqual(sr.response_proof(self.response('{}'), broken, True)['tool_json_errors'][0]['kind'], 'invalid-schema')


if __name__ == '__main__':
    unittest.main()
