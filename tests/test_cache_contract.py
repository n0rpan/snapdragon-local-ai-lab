"""Model-free tests for the CACHE-PERFECT contract checks (C1, C2, C3) and the GGUF header reader."""
import itertools
import os
import shutil
import struct
import sys
import tempfile
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
sys.path.insert(0, str(TOOLS))
import cache_contract as cc  # noqa: E402

REAL_GGUF = Path.home() / '.local/share/ai-stack/models/images/Qwen3-4B-Instruct-2507-Q4_K_M.gguf'


def toy_render(messages, drop_reasoning=True):
    """A ChatML-like stand-in: one byte is one token. With drop_reasoning, reasoning before the last
    user message is removed (the Qwen-family behaviour that breaks C1)."""
    last_user = max((i for i, m in enumerate(messages) if m['role'] == 'user'), default=-1)
    out = []
    for i, m in enumerate(messages):
        body = m.get('content', '')
        reasoning = m.get('reasoning', '')
        if reasoning and (not drop_reasoning or i > last_user):
            body = '<think>' + reasoning + '</think>' + body
        out.append('<|im_start|>' + m['role'] + '\n' + body + '<|im_end|>\n')
    return list(''.join(out).encode('utf-8'))


def toy_parse(text):
    """Completion text back to a message: the think block becomes `reasoning`."""
    msg = {'role': 'assistant', 'content': text}
    if text.startswith('<think>') and '</think>' in text:
        reasoning, rest = text[len('<think>'):].split('</think>', 1)
        msg.update(reasoning=reasoning, content=rest)
    return msg


def history(n_turns):
    msgs = [{'role': 'system', 'content': 'sys prompt'}]
    for i in range(n_turns):
        msgs.append({'role': 'user', 'content': f'question {i}'})
        msgs.append({'role': 'assistant', 'content': f'answer {i}', 'reasoning': f'think {i}'})
    msgs.append({'role': 'user', 'content': 'final'})
    return msgs


def gguf_bytes(kv):
    """A minimal GGUF v3 header: no tensors, the given metadata (name -> (type, payload bytes))."""
    out = bytearray(b'GGUF' + struct.pack('<I', 3) + struct.pack('<QQ', 0, len(kv)))
    for key, (kind, payload) in kv.items():
        raw = key.encode()
        out += struct.pack('<Q', len(raw)) + raw + struct.pack('<I', kind) + payload
    return bytes(out)


def gguf_str(s):
    raw = s.encode()
    return struct.pack('<Q', len(raw)) + raw


class PrefixMonotoneTests(unittest.TestCase):
    def test_keeping_reasoning_is_prefix_monotone_for_every_history(self):
        for n in range(0, 4):
            self.assertEqual(cc.prefix_monotone_violations(lambda h: toy_render(h, False), history(n)), [])

    def test_dropping_reasoning_breaks_prefix_monotone_rendering(self):
        viol = cc.prefix_monotone_violations(lambda h: toy_render(h, True), history(2))
        self.assertTrue(viol)
        k, first = viol[0]
        self.assertGreater(first, 0)

    def test_render_is_deterministic(self):
        self.assertTrue(cc.render_deterministic(lambda h: toy_render(h, False), history(2)))
        counter = itertools.count()
        self.assertFalse(cc.render_deterministic(lambda h: [next(counter)], history(1)))


class RoundTripTests(unittest.TestCase):
    def test_completion_round_trips_when_rendered_back_the_same_way(self):
        text = '<think>plan it</think>call the tool'
        generated = list(text.encode())
        def body(m):
            prefix = '<think>' + m['reasoning'] + '</think>' if m.get('reasoning') else ''
            return list((prefix + m['content']).encode())
        ok, where = cc.roundtrip(toy_parse, body, text, generated)
        self.assertTrue(ok)
        self.assertIsNone(where)

    def test_reserialised_tool_arguments_report_the_first_divergent_offset(self):
        generated_text = '{"path": "a.txt", "limit": 5}'
        client_text = '{"limit":5,"path":"a.txt"}'          # same JSON, key order changed by the client
        generated = list(generated_text.encode())
        ok, where = cc.roundtrip(lambda t: t, lambda t: list(t.encode()), client_text, generated)
        self.assertFalse(ok)
        self.assertEqual(where, 2)


class InvariantTests(unittest.TestCase):
    def test_c3_passes_at_the_exact_bound_and_fails_one_below(self):
        prev = {'cache_n': 100, 'prompt_n': 50, 'predicted_n': 20}
        self.assertTrue(cc.turn_invariant_ok(prev, {'cache_n': 169}))
        self.assertFalse(cc.turn_invariant_ok(prev, {'cache_n': 168}))

    def test_edited_history_is_exempt(self):
        prev = {'cache_n': 100, 'prompt_n': 50, 'predicted_n': 20}
        self.assertTrue(cc.turn_invariant_ok(prev, {'cache_n': 10}, edited=True))

    def test_missing_predicted_count_counts_as_zero(self):
        prev = {'cache_n': 100, 'prompt_n': 50}
        self.assertTrue(cc.turn_invariant_ok(prev, {'cache_n': 149}))
        self.assertFalse(cc.turn_invariant_ok(prev, {'cache_n': 148}))


class GgufHeaderTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix='gguf-test-')

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_reads_string_number_and_array_count_without_tensor_data(self):
        arr = struct.pack('<IQ', 8, 3) + gguf_str('a') + gguf_str('bc') + gguf_str('def')
        data = gguf_bytes({
            'general.architecture': (8, gguf_str('qwen3')),
            'tokenizer.ggml.tokens': (9, arr),
            'tokenizer.ggml.bos_token_id': (4, struct.pack('<I', 7)),
            'tokenizer.chat_template': (8, gguf_str('{% if x %}preserve_thinking{% endif %}')),
        })
        path = os.path.join(self.dir, 'tiny.gguf')
        Path(path).write_bytes(data + b'\x00' * 64)   # trailing bytes are never read
        meta = cc.read_gguf_metadata(path, ('general.architecture', 'tokenizer.ggml.tokens',
                                            'tokenizer.ggml.bos_token_id', 'tokenizer.chat_template'))
        self.assertEqual(meta, {'general.architecture': 'qwen3', 'tokenizer.ggml.tokens': 3,
                                'tokenizer.ggml.bos_token_id': 7,
                                'tokenizer.chat_template': '{% if x %}preserve_thinking{% endif %}'})

    def test_refuses_non_gguf_and_truncated_headers(self):
        bad = os.path.join(self.dir, 'bad.gguf')
        Path(bad).write_bytes(b'NOPE' + b'\x00' * 32)
        with self.assertRaises(cc.ContractError):
            cc.read_gguf_metadata(bad, ('general.architecture',))
        short = os.path.join(self.dir, 'short.gguf')
        Path(short).write_bytes(b'GGUF' + struct.pack('<I', 3))
        with self.assertRaises(cc.ContractError):
            cc.read_gguf_metadata(short, ('general.architecture',))

    def test_counts_and_seeks_are_bounded_by_the_file(self):
        cases = {
            'truncated-scalar-array': {'a.ids': (9, struct.pack('<IQ', 4, 1000) + b'\x00' * 8),
                                       'general.architecture': (8, gguf_str('qwen3'))},
            'huge-string-array': {'a.toks': (9, struct.pack('<IQ', 8, 1 << 40))},
            'huge-scalar-array': {'a.ids': (9, struct.pack('<IQ', 0, 1 << 62))},
            'nested-too-deep': {'a.n': (9, struct.pack('<IQ', 9, 1) + struct.pack('<IQ', 9, 1) + struct.pack('<IQ', 0, 0))},
        }
        for name, kv in cases.items():
            path = os.path.join(self.dir, name + '.gguf')
            Path(path).write_bytes(gguf_bytes(kv))
            with self.assertRaises(cc.ContractError, msg=name):
                cc.read_gguf_metadata(path, ('general.architecture', 'missing.key'))
        huge = os.path.join(self.dir, 'huge-kv.gguf')
        Path(huge).write_bytes(b'GGUF' + struct.pack('<I', 3) + struct.pack('<QQ', 0, 1 << 50) + b'\x00' * 64)
        with self.assertRaises(cc.ContractError):
            cc.read_gguf_metadata(huge, ('general.architecture',))

    def test_template_shape_flags_reasoning_drop_pattern(self):
        tmpl = '{% set ns.last_query_index = 3 %}...'
        path = os.path.join(self.dir, 'q.gguf')
        Path(path).write_bytes(gguf_bytes({'tokenizer.chat_template': (8, gguf_str(tmpl)),
                                           'general.architecture': (8, gguf_str('qwen3'))}))
        shape = cc.template_shape(path)
        self.assertTrue(shape['drops_reasoning_before_last_query'])
        self.assertFalse(shape['has_preserve_thinking'])

    @unittest.skipUnless(REAL_GGUF.exists(), 'no local GGUF to read the header of')
    def test_real_header_reads_template_metadata(self):
        shape = cc.template_shape(str(REAL_GGUF))
        self.assertEqual(shape['architecture'], 'qwen3')
        self.assertGreater(shape['template_chars'], 0)
        self.assertGreater(shape['vocab_size'], 0)


if __name__ == '__main__':
    unittest.main()
