"""Offline tests for tools/model_inspect.py on tiny synthetic GGUF files."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('model_inspect', ROOT / 'tools' / 'model_inspect.py')
mi = importlib.util.module_from_spec(spec)
sys.modules['model_inspect'] = mi
spec.loader.exec_module(mi)

TYPE_IDS = {'F32': 0, 'Q8_0': 8, 'Q3_K': 11, 'Q4_K': 12, 'IQ2_S': 22}


def _string(value):
    data = value.encode()
    return struct.pack('<Q', len(data)) + data


def _value(value):
    if isinstance(value, bool):
        return struct.pack('<I?', 7, value)
    if isinstance(value, int):
        return struct.pack('<II', 4, value)
    if isinstance(value, float):
        return struct.pack('<If', 6, value)
    if isinstance(value, str):
        return struct.pack('<I', 8) + _string(value)
    if isinstance(value, list) and all(isinstance(v, str) for v in value):
        return struct.pack('<IIQ', 9, 8, len(value)) + b''.join(_string(v) for v in value)
    if isinstance(value, list):
        return struct.pack('<IIQ', 9, 4, len(value)) + b''.join(struct.pack('<I', v) for v in value)
    raise TypeError(value)


def write_gguf(path, metadata, tensors, *, payload=b'\xAB', alignment=32):
    """tensors: list of (name, shape, type_name). Returns the data offset."""
    infos, offset = [], 0
    for name, shape, type_name in tensors:
        _, block, size = mi.GGML_TYPES[TYPE_IDS[type_name]]
        n = 1
        for d in shape:
            n *= d
        nbytes = n // block * size
        infos.append((name, shape, TYPE_IDS[type_name], offset, nbytes))
        offset += (nbytes + alignment - 1) // alignment * alignment
    head = b'GGUF' + struct.pack('<IQQ', 3, len(tensors), len(metadata))
    for key, value in metadata.items():
        head += _string(key) + _value(value)
    for name, shape, type_id, off, _ in infos:
        head += _string(name) + struct.pack('<I', len(shape)) + struct.pack('<' + 'Q' * len(shape), *shape)
        head += struct.pack('<IQ', type_id, off)
    data_offset = (len(head) + alignment - 1) // alignment * alignment
    with open(path, 'wb') as stream:
        stream.write(head + b'\0' * (data_offset - len(head)))
        stream.write(payload * offset)
    return data_offset


def hybrid_moe(path, **extra_meta):
    """5 blocks: 0-2 recurrent, 3 full attention, 4 MTP (nextn). 4 experts, top-2."""
    a = 'qwen35moe'
    meta = {'general.architecture': a, 'general.name': 'Tiny MoE', 'general.license': 'mit',
            f'{a}.block_count': 5, f'{a}.nextn_predict_layers': 1, f'{a}.embedding_length': 256,
            f'{a}.context_length': 4096, f'{a}.attention.head_count': 4, f'{a}.attention.head_count_kv': 2,
            f'{a}.attention.key_length': 64, f'{a}.attention.value_length': 64,
            f'{a}.full_attention_interval': 4, f'{a}.expert_count': 4, f'{a}.expert_used_count': 2,
            f'{a}.ssm.conv_kernel': 4, f'{a}.ssm.state_size': 16, f'{a}.ssm.group_count': 2,
            f'{a}.ssm.inner_size': 64, f'{a}.ssm.time_step_rank': 4,
            'tokenizer.chat_template': '{% if enable_thinking %}{% endif %}{{ tools }}',
            'tokenizer.ggml.tokens': [f't{i}' for i in range(5000)]}
    meta.update(extra_meta)
    tensors = [('token_embd.weight', [256, 64], 'Q8_0'), ('output.weight', [256, 64], 'Q8_0'),
               ('output_norm.weight', [256], 'F32')]
    for i in range(5):
        tensors += [(f'blk.{i}.attn_norm.weight', [256], 'F32'),
                    (f'blk.{i}.ffn_gate_inp.weight', [256, 4], 'F32'),
                    (f'blk.{i}.ffn_gate_exps.weight', [256, 8, 4], 'Q3_K'),
                    (f'blk.{i}.ffn_down_exps.weight', [256, 8, 4], 'Q4_K'),
                    (f'blk.{i}.ffn_up_shexp.weight', [256, 8], 'Q8_0')]
        if i < 3:
            tensors += [(f'blk.{i}.ssm_a', [4], 'F32'), (f'blk.{i}.ssm_conv1d.weight', [4, 128], 'F32'),
                        (f'blk.{i}.ssm_out.weight', [64, 256], 'Q8_0')]
        else:
            tensors += [(f'blk.{i}.attn_k.weight', [256, 128], 'Q8_0'),
                        (f'blk.{i}.attn_q.weight', [256, 256], 'Q8_0')]
        if i == 4:
            tensors += [(f'blk.{i}.nextn.eh_proj.weight', [512, 256], 'Q8_0')]
    return write_gguf(path, meta, tensors), tensors


def tensor_bytes(shape, type_name):
    _, block, size = mi.GGML_TYPES[TYPE_IDS[type_name]]
    n = 1
    for d in shape:
        n *= d
    return n // block * size


class ModelInspectTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_parser_reads_header_only(self):
        path = self.dir / 'tiny.gguf'
        data_offset, tensors = hybrid_moe(path)
        parsed = mi.parse_builtin(path)
        self.assertEqual(parsed['data_offset'], data_offset)
        self.assertLessEqual(parsed['header_bytes_read'], data_offset)
        self.assertEqual(len(parsed['tensors']), len(tensors))
        self.assertEqual(parsed['metadata']['tokenizer.ggml.tokens'], {'array_length': 5000})
        # Truncating the payload away must not change anything: the parser never reads it.
        path.write_bytes(path.read_bytes()[:data_offset])
        again = mi.parse_builtin(path)
        self.assertEqual(again['tensors'], parsed['tensors'])

    def test_rejects_non_gguf(self):
        path = self.dir / 'bad.gguf'
        path.write_bytes(b'GGML' + b'\0' * 64)
        with self.assertRaises(mi.GGUFError):
            mi.parse_builtin(path)

    def test_layout_kv_and_state(self):
        path = self.dir / 'tiny.gguf'
        hybrid_moe(path)
        info = mi.analyse(path, llama_src=self.dir)
        self.assertEqual(info['layers']['mtp'], [4])
        self.assertEqual(info['layers']['target'], 4)
        self.assertEqual(info['layers']['full_attention'], [3])
        self.assertEqual(info['layers']['recurrent'], 3)
        self.assertEqual(info['kv']['bytes_per_token_f16'], 2 * (64 + 64) * 2)   # one full layer
        self.assertEqual(info['kv']['mtp_bytes_per_token_f16'], 2 * (64 + 64) * 2)
        per_layer = 4 * (3 * (64 + 2 * 2 * 16) + 16 * 64)
        self.assertEqual(info['recurrent_state_bytes'], 3 * per_layer)
        self.assertEqual(info['experts']['count'], 4)
        self.assertEqual(info['experts']['used'], 2)
        self.assertEqual(info['experts']['shared'], 1)
        self.assertEqual(info['experts']['types']['Q3_K']['tensors'], 4)   # MTP layer excluded
        self.assertIsNone(info['llama_cpp_arch_known'])
        self.assertFalse(info['tensor_payload_read'])

    def test_repack_and_memory_split(self):
        path = self.dir / 'tiny.gguf'
        _, tensors = hybrid_moe(path)
        info = mi.analyse(path, llama_src=self.dir)
        inv = info['inventory']
        self.assertEqual(inv['Q3_K']['repack'], 'q3k-patch')
        self.assertEqual(inv['Q4_K']['repack'], 'stock')
        self.assertEqual(inv['F32']['repack'], 'no')
        total = sum(tensor_bytes(s, t) for _, s, t in tensors)
        self.assertEqual(info['tensor_bytes'], total)
        embd = tensor_bytes([256, 64], 'Q8_0')
        f32 = sum(tensor_bytes(s, t) for _, s, t in tensors if t == 'F32')
        q3k = sum(tensor_bytes(s, t) for _, s, t in tensors if t == 'Q3_K')
        stock = info['memory']['stock_build']
        patched = info['memory']['q3k_patch_build']
        self.assertEqual(stock['file_backed_bytes'], embd + f32 + q3k)
        self.assertEqual(patched['file_backed_bytes'], embd + f32)
        self.assertEqual(stock['anonymous_bytes'] + stock['file_backed_bytes'], total)
        self.assertIn('Q3_K experts', ' '.join(info['warnings']))

    def test_repack_requires_row_multiple(self):
        t = {'name': 'blk.0.attn_k.weight', 'shape': [256, 6], 'type': 'Q4_K'}
        self.assertIsNone(mi.repack_choice(t, False))
        t['shape'] = [256, 8]
        self.assertEqual(mi.repack_choice(t, False), '8x8')
        self.assertIsNone(mi.repack_choice({'name': 'token_embd.weight', 'shape': [256, 8], 'type': 'Q4_K'}, False))
        self.assertIsNone(mi.repack_choice({'name': 'blk.0.x.weight', 'shape': [256, 8], 'type': 'IQ2_S'}, True))

    def test_active_bytes_and_roofline(self):
        path = self.dir / 'tiny.gguf'
        _, tensors = hybrid_moe(path)
        info = mi.analyse(path, llama_src=self.dir)
        expected = 0.0
        for name, shape, t in tensors:
            if name.startswith('blk.4.') or name.startswith('token_embd'):
                continue
            nbytes = tensor_bytes(shape, t)
            expected += nbytes * 2 / 4 if '_exps' in name else nbytes
        self.assertEqual(info['decode']['active_bytes_per_token'], int(expected))
        self.assertAlmostEqual(info['decode']['roofline_tok_s_at_116gbs'], round(116e9 / expected, 2))
        self.assertEqual(info['mtp']['bytes'],
                         sum(tensor_bytes(s, t) for n, s, t in tensors if n.startswith('blk.4.')))

    def test_suggested_profile_moe_with_mtp(self):
        path = self.dir / 'tiny.gguf'
        hybrid_moe(path)
        prof = mi.analyse(path, llama_src=self.dir)['suggested_profile']
        self.assertFalse(prof['ready'])
        self.assertEqual(prof['context_size'], 16384)
        self.assertEqual(prof['memlat_overrides']['threads'], 8)
        self.assertEqual(prof['memlat_overrides']['cpu_mask'], '0xff')
        self.assertEqual(prof['memlat_overrides']['spec_type'], 'draft-mtp')
        self.assertEqual(prof['environment'], {'LLAMA_CPU_SMALL_BATCH_LIMIT': '3'})
        self.assertIn('--chat-template-kwargs', prof['extra_args'])
        self.assertIn('--cache-ram', prof['extra_args'])
        self.assertEqual(prof['extra_args'][prof['extra_args'].index('--cache-ram') + 1], '0')
        self.assertFalse(prof['prefix_cache']['enabled'])

    def test_dense_without_mtp_or_thinking(self):
        a = 'llama'
        meta = {'general.architecture': a, f'{a}.block_count': 2, f'{a}.embedding_length': 256,
                f'{a}.attention.head_count': 4, f'{a}.attention.head_count_kv': 4,
                'tokenizer.chat_template': '{{ messages }}'}
        tensors = [('token_embd.weight', [256, 32], 'Q4_K')]
        for i in range(2):
            tensors += [(f'blk.{i}.attn_k.weight', [256, 256], 'Q4_K'), (f'blk.{i}.ffn_up.weight', [256, 64], 'Q4_K')]
        path = self.dir / 'dense.gguf'
        write_gguf(path, meta, tensors)
        info = mi.analyse(path, llama_src=self.dir)
        self.assertEqual(info['kv']['bytes_per_token_f16'], 2 * 4 * (64 + 64) * 2)  # head dim 256/4
        # tied embeddings: no output.weight, so the embedding table is read as the head
        self.assertEqual(info['decode']['active_bytes_per_token'],
                         sum(tensor_bytes(s, t) for _, s, t in tensors))
        prof = info['suggested_profile']
        self.assertEqual(prof['memlat_overrides']['threads'], 6)
        self.assertEqual(prof['memlat_overrides']['spec_type'], 'none')
        self.assertNotIn('--reasoning-budget', prof['extra_args'])
        self.assertNotIn('--chat-template-kwargs', prof['extra_args'])
        joined = ' '.join(info['warnings'])
        self.assertIn('no embedded MTP head', joined)
        self.assertIn('no general.license', joined)
        self.assertIn('enable_thinking', joined)

    def test_split_shards(self):
        a = 'llama'
        meta = {'general.architecture': a, f'{a}.block_count': 1, f'{a}.attention.head_count': 1,
                f'{a}.embedding_length': 256}
        write_gguf(self.dir / 'm-00001-of-00002.gguf', meta, [('blk.0.attn_k.weight', [256, 8], 'Q8_0')])
        with self.assertRaises(mi.GGUFError):
            mi.analyse(self.dir / 'm-00001-of-00002.gguf', llama_src=self.dir)
        write_gguf(self.dir / 'm-00002-of-00002.gguf', meta, [('blk.0.ffn_up.weight', [256, 8], 'Q8_0')])
        info = mi.analyse(self.dir / 'm-00002-of-00002.gguf', llama_src=self.dir)
        self.assertEqual(len(info['shards']), 2)
        self.assertEqual(info['tensor_bytes'], 2 * tensor_bytes([256, 8], 'Q8_0'))

    def test_arch_check_and_location_warning(self):
        src = self.dir / 'llama'
        (src / 'src').mkdir(parents=True)
        (src / 'src/llama-arch.cpp').write_text('{ LLM_ARCH_LLAMA, "llama" },\n')
        path = self.dir / 'tiny.gguf'
        hybrid_moe(path)
        self.assertFalse(mi.analyse(path, llama_src=src)['llama_cpp_arch_known'])
        info = mi.analyse(path, llama_src=self.dir)
        info['path'] = str(Path.home() / 'Work' / 'x.gguf')
        self.assertIn('~/Work', ' '.join(mi.warnings(info)))

    def test_cli_outputs(self):
        path = self.dir / 'tiny.gguf'
        hybrid_moe(path)
        for flag, check in (('--json', lambda v: v['architecture'] == 'qwen35moe'),
                            ('--profile', lambda v: v['ready'] is False)):
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(mi.main([str(path), flag, '--llama-src', str(self.dir)]), 0)
            self.assertTrue(check(json.loads(out.getvalue())))
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            mi.main([str(path), '--llama-src', str(self.dir)])
        self.assertIn('roofline', out.getvalue())
        runtime = self.dir / 'runtime.json'
        runtime.write_text(json.dumps({'profiles': {'tiny-local': {
            'model_alias': 'tiny-local', 'model_path': str(path), 'server_binary': '/x/llama-server',
            'server_sha256': 'a' * 64}}}))
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            mi.main(['--runtime', str(runtime), '--profile', '--llama-src', str(self.dir)])
        self.assertEqual(json.loads(out.getvalue())['server_binary'], '/x/llama-server')

    @unittest.skipUnless(importlib.util.find_spec('gguf'), 'gguf package not installed')
    def test_gguf_package_backend_matches(self):
        path = self.dir / 'tiny.gguf'
        hybrid_moe(path)
        a = mi.analyse(path, 'builtin', self.dir)
        b = mi.analyse(path, 'gguf', self.dir)
        a.pop('parser'), b.pop('parser')
        self.assertEqual(a, b)


if __name__ == '__main__':
    unittest.main()
