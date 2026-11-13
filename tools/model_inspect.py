#!/usr/bin/env python3
"""Offline GGUF inspection for onboarding a model on the Snapdragon X Elite laptop.

Reads only the GGUF header, metadata and tensor-info table (never tensor data),
then reports the facts the onboarding guide needs: architecture, layer and
attention layout, KV bytes per token, experts, a per-type byte inventory, which
tensors the CPU repack path takes (anonymous RAM) versus which stay mmapped
(file-backed), the MTP head, active bytes per decoded token, a decode roofline
and a suggested starting runtime profile.

Usage:
  python3 tools/model_inspect.py MODEL.gguf [--json] [--profile]
  python3 tools/model_inspect.py --runtime ~/.config/local-ai/llm/runtime.json

The built-in parser is used by default because it reads a bounded prefix of
the file. If the `gguf` package is installed, `--backend gguf` uses
gguf.GGUFReader instead (it memory-maps the file but touches only the header
pages); `--backend auto` prefers it and falls back to the built-in parser on
type ids the package does not know. Estimates are formula-based, not
measurements; docs/MODEL-ONBOARDING.md says which run confirms each one.
"""
import argparse
import json
import os
from pathlib import Path
import re
import struct
import sys

GIB = 1024 ** 3
MEMORY_CAP_BYTES = 28 * GIB          # local-ai-chat.service MemoryMax=28G
CAP_MARGIN_BYTES = 1536 * 1024 ** 2  # controller stops below 1.5 GiB MemAvailable
READ_BANDWIDTH_GBS = 116.0           # measured memlat CPU read ceiling (115.85 GB/s)
MAX_HEADER_BYTES = 256 * 1024 ** 2   # refuse anything that needs more header than this
LONG_ARRAY = 4096                    # arrays longer than this are counted, not stored
DEFAULT_LLAMA_SRC = Path.home() / '.local/share/ai-stack/src/llama.cpp'
MODEL_HOME = Path.home() / '.local/share/ai-stack/models/llm'
PREFIX_DIR = Path.home() / '.cache/local-ai/prefixes'

# ggml_type id -> (name, block elements, block bytes), from the pinned
# llama.cpp 954de24 ggml.h / ggml-common.h.
GGML_TYPES = {
    0: ('F32', 1, 4), 1: ('F16', 1, 2), 2: ('Q4_0', 32, 18), 3: ('Q4_1', 32, 20),
    6: ('Q5_0', 32, 22), 7: ('Q5_1', 32, 24), 8: ('Q8_0', 32, 34), 9: ('Q8_1', 32, 36),
    10: ('Q2_K', 256, 84), 11: ('Q3_K', 256, 110), 12: ('Q4_K', 256, 144),
    13: ('Q5_K', 256, 176), 14: ('Q6_K', 256, 210), 15: ('Q8_K', 256, 292),
    16: ('IQ2_XXS', 256, 66), 17: ('IQ2_XS', 256, 74), 18: ('IQ3_XXS', 256, 98),
    19: ('IQ1_S', 256, 50), 20: ('IQ4_NL', 32, 18), 21: ('IQ3_S', 256, 110),
    22: ('IQ2_S', 256, 82), 23: ('IQ4_XS', 256, 136), 24: ('I8', 1, 1), 25: ('I16', 1, 2),
    26: ('I32', 1, 4), 27: ('I64', 1, 8), 28: ('F64', 1, 8), 29: ('IQ1_M', 256, 56),
    30: ('BF16', 1, 2), 34: ('TQ1_0', 256, 54), 35: ('TQ2_0', 256, 66),
    39: ('MXFP4', 32, 17), 40: ('NVFP4', 64, 36), 41: ('Q1_0', 128, 18),
    42: ('Q2_0', 64, 18), 43: ('F8_E4M3', 1, 1),
}

# CPU_REPACK selection on this CPU (NEON + dotprod + i8mm, no SVE), from
# ggml_repack_get_optimal_repack_type in the pinned source: type -> (layout, ne[1] multiple).
REPACK_STOCK = {'Q4_0': ('4x8', 4), 'Q4_K': ('8x8', 8), 'Q5_K': ('8x8', 8), 'Q6_K': ('8x8', 8),
                'Q8_0': ('4x8', 4), 'IQ4_NL': ('4x4', 4), 'MXFP4': ('4x4', 4), 'Q1_0': ('4x8', 4)}
# patches/llama-q3k-repack.patch (GGML_CPU_REPACK_Q3K unset or != 0)
REPACK_Q3K_PATCH = {'Q3_K': ('8x8', 8)}
# Expert types with no ARM fast path: the IQ2/IQ3/IQ1 family runs scalar-ish vec_dot.
SLOW_EXPERT_TYPES = {'IQ1_S', 'IQ1_M', 'IQ2_XXS', 'IQ2_XS', 'IQ2_S', 'IQ3_XXS', 'IQ3_S', 'Q2_K', 'TQ1_0', 'TQ2_0'}
# Weights consumed by ops other than MUL_MAT / MUL_MAT_ID never enter the repack buffer.
NOT_MATMUL = re.compile(r'^(token_embd|per_layer_token_embd|position_embd|rope_freqs)\.'
                        r'|norm\.weight$|conv1d|\.ssm_(a|d|dt)\b|\.bias$')
STANDARD_ARGS = ['--device', 'none', '--no-op-offload', '--no-host', '--parallel', '1',
                 '--threads-batch', '10', '--batch-size', '256', '--ubatch-size', '256', '--repack',
                 '--flash-attn', 'off', '--cache-type-k', 'f16', '--cache-type-v', 'f16',
                 '--cache-ram', '0', '--ctx-checkpoints', '4', '--no-warmup', '--jinja',
                 '--reasoning-budget', '768', '--log-disable']


class GGUFError(ValueError):
    pass


def type_info(type_id):
    if type_id in GGML_TYPES:
        return GGML_TYPES[type_id]
    try:  # optional: newer gguf packages know more type ids
        import gguf
        quant = gguf.GGMLQuantizationType(type_id)
        block, size = gguf.GGML_QUANT_SIZES[quant]
        return (quant.name, block, size)
    except Exception:
        raise GGUFError(f'unknown ggml type id {type_id}') from None


# ---------------------------------------------------------------- parsers

class _Reader:
    """Bounded sequential reader over the start of a file."""

    def __init__(self, stream):
        self.stream = stream
        self.pos = 0

    def read(self, n):
        if self.pos + n > MAX_HEADER_BYTES:
            raise GGUFError('header larger than the inspection bound')
        data = self.stream.read(n)
        if len(data) != n:
            raise GGUFError('truncated GGUF header')
        self.pos += n
        return data

    def unpack(self, fmt):
        return struct.unpack('<' + fmt, self.read(struct.calcsize('<' + fmt)))

    def string(self):
        (n,) = self.unpack('Q')
        return self.read(n).decode('utf-8', 'replace')


_SCALAR = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i', 6: 'f', 7: '?', 10: 'Q', 11: 'q', 12: 'd'}


def _value(reader, vtype):
    if vtype in _SCALAR:
        return reader.unpack(_SCALAR[vtype])[0]
    if vtype == 8:
        return reader.string()
    if vtype == 9:
        item_type, count = reader.unpack('IQ')
        if count > LONG_ARRAY:
            if item_type in _SCALAR:  # skip fixed-size items without decoding them
                reader.read(struct.calcsize('<' + _SCALAR[item_type]) * count)
            else:
                for _ in range(count):
                    _value(reader, item_type)
            return {'array_length': count}
        return [_value(reader, item_type) for _ in range(count)]
    raise GGUFError(f'unknown metadata value type {vtype}')


def parse_builtin(path):
    with open(path, 'rb') as stream:
        reader = _Reader(stream)
        if reader.read(4) != b'GGUF':
            raise GGUFError('not a GGUF file')
        (version,) = reader.unpack('I')
        if version not in (2, 3):
            raise GGUFError(f'unsupported GGUF version {version}')
        n_tensors, n_kv = reader.unpack('QQ')
        metadata = {}
        for _ in range(n_kv):
            key = reader.string()
            (vtype,) = reader.unpack('I')
            metadata[key] = _value(reader, vtype)
        tensors = []
        for _ in range(n_tensors):
            name = reader.string()
            (n_dims,) = reader.unpack('I')
            shape = list(reader.unpack('Q' * n_dims))
            type_id, offset = reader.unpack('IQ')
            tensors.append({'name': name, 'shape': shape, 'type_id': type_id, 'offset': offset})
        alignment = int(metadata.get('general.alignment', 32))
        data_offset = (reader.pos + alignment - 1) // alignment * alignment
        return {'version': version, 'metadata': metadata, 'tensors': tensors,
                'data_offset': data_offset, 'header_bytes_read': reader.pos, 'backend': 'builtin'}


def parse_gguf_package(path):
    import gguf
    reader = gguf.GGUFReader(path, 'r')
    metadata = {}
    for name, field in reader.fields.items():
        if name.startswith('GGUF.'):
            continue
        if field.types and field.types[0] == gguf.GGUFValueType.ARRAY and len(field.data) > LONG_ARRAY:
            metadata[name] = {'array_length': len(field.data)}
            continue
        value = field.contents()
        metadata[name] = value.tolist() if hasattr(value, 'tolist') else value
    tensors = [{'name': t.name, 'shape': [int(x) for x in t.shape], 'type_id': int(t.tensor_type),
                'offset': int(t.data_offset) - int(reader.data_offset)} for t in reader.tensors]
    return {'version': int(reader.fields['GGUF.version'].contents()), 'metadata': metadata,
            'tensors': tensors, 'data_offset': int(reader.data_offset),
            'header_bytes_read': int(reader.data_offset), 'backend': 'gguf'}


def parse(path, backend='builtin'):
    if backend in ('gguf', 'auto'):
        try:
            import gguf  # noqa: F401
        except ImportError:
            if backend == 'gguf':
                raise GGUFError('the gguf package is not installed')
            return parse_builtin(path)
        try:
            return parse_gguf_package(path)
        except Exception:
            if backend == 'gguf':
                raise
            return parse_builtin(path)
    return parse_builtin(path)


def shard_paths(path):
    """All shards of a split GGUF given any shard, else just the file."""
    match = re.match(r'^(.*)-(\d{5})-of-(\d{5})\.gguf$', path.name)
    if not match:
        return [path]
    total = int(match.group(3))
    paths = [path.with_name(f'{match.group(1)}-{i:05d}-of-{total:05d}.gguf') for i in range(1, total + 1)]
    missing = [p.name for p in paths if not p.is_file()]
    if missing:
        raise GGUFError('missing shards: ' + ', '.join(missing))
    return paths


# ---------------------------------------------------------------- analysis

def _elements(shape):
    n = 1
    for d in shape:
        n *= d
    return n


def _layer(name):
    match = re.match(r'^blk\.(\d+)\.', name)
    return int(match.group(1)) if match else None


def _meta(metadata, arch, key, default=None):
    return metadata.get(f'{arch}.{key}', default)


def _per_layer(value, layer, default=0):
    if isinstance(value, list):
        return value[layer] if layer < len(value) else default
    return default if value is None else value


def repack_choice(tensor, q3k_patch):
    """Return the CPU_REPACK layout this tensor gets on this CPU, or None."""
    name, shape, type_name = tensor['name'], tensor['shape'], tensor['type']
    if len(shape) not in (2, 3) or NOT_MATMUL.search(name):
        return None
    if len(shape) == 3 and '_exps' not in name:
        return None
    table = dict(REPACK_STOCK, **(REPACK_Q3K_PATCH if q3k_patch else {}))
    if type_name not in table:
        return None
    layout, multiple = table[type_name]
    return layout if shape[1] % multiple == 0 else None


def analyse(path, backend='builtin', llama_src=DEFAULT_LLAMA_SRC):
    path = Path(path).expanduser()
    shards = shard_paths(path)
    parsed = [parse(p, backend) for p in shards]
    metadata = dict(parsed[0]['metadata'])
    tensors = []
    for part in parsed:
        for t in part['tensors']:
            name, block, size = type_info(t['type_id'])
            n = _elements(t['shape'])
            if n % block:
                raise GGUFError(f"{t['name']}: {n} elements not a multiple of {name} block {block}")
            tensors.append(dict(t, type=name, elements=n, bytes=n // block * size))
    file_bytes = sum(p.stat().st_size for p in shards)
    tensor_bytes = sum(t['bytes'] for t in tensors)
    header_bytes = sum(p['data_offset'] for p in parsed)
    arch = metadata.get('general.architecture', 'unknown')

    # ---- layer layout from the tensor table (metadata cross-checks it)
    n_layers = int(_meta(metadata, arch, 'block_count', 0) or 0)
    n_nextn = int(_meta(metadata, arch, 'nextn_predict_layers', 0) or 0)
    by_layer = {}
    for t in tensors:
        layer = _layer(t['name'])
        if layer is not None:
            by_layer.setdefault(layer, []).append(t)
    mtp_layers = sorted(i for i in by_layer if i >= n_layers - n_nextn) if n_nextn else []
    mtp_layers += sorted(i for i, ts in by_layer.items()
                         if i not in mtp_layers and any('.nextn.' in t['name'] for t in ts))
    target_layers = [i for i in sorted(by_layer) if i not in mtp_layers]
    recurrent, full = [], []
    for i in sorted(by_layer):
        names = [t['name'] for t in by_layer[i]]
        if any('.ssm_' in n for n in names):
            recurrent.append(i)
        elif any(re.search(r'\.attn_(k|kv|qkv|k_b|kv_a_mqa)\.weight$', n) for n in names):
            full.append(i)

    # ---- attention / KV
    head_count = _meta(metadata, arch, 'attention.head_count', 0)
    head_kv = _meta(metadata, arch, 'attention.head_count_kv', head_count)
    embd = int(_meta(metadata, arch, 'embedding_length', 0) or 0)
    max_heads = max(head_count) if isinstance(head_count, list) else int(head_count or 0)
    key_len = _meta(metadata, arch, 'attention.key_length') or (embd // max_heads if max_heads else 0)
    value_len = _meta(metadata, arch, 'attention.value_length') or key_len
    kv_lora = _meta(metadata, arch, 'attention.kv_lora_rank')
    rope_dims = _meta(metadata, arch, 'rope.dimension_count', 0) or 0

    def layer_kv(i, bytes_per_elem=2):
        if kv_lora:  # MLA caches the compressed latent plus the rope part
            return int((kv_lora + rope_dims) * bytes_per_elem)
        kv = _per_layer(head_kv, i, max_heads)
        return int(kv * (key_len + value_len) * bytes_per_elem)

    target_full = [i for i in full if i in target_layers]
    kv_target = sum(layer_kv(i) for i in target_full)
    kv_mtp = sum(layer_kv(i) for i in full if i in mtp_layers)
    sliding = _meta(metadata, arch, 'attention.sliding_window')

    # ---- recurrent state (gated delta net / mamba style), f32, independent of context
    conv_kernel = _meta(metadata, arch, 'ssm.conv_kernel', 0) or 0
    state_size = _meta(metadata, arch, 'ssm.state_size', 0) or 0
    groups = _meta(metadata, arch, 'ssm.group_count', 0) or 0
    inner = _meta(metadata, arch, 'ssm.inner_size', 0) or 0
    per_layer_state = 4 * ((max(conv_kernel - 1, 0)) * (inner + 2 * groups * state_size) + state_size * inner)
    recurrent_target = [i for i in recurrent if i in target_layers]
    recurrent_bytes = per_layer_state * len(recurrent_target)

    # ---- experts
    expert_count = int(_meta(metadata, arch, 'expert_count', 0) or 0)
    expert_used = int(_meta(metadata, arch, 'expert_used_count', 0) or 0)
    expert_shared = int(_meta(metadata, arch, 'expert_shared_count', 0) or 0)
    has_shared_tensors = any('_shexp' in t['name'] for t in tensors)

    # ---- per-tensor classification
    def role(t):
        layer = _layer(t['name'])
        if layer is not None and layer in mtp_layers:
            return 'mtp'
        if '_exps' in t['name']:
            return 'expert'
        if re.match(r'^(token_embd|per_layer_token_embd)\.', t['name']):
            return 'embedding'
        return 'dense'

    inventory = {}
    totals = {'stock': {'anon': 0, 'file': 0}, 'q3k_patch': {'anon': 0, 'file': 0}}
    active_bytes = 0.0
    mtp_bytes = 0
    expert_types = {}
    has_output = any(t['name'] == 'output.weight' for t in tensors)
    for t in tensors:
        r = role(t)
        stock = repack_choice(t, False)
        patched = repack_choice(t, True)
        row = inventory.setdefault(t['type'], {'tensors': 0, 'bytes': 0, 'expert_bytes': 0, 'mtp_bytes': 0,
                                               'repacked_stock_bytes': 0, 'repacked_q3k_patch_bytes': 0})
        row['tensors'] += 1
        row['bytes'] += t['bytes']
        if r == 'expert':
            row['expert_bytes'] += t['bytes']
            expert_types.setdefault(t['type'], {'tensors': 0, 'bytes': 0})
            expert_types[t['type']]['tensors'] += 1
            expert_types[t['type']]['bytes'] += t['bytes']
        if r == 'mtp':
            row['mtp_bytes'] += t['bytes']
            mtp_bytes += t['bytes']
        if stock:
            row['repacked_stock_bytes'] += t['bytes']
        if patched:
            row['repacked_q3k_patch_bytes'] += t['bytes']
        totals['stock']['anon' if stock else 'file'] += t['bytes']
        totals['q3k_patch']['anon' if patched else 'file'] += t['bytes']
        # decode traffic: whole target weights, routed experts at top-k/count,
        # one embedding row per token, MTP excluded (only read when drafting)
        if r == 'expert' and expert_count:
            active_bytes += t['bytes'] * expert_used / expert_count
        elif r == 'dense':
            active_bytes += t['bytes']
        elif r == 'embedding' and not has_output and t['name'].startswith('token_embd'):
            active_bytes += t['bytes']  # tied embeddings double as the output head
    for row in inventory.values():
        row['repack'] = ('stock' if row['repacked_stock_bytes'] else
                         'q3k-patch' if row['repacked_q3k_patch_bytes'] else 'no')

    # ---- llama.cpp support (pinned source, read-only)
    arch_known = None
    arch_file = Path(llama_src).expanduser() / 'src/llama-arch.cpp'
    if arch_file.is_file():
        arch_known = f'"{arch}"' in arch_file.read_text(errors='replace')

    template = metadata.get('tokenizer.chat_template')
    template = template if isinstance(template, str) else None
    import hashlib
    result = {
        'path': str(path),
        'shards': [str(p) for p in shards],
        'gguf_version': parsed[0]['version'],
        'parser': parsed[0]['backend'],
        'file_bytes': file_bytes,
        'header_bytes': header_bytes,
        'tensor_bytes': tensor_bytes,
        'tensor_payload_read': False,
        'name': metadata.get('general.name'),
        'architecture': arch,
        'llama_cpp_arch_known': arch_known,
        'license': metadata.get('general.license') or metadata.get('general.license.name'),
        'license_link': metadata.get('general.license.link'),
        'base_model': metadata.get('general.base_model.0.repo_url') or metadata.get('general.base_model.0.name'),
        'file_type': metadata.get('general.file_type'),
        'context_length_trained': _meta(metadata, arch, 'context_length'),
        'embedding_length': embd,
        'vocab_size': (metadata.get('tokenizer.ggml.tokens') or {}).get('array_length')
        if isinstance(metadata.get('tokenizer.ggml.tokens'), dict) else
        len(metadata.get('tokenizer.ggml.tokens') or []) or None,
        'layers': {'block_count': n_layers, 'target': len(target_layers), 'mtp': mtp_layers,
                   'full_attention': [i for i in target_full],
                   'recurrent': len(recurrent_target),
                   'full_attention_interval': _meta(metadata, arch, 'full_attention_interval')},
        'attention': {'heads': head_count, 'kv_heads': head_kv if not isinstance(head_kv, list)
                      else sorted(set(head_kv)), 'key_length': key_len, 'value_length': value_len,
                      'mla_kv_lora_rank': kv_lora, 'sliding_window': sliding},
        'kv': {'bytes_per_token_f16': kv_target, 'mtp_bytes_per_token_f16': kv_mtp,
               'gib_f16': {str(c): round((kv_target + kv_mtp) * c / GIB, 3) for c in (16384, 32768, 65536)},
               'gib_q8_0_k_and_v': {str(c): round((kv_target + kv_mtp) * c * 34 / 64 / GIB, 3)
                                    for c in (16384, 32768, 65536)},
               'upper_bound': bool(sliding)},
        'recurrent_state_bytes': recurrent_bytes,
        'experts': {'count': expert_count, 'used': expert_used, 'shared': expert_shared or
                    (1 if has_shared_tensors else 0), 'types': expert_types},
        'mtp': {'nextn_predict_layers': n_nextn, 'layers': mtp_layers, 'bytes': mtp_bytes},
        'inventory': inventory,
        'memory': {
            'stock_build': {'anonymous_bytes': totals['stock']['anon'], 'file_backed_bytes': totals['stock']['file']},
            'q3k_patch_build': {'anonymous_bytes': totals['q3k_patch']['anon'],
                                'file_backed_bytes': totals['q3k_patch']['file']},
        },
        'decode': {'active_bytes_per_token': int(active_bytes),
                   'roofline_tok_s_at_116gbs': round(READ_BANDWIDTH_GBS * 1e9 / active_bytes, 2) if active_bytes else None,
                   'kv_read_bytes_per_token_at_16k': kv_target * 16384,
                   'recurrent_state_bytes_per_token': recurrent_bytes},
        'chat_template': {'present': template is not None,
                          'sha256': hashlib.sha256(template.encode()).hexdigest() if template else None,
                          'enable_thinking': bool(template and 'enable_thinking' in template),
                          'tools': bool(template and 'tools' in template)},
    }
    result['context_fit'] = context_fit(result)
    result['warnings'] = warnings(result)
    result['suggested_profile'] = suggest_profile(result)
    result['onboarding_hints'] = {
        'context_fits_28gib_cap': {k: v['fits'] for k, v in result['context_fit'].items()},
        'mtp_screen': ('draft-mtp n_max 1/2/3 vs spec-type none' if result['mtp']['nextn_predict_layers']
                       else 'no embedded MTP head'),
        'cache_ram_mib': 0,
        'cache_ram_note': 'production keeps --cache-ram 0 until the cache-keep work is qualified',
    }
    return result


def context_fit(info, checkpoints=4, ubatch=256, overhead=768 * 1024 ** 2):
    """Formula estimate of the steady working set against the 28 GiB service cap."""
    heads = info['attention']['heads']
    heads = max(heads) if isinstance(heads, list) else int(heads or 0)
    fa_off_scores_per_ctx = heads * ubatch * 4
    rows = {}
    for ctx in (16384, 32768, 65536, 131072):
        kv = (info['kv']['bytes_per_token_f16'] + info['kv']['mtp_bytes_per_token_f16']) * ctx
        total = (info['file_bytes'] + kv + info['recurrent_state_bytes'] * (1 + checkpoints)
                 + fa_off_scores_per_ctx * ctx + overhead)
        rows[str(ctx)] = {'estimate_gib': round(total / GIB, 2),
                          'fits': total + CAP_MARGIN_BYTES <= MEMORY_CAP_BYTES}
    return rows


def warnings(info):
    out = []
    home = Path.home()
    path = Path(info['path'])
    if str(path).startswith(str(home / 'Work') + os.sep):
        out.append('model lives under ~/Work: move it to ~/.local/share/ai-stack/models/llm/ before any profile points at it')
    elif not str(path).startswith(str(MODEL_HOME) + os.sep):
        out.append('model is outside ~/.local/share/ai-stack/models/llm/')
    if info['llama_cpp_arch_known'] is False:
        out.append(f"architecture {info['architecture']!r} is not in the pinned llama.cpp llama-arch.cpp")
    expert_types = info['experts']['types']
    if 'Q3_K' in expert_types:
        out.append('Q3_K experts: repacked only with patches/llama-q3k-repack.patch (+anonymous RAM); stock build runs vec_dot')
    slow = sorted(set(expert_types) & SLOW_EXPERT_TYPES)
    if slow:
        out.append('experts in types without an ARM repack fast path: ' + ', '.join(slow))
    if info['file_bytes'] + CAP_MARGIN_BYTES > MEMORY_CAP_BYTES:
        out.append('weights alone exceed the 28 GiB service cap minus margin: classic profile cannot be resident')
    elif not info['context_fit']['16384']['fits']:
        out.append('estimated 16K working set exceeds the 28 GiB cap minus margin')
    if not info['license']:
        out.append('no general.license in the header: check the publisher card')
    if not info['chat_template']['present']:
        out.append('no embedded chat template: --jinja needs one')
    elif not info['chat_template']['enable_thinking']:
        out.append('template has no enable_thinking switch: drop --chat-template-kwargs/--reasoning-budget or verify behaviour')
    if info['kv']['upper_bound']:
        out.append('sliding-window attention present: KV figures are an upper bound')
    if info['mtp']['nextn_predict_layers'] == 0:
        out.append('no embedded MTP head: spec-type none (separate draft files are a different, unqualified path)')
    return out


def slug(info):
    stem = Path(info['shards'][0]).name
    stem = re.sub(r'(-\d{5}-of-\d{5})?\.gguf$', '', stem)
    return re.sub(r'[^a-z0-9.]+', '-', stem.lower()).strip('-')[:48] + '-local'


def suggest_profile(info, runtime_profile=None):
    """Runtime.json profile skeleton with the memlat-qualified starting settings."""
    moe = info['experts']['count'] > 0
    mtp = info['mtp']['nextn_predict_layers'] > 0
    spec = ['--spec-type', 'draft-mtp', '--spec-draft-n-max', '2'] if mtp else ['--spec-type', 'none']
    kwargs = ['--chat-template-kwargs', '{"enable_thinking":false}'] if info['chat_template']['enable_thinking'] else []
    args = list(STANDARD_ARGS)
    if not info['chat_template']['enable_thinking']:
        i = args.index('--reasoning-budget')
        del args[i:i + 2]
    memlat = ({'threads': 8, 'cpu_mask': '0xff', 'cpu_strict': 1} if moe else {'threads': 6})
    memlat.update(threads_batch=12, cpu_mask_batch='0xfff', cpu_strict_batch=1,
                  small_batch_limit=3 if mtp else 1)
    memlat.update({'spec_type': 'draft-mtp', 'spec_draft_n_max': 2} if mtp else {'spec_type': 'none'})
    base = runtime_profile or {}
    return {
        'ready': False,
        'model_alias': slug(info),
        'display_name': info['name'] or slug(info),
        'model_path': info['path'],
        'sha256': base.get('sha256'),
        'expected_bytes': info['file_bytes'],
        'server_binary': base.get('server_binary'),
        'server_sha256': base.get('server_sha256'),
        'context_size': 16384,
        'threads': 4,
        'gpu_layers': 0,
        'backend': 'cpu',
        'extra_args': args + kwargs + spec,
        'environment': {'LLAMA_CPU_SMALL_BATCH_LIMIT': '3' if mtp else '1'},
        'prefix_cache': {'enabled': False, 'directory': str(PREFIX_DIR), 'max_bytes': 3 * GIB},
        'memlat_overrides': memlat,
        'notes': ('Starting point from tools/model_inspect.py. Stock-boot values are the old 4/10 baseline; '
                  'memlat_overrides are the qualified MoE (8 packed/12) or dense (6/12) winners. '
                  'Screen MTP on vs off; keep --cache-ram 0 until cache-keep is qualified. '
                  'ready stays false until the onboarding gates pass.'),
    }


# ---------------------------------------------------------------- output

def _gib(n):
    return f'{n / GIB:.2f} GiB'


def summary(info):
    lay, att, kv, ex = info['layers'], info['attention'], info['kv'], info['experts']
    mem, dec = info['memory'], info['decode']
    lines = [
        f"{Path(info['path']).name}" + (f" (+{len(info['shards']) - 1} shards)" if len(info['shards']) > 1 else ''),
        f"  name: {info['name']}   arch: {info['architecture']}   llama.cpp pinned: "
        f"{ {True: 'known', False: 'UNKNOWN', None: 'not checked'}[info['llama_cpp_arch_known']] }",
        f"  file: {info['file_bytes']:,} B ({_gib(info['file_bytes'])}), header {info['header_bytes']:,} B, "
        f"GGUF v{info['gguf_version']}, parser {info['parser']}, tensor data read: no",
        f"  license: {info['license']}   trained context: {info['context_length_trained']}",
        f"  layers: {lay['block_count']} blocks = {lay['target']} target + {len(lay['mtp'])} MTP; "
        f"{len(lay['full_attention'])} full-attention, {lay['recurrent']} linear/recurrent"
        + (f" (interval {lay['full_attention_interval']})" if lay['full_attention_interval'] else ''),
        f"  attention: {att['heads']} heads, {att['kv_heads']} KV heads, head dim {att['key_length']}/{att['value_length']}",
        f"  KV f16: {kv['bytes_per_token_f16']:,} B/token (+{kv['mtp_bytes_per_token_f16']:,} MTP); "
        + ', '.join(f"{int(c) // 1024}K {g:.2f} GiB" for c, g in kv['gib_f16'].items())
        + (' (upper bound)' if kv['upper_bound'] else ''),
        f"  recurrent state: {info['recurrent_state_bytes'] / 1024 ** 2:.1f} MiB (x5 with 4 checkpoints)",
    ]
    if ex['count']:
        lines.append(f"  experts: {ex['count']} routed, top-{ex['used']}, shared {ex['shared']}; expert types: "
                     + ', '.join(f"{k} x{v['tensors']} {_gib(v['bytes'])}" for k, v in
                                 sorted(ex['types'].items(), key=lambda kv_: -kv_[1]['bytes'])))
    else:
        lines.append('  experts: none (dense)')
    mtp = info['mtp']
    lines.append(f"  MTP: {mtp['nextn_predict_layers']} nextn layer(s) {mtp['layers']}, {_gib(mtp['bytes'])}"
                 if mtp['nextn_predict_layers'] else '  MTP: none embedded')
    lines.append('  type inventory (all / experts / repack on this CPU):')
    for name, row in sorted(info['inventory'].items(), key=lambda kv_: -kv_[1]['bytes']):
        lines.append(f"    {name:8s} {row['tensors']:5d} tensors {_gib(row['bytes']):>11s}  experts "
                     f"{_gib(row['expert_bytes']):>11s}  repack {row['repack']}")
    lines.append(f"  RAM, stock build: anonymous (repacked) {_gib(mem['stock_build']['anonymous_bytes'])}, "
                 f"file-backed mmap {_gib(mem['stock_build']['file_backed_bytes'])}")
    if mem['q3k_patch_build'] != mem['stock_build']:
        lines.append(f"  RAM, Q3_K patch: anonymous {_gib(mem['q3k_patch_build']['anonymous_bytes'])}, "
                     f"file-backed {_gib(mem['q3k_patch_build']['file_backed_bytes'])}")
    lines.append(f"  decode: {dec['active_bytes_per_token'] / 1e9:.3f} GB weights/token -> roofline "
                 f"{dec['roofline_tok_s_at_116gbs']} tok/s at {READ_BANDWIDTH_GBS:.0f} GB/s "
                 f"(+{dec['kv_read_bytes_per_token_at_16k'] / 1e9:.3f} GB KV/token at a full 16K context)")
    lines.append('  working set vs 28 GiB cap: ' + ', '.join(
        f"{int(c) // 1024}K {v['estimate_gib']} GiB {'ok' if v['fits'] else 'NO'}" for c, v in info['context_fit'].items()))
    prof = info['suggested_profile']
    lines.append(f"  suggested: alias {prof['model_alias']}, ctx {prof['context_size']}, memlat threads "
                 f"{prof['memlat_overrides']['threads']}/{prof['memlat_overrides']['threads_batch']}, "
                 f"spec {prof['memlat_overrides']['spec_type']}, cache-ram 0")
    for w in info['warnings']:
        lines.append(f'  ! {w}')
    return '\n'.join(lines)


def runtime_models(runtime_path):
    value = json.loads(Path(runtime_path).expanduser().read_text())
    seen = {}
    for alias, prof in value.get('profiles', {}).items():
        path = prof.get('model_path')
        if path and path not in seen:
            seen[path] = (alias, prof)
    return seen


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('models', nargs='*', help='GGUF file (any shard of a split model)')
    parser.add_argument('--runtime', help='inspect every model referenced by this runtime.json (read-only)')
    parser.add_argument('--json', action='store_true', help='print the full inspection as JSON')
    parser.add_argument('--profile', action='store_true', help='print only the suggested runtime.json profile')
    parser.add_argument('--backend', choices=('builtin', 'gguf', 'auto'), default='builtin')
    parser.add_argument('--llama-src', default=str(DEFAULT_LLAMA_SRC),
                        help='llama.cpp source tree to check architecture support against')
    args = parser.parse_args(argv)
    targets = [(m, None) for m in args.models]
    if args.runtime:
        targets += [(path, prof) for path, (_, prof) in runtime_models(args.runtime).items()]
    if not targets:
        parser.error('give a GGUF path or --runtime')
    results = []
    for model, prof in targets:
        info = analyse(model, args.backend, args.llama_src)
        if prof is not None:
            info['runtime_alias'] = prof.get('model_alias')
            if prof.get('backend_kind') not in (None, 'llama'):
                info['warnings'].append(f"runtime uses backend {prof['backend_kind']}: classic profile hints do not apply")
        info['suggested_profile'] = suggest_profile(info, prof)
        results.append(info)
    if args.profile:
        out = [r['suggested_profile'] for r in results]
        print(json.dumps(out[0] if len(out) == 1 else out, indent=2))
    elif args.json:
        print(json.dumps(results[0] if len(results) == 1 else results, indent=2))
    else:
        print('\n\n'.join(summary(r) for r in results))
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (GGUFError, OSError) as error:
        print(f'model_inspect: {error}', file=sys.stderr)
        sys.exit(2)
