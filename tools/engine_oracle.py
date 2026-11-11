#!/usr/bin/env python3
"""Private synthetic oracles; run under the existing roofline lease/memory guard.

Reads installed model/profile metadata only; never reads history or user prompts.
The binary record format is shared with snapdragon-engine's `oxe gate --ref`.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import threading

PRODUCTION_SHA = 'acd47ff19a161f9303be23ae1e99a80050579dad17071302ba77b3a0d6060efc'
MAX_ARTIFACT_BYTES = 512 * 1024**2
FIXTURES = ('code', 'math', 'json')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024**2), b''): h.update(block)
    return h.hexdigest()


def tensor_index(path):
    """Validate the bounded LE record stream without decoding activation values."""
    result = []
    size = Path(path).stat().st_size
    if size > MAX_ARTIFACT_BYTES: raise ValueError('tensor_file_too_large')
    with Path(path).open('rb') as f:
        while f.tell() < size:
            raw = f.read(4)
            if len(raw) != 4: raise ValueError('truncated_tensor_name_length')
            n, = struct.unpack('<I', raw)
            if not 1 <= n <= 256: raise ValueError('invalid_tensor_name_length')
            name = f.read(n).decode('utf-8')
            raw = f.read(8)
            if len(raw) != 8: raise ValueError('truncated_tensor_header')
            dtype, dims = struct.unpack('<II', raw)
            if dtype not in (0, 1) or not 1 <= dims <= 4: raise ValueError('invalid_tensor_type_or_dims')
            raw = f.read(8*dims)
            if len(raw) != 8*dims: raise ValueError('truncated_tensor_shape')
            shape = list(struct.unpack('<'+'q'*dims, raw))
            elements = 1
            for d in shape:
                if d < 0: raise ValueError('invalid_tensor_shape')
                elements *= d
            payload = elements * 4
            if payload > MAX_ARTIFACT_BYTES or f.tell()+payload > size: raise ValueError('truncated_tensor_payload')
            result.append({'name': name, 'type': dtype, 'shape': shape, 'offset': f.tell(), 'bytes': payload})
            if len(result) > 20000: raise ValueError('too_many_tensor_records')
            f.seek(payload, 1)
    return result


def validate_artifact(path):
    path = Path(path)
    m = json.loads((path/'manifest.json').read_text())
    if m.get('complete') is not True or not m.get('tokens') or not m.get('generated'):
        raise ValueError('incomplete_oracle')
    expected = m['n_eval'] * m['n_vocab'] * 4
    if (path/'logits.f32').stat().st_size != expected: raise ValueError('wrong_logit_row_count')
    total = sum(f.stat().st_size for f in path.iterdir() if f.is_file())
    if total > MAX_ARTIFACT_BYTES: raise ValueError('artifact_budget_exceeded')
    index = {}
    for file in sorted(path.glob('*.tensors')):
        index[file.name] = tensor_index(file)
    layers = sorted({int(match[1]) for records in index.values() for row in records
                     if (match := re.fullmatch(r'l_out-(\d+)', row['name']))})
    return {'tokens': len(m['tokens']), 'generated': len(m['generated']),
            'logit_rows': m['n_eval'], 'logits_sha256': sha(path/'logits.f32'),
            'layers_captured': layers, 'tensor_records': sum(map(len, index.values())),
            'bytes': total, 'index': index}


def run(args):
    os.umask(0o077)
    out = args.output.expanduser().absolute()
    root = (Path.home()/'.cache/local-ai/engine-oracle').resolve()
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    if out.exists() or root not in out.parents or any(p.is_symlink() for p in (out, *out.parents)):
        raise ValueError('fresh_private_engine_oracle_path_required')
    if root.stat().st_uid != os.getuid() or root.stat().st_mode & 0o077:
        raise ValueError('private_oracle_root_required')
    out.mkdir(mode=0o700, parents=True)
    binary = args.binary.resolve()
    binary_sha = sha(binary)
    runtime = json.loads(args.runtime.read_text())
    alias = {'ornith': 'ornith-35b-a3b-local', 'qwen': 'qwen3.6-35b-a3b-local'}[args.model]
    profile = {**runtime, **runtime['profiles'][alias], 'model_alias': alias}
    source = Path(__file__).resolve().parents[1]/'src/snapdragon_ai_lab/runtime'
    sys.path.insert(0, str(source))
    import profile_policy
    profile = profile_policy.resolve(profile)
    model = Path(profile['model_path'])
    before = model.stat()
    if before.st_size != profile['expected_bytes']: raise ValueError('model_size_changed')
    environment = dict(os.environ, GGML_CPU_REPACK_Q3K='1', GGML_CPU_MOE_GEMM='1',
                       GGML_CPU_REPACK_Q3K_THREADS='8', LLAMA_CPU_SMALL_BATCH_LIMIT='3',
                       GGML_CPU_REPACK_CACHE=str(root/'repack'), GGML_CPU_REPACK_CACHE_VERIFY='once',
                       GGML_CPU_REPACK_CACHE_MODEL_SHA256=profile['sha256'],
                       GGML_CPU_REPACK_CACHE_ENGINE_ID=binary_sha, GGML_CPU_REPACK_DROP_SOURCE='1')
    rows = []
    for fixture in (args.fixture,):
        pair = []
        for capture in (not args.no_activations,):
            dest = out/(fixture+('-captured' if capture else '-control'))
            command = [str(binary), '--model', str(model), '--out', str(dest), '--fixture', fixture,
                       '--threads', '8', '--threads-batch', '12', '--ctx', '4096', '--gen', '8', '--logits']
            if capture: command += ['--dump-pos', '0-1', '--max-elems', '65536']
            with (out/(dest.name+'.log')).open('x') as log:
                child = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                         env=environment, text=True)
                def drain():
                    for line in child.stderr:
                        log.write(line)
                        if line.strip() == 'ORACLE_MODEL_READY' and args.ready_log:
                            args.ready_log.write_text('LOCAL_AI_MODEL_READY\n')
                reader = threading.Thread(target=drain, daemon=True); reader.start()
                try:
                    child.wait(timeout=300)
                finally:
                    if child.poll() is None:
                        child.terminate()
                        try: child.wait(timeout=3)
                        except subprocess.TimeoutExpired: child.kill(); child.wait()
                    reader.join(timeout=5)
                if child.returncode or reader.is_alive(): raise RuntimeError('oracle_child_failed_or_reader_alive')
            manifest = json.loads((dest/'manifest.json').read_text())
            details = validate_artifact(dest)
            if capture and details['layers_captured'] != list(range(40)):
                raise ValueError('missing_layer_activations')
            (dest/'tensor-index.json').write_text(json.dumps(details.pop('index'), indent=2)+'\n')
            details.update(path=str(dest), fixture=fixture, capture=capture,
                           generated_ids=manifest['generated'])
            pair.append(details)
        exact = None
        rows.append({'fixture': fixture, 'capture_preserves_logits_and_greedy': exact, 'artifacts': pair})

    after = model.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise ValueError('model_identity_changed')
    result = {'kind': 'private-production-library-oracles', 'model_alias': alias,
              'model_sha256': profile['sha256'], 'oracle_binary_sha256': binary_sha,
              'reference_library_server_sha256': PRODUCTION_SHA,
              'installed_profile_server_sha256': profile['server_sha256'],
              'numeric_domain': 'production Ornith runtime libraries, KleidiAI ON; MTP OFF, serial prefill',
              'source_fixture_only': True, 'production_changed': False, 'cases': rows, 'passed': True}
    (out/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'kind': result['kind'], 'model_alias': alias, 'fixtures': len(rows),
                      'capture_parity_pending': True, 'private_path': str(out), 'passed': True}), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--binary', type=Path, required=True)
    ap.add_argument('--runtime', type=Path, default=Path.home()/'.config/local-ai/llm/runtime.json')
    ap.add_argument('--model', choices=['ornith', 'qwen'], required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--fixture', choices=FIXTURES, default='code')
    ap.add_argument('--no-activations', action='store_true')
    ap.add_argument('--ready-log', type=Path)
    run(ap.parse_args())


if __name__ == '__main__': main()
