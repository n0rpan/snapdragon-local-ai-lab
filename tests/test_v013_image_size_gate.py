"""v0.1.3 lane H (F-IMG-SHIP5): the per-size gate tool and Studio's gated routing.

Model-free: tiny generated PNGs and stub "engines" (Python scripts writing those PNGs). Records go
to temp dirs; Studio's config/output/stack paths are temp dirs (HOME/XDG from tests/_isolation).
"""
import importlib.machinery
import importlib.util
import json
import math
import os
from pathlib import Path
import stat
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch
import zlib

from snapdragon_ai_lab.runtime import image_size_gate as gate

SOURCE = Path(__file__).resolve().parents[1] / 'src/snapdragon_ai_lab/runtime/studio'
W = H = 32


def chunk(kind, body):
    return struct.pack('>I', len(body)) + kind + body + struct.pack('>I', zlib.crc32(kind + body))


def paeth(a, b, c):
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    return a if pa <= pb and pa <= pc else b if pb <= pc else c


def encode_png(path, width, height, rgb, alpha=False, filters=(0,)):
    """8-bit RGB/RGBA PNG; row y uses filter filters[y % len(filters)] (exercises the reader)."""
    step = 4 if alpha else 3
    if alpha:
        pixels = bytearray()
        for i in range(0, len(rgb), 3):
            pixels += rgb[i:i + 3] + b'\xff'
    else:
        pixels = bytearray(rgb)
    stride = width * step
    raw = bytearray()
    previous = bytearray(stride)
    for y in range(height):
        row = pixels[y * stride:(y + 1) * stride]
        mode = filters[y % len(filters)]
        out = bytearray(stride)
        for i in range(stride):
            a = row[i - step] if i >= step else 0
            b = previous[i]
            c = previous[i - step] if i >= step else 0
            predictor = (0, a, b, (a + b) >> 1, paeth(a, b, c))[mode]
            out[i] = (row[i] - predictor) & 255
        raw += bytes([mode]) + out
        previous = row
    header = struct.pack('>IIBBBBB', width, height, 8, 6 if alpha else 2, 0, 0, 0)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', header) + chunk(b'IDAT', zlib.compress(bytes(raw)))
                           + chunk(b'IEND', b''))


def base_image(seed, width=W, height=H):
    return bytearray(((x * 7 + y * 13 + c * 29 + seed * 31) % 150) + 20
                     for y in range(height) for x in range(width) for c in range(3))


def perturbed(rgb, count, delta):
    out = bytearray(rgb)
    for i in range(count):
        j = (i * 37) % len(out)
        out[j] = max(0, min(255, out[j] + delta))
    return out


def expected_psnr(n, squared):
    return 10 * math.log10(n * 255.0 * 255.0 / squared)


class Fixtures(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.oxi = self.root / 'bin/oxi'
        self.oxi.parent.mkdir(parents=True)
        self.oxi.write_bytes(b'stub oxi build 1')
        self.records = self.root / 'records'

    def triple(self, seed, control_delta=1, control_count=10, candidate_delta=40, candidate_count=100,
               width=W, height=H):
        rgb = base_image(seed, width, height)
        paths = []
        for role, image in (('ref', rgb), ('ctl', perturbed(rgb, control_count, control_delta)),
                            ('cand', perturbed(rgb, candidate_count, candidate_delta))):
            path = self.root / f'img/{role}-{seed}.png'
            encode_png(path, width, height, image)
            paths.append(path)
        return (seed, *paths)


class PngAndPsnr(Fixtures):
    def test_reader_decodes_every_filter_rgb_and_rgba(self):
        rgb = base_image(5)
        for alpha in (False, True):
            for filters in ((0,), (1,), (2,), (3,), (4,), (0, 1, 2, 3, 4)):
                with self.subTest(alpha=alpha, filters=filters):
                    path = self.root / 'f.png'
                    encode_png(path, W, H, rgb, alpha, filters)
                    self.assertEqual(gate.read_png_rgb(path), (W, H, bytes(rgb)))

    def test_reader_refuses_unsupported_pngs(self):
        path = self.root / 'gray.png'
        path.write_bytes(b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', 2, 2, 8, 0, 0, 0, 0))
                         + chunk(b'IDAT', zlib.compress(b'\0\0\0\0\0\0')) + chunk(b'IEND', b''))
        with self.assertRaises(gate.GateError):
            gate.read_png_rgb(path)
        (self.root / 'junk.png').write_bytes(b'not a png')
        with self.assertRaises(gate.GateError):
            gate.read_png_rgb(self.root / 'junk.png')

    def test_psnr_is_rgb_normalised(self):
        a = bytes(base_image(1))
        b = bytes(perturbed(a, 100, 40))
        self.assertAlmostEqual(gate.psnr_rgb(a, b), expected_psnr(len(a), 100 * 40 * 40), places=9)
        self.assertEqual(gate.psnr_rgb(a, a), math.inf)
        self.assertEqual(gate.gate_db(35.4068), 35.4068 - 6)
        self.assertEqual(gate.gate_db(37.5917), 30.0)
        self.assertEqual(gate.gate_db(math.inf), 30.0)


class Judge(Fixtures):
    def test_pass_record_is_atomic_private_and_free_of_paths(self):
        triples = [self.triple(seed) for seed in (42, 43, 44)]
        record = gate.judge(W, H, self.oxi, triples, {'threads': 12, 'chunk1': '32', 'attn_prec': 'f32'},
                            self.records)
        self.assertEqual(record['verdict'], 'PASS')
        path = self.records / f'{W}x{H}.json'
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
        self.assertEqual(sorted(p.name for p in self.records.iterdir()), [path.name])  # no temp leftovers
        stored = json.loads(path.read_text())
        self.assertEqual(stored, record)
        self.assertEqual(stored['schema'], 'oxi-size-gate/1')
        self.assertEqual(stored['oxi_sha256'], gate.sha256_file(self.oxi))
        self.assertEqual(stored['oxi_settings'], {'threads': 12, 'chunk1': '32', 'attn_prec': 'f32'})
        self.assertEqual([e['seed'] for e in stored['seeds']], [42, 43, 44])
        n = W * H * 3
        entry = stored['seeds'][0]
        self.assertAlmostEqual(entry['n_db'], round(expected_psnr(n, 10), 4))
        self.assertEqual(entry['gate_db'], 30.0)
        self.assertAlmostEqual(entry['p_db'], round(expected_psnr(n, 100 * 1600), 4))
        text = path.read_text()
        self.assertNotIn(str(self.root), text)
        for name in ('ref-42', 'ctl-42', 'cand-42', 'img/'):
            self.assertNotIn(name, text)
        for key in ('created_at', 'tool_version', 'width', 'height'):
            self.assertIn(key, stored)

    def test_one_failing_seed_fails_the_size(self):
        triples = [self.triple(42), self.triple(43, candidate_delta=50), self.triple(44)]
        record = gate.judge(W, H, self.oxi, triples, None, self.records)
        self.assertEqual(record['verdict'], 'FAIL')
        self.assertEqual([e['pass'] for e in record['seeds']], [True, False, True])
        self.assertLess(record['seeds'][1]['p_db'], 30.0)

    def test_gate_follows_a_low_noise_floor(self):
        # N ~ 24.9 dB -> G = N - 6 ~ 18.9 dB, so a 29 dB candidate passes (min(30, N - 6) rule)
        triples = [self.triple(s, control_delta=80, control_count=100, candidate_delta=50) for s in (42, 43, 44)]
        record = gate.judge(W, H, self.oxi, triples, None, self.records)
        entry = record['seeds'][0]
        self.assertAlmostEqual(entry['gate_db'], round(entry['n_db'] - 6, 4), places=3)
        self.assertLess(entry['gate_db'], entry['p_db'])
        self.assertEqual(record['verdict'], 'PASS')

    def test_fewer_than_three_distinct_seeds_never_pass(self):
        record = gate.judge(W, H, self.oxi, [self.triple(42), self.triple(43)], None, self.records)
        self.assertEqual(record['verdict'], 'FAIL')
        self.assertIn('distinct seeds', record['reason'])
        record = gate.judge(W, H, self.oxi, [self.triple(42)] * 3, None, self.records)
        self.assertEqual(record['verdict'], 'FAIL')

    def test_wrong_size_image_writes_nothing(self):
        triples = [self.triple(42, width=16, height=16)]
        with self.assertRaises(gate.GateError):
            gate.judge(W, H, self.oxi, triples, None, self.records)
        self.assertFalse(self.records.exists())

    def test_a_later_fail_replaces_an_earlier_pass(self):
        gate.judge(W, H, self.oxi, [self.triple(s) for s in (42, 43, 44)], None, self.records)
        gate.judge(W, H, self.oxi, [self.triple(s, candidate_delta=60) for s in (42, 43, 44)], None, self.records)
        self.assertEqual(json.loads((self.records / f'{W}x{H}.json').read_text())['verdict'], 'FAIL')

    def test_cli_judge_exit_codes(self):
        args = ['judge', '--width', str(W), '--height', str(H), '--oxi', str(self.oxi), '--record-dir', str(self.records)]
        for seed in (42, 43, 44):
            args += ['--seed', *map(str, self.triple(seed))]
        with patch('sys.stdout'):
            self.assertEqual(gate.main(args), 0)
        bad = args[:9] + ['--seed', '42', str(self.root / 'missing.png'), str(self.root / 'x'), str(self.root / 'y')]
        with patch('sys.stdout'), patch('sys.stderr') as err:
            self.assertEqual(gate.main(bad), 2)
        written = ''.join(str(c.args[0]) for c in err.write.call_args_list)
        self.assertNotIn(str(self.root), written)

    def test_default_record_dir_matches_studio(self):
        with patch.dict(os.environ, {'LOCAL_IMAGE_CONFIG_DIR': str(self.root / 'cfg')}):
            os.environ.pop('OXI_SIZE_GATE_DIR', None)
            self.assertEqual(gate.default_record_dir(), self.root / 'cfg/oxi-size-gates')
            with patch.dict(os.environ, {'OXI_SIZE_GATE_DIR': str(self.root / 'override')}):
                self.assertEqual(gate.default_record_dir(), self.root / 'override')

    def test_parity_gap_note_is_present(self):
        doc = gate.__doc__
        for phrase in ('parity-emulation gaps', 'SIMD summation order', '0.037 %', 'round-half-even',
                       'F16 K/V', 'sage attention', 'KleidiAI', 'one-ULP noise floor'):
            self.assertIn(phrase, doc)


STUB_ENGINE = '''import math, struct, sys, zlib
role, seed, out, width, height = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
print('echo of a private prompt', flush=True)
if role == 'broken':
    sys.exit(3)
rgb = bytearray(((x * 7 + y * 13 + c * 29 + seed * 31) %% 150) + 20
                for y in range(height) for x in range(width) for c in range(3))
count, delta = {'reference': (0, 0), 'control': (10, 1), 'candidate': (100, %(cand_delta)d)}[role]
for i in range(count):
    j = (i * 37) %% len(rgb)
    rgb[j] = max(0, min(255, rgb[j] + delta))
raw = b''.join(b'\\0' + bytes(rgb[y * width * 3:(y + 1) * width * 3]) for y in range(height))
def chunk(kind, body):
    return struct.pack('>I', len(body)) + kind + body + struct.pack('>I', zlib.crc32(kind + body))
open(out, 'wb').write(b'\\x89PNG\\r\\n\\x1a\\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0))
                      + chunk(b'IDAT', zlib.compress(raw)) + chunk(b'IEND', b''))
'''


class Run(Fixtures):
    def engine(self, cand_delta=40):
        path = self.root / 'bin/engine'
        path.write_text(f'#!{sys.executable}\n' + STUB_ENGINE % {'cand_delta': cand_delta})
        path.chmod(0o700)
        return path

    def recipe(self, engine, **extra):
        template = lambda role: [str(engine), role, '{seed}', '{out}', '{width}', '{height}']
        return {'width': W, 'height': H, 'oxi_bin': str(self.oxi), 'reference': template('reference'),
                'control': template('control'), 'candidate': template('candidate'), **extra}

    def test_recipe_renders_judges_and_writes_the_record(self):
        record = gate.run(self.recipe(self.engine()), self.records)
        self.assertEqual(record['verdict'], 'PASS')
        self.assertEqual([e['seed'] for e in record['seeds']], [42, 43, 44])
        self.assertTrue((self.records / f'{W}x{H}.json').is_file())
        self.assertNotIn('private prompt', (self.records / f'{W}x{H}.json').read_text())

    def test_recipe_fail_and_cli_exit_status(self):
        recipe = self.root / 'recipe.json'
        recipe.write_text(json.dumps(self.recipe(self.engine(cand_delta=60))))
        with patch('sys.stdout') as out:
            self.assertEqual(gate.main(['run', '--recipe', str(recipe), '--record-dir', str(self.records)]), 1)
        printed = ''.join(str(c.args[0]) for c in out.write.call_args_list)
        self.assertIn('FAIL', printed)
        self.assertNotIn('private prompt', printed)

    def test_command_line_templates(self):
        engine = self.engine()
        args = ['run', '--width', str(W), '--height', str(H), '--oxi', str(self.oxi), '--seeds', '42,43,44',
                '--record-dir', str(self.records)]
        for role in ('reference', 'control', 'candidate'):
            args += [f'--{role}-cmd', f'{engine} {role} {{seed}} {{out}} {{width}} {{height}}']
        with patch('sys.stdout'):
            self.assertEqual(gate.main(args), 0)

    def test_failed_render_writes_no_record(self):
        engine = self.engine()
        recipe = self.recipe(engine)
        recipe['control'] = [str(engine), 'broken', '{seed}', '{out}', '{width}', '{height}']
        with self.assertRaises(gate.GateError) as caught:
            gate.run(recipe, self.records)
        self.assertIn('exited 3', str(caught.exception))
        self.assertNotIn(str(self.root), str(caught.exception))
        self.assertFalse(self.records.exists())


class StudioRouting(Fixtures):
    def setUp(self):
        super().setUp()
        environment = {'IMG_OUTDIR': str(self.root / 'output'), 'LOCAL_IMAGE_CONFIG_DIR': str(self.root / 'config'),
                       'LOCAL_AI_LEASE': str(self.root / 'inference.lock'), 'SD_HOME': str(self.root / 'stack')}
        with patch.dict(os.environ, environment):
            for key in ('OXI_BIN', 'SD_CLI_PROBE', 'OXI_COND_CACHE', 'OXI_WORK', 'SD_CLI', 'OXI_SIZE_GATE_DIR'):
                os.environ.pop(key, None)
            loader = importlib.machinery.SourceFileLoader('v013_gate_studio', str(SOURCE))
            spec = importlib.util.spec_from_loader(loader.name, loader)
            self.studio = importlib.util.module_from_spec(spec)
            loader.exec_module(self.studio)
        s = self.studio
        self.assertEqual(s.OXI_GATE_DIR, self.root / 'config/oxi-size-gates')
        for path in (s.KLEIN_FILES['q8'], s.KLEIN_LLM, s.KLEIN_VAE):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'stub')
        s.PROJECT.mkdir(parents=True, exist_ok=True)
        for binary in (s.OXI_BIN, s.SD_PROBE_CLI):
            binary.parent.mkdir(parents=True, exist_ok=True)
            binary.write_text('#!/bin/sh\nexit 0\n')
            binary.chmod(0o700)

    def record(self, width, height, verdict='PASS', sha=None, **extra):
        s = self.studio
        record = {'schema': 'oxi-size-gate/1', 'width': width, 'height': height, 'verdict': verdict,
                  'oxi_sha256': sha or gate.sha256_file(s.OXI_BIN),
                  'seeds': [{'seed': x, 'pass': verdict == 'PASS'} for x in (42, 43, 44)], **extra}
        s.OXI_GATE_DIR.mkdir(parents=True, exist_ok=True)
        (s.OXI_GATE_DIR / f'{width}x{height}.json').write_text(json.dumps(record))

    def route(self, width, height, setting='auto'):
        return self.studio.choose_t2i_engine(setting, 'klein-q8', width, height, [])

    def test_no_record_keeps_sd_cpp_with_todays_reason(self):
        self.assertEqual(self.route(768, 768),
                         ('sdcpp', 'oxi fast path is 512 × 512 only; 768 × 768 runs on sd.cpp (slower)'))
        self.assertEqual(self.route(1024, 1024)[0], 'sdcpp')

    def test_512_is_unchanged(self):
        self.assertEqual(self.route(512, 512), ('oxi', 'oxi fast path (512 × 512)'))
        self.record(512, 512, verdict='FAIL')
        self.assertEqual(self.route(512, 512), ('oxi', 'oxi fast path (512 × 512)'))

    def test_pass_with_matching_sha_routes_to_oxi(self):
        for width, height in ((768, 768), (1024, 1024), (768, 512)):
            self.record(width, height)
            engine, reason = self.route(width, height)
            self.assertEqual(engine, 'oxi', (width, height))
            self.assertIn('passed its size gate', reason)
            self.assertEqual(self.route(width, height, 'oxi')[0], 'oxi')
            self.assertEqual(self.route(width, height, 'sdcpp')[0], 'sdcpp')
        self.assertEqual(self.route(512, 768)[0], 'sdcpp')   # exact width x height only
        self.assertEqual(self.studio.choose_t2i_engine('auto', 'klein-q4', 768, 768, [])[0], 'sdcpp')
        self.assertEqual(self.studio.choose_t2i_engine('auto', 'klein-q8', 768, 768, ['oxi'])[0], 'sdcpp')

    def test_record_written_by_the_gate_tool_is_honoured(self):
        s = self.studio
        triples = [self.triple(seed, width=48, height=32) for seed in (42, 43, 44)]
        gate.judge(48, 32, s.OXI_BIN, triples, {'threads': 12, 'chunk1': '32', 'chunk2': '32', 'attn_prec': 'f32',
                                                 'vae_conv': 'f32'}, s.OXI_GATE_DIR)
        self.assertEqual(self.route(48, 32)[0], 'oxi')
        self.assertEqual(s.engine_info()['oxi_gated_sizes'], ['48x32'])

    def test_wrong_sha_fail_and_malformed_records_keep_sd_cpp(self):
        s = self.studio
        self.record(768, 768, sha='0' * 64)
        self.assertEqual(self.route(768, 768)[0], 'sdcpp')
        self.record(768, 768, verdict='FAIL')
        self.assertEqual(self.route(768, 768)[0], 'sdcpp')
        for text in ('{broken', '[]', '"PASS"', json.dumps({'verdict': 'PASS'}), ''):
            (s.OXI_GATE_DIR / '768x768.json').write_text(text)
            self.assertEqual(self.route(768, 768)[0], 'sdcpp', text)
        good = {'schema': 'oxi-size-gate/1', 'width': 768, 'height': 768, 'verdict': 'PASS',
                'oxi_sha256': gate.sha256_file(s.OXI_BIN), 'seeds': [{'seed': 42, 'pass': True}]}
        for change in ({'width': 1024}, {'width': '768'}, {'schema': 'other/1'}, {'seeds': []},
                       {'seeds': [{'seed': 42, 'pass': False}]}, {'seeds': 'all'}, {'oxi_sha256': 'ABC'},
                       {'oxi_settings': 'x'}, {'oxi_settings': {'vae_conv': 'int8'}},
                       {'oxi_settings': {'attn_prec': 'f16'}}, {'oxi_settings': {'chunk1': 64}}):
            with self.subTest(change=change):
                (s.OXI_GATE_DIR / '768x768.json').write_text(json.dumps(dict(good, **change)))
                self.assertEqual(self.route(768, 768)[0], 'sdcpp')
        (s.OXI_GATE_DIR / '768x768.json').write_text(json.dumps(good))
        self.assertEqual(self.route(768, 768)[0], 'oxi')
        self.assertEqual(s.engine_info()['oxi_gated_sizes'], ['768x768'])

    def test_record_must_be_an_owned_regular_file(self):
        s = self.studio
        self.record(768, 768)
        target = s.OXI_GATE_DIR / 'elsewhere.json'
        (s.OXI_GATE_DIR / '768x768.json').rename(target)
        (s.OXI_GATE_DIR / '768x768.json').symlink_to(target)
        self.assertEqual(self.route(768, 768)[0], 'sdcpp')

    def test_a_replaced_oxi_binary_drops_the_route(self):
        s = self.studio
        self.record(768, 768)
        self.assertEqual(self.route(768, 768)[0], 'oxi')
        s.OXI_BIN.write_text('#!/bin/sh\n# rebuilt\nexit 0\n')
        self.assertEqual(self.route(768, 768)[0], 'sdcpp')
        s.OXI_BIN.unlink()
        self.assertEqual(self.route(768, 768)[0], 'sdcpp')

    def test_gated_plan_runs_oxi_at_its_size_on_the_cpu_only(self):
        s = self.studio
        self.record(768, 768)
        plan = s.plan_generation({'prompt': 'x', 'model': 'klein-q8', 'width': 768, 'height': 768})
        self.assertEqual(plan['engine']['engine'], 'oxi')
        self.assertEqual(plan['steps'][0]['expect_size'], (768, 768))
        args = s.oxi_generate_args({'width': 768, 'height': 768}, 42, s.OUT / 'x.png')
        self.assertEqual((args[args.index('--width') + 1], args[args.index('--height') + 1]), ('768', '768'))
        # an NPU-enabled install still takes the CPU lease for a gated size (NPU profile is 512 px)
        with patch.object(s, 'oxi_npu_settings', return_value=(True, '/owned/profile.json')), \
                patch.object(s.image_slot, 'connect') as connect, \
                patch.object(s, 'claim_generation_slot', return_value=True), \
                patch.object(s, 'run_pipeline'):
            job_id, error = s.start_job(plan)
        self.assertIsNone(error)
        connect.assert_not_called()
        self.assertGreater(s.oxi_estimate_seconds(1, 768, 768), s.oxi_estimate_seconds(1))
        self.assertEqual(s.oxi_estimate_seconds(1), s.oxi_estimate_seconds(1, 512, 512))


if __name__ == '__main__':
    unittest.main()
