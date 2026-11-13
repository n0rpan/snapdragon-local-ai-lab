"""V013 lane C: context modes (fast/deep/max), the RSS-proven 256K cap, preload and failure fields in
/_local/status, and the controller leftovers (thermal ledger sample, token directory, echo naming,
OpenCode limits per mode). Model-free; isolated HOME/XDG (tests/_isolation.py); no real config path."""
import copy
import hashlib
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from snapdragon_ai_lab.runtime import (controller, harness_registry as hr, launch_local_plan as launcher,
                                       oxe_runtime, profile_policy as pp, state_cache as sc)

import openai_api_stack as stack_module
from openai_api_stack import FakeEngine, Stack, ORNITH, QWEN, FLASH

spec = importlib.util.spec_from_file_location('openai_api_check', Path(__file__).resolve().parents[1] / 'tools/openai_api_check.py')
check = importlib.util.module_from_spec(spec); spec.loader.exec_module(check)
RUNTIME = Path(__file__).resolve().parents[1] / 'src/snapdragon_ai_lab/runtime'
_cli_loader = importlib.machinery.SourceFileLoader('local_ai_cli_v013', str(RUNTIME / 'local-ai'))
cli = importlib.util.module_from_spec(importlib.util.spec_from_loader(_cli_loader.name, _cli_loader))
_cli_loader.exec_module(cli)

GIB = 1024**3
ENGINE = 'b' * 64  # openai_api_stack's server_sha256
PROOF = {'context': 262144, 'peak_rss_bytes': 26 * GIB, 'engine_sha256': ENGINE, 'date': '2026-11-20'}
CONTEXT_KEYS = {'cap', 'mode', 'modes'}
MODE_KEYS = {'id', 'compact_at', 'cap', 'available', 'reason'}
PRELOAD_KEYS = {'state', 'src', 'ms', 'age_s'}


def oxe_config(alias=ORNITH, context=262144, proof=PROOF, **extra):
    value = {'backend_kind': 'oxe', 'model_alias': alias, 'server_binary': '/synthetic/oxe-server',
             'model_path': '/synthetic/m.gguf', 'server_sha256': ENGINE, 'sha256': 'a' * 64, 'threads': 8,
             'context_size': context, 'expected_bytes': 100,
             'oxe_args': ['--model', '{model}', '--alias', '{alias}', '--ctx-size', '{context}'], **extra}
    if proof is not None:
        value['rss_proof'] = copy.deepcopy(proof)
    return value


def rust_profile(alias=ORNITH, context=131072, proof=None):
    """A classic profile with a top-level rust_engine.oxe block (the production shape)."""
    oxe = {'server_binary': '/synthetic/oxe-server', 'server_sha256': ENGINE, 'context_size': context,
           'oxe_args': ['--model', '{model}', '--ctx-size', '{context}']}
    if proof is not None:
        oxe['rss_proof'] = copy.deepcopy(proof)
    return {'model_alias': alias, 'sha256': 'a' * 64, 'context_size': 16384, 'rust_engine': {'oxe': oxe}}


class ModeTable(unittest.TestCase):
    def test_compaction_points_and_caps(self):
        self.assertEqual([hr.compact_at(m) for m in hr.MODES], [49536, 102000, 199750])
        self.assertEqual([hr.MODE_CAP[m] for m in hr.MODES], [131072, 131072, 262144])
        self.assertEqual(hr.DEFAULT_MODE, 'fast')  # O-105 (owner decision): fast by default
        self.assertEqual(hr.MODE_APPLIES, 'next_session')
        with self.assertRaisesRegex(ValueError, 'unsupported_opencode_mode'):
            hr.opencode_limits('medium')

    def test_standalone_launcher_mirrors_the_registry(self):
        # launch_local_plan.py is installed alone in ~/.local/bin, so it carries its own copy
        self.assertEqual(launcher.OPENCODE_MODE_INPUT, hr.MODE_INPUT)
        self.assertEqual(launcher.OPENCODE_MODE_CONTEXT, hr.MODE_CAP)
        self.assertEqual(launcher.MOE_OUTPUT, hr.OPENCODE_OUTPUT)
        self.assertEqual(launcher.MOE_KEEP_TOKENS, hr.OPENCODE_KEEP)
        self.assertNotRegex((RUNTIME / 'launch_local_plan.py').read_text(), r'(?m)^\s*(from\s+\S+\s+)?import\s+harness_registry')

    def test_harness_limits_rendered_per_mode(self):
        for mode in hr.MODES:
            for model in launcher.MOE_PRESET_MODELS:
                with self.subTest(mode=mode, model=model):
                    env = launcher.environment_for_classic_model({}, model, mode)
                    overlay = json.loads(env['OPENCODE_CONFIG_CONTENT'])
                    entry = overlay['providers']['local-qwen']['models'][launcher.CLASSIC_MODELS[model].split('/', 1)[1]]
                    self.assertEqual(entry['limit'], hr.opencode_limits(mode))
                    self.assertEqual(overlay['compaction'], {'auto': True, 'keep': {'tokens': 8000}})

    def test_launcher_takes_the_stored_mode_argument(self):
        launched = {}
        with patch.dict(launcher.os.environ, {}, clear=True), \
                patch.object(launcher.os, 'execvpe', side_effect=lambda *a: launched.update(args=a)):
            launcher.main(['--model', 'ornith', '--mode', 'max', '/synthetic/project'])
        overlay = json.loads(launched['args'][2]['OPENCODE_CONFIG_CONTENT'])
        self.assertEqual(overlay['providers']['local-qwen']['models']['ornith-35b-a3b-local']['limit']['context'], 262144)
        for bad in (['--model', 'ornith', '--mode', 'huge', '/p'], ['--model', 'flash', '--mode', 'deep', '/p'],
                    ['--model', 'moe', '--mode', 'fast', 'relative']):
            with self.subTest(bad=bad), self.assertRaises(ValueError), \
                    patch.object(launcher.os, 'execvpe', side_effect=AssertionError('must not launch')):
                launcher.main(bad)

    def test_cli_code_passes_the_controller_mode(self):
        status = {'selected_model': 'ornith', 'context': {'cap': 131072, 'mode': 'fast', 'modes': []}}
        with patch.object(cli, 'ensure_service', return_value=True), patch.object(cli, 'status', return_value=status), \
                patch.object(cli.subprocess, 'Popen') as launch:
            self.assertEqual(cli.action('code'), 0)
        self.assertEqual(launch.call_args.args[0][-4:], ['ornith', '--mode', 'fast', str(cli.CODE_DIRECTORY)])

    def test_cli_has_no_dense_key_and_a_mode_action(self):
        self.assertEqual(cli.main(['dense']), 2)  # audit S1: the retired key explains and exits 2
        status = {'selected_model': 'moe'}
        with patch.object(cli, 'ensure_service', return_value=True), patch.object(cli, 'status', return_value=status), \
                patch.object(cli, 'request', return_value=(200, {})) as request:
            self.assertEqual(cli.main(['mode', '--mode', 'deep']), 0)
        self.assertEqual(request.call_args.kwargs['payload'], {'model': QWEN, 'mode': 'deep'})
        with patch.object(cli, 'ensure_service', return_value=True):
            self.assertEqual(cli.action('mode', pref='huge'), 2)


class RssProof(unittest.TestCase):
    def test_reason_per_case(self):
        self.assertIsNone(pp.rss_proof_reason(oxe_config()))
        self.assertIsNone(pp.rss_proof_reason(oxe_config(proof=None), 131072))
        self.assertEqual(pp.rss_proof_reason(oxe_config(proof=None)), 'needs RSS proof')
        self.assertEqual(pp.rss_proof_reason(oxe_config(alias=QWEN)), 'Ornith only')
        self.assertEqual(pp.rss_proof_reason(oxe_config(proof=dict(PROOF, engine_sha256='c' * 64))),
                         'RSS proof is for another engine')
        self.assertEqual(pp.rss_proof_reason(oxe_config(proof=dict(PROOF, peak_rss_bytes=27 * GIB + 1))), 'RSS above 27 GiB')
        self.assertEqual(pp.rss_proof_reason(oxe_config(proof=dict(PROOF, context=196608))), 'needs RSS proof')
        self.assertEqual(pp.rss_proof_reason(oxe_config(), 262145), 'above 262144')

    def test_malformed_records_are_refused(self):
        for bad in (dict(PROOF, extra=1), {k: v for k, v in PROOF.items() if k != 'date'}, dict(PROOF, context=131072),
                    dict(PROOF, peak_rss_bytes=True), dict(PROOF, engine_sha256='B' * 64), dict(PROOF, date='20.11.2026')):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, 'oxe_rss_proof_invalid'):
                pp.validate_rss_proof(bad)

    def test_262144_refused_without_proof_and_accepted_with_it(self):
        with self.assertRaisesRegex(ValueError, 'oxe_context_above_131072_needs_rss_proof'):
            pp.resolve(rust_profile(context=262144), oxe_enabled=True)
        with self.assertRaisesRegex(ValueError, 'oxe_context_above_131072_needs_rss_proof'):
            pp.resolve(rust_profile(context=262144, proof=dict(PROOF, engine_sha256='c' * 64)), oxe_enabled=True)
        with self.assertRaisesRegex(ValueError, 'invalid_memlat_context_size'):
            pp.resolve(rust_profile(context=262145, proof=PROOF), oxe_enabled=True)
        resolved = pp.resolve(rust_profile(context=262144, proof=PROOF), oxe_enabled=True)
        self.assertEqual((resolved['context_size'], resolved['rss_proof']), (262144, PROOF))
        with self.assertRaisesRegex(ValueError, 'ornith_only'):
            pp.resolve(rust_profile(alias=QWEN, context=262144, proof=PROOF), oxe_enabled=True)
        # 131072 stays the default and needs nothing; a proof alone does not raise the cap
        self.assertEqual(pp.resolve(rust_profile(proof=PROOF), oxe_enabled=True)['context_size'], 131072)

    def test_launch_validation_follows_the_proof(self):
        oxe_runtime._validate(oxe_config())
        oxe_runtime._validate(oxe_config(context=131072, proof=None))
        for config in (oxe_config(proof=None), oxe_config(alias=QWEN), oxe_config(proof=dict(PROOF, engine_sha256='c' * 64))):
            with self.subTest(config=config.get('rss_proof')), \
                    self.assertRaisesRegex(ValueError, 'oxe_context_above_131072_needs_rss_proof'):
                oxe_runtime._validate(config)
        with self.assertRaisesRegex(ValueError, 'oxe_invalid_integer'):
            oxe_runtime._validate(oxe_config(context=262145))

    def test_apply_mode(self):
        base = oxe_config(context=131072)
        self.assertEqual(pp.apply_mode(base, 'max')['context_size'], 262144)
        self.assertEqual(pp.apply_mode(base, 'fast')['context_size'], 131072)
        stale = oxe_config(context=131072, proof=dict(PROOF, engine_sha256='c' * 64))
        self.assertEqual((pp.apply_mode(stale, 'max')['context_mode'], pp.apply_mode(stale, 'max')['context_size']),
                         ('fast', 131072))  # a stale proof falls back to the default (fast) on the standard cap
        flash = {'model_alias': FLASH, 'context_size': 16384}
        self.assertIs(pp.apply_mode(flash, 'max'), flash)
        self.assertEqual(pp.stored_mode({}, ORNITH), 'fast')
        self.assertEqual(pp.stored_mode({'context_modes': {'default': 'fast', 'models': {QWEN: 'bogus'}}}, QWEN), 'fast')
        written = pp.with_stored_mode({'x': 1}, ORNITH, 'fast')
        self.assertEqual(written, {'x': 1, 'context_modes': {'models': {ORNITH: 'fast'}}})


class StatusContract(unittest.TestCase):
    """POST /_local/mode round-trip and the status shape of V013-PLAN sec. 3, on the stub stack."""

    @classmethod
    def setUpClass(cls):
        cls.engine = FakeEngine()

    @classmethod
    def tearDownClass(cls):
        cls.engine.close()

    def stack(self, **config):
        stack = Stack(self.engine.port, **config)
        self.addCleanup(stack.close)
        return stack

    def status(self, stack):
        code, _, value = check.call(stack.base, 'GET', '/_local/status')
        self.assertEqual(code, 200)
        return value

    def test_status_shape_matches_the_contract(self):
        stack = self.stack()
        value = self.status(stack)
        for key in ('context', 'preload', 'error_at', 'error_origin', 'last_failure'):
            self.assertIn(key, value)
        self.assertEqual(set(value['context']), CONTEXT_KEYS)
        self.assertEqual(set(value['preload']), PRELOAD_KEYS)
        self.assertEqual(json.dumps(value['context'], sort_keys=True), json.dumps({
            'cap': 32768, 'mode': 'fast', 'modes': [  # O-105 default
                {'id': 'fast', 'compact_at': 49536, 'cap': 131072, 'available': True, 'reason': None},
                {'id': 'deep', 'compact_at': 102000, 'cap': 131072, 'available': True, 'reason': None},
                {'id': 'max', 'compact_at': 199750, 'cap': 262144, 'available': False, 'reason': 'needs RSS proof'}]},
            sort_keys=True))
        self.assertEqual(value['preload'], {'state': 'idle', 'src': None, 'ms': None, 'age_s': None})
        self.assertEqual((value['error_at'], value['error_origin'], value['last_failure']), (None, None, None))
        self.assertIn('context_modes', value['capabilities'])
        self.assertIn('preload', value['capabilities'])

    def test_failure_time_and_origin(self):
        stack = self.stack()
        with stack.manager.mu:
            stack.manager.record_failure('No room', kind='start', alias=ORNITH, origin='lab')
        value = self.status(stack)
        self.assertEqual(value['error_origin'], 'lab')
        self.assertEqual(value['error_at'], value['last_failure']['t'])
        self.assertIsNone(value['last_error'])  # lab failures stay out of the user-facing text

    def test_mode_round_trip(self):
        stack = self.stack()
        # O-105: fast is the default, so the round trip goes to deep and back
        code, _, got = check.call(stack.base, 'POST', '/_local/mode', {'model': ORNITH, 'mode': 'deep'})
        self.assertEqual((code, set(got), got['mode']), (200, CONTEXT_KEYS, 'deep'))
        self.assertEqual(json.loads(stack.config_path.read_text())['context_modes'], {'models': {ORNITH: 'deep'}})
        self.assertEqual(self.status(stack)['context']['mode'], 'deep')
        self.assertEqual(stack.manager.config(ORNITH)['context_mode'], 'deep')
        # persisted per model: the other MoE keeps the default
        self.assertEqual(stack.manager.context_block(stack.manager.config(QWEN))['mode'], 'fast')
        code, _, got = check.call(stack.base, 'POST', '/_local/mode', {'model': ORNITH, 'mode': 'fast'})
        self.assertEqual((code, got['mode']), (200, 'fast'))

    def test_max_needs_the_proof(self):
        stack = self.stack()
        code, _, got = check.call(stack.base, 'POST', '/_local/mode', {'model': ORNITH, 'mode': 'max'})
        self.assertEqual((code, got), (409, {'error': 'needs RSS proof'}))
        self.assertNotIn('context_modes', json.loads(stack.config_path.read_text()))
        code, _, got = check.call(stack.base, 'POST', '/_local/mode', {'model': QWEN, 'mode': 'max'})
        self.assertEqual((code, got), (409, {'error': 'Ornith only'}))
        value = json.loads(stack.config_path.read_text())
        value['profiles'][ORNITH]['rss_proof'] = PROOF
        stack.config_path.write_text(json.dumps(value))
        code, _, got = check.call(stack.base, 'POST', '/_local/mode', {'model': ORNITH, 'mode': 'max'})
        self.assertEqual((code, got['mode'], got['cap']), (200, 'max', 262144))
        self.assertTrue(got['modes'][2]['available'])
        self.assertEqual(stack.manager.config(ORNITH)['context_size'], 262144)

    def test_cap_change_unloads_an_idle_loaded_model(self):
        stack = self.stack()
        value = json.loads(stack.config_path.read_text())
        value['profiles'][ORNITH]['rss_proof'] = PROOF
        stack.config_path.write_text(json.dumps(value))
        stack.manager.begin(allow_start=True, model_alias=ORNITH); stack.manager.end()
        self.assertIsNotNone(stack.manager.process)
        check.call(stack.base, 'POST', '/_local/mode', {'model': ORNITH, 'mode': 'fast'})
        self.assertIsNotNone(stack.manager.process)  # same cap: the engine stays
        stack.manager.active = 1
        code, _, got = check.call(stack.base, 'POST', '/_local/mode', {'model': ORNITH, 'mode': 'max'})
        stack.manager.active = 0
        self.assertEqual(code, 409)
        code, _, _ = check.call(stack.base, 'POST', '/_local/mode', {'model': ORNITH, 'mode': 'max'})
        self.assertEqual(code, 200)
        self.assertIsNone(stack.manager.process)  # the next request loads at 262144

    def test_bad_requests(self):
        stack = self.stack()
        for body in ({'model': ORNITH}, {'model': ORNITH, 'mode': 'fast', 'x': 1}, {}):
            self.assertEqual(check.call(stack.base, 'POST', '/_local/mode', body)[0], 400, body)
        for body in ({'model': ORNITH, 'mode': 'huge'}, {'model': FLASH, 'mode': 'deep'}, {'model': 'nope', 'mode': 'deep'},
                     {'model': 7, 'mode': 'deep'}):
            code, _, got = check.call(stack.base, 'POST', '/_local/mode', body)
            self.assertEqual((code, set(got)), (409, {'error'}), body)

    def test_cli_status_allowlists_the_new_fields(self):
        chat = {'context': {'cap': 131072, 'mode': 'deep', 'modes': [
                    {'id': 'max', 'compact_at': 199750, 'cap': 262144, 'available': False, 'reason': 'needs RSS proof\n'}]},
                'preload': {'state': 'done', 'src': 'disk', 'ms': 812.5, 'age_s': 3, 'pos': 9},
                'error_at': 1_800_000_000, 'error_origin': 'organic'}
        result = cli.optional_status(chat, {})
        self.assertEqual(result['context']['modes'][0]['reason'], 'needs RSS proof')
        self.assertEqual(result['preload'], {'state': 'done', 'src': 'disk', 'ms': 812.5, 'age_s': 3})
        self.assertEqual((result['error_at'], result['error_origin']), (1_800_000_000, 'organic'))
        bad = cli.optional_status({'context': {'modes': [{'id': 'huge'}]}, 'preload': {'state': 'x'},
                                   'error_origin': 'evil'}, {})
        self.assertNotIn('context', bad); self.assertNotIn('preload', bad)
        self.assertIsNone(bad['error_origin'])


class Preload(unittest.TestCase):
    def test_engine_status_mapping(self):
        view = controller.preload_view
        self.assertEqual(view({'status': 'started'}), ('running', None, None))
        self.assertEqual(view({'status': 'done', 'pos': 9000, 'ms': 412.25}), ('done', 'disk', 412.2))
        self.assertEqual(view({'status': 'used'}), ('done', 'disk', None))
        self.assertEqual(view({'status': 'live'}), ('done', 'live', None))
        self.assertEqual(view({'status': 'cancelled', 'ms': 3}), ('cancelled', None, 3.0))
        self.assertEqual(view({'status': 'busy'}), ('cancelled', None, None))
        self.assertEqual(view({'status': 'none'}), ('idle', None, None))
        self.assertEqual(view(None), ('unavailable', None, None))
        self.assertEqual(view({'status': 'done', 'ms': float('nan')}), ('done', 'disk', None))
        for state, *_ in (view({'status': s}) for s in ('started', 'done', 'live', 'busy', 'none', 'weird')):
            self.assertIn(state, controller.PRELOAD_STATES)

    def test_status_follows_the_loaded_engine(self):
        manager = controller.Manager(); manager.stop_event.set(); manager.thread.join(2)
        try:
            self.assertEqual(manager.preload_status()['state'], 'idle')
            process = type('P', (), {'poll': lambda self: None, 'pid': None})()
            manager.process = process
            self.assertEqual(manager.preload_status()['state'], 'unavailable')  # no state store
            manager.state_engine = manager.state_preload = True
            self.assertEqual(manager.preload_status()['state'], 'idle')
            manager.note_preload_engine({'status': 'done', 'ms': 50}, process)
            got = manager.preload_status()
            self.assertEqual((got['state'], got['src'], got['ms']), ('done', 'disk', 50.0))
            self.assertIsInstance(got['age_s'], int)
            manager.note_preload('running', object())  # another engine's preload is ignored
            self.assertEqual(manager.preload_status()['state'], 'done')
        finally:
            manager.process = None
            manager.close()


class Leftovers(unittest.TestCase):
    def test_thermal_sample_numbers_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            thermal, cpufreq = Path(tmp) / 'thermal', Path(tmp) / 'cpufreq'
            for name, value in (('thermal_zone0', '45000'), ('thermal_zone1', '71250'), ('thermal_zone2', 'junk'),
                                ('thermal_zone3', '999999'), ('cooling_device0', '99000')):
                (thermal / name).mkdir(parents=True); (thermal / name / 'temp').write_text(value + '\n')
            for name, value in (('policy8', '3417600'), ('policy0', '2956800'), ('policy4', '1200000'), ('policyx', '1')):
                (cpufreq / name).mkdir(parents=True); (cpufreq / name / 'scaling_cur_freq').write_text(value)
            self.assertEqual(controller.thermal_sample(thermal, cpufreq),
                             {'thermal_max_c': 71.2, 'cluster_mhz': [2956, 1200, 3417]})
            self.assertEqual(controller.thermal_sample(Path(tmp) / 'none', Path(tmp) / 'none'),
                             {'thermal_max_c': None, 'cluster_mhz': []})

    def test_temporary_token_directory_is_removed(self):
        token, path = oxe_runtime.new_api_key(8093, {})  # no runtime dir: a fresh mkdtemp directory
        self.assertTrue(path.parent.name.startswith(oxe_runtime.TEMP_KEY_DIR_PREFIX))
        oxe_runtime.remove_api_key(path)
        self.assertFalse(path.parent.exists())
        with tempfile.TemporaryDirectory() as tmp:  # a runtime dir is never removed
            token, path = oxe_runtime.new_api_key(8093, {'XDG_RUNTIME_DIR': tmp})
            oxe_runtime.remove_api_key(path)
            self.assertTrue(path.parent.is_dir())

    def test_echo_stem_matches_the_engine_scheme(self):
        # state_store::hint_hash: first 32 hex of SHA-256; echo.rs group_stem strips "s:"
        self.assertEqual(sc.hint_hash('conv-p'), hashlib.sha256(b'conv-p').hexdigest()[:32])
        self.assertEqual(sc.echo_stem('s:conv-p'), sc.hint_hash('conv-p'))
        self.assertEqual(sc.echo_stem('k:0123'), sc.hint_hash('k:0123'))

    def test_gc_sweeps_only_stale_orphan_echo_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'state'; model = root / ('a' * 16)
            now = time.time()
            (model / 'sessions').mkdir(parents=True); (model / 'echo').mkdir()
            (model / 'identity.json').write_text('{}')
            named = sc.hint_hash('conv')
            (model / 'sessions/s1.json').write_text(json.dumps({'version': 1, 'blocks': [], 'snaps': [], 'atime': now,
                                                                 'n_tokens': 0, 'hint': named}))
            files = {named: now - 30 * 86400, sc.echo_stem('k:old'): now - 30 * 86400, sc.echo_stem('k:new'): now - 60}
            for stem, mtime in files.items():
                path = model / 'echo' / (stem + '.jsonl'); path.write_text('x' * 10); os.utime(path, (mtime, mtime))
            freed = sc.gc(root, 64_000_000_000, 7 * 86400, now)
            self.assertEqual(freed, 10)
            left = sorted(p.stem for p in (model / 'echo').iterdir())
            self.assertEqual(left, sorted([named, sc.echo_stem('k:new')]))

    def test_preload_is_a_cache_src(self):
        from snapdragon_ai_lab.runtime import engine_ledger
        self.assertIn('preload', engine_ledger.ENUMS['cache_src'])
        self.assertIn('preload', cli.ENGINE_CACHE_SOURCES)


class NoRealConfigWrites(unittest.TestCase):
    """P56: every runtime.json writer goes through guard_production_path. The temp config is
    declared 'production' for the test; each writer must refuse and leave it byte-identical."""

    def test_writers_refuse_a_production_config(self):
        engine = FakeEngine()
        self.addCleanup(engine.close)
        stack = Stack(engine.port)
        self.addCleanup(stack.close)
        before = stack.config_path.read_bytes()
        real = (*controller.REAL_PRODUCTION_PATHS, stack.config_path.parent)
        writers = {'mode': lambda: stack.manager.set_mode(ORNITH, 'fast'),
                   'select': lambda: stack.manager.select(QWEN),
                   'budget': lambda: stack.manager.set_cache_budget(5_000_000_000),
                   'learning': lambda: stack.manager.set_harness_learning(True)}
        with patch.object(controller, 'REAL_PRODUCTION_PATHS', real):
            for name, write in writers.items():
                with self.subTest(writer=name):
                    with self.assertRaises((RuntimeError, controller.Unavailable)):
                        write()
                    self.assertEqual(stack.config_path.read_bytes(), before)
            code, _, _ = check.call(stack.base, 'POST', '/_local/mode', {'model': ORNITH, 'mode': 'fast'})
            self.assertNotEqual(code, 200)
            self.assertEqual(stack.config_path.read_bytes(), before)

    def test_example_runtime_modes_block(self):
        value = json.loads((Path(__file__).resolve().parents[1] / 'examples/runtime/llm-runtime.json').read_text())
        self.assertEqual(value['context_modes']['default'], 'fast')
        self.assertEqual(pp.stored_mode(value, ORNITH), 'fast')
        example = value['context_modes']['_example_256k']['rss_proof']
        self.assertEqual(set(example), pp.RSS_PROOF_KEYS)
        self.assertLessEqual(example['peak_rss_bytes'], pp.RSS_PROOF_PEAK_MAX)
        for alias in (ORNITH, QWEN):
            oxe = value['profiles'][alias]['rust_engine']['oxe']
            self.assertNotIn('rss_proof', oxe)  # max stays unavailable until the ladder writes one
            self.assertEqual(oxe['context_size'], 131072)
            args = oxe['oxe_args']
            self.assertEqual(args[args.index('--prefix-cache-ttl-h') + 1], '168')


if __name__ == '__main__':
    unittest.main()
