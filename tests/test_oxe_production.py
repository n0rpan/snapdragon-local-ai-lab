"""Synthetic MEMLAT-only Rust (oxe) alias switch; never starts an engine or reads weights."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from snapdragon_ai_lab.runtime import classic_beta as beta, controller, oxe_runtime, profile_policy

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('stage_oxe_production', ROOT / 'tools/stage_oxe_production.py')
stage_tool = importlib.util.module_from_spec(spec); spec.loader.exec_module(stage_tool)

RUST_SHA = stage_tool.QUALIFIED_SERVER_SHA
CLASSIC_MEMLAT = dict(threads=8, threads_batch=12, cpu_mask='0xff', cpu_mask_batch='0xfff', cpu_strict=1,
                      cpu_strict_batch=1, small_batch_limit=3, spec_type='none', prefix_cache_enabled=True,
                      server_binary='/synthetic/q3k/llama-server', server_sha256=beta.PATCHED_SERVER_SHA,
                      environment={'GGML_CPU_REPACK_Q3K': '1', 'LLAMA_CPU_SMALL_BATCH_LIMIT': '3'})


def profile(alias=beta.ALIAS, model_sha=beta.MODEL_SHA):
    return {**copy.deepcopy(beta.POLICY), 'ready': True, 'model_alias': alias,
            'model_path': '/synthetic/model.gguf', 'sha256': model_sha, 'expected_bytes': 100,
            'server_binary': '/synthetic/classic/llama-server', 'server_sha256': beta.SERVER_SHA,
            'prefix_cache': {'enabled': False, 'directory': '/synthetic/prefixes', 'max_bytes': 1},
            'validation': {'kind': 'synthetic-stock'}, 'memlat_validation': {'kind': 'synthetic-memlat'},
            'memlat_overrides': {**copy.deepcopy(CLASSIC_MEMLAT),
                                 'oxe': stage_tool.oxe_override('/synthetic/oxe-server', RUST_SHA)}}


class OxeMemlatPolicy(unittest.TestCase):
    def test_context_override_is_rust_only_and_preserves_fallbacks(self):
        original = profile()
        original['memlat_overrides']['oxe']['context_size'] = 32768
        before = copy.deepcopy(original)
        with patch.dict(os.environ, {'LOCAL_AI_ENABLE_OXE': '1'}):
            rust = profile_policy.resolve(original, available=True, oxe_enabled=True)
            command = oxe_runtime.command(rust, 8093)
        self.assertEqual(rust['context_size'], 32768)
        self.assertEqual(command[command.index('--ctx-size') + 1], '32768')
        self.assertEqual(profile_policy.resolve(original, available=False, oxe_enabled=True), before)
        self.assertEqual(profile_policy.resolve(original, available=True, oxe_enabled=False)['context_size'], 16384)
        self.assertEqual(original, before)

    def test_context_override_rejects_invalid_capacity(self):
        for value in (True, '32768', 0, 255, 131073):
            p = profile(); p['memlat_overrides']['oxe']['context_size'] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                profile_policy.resolve(p, available=True, oxe_enabled=True)

    def test_stock_boot_is_exact_even_when_enabled(self):
        original = profile()
        self.assertEqual(profile_policy.resolve(original, available=False, oxe_enabled=True), original)
        self.assertEqual(original, profile())

    def test_memlat_without_enable_keeps_classic_memlat(self):
        without_oxe = profile(); del without_oxe['memlat_overrides']['oxe']
        expected = profile_policy.resolve(without_oxe, available=True, oxe_enabled=False)
        for enabled in (False, None):
            with self.subTest(enabled=enabled), patch.dict(os.environ, {'LOCAL_AI_ENABLE_OXE': '0'}):
                result = profile_policy.resolve(profile(), available=True, oxe_enabled=enabled)
                result['memlat_overrides'].pop('oxe')
                self.assertEqual(result, expected)
                self.assertNotIn('backend_kind', result)
                self.assertEqual(result['server_sha256'], beta.PATCHED_SERVER_SHA)

    def test_memlat_enabled_switches_alias_to_clean_rust_launch(self):
        original = profile()
        with patch.dict(os.environ, {'LOCAL_AI_ENABLE_OXE': '1'}):
            result = profile_policy.resolve(original, available=True)
            self.assertTrue(oxe_runtime.eligible(result))
            command = oxe_runtime.command(result, 8093)
        self.assertEqual(original, profile())
        self.assertEqual((result['backend_kind'], result['server_binary'], result['server_sha256']),
                         ('oxe', '/synthetic/oxe-server', RUST_SHA))
        self.assertEqual((result['model_alias'], result['sha256'], result['ready']), (beta.ALIAS, beta.MODEL_SHA, True))
        self.assertEqual(result['extra_args'], [])
        self.assertEqual(result['environment'], {'OXE_MODEL_SHA256': beta.MODEL_SHA, 'OXE_ATTN_TILED_MIN_KV': '0'})
        self.assertEqual(result['prefix_cache'], {'enabled': False, 'directory': '/synthetic/prefixes', 'max_bytes': 1})
        self.assertEqual(result['threads'], 8)
        self.assertEqual(result['validation']['kind'], 'oxe-rust-server-qualification')
        self.assertEqual(result['execution_policy_mode'], 'memlat')
        self.assertEqual(command[command.index('--spec-type') + 1], 'none')
        self.assertEqual(command[command.index('--prefix-cache') + 1], 'default')
        self.assertEqual(result['environment']['OXE_ATTN_TILED_MIN_KV'], '0')
        self.assertEqual(command[command.index('--oxe-cpus') + 1], '0-7')
        self.assertEqual(command[command.index('--oxe-prefill-cpus') + 1], '0-11')
        self.assertEqual(command[command.index('--ctx-size') + 1], '16384')
        self.assertEqual(command[command.index('--alias') + 1], beta.ALIAS)
        # Detached: mutating the effective snapshot never reaches the profile.
        result['oxe_args'].append('--x'); result['environment']['X'] = '1'
        self.assertEqual(profile(), original)

    def test_prefix_store_forced_off_even_if_classic_memlat_enables_it(self):
        p = profile(); p['prefix_cache']['enabled'] = True
        self.assertFalse(profile_policy.resolve(p, available=True, oxe_enabled=True)['prefix_cache']['enabled'])

    def test_malformed_oxe_overrides_rejected(self):
        cases = [lambda o: o.pop('oxe_args'), lambda o: o.pop('server_sha256'),
                 lambda o: o.update(server_sha256='X' * 64), lambda o: o.update(server_binary='relative'),
                 lambda o: o.update(oxe_args=[]), lambda o: o.update(oxe_args=['ok', 1]),
                 lambda o: o.update(threads=13), lambda o: o.update(backend_kind='oxe'),
                 lambda o: o.update(environment={'GGML_CPU_REPACK': '1'}),
                 lambda o: o.update(environment={'LLAMA_CPU_SMALL_BATCH_LIMIT': '3'}),
                 lambda o: o.update(validation=[])]
        for index, change in enumerate(cases):
            p = profile(); change(p['memlat_overrides']['oxe'])
            with self.subTest(index=index), self.assertRaises(ValueError):
                profile_policy.resolve(p, available=True, oxe_enabled=False)
        p = profile(); p['memlat_overrides']['oxe']['environment'] = {'OXE_MODEL_SHA256': 'c' * 64}
        with self.assertRaisesRegex(ValueError, 'model_sha'):
            profile_policy.resolve(p, available=True, oxe_enabled=True)
        p = profile(); p['memlat_overrides']['oxe'] = 'oxe'
        with self.assertRaises(ValueError):
            profile_policy.resolve(p, available=True)


class OxeBetaAliasStartup(unittest.TestCase):
    def test_beta_alias_verifies_rust_engine_only_and_drops_legacy_env(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            model = tmp / 'model.gguf'; model.write_bytes(b'm' * 100)
            engine = tmp / 'oxe-server'; engine.write_bytes(b'synthetic rust server')
            p = profile(); p['model_path'] = str(model)
            p['memlat_overrides']['oxe'] = stage_tool.oxe_override(
                str(engine), hashlib.sha256(engine.read_bytes()).hexdigest())
            config = tmp / 'runtime.json'
            config.write_text(json.dumps({**profile(alias='qwen3.8-27b-local'), 'profiles': {beta.ALIAS: p}}))
            diagnostic_log = tmp / 'engine-stderr.log'
            launched = {}

            def popen(args, **kwargs):
                launched.update(args=args, env=kwargs['env'])
                raise RuntimeError('synthetic stop before launch')
            manager = controller.Manager(); manager.stop_event.set()
            try:
                with patch.dict(os.environ, {'LOCAL_AI_ENABLE_OXE': '1', 'GGML_CPU_MOE_GEMM': '1',
                                             'LLAMA_ARG_THREADS': '4', 'KEEP_ME': 'yes'}), \
                        patch.object(controller, 'CONFIG', config), patch.object(controller, 'LEASE', tmp / 'lease'), \
                        patch.object(controller, 'ENGINE_DIAGNOSTIC_LOG', diagnostic_log), \
                        patch.object(profile_policy, 'memlat_available', return_value=True), \
                        patch.object(controller, 'available_memory_bytes', return_value=None), \
                        patch.object(controller.memory_policy, 'MemoryGuard', return_value=Mock()), \
                        patch.object(beta, 'verify_engine', side_effect=AssertionError('classic verify')), \
                        patch.object(controller.subprocess, 'Popen', side_effect=popen):
                    with self.assertRaisesRegex(RuntimeError, 'synthetic stop'):
                        manager.begin(allow_start=True, model_alias=beta.ALIAS)
            finally:
                manager.close()
            self.assertEqual(launched['args'][0], str(engine))
            self.assertEqual(launched['args'][launched['args'].index('--spec-type') + 1], 'none')
            self.assertNotIn('--slot-save-path', launched['args'])
            env = launched['env']
            self.assertFalse([k for k in env if k.startswith(('GGML_', 'LLAMA_'))])
            self.assertEqual((env['OXE_MODEL_SHA256'], env['KEEP_ME']), (beta.MODEL_SHA, 'yes'))
            self.assertIsNone(manager.lease)


def runtime():
    base = profile(alias='qwen3.8-27b-local', model_sha='9' * 64)
    profiles = {}
    for alias, model_sha in stage_tool.QUALIFIED_MODELS.items():
        p = profile(alias=alias, model_sha=model_sha); del p['memlat_overrides']['oxe']
        profiles[alias] = p
    profiles['qwen3.8-27b-local'] = copy.deepcopy(base)
    return {**base, 'selected_model': beta.ALIAS, 'profiles': profiles}


UNIT = '''[Unit]
Description=Private local AI runtime (controller.py)
[Service]
WorkingDirectory=/synthetic/runtime-releases/old
ExecStart=/usr/bin/python3 "/synthetic/runtime-releases/old/controller.py"
Environment="LOCAL_LLM_CONFIG=/synthetic/runtime.json"
Nice=15
'''


class StageTool(unittest.TestCase):
    def test_adds_only_nested_override_to_the_two_aliases(self):
        before = runtime()
        after = stage_tool.add_oxe_overrides(before, '/synthetic/oxe-server', RUST_SHA)
        self.assertEqual(before, runtime())
        for alias in stage_tool.QUALIFIED_MODELS:
            changed = copy.deepcopy(after['profiles'][alias])
            self.assertEqual(changed['memlat_overrides'].pop('oxe')['oxe_args'][-2:], ['--prefix-cache', 'default'])
            self.assertEqual(changed, before['profiles'][alias])
        self.assertEqual(after['profiles']['qwen3.8-27b-local'], before['profiles']['qwen3.8-27b-local'])
        self.assertEqual({k: v for k, v in after.items() if k != 'profiles'},
                         {k: v for k, v in before.items() if k != 'profiles'})

    def test_refuses_unqualified_identities(self):
        with self.assertRaisesRegex(ValueError, 'unqualified'):
            stage_tool.add_oxe_overrides(runtime(), '/x', 'a' * 64)
        bad = runtime(); bad['profiles']['qwen3.6-35b-a3b-local']['sha256'] = 'f' * 64
        with self.assertRaisesRegex(ValueError, 'identity'):
            stage_tool.add_oxe_overrides(bad, '/x', RUST_SHA)
        twice = stage_tool.add_oxe_overrides(runtime(), '/x', RUST_SHA)
        with self.assertRaisesRegex(ValueError, 'already'):
            stage_tool.add_oxe_overrides(twice, '/x', RUST_SHA)

    def test_unit_rewrite_points_at_release_and_enables_oxe_once(self):
        unit = stage_tool.rewrite_unit(UNIT, '/synthetic/runtime-releases/new')
        self.assertIn('WorkingDirectory=/synthetic/runtime-releases/new\n', unit)
        self.assertIn('ExecStart=/usr/bin/python3 "/synthetic/runtime-releases/new/controller.py"\n', unit)
        self.assertEqual(unit.count('Environment="LOCAL_AI_ENABLE_OXE=1"'), 1)
        self.assertNotIn('/old', unit)
        with self.assertRaises(ValueError):
            stage_tool.rewrite_unit(unit, '/synthetic/runtime-releases/new')
        with self.assertRaises(ValueError):
            stage_tool.rewrite_unit('[Service]\nNice=1\n', '/r')

    def test_stage_writes_fresh_bound_payload_and_rollback(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            live_runtime = tmp / 'runtime.json'; live_runtime.write_text(json.dumps(runtime()))
            live_unit = tmp / 'unit.service'; live_unit.write_text(UNIT)
            engine = tmp / 'oxe-server'; engine.write_bytes(b'synthetic rust server')
            digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
            with patch.object(stage_tool, 'QUALIFIED_SERVER_SHA', digest(engine)):
                with self.assertRaisesRegex(ValueError, 'identity_changed'):
                    stage_tool.stage(tmp / 'bad', runtime_path=live_runtime, runtime_sha='0' * 64, unit_path=live_unit,
                                     unit_sha=digest(live_unit), server_binary=engine, releases_root=tmp / 'releases')
                self.assertFalse((tmp / 'bad').exists())
                result = stage_tool.stage(tmp / 'stage', runtime_path=live_runtime, runtime_sha=digest(live_runtime),
                                          unit_path=live_unit, unit_sha=digest(live_unit), server_binary=engine,
                                          releases_root=tmp / 'releases')
            stage = tmp / 'stage'
            self.assertEqual((stage / 'rollback/runtime.original').read_bytes(), live_runtime.read_bytes())
            self.assertEqual((stage / 'rollback/unit.original').read_text(), UNIT)
            self.assertIn(result['release_dir'], (stage / 'local-ai-chat.service').read_text())
            staged = json.loads((stage / 'runtime.json').read_text())
            self.assertEqual(staged['profiles'][beta.ALIAS]['memlat_overrides']['oxe']['server_binary'], str(engine))
            self.assertTrue((stage / 'runtime-source/controller.py').is_file())
            self.assertTrue(os.access(stage / 'rollback/rollback.py', os.X_OK))
            self.assertEqual(live_unit.read_text(), UNIT)
            compile((stage / 'rollback/rollback.py').read_text(), 'rollback.py', 'exec')


if __name__ == '__main__':
    unittest.main()
