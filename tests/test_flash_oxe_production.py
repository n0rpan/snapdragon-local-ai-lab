"""Synthetic Rust Flash (oxe --engine flash) switch; never starts an engine, loads weights or reads ~/.config."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

from snapdragon_ai_lab.runtime import bigmoe_runtime, controller, oxe_runtime, profile_policy

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('stage_flash_oxe_production', ROOT / 'tools/stage_flash_oxe_production.py')
stage_tool = importlib.util.module_from_spec(spec); spec.loader.exec_module(stage_tool)

ALIAS = stage_tool.ALIAS
GIB = 1024**3
RUST_SHA = 'd' * 64
PEAK = 26 * GIB
SIDECAR = '/synthetic/flash/sidecar.bin'
SIDECAR_BYTES = 4096
FLASH_ON = {profile_policy.OXE_FLASH_ENABLE_ENV: '1'}
MOE_ON = {oxe_runtime.ENABLE_ENV: '1'}


def flash_profile(alias=ALIAS):
    shards = [{'path': '/synthetic/flash/shard-%d.gguf' % i, 'bytes': size, 'sha256': digest}
              for i, (size, digest) in enumerate(bigmoe_runtime.PINNED, 1)]
    return {'ready': True, 'backend_kind': bigmoe_runtime.KIND, 'model_alias': alias,
            'display_name': 'Flash synthetic', 'model_path': shards[0]['path'], 'expected_bytes': shards[0]['bytes'],
            'server_binary': '/synthetic/bmoe-server', 'server_sha256': 'c' * 64, 'sha256': '9' * 64,
            'context_size': 16384, 'threads': 4, 'gpu_layers': 0, 'backend': 'cpu', 'extra_args': [],
            'environment': {}, 'prefix_cache': {'enabled': False},
            'bigmoe': {'revision': bigmoe_runtime.REVISION, 'cache_mib': 12288, 'models_dir': '/synthetic/flash',
                       'data_dir': '/synthetic/data', 'shards': shards, 'microbatch_size': 4096},
            'memlat_overrides': {'oxe_flash': stage_tool.flash_override(
                '/synthetic/oxe-server', RUST_SHA, sidecar_path=SIDECAR, sidecar_bytes=SIDECAR_BYTES,
                measured_peak_bytes=PEAK, cache_gib=20)}}


def effective_flash(**environment):
    with patch.dict(os.environ, {**environment}, clear=True):
        return profile_policy.resolve(flash_profile(), available=True)


class FlashPolicy(unittest.TestCase):
    def test_stock_boot_is_exact_even_when_flag_is_on(self):
        original = flash_profile()
        self.assertEqual(profile_policy.resolve(original, available=False, oxe_flash_enabled=True), original)
        self.assertEqual(original, flash_profile())

    def test_memlat_without_flag_keeps_stock_bigmoe(self):
        original = flash_profile()
        for env in ({}, {profile_policy.OXE_FLASH_ENABLE_ENV: '0'}):
            with self.subTest(env=env), patch.dict(os.environ, env, clear=True):
                result = profile_policy.resolve(original, available=True)
                self.assertEqual(result, original)
                self.assertEqual(result['backend_kind'], bigmoe_runtime.KIND)
        self.assertEqual(profile_policy.resolve(original, available=True, oxe_flash_enabled=False), original)

    def test_flag_switches_to_clean_rust_flash_launch(self):
        original = flash_profile()
        with patch.dict(os.environ, FLASH_ON, clear=True):
            result = profile_policy.resolve(original, available=True)
            self.assertTrue(oxe_runtime.eligible(result))
            command = oxe_runtime.command(result, 8093)
        self.assertEqual(original, flash_profile())
        digest = profile_policy.flash_model_sha(original['bigmoe']['shards'])
        self.assertEqual(result['backend_kind'], 'oxe')
        self.assertEqual((result['server_binary'], result['server_sha256']), ('/synthetic/oxe-server', RUST_SHA))
        self.assertEqual((result['sha256'], result['environment']['OXE_MODEL_SHA256']), (digest, digest))
        self.assertEqual((result['extra_args'], result['threads'], result['expected_bytes']),
                         ([], 8, bigmoe_runtime.PINNED[0][0]))
        self.assertEqual(result['prefix_cache']['enabled'], False)
        self.assertEqual(result['execution_policy_mode'], 'memlat')
        self.assertEqual(result['oxe_flash']['idle_unload_seconds'], 600)
        self.assertEqual(len(result['oxe_flash']['shards']), 4)
        self.assertEqual(command[:5], ['/synthetic/oxe-server', '--host', '127.0.0.1', '--port', '8093'])
        self.assertEqual(command[command.index('--engine') + 1], 'flash')
        self.assertEqual(command[command.index('--flash-cache-gib') + 1], '20')
        self.assertEqual(command[command.index('--ctx-size') + 1], '16384')
        self.assertEqual(command[command.index('--alias') + 1], ALIAS)
        # Detached: mutating the effective snapshot never reaches the profile.
        result['oxe_args'].append('--x'); result['oxe_flash']['shards'][0]['bytes'] = 1
        self.assertEqual(flash_profile(), original)

    def test_moe_flag_and_flash_flag_are_independent(self):
        moe = {'backend_kind': 'oxe', 'ready': True, 'model_alias': 'ornith-oxe', 'model_path': '/m.gguf',
               'server_binary': '/e', 'server_sha256': 'a' * 64, 'sha256': 'b' * 64, 'expected_bytes': 100,
               'threads': 8, 'context_size': 16384, 'prefix_cache': {'enabled': False},
               'oxe_args': ['--model', '{model}']}
        flash_cfg = effective_flash(**FLASH_ON)
        with patch.dict(os.environ, MOE_ON, clear=True):
            self.assertEqual(profile_policy.resolve(flash_profile(), available=True), flash_profile())
            self.assertTrue(oxe_runtime.eligible(moe))
            self.assertFalse(oxe_runtime.eligible(flash_cfg))
        with patch.dict(os.environ, FLASH_ON, clear=True):
            self.assertFalse(oxe_runtime.eligible(moe))
            self.assertTrue(oxe_runtime.eligible(flash_cfg))

    def test_flash_override_refused_on_classic_profile(self):
        classic = {'ready': True, 'model_path': '/m.gguf', 'sha256': 'a' * 64,
                   'memlat_overrides': {'oxe_flash': flash_profile()['memlat_overrides']['oxe_flash']}}
        with self.assertRaisesRegex(ValueError, 'requires_flash_backend'):
            profile_policy.resolve(classic, available=True, oxe_flash_enabled=True)


class FlashAdmission(unittest.TestCase):
    def test_measured_host_floor_is_independent_of_scope_peak(self):
        cfg = effective_flash(**FLASH_ON)
        baseline = oxe_runtime.flash_required_available(cfg)
        minimum = baseline + 512 * 1024**2
        cfg['oxe_flash']['admission']['minimum_available_bytes'] = minimum
        self.assertEqual(oxe_runtime.flash_required_available(cfg), minimum)
        with self.assertRaisesRegex(ValueError, 'insufficient_available_memory'):
            oxe_runtime.flash_admit(cfg, minimum - 1)
        self.assertIsNone(oxe_runtime.flash_admit(cfg, minimum))

    def test_admit_passes_at_peak_plus_floor_and_fails_one_byte_below(self):
        cfg = effective_flash(**FLASH_ON)
        required = PEAK + 1536 * 1024**2
        self.assertEqual(oxe_runtime.flash_required_available(cfg), required)
        self.assertIsNone(oxe_runtime.flash_admit(cfg, required))
        with self.assertRaisesRegex(ValueError, 'insufficient_available_memory'):
            oxe_runtime.flash_admit(cfg, required - 1)
        with self.assertRaisesRegex(ValueError, 'insufficient_available_memory'):
            oxe_runtime.flash_admit(cfg, None)

    def test_budget_above_cap_or_peak_above_memory_max_fails(self):
        cfg = effective_flash(**FLASH_ON)
        cfg['oxe_flash']['admission']['admitted_budget_bytes'] = 29 * GIB
        with self.assertRaisesRegex(ValueError, 'exceeds_cap'):
            oxe_runtime.flash_admit(cfg, 64 * GIB)
        cfg = effective_flash(**FLASH_ON)
        with self.assertRaisesRegex(ValueError, 'exceeds_cap'):
            oxe_runtime.flash_admit(cfg, 64 * GIB, memory_max=PEAK - 1)
        self.assertIsNone(oxe_runtime.flash_admit(cfg, 64 * GIB, memory_max=PEAK))


class FlashPower(unittest.TestCase):
    def state(self, percent, status='Discharging'):
        return {'external_online': False, 'battery_status': status, 'battery_percent': percent}

    def test_start_and_stop_thresholds(self):
        self.assertEqual(oxe_runtime.flash_power_failure(self.state(29), starting=True), 'flash_battery_start_reserve')
        self.assertIsNone(oxe_runtime.flash_power_failure(self.state(30), starting=True))
        self.assertIsNone(oxe_runtime.flash_power_failure(self.state(29)))
        self.assertEqual(oxe_runtime.flash_power_failure(self.state(20)), 'flash_battery_stop_reserve')
        self.assertIsNone(oxe_runtime.flash_power_failure(self.state(21)))

    def test_missing_telemetry_fails(self):
        for value in ({'battery_status': None, 'battery_percent': None}, {}, self.state(None)):
            with self.subTest(value=value):
                self.assertEqual(oxe_runtime.flash_power_failure(value), 'flash_power_telemetry_unavailable')


class FakeProcess:
    pid = 4242

    def __init__(self, waits=None):
        self.waits = list(waits or [])

    def poll(self):
        return None

    def wait(self, timeout=None):
        value = self.waits.pop(0) if self.waits else 0
        if isinstance(value, BaseException):
            raise value
        return value


class FlashPowerGuardTests(unittest.TestCase):
    def test_low_battery_signals_process_group_then_kills_if_needed(self):
        power = lambda: {'external_online': False, 'battery_status': 'Discharging', 'battery_percent': 15}
        with patch.object(oxe_runtime.os, 'killpg') as killpg:
            guard = oxe_runtime.FlashPowerGuard(FakeProcess(), power=power, interval=0.01)
            guard.start(); guard.thread.join(timeout=5)
            guard.stop()
            self.assertEqual(guard.failure, 'flash_battery_stop_reserve')
            killpg.assert_called_once_with(4242, signal.SIGTERM)
        with patch.object(oxe_runtime.os, 'killpg') as killpg:
            process = FakeProcess(waits=[subprocess.TimeoutExpired('x', 10), 0])
            guard = oxe_runtime.FlashPowerGuard(process, power=power, interval=0.01)
            guard.start(); guard.thread.join(timeout=5)
            self.assertEqual([c.args[1] for c in killpg.call_args_list], [signal.SIGTERM, signal.SIGKILL])

    def test_healthy_power_never_signals(self):
        power = lambda: {'external_online': False, 'battery_status': 'Discharging', 'battery_percent': 80}
        with patch.object(oxe_runtime.os, 'killpg') as killpg:
            guard = oxe_runtime.FlashPowerGuard(FakeProcess(), power=power, interval=0.01)
            guard.start()
            guard.stop()
            self.assertIsNone(guard.failure)
            killpg.assert_not_called()

    def test_stop_without_start_is_safe(self):
        guard = oxe_runtime.FlashPowerGuard(FakeProcess(), power=lambda: {})
        guard.stop()


class FlashInvalidOverrides(unittest.TestCase):
    def resolve_with(self, change, flag=True):
        p = flash_profile()
        change(p['memlat_overrides']['oxe_flash'])
        return profile_policy.resolve(p, available=True, oxe_flash_enabled=flag)

    def test_malformed_flash_overrides_rejected(self):
        cases = {
            'cache_gib 22': lambda o: o.update(cache_gib=22),
            'missing admission': lambda o: o.pop('admission'),
            'peak below dense bound': lambda o: o['admission'].update(measured_peak_bytes=20 * GIB),
            'budget above the absolute cap': lambda o: o['admission'].update(admitted_budget_bytes=29 * GIB + 1),
            'extra admission key': lambda o: o['admission'].update(note=1),
            'relative sidecar': lambda o: o.update(sidecar_path='relative.bin'),
            'idle too short': lambda o: o.update(idle_unload_seconds=30),
            'bad server sha': lambda o: o.update(server_sha256='X' * 64),
        }
        for name, change in cases.items():
            with self.subTest(name), self.assertRaises(ValueError):
                self.resolve_with(change, flag=False)

    def test_other_override_keys_on_flash_profile_rejected(self):
        p = flash_profile()
        p['memlat_overrides']['threads'] = 6
        with self.assertRaisesRegex(ValueError, 'require_classic_backend'):
            profile_policy.resolve(p, available=True, oxe_flash_enabled=False)

    def test_args_must_match_cache_and_shas_must_match_shards(self):
        effective = self.resolve_with(lambda o: None)
        i = effective['oxe_args'].index('--flash-cache-gib')
        effective['oxe_args'][i + 1] = '18'
        with self.assertRaisesRegex(ValueError, 'args_must_match_admission'):
            oxe_runtime._validate(effective)
        effective = self.resolve_with(lambda o: None)
        effective['sha256'] = 'e' * 64
        with self.assertRaisesRegex(ValueError, 'model_sha_mismatch'):
            oxe_runtime._validate(effective)

    def test_model_sha_mismatch_in_environment_rejected(self):
        p = flash_profile()
        p['memlat_overrides']['oxe_flash']['environment'] = {'OXE_MODEL_SHA256': 'c' * 64}
        with self.assertRaisesRegex(ValueError, 'model_sha'):
            profile_policy.resolve(p, available=True, oxe_flash_enabled=True)


class FlashFiles(unittest.TestCase):
    def test_metadata_checks_sizes_and_types(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sizes = [3, 5, 7, 11]
            paths = []
            for i, size in enumerate(sizes):
                path = root / ('shard-%d.gguf' % i); path.write_bytes(b'x' * size); paths.append(path)
            sidecar = root / 'sidecar.bin'; sidecar.write_bytes(b's' * 13)
            cfg = effective_flash(**FLASH_ON)
            cfg['oxe_flash']['shards'] = [{'path': str(p), 'bytes': s, 'sha256': 'f' * 64} for p, s in zip(paths, sizes)]
            cfg['oxe_flash']['sidecar_path'] = str(sidecar)
            cfg['oxe_flash']['sidecar_bytes'] = 13
            self.assertTrue(oxe_runtime.flash_files_ready(cfg))
            cfg['oxe_flash']['shards'][1]['bytes'] = 6
            self.assertFalse(oxe_runtime.flash_files_ready(cfg))
            cfg['oxe_flash']['shards'][1]['bytes'] = 5
            cfg['oxe_flash']['sidecar_bytes'] = 12
            self.assertFalse(oxe_runtime.flash_files_ready(cfg))
            cfg['oxe_flash']['sidecar_bytes'] = 13
            cfg['oxe_flash']['shards'][2]['path'] = str(root / 'missing.gguf')
            self.assertFalse(oxe_runtime.flash_files_ready(cfg))
            cfg['oxe_flash']['shards'][2]['path'] = str(root)
            self.assertFalse(oxe_runtime.flash_files_ready(cfg))


class IdleAndCgroup(unittest.TestCase):
    def test_idle_timeout_is_flash_specific(self):
        cfg = effective_flash(**FLASH_ON)
        self.assertEqual(controller.idle_seconds(cfg), 600)
        cfg['oxe_flash']['idle_unload_seconds'] = 900
        self.assertEqual(controller.idle_seconds(cfg), 900)
        self.assertEqual(controller.idle_seconds({'backend_kind': 'oxe'}), controller.IDLE_SECONDS)
        self.assertEqual(controller.idle_seconds(None), controller.IDLE_SECONDS)


def staged_runtime():
    flash = flash_profile()
    base = {k: v for k, v in flash.items() if k != 'memlat_overrides'}
    return {'model_alias': 'qwen3.8-27b-local', 'context_size': 8192,
            'selected_model': ALIAS, 'profiles': {ALIAS: base}}


UNIT = '''[Unit]
Description=Private local AI runtime (controller.py)
[Service]
WorkingDirectory=/synthetic/runtime-releases/old
ExecStart=/usr/bin/python3 "/synthetic/runtime-releases/old/controller.py"
Environment="LOCAL_LLM_CONFIG=/synthetic/runtime.json"
Nice=15
'''


class FlashStageTool(unittest.TestCase):
    def test_refuses_while_no_flash_server_is_qualified(self):
        self.assertIsNone(stage_tool.QUALIFIED_FLASH_SERVER_SHA)
        with self.assertRaisesRegex(ValueError, 'unqualified_rust_flash_server'):
            stage_tool.add_flash_override(staged_runtime(), '/x', RUST_SHA, sidecar_path=__file__,
                                          measured_peak_bytes=PEAK)

    def test_qualified_override_is_detached_and_resolves_both_ways(self):
        before = staged_runtime()
        with tempfile.TemporaryDirectory() as tmp:
            sidecar = Path(tmp) / 'sidecar.bin'; sidecar.write_bytes(b's' * 64)
            with self.assertRaisesRegex(ValueError, 'unqualified'):
                stage_tool.add_flash_override(before, '/x', RUST_SHA, qualified_sha='b' * 64,
                                              sidecar_path=sidecar, measured_peak_bytes=PEAK)
            after = stage_tool.add_flash_override(before, '/synthetic/oxe-server', RUST_SHA, qualified_sha=RUST_SHA,
                                                  sidecar_path=sidecar, measured_peak_bytes=PEAK)
        self.assertEqual(before, staged_runtime())
        merged = {**{k: v for k, v in after.items() if k != 'profiles'}, **after['profiles'][ALIAS]}
        self.assertEqual(profile_policy.resolve(merged, available=True, oxe_flash_enabled=False), merged)
        with patch.dict(os.environ, FLASH_ON, clear=True):
            effective = profile_policy.resolve(merged, available=True)
        self.assertEqual(effective['backend_kind'], 'oxe')
        self.assertEqual(effective['oxe_flash']['sidecar_bytes'], 64)

    def test_cache_18_changes_only_the_cache_argument(self):
        args_20 = stage_tool.flash_override('/x', RUST_SHA, sidecar_path=SIDECAR, sidecar_bytes=1,
                                            measured_peak_bytes=PEAK, cache_gib=20)['oxe_args']
        args_18 = stage_tool.flash_override('/x', RUST_SHA, sidecar_path=SIDECAR, sidecar_bytes=1,
                                            measured_peak_bytes=PEAK, cache_gib=18)['oxe_args']
        self.assertEqual(args_18[args_18.index('--flash-cache-gib') + 1], '18')
        self.assertEqual([a for a in args_18 if a != '18'], [a for a in args_20 if a != '20'])

    def test_unit_rewrite_keeps_moe_line_and_adds_flash_once(self):
        moe_unit = UNIT.replace('Nice=15', 'Nice=15\nEnvironment="LOCAL_AI_ENABLE_OXE=1"')
        unit = stage_tool.rewrite_unit(moe_unit, '/synthetic/runtime-releases/new')
        self.assertEqual(unit.count('Environment="LOCAL_AI_ENABLE_OXE=1"'), 1)
        self.assertEqual(unit.count('Environment="LOCAL_AI_ENABLE_OXE_FLASH=1"'), 1)
        self.assertLess(unit.index('LOCAL_LLM_CONFIG'), unit.index('LOCAL_AI_ENABLE_OXE_FLASH'))
        self.assertIn('WorkingDirectory=/synthetic/runtime-releases/new\n', unit)
        self.assertIn('ExecStart=/usr/bin/python3 "/synthetic/runtime-releases/new/controller.py"\n', unit)
        with self.assertRaisesRegex(ValueError, 'already_mentions'):
            stage_tool.rewrite_unit(unit, '/synthetic/runtime-releases/new')
        with self.assertRaises(ValueError):
            stage_tool.rewrite_unit('[Service]\nNice=1\n', '/r')

    def test_stage_writes_fresh_bound_payload_and_exact_rollback(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            live_runtime = tmp / 'runtime.json'
            live_runtime.write_text(json.dumps(staged_runtime()))
            live_unit = tmp / 'unit.service'
            live_unit.write_text(UNIT.replace('Nice=15', 'Nice=15\nEnvironment="LOCAL_AI_ENABLE_OXE=1"'))
            engine = tmp / 'oxe-server'; engine.write_bytes(b'synthetic rust server')
            sidecar = tmp / 'sidecar.bin'; sidecar.write_bytes(b's' * 4096)
            digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
            live_before = (live_runtime.read_bytes(), live_unit.read_bytes())
            stage = tmp / 'stage'
            result = stage_tool.stage(stage, runtime_path=live_runtime, runtime_sha=digest(live_runtime),
                                      unit_path=live_unit, unit_sha=digest(live_unit), server_binary=engine,
                                      releases_root=tmp / 'releases', engine_releases_root=tmp / 'engines',
                                      sidecar_path=sidecar, measured_peak_bytes=PEAK, qualified_sha=digest(engine))
            self.assertEqual((live_runtime.read_bytes(), live_unit.read_bytes()), live_before)
            self.assertEqual((stage / 'rollback/runtime.original').read_bytes(), live_before[0])
            self.assertEqual((stage / 'rollback/unit.original').read_bytes(), live_before[1])
            unit = (stage / 'local-ai-chat.service').read_text()
            self.assertIn(result['release_dir'], unit)
            self.assertIn('flash-oxe-production-', result['release_dir'])
            self.assertEqual(unit.count('Environment="LOCAL_AI_ENABLE_OXE=1"'), 1)
            self.assertEqual(unit.count('Environment="LOCAL_AI_ENABLE_OXE_FLASH=1"'), 1)
            staged = json.loads((stage / 'runtime.json').read_text())
            profile = staged['profiles'][ALIAS]
            # The override names the content-addressed install path, not the build output it was copied from.
            self.assertEqual(profile['memlat_overrides']['oxe_flash']['server_binary'],
                             str(stage_tool.engine_install_path(tmp / 'engines', digest(engine))))
            merged = {**{k: v for k, v in staged.items() if k != 'profiles'}, **profile}
            with patch.dict(os.environ, FLASH_ON, clear=True):
                self.assertEqual(profile_policy.resolve(merged, available=True)['backend_kind'], 'oxe')
            with patch.dict(os.environ, {}, clear=True):
                self.assertEqual(profile_policy.resolve(merged, available=True), merged)
            self.assertTrue((stage / 'runtime-source/controller.py').is_file())
            self.assertTrue(os.access(stage / 'rollback/rollback.py', os.X_OK))
            text = (stage / 'rollback/rollback.py').read_text()
            self.assertIn('Flash is back on BigMoeOnEdge; no model loaded.', text)
            compile(text, 'rollback.py', 'exec')
            self.assertFalse(result['installed'] or result['models_started'])
            with self.assertRaisesRegex(ValueError, 'fresh_absolute'):
                stage_tool.stage(stage, runtime_path=live_runtime, runtime_sha=digest(live_runtime),
                                 unit_path=live_unit, unit_sha=digest(live_unit), server_binary=engine,
                                 releases_root=tmp / 'releases', engine_releases_root=tmp / 'engines',
                                 sidecar_path=sidecar, measured_peak_bytes=PEAK, qualified_sha=digest(engine))


class FlashCacheFallback(unittest.TestCase):
    """20 GiB by default; the 18 GiB admission is used only when 20 cannot start now."""
    SMALL = PEAK - 2 * oxe_runtime.GIB

    def config(self):
        profile = flash_profile()
        profile['memlat_overrides']['oxe_flash'] = stage_tool.flash_override(
            '/synthetic/oxe-server', RUST_SHA, sidecar_path=SIDECAR, sidecar_bytes=SIDECAR_BYTES,
            measured_peak_bytes=PEAK, cache_gib=20, fallback_peak_bytes=self.SMALL)
        return profile_policy.resolve(profile, available=True, oxe_flash_enabled=True)

    def test_primary_is_kept_when_admissible(self):
        config = self.config()
        chosen = oxe_runtime.flash_select(config, PEAK + oxe_runtime.FLASH_HOST_FLOOR_BYTES)
        self.assertIs(chosen, config)
        self.assertEqual(chosen['oxe_flash']['cache_gib'], 20)

    def test_fallback_is_a_detached_valid_18_gib_launch(self):
        config = self.config(); before = copy.deepcopy(config)
        chosen = oxe_runtime.flash_select(config, self.SMALL + oxe_runtime.FLASH_HOST_FLOOR_BYTES)
        self.assertEqual(config, before)
        flash = chosen['oxe_flash']
        self.assertEqual((flash['cache_gib'], flash['admission']['measured_peak_bytes'], flash['fallback_active']),
                         (18, self.SMALL, True))
        self.assertNotIn('fallback', flash)
        args = chosen['oxe_args']
        self.assertEqual(args[args.index('--flash-cache-gib') + 1], '18')
        with patch.dict(os.environ, {'LOCAL_AI_ENABLE_OXE_FLASH': '1'}):
            self.assertIn('18', oxe_runtime.command(chosen, 8093))
        self.assertEqual(oxe_runtime.flash_required_available(chosen), self.SMALL + oxe_runtime.FLASH_HOST_FLOOR_BYTES)

    def test_neither_admissible_raises_and_reports_the_smaller_need(self):
        config = self.config()
        with self.assertRaises(ValueError):
            oxe_runtime.flash_select(config, self.SMALL + oxe_runtime.FLASH_HOST_FLOOR_BYTES - 1)
        self.assertEqual(oxe_runtime.flash_smallest_required_available(config),
                         self.SMALL + oxe_runtime.FLASH_HOST_FLOOR_BYTES)

    def test_without_fallback_the_primary_error_stands(self):
        config = profile_policy.resolve(flash_profile(), available=True, oxe_flash_enabled=True)
        with self.assertRaises(ValueError):
            oxe_runtime.flash_select(config, PEAK)
        self.assertEqual(oxe_runtime.flash_smallest_required_available(config),
                         PEAK + oxe_runtime.FLASH_HOST_FLOOR_BYTES)

    def test_invalid_fallbacks_rejected(self):
        for change in (lambda f: f.update(cache_gib=20), lambda f: f['admission'].update(measured_peak_bytes=PEAK),
                       lambda f: f.update(extra=1)):
            profile = flash_profile()
            flash = stage_tool.flash_override('/synthetic/oxe-server', RUST_SHA, sidecar_path=SIDECAR,
                                              sidecar_bytes=SIDECAR_BYTES, measured_peak_bytes=PEAK, cache_gib=20,
                                              fallback_peak_bytes=self.SMALL)
            change(flash['fallback']); profile['memlat_overrides']['oxe_flash'] = flash
            with self.assertRaises(ValueError):
                profile_policy.resolve(profile, available=True, oxe_flash_enabled=True)


class UnloadControl(unittest.TestCase):
    """Status fields for an 'Unload now' button: loaded model, its RAM and the idle countdown."""
    def manager(self, **fields):
        manager = controller.Manager.__new__(controller.Manager)
        base = dict(process=None, loaded_config=None, state='unloaded', active=0, last_used=0)
        for key, value in {**base, **fields}.items():
            setattr(manager, key, value)
        return manager

    def test_resident_bytes_reads_only_the_rss_line(self):
        with tempfile.TemporaryDirectory() as root:
            (Path(root) / '42').mkdir()
            (Path(root) / '42/status').write_text('Name:\toxe-server\nVmHWM:\t  900 kB\nVmRSS:\t  512 kB\n')
            self.assertEqual(controller.resident_bytes(42, root=root), 512 * 1024)
            self.assertIsNone(controller.resident_bytes(43, root=root))
            (Path(root) / '42/status').write_text('VmRSS:\t many kB\n')
            self.assertIsNone(controller.resident_bytes(42, root=root))

    def test_unloaded_reports_nothing_to_unload(self):
        self.assertEqual(self.manager().unload_info(),
                         {'loaded_model': None, 'resident_bytes': None, 'idle_unload_remaining_seconds': None,
                          'unload': {'method': 'POST', 'path': '/_local/unload', 'available': False}})

    def test_loaded_flash_counts_down_from_its_own_timeout(self):
        config = effective_flash(**FLASH_ON)
        process = Mock(pid=4242)
        with patch.object(controller.time, 'monotonic', return_value=1000.0), \
             patch.object(controller, 'resident_bytes', return_value=27 * GIB) as rss:
            idle = self.manager(process=process, loaded_config=config, state='ready', last_used=900.0).unload_info()
            busy = self.manager(process=process, loaded_config=config, state='ready', last_used=900.0,
                                active=1).unload_info()
        rss.assert_called_with(4242)
        self.assertEqual((idle['loaded_model'], idle['resident_bytes'], idle['idle_unload_remaining_seconds']),
                         (ALIAS, 27 * GIB, 500.0))
        self.assertTrue(idle['unload']['available'])
        self.assertIsNone(busy['idle_unload_remaining_seconds'])
        self.assertFalse(busy['unload']['available'])


class FlashMemoryScope(unittest.TestCase):
    """With the profile's flash_memory policy: 1 GiB floor, 29 GiB budget, a 21 GiB cache."""
    POLICY = {'scope_max_bytes': 29 * GIB, 'host_floor_bytes': GIB}

    def config(self, cache_gib=21, peak=28 * GIB, budget=29 * GIB, policy=True):
        profile = flash_profile()
        if policy:
            profile['flash_memory'] = dict(self.POLICY)
        profile['memlat_overrides']['oxe_flash'] = stage_tool.flash_override(
            '/synthetic/oxe-server', RUST_SHA, sidecar_path=SIDECAR, sidecar_bytes=SIDECAR_BYTES,
            measured_peak_bytes=peak, cache_gib=cache_gib, budget_bytes=budget)
        return profile_policy.resolve(profile, available=True, oxe_flash_enabled=True)

    def test_limits_follow_the_policy(self):
        self.assertEqual(oxe_runtime.flash_limits({}), (1536 * 1024**2, 28 * GIB))
        config = self.config()
        self.assertEqual(oxe_runtime.flash_limits(config), (GIB, 29 * GIB))
        oxe_runtime._validate(config)
        self.assertEqual(oxe_runtime.flash_required_available(config), 29 * GIB)
        oxe_runtime.flash_admit(config, 29 * GIB, 29 * GIB)
        with self.assertRaises(ValueError):
            oxe_runtime.flash_admit(config, 29 * GIB - 1, 29 * GIB)

    def test_without_the_policy_21_gib_and_a_29_gib_budget_are_refused(self):
        for arguments in (dict(cache_gib=21, peak=26 * GIB, budget=28 * GIB), dict(cache_gib=20, peak=26 * GIB)):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                self.config(policy=False, **arguments)
        stock = self.config(cache_gib=20, peak=26 * GIB, budget=28 * GIB, policy=False)
        self.assertEqual(oxe_runtime.flash_required_available(stock), 26 * GIB + 1536 * 1024**2)

    def test_runtime_validation_rechecks_the_cap(self):
        config = self.config()
        del config['flash_memory']
        with self.assertRaises(ValueError):
            oxe_runtime._validate(config)

    def test_stage_tool_uses_the_profile_scope_for_its_budget(self):
        runtime = staged_runtime()
        runtime['profiles'][ALIAS]['flash_memory'] = dict(self.POLICY)
        with tempfile.NamedTemporaryFile() as sidecar:
            sidecar.write(b'x' * 16); sidecar.flush()
            updated = stage_tool.add_flash_override(runtime, '/synthetic/oxe-server', RUST_SHA,
                                                    sidecar_path=sidecar.name, measured_peak_bytes=28 * GIB,
                                                    cache_gib=21, qualified_sha=RUST_SHA)
        override = updated['profiles'][ALIAS]['memlat_overrides']['oxe_flash']
        self.assertEqual(override['admission']['admitted_budget_bytes'], 29 * GIB)
        self.assertIn('21', override['oxe_args'])


if __name__ == '__main__':
    unittest.main()
