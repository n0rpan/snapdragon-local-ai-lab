"""Opt-in native cache selection; no files, engines or model sessions."""
import copy
import unittest
from unittest.mock import patch
from snapdragon_ai_lab.runtime import flash_cache_pref as fcp, oxe_runtime, profile_policy, model_catalog, controller
from test_flash_admission_rule import make_config, flash_profile, GIB


class ControllerPreference(unittest.TestCase):
    def config(self):
        return make_config(cache_gib=14, peak=22 * GIB)

    def test_fixed_path_is_same_object_and_argv(self):
        cfg = self.config(); before = copy.deepcopy(cfg)
        self.assertIs(oxe_runtime.flash_select(cfg, 26 * GIB, env={}), cfg)
        self.assertEqual(cfg, before)
        self.assertFalse(fcp.enabled({}, cfg['oxe_flash']))

    def test_profile_fields_survive_resolution(self):
        profile = flash_profile()
        profile['flash_memory'] = {'scope_max_bytes': 29 * GIB, 'host_floor_bytes': GIB}
        profile['memlat_overrides']['oxe_flash'].update(cache_pref='20', desktop_floor_gib=2.0)
        cfg = profile_policy.resolve(profile, available=True, oxe_flash_enabled=True)
        self.assertEqual(cfg['oxe_flash']['cache_pref'], '20')
        self.assertEqual(cfg['oxe_flash']['desktop_floor_gib'], 2.0)

    def test_twenty_uses_total_and_exact_native_flags(self):
        cfg = self.config(); before = copy.deepcopy(cfg)
        cfg['oxe_flash'].update(cache_pref='20', desktop_floor_gib=2.0)
        selected = oxe_runtime.flash_select(cfg, 25 * GIB, 29 * GIB, env={}, mem_total=int(30.72 * GIB), swap_free=16 * GIB)
        args = selected['oxe_args']
        self.assertEqual(args[args.index('--flash-cache-gib') + 1], 'auto')
        self.assertEqual(args[-4:], ['--flash-cache-pref', '20', '--flash-desktop-floor-gib', '2.0'])
        self.assertEqual(selected['flash_memory'], before['flash_memory'])
        self.assertEqual(selected['oxe_flash']['admission'], before['oxe_flash']['admission'])
        self.assertEqual(cfg['oxe_args'], before['oxe_args'])
        oxe_runtime._validate(selected)

    def test_desktop_floor_replaces_existing_native_option(self):
        cfg = self.config(); cfg['oxe_flash'].update(cache_pref='20', desktop_floor_gib=2.0)
        for existing in (['--flash-desktop-floor-gib', '3.0'], ['--flash-desktop-floor-gib=3.0']):
            original = copy.deepcopy(cfg)
            original['oxe_args'] += existing
            selected = oxe_runtime.flash_select(original, 25 * GIB, env={}, mem_total=31 * GIB, swap_free=16 * GIB)
            self.assertEqual(selected['oxe_args'].count('--flash-desktop-floor-gib'), 1)
            self.assertEqual(selected['oxe_args'][-2:], ['--flash-desktop-floor-gib', '2.0'])

    def test_env_override_and_conservative_preferences(self):
        cfg = self.config(); cfg['oxe_flash'].update(cache_pref='20', desktop_floor_gib=2.0)
        for pref in ('auto', '16', '14'):
            with self.subTest(pref=pref), self.assertRaises(ValueError):
                oxe_runtime.flash_select(cfg, 24 * GIB, env={fcp.ENV: pref}, mem_total=40 * GIB)
        selected = oxe_runtime.flash_select(cfg, 25 * GIB, env={fcp.ENV: '18'}, mem_total=31 * GIB)
        self.assertEqual(selected['oxe_flash']['cache_pref'], '18')

    def test_floor_cap_and_missing_total_fail_closed(self):
        cfg = self.config(); cfg['oxe_flash'].update(cache_pref='20', desktop_floor_gib=2.0)
        for avail, total, cap in ((GIB - 1, 31 * GIB, 29 * GIB),
                                  (23 * GIB, None, 29 * GIB),
                                  (25 * GIB, 31 * GIB, 28 * GIB)):
            with self.subTest(avail=avail, total=total, cap=cap), self.assertRaises(ValueError):
                oxe_runtime.flash_select(cfg, avail, cap, env={}, mem_total=total, swap_free=16 * GIB)
        for floor in (True, float('nan'), -1, 17):
            cfg['oxe_flash']['desktop_floor_gib'] = floor
            with self.assertRaises(ValueError):
                oxe_runtime.flash_select(cfg, 25 * GIB, env={}, mem_total=31 * GIB)


class CacheMetadata(unittest.TestCase):
    def test_fixed_flash_reports_qualified_working_set(self):
        cfg = make_config(cache_gib=14, peak=22 * GIB)
        with patch.dict('os.environ', {}, clear=True):
            row = model_catalog.row('flash', {}, cfg, selected='flash', loaded_alias=None, state='unloaded')
        self.assertEqual(row['memory_gib'], 22.0)
        self.assertEqual(row['memory_basis'], 'qualified_working_set')

    def test_opt_in_estimate_is_not_free_ram_requirement(self):
        cfg = make_config(cache_gib=14, peak=22 * GIB)
        cfg['oxe_flash'].update(cache_pref='20', desktop_floor_gib=2.0)
        with patch.dict('os.environ', {}, clear=True):
            row = model_catalog.row('flash', {}, cfg, selected='flash', loaded_alias=None, state='unloaded')
        self.assertIsNone(row['memory_gib'])
        self.assertNotIn('tier_gib', row['flash_cache'])
        self.assertEqual(row['flash_cache']['estimated_working_set_gib'], 28.325)
        self.assertNotIn('reason', row['flash_cache'])

    def test_actual_props_are_authoritative_and_static_status_never_calls_backend(self):
        props = {'engine_status': {'flash_cache': {'cache_gib': 18, 'pref_gib': 20,
                    'reason': 'preferred-swap-tolerant', 'margin_gib': 1.4}}}
        actual = fcp.status_from_props(props)
        cfg = make_config(cache_gib=14, peak=22 * GIB)
        cfg['oxe_flash'].update(cache_pref='20', desktop_floor_gib=2.0)
        self.assertEqual(fcp.catalog_status(cfg, actual)['estimated_working_set_gib'], 26.325)
        with patch.object(controller.Manager, 'reaper'), patch.object(controller, 'CONFIG') as config_path:
            config_path.read_text.return_value = '{}'
            config_path.parent.__truediv__.return_value.read_text.return_value = '{}'
            manager = controller.Manager()
            try:
                manager.flash_cache_status = actual
                with patch.object(manager, 'config', return_value=cfg), patch.object(manager, 'eligible', return_value=True), \
                     patch.object(controller.http.client, 'HTTPConnection', side_effect=AssertionError('metadata must not wake')):
                    self.assertEqual(manager.status()['flash_cache']['tier_gib'], 18)
            finally:
                manager.flash_cache_status = None
                manager.close()

    def test_props_rejects_invalid_nonfinite_tiers(self):
        for tier in (True, 19, float('nan'), float('inf')):
            self.assertIsNone(fcp.status_from_props({'engine_status': {'flash_cache': {
                'cache_gib': tier, 'pref_gib': 20, 'reason': 'preferred'}}}))
