"""FLASH-PREFIX-1: the Flash prefix store switch (oxe_flash.prefix_store); never starts an engine, loads weights or reads the real home."""
import copy
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from snapdragon_ai_lab.runtime import bigmoe_runtime, oxe_runtime, profile_policy

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
PREFIX_SUBDIR = ('.cache', 'local-ai', 'snapdragon-engine', 'prefix-flash')


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


def profile_with(setting=None, extra_args=()):
    """A flash profile with an optional prefix_store setting and extra oxe_args appended to the template."""
    profile = flash_profile()
    flash = profile['memlat_overrides']['oxe_flash']
    if setting is not None:
        flash['prefix_store'] = setting
    flash['oxe_args'] = [*flash['oxe_args'], *extra_args]
    return profile


def resolve_flash(profile):
    with patch.dict(os.environ, FLASH_ON, clear=True):
        return profile_policy.resolve(profile, available=True)


class FlashPrefixStore(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        home = patch.object(Path, 'home', return_value=self.tmp)
        home.start()
        self.addCleanup(home.stop)

    def command(self, result, port=8093):
        with patch.dict(os.environ, FLASH_ON, clear=True):
            return oxe_runtime.command(result, port)

    def test_default_without_key_is_off(self):
        original = flash_profile()
        result = resolve_flash(original)
        with patch.dict(os.environ, FLASH_ON, clear=True):
            self.assertTrue(oxe_runtime.eligible(result))
        self.assertEqual(oxe_runtime.flash_prefix_args(result), [])
        self.assertNotIn('--prefix-cache', self.command(result))
        self.assertNotIn('prefix_store', result['oxe_flash'])
        self.assertEqual(original, flash_profile())

    def test_disabled_setting_matches_default(self):
        result = resolve_flash(profile_with({'enabled': False}))
        with patch.dict(os.environ, FLASH_ON, clear=True):
            self.assertTrue(oxe_runtime.eligible(result))
        self.assertEqual(oxe_runtime.flash_prefix_args(result), [])
        self.assertNotIn('--prefix-cache', self.command(result))
        self.assertEqual(result['oxe_flash']['prefix_store'],
                         {**profile_policy.FLASH_PREFIX_DEFAULTS, 'enabled': False})

    def test_enabled_true_fills_defaults_and_appends_flags(self):
        result = resolve_flash(profile_with({'enabled': True}))
        self.assertEqual(result['oxe_flash']['prefix_store'],
                         {'enabled': True, 'max_bytes': 2 * 1024**3, 'min_tokens': 1024, 'ttl_hours': 168})
        command = self.command(result)
        store = str(self.tmp.joinpath(*PREFIX_SUBDIR))
        self.assertEqual(command[-len(oxe_runtime.FLASH_PREFIX_RAW_FLAGS) * 2:],
                         ['--prefix-cache', store, '--prefix-cache-max-bytes', '2147483648',
                          '--prefix-cache-min-tokens', '1024', '--prefix-cache-ttl-h', '168'])
        self.assertEqual(command[:5], [result['server_binary'], '--host', '127.0.0.1', '--port', '8093'])
        self.assertEqual(command[1:3], ['--host', '127.0.0.1'])

    def test_explicit_values_pass_through(self):
        setting = {'enabled': True, 'max_bytes': 1024**3, 'min_tokens': 2048, 'ttl_hours': 0}
        result = resolve_flash(profile_with(setting))
        self.assertEqual(result['oxe_flash']['prefix_store'], setting)
        self.assertEqual(oxe_runtime.flash_prefix_args(result, home=self.tmp),
                         ['--prefix-cache', str(self.tmp.joinpath(*PREFIX_SUBDIR)),
                          '--prefix-cache-max-bytes', str(GIB), '--prefix-cache-min-tokens', '2048',
                          '--prefix-cache-ttl-h', '0'])

    def test_malformed_settings_rejected_by_resolve(self):
        bad = {
            'not a dict': 'yes',
            'unknown key': {'enabled': True, 'bogus': 1},
            'enabled missing': {'max_bytes': 2 * 1024**3},
            "enabled 'yes'": {'enabled': 'yes'},
            'max_bytes 100': {'enabled': True, 'max_bytes': 100},
            'max_bytes 17 GiB': {'enabled': True, 'max_bytes': 17 * GIB},
            'min_tokens 10': {'enabled': True, 'min_tokens': 10},
            'ttl_hours -1': {'enabled': True, 'ttl_hours': -1},
            'ttl_hours True': {'enabled': True, 'ttl_hours': True},
        }
        for label, setting in bad.items():
            with self.subTest(case=label):
                with self.assertRaises(ValueError) as cm:
                    resolve_flash(profile_with(setting))
                self.assertEqual(cm.exception.args, ('memlat_oxe_flash_prefix_store_invalid',))

    def test_raw_prefix_flags_in_argv_make_config_ineligible(self):
        for flag in ('--prefix-cache', '--prefix-cache=/x', '--prefix-cache-ttl-h', '--prefix-cache-ttl-h=24'):
            with self.subTest(flag=flag):
                result = resolve_flash(profile_with(None, extra_args=[flag, '/x']
                                                    if '=' not in flag else [flag]))
                with patch.dict(os.environ, FLASH_ON, clear=True):
                    self.assertFalse(oxe_runtime.eligible(result))
                    with self.assertRaises(ValueError) as cm:
                        oxe_runtime._validate(result)
                self.assertEqual(cm.exception.args, ('oxe_flash_prefix_flags_only_via_profile_key',))

    def test_store_with_unqualified_warmup_or_session_reads_is_refused(self):
        for extra in (['--flash-warmup', '1'], ['--flash-warmup=1'], ['--flash-hotset', 'default'],
                      ['--session-cache', '/x']):
            with self.subTest(extra=extra):
                result = resolve_flash(profile_with({'enabled': True}, extra_args=extra))
                with patch.dict(os.environ, FLASH_ON, clear=True):
                    with self.assertRaises(ValueError) as cm:
                        oxe_runtime._validate(result)
                self.assertEqual(cm.exception.args, ('oxe_flash_prefix_store_with_unqualified_warmup',))

    def test_store_with_warmup_zero_is_allowed(self):
        result = resolve_flash(profile_with({'enabled': True}, extra_args=['--flash-warmup', '0']))
        with patch.dict(os.environ, FLASH_ON, clear=True):
            oxe_runtime._validate(result)
            self.assertTrue(oxe_runtime.eligible(result))
        self.assertIn('--prefix-cache', self.command(result))

    def test_prefix_dir_on_weights_partition_is_refused(self):
        result = resolve_flash(profile_with({'enabled': True}))
        home = self.tmp / 'flash'
        result['oxe_flash']['sidecar_path'] = str(home / 'sidecar.bin')
        with self.assertRaises(ValueError) as cm:
            oxe_runtime.flash_prefix_dir(result, home=home)
        self.assertEqual(cm.exception.args, ('oxe_flash_prefix_dir_on_weights_partition',))

    def test_prefix_dir_elsewhere_is_accepted(self):
        result = resolve_flash(profile_with({'enabled': True}))
        home = self.tmp / 'home'
        result['oxe_flash']['sidecar_path'] = str(self.tmp / 'flash' / 'sidecar.bin')
        self.assertEqual(oxe_runtime.flash_prefix_dir(result, home=home), home.joinpath(*PREFIX_SUBDIR))

    def test_resolved_config_is_detached_from_profile(self):
        profile = profile_with({'enabled': True})
        snapshot = copy.deepcopy(profile)
        result = resolve_flash(profile)
        result['oxe_flash']['prefix_store']['max_bytes'] = 1
        result['oxe_flash']['prefix_store']['enabled'] = False
        self.assertEqual(profile, snapshot)
        self.assertEqual(profile['memlat_overrides']['oxe_flash']['prefix_store'], {'enabled': True})
        self.assertEqual(profile_policy.FLASH_PREFIX_DEFAULTS['max_bytes'], 2 * 1024**3)
        self.assertEqual(profile_policy.FLASH_PREFIX_DEFAULTS['enabled'], False)


if __name__ == '__main__':
    unittest.main()
