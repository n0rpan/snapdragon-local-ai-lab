"""Flash admission rule: cache GiB + 5.87 GiB (1 GiB floor) or the measured peak + floor, whichever is larger.
Synthetic configs only; never starts an engine, loads weights or reads ~/.config."""
import copy
import importlib.util
from pathlib import Path
import unittest

from snapdragon_ai_lab.runtime import bigmoe_runtime, oxe_runtime, profile_policy

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('stage_flash_oxe_production', ROOT / 'tools/stage_flash_oxe_production.py')
stage_tool = importlib.util.module_from_spec(spec); spec.loader.exec_module(stage_tool)

GIB = 1024**3
ALIAS = stage_tool.ALIAS
RUST_SHA = 'd' * 64
SIDECAR = '/synthetic/flash/sidecar.bin'
SIDECAR_BYTES = 4096
OVERHEAD = oxe_runtime.FLASH_ADMISSION_OVERHEAD_BYTES
OVER_4_62 = int(4.62 * GIB)
POLICY = {'scope_max_bytes': 29 * GIB, 'host_floor_bytes': GIB}


def flash_profile():
    shards = [{'path': '/synthetic/flash/shard-%d.gguf' % i, 'bytes': size, 'sha256': digest}
              for i, (size, digest) in enumerate(bigmoe_runtime.PINNED, 1)]
    return {'ready': True, 'backend_kind': bigmoe_runtime.KIND, 'model_alias': ALIAS,
            'display_name': 'Flash synthetic', 'model_path': shards[0]['path'], 'expected_bytes': shards[0]['bytes'],
            'server_binary': '/synthetic/bmoe-server', 'server_sha256': 'c' * 64, 'sha256': '9' * 64,
            'context_size': 16384, 'threads': 4, 'gpu_layers': 0, 'backend': 'cpu', 'extra_args': [],
            'environment': {}, 'prefix_cache': {'enabled': False},
            'bigmoe': {'revision': bigmoe_runtime.REVISION, 'cache_mib': 12288, 'models_dir': '/synthetic/flash',
                       'data_dir': '/synthetic/data', 'shards': shards, 'microbatch_size': 4096},
            'memlat_overrides': {'oxe_flash': stage_tool.flash_override(
                '/synthetic/oxe-server', RUST_SHA, sidecar_path=SIDECAR, sidecar_bytes=SIDECAR_BYTES,
                measured_peak_bytes=20 * GIB + OVER_4_62, cache_gib=20)}}


def make_config(cache_gib=20, peak=20 * GIB + OVER_4_62, budget=29 * GIB, policy=True, fallback_peak=None):
    profile = flash_profile()
    if policy:
        profile['flash_memory'] = dict(POLICY)
    override = stage_tool.flash_override(
        '/synthetic/oxe-server', RUST_SHA, sidecar_path=SIDECAR, sidecar_bytes=SIDECAR_BYTES,
        measured_peak_bytes=peak, cache_gib=cache_gib, budget_bytes=budget, fallback_peak_bytes=fallback_peak)
    # profile_policy does not carry session_reserve_bytes yet; tests set it on the resolved config.
    override.pop('session_reserve_bytes', None)
    profile['memlat_overrides']['oxe_flash'] = override
    return profile_policy.resolve(profile, available=True, oxe_flash_enabled=True)


class AdmissionRule(unittest.TestCase):
    def test_twenty_gib_cache_needs_cache_plus_587_at_one_gib_floor(self):
        cfg = make_config(cache_gib=20, peak=20 * GIB + OVER_4_62)
        required = 20 * GIB + OVERHEAD + GIB
        self.assertEqual(oxe_runtime.flash_required_available(cfg), required)
        self.assertEqual(oxe_runtime.flash_required_available(cfg), 20 * GIB + int(4.87 * GIB) + GIB)
        self.assertIsNone(oxe_runtime.flash_admit(cfg, required))
        with self.assertRaisesRegex(ValueError, 'insufficient_available_memory'):
            oxe_runtime.flash_admit(cfg, required - 1)

    def test_larger_measured_peak_wins_the_max(self):
        cfg = make_config(cache_gib=20, peak=27 * GIB)
        self.assertEqual(oxe_runtime.flash_required_available(cfg), 27 * GIB + GIB)

    def test_session_reserve_is_added_and_validated(self):
        cfg = make_config(cache_gib=20, peak=20 * GIB + OVER_4_62)
        cfg['oxe_flash']['session_reserve_bytes'] = 2 * GIB
        self.assertEqual(oxe_runtime.flash_required_available(cfg), 20 * GIB + OVERHEAD + GIB + 2 * GIB)
        oxe_runtime._validate(cfg)
        for value in (0, 4 * GIB):
            with self.subTest(value=value):
                cfg['oxe_flash']['session_reserve_bytes'] = value
                oxe_runtime._validate(cfg)
        for value in (True, False, -1, 4 * GIB + 1, 1.5, float(GIB), '1'):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'session_reserve_invalid'):
                cfg['oxe_flash']['session_reserve_bytes'] = value
                oxe_runtime._validate(cfg)

    def test_without_flash_memory_policy_floor_is_one_and_a_half_gib(self):
        cfg = make_config(cache_gib=20, peak=20 * GIB + OVER_4_62, budget=28 * GIB, policy=False)
        self.assertEqual(oxe_runtime.flash_limits(cfg)[0], 1536 * 1024**2)
        self.assertEqual(oxe_runtime.flash_required_available(cfg), 20 * GIB + OVERHEAD + 1536 * 1024**2)


class FallbackRule(unittest.TestCase):
    PRIMARY_PEAK = 20 * GIB + OVER_4_62
    FALLBACK_PEAK = 18 * GIB + OVER_4_62

    def config(self):
        return make_config(cache_gib=20, peak=self.PRIMARY_PEAK, fallback_peak=self.FALLBACK_PEAK)

    def test_smallest_requirement_uses_the_18_gib_rule(self):
        cfg = self.config()
        small = 18 * GIB + OVERHEAD + GIB
        self.assertEqual(oxe_runtime.flash_smallest_required_available(cfg), small)
        self.assertEqual(oxe_runtime.flash_required_available(cfg), 20 * GIB + OVERHEAD + GIB)

    def test_fallback_chosen_between_18_and_20_rules(self):
        cfg = self.config(); before = copy.deepcopy(cfg)
        chosen = oxe_runtime.flash_select(cfg, 24 * GIB)
        self.assertEqual(cfg, before)
        flash = chosen['oxe_flash']
        self.assertEqual((flash['cache_gib'], flash['fallback_active']), (18, True))
        self.assertEqual(oxe_runtime.flash_required_available(chosen), 18 * GIB + OVERHEAD + GIB)

    def test_below_the_18_rule_raises(self):
        cfg = self.config()
        small = 18 * GIB + OVERHEAD + GIB
        with self.assertRaises(ValueError):
            oxe_runtime.flash_select(cfg, small - 1)
        chosen = oxe_runtime.flash_select(cfg, small)
        self.assertEqual((chosen['oxe_flash']['cache_gib'], chosen['oxe_flash']['fallback_active']), (18, True))


if __name__ == '__main__':
    unittest.main()
