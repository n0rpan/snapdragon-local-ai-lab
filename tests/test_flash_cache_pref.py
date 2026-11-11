"""O-27 Flash cache preference, controller side. Synthetic values only; never starts an engine or reads ~/.config."""
import unittest

from snapdragon_ai_lab.runtime import flash_cache_pref as fcp

GIB = 1024 ** 3
BASE = ['--engine', 'flash', '--model', '{model}', '--flash-threads', '8', '--flash-cache-gib', '14', '--reasoning', 'off']


class ReadPref(unittest.TestCase):
    def test_default_is_auto(self):
        self.assertEqual(fcp.read_pref({}, {}), 'auto')
        self.assertEqual(fcp.read_pref(None, None), 'auto')

    def test_profile_field_and_env_override(self):
        self.assertEqual(fcp.read_pref({}, {'cache_pref': 20}), '20')
        self.assertEqual(fcp.read_pref({fcp.ENV: '16'}, {'cache_pref': 20}), '16')
        self.assertEqual(fcp.read_pref({fcp.ENV: ' 20GiB '}, {}), '20')
        self.assertEqual(fcp.read_pref({fcp.ENV: '18gb'}, {}), '18')
        self.assertEqual(fcp.read_pref({fcp.ENV: 'AUTO'}, {'cache_pref': 14}), 'auto')
        self.assertEqual(fcp.read_pref({fcp.ENV: ''}, {'cache_pref': '14'}), '14')

    def test_invalid_values_raise(self):
        for bad in ('21', '12', 'big', '-16', 'gib'):
            with self.assertRaises(ValueError):
                fcp.read_pref({fcp.ENV: bad}, {})
        with self.assertRaises(ValueError):
            fcp.read_pref({}, {'cache_pref': 15})


class ApplyPref(unittest.TestCase):
    def test_auto_replaces_fixed_size(self):
        out = fcp.apply_pref(BASE, 'auto')
        self.assertEqual(out, ['--engine', 'flash', '--model', '{model}', '--flash-threads', '8', '--flash-cache-gib', 'auto',
                               '--reasoning', 'off'])
        self.assertEqual(BASE[7], '14')  # input untouched

    def test_each_preference(self):
        for p in ('20', '18', '16', '14'):
            out = fcp.apply_pref(BASE, p)
            self.assertEqual(out.count('--flash-cache-gib'), 1)
            self.assertEqual(out[out.index('--flash-cache-gib') + 1], 'auto')
            self.assertEqual(out[-2:], ['--flash-cache-pref', p])

    def test_idempotent_and_adds_missing(self):
        once = fcp.apply_pref(BASE, '20')
        self.assertEqual(fcp.apply_pref(once, '20'), once)
        self.assertEqual(fcp.apply_pref(once, 'auto').count('--flash-cache-pref'), 0)
        bare = ['--engine', 'flash']
        self.assertEqual(fcp.apply_pref(bare, '16'), ['--engine', 'flash', '--flash-cache-gib', 'auto', '--flash-cache-pref', '16'])


class Needs(unittest.TestCase):
    def test_needs_match_engine(self):
        # crates/qwen4exp/src/pf_plan.rs admit_need_matches_receipts: 30.125 / 28.125 / 26.125 / 24.125
        for t, n in ((20, 30.125), (18, 28.125), (16, 26.125), (14, 24.125)):
            self.assertAlmostEqual(fcp.tier_need_gib(t), n, places=9)
        self.assertLessEqual(abs(fcp.smallest_tier_need_bytes() - int(24.125 * GIB)), 1)

    def test_expected_tier_every_tier_and_fallback(self):
        g = lambda x: int(x * GIB)
        self.assertEqual(fcp.expected_tier(g(30.2)), (20, 'auto'))
        self.assertEqual(fcp.expected_tier(g(29.0)), (18, 'auto'))
        self.assertEqual(fcp.expected_tier(g(27.0)), (16, 'auto'))
        self.assertEqual(fcp.expected_tier(g(25.6)), (14, 'auto'))
        self.assertEqual(fcp.expected_tier(g(24.0)), (None, 'refuse'))
        self.assertEqual(fcp.expected_tier(g(30.2), '16'), (16, 'preferred'))
        self.assertEqual(fcp.expected_tier(g(25.6), '20'), (14, 'preferred-does-not-fit'))
        self.assertEqual(fcp.expected_tier(g(29.0), '20'), (18, 'preferred-does-not-fit'))
        self.assertEqual(fcp.expected_tier(g(24.0), '20'), (None, 'refuse'))


class SwapTolerant(unittest.TestCase):
    MT = 32215352 * 1024  # this machine's MemTotal

    def test_budgets_match_engine(self):
        # crates/qwen4exp/src/pf_plan.rs swap_tolerant_budget_terms: 20 -> -0.605 at floor 3, 18 -> +1.395
        mt = 30.72 * GIB
        self.assertAlmostEqual(fcp.swap_tolerant_budget_gib(mt, 20, 3.0), -0.605, places=6)
        self.assertAlmostEqual(fcp.swap_tolerant_budget_gib(mt, 18, 3.0), 1.395, places=6)

    def test_expected_tier_opt_in_only(self):
        g = lambda x: int(x * GIB)
        self.assertEqual(fcp.expected_tier(g(25.5), '20', mem_total_bytes=self.MT, desktop_floor_gib=2.0, swap_free_bytes=16 * GIB), (20, 'preferred-swap-tolerant'))
        self.assertEqual(fcp.expected_tier(g(25.5), '20', mem_total_bytes=self.MT, swap_free_bytes=16 * GIB), (18, 'preferred-swap-tolerant'))
        self.assertEqual(fcp.expected_tier(g(25.5), '20', mem_total_bytes=self.MT, desktop_floor_gib=5.0), (14, 'preferred-does-not-fit'))
        self.assertEqual(fcp.expected_tier(g(25.5), 'auto', mem_total_bytes=self.MT, desktop_floor_gib=0.0), (14, 'auto'))
        self.assertEqual(fcp.expected_tier(g(25.5), '16', mem_total_bytes=self.MT, desktop_floor_gib=0.0), (14, 'preferred-does-not-fit'))


class Status(unittest.TestCase):
    def props(self, **fc):
        return {'model_alias': 'x', 'engine_status': {'flash_cache': fc}}

    def test_fallback_label(self):
        s = fcp.status_from_props(self.props(cache_gib=14, pref_gib=20, reason='preferred-does-not-fit', refuse=False,
                                             margin_gib=1.4751, mem_available_gib=25.6))
        self.assertEqual(s['tier_gib'], 14)
        self.assertEqual(s['pref'], '20')
        self.assertEqual(s['label'], '14 GiB (preferred 20 GiB does not fit now)')
        self.assertEqual(s['margin_gib'], 1.48)
        self.assertEqual(s['choices'], ['auto', '20', '18', '16', '14'])

    def test_auto_and_preferred(self):
        self.assertEqual(fcp.status_from_props(self.props(cache_gib=16, pref_gib=None, reason='auto'))['label'],
                         '16 GiB (largest tier that fits)')
        self.assertEqual(fcp.status_from_props(self.props(cache_gib=20, pref_gib=20, reason='preferred'))['label'],
                         '20 GiB (your preference)')

    def test_swap_tolerant_label(self):
        self.assertEqual(fcp.status_from_props(self.props(cache_gib=20, pref_gib=20, reason='preferred-swap-tolerant'))['label'],
                         '20 GiB (your preference; idle desktop pages may go to swap)')

    def test_old_engine_or_garbage(self):
        self.assertIsNone(fcp.status_from_props({'model_alias': 'x'}))
        self.assertIsNone(fcp.status_from_props({'engine_status': None}))
        self.assertIsNone(fcp.status_from_props(self.props(cache_gib=14, reason='weird')))
        self.assertIsNone(fcp.status_from_props(None))


if __name__ == '__main__':
    unittest.main()
