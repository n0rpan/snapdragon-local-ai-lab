"""S2 (v0.1.1), lab side: an explicit 20/18 Flash cache preference is admitted on the MemTotal
(swap-tolerant) rule only when SwapFree can hold what must leave RAM; the controller never passes
--flash-cache-pref 20/18 to an engine otherwise (a v0.1.0 engine does not read swap). Model-free."""
import unittest

from snapdragon_ai_lab.runtime import flash_cache_pref as fcp, oxe_runtime
from test_flash_admission_rule import make_config, GIB
from test_swap_tolerant_guard import MEM_TOTAL


class Mirror(unittest.TestCase):
    def test_review_case_no_swap_is_not_swap_tolerant(self):
        # REVIEW S2: total 32, available 12, swap free 0 must not be PreferredSwapTolerant.
        for swap_free in (0, None):
            tier, reason = fcp.expected_tier(12 * GIB, '20', 16384, 32 * GIB, 2.0, swap_free)
            self.assertNotEqual(reason, 'preferred-swap-tolerant')
            self.assertEqual((tier, reason), (None, 'refuse'))

    def test_64_gib_host_with_39_in_use_and_no_swap_falls_back_to_memavailable(self):
        tier, reason = fcp.expected_tier(25 * GIB, '20', 16384, 64 * GIB, 3.0, 0)
        self.assertEqual(reason, 'preferred-does-not-fit')
        self.assertEqual(tier, 14)

    def test_swap_must_cover_what_leaves_ram_plus_margin(self):
        need = fcp.swap_needed_bytes(25 * GIB, 20)
        self.assertGreater(need, 0)
        margin = int(fcp.SWAP_MARGIN_GIB * GIB)
        self.assertFalse(fcp.swap_covers(25 * GIB, need + margin - 1, 20))
        self.assertTrue(fcp.swap_covers(25 * GIB, need + margin, 20))
        self.assertEqual(fcp.expected_tier(25 * GIB, '20', 16384, MEM_TOTAL, 2.0, need + margin),
                         (20, 'preferred-swap-tolerant'))
        # Swap for 18 but not for 20: the 18 tier on the MemTotal rule.
        need18 = fcp.swap_needed_bytes(25 * GIB, 18)
        self.assertEqual(fcp.expected_tier(25 * GIB, '20', 16384, MEM_TOTAL, 2.0, need18 + margin),
                         (18, 'preferred-swap-tolerant'))
        # Enough MemAvailable: nothing must leave RAM, the margin alone suffices.
        self.assertEqual(fcp.swap_needed_bytes(40 * GIB, 20), 0)
        self.assertTrue(fcp.swap_covers(40 * GIB, margin, 20))
        self.assertFalse(fcp.swap_covers(40 * GIB, 0, 20))

    def test_offered_prefs(self):
        self.assertEqual(fcp.offered_prefs(16 * GIB), ['auto', '20', '18', '16', '14'])
        for total in (0, None, -1, 1.5):
            self.assertEqual(fcp.offered_prefs(total), ['auto', '16', '14'])


class Launch(unittest.TestCase):
    def config(self, pref='20'):
        cfg = make_config(cache_gib=14, peak=22 * GIB)
        cfg['oxe_flash'].update(cache_pref=pref, desktop_floor_gib=2.0)
        return cfg

    @staticmethod
    def engine_pref(selected):
        args = selected['oxe_args']
        return args[args.index('--flash-cache-pref') + 1] if '--flash-cache-pref' in args else 'auto'

    def test_without_swap_the_engine_never_gets_20_or_18(self):
        for pref in ('20', '18'):
            with self.subTest(pref=pref):
                selected = oxe_runtime.flash_select(self.config(pref), 25 * GIB, 29 * GIB, env={},
                                                    mem_total=MEM_TOTAL, swap_free=0)
                self.assertEqual(self.engine_pref(selected), '14')
                self.assertEqual(selected['oxe_flash']['cache_pref'], pref)   # the user's choice is kept
                oxe_runtime._validate(selected)

    def test_with_swap_the_preference_passes_unchanged(self):
        selected = oxe_runtime.flash_select(self.config('20'), 25 * GIB, 29 * GIB, env={},
                                            mem_total=MEM_TOTAL, swap_free=16 * GIB)
        self.assertEqual(self.engine_pref(selected), '20')

    def test_memavailable_fit_keeps_20_even_without_swap(self):
        cfg = self.config('20')
        selected = oxe_runtime.flash_select(cfg, 31 * GIB, 64 * GIB, env={}, mem_total=64 * GIB, swap_free=0)
        self.assertEqual(self.engine_pref(selected), '20')

    def test_guard_is_standard_without_swap(self):
        cfg = self.config('20'); cfg['oxe_flash']['swap_tolerant_guard'] = True
        self.assertIsNone(oxe_runtime.flash_swap_tolerant_guard(cfg, 25 * GIB, MEM_TOTAL, env={}, swap_free=0))
        self.assertIsNone(oxe_runtime.flash_swap_tolerant_guard(cfg, 25 * GIB, MEM_TOTAL, env={}))
        self.assertIsNotNone(oxe_runtime.flash_swap_tolerant_guard(cfg, 25 * GIB, MEM_TOTAL, env={},
                                                                   swap_free=16 * GIB))


if __name__ == '__main__':
    unittest.main()
