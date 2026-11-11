"""S3 (v0.1.1): the swap-tolerant guard's counter-only PSI stop. One avg10 sample under the threshold
resets the avg10 timer, so oscillating thrash never tripped it; real stall time per 60 s cannot be reset
by a dip. Also F127-7: no MemAvailable floor is reported in swap-tolerant mode. Model-free."""
import unittest

from snapdragon_ai_lab.runtime import memory_policy as policy
from test_swap_tolerant_guard import FLOOR, Exited, sample


def tracks(value=None):
    value = value or policy.swap_tolerant_policy(True)
    return {'full': policy.PressureTrack(value['psi_full_avg10_percent'], value['psi_full_seconds'],
                                         value['psi_counter_window_seconds']),
            'some': policy.PressureTrack(value['psi_some_avg10_percent'], value['psi_some_seconds'],
                                         value['psi_counter_window_seconds'])}


def oscillate(seconds, high, low, full_share, some_share, step=1.0, period=4):
    """avg10 alternates high/low every `period` samples; totals rise at the given real stall shares."""
    full_total = some_total = 0
    now = 0.
    for i in range(int(seconds / step) + 1):
        avg = high if (i // period) % 2 == 0 else low
        yield now, sample(psi_full_avg10=avg, psi_full_total_us=full_total,
                          psi_some_avg10=avg, psi_some_total_us=some_total)
        now += step
        full_total += int(full_share * step * 1e6)
        some_total += int(some_share * step * 1e6)


class CounterStop(unittest.TestCase):
    def run_policy(self, seconds, high, low, full_share, some_share, **kw):
        value = policy.swap_tolerant_policy(True)
        t = tracks(value)
        baseline = {'host_oom_kill': 7}
        for now, s in oscillate(seconds, high, low, full_share, some_share, **kw):
            reason = policy.evaluate_swap_tolerant(s, now, baseline, value, t, in_grace=False)
            if reason:
                return reason, now, t
        return None, None, t

    def test_oscillating_full_thrash_stops_on_the_counter(self):
        # REVIEW S3: avg10 alternating 40 / 9 with real full stalls at 25 % of wall time for 120 s.
        reason, at, t = self.run_policy(120, 40., 9., .25, .25)
        self.assertEqual(reason, 'host_memory_pressure_full_sustained')
        self.assertAlmostEqual(at, 60, delta=1)
        self.assertEqual(t['full'].tripped_by, 'counter')

    def test_the_avg10_track_alone_never_trips_on_that_pattern(self):
        value = policy.swap_tolerant_policy(True)
        track = policy.PressureTrack(value['psi_full_avg10_percent'], value['psi_full_seconds'])  # no long window
        for now, s in oscillate(300, 40., 9., .25, .25):
            self.assertFalse(track.update(s['psi_full_avg10'], s['psi_full_total_us'], now))

    def test_oscillating_some_thrash_stops_on_the_counter(self):
        reason, at, t = self.run_policy(120, 45., 20., .05, .35)
        self.assertEqual(reason, 'host_memory_pressure_some_sustained')
        self.assertAlmostEqual(at, 60, delta=1)
        self.assertEqual(t['some'].tripped_by, 'counter')

    def test_bursts_below_the_real_share_do_not_stop(self):
        # avg10 spikes to 40 % but real stalls are 8 % (full) and 25 % (some) of wall time.
        self.assertIsNone(self.run_policy(600, 40., 2., .08, .25)[0])

    def test_counter_needs_a_full_window(self):
        track = policy.PressureTrack(10., 20., 60.)
        self.assertFalse(track.update(0., 0, 0.))
        self.assertFalse(track.update(5., 59_000_000, 59.))   # 100 % stalls, but only 59 s of history
        self.assertTrue(track.update(5., 60_000_000, 60.))
        self.assertEqual(track.tripped_by, 'counter')

    def test_window_and_validation(self):
        with self.assertRaises(ValueError):
            policy.PressureTrack(10., 20., 10.)
        self.assertEqual(policy.swap_tolerant_policy({'enabled': True, 'psi_counter_window_seconds': 120})
                         ['psi_counter_window_seconds'], 120.)
        for bad in (10, 601, True, float('inf')):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, 'swap_tolerant_guard_invalid'):
                policy.swap_tolerant_policy({'enabled': True, 'psi_counter_window_seconds': bad})


class GuardWiring(unittest.TestCase):
    def guard(self, **overrides):
        self.now = 0.
        value = policy.swap_tolerant_policy({'enabled': True, **overrides})
        guard = policy.MemoryGuard(reader=sample, clock=lambda: self.now,
                                   minimum_available_bytes=FLOOR, swap_tolerant=value)
        guard.start(Exited()); guard._thread.join(timeout=1)
        return guard

    def test_guard_stops_oscillating_thrash_after_grace(self):
        guard = self.guard()
        reason = None
        for now, s in oscillate(200, 40., 9., .25, .05):
            self.now = now
            reason = guard._observe(s)
            if reason:
                break
        self.assertEqual(reason, 'host_memory_pressure_full_sustained')
        # grace ends at attach + 60 s; then a full 60 s counter window
        self.assertAlmostEqual(self.now, 120, delta=1.5)
        meta = guard.metadata()['swap_tolerant']
        self.assertEqual(meta['psi_full_tripped_by'], 'counter')
        self.assertGreaterEqual(meta['psi_full_counter_window_percent'], 10)
        self.assertGreaterEqual(meta['psi_full_counter_window_seconds'], 60)

    def test_counter_window_follows_config(self):
        guard = self.guard(psi_counter_window_seconds=120)
        reason = None
        for now, s in oscillate(300, 40., 9., .25, .05):
            self.now = now
            reason = guard._observe(s)
            if reason:
                break
        self.assertAlmostEqual(self.now, 180, delta=1.5)

    def test_f127_7_no_memavailable_floor_reported_in_swap_tolerant_mode(self):
        guard = self.guard()
        meta = guard.metadata()
        self.assertEqual(meta['mode'], policy.SWAP_TOLERANT_MODE)
        self.assertIsNone(meta['policy']['minimum_available_bytes'])
        self.assertEqual(meta['swap_tolerant']['last_resort_floor_bytes'], 1024**3)
        guard.leave_swap_tolerant()
        self.assertEqual(guard.metadata()['policy']['minimum_available_bytes'], FLOOR)


if __name__ == '__main__':
    unittest.main()
