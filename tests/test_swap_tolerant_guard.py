"""F127 (O-30): opt-in swap-tolerant MemoryGuard mode. Model-free: fake PSI / meminfo / OOM samples only."""
import copy
import functools
import unittest
from unittest.mock import patch

from snapdragon_ai_lab.runtime import memory_policy as policy, oxe_runtime, profile_policy
from snapdragon_ai_lab.runtime import flash_cache_pref as fcp
from test_flash_admission_rule import make_config, flash_profile, GIB

FLOOR = policy.FLASH_HOST_FLOOR_BYTES
MEM_TOTAL = int(30.72 * GIB)        # this machine (064 block 0)
TRIP_AVAILABLE = 1070514176         # 064 0905: MemAvailable at the 1 GiB floor trip


def sample(**changes):
    """An extended (swap-tolerant) sample: calm host, lots of swap left."""
    value = {'available_bytes': 4 * GIB, 'swap_free_bytes': 8 * GIB,
             'psi_full_avg10': 0., 'psi_full_total_us': 0,
             'psi_some_avg10': 0., 'psi_some_total_us': 0, 'host_oom_kill': 7,
             'swap_current_bytes': 0, 'swap_max': '0', 'memory_high': 'max', 'memory_max': str(29 * GIB),
             'events': {'high': 0, 'max': 0, 'oom': 0, 'oom_kill': 0, 'oom_group_kill': 0},
             'pids_max_events': 0}
    value.update(changes)
    return value


class Exited:
    """An already-exited child: start() samples once and its thread returns without signalling."""
    pid = 4242
    exit = 0
    def poll(self): return self.exit
    def terminate(self): raise AssertionError('must not signal')
    kill = terminate


class Pressure:
    """Fake /proc/pressure/memory: avg10 held at a level, totals growing at that real stall rate."""
    def __init__(self):
        self.full_total = self.some_total = 0
        self.at = 0.

    def advance(self, now, full, some):
        dt = now - self.at
        self.full_total += int(full / 100 * dt * 1e6)
        self.some_total += int(some / 100 * dt * 1e6)
        self.at = now
        return {'psi_full_avg10': full, 'psi_full_total_us': self.full_total,
                'psi_some_avg10': some, 'psi_some_total_us': self.some_total}


class PolicyValues(unittest.TestCase):
    def test_off_by_default_and_defaults_from_064(self):
        self.assertIsNone(policy.swap_tolerant_policy(None))
        self.assertIsNone(policy.swap_tolerant_policy(False))
        self.assertIsNone(policy.swap_tolerant_policy({'enabled': False, 'grace_seconds': 30}))
        self.assertEqual(policy.swap_tolerant_policy(True), {
            'psi_full_avg10_percent': 10.0, 'psi_full_seconds': 20.0,
            'psi_some_avg10_percent': 30.0, 'psi_some_seconds': 30.0,
            'grace_seconds': 60.0, 'last_resort_floor_bytes': GIB, 'psi_counter_window_seconds': 60.0})

    def test_overrides_are_config_values(self):
        value = policy.swap_tolerant_policy({'enabled': True, 'psi_full_avg10_percent': 15,
                                             'psi_some_seconds': 45.5, 'last_resort_floor_bytes': 2 * GIB})
        self.assertEqual(value['psi_full_avg10_percent'], 15.0)
        self.assertEqual(value['psi_some_seconds'], 45.5)
        self.assertEqual(value['last_resort_floor_bytes'], 2 * GIB)
        self.assertEqual(value['grace_seconds'], 60.0)

    def test_malformed_values_fail_closed(self):
        for bad in ('on', 1, [], {}, {'enabled': 'yes'}, {'enabled': True, 'unknown': 1},
                    {'enabled': True, 'psi_full_avg10_percent': True},
                    {'enabled': True, 'psi_full_avg10_percent': 0.5},
                    {'enabled': True, 'psi_some_avg10_percent': 101},
                    {'enabled': True, 'psi_full_seconds': float('nan')},
                    {'enabled': True, 'grace_seconds': 301},
                    {'enabled': True, 'last_resort_floor_bytes': 1.5 * GIB},
                    {'enabled': True, 'last_resort_floor_bytes': 100 * 1024**2}):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, 'swap_tolerant_guard_invalid'):
                policy.swap_tolerant_policy(bad)
        with self.assertRaisesRegex(ValueError, 'swap_tolerant_guard_invalid'):
            policy.MemoryGuard(reader=sample, swap_tolerant={'grace_seconds': 60.0})

    def test_parsers(self):
        psi = ('some avg10=27.31 avg60=5.00 avg300=1.00 total=123\n'
               'full avg10=27.16 avg60=4.00 avg300=1.00 total=99\n')
        self.assertEqual(policy.parse_psi(psi, 'some'), (27.31, 123))
        self.assertEqual(policy.parse_psi_full(psi), (27.16, 99))
        self.assertEqual(policy.parse_swap_free_bytes('SwapTotal: 16 kB\nSwapFree:  8 kB\n'), 8192)
        self.assertEqual(policy.parse_host_oom_kill('oom_kill_x 9\noom_kill 3\n'), 3)
        for bad in ('', 'SwapFree: 8 MB\n'):
            with self.assertRaises(ValueError):
                policy.parse_swap_free_bytes(bad)
        with self.assertRaises(ValueError):
            policy.parse_host_oom_kill('pswpout 1\n')


class SwapTolerantGuard(unittest.TestCase):
    def guard(self, initial=None, *, loading=None, **overrides):
        self.now = 0.
        value = policy.swap_tolerant_policy({'enabled': True, **overrides})
        return policy.MemoryGuard(reader=lambda: copy.deepcopy(initial or sample()), clock=lambda: self.now,
                                  loading_timeout_seconds=loading, minimum_available_bytes=FLOOR,
                                  swap_tolerant=value)

    def attach(self, guard):
        guard.start(Exited())
        guard._thread.join(timeout=1)
        self.assertFalse(guard._thread.is_alive())

    def test_064_trip_point_does_not_stop_but_standard_mode_still_does(self):
        trip = sample(available_bytes=TRIP_AVAILABLE, psi_full_avg10=.29, psi_some_avg10=.49)
        guard = self.guard()
        self.attach(guard)
        self.now = 120
        self.assertIsNone(guard._observe(trip))
        self.assertEqual(guard.mode(), 'swap-tolerant')
        self.assertEqual(guard.metadata()['mode'], 'swap-tolerant')
        # Unchanged standard Flash guard on the same sample.
        standard = policy.MemoryGuard(reader=lambda: sample(), clock=lambda: 0., minimum_available_bytes=FLOOR)
        self.assertEqual(standard.mode(), 'standard')
        self.assertEqual(standard._observe(trip), 'host_available_below_floor')

    def test_last_resort_floor_counts_swap_and_is_never_in_grace(self):
        for loading in (None, 300):
            with self.subTest(loading=loading):
                guard = self.guard(loading=loading)
                exact = sample(available_bytes=GIB // 4, swap_free_bytes=GIB - GIB // 4)
                self.assertIsNone(guard._observe(exact))
                self.assertEqual(guard._observe(sample(available_bytes=GIB // 4, swap_free_bytes=GIB - GIB // 4 - 1)),
                                 'host_memory_and_swap_exhausted')
                self.assertEqual(guard._observe(sample()), 'host_memory_and_swap_exhausted')  # latched
        guard = self.guard(last_resort_floor_bytes=2 * GIB)
        self.assertEqual(guard._observe(sample(available_bytes=GIB, swap_free_bytes=GIB - 1)),
                         'host_memory_and_swap_exhausted')

    def test_engine_swap_must_stay_zero(self):
        guard = self.guard(loading=300)
        self.assertEqual(guard._observe(sample(swap_current_bytes=4096)), 'own_swap_nonzero')
        with self.assertRaisesRegex(policy.MemoryPolicyError, 'own_swap_limit_not_zero'):
            self.guard(sample(swap_max='max'))

    def test_scope_oom_events_stop(self):
        for key in ('oom', 'oom_kill', 'oom_group_kill'):
            with self.subTest(key=key):
                guard = self.guard(loading=300)
                value = sample()
                value['events'][key] = 1
                self.assertEqual(guard._observe(value), 'cgroup_oom')

    def test_global_oom_kill_stops_and_history_does_not(self):
        guard = self.guard(sample(host_oom_kill=40), loading=300)
        self.assertIsNone(guard._observe(sample(host_oom_kill=40)))
        self.assertEqual(guard._observe(sample(host_oom_kill=41)), 'host_oom_kill')

    def test_global_oom_baseline_survives_scope_move(self):
        from pathlib import Path
        initial = sample(host_oom_kill=40)
        initial['events']['oom_kill'] = 5          # the controller unit's history
        guard = self.guard(initial)
        guard._process = Exited()
        guard._home_cgroup = Path('/sys/fs/cgroup/unit')
        reads = []
        def fake_read(cgroup, extended=False):
            reads.append(extended)
            return sample(host_oom_kill=40)
        with patch.object(policy, 'process_cgroup', return_value=Path('/sys/fs/cgroup/scope')), \
                patch.object(policy, 'read_sample', side_effect=fake_read):
            moved = guard._read_followed()
        self.assertEqual(reads, [True])
        self.assertEqual(guard._initial_events['host_oom_kill'], 40)
        self.assertEqual(guard._initial_events['oom_kill'], 0)
        self.assertIsNone(guard._observe(moved))
        self.assertEqual(guard._observe(sample(host_oom_kill=41)), 'host_oom_kill')

    def test_missing_extended_metadata_fails_closed(self):
        for key in ('swap_free_bytes', 'psi_some_avg10', 'host_oom_kill'):
            with self.subTest(key=key):
                value = sample()
                del value[key]
                with self.assertRaisesRegex(policy.MemoryPolicyError, 'memory_metadata_unavailable'):
                    self.guard(value)

    def _run_pressure(self, guard, start, end, full, some, pressure, step=.25):
        reason, at = None, start
        while at <= end + 1e-9 and reason is None:
            self.now = at
            reason = guard._observe(sample(**pressure.advance(at, full, some)))
            at = round(at + step, 6)
        return reason, self.now

    def test_load_spike_inside_grace_window_does_not_stop(self):
        # 064: reclaim/load PSI peaked at ~27 % full/some; ready at 40 s, grace ends at attach + 60 s.
        guard = self.guard(loading=300)
        self.attach(guard)
        pressure = Pressure()
        reason, _ = self._run_pressure(guard, .25, 40, 27., 27.3, pressure)
        self.assertIsNone(reason)
        guard.mark_ready()
        reason, _ = self._run_pressure(guard, 40.25, 59.75, 27., 27.3, pressure)
        self.assertIsNone(reason)
        self.assertIsNone(guard.failure())

    def test_sustained_full_after_grace_stops_at_grace_plus_Y(self):
        guard = self.guard(loading=300)
        self.attach(guard)
        guard.mark_ready()
        pressure = Pressure()
        reason, at = self._run_pressure(guard, .25, 200, 12., 12., pressure)
        self.assertEqual(reason, 'host_memory_pressure_full_sustained')
        self.assertAlmostEqual(at, 60 + 20, delta=.5)

    def test_sustained_some_stops_at_W(self):
        guard = self.guard()
        self.attach(guard)
        pressure = Pressure()
        reason, at = self._run_pressure(guard, .25, 200, 2., 35., pressure)
        self.assertEqual(reason, 'host_memory_pressure_some_sustained')
        self.assertAlmostEqual(at, 60 + 30, delta=.5)

    def test_064_loaded_phase_levels_do_not_stop(self):
        # Loaded maxima 16.47 % (some = full), full > 10 % for < 7.1 s: bursts of 7 s at 16.47 % between calm.
        guard = self.guard()
        self.attach(guard)
        pressure = Pressure()
        at = 60.
        for _ in range(10):
            reason, _ = self._run_pressure(guard, at + .25, at + 7, 16.47, 16.47, pressure)
            self.assertIsNone(reason)
            reason, _ = self._run_pressure(guard, at + 7.25, at + 30, 3., 5., pressure)
            self.assertIsNone(reason)
            at += 30
        self.assertIsNone(guard.failure())

    def test_avg10_without_real_stalls_is_not_corroborated(self):
        guard = self.guard()
        self.attach(guard)
        reason = None
        for i in range(1, 400):
            self.now = 60 + i * .25
            reason = guard._observe(sample(psi_full_avg10=50., psi_some_avg10=50.))   # totals never grow
            self.assertIsNone(reason)
        meta = guard.metadata()['swap_tolerant']
        self.assertGreater(meta['psi_full_high_seconds'], 20)
        self.assertEqual(meta['psi_full_window_percent'], 0)

    def test_a_dip_below_threshold_restarts_the_count(self):
        guard = self.guard()
        self.attach(guard)
        pressure = Pressure()
        self.assertIsNone(self._run_pressure(guard, 60.25, 79, 12., 12., pressure)[0])
        self.assertIsNone(self._run_pressure(guard, 79.25, 79.25, 9.99, 12., pressure)[0])
        reason, at = self._run_pressure(guard, 79.5, 200, 12., 12., pressure)
        self.assertEqual(reason, 'host_memory_pressure_full_sustained')
        self.assertAlmostEqual(at, 79.5 + 20, delta=.5)

    def test_thresholds_follow_config(self):
        guard = self.guard(psi_full_avg10_percent=20, psi_full_seconds=5, grace_seconds=0)
        self.attach(guard)
        pressure = Pressure()
        self.assertIsNone(self._run_pressure(guard, .25, 30, 15., 15., pressure)[0])
        reason, at = self._run_pressure(guard, 30.25, 60, 25., 25., pressure)
        self.assertEqual(reason, 'host_memory_pressure_full_sustained')
        self.assertAlmostEqual(at, 30.25 + 5, delta=.5)

    def test_loading_timeout_still_applies(self):
        guard = self.guard(loading=30)
        self.attach(guard)
        self.now = 30
        self.assertEqual(guard._observe(sample()), 'model_loading_timeout')

    def test_leave_restores_standard_floor(self):
        guard = self.guard()
        self.attach(guard)
        self.now = 100
        low = sample(available_bytes=TRIP_AVAILABLE)
        self.assertIsNone(guard._observe(low))
        guard.leave_swap_tolerant()
        guard.leave_swap_tolerant()   # idempotent
        self.assertEqual(guard.mode(), 'standard')
        self.assertIsNone(guard.metadata()['swap_tolerant'])
        self.assertEqual(guard._observe(low), 'host_available_below_floor')

    def test_thread_signals_on_swap_tolerant_stop(self):
        class Live:
            pid = 77
            def __init__(self): self.exit, self.signals = None, []
            def poll(self): return self.exit
            def terminate(self): self.signals.append('term')
            def kill(self): self.signals.append('kill'); self.exit = -9
        values = iter([sample(), sample(host_oom_kill=8)])
        guard = policy.MemoryGuard(reader=lambda: next(values), swap_tolerant=policy.swap_tolerant_policy(True))
        process = Live()
        with patch.object(policy, 'TERM_GRACE_SECONDS', .01):
            guard.start(process)
            guard._thread.join(timeout=1)
        self.assertEqual(guard.failure(), 'host_oom_kill')
        self.assertEqual(process.signals, ['term', 'kill'])
        guard.stop()


class Selection(unittest.TestCase):
    def config(self, **flash):
        cfg = make_config(cache_gib=14, peak=22 * GIB)
        cfg['oxe_flash'].update(flash)
        return cfg

    def test_only_pref_20_or_18_on_the_memtotal_rule_and_opted_in(self):
        on = {'swap_tolerant_guard': True, 'desktop_floor_gib': 2.0}
        pick = functools.partial(oxe_runtime.flash_swap_tolerant_guard, swap_free=16 * GIB)  # S2: swap present
        self.assertEqual(pick(self.config(cache_pref='20', **on), 25 * GIB, MEM_TOTAL, env={}),
                         policy.swap_tolerant_policy(True))
        self.assertIsNotNone(pick(self.config(cache_pref='18', **on), 25 * GIB, MEM_TOTAL, env={}))
        for pref in ('auto', '16', '14'):
            with self.subTest(pref=pref):
                self.assertIsNone(pick(self.config(cache_pref=pref, **on), 25 * GIB, MEM_TOTAL, env={}))
        # Opt-in missing or off: standard guard (the public default).
        self.assertIsNone(pick(self.config(cache_pref='20', desktop_floor_gib=2.0), 25 * GIB, MEM_TOTAL, env={}))
        self.assertIsNone(pick(self.config(cache_pref='20', desktop_floor_gib=2.0, swap_tolerant_guard=False),
                               25 * GIB, MEM_TOTAL, env={}))
        # MemTotal unknown, or the MemTotal rule does not fit (falls back to MemAvailable): standard guard.
        self.assertIsNone(pick(self.config(cache_pref='20', **on), 25 * GIB, None, env={}))
        self.assertIsNone(pick(self.config(cache_pref='20', swap_tolerant_guard=True, desktop_floor_gib=16),
                               40 * GIB, MEM_TOTAL, env={}))
        # Environment preference wins, as for the cache choice.
        self.assertIsNone(pick(self.config(cache_pref='20', **on), 25 * GIB, MEM_TOTAL, env={fcp.ENV: 'auto'}))
        # No own scope (flash_memory) -> never.
        cfg = self.config(cache_pref='20', **on)
        cfg.pop('flash_memory', None)
        self.assertIsNone(pick(cfg, 25 * GIB, MEM_TOTAL, env={}))

    def test_overrides_reach_the_guard(self):
        cfg = self.config(cache_pref='20', desktop_floor_gib=2.0,
                          swap_tolerant_guard={'enabled': True, 'psi_full_seconds': 25})
        self.assertEqual(oxe_runtime.flash_swap_tolerant_guard(cfg, 25 * GIB, MEM_TOTAL, env={}, swap_free=16 * GIB)['psi_full_seconds'], 25.0)

    def test_profile_validation_and_resolution(self):
        profile = flash_profile()
        profile['flash_memory'] = {'scope_max_bytes': 29 * GIB, 'host_floor_bytes': GIB}
        profile['memlat_overrides']['oxe_flash'].update(cache_pref='20', desktop_floor_gib=2.0,
                                                        swap_tolerant_guard={'enabled': True, 'grace_seconds': 90})
        cfg = profile_policy.resolve(profile, available=True, oxe_flash_enabled=True)
        self.assertEqual(cfg['oxe_flash']['swap_tolerant_guard'], {'enabled': True, 'grace_seconds': 90})
        oxe_runtime._validate(cfg)
        profile['memlat_overrides']['oxe_flash']['swap_tolerant_guard'] = {'enabled': True, 'grace_seconds': -1}
        with self.assertRaises(ValueError):
            profile_policy.resolve(profile, available=True, oxe_flash_enabled=True)



class ControllerWiring(unittest.TestCase):
    """Manager.begin() with a fake engine: which guard is built, and leaving the mode after /props."""
    def setUp(self):
        import shutil, tempfile
        from pathlib import Path
        from snapdragon_ai_lab.runtime import controller
        from test_flash_oxe_production import flash_profile as production_profile
        self.controller = controller
        self.tmp = Path(tempfile.mkdtemp(prefix='f127-'))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.manager = controller.Manager()
        self.manager.stop_event.set()
        self.addCleanup(self.manager.close)
        profile = production_profile()
        profile['flash_memory'] = {'scope_max_bytes': 29 * GIB, 'host_floor_bytes': GIB}
        self.profile = profile

    def begin(self, flash, props):
        from unittest.mock import Mock
        from test_flash_idle_qos import FakeEngine
        import json
        import subprocess
        profile = copy.deepcopy(self.profile)
        profile['memlat_overrides']['oxe_flash'].update(flash)
        config = profile_policy.resolve(profile, available=True, oxe_flash_enabled=True)
        guard = Mock(failure=Mock(return_value=None), mode=Mock(return_value='swap-tolerant'))
        body = json.dumps({'engine_status': {'flash_cache': props}}).encode()
        with patch.object(self.controller, 'total_memory_bytes', return_value=MEM_TOTAL), \
                patch.object(self.controller, 'swap_bytes', return_value=(16 * GIB, 16 * GIB)), \
                patch.object(self.controller.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0)), \
                FakeEngine(self.tmp, self.manager, config, props_body=body, guard=guard), \
                patch.object(self.controller.memory_policy, 'MemoryGuard', return_value=guard) as built:
            self.manager.begin(allow_start=True)
            self.manager.cleanup()
        return built.call_args.kwargs, guard

    def test_controller_uses_guard2_even_if_legacy_f127_is_requested(self):
        kwargs, guard = self.begin({'cache_pref': '20', 'desktop_floor_gib': 2.0, 'swap_tolerant_guard': True},
                                   {'reason': 'preferred-swap-tolerant', 'cache_gib': 20, 'pref_gib': 20})
        self.assertNotIn('swap_tolerant', kwargs)
        self.assertEqual(kwargs['host_guard_version'], policy.GUARD_2_VERSION)
        self.assertEqual(kwargs['minimum_available_bytes'], GIB)
        guard.leave_swap_tolerant.assert_not_called()

    def test_native_reporting_another_reason_returns_to_the_floor(self):
        kwargs, guard = self.begin({'cache_pref': '20', 'desktop_floor_gib': 2.0, 'swap_tolerant_guard': True},
                                   {'reason': 'preferred', 'cache_gib': 20, 'pref_gib': 20})
        self.assertNotIn('swap_tolerant', kwargs)
        self.assertEqual(kwargs['host_guard_version'], policy.GUARD_2_VERSION)
        guard.leave_swap_tolerant.assert_called_once_with()

    def test_without_opt_in_the_guard_is_unchanged(self):
        kwargs, _ = self.begin({'cache_pref': '20', 'desktop_floor_gib': 2.0},
                               {'reason': 'preferred-swap-tolerant', 'cache_gib': 20, 'pref_gib': 20})
        self.assertNotIn('swap_tolerant', kwargs)
        self.assertEqual(kwargs['host_guard_version'], policy.GUARD_2_VERSION)
        self.assertEqual(kwargs['minimum_available_bytes'], GIB)


if __name__ == '__main__':
    unittest.main()
