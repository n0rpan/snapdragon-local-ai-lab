import copy
import importlib.util
from pathlib import Path
import threading
import unittest
from unittest.mock import patch

PATH = Path(__file__).resolve().parents[1] / 'src/snapdragon_ai_lab/runtime/memory_policy.py'
spec = importlib.util.spec_from_file_location('tested_memory_policy', PATH)
policy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy)


def sample(**changes):
    value = {'available_bytes': 4 * 1024**3, 'psi_full_avg10': 0., 'psi_full_total_us': 0,
             'swap_current_bytes': 0, 'swap_max': '0', 'memory_high': 'max',
             'memory_max': str(27 * 1024**3),
             'events': {'high': 0, 'max': 0, 'oom': 0, 'oom_kill': 0, 'oom_group_kill': 0}}
    value.update(changes)
    return value


class Process:
    pid = 12345
    def __init__(self):
        self.exit = None
        self.signals = []
        self.terminated = threading.Event()
    def poll(self): return self.exit
    def terminate(self):
        self.signals.append('term')
        self.terminated.set()
    def kill(self):
        self.signals.append('kill')
        self.exit = -9


class MemoryPolicyTests(unittest.TestCase):
    def guard(self, initial=None):
        self.now = 0.
        return policy.MemoryGuard(reader=lambda: copy.deepcopy(initial or sample()), clock=lambda: self.now)

    def test_exact_available_floor_and_latched_reason(self):
        guard = self.guard()
        self.assertIsNone(guard._observe(sample(available_bytes=policy.MIN_AVAILABLE_BYTES)))
        self.assertEqual(guard._observe(sample(available_bytes=policy.MIN_AVAILABLE_BYTES - 1)), 'host_available_below_floor')
        self.assertEqual(guard._observe(sample()), 'host_available_below_floor')

    def test_sustained_inclusive_PSI_threshold_requires_ten_seconds(self):
        guard = self.guard()
        self.assertIsNone(guard._observe(sample(psi_full_avg10=10.)))
        for at in (.25, 3, 9.999):
            self.now = at
            self.assertIsNone(guard._observe(sample(psi_full_avg10=10., psi_full_total_us=int(at * 100000))))
        self.now = 10
        self.assertEqual(guard._observe(sample(psi_full_avg10=10., psi_full_total_us=1000000)), 'host_memory_pressure_sustained')

    def test_short_and_intermittent_pressure_does_not_accumulate(self):
        guard = self.guard()
        for start in (0, 10, 20):
            self.now = start
            self.assertIsNone(guard._observe(sample(psi_full_avg10=12.)))
            self.now = start + 9
            self.assertIsNone(guard._observe(sample(psi_full_avg10=12.)))
            self.now = start + 9.5
            self.assertIsNone(guard._observe(sample(psi_full_avg10=9.999)))
        self.assertEqual(guard.metadata()['psi_high_seconds'], 0)

    def test_high_max_events_and_global_swap_are_not_failure_rules(self):
        guard = self.guard()
        value = sample()
        value['events'].update(high=300, max=200)
        # Extraneous host swap/pageout metadata has no stop semantics.
        value.update(host_swapouts=99999999, host_swap_used=2 * 1024**3)
        self.assertIsNone(guard._observe(value))
        self.assertEqual(guard.metadata()['memory_events']['max'], 200)
        self.assertTrue(guard.metadata()['policy']['host_swap_activity_allowed'])

    def test_each_new_OOM_counter_is_failure_but_historical_counts_are_not(self):
        for key in ('oom', 'oom_kill', 'oom_group_kill'):
            with self.subTest(key=key):
                initial = sample()
                initial['events'][key] = 3
                guard = self.guard(initial)
                self.assertIsNone(guard._observe(initial))
                value = copy.deepcopy(initial)
                value['events'][key] = 4
                self.assertEqual(guard._observe(value), 'cgroup_oom')

    def test_own_swap_and_missing_zero_swap_cap_fail(self):
        guard = self.guard()
        self.assertEqual(guard._observe(sample(swap_current_bytes=1)), 'own_swap_nonzero')
        with self.assertRaisesRegex(policy.MemoryPolicyError, 'own_swap_limit_not_zero'):
            self.guard(sample(swap_max='max'))

    def test_bad_metadata_prevents_attach_without_any_process(self):
        def bad(): raise OSError('no metadata')
        with self.assertRaisesRegex(policy.MemoryPolicyError, 'memory_metadata_unavailable'):
            policy.MemoryGuard(reader=bad)

    def test_thread_signals_only_attached_process_and_escalates(self):
        values = iter([sample(), sample(available_bytes=1)])
        guard = policy.MemoryGuard(reader=lambda: next(values))
        process = Process()
        unrelated = Process()
        with patch.object(policy, 'TERM_GRACE_SECONDS', .01):
            guard.start(process)
            guard._thread.join(timeout=1)
        self.assertFalse(guard._thread.is_alive())
        self.assertEqual(process.signals, ['term', 'kill'])
        self.assertEqual(unrelated.signals, [])
        self.assertEqual(guard.failure(), 'host_available_below_floor')
        self.assertFalse(hasattr(guard, 'lease'))
        guard.stop()

    def test_stop_prevents_escalation_and_future_signals(self):
        values = iter([sample(), sample(available_bytes=1)])
        guard = policy.MemoryGuard(reader=lambda: next(values))
        process = Process()
        guard.start(process)
        self.assertTrue(process.terminated.wait(timeout=1))
        guard.stop()
        guard._signal(kill=True)
        self.assertEqual(process.signals, ['term'])
        self.assertFalse(guard._thread.is_alive())
        with self.assertRaisesRegex(ValueError, 'only_once'):
            guard.start(Process())
        guard.stop()

    def test_already_exited_child_is_never_signaled(self):
        guard = self.guard()
        process = Process()
        process.exit = 0
        guard.start(process)
        guard._thread.join(timeout=1)
        self.assertEqual(process.signals, [])
        self.assertIsNone(guard.failure())
        guard.stop()

    def test_oom_is_recorded_when_child_exits_between_samples(self):
        final = sample()
        final['events']['oom_kill'] = 1
        values = iter([sample(), final])
        guard = policy.MemoryGuard(reader=lambda: next(values))
        process = Process()
        process.exit = -9
        guard.start(process)
        guard._thread.join(timeout=1)
        self.assertEqual(guard.failure(), 'cgroup_oom')
        self.assertEqual(process.signals, [])
        guard.stop()

    def test_metadata_failure_during_monitor_stops_owned_process(self):
        values = iter([sample()])
        def reader():
            try: return next(values)
            except StopIteration: raise ValueError('metadata lost')
        guard = policy.MemoryGuard(reader=reader)
        process = Process()
        guard.start(process)
        self.assertTrue(process.terminated.wait(timeout=1))
        guard.stop()
        self.assertEqual(guard.failure(), 'memory_metadata_unavailable')

    def test_numeric_metadata_only_status(self):
        value = self.guard().metadata()
        self.assertEqual(value['policy']['minimum_available_bytes'], 1536 * 1024**2)
        self.assertEqual(value['policy']['psi_continuous_seconds'], 10)
        self.assertFalse(any(word in repr(value).lower() for word in ['prompt', 'completion', 'output', 'model_path']))

    def test_loading_grace_then_fresh_runtime_PSI_window(self):
        self.now = 0.
        guard = policy.MemoryGuard(reader=lambda: sample(psi_full_avg10=50.),
                                   clock=lambda: self.now, loading_timeout_seconds=60)
        for at in (10, 30, 59):
            self.now = at
            self.assertIsNone(guard._observe(sample(psi_full_avg10=50.)))
        self.assertEqual(guard.metadata()['phase'], 'loading')
        guard.mark_ready()
        self.assertEqual(guard.metadata()['phase'], 'ready')
        self.assertIsNone(guard._observe(sample(psi_full_avg10=50., psi_full_total_us=10000000)))
        self.now = 68
        guard.mark_ready()  # Repeated readiness must not reset the PSI timer.
        self.assertIsNone(guard._observe(sample(psi_full_avg10=50., psi_full_total_us=10900000)))
        self.now = 69
        self.assertEqual(guard._observe(sample(psi_full_avg10=50., psi_full_total_us=11000000)), 'host_memory_pressure_sustained')

    def test_loading_deadline_is_bounded_and_cannot_be_evaded_by_late_readiness(self):
        for late_ready in (False, True):
            self.now = 0.
            guard = policy.MemoryGuard(reader=sample, clock=lambda: self.now,
                                       loading_timeout_seconds=30)
            self.now = 30
            if late_ready:
                guard.mark_ready()
            self.assertEqual(guard._observe(sample()), 'model_loading_timeout')
        for timeout in (0, -1, 301, float('nan'), float('inf'), True, '30'):
            with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                policy.MemoryGuard(reader=sample, loading_timeout_seconds=timeout)

    def test_loading_deadline_starts_at_attach_and_metadata_failure_still_stops(self):
        self.now = 0.
        guard = policy.MemoryGuard(reader=sample, clock=lambda: self.now,
                                   loading_timeout_seconds=30)
        self.now = 100
        process = Process()
        process.exit = 0
        guard.start(process)
        guard._thread.join(timeout=1)
        self.assertEqual(guard.metadata()['loading_deadline_monotonic'], 130)
        self.assertIsNone(guard.failure())
        guard.stop()
        calls = iter([sample()])
        def reader():
            try:
                return next(calls)
            except StopIteration:
                raise ValueError('metadata lost while loading')
        guard = policy.MemoryGuard(reader=reader, loading_timeout_seconds=30)
        process = Process()
        guard.start(process)
        self.assertTrue(process.terminated.wait(timeout=1))
        guard.stop()
        self.assertEqual(guard.failure(), 'memory_metadata_unavailable')

    def test_loading_grace_preserves_all_hard_stops(self):
        cases = [(sample(available_bytes=policy.MIN_AVAILABLE_BYTES - 1), 'host_available_below_floor'),
                 (sample(swap_current_bytes=1), 'own_swap_nonzero'),
                 (sample(swap_max='max'), 'own_swap_limit_not_zero'),
                 (sample(pids_max_events=1), 'cgroup_pids_limit')]
        oom = sample()
        oom['events']['oom_kill'] = 1
        cases.append((oom, 'cgroup_oom'))
        for value, expected in cases:
            with self.subTest(expected=expected):
                guard = policy.MemoryGuard(reader=sample, loading_timeout_seconds=60)
                self.assertEqual(guard._observe(value), expected)
                guard.mark_ready()
                self.assertEqual(guard.failure(), expected)

    def test_high_ema_without_new_stalls_never_stops_after_loading(self):
        self.now = 0.
        guard = policy.MemoryGuard(reader=lambda: sample(psi_full_avg10=50.),
                                   clock=lambda: self.now, loading_timeout_seconds=60)
        self.now = 20
        self.assertIsNone(guard._observe(sample(psi_full_avg10=50., psi_full_total_us=15000000)))
        guard.mark_ready()
        for at in (20, 25, 30, 40):
            self.now = at
            self.assertIsNone(guard._observe(sample(psi_full_avg10=50., psi_full_total_us=15000000)))
        self.assertEqual(guard.metadata()['psi_window_full_percent'], 0.)
        self.assertFalse(guard.metadata()['psi_stop_corroborated'])

    def test_recent_counter_window_keeps_boundary_sample_and_stays_bounded(self):
        guard = self.guard()
        for at in (.5, 1.5, 2.8, 4.1, 5.4, 7., 9.2, 10.6, 11.7):
            self.now = at
            self.assertIsNone(guard._observe(sample(psi_full_avg10=50.)))
        self.assertEqual(guard._psi_window[0][0], 1.5)
        self.assertAlmostEqual(guard.metadata()['psi_window_seconds'], 10.2)
        for step in range(48, 401):
            self.now = step * .25
            self.assertIsNone(guard._observe(sample(psi_full_avg10=50.)))
        self.assertLessEqual(len(guard._psi_window), 42)
        self.assertGreaterEqual(guard.metadata()['psi_window_seconds'], 10.)

    def test_below_ten_percent_actual_stalls_do_not_confirm_high_ema(self):
        guard = self.guard()
        self.assertIsNone(guard._observe(sample(psi_full_avg10=50.)))
        for at in (1., 5., 10., 11.):
            self.now = at
            self.assertIsNone(guard._observe(sample(psi_full_avg10=50., psi_full_total_us=int(at * 99999))))
        self.assertLess(guard.metadata()['psi_window_full_percent'], 10.)

    def test_counter_reset_missing_and_malformed_stop_only_the_owned_child(self):
        for bad in (None, -1, True, '7', 6, 2**64, 'missing'):
            value = sample(psi_full_total_us=bad)
            if bad == 'missing':value.pop('psi_full_total_us')
            values = iter([sample(psi_full_total_us=7), sample(psi_full_total_us=7), value])
            guard = policy.MemoryGuard(reader=lambda: next(values))
            process, unrelated = Process(), Process()
            with patch.object(policy, 'POLL_SECONDS', .001), patch.object(policy, 'TERM_GRACE_SECONDS', .001):
                guard.start(process)
                guard._thread.join(timeout=1)
            self.assertFalse(guard._thread.is_alive())
            self.assertEqual(guard.failure(), 'memory_metadata_unavailable')
            self.assertEqual(process.signals, ['term', 'kill'])
            self.assertEqual(unrelated.signals, [])
            guard.stop()

    def test_pressure_file_is_read_once_and_counter_parser_is_strict(self):
        files = {'/metadata/pids.events':'max 0', '/proc/meminfo':'MemAvailable: 4000000 kB',
                 '/proc/pressure/memory':'some avg10=0 total=30\nfull avg10=12.5 total=27\n',
                 '/metadata/memory.swap.current':'0', '/metadata/memory.swap.max':'0',
                 '/metadata/memory.high':'max', '/metadata/memory.max':'max',
                 '/metadata/memory.events':'oom 0\noom_kill 0\n'}
        seen = []
        def read(path):
            seen.append(str(path))
            return files[str(path)]
        with patch.object(Path, 'read_text', read):
            value = policy.read_sample('/metadata')
        self.assertEqual(value['psi_full_total_us'], 27)
        self.assertEqual(value['psi_full_avg10'], 12.5)
        self.assertEqual(seen.count('/proc/pressure/memory'), 1)
        for text in ('full avg10=10', 'full avg10=10 total=-1', 'full avg10=10 total=nan'):
            with self.subTest(text=text), self.assertRaises((ValueError, KeyError)):
                policy.parse_psi_full(text)

    def test_pure_parsers_and_reusable_policy(self):
        self.assertEqual(policy.parse_available_bytes('MemTotal: 999 kB\nMemAvailable: 1572864 kB\n'), policy.MIN_AVAILABLE_BYTES)
        self.assertEqual(policy.parse_psi_full_avg10('some avg10=99.0 avg60=0 total=1\nfull avg10=10.00 avg60=0.00 avg300=0.00 total=2\n'), 10.)
        for invalid in ['full avg10=nan', 'full avg10=inf', 'full avg10=-1', 'some avg10=0']:
            with self.subTest(invalid=invalid):
                with self.assertRaises((ValueError, KeyError)): policy.parse_psi_full_avg10(invalid)
        initial = sample()
        baseline = policy.event_baseline(initial)
        high = sample(psi_full_avg10=10.)
        self.assertEqual(policy.evaluate_sample(high, 1., None, baseline), (None, 1.))
        self.assertEqual(policy.evaluate_sample(high, 11., 1., baseline), ('host_memory_pressure_sustained', 1.))
        self.assertEqual(policy.evaluate_sample(sample(pids_max_events=1), 1., None, baseline)[0], 'cgroup_pids_limit')
        self.assertEqual(policy.parse_events('high 9\nmax 99\noom 0\noom_kill 0\n')['max'], 99)


if __name__ == '__main__': unittest.main()
