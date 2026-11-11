"""O-39 host stop policy: synthetic metadata, no models or host mutations."""
import copy
import unittest
from unittest.mock import patch

from snapdragon_ai_lab.runtime import memory_policy as policy

GIB = 1024**3
MIB = 1024**2


def sample(**changes):
    value = dict(available_bytes=4 * GIB, swap_free_bytes=16 * GIB,
                 psi_full_avg10=0., psi_full_total_us=0,
                 psi_some_avg10=0., psi_some_total_us=0, host_oom_kill=7,
                 swap_current_bytes=0, swap_max='0', memory_high='max',
                 memory_max=str(29 * GIB), pids_max_events=0,
                 events={'high': 0, 'max': 0, 'oom': 0, 'oom_kill': 0, 'oom_group_kill': 0})
    value.update(changes)
    return value


class Guard2Tests(unittest.TestCase):
    def guard(self, initial=None, **kwargs):
        self.now = 0.
        return policy.MemoryGuard(reader=lambda: copy.deepcopy(initial or sample()),
                                  clock=lambda: self.now, host_guard_version='GUARD-2', **kwargs)

    def assert_sustained(self, value, expected):
        guard = self.guard()
        self.assertIsNone(guard._observe(value))
        self.now = 2.999
        self.assertIsNone(guard._observe(value))
        self.now = 3.
        self.assertEqual(guard._observe(value), expected)
        self.assertEqual(guard._observe(sample()), expected)  # Failure stays latched.

    def test_combined_memory_and_swap_requires_three_seconds(self):
        self.assert_sustained(sample(available_bytes=GIB, swap_free_bytes=GIB - 1),
                              'host_memory_and_swap_exhausted')

    def test_full_psi_requires_strictly_above_twenty_for_three_seconds(self):
        self.assert_sustained(sample(psi_full_avg10=20.001), 'host_memory_pressure_full_sustained')

    def test_emergency_available_requires_three_seconds_even_with_free_swap(self):
        self.assert_sustained(sample(available_bytes=256 * MIB - 1), 'host_available_emergency')

    def test_exact_boundaries_are_allowed(self):
        guard = self.guard()
        value = sample(available_bytes=256 * MIB, swap_free_bytes=2 * GIB - 256 * MIB,
                       psi_full_avg10=20.)
        for at in (0., 3., 10., 100.):
            self.now = at
            self.assertIsNone(guard._observe(value))
        self.assertEqual(guard.metadata()['host_high_seconds'], 0)

    def test_responsive_low_available_with_free_swap_is_allowed(self):
        guard = self.guard()
        for at in (0., 3., 30.):
            self.now = at
            self.assertIsNone(guard._observe(sample(available_bytes=900 * MIB, psi_some_avg10=99.)))
        self.assertEqual(guard.metadata()['swap_free_bytes'], 16 * GIB)
        self.assertEqual(guard.metadata()['psi_some_avg10'], 99.)

    def test_a_healthy_sample_resets_the_continuous_or_timer(self):
        guard = self.guard()
        for start in (0., 5., 10.):
            self.now = start
            self.assertIsNone(guard._observe(sample(available_bytes=1)))
            self.now = start + 2.9
            self.assertIsNone(guard._observe(sample(available_bytes=1)))
            self.now = start + 3.
            self.assertIsNone(guard._observe(sample()))
        self.assertEqual(guard.metadata()['host_high_seconds'], 0)

    def test_continuous_unsafe_or_is_not_reset_by_switching_conditions(self):
        guard = self.guard()
        self.assertIsNone(guard._observe(sample(psi_full_avg10=21.)))
        self.now = 2.
        self.assertIsNone(guard._observe(sample(available_bytes=1)))
        self.now = 3.
        self.assertEqual(guard._observe(sample(available_bytes=1)), 'host_available_emergency')

    def test_loading_and_ready_do_not_exempt_or_reset_host_timer(self):
        guard = self.guard(loading_timeout_seconds=60)
        high = sample(available_bytes=128 * MIB)
        self.assertIsNone(guard._observe(high))
        self.now = 2.
        guard.mark_ready()
        self.now = 3.
        self.assertEqual(guard._observe(high), 'host_available_emergency')
        self.assert_sustained(sample(available_bytes=1), 'host_available_emergency')

    def test_loading_timeout_remains_bounded(self):
        guard = self.guard(loading_timeout_seconds=30)
        self.now = 30.
        self.assertEqual(guard._observe(sample()), 'model_loading_timeout')

    def test_loading_pressure_above_twenty_is_not_a_runtime_pressure_stop(self):
        guard = self.guard(loading_timeout_seconds=120)
        for at in (0., 3., 5., 20., 60.):
            self.now = at
            self.assertIsNone(guard._observe(sample(psi_full_avg10=60.)))
        self.assertEqual(guard.metadata()['loading_psi_high_seconds'], 0)

    def test_loading_pressure_strictly_above_sixty_requires_five_seconds(self):
        guard = self.guard(loading_timeout_seconds=60)
        high = sample(psi_full_avg10=60.001)
        self.assertIsNone(guard._observe(high))
        self.now = 4.999
        self.assertIsNone(guard._observe(high))
        self.now = 5.
        self.assertEqual(guard._observe(high), 'host_memory_pressure_full_loading_sustained')

    def test_loading_pressure_dip_resets_its_own_timer(self):
        guard = self.guard(loading_timeout_seconds=60)
        for start in (0., 10., 20.):
            self.now = start
            self.assertIsNone(guard._observe(sample(psi_full_avg10=90.)))
            self.now = start + 4.999
            self.assertIsNone(guard._observe(sample(psi_full_avg10=90.)))
            self.now = start + 5.
            self.assertIsNone(guard._observe(sample(psi_full_avg10=60.)))

    def test_ready_switches_to_twenty_three_with_no_inherited_loading_psi_time(self):
        guard = self.guard(loading_timeout_seconds=120)
        self.assertIsNone(guard._observe(sample(psi_full_avg10=40.)))
        self.now = 30.
        guard.mark_ready()
        self.assertIsNone(guard._observe(sample(psi_full_avg10=40.)))
        self.now = 32.999
        self.assertIsNone(guard._observe(sample(psi_full_avg10=40.)))
        self.now = 33.
        self.assertEqual(guard._observe(sample(psi_full_avg10=40.)), 'host_memory_pressure_full_sustained')

    def test_load_host_floors_stay_three_seconds_despite_catastrophic_psi(self):
        for value, reason in ((sample(available_bytes=1, psi_full_avg10=90.), 'host_available_emergency'),
                              (sample(available_bytes=GIB, swap_free_bytes=0, psi_full_avg10=90.), 'host_memory_and_swap_exhausted')):
            with self.subTest(reason=reason):
                guard = self.guard(loading_timeout_seconds=60)
                self.assertIsNone(guard._observe(value))
                self.now = 3.
                self.assertEqual(guard._observe(value), reason)

    def test_loading_policy_reports_the_actual_phase_and_revision(self):
        guard = self.guard(loading_timeout_seconds=60)
        metadata = guard.metadata()
        self.assertEqual(metadata['guard_revision'], 'GUARD-2b')
        self.assertEqual(metadata['policy']['phase_psi_full_avg10_percent'], 60.)
        self.assertEqual(metadata['policy']['phase_psi_continuous_seconds'], 5.)
        guard.mark_ready()
        self.assertEqual(guard.metadata()['policy']['phase_psi_full_avg10_percent'], 20.)
        self.assertEqual(guard.metadata()['policy']['phase_psi_continuous_seconds'], 3.)

    def test_loading_psi_cannot_start_the_unchanged_host_floor_timer(self):
        guard = self.guard(loading_timeout_seconds=60)
        self.assertIsNone(guard._observe(sample(psi_full_avg10=90.)))
        self.now = 4.
        low = sample(available_bytes=1, psi_full_avg10=20.)
        self.assertIsNone(guard._observe(low))
        self.now = 6.999
        self.assertIsNone(guard._observe(low))
        self.now = 7.
        self.assertEqual(guard._observe(low), 'host_available_emergency')

    def test_ready_discards_the_loading_psi_timer(self):
        guard = self.guard(loading_timeout_seconds=60)
        high = sample(psi_full_avg10=90.)
        self.assertIsNone(guard._observe(high))
        self.now = 4.9
        guard.mark_ready()
        self.assertIsNone(guard._observe(high))
        self.now = 5.
        self.assertIsNone(guard._observe(high))
        self.now = 7.901
        self.assertEqual(guard._observe(high), 'host_memory_pressure_full_sustained')

    def test_attach_preserves_an_existing_unsafe_host_interval(self):
        from test_memory_policy import Process
        guard = self.guard(initial=sample(psi_full_avg10=21.))
        self.now = 2.
        process = Process()
        process.exit = 0  # No signal; inspect the first attached observation.
        guard.start(process)
        guard._thread.join(timeout=1)
        self.assertEqual(guard.metadata()['host_high_seconds'], 2.)
        self.now = 3.
        self.assertEqual(guard._observe(sample(psi_full_avg10=21.)), 'host_memory_pressure_full_sustained')
        guard.stop()

    def test_native_swap_limit_and_usage_fail_immediately(self):
        for value, reason in ((sample(swap_max='max'), 'own_swap_limit_not_zero'),
                              (sample(swap_current_bytes=1), 'own_swap_nonzero')):
            with self.subTest(reason=reason):
                guard = self.guard()
                self.assertEqual(guard._observe(value), reason)

    def test_oom_and_pid_fail_immediately(self):
        values = [(sample(host_oom_kill=8), 'host_oom_kill'),
                  (sample(pids_max_events=1), 'cgroup_pids_limit')]
        for key in ('oom', 'oom_kill', 'oom_group_kill'):
            value = sample()
            value['events'][key] = 1
            values.append((value, 'cgroup_oom'))
        for value, reason in values:
            with self.subTest(reason=reason):
                guard = self.guard()
                self.assertEqual(guard._observe(value), reason)

    def test_high_max_and_host_swap_activity_do_not_stop(self):
        guard = self.guard()
        value = sample()
        value['events'].update(high=50, max=20)
        value.update(pswpin=1000000, pswpout=2000000)
        self.now = 20.
        self.assertIsNone(guard._observe(value))

    def test_metadata_names_actual_policy_and_keeps_numeric_samples(self):
        value = self.guard().metadata()
        self.assertEqual(value['guard_version'], 'GUARD-2')
        self.assertEqual(value['mode'], 'standard')
        self.assertIsNone(value['policy']['minimum_available_bytes'])
        self.assertEqual(value['policy']['host_memory_and_swap_floor_bytes'], 2 * GIB)
        self.assertEqual(value['policy']['host_emergency_available_bytes'], 256 * MIB)
        self.assertEqual(value['policy']['host_continuous_seconds'], 3.)
        self.assertEqual(value['policy']['psi_full_avg10_percent'], 20.)
        self.assertEqual(value['policy']['threshold_comparisons'], 'strict')
        self.assertTrue(value['policy']['stop_only'])

    def test_reads_extended_host_fields_for_guard2_only(self):
        with patch.object(policy, 'current_cgroup', return_value='/fake'), \
                patch.object(policy, 'read_sample', return_value=sample()) as read:
            guard = policy.MemoryGuard(host_guard_version='GUARD-2')
        read.assert_called_once_with('/fake', extended=True)
        self.assertIsNone(guard.failure())

    def test_unknown_or_conflicting_policy_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'unknown_host_guard_version'):
            policy.MemoryGuard(reader=sample, host_guard_version='typo')
        with self.assertRaisesRegex(ValueError, 'mutually_exclusive'):
            policy.MemoryGuard(reader=sample, host_guard_version='GUARD-2',
                               swap_tolerant=policy.swap_tolerant_policy(True))

    def test_bad_required_host_metadata_is_not_assumed_safe(self):
        for key, bad in (('swap_free_bytes', -1), ('available_bytes', True),
                         ('host_oom_kill', None), ('psi_full_avg10', float('nan'))):
            with self.subTest(key=key), self.assertRaises(ValueError):
                policy.evaluate_guard2(sample(**{key: bad}), 0., None, policy.event_baseline(sample()))
        value = sample()
        del value['swap_free_bytes']
        with self.assertRaises(KeyError):
            policy.evaluate_guard2(value, 0., None, policy.event_baseline(sample()))


if __name__ == '__main__':
    unittest.main()
