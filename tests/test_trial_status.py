"""Read-only trial status accepts bounded complete systemd metadata only."""
from pathlib import Path
import runpy
import unittest

CLI = Path(__file__).resolve().parents[1] / 'src/snapdragon_ai_lab/runtime/local-ai'


class TrialStatus(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cli = runpy.run_path(str(CLI))

    def block(self, unit, state='active', stamp='1000000'):
        return f'Id={unit}\nActiveState={state}\nActiveEnterTimestampMonotonic={stamp}\n'

    def parse(self, value, now=5_000_000):
        return self.cli['parse_flash_trials'](value, now)

    def test_current_cache_and_protected_jobs_show_elapsed_stage(self):
        for unit, kind in {
            'local-ai-flash-profile-cache-abba.service': 'memory',
            'local-ai-flash-profile-hot-parity.service': 'baseline',
            'local-ai-flash-profile-hot-abba.service': 'memory',
            'local-ai-flash-profile-cpu-quota-abba.service': 'coding',
            'local-ai-flash-profile-decode46.service': 'coding',
            'local-ai-flash-profile-decode48.service': 'coding',
            'local-ai-flash-profile-selective-q8.service': 'coding',
            'local-ai-flash-profile-placement.service': 'coding',
            'local-ai-flash-profile-combined-parity.service': 'coding',
            'local-ai-flash-profile-combined-coalescing.service': 'coding',
            'local-ai-flash-profile-combined-predictive.service': 'coding',
            'local-ai-flash-profile-ngram-capture.service': 'memory',
            'local-ai-flash-profile-ngram-screen.service': 'coding',
            'local-ai-flash-profile-cache-full-12288.service': 'coding',
            'local-ai-flash-profile-cache-full-13312.service': 'coding',
        }.items():
            with self.subTest(unit=unit):
                self.assertEqual(self.parse(self.block(unit)), {
                    'flash_trial_active': True, 'flash_trial_kind': kind,
                    'flash_trial_elapsed_seconds': 4,
                })

    def test_inactive_unknown_markup_and_bad_clock_never_show_trial(self):
        unit = 'local-ai-flash-profile-cache-full-12288.service'
        for value in (
            self.block(unit, state='inactive'),
            self.block('<img src="https://example.invalid/">'),
            self.block('unrelated.service'),
            self.block(unit, stamp='0'), self.block(unit, stamp='-1'),
            self.block(unit, stamp='6000000'),
            self.block(unit) + 'Unexpected=payload\n', 'x' * 8193,
        ):
            with self.subTest(value=value[:100]):
                self.assertFalse(self.parse(value)['flash_trial_active'])
        self.assertFalse(self.parse(self.block(unit), now=100_000_000_000)['flash_trial_active'])

    def test_duplicate_metadata_cannot_override_current_state(self):
        unit = 'local-ai-flash-profile-cache-full-12288.service'
        value = self.block(unit)
        self.assertFalse(self.parse(value + '\n' + value)['flash_trial_active'])
        self.assertFalse(self.parse(value + 'ActiveState=active\n')['flash_trial_active'])


if __name__ == '__main__':
    unittest.main()
