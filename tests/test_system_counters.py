import copy
from pathlib import Path
import tempfile
import unittest

from snapdragon_ai_lab import system_counters as counters


def residency(low, high):
    return ('     From  :   To\n'
            '           : 2000000004224000000   time(ms)\n'
            f'  200000000: 0 1 {low}\n'
            f'*4224000000: 1 0 {high}\n'
            'Total transition : 2\n')


class CounterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def put(self, path, text):
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True); target.write_text(text)

    def base(self):
        self.put('proc/vmstat', 'pswpin 100\npswpout 50\nzswpin 20\npgmajfault 30\nallocstall_normal 2\nworkingset_refault_file 8\nother 999\n')
        self.put('proc/meminfo', 'MemAvailable: 2048 kB\nSwapTotal: 8192 kB\nSwapFree: 4096 kB\n')
        for kind in ('cpu', 'memory', 'io'):
            self.put(f'proc/pressure/{kind}', 'some avg10=0.00 avg60=0.00 avg300=0.00 total=1000\nfull avg10=0.00 avg60=0.00 avg300=0.00 total=500\n')
        for name in counters.DEVFREQ:
            self.put(f'sys/class/devfreq/{name}/trans_stat', residency(100, 900))
        self.put('sys/module/zswap/parameters/enabled', 'Y\n')
        self.put('sys/module/zswap/parameters/max_pool_percent', '5\n')
        self.put('sys/devices/system/cpu/cpufreq/policy0/cpuinfo_avg_freq', '3400000\n')
        self.put('sys/class/thermal/thermal_zone0/type', 'cpu0\n')
        self.put('sys/class/thermal/thermal_zone0/temp', '75000\n')

    def test_packed_frequency_header_does_not_corrupt_residency(self):
        row = counters.parse_trans_stat(residency(10, 90))
        self.assertEqual(row['time_ms_by_freq_hz'], {'200000000': 10, '4224000000': 90})
        self.assertEqual(row['current_freq_hz'], 4224000000)
        self.assertEqual(row['max_freq_hz'], 4224000000)
        with self.assertRaisesRegex(ValueError, 'unit'):
            counters.parse_trans_stat('From: To\n200000000: 0 1 10\n')

    def test_snapshot_is_read_only_and_missing_sources_are_explicit(self):
        self.base()
        before = {p.relative_to(self.root): p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        result = counters.snapshot(self.root)
        after = {p.relative_to(self.root): p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        self.assertEqual(before, after)
        self.assertEqual(result['memory_bytes']['value']['MemAvailable'], 2048 * 1024)
        self.assertNotIn('other', result['vmstat']['value'])
        self.assertEqual(result['cpu_policies']['policy0']['cpuinfo_avg_freq']['value'], 3400000)
        self.assertFalse(result['cpu_policies']['policy0']['scaling_cur_freq']['available'])
        self.assertEqual(result['thermal']['thermal_zone0']['temp_millic']['value'], 75000)
        self.assertEqual(result['zswap']['shrinker_enabled']['reason'], 'ENOENT')

    def test_delta_uses_actual_stalls_and_wall_time_not_avg10(self):
        self.base(); before = counters.snapshot(self.root); before['monotonic_s'] = 100
        after = copy.deepcopy(before); after['monotonic_s'] = 101
        after['vmstat']['value']['pswpin'] += 1200
        after['psi']['memory']['value']['some']['total'] += 60000
        after['psi']['memory']['value']['some']['avg10'] = 99
        after['devfreq']['ddr']['value'] = counters.parse_trans_stat(residency(130, 1870))
        diff = counters.difference(before, after)
        self.assertEqual(diff['swapin_pages_per_s'], 1200)
        self.assertEqual(diff['psi']['memory']['value']['some']['stall_percent'], 6)
        ddr = diff['devfreq']['ddr']
        self.assertAlmostEqual(ddr['below_max_fraction'], .03)
        self.assertEqual(ddr['coverage_fraction'], 1)
        self.assertTrue(ddr['below_max_exceeds_2percent'])

    def test_reset_or_sparse_observation_cannot_authorize_o14(self):
        self.base(); before = counters.snapshot(self.root); before['monotonic_s'] = 100
        after = copy.deepcopy(before); after['monotonic_s'] = 101
        after['devfreq']['ddr']['value'] = counters.parse_trans_stat(residency(1, 9))
        after['vmstat']['value']['pswpin'] = 0
        diff = counters.difference(before, after)
        self.assertFalse(diff['devfreq']['ddr']['available'])
        self.assertIn('pswpin', diff['vmstat']['reset'])
        self.assertIsNone(diff['swapin_pages_per_s'])
        after['devfreq']['ddr']['value'] = counters.parse_trans_stat(residency(200, 900))
        ddr = counters.difference(before, after)['devfreq']['ddr']
        self.assertEqual(ddr['below_max_fraction'], .1)
        self.assertFalse(ddr['usable_for_o14'])
        self.assertFalse(ddr['below_max_exceeds_2percent'])

    def test_cgroup_deltas_separate_gauges_from_counters(self):
        self.base(); group = '/sys/fs/cgroup/user.slice/run-test.scope'
        self.put(group.lstrip('/') + '/cpu.stat', 'usage_usec 1000\nnr_throttled 2\nthrottled_usec 100\n')
        self.put(group.lstrip('/') + '/memory.events', 'oom 0\noom_kill 0\n')
        self.put(group.lstrip('/') + '/memory.stat', 'anon 100\nfile 200\npgmajfault 5\n')
        before = counters.snapshot(self.root, [group]); before['monotonic_s'] = 100
        after = copy.deepcopy(before); after['monotonic_s'] = 102
        g = after['cgroups'][group]
        g['cpu.stat']['value']['nr_throttled'] = 5
        g['memory.stat']['value'].update(anon=50, pgmajfault=8)
        diff = counters.difference(before, after)['cgroups'][group]
        self.assertEqual(diff['cpu.stat']['delta']['nr_throttled'], 3)
        self.assertEqual(diff['memory_stat_counters']['delta'], {'pgmajfault': 3})
        self.assertNotIn('anon', diff['memory_stat_counters']['delta'])

    def test_pid_resolution_reads_only_explicit_unified_cgroup(self):
        self.put('proc/123/cgroup', '0::/user.slice/test.scope\n')
        self.assertEqual(counters.cgroup_for_pid(123, self.root), '/sys/fs/cgroup/user.slice/test.scope')
        self.put('proc/123/cgroup', '0::/../../etc\n')
        with self.assertRaises(ValueError): counters.cgroup_for_pid(123, self.root)
        for group in ['/etc', '/sys/fs/cgroup/../../etc']:
            with self.assertRaises(ValueError): counters.snapshot(self.root, [group])

    def test_malformed_or_large_source_stays_unavailable(self):
        self.put('proc/vmstat', 'pswpin nope\n')
        result = counters.snapshot(self.root)
        self.assertEqual(result['vmstat']['reason'], 'unrecognized_counter_format')
        self.put('proc/vmstat', 'x' * 65537)
        self.assertEqual(counters.snapshot(self.root)['vmstat']['reason'], 'source_exceeds_64k')


if __name__ == '__main__':
    unittest.main()
