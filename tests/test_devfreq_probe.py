import json
import tempfile
import time
from pathlib import Path
import unittest
from snapdragon_ai_lab.devfreq_probe import policies, temporary_max_vote, Sampler, _write_vote

class DevfreqProbeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for name in ('ddr', 'llcc', 'ddr-qos'):
            (self.root / name).mkdir()
            for field, value in {'min_freq': '100', 'max_freq': '900', 'cur_freq': '400',
                                 'governor': 'remote', 'available_frequencies': '100 400 900'}.items():
                (self.root / name / field).write_text(value)
    def writer(self, path, value):
        path.write_text(str(value))
    def test_pin_restores_both_on_body_exception(self):
        with self.assertRaisesRegex(RuntimeError, 'fixture'):
            with temporary_max_vote(root=self.root, writer=self.writer) as record:
                self.assertEqual(record['applied']['ddr']['readback'], 900)
                raise RuntimeError('fixture')
        self.assertEqual(policies(self.root)['ddr']['min_freq'], 100)
        self.assertEqual(policies(self.root)['llcc']['min_freq'], 100)
        self.assertEqual(record['restore_errors'], [])
    def test_partial_failed_write_is_also_restored(self):
        calls = []
        def partial(path, value):
            calls.append((path.parent.name, value)); self.writer(path, value)
            if path.parent.name == 'llcc' and value == 900:
                raise OSError('partial write')
        with self.assertRaises(OSError):
            with temporary_max_vote(root=self.root, writer=partial):
                self.fail('body must not run')
        self.assertEqual(calls, [('ddr',900), ('llcc',900), ('llcc',100), ('ddr',100)])
        self.assertEqual(policies(self.root)['llcc']['min_freq'], 100)
    def test_validate_all_before_writing_and_reject_arbitrary_sudo_path(self):
        (self.root/'llcc/max_freq').write_text('0')
        calls = []
        with self.assertRaises(ValueError):
            with temporary_max_vote(root=self.root, writer=lambda *v: calls.append(v)):
                self.fail('body must not run')
        self.assertEqual(calls, [])
        with self.assertRaises(ValueError): _write_vote(self.root/'ddr/min_freq', 900)
    def test_failed_restore_is_explicit(self):
        def bad(path,value):
            if value == 100: raise OSError('restore')
            self.writer(path,value)
        with self.assertRaisesRegex(RuntimeError, 'devfreq_restore_failed'):
            with temporary_max_vote(root=self.root, writer=bad): pass
    def test_trace_handles_missing_node_and_distinguishes_qos_units(self):
        (self.root/'llcc/cur_freq').unlink()
        out = self.root/'trace.jsonl'
        with Sampler(out, interval_ms=10, root=self.root) as sampler:
            time.sleep(.05)
        rows = [json.loads(x) for x in out.read_text().splitlines()]
        self.assertGreaterEqual(sampler.count, 2)
        self.assertEqual(rows[0]['policies']['ddr-qos']['units'], 'driver_value')
        self.assertTrue(all(x['cur_value']['llcc'] is None for x in rows if x['event']=='sample'))
        self.assertEqual(out.stat().st_mode & 0o777, 0o600)
        self.assertEqual(rows[-1]['event'], 'end')
        with self.assertRaises(FileExistsError): Sampler(out, root=self.root).start()
