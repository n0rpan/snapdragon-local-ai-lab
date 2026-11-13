"""Model-free tests for tools/tune_from_ledger.py (paper model on synthetic organic reports)."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location('tune_from_ledger', ROOT / 'tools/tune_from_ledger.py')
tune = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tune)
_ospec = importlib.util.spec_from_file_location('organic_report', ROOT / 'tools/organic_report.py')
org = importlib.util.module_from_spec(_ospec)
_ospec.loader.exec_module(org)
sys.path.insert(0, str(ROOT / 'tests'))
from test_tools_organic_report import synthetic_rows  # noqa: E402


class TuneTests(unittest.TestCase):
    def setUp(self):
        self.report = org.build(synthetic_rows())

    def test_decode_cost_grows_with_context_and_matches_ag2_points(self):
        # LONGCTX-ZERO-H sec. 2: Ornith 30.4 ms/token at 32K (after acceptance 0.61)
        self.assertAlmostEqual(tune.decode_ms('ornith', 32 * 1024), 30.4, delta=0.2)
        self.assertLess(tune.decode_ms('ornith', 32 * 1024), tune.decode_ms('ornith', 128 * 1024))

    def test_smaller_cap_compacts_more(self):
        _, few = tune.simulate('ornith', 128, 50)
        _, many = tune.simulate('ornith', 24, 50)
        self.assertGreater(many, few)

    def test_recommendations_are_tagged_and_mtp_undecided_without_fit(self):
        lines = tune.recommend(self.report, {})
        text = '\n'.join(lines)
        self.assertIn('inputs:', text)
        self.assertIn('[M]', text)
        self.assertIn('[E]', text)
        self.assertIn('NOT DECIDED', text)
        self.assertIn('levers:', text)
        self.assertIn('worth >= 3%: yes', text)  # prefill share 0.25 of wall in the synthetic report

    def test_mtp_cutoff_with_a_measured_plain_fit(self):
        # plain decode far slower than MTP: MTP wins everywhere, so no cutoff
        lines = tune.mtp_decision(self.report, {'ornith': (500.0, 0.0)})
        self.assertIn('no cutoff', '\n'.join(lines))
        # plain decode nearly as fast as one verify: break-even acceptance ~0, measured 0.6 wins -> still no cutoff
        lines = tune.mtp_decision(self.report, {'ornith': (1.0, 0.0)})
        self.assertIn('mtp-max-ctx ornith:', '\n'.join(lines))

    def test_refuses_insufficient_report(self):
        bad = org.build(synthetic_rows(sessions=3))
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'report.json'
            path.write_text(json.dumps(bad), encoding='utf-8')
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                code = tune.main(['--report', str(path)])
        self.assertEqual(code, 2)

    def test_main_prints_recommendations_for_ok_report(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'report.json'
            path.write_text(json.dumps(self.report), encoding='utf-8')
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = tune.main(['--report', str(path)])
        self.assertEqual(code, 0)
        self.assertIn('fast compaction', out.getvalue())


if __name__ == '__main__':
    unittest.main()
