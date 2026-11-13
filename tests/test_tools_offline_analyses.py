"""Model-free tests for tools/offline_analyses.py on synthetic rows and receipts."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location('offline_analyses', ROOT / 'tools/offline_analyses.py')
oa = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(oa)


def rows():
    out = []
    for i in range(10):  # short gap: 20 ms per token
        out.append({'queue_wait_ms': float(i), 'gap_ms': 5000, 'decode_ms': 2000.0, 'output_tokens': 100})
    for _ in range(10):  # long gap: 10 ms per token
        out.append({'queue_wait_ms': 0.0, 'gap_ms': 900000, 'decode_ms': 1000.0, 'output_tokens': 100})
    out.append({'gap_ms': None, 'decode_ms': 1.0, 'output_tokens': 1})  # no gap: not counted in heat
    return out


class LedgerTests(unittest.TestCase):
    def test_heat_ratio_and_queue(self):
        rep = oa.ledger_report(rows())
        self.assertEqual(rep['rows'], 21)
        self.assertEqual(rep['decode ms per token after gap < 60 s [M]']['p50'], 20.0)
        self.assertEqual(rep['decode ms per token after gap >= 600 s [M]']['p50'], 10.0)
        self.assertEqual(rep['heat carry-over ratio (short / long p50) [D]'], 2.0)
        self.assertEqual(rep['queue_wait_ms [M]']['n'], 20)

    def test_cli_reads_a_file_and_prints_counts_only(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'requests-20261112.jsonl'
            path.write_text('\n'.join(json.dumps(r) for r in rows()) + '\n', encoding='utf-8')
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = oa.main(['ledger', str(path)])
        self.assertEqual(code, 0)
        self.assertNotIn(d, out.getvalue())


class ReceiptTests(unittest.TestCase):
    def receipt(self):
        return {'decode': {'ratio18_over14': 0.975},
                'qualification_arm': {'desktop': {'loaded': {'psi_full_above_2_seconds': 60.0, 'samples': 600}}}}

    def test_stall_share_needs_the_interval(self):
        rep = oa.receipt_report(self.receipt(), None)
        self.assertIn('not computed', rep['stall share'])

    def test_stall_share_with_interval(self):
        rep = oa.receipt_report(self.receipt(), 2.0)
        self.assertEqual(rep['loaded wall seconds [D: samples x sample-s]'], 1200.0)
        self.assertEqual(rep['stall share of loaded wall [D]'], 0.05)

    def test_missing_fields_are_reported_not_guessed(self):
        rep = oa.receipt_report({}, 2.0)
        self.assertIn('not measurable', rep['stall share'])


if __name__ == '__main__':
    unittest.main()
