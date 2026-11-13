"""Model-free tests for tools/organic_report.py on synthetic ledger rows (no real ledger is read)."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location('organic_report', ROOT / 'tools/organic_report.py')
org = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(org)


def synthetic_rows(sessions=25, per_session=4):
    rows = []
    for s in range(sessions):
        sid = f'{s:016x}'
        for k in range(per_session):
            cached = 0 if k == 0 else 5000
            rows.append({
                'origin': 'organic', 'session': sid, 't': 1000.0 + s * 100 + k,
                'cached_tokens': cached, 'new_tokens': 1000, 'ctx_tokens': cached + 1000,
                'output_tokens': 200, 'ttfb_ms': 900.0 if k == 0 else 300.0,
                'prefill_ms': 1000.0, 'decode_ms': 3000.0, 'wall_ms': 4000.0, 'queue_wait_ms': 0.0,
                'miss_class': 'history-edit' if k == 1 else None,
                'resume_src': 'ram' if k % 2 else None,
                'draft_n': 10, 'draft_n_accepted': 6,
            })
    # rows that must never be counted: lab replay, and rows from before REAL-1 (no origin)
    rows += [{'origin': 'lab', 'session': 'ff' * 8, 't': 5.0, 'cached_tokens': 0, 'new_tokens': 99999,
              'wall_ms': 1.0}] * 5
    rows += [{'session': 'ee' * 8, 't': 6.0, 'cached_tokens': 0, 'new_tokens': 99999, 'wall_ms': 1.0}] * 3
    return rows


def write_ledger(rows, directory):
    path = Path(directory) / 'requests-20261112.jsonl'
    with open(path, 'w', encoding='utf-8') as stream:
        for r in rows:
            stream.write(json.dumps(r) + '\n')
    return str(path)


class OrganicReportTests(unittest.TestCase):
    def test_known_answers(self):
        rep = org.build(synthetic_rows())
        self.assertEqual(rep['status'], 'ok')
        self.assertEqual(rep['sessions'], 25)
        self.assertEqual(rep['organic_rows'], 100)  # lab and unlabelled rows excluded
        self.assertEqual(rep['turns_per_session']['p50'], 4)
        self.assertEqual(rep['new_tokens_per_turn']['p50'], 1000)
        self.assertEqual(rep['context_tokens']['max'], 6000)
        self.assertEqual(rep['turn1_ttfb_ms']['cold']['n'], 25)
        self.assertEqual(rep['turn1_ttfb_ms']['cold']['p50'], 900.0)
        self.assertEqual(rep['turn1_ttfb_ms']['warm']['n'], 0)
        self.assertEqual(rep['prefill_share_of_engine_ms'], 0.25)
        self.assertEqual(rep['c3'], {'judged': 25 * 1, 'violations_by_class': {'history-edit': 25}})
        self.assertEqual(rep['resume_src_mix'], {'absent': 50, 'ram': 50})
        self.assertEqual(rep['wall_shares'], {'queue_wait': 0.0, 'prefill': 0.25, 'decode': 0.75})
        # every organic context (1000 and 6000 tokens) sits in the <8K band
        self.assertEqual(rep['mtp_acceptance_by_band'], {'<8K': {'drafts': 1000, 'acceptance': 0.6}})
        self.assertEqual(rep['wall_share_by_context_band'], {'<8K': 1.0})

    def test_refuses_below_twenty_sessions(self):
        rep = org.build(synthetic_rows(sessions=19))
        self.assertEqual(rep['status'], 'insufficient')
        self.assertNotIn('context_tokens', rep)
        with tempfile.TemporaryDirectory() as d:
            path = write_ledger(synthetic_rows(sessions=19), d)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = org.main(['--json', path])
            text = out.getvalue()
        self.assertEqual(code, 2)
        self.assertIn('not enough organic data', text)
        self.assertNotIn('p50', text)
        self.assertNotIn('ff' * 8, text)

    def test_json_output_has_no_session_hashes(self):
        with tempfile.TemporaryDirectory() as d:
            path = write_ledger(synthetic_rows(), d)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = org.main(['--json', path])
        self.assertEqual(code, 0)
        self.assertNotIn(f'{0:016x}', out.getvalue())
        self.assertEqual(json.loads(out.getvalue())['schema'], org.SCHEMA)

    def test_band_edges(self):
        self.assertEqual(org.band_of(8 * 1024), '<8K')
        self.assertEqual(org.band_of(8 * 1024 + 1), '8K-32K')
        self.assertEqual(org.band_of(300 * 1024), '>256K')


if __name__ == '__main__':
    unittest.main()
