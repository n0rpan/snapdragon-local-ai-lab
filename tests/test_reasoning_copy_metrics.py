import importlib.util
import json
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('copy_metrics', Path(__file__).resolve().parents[1] / 'tools/reasoning_copy_metrics.py')
metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics)


class ReasoningCopyMetricsTests(unittest.TestCase):
    def test_full_and_disjoint_copy_have_expected_coverage(self):
        for source, expected in ((list(range(30)), 1.0), (list(range(100, 130)), 0.0)):
            row = metrics.reasoning_copy_metrics(source, list(range(10, 20)))
            self.assertTrue(all(x['copy_share'] == expected for x in row['historical_ngram_coverage']))

    def test_overlap_counted_once_and_short_sequences(self):
        row = metrics.reasoning_copy_metrics([1, 2, 3, 4], [1, 2, 3, 4, 9])
        self.assertEqual(row['historical_ngram_coverage'][0]['covered_tokens'], 4)
        self.assertEqual(row['historical_ngram_coverage'][1]['covered_tokens'], 4)
        self.assertEqual(row['historical_ngram_coverage'][2]['covered_tokens'], 0)
        self.assertTrue(all(x['copy_share'] == 0 for x in metrics.reasoning_copy_metrics([], [])['historical_ngram_coverage']))

    def test_lookup_cannot_read_future_reasoning(self):
        # One novel 4-gram repeated later becomes a source, never before it exists.
        row = metrics.reasoning_copy_metrics([], [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6])
        self.assertTrue(all(x['accepted_tokens'] == 2 for x in row['causal_prompt_lookup']))
        self.assertTrue(all(x['covered_tokens'] == 0 for x in row['historical_ngram_coverage']))

    def test_exact_match_and_draft_cap(self):
        context = list(range(20)) + [0, 1, 2, 3]
        row = metrics.reasoning_copy_metrics(context, list(range(4, 12)))
        self.assertEqual([x['accepted_tokens'] for x in row['causal_prompt_lookup']], [8, 8])
        self.assertEqual(row['causal_prompt_lookup'][0]['successful_lookups'], 2)
        self.assertEqual(row['causal_prompt_lookup'][1]['successful_lookups'], 1)
        self.assertNotIn('context', row)
        self.assertNotIn('reasoning', row)
        self.assertNotIn('token_ids', json.dumps(row))

    def test_rejects_text_and_invalid_ids_without_mutating_inputs(self):
        for ids in (['private-reasoning'], [-1], [True]):
            with self.assertRaises(ValueError):
                metrics.reasoning_copy_metrics([], ids)
        context, reasoning = [1, 2, 3], [4, 5, 6]
        metrics.reasoning_copy_metrics(context, reasoning)
        self.assertEqual((context, reasoning), ([1, 2, 3], [4, 5, 6]))


if __name__ == '__main__':
    unittest.main()
