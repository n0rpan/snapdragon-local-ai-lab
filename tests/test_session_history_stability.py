import copy
import importlib.util
from pathlib import Path
import sys
import unittest

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
sys.path.insert(0, str(TOOLS))
spec = importlib.util.spec_from_file_location('stability', TOOLS / 'session_history_stability.py')
hs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hs)


class HistoryStabilityTests(unittest.TestCase):
    def case(self):
        first = [{'role': 'system', 'content': 'synthetic'}, {'role': 'user', 'content': 'first'}]
        turns = [{'seq': 1, 'status': 200, 'path': '/v1/chat/completions', 'body': {'messages': first}},
                 {'seq': 2, 'status': 200, 'path': '/v1/chat/completions', 'body': {'messages':
                     first + [{'role': 'assistant', 'content': 'done'}, {'role': 'user', 'content': 'next'}]}}]
        default = [{'seq': 1, 'prompt_ids': [1, 2, 3, 4, 5, 90], 'generation_prompt_start': 5},
                   {'seq': 2, 'prompt_ids': [1, 2, 8, 7, 6, 5, 90], 'generation_prompt_start': 6}]
        fixed = copy.deepcopy(default)
        fixed[1]['prompt_ids'] = [1, 2, 3, 4, 5, 90, 91, 92, 90]
        fixed[1]['generation_prompt_start'] = 8
        return turns, default, fixed

    def test_counts_only_previously_supplied_input_and_passes_exact_stability(self):
        turns, old, fixed = self.case()
        snapshot = copy.deepcopy((turns, old, fixed))
        result = hs.compare(turns, old, fixed)
        self.assertTrue(result['passed'])
        self.assertEqual(result['baseline']['rerendered_input_tokens'], 3)
        self.assertEqual(result['preserved']['rerendered_input_tokens'], 0)
        self.assertEqual(result['reduction'], 1)
        self.assertEqual((turns, old, fixed), snapshot)
        self.assertNotIn('synthetic', str(result))

    def test_content_changes_are_not_mislabelled_as_template_waste(self):
        turns, rows, _ = self.case()
        turns[1]['body']['messages'] = copy.deepcopy(turns[1]['body']['messages'])
        turns[1]['body']['messages'][1]['content'] = 'changed user history'
        result = hs.attribute_history(turns, rows)
        self.assertEqual(result['rerendered_input_tokens'], 0)
        self.assertFalse(result['turns'][-1]['eligible'])

    def test_missing_boundary_and_different_coverage_fail_closed(self):
        turns, old, fixed = self.case()
        fixed[1].pop('generation_prompt_start')
        with self.assertRaises(hs.UsageError): hs.compare(turns, old, fixed)
        with self.assertRaises(hs.UsageError): hs.compare(turns, old, old[:1])

    def test_schema_property_reordering_is_not_template_instability(self):
        turns, old, _ = self.case()
        turns[0]['body']['tools'] = [{'a': 1, 'b': 2}]
        turns[1]['body']['tools'] = [{'b': 2, 'a': 1}]
        result = hs.attribute_history(turns, old)
        self.assertEqual(result['turns'][-1]['reason'], 'tool-schema-or-choice-changed')


if __name__ == '__main__': unittest.main()
