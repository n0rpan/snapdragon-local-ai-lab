"""Model-free tests for the 256K shape, the edit-back case and the replay key file in tools/long_session.py."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location('long_session', ROOT / 'tools/long_session.py')
ls = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ls)


class ShapeAndEditTests(unittest.TestCase):
    def test_256k_shape_is_200_turns_to_256k(self):
        self.assertEqual(ls.SHAPES['256k-cross'], {'turns': 200, 'target': 262144})
        args = ls.build_parser().parse_args(['generate', '--out', 'x.json', '--shape', '256k-cross'])
        self.assertEqual(args.shape, '256k-cross')

    def test_edit_back_edits_one_earlier_message_in_the_last_request_only(self):
        base = ls.generate_session(3, 30, 16 * 1024, True)
        edited = ls.edit_back(base, 4 * 1024)
        edit = edited['params']['edit_back']
        self.assertLess(edit['turn_index'], len(base['turns']) - 1)
        self.assertEqual(edit['tokens_back'], 4096)
        plain = list(ls.request_plan(base))
        changed = list(ls.request_plan(edited))
        self.assertEqual([m for _, m, _ in plain[:-1]], [m for _, m, _ in changed[:-1]])
        last_plain, last_edit = plain[-1][1], changed[-1][1]
        pos = len(base['prefix']['messages']) + 3 * edit['turn_index']
        self.assertNotEqual(last_plain[pos]['content'], last_edit[pos]['content'])
        self.assertEqual(len(last_plain[pos]['content']), len(last_edit[pos]['content']))
        self.assertTrue(last_edit[pos]['content'].startswith(ls.EDIT_PREFIX))
        self.assertEqual(plain[-1][2], changed[-1][2])  # same estimated tokens

    def test_edit_back_refuses_when_history_is_too_short(self):
        base = ls.generate_session(3, 5, 16 * 1024, True)
        with self.assertRaises(ls.UsageError):
            ls.edit_back(base, 10 ** 7)

    def test_generate_cli_writes_the_edit(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 's.json')
            with contextlib.redirect_stdout(io.StringIO()):
                code = ls.main(['generate', '--out', path, '--seed', '3', '--turns', '30', '--target-context', '16K',
                                '--edit-back', '4K'])
            self.assertEqual(code, 0)
            with open(path, encoding='utf-8') as stream:
                data = json.load(stream)
        self.assertIn('edit_back', data['params'])


class KeyFileTests(unittest.TestCase):
    def test_owner_only_key_file_is_read(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'key')
            with open(path, 'w', encoding='ascii') as stream:
                stream.write('k' * 32 + '\n')
            os.chmod(path, 0o600)
            self.assertEqual(ls.load_key_file(path), 'k' * 32)
            os.chmod(path, 0o644)
            with self.assertRaises(ls.UsageError):
                ls.load_key_file(path)


if __name__ == '__main__':
    unittest.main()
