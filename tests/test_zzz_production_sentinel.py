"""Runs last under `unittest discover`: the real runtime config and Omarchy panel files were not
modified while the suite ran (mtime and size recorded by tests/_isolation.py before any test).

A legitimate change by the live system during the run (for example a model switch in the
panel) also fails this test; rerun when the machine is quiet.
"""
import unittest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _isolation  # noqa: E402


class ProductionSentinel(unittest.TestCase):
    def test_real_files_untouched(self):
        if not _isolation.BEFORE:
            self.skipTest('isolation was not initialised (module run on its own)')
        for name, before in _isolation.BEFORE.items():
            with self.subTest(path=name):
                self.assertEqual(_isolation.stamp(Path(name)), before, f'{name} changed during the test run')


if __name__ == '__main__':
    unittest.main()
