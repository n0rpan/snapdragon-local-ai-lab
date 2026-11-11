import unittest

from minibuild.errors import CycleError, MissingSourceError
from minibuild.planner import plan


class PlanTests(unittest.TestCase):
    def test_nothing_stale(self):
        rules = {'out': ['in.c']}
        self.assertEqual(plan(['out'], rules, {'in.c': 1, 'out': 2}), [])

    def test_stale_source_propagates_up_chain(self):
        rules = {'app': ['lib'], 'lib': ['main.c']}
        mtimes = {'main.c': 5, 'lib': 3, 'app': 4}
        self.assertEqual(plan(['app'], rules, mtimes), ['lib', 'app'])

    def test_diamond_order(self):
        rules = {'top': ['left', 'right'], 'left': ['base'], 'right': ['base'], 'base': ['src']}
        mtimes = {'src': 10, 'base': 1, 'left': 20, 'right': 20, 'top': 20}
        self.assertEqual(plan(['top'], rules, mtimes), ['base', 'left', 'right', 'top'])

    def test_missing_source_error_name(self):
        with self.assertRaises(MissingSourceError) as ctx:
            plan(['out'], {'out': ['in.c']}, {'out': 1})
        self.assertEqual(ctx.exception.name, 'in.c')

    def test_phony_forces_dependents(self):
        rules = {'all': ['obj'], 'obj': ['a.c']}
        mtimes = {'a.c': 1, 'obj': 5, 'all': 6}
        self.assertEqual(plan(['all'], rules, mtimes, phony={'obj'}), ['obj', 'all'])

    def test_three_cycle_path(self):
        rules = {'a': ['b'], 'b': ['c'], 'c': ['a']}
        with self.assertRaises(CycleError) as ctx:
            plan(['a'], rules, {})
        self.assertEqual(ctx.exception.cycle, ['a', 'b', 'c', 'a'])


if __name__ == '__main__':
    unittest.main()
