import os
import random
import sys
import unittest

sys.path.insert(0, os.environ["PROJECT"])

from minibuild.errors import CycleError, MissingSourceError  # noqa: E402
from minibuild.planner import plan  # noqa: E402


def ref_plan(goals, rules, mtimes, phony=frozenset()):
    """Recursive reference implementing the docstring spec (small graphs only)."""
    active, stale_of, stack, out = set(), {}, [], []

    def visit(name):
        active.add(name)
        stack.append(name)
        for dep in rules[name]:
            if dep in rules:
                if dep in active:
                    i = stack.index(dep)
                    raise CycleError(stack[i:] + [dep])
                if dep not in stale_of:
                    visit(dep)
            elif dep not in mtimes:
                raise MissingSourceError(dep)
        stack.pop()
        active.discard(name)
        s = name in phony or name not in mtimes or any(
            (d in rules and stale_of[d]) or (d in mtimes and mtimes[d] > mtimes[name])
            for d in rules[name])
        stale_of[name] = s
        if s:
            out.append(name)

    for g in goals:
        if g not in rules:
            if g not in mtimes:
                raise MissingSourceError(g)
            continue
        if g not in stale_of:
            visit(g)
    return out


class PlanHidden(unittest.TestCase):
    def test_nothing_stale_multilevel(self):
        rules = {'app': ['lib', 'cfg'], 'lib': ['a.c', 'a.h']}
        self.assertEqual(plan(['app'], rules, {'a.c': 1, 'a.h': 2, 'lib': 3, 'cfg': 4, 'app': 5}), [])

    def test_stale_chain_order(self):
        rules = {'a': ['b'], 'b': ['c'], 'c': ['src']}
        self.assertEqual(plan(['a'], rules, {'src': 9, 'c': 5, 'b': 6, 'a': 7}), ['c', 'b', 'a'])

    def test_older_than_one_of_two_deps(self):
        rules = {'t': ['x', 'y']}
        self.assertEqual(plan(['t'], rules, {'x': 1, 'y': 9, 't': 5}), ['t'])

    def test_equal_mtimes_not_stale(self):
        self.assertEqual(plan(['t'], {'t': ['x']}, {'x': 5, 't': 5}), [])

    def test_missing_target_file_is_stale(self):
        self.assertEqual(plan(['t'], {'t': ['x']}, {'x': 1}), ['t'])

    def test_phony_no_deps_stale(self):
        self.assertEqual(plan(['clean'], {'clean': []}, {}, phony={'clean'}), ['clean'])
        self.assertEqual(plan(['clean'], {'clean': []}, {'clean': 1}, phony={'clean'}), ['clean'])

    def test_phony_forces_dependents_chain(self):
        rules = {'top': ['mid'], 'mid': ['leaf'], 'leaf': ['s']}
        self.assertEqual(plan(['top'], rules, {'s': 1, 'leaf': 2, 'mid': 3, 'top': 4}, phony={'leaf'}),
                         ['leaf', 'mid', 'top'])

    def test_phony_name_that_is_not_a_target_is_ignored(self):
        self.assertEqual(plan(['t'], {'t': ['s']}, {'s': 1, 't': 2}, phony={'s'}), [])

    def test_stale_dep_counts_as_newer(self):
        self.assertEqual(plan(['t'], {'t': ['u'], 'u': ['s']}, {'s': 10, 'u': 1, 't': 5}), ['u', 't'])
        self.assertEqual(plan(['t'], {'t': ['u'], 'u': ['s']}, {'s': 1, 'u': 100, 't': 50}), ['t'])

    def test_non_stale_intermediate_not_output(self):
        rules = {'top': ['mid', 'x'], 'mid': ['s']}
        self.assertEqual(plan(['top'], rules, {'s': 1, 'mid': 5, 'x': 100, 'top': 6}), ['top'])

    def test_dependency_order_matters(self):
        rules = {'t': ['b', 'a'], 'a': ['s'], 'b': ['s']}
        self.assertEqual(plan(['t'], rules, {'s': 9, 'a': 1, 'b': 1, 't': 5}), ['b', 'a', 't'])

    def test_goal_order_matters(self):
        rules = {'a': ['s'], 'b': ['s']}
        mtimes = {'s': 9, 'a': 1, 'b': 1}
        self.assertEqual(plan(['b', 'a'], rules, mtimes), ['b', 'a'])
        self.assertEqual(plan(['a', 'b'], rules, mtimes), ['a', 'b'])

    def test_diamond_shared_dep_once_multiple_goals(self):
        rules = {'a': ['m'], 'b': ['m'], 'm': ['s']}
        self.assertEqual(plan(['a', 'b'], rules, {'s': 10, 'm': 1, 'a': 5, 'b': 5}), ['m', 'a', 'b'])

    def test_duplicate_goals_and_duplicate_deps(self):
        self.assertEqual(plan(['t', 't'], {'t': ['s', 's']}, {'s': 2, 't': 1}), ['t'])

    def test_self_loop(self):
        with self.assertRaises(CycleError) as ctx:
            plan(['a'], {'a': ['a']}, {'a': 1})
        self.assertEqual(ctx.exception.cycle, ['a', 'a'])

    def test_cycle_path_starts_at_first_repeated_node(self):
        rules = {'x': ['b'], 'a': ['b'], 'b': ['c'], 'c': ['a']}
        with self.assertRaises(CycleError) as ctx:
            plan(['x'], rules, {})
        self.assertEqual(ctx.exception.cycle, ['b', 'c', 'a', 'b'])

    def test_cycle_only_via_second_goal(self):
        rules = {'a': ['s'], 'q': ['r'], 'r': ['q']}
        with self.assertRaises(CycleError) as ctx:
            plan(['a', 'q'], rules, {'s': 1, 'a': 2})
        self.assertEqual(ctx.exception.cycle, ['q', 'r', 'q'])

    def test_missing_source_name_nested(self):
        with self.assertRaises(MissingSourceError) as ctx:
            plan(['out'], {'out': ['mid'], 'mid': ['deep.h']}, {'out': 1, 'mid': 1})
        self.assertEqual(ctx.exception.name, 'deep.h')

    def test_missing_source_not_reachable_no_error(self):
        rules = {'out': ['a.c'], 'unused': ['nope.h']}
        self.assertEqual(plan(['out'], rules, {'a.c': 1, 'out': 2}), [])

    def test_missing_source_reachable_via_second_goal(self):
        rules = {'out': ['a.c'], 'unused': ['nope.h']}
        with self.assertRaises(MissingSourceError) as ctx:
            plan(['out', 'unused'], rules, {'a.c': 1, 'out': 2})
        self.assertEqual(ctx.exception.name, 'nope.h')

    def test_goal_that_is_a_source(self):
        self.assertEqual(plan(['src'], {}, {'src': 1}), [])
        self.assertEqual(plan(['src', 't'], {'t': ['src']}, {'src': 5, 't': 1}), ['t'])
        with self.assertRaises(MissingSourceError) as ctx:
            plan(['src'], {}, {})
        self.assertEqual(ctx.exception.name, 'src')

    def test_deep_chain_no_recursion_error(self):
        n = 3000
        rules = {f'n{i}': [f'n{i + 1}'] for i in range(n)}
        mtimes = {f'n{i}': 5 for i in range(n)}
        mtimes[f'n{n}'] = 10
        self.assertEqual(plan(['n0'], rules, mtimes), [f'n{i}' for i in range(n - 1, -1, -1)])
        mtimes[f'n{n}'] = 1
        self.assertEqual(plan(['n0'], rules, mtimes), [])

    def test_deep_chain_cycle(self):
        n = 3000
        rules = {f'n{i}': [f'n{i + 1}'] for i in range(n)}
        rules[f'n{n}'] = ['n0']
        with self.assertRaises(CycleError) as ctx:
            plan(['n0'], rules, {})
        self.assertEqual(ctx.exception.cycle, [f'n{i}' for i in range(n + 1)] + ['n0'])

    def test_random_graphs_match_reference(self):
        rng = random.Random(20240611)
        for _ in range(400):
            n = rng.randint(1, 9)
            names = [f'n{i}' for i in range(n)]
            rules = {}
            for i in names:
                if rng.random() < 0.6:
                    deps = [rng.choice(names) for _ in range(rng.randint(0, 3))]
                    if rng.random() < 0.8:
                        # mostly a DAG: keep only deps that come later
                        deps = [d for d in deps if int(d[1:]) > int(i[1:])]
                    rules[i] = deps
            sources = [x for x in names if x not in rules]
            mtimes = {}
            for x in names + ['extra.h']:
                if rng.random() < 0.85:
                    mtimes[x] = rng.randint(0, 3)
            phony = {x for x in names if rng.random() < 0.15}
            goals = [rng.choice(names + sources) for _ in range(rng.randint(1, 3))]

            def outcome(fn):
                try:
                    return ('ok', fn(goals, rules, dict(mtimes), phony))
                except CycleError as e:
                    return ('cycle', list(e.cycle))
                except MissingSourceError as e:
                    return ('missing', e.name)

            self.assertEqual(outcome(plan), outcome(ref_plan), (goals, rules, mtimes, phony))

    def test_returns_list_of_unique_targets(self):
        rules = {'t': ['u', 'v'], 'u': ['s'], 'v': ['s']}
        result = plan(['t'], rules, {'s': 9, 'u': 1, 'v': 1, 't': 2})
        self.assertIsInstance(result, list)
        self.assertEqual(result, ['u', 'v', 't'])


if __name__ == '__main__':
    unittest.main()
