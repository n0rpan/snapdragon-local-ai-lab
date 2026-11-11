import os
import random
import string
import sys
import unittest
from fractions import Fraction

sys.path.insert(0, os.environ["PROJECT"])
from alloc import allocate  # noqa: E402


def ref_allocate(budget, requests):
    """Independent reference: round-based water filling, then rounding by sort."""
    names = [n for n, _, _ in requests]
    total = sum(d for _, d, _ in requests)
    target = min(budget, total)
    x = {n: Fraction(0) for n in names}
    active = [(n, Fraction(d), w) for n, d, w in requests if d > 0]
    remaining = Fraction(target)
    rounds = 0
    while active:
        weight_sum = sum(w for _, _, w in active)
        level = remaining / weight_sum
        sat = [a for a in active if a[1] <= a[2] * level]
        if not sat:
            for n, _, w in active:
                x[n] = w * level
            break
        rounds += 1
        for n, d, _ in sat:
            x[n] = d
            remaining -= d
        active = [a for a in active if a not in sat]
    floors = {n: v.numerator // v.denominator for n, v in x.items()}
    extra = target - sum(floors.values())
    weights = {n: w for n, _, w in requests}
    cands = [n for n in names if x[n] > floors[n]]
    cands.sort(key=lambda n: (-(x[n] - floors[n]), -weights[n], n))
    out = dict(floors)
    for n in cands[:extra]:
        out[n] += 1
    return out, rounds


class TestExamples(unittest.TestCase):
    def test_all_satisfied(self):
        self.assertEqual(allocate(100, [("a", 5, 1), ("b", 3, 2)]), {"a": 5, "b": 3})
        self.assertEqual(allocate(8, [("a", 5, 1), ("b", 3, 2)]), {"a": 5, "b": 3})

    def test_budget_zero(self):
        self.assertEqual(allocate(0, [("a", 5, 1), ("b", 3, 2)]), {"a": 0, "b": 0})
        self.assertEqual(allocate(0, [("a", 0, 1)]), {"a": 0})

    def test_empty(self):
        self.assertEqual(allocate(5, []), {})
        self.assertEqual(allocate(0, []), {})

    def test_single(self):
        self.assertEqual(allocate(3, [("x", 4, 3)]), {"x": 3})
        self.assertEqual(allocate(9, [("x", 4, 3)]), {"x": 4})

    def test_equal_weights_thirds(self):
        r = [("a", 100, 1), ("b", 100, 1), ("c", 100, 1)]
        self.assertEqual(allocate(10, r), {"a": 4, "b": 3, "c": 3})
        self.assertEqual(allocate(2, r), {"a": 1, "b": 1, "c": 0})
        self.assertEqual(allocate(10, list(reversed(r))), {"a": 4, "b": 3, "c": 3})

    def test_unequal_weights(self):
        r = [("a", 100, 1), ("b", 100, 2), ("c", 100, 3)]
        self.assertEqual(allocate(10, r), {"a": 2, "b": 3, "c": 5})
        self.assertEqual(allocate(12, r), {"a": 2, "b": 4, "c": 6})
        self.assertEqual(allocate(3, r), {"a": 0, "b": 1, "c": 2})

    def test_weight_tiebreak_over_name(self):
        # a has the smaller name but the smaller weight; c wins the tie.
        self.assertEqual(allocate(3, [("a", 100, 1), ("b", 100, 2), ("c", 100, 3)]),
                         {"a": 0, "b": 1, "c": 2})

    def test_name_tiebreak(self):
        self.assertEqual(allocate(5, [("z", 100, 2), ("m", 100, 2)]), {"m": 3, "z": 2})
        self.assertEqual(allocate(5, [("m", 100, 2), ("z", 100, 2)]), {"m": 3, "z": 2})

    def test_zero_demand(self):
        self.assertEqual(allocate(5, [("a", 0, 1), ("b", 10, 1)]), {"a": 0, "b": 5})
        self.assertEqual(allocate(1, [("a", 0, 5), ("b", 0, 1)]), {"a": 0, "b": 0})
        self.assertEqual(allocate(4, [("a", 0, 9), ("b", 2, 1), ("c", 9, 1)]),
                         {"a": 0, "b": 2, "c": 2})

    def test_multi_round_frees_capacity(self):
        self.assertEqual(allocate(10, [("a", 1, 1), ("b", 100, 1), ("c", 100, 1)]),
                         {"a": 1, "b": 5, "c": 4})

    def test_three_rounds(self):
        r = [("a", 2, 1), ("b", 5, 2), ("c", 100, 1), ("d", 100, 3)]
        self.assertEqual(allocate(20, r), {"a": 2, "b": 5, "c": 3, "d": 10})

    def test_large_numbers_exact(self):
        r = [("a", 10 ** 12, 1), ("b", 10 ** 12, 3), ("c", 7, 1)]
        self.assertEqual(allocate(10 ** 12, r),
                         {"a": 249999999998, "b": 749999999995, "c": 7})
        self.assertEqual(allocate(10 ** 15 + 1, [("p", 10 ** 15, 2), ("q", 10 ** 15, 1)]),
                         {"p": 666666666666667, "q": 333333333333334})


class TestValidation(unittest.TestCase):
    def test_bad_budget(self):
        for bad in (-1, True, False, 2.0, "3"):
            with self.subTest(budget=bad):
                with self.assertRaises(ValueError):
                    allocate(bad, [])
                with self.assertRaises(ValueError):
                    allocate(bad, [("a", 1, 1)])

    def test_bad_demand(self):
        for bad in (-1, True, 1.0):
            with self.subTest(demand=bad):
                with self.assertRaises(ValueError):
                    allocate(5, [("a", bad, 1)])

    def test_bad_weight(self):
        for bad in (0, -2, True, 1.5):
            with self.subTest(weight=bad):
                with self.assertRaises(ValueError):
                    allocate(5, [("a", 1, bad)])

    def test_duplicate_names(self):
        with self.assertRaises(ValueError):
            allocate(5, [("a", 1, 1), ("a", 2, 2)])
        with self.assertRaises(ValueError):
            allocate(0, [("x", 0, 1), ("y", 0, 1), ("x", 0, 1)])


def _random_cases():
    rng = random.Random(1234)
    cases = []
    for k in range(300):
        n = rng.randint(4, 8)
        names = rng.sample(string.ascii_lowercase, n)
        scale = 10 ** 9 if k % 10 == 0 else 1
        reqs = []
        for nm in names:
            d = rng.randint(1, 10 ** rng.randint(0, 3)) * scale
            reqs.append((nm, d, rng.randint(1, 6)))
        total = sum(d for _, d, _ in reqs)
        if k % 7 == 6:
            budget = rng.randint(0, total + 5)
        else:
            budget = rng.randint(total // 5, (total * 4) // 5)
        cases.append((budget, reqs))
    return cases


class TestRandomized(unittest.TestCase):
    def test_against_reference_and_invariants(self):
        many_rounds = 0
        for idx, (budget, reqs) in enumerate(_random_cases()):
            with self.subTest(case=idx):
                got = allocate(budget, reqs)
                want, rounds = ref_allocate(budget, reqs)
                if rounds >= 3:
                    many_rounds += 1
                self.assertEqual(got, want)
                self.assertEqual(set(got), {n for n, _, _ in reqs})
                total = sum(d for _, d, _ in reqs)
                self.assertEqual(sum(got.values()), min(budget, total))
                for n, d, _ in reqs:
                    self.assertGreaterEqual(got[n], 0)
                    self.assertLessEqual(got[n], d)
        self.assertGreaterEqual(many_rounds, 10)


if __name__ == "__main__":
    unittest.main()
