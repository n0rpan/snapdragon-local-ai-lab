import unittest

from alloc import allocate


class TestAllocate(unittest.TestCase):
    def test_all_satisfied(self):
        self.assertEqual(allocate(100, [("a", 5, 1), ("b", 3, 2)]), {"a": 5, "b": 3})

    def test_budget_zero(self):
        self.assertEqual(allocate(0, [("a", 5, 1), ("b", 3, 2)]), {"a": 0, "b": 0})

    def test_equal_weights_thirds(self):
        self.assertEqual(allocate(10, [("a", 100, 1), ("b", 100, 1), ("c", 100, 1)]),
                         {"a": 4, "b": 3, "c": 3})

    def test_unequal_weights(self):
        self.assertEqual(allocate(10, [("a", 100, 1), ("b", 100, 2), ("c", 100, 3)]),
                         {"a": 2, "b": 3, "c": 5})

    def test_small_demand_frees_capacity(self):
        self.assertEqual(allocate(10, [("a", 1, 1), ("b", 100, 1), ("c", 100, 1)]),
                         {"a": 1, "b": 5, "c": 4})

    def test_bool_weight_rejected(self):
        with self.assertRaises(ValueError):
            allocate(5, [("a", 3, True)])


if __name__ == "__main__":
    unittest.main()
