import unittest

from ratelimit.analyze import flag_bursts, top_paths


class FlagBurstsTest(unittest.TestCase):
    def test_basic_window(self):
        events = [(0.0, "a", "/x"), (1.0, "a", "/x"), (2.0, "a", "/x"), (10.0, "a", "/x")]
        # window 2: event 2 sees t in (0, 2] -> {1, 2} = 2; event 1 sees {0, 1} = 2
        self.assertEqual(flag_bursts(events, 2, 1), [1, 2])

    def test_window_lower_bound_is_exclusive(self):
        events = [(0, "a", "/"), (5, "a", "/")]
        self.assertEqual(flag_bursts(events, 5, 1), [])
        self.assertEqual(flag_bursts(events, 6, 1), [1])

    def test_equal_timestamps_and_other_clients(self):
        events = [(3, "b", "/"), (3, "a", "/"), (3, "a", "/"), (3, "a", "/")]
        self.assertEqual(flag_bursts(events, 1, 2), [1, 2, 3])
        self.assertEqual(flag_bursts(events, 1, 0), [0, 1, 2, 3])
        self.assertEqual(flag_bursts(events, 1, 3), [])
        self.assertEqual(flag_bursts(events, 0, 0), [])

    def test_out_of_order_input(self):
        events = [(5, "a", "/"), (1, "a", "/"), (3, "a", "/")]
        self.assertEqual(flag_bursts(events, 10, 1), [0, 2])

    def test_empty(self):
        self.assertEqual(flag_bursts([], 5, 0), [])


class TopPathsTest(unittest.TestCase):
    def test_ties_broken_by_path(self):
        events = [(0, "a", "/b"), (0, "a", "/a"), (0, "a", "/c"), (0, "a", "/c"), (0, "a", "/b")]
        self.assertEqual(top_paths(events, 2), [("/b", 2), ("/c", 2)])
        self.assertEqual(top_paths(events, 10), [("/b", 2), ("/c", 2), ("/a", 1)])

    def test_empty(self):
        self.assertEqual(top_paths([], 3), [])


if __name__ == "__main__":
    unittest.main()
