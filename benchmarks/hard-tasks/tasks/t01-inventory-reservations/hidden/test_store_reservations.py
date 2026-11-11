import os
import sys
import unittest

sys.path.insert(0, os.environ["PROJECT"])

from inventory.models import Item  # noqa: E402
from inventory.store import InsufficientStock, Store  # noqa: E402


def store_with_stock(on_hand=10):
    s = Store()
    s.add_item("A1", "Apple")
    s.receive("A1", on_hand)
    return s


class ExistingStoreBehaviour(unittest.TestCase):
    def test_ship_reduces_on_hand(self):
        s = store_with_stock()
        s.ship("A1", 4)
        self.assertEqual(s.get("A1").on_hand, 6)

    def test_ship_over_on_hand_is_insufficient(self):
        with self.assertRaises(InsufficientStock):
            store_with_stock().ship("A1", 11)

    def test_unknown_sku_is_keyerror(self):
        s = store_with_stock()
        calls = [
            lambda: s.get("ZZ"),
            lambda: s.receive("ZZ", 1),
            lambda: s.ship("ZZ", 1),
            lambda: s.reserve("ZZ", 1, "o1"),
            lambda: s.release("ZZ", "o1"),
        ]
        for call in calls:
            with self.subTest(call=call):
                with self.assertRaises(KeyError):
                    call()

    def test_bad_quantities_are_valueerror(self):
        s = store_with_stock()
        for bad in (0, -3, 1.5, "2", None):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    s.receive("A1", bad)
                with self.assertRaises(ValueError):
                    s.ship("A1", bad)
                with self.assertRaises(ValueError):
                    s.reserve("A1", bad, "o1")

    def test_items_sorted_by_sku(self):
        s = Store()
        for sku in ("C3", "A1", "B2"):
            s.add_item(sku, sku.lower())
        self.assertEqual([i.sku for i in s.items()], ["A1", "B2", "C3"])

    def test_item_defaults_and_positional_construction(self):
        it = Item("A1", "Apple", 5)
        self.assertEqual(it.reservations, {})
        self.assertEqual(it.reserved, 0)
        self.assertEqual(it.available, 5)
        other = Item("B2", "Banana", 1)
        it.reservations["x"] = 1
        self.assertEqual(other.reservations, {})

    def test_reserved_and_available_are_read_only(self):
        it = Item("A1", "Apple", 5)
        with self.assertRaises(AttributeError):
            it.reserved = 3
        with self.assertRaises(AttributeError):
            it.available = 3


class Reservations(unittest.TestCase):
    def test_reserve_updates_reserved_and_available(self):
        s = store_with_stock(10)
        s.reserve("A1", 6, "o1")
        it = s.get("A1")
        self.assertEqual(it.reservations, {"o1": 6})
        self.assertEqual(it.reserved, 6)
        self.assertEqual(it.available, 4)
        self.assertEqual(it.on_hand, 10)

    def test_reserve_checks_available(self):
        s = store_with_stock(10)
        s.reserve("A1", 6, "o1")
        with self.assertRaises(InsufficientStock):
            s.reserve("A1", 5, "o2")
        self.assertNotIn("o2", s.get("A1").reservations)
        s.reserve("A1", 4, "o2")
        self.assertEqual(s.get("A1").available, 0)

    def test_duplicate_order_is_valueerror_and_leaves_state(self):
        s = store_with_stock(10)
        s.reserve("A1", 2, "o1")
        with self.assertRaises(ValueError):
            s.reserve("A1", 1, "o1")
        self.assertEqual(s.get("A1").reservations, {"o1": 2})

    def test_same_order_on_other_sku_is_fine(self):
        s = store_with_stock(10)
        s.add_item("B2", "Banana")
        s.receive("B2", 3)
        s.reserve("A1", 2, "o1")
        s.reserve("B2", 3, "o1")
        self.assertEqual(s.get("B2").reservations, {"o1": 3})

    def test_release_frees_stock(self):
        s = store_with_stock(10)
        s.reserve("A1", 7, "o1")
        s.release("A1", "o1")
        self.assertEqual(s.get("A1").reservations, {})
        self.assertEqual(s.get("A1").available, 10)
        with self.assertRaises(KeyError):
            s.release("A1", "o1")

    def test_ship_without_order_uses_available(self):
        s = store_with_stock(10)
        s.reserve("A1", 6, "o1")
        with self.assertRaises(InsufficientStock):
            s.ship("A1", 5)
        self.assertEqual(s.get("A1").on_hand, 10)
        s.ship("A1", 4)
        it = s.get("A1")
        self.assertEqual((it.on_hand, it.reserved, it.available), (6, 6, 0))

    def test_partial_ship_against_reservation(self):
        s = store_with_stock(10)
        s.reserve("A1", 6, "o1")
        s.ship("A1", 2, order_id="o1")
        it = s.get("A1")
        self.assertEqual(it.reservations, {"o1": 4})
        self.assertEqual((it.on_hand, it.reserved, it.available), (8, 4, 4))

    def test_full_ship_removes_reservation(self):
        s = store_with_stock(10)
        s.reserve("A1", 6, "o1")
        s.ship("A1", 6, order_id="o1")
        it = s.get("A1")
        self.assertEqual(it.reservations, {})
        self.assertEqual((it.on_hand, it.available), (4, 4))

    def test_ship_against_order_exceeding_its_reservation(self):
        s = store_with_stock(10)
        s.reserve("A1", 3, "o1")
        s.reserve("A1", 5, "o2")
        with self.assertRaises(ValueError):
            s.ship("A1", 4, order_id="o1")
        self.assertEqual(s.get("A1").on_hand, 10)
        self.assertEqual(s.get("A1").reservations, {"o1": 3, "o2": 5})

    def test_ship_against_order_ignores_free_stock(self):
        s = store_with_stock(10)
        s.reserve("A1", 5, "o1")
        s.reserve("A1", 5, "o2")
        self.assertEqual(s.get("A1").available, 0)
        s.ship("A1", 3, order_id="o1")
        it = s.get("A1")
        self.assertEqual(it.reservations, {"o1": 2, "o2": 5})
        self.assertEqual((it.on_hand, it.available), (7, 0))

    def test_ship_against_unknown_order_is_keyerror(self):
        s = store_with_stock(10)
        s.reserve("A1", 3, "o1")
        with self.assertRaises(KeyError):
            s.ship("A1", 1, order_id="o9")
        self.assertEqual(s.get("A1").on_hand, 10)

    def test_ship_with_order_rejects_bad_qty(self):
        s = store_with_stock(10)
        s.reserve("A1", 3, "o1")
        for bad in (0, -1):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    s.ship("A1", bad, order_id="o1")


if __name__ == "__main__":
    unittest.main()
