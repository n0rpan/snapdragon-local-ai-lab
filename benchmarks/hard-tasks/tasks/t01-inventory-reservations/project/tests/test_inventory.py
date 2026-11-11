import contextlib
import io
import os
import tempfile
import unittest

from inventory import persist
from inventory.cli import main
from inventory.report import render
from inventory.store import InsufficientStock, Store


def make_store():
    store = Store()
    store.add_item("A1", "Apple")
    store.add_item("B2", "Banana")
    store.receive("A1", 10)
    return store


class ExistingBehaviourTests(unittest.TestCase):
    def test_receive_and_ship(self):
        store = make_store()
        store.ship("A1", 4)
        self.assertEqual(store.get("A1").on_hand, 6)

    def test_ship_more_than_on_hand_raises(self):
        with self.assertRaises(InsufficientStock):
            make_store().ship("A1", 11)

    def test_unknown_sku_raises_keyerror(self):
        with self.assertRaises(KeyError):
            make_store().receive("ZZ", 1)

    def test_qty_must_be_positive(self):
        with self.assertRaises(ValueError):
            make_store().receive("A1", 0)
        with self.assertRaises(ValueError):
            make_store().ship("A1", -2)

    def test_items_sorted_by_sku(self):
        store = Store()
        store.add_item("Z9", "Zed")
        store.add_item("A1", "Ay")
        self.assertEqual([i.sku for i in store.items()], ["A1", "Z9"])

    def test_persist_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "db.json")
            persist.save(make_store(), path)
            loaded = persist.load(path)
            self.assertEqual(loaded.get("A1").on_hand, 10)
            self.assertEqual(loaded.get("B2").name, "Banana")

    def test_report_has_item_line(self):
        lines = render(make_store()).split("\n")
        self.assertTrue(lines[0].startswith("SKU"))
        self.assertTrue(lines[1].startswith("A1"))

    def test_cli_ship_error_returns_1(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "db.json")
            self.assertEqual(main(["add", "A1", "Apple"], db), 0)
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                self.assertEqual(main(["ship", "A1", "3"], db), 1)
            self.assertTrue(err.getvalue().startswith("error:"))


class ReservationTests(unittest.TestCase):
    def test_reserve_reduces_available(self):
        store = make_store()
        store.reserve("A1", 4, "o1")
        item = store.get("A1")
        self.assertEqual(item.reserved, 4)
        self.assertEqual(item.available, 6)

    def test_ship_without_order_respects_reservations(self):
        store = make_store()
        store.reserve("A1", 8, "o1")
        with self.assertRaises(InsufficientStock):
            store.ship("A1", 3)

    def test_partial_ship_against_order(self):
        store = make_store()
        store.reserve("A1", 4, "o1")
        store.ship("A1", 1, order_id="o1")
        self.assertEqual(store.get("A1").reservations, {"o1": 3})
        self.assertEqual(store.get("A1").on_hand, 9)

    def test_release_unknown_order_raises(self):
        with self.assertRaises(KeyError):
            make_store().release("A1", "nope")


if __name__ == "__main__":
    unittest.main()
