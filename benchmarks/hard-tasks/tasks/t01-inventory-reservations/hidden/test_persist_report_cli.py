import contextlib
import io
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.environ["PROJECT"])

from inventory import persist  # noqa: E402
from inventory.cli import main  # noqa: E402
from inventory.models import Item  # noqa: E402
from inventory.report import render  # noqa: E402
from inventory.store import Store  # noqa: E402

REPORT_3 = "\n".join([
    "SKU      NAME             ON_HAND RESERVED AVAILABLE",
    "A1       Apple                 10        5         5",
    "B2       Banana                 7        3         4",
    "C3       Cherry Tomatoes     1200        0      1200",
])


def three_item_store():
    return Store([
        Item("B2", "Banana", 7, {"o1": 3}),
        Item("A1", "Apple", 10, {"o2": 4, "o1": 1}),
        Item("C3", "Cherry Tomatoes", 1200),
    ])


def run_cli(argv, db):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = main(argv, db)
    return rc, out.getvalue(), err.getvalue()


class PersistTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self._tmp.name, "db.json")

    def tearDown(self):
        self._tmp.cleanup()

    def test_round_trip_keeps_reservations(self):
        persist.save(three_item_store(), self.path)
        loaded = persist.load(self.path)
        a1 = loaded.get("A1")
        self.assertEqual(a1.reservations, {"o1": 1, "o2": 4})
        self.assertEqual((a1.on_hand, a1.reserved, a1.available), (10, 5, 5))
        self.assertEqual(loaded.get("B2").reservations, {"o1": 3})
        self.assertEqual(loaded.get("C3").reservations, {})

    def test_saved_reservation_keys_are_sorted(self):
        s = Store([Item("A1", "Apple", 10, {"z9": 1, "a1": 2, "m5": 3})])
        persist.save(s, self.path)
        with open(self.path, encoding="utf-8") as f:
            data = json.load(f)
        item = data["items"][0]
        self.assertEqual(set(item), {"sku", "name", "on_hand", "reservations"})
        self.assertEqual(list(item["reservations"]), ["a1", "m5", "z9"])

    def test_old_format_file_loads_without_reservations(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump({"items": [{"sku": "A1", "name": "Apple", "on_hand": 5}]}, f)
        it = persist.load(self.path).get("A1")
        self.assertEqual(it.reservations, {})
        self.assertEqual((it.reserved, it.available), (0, 5))

    def test_missing_file_is_empty_store(self):
        self.assertEqual(persist.load(self.path).items(), [])


class ReportTests(unittest.TestCase):
    def test_exact_three_item_report(self):
        self.assertEqual(render(three_item_store()), REPORT_3)

    def test_empty_store_is_header_only(self):
        self.assertEqual(render(Store()), "SKU      NAME             ON_HAND RESERVED AVAILABLE")

    def test_no_trailing_newline(self):
        self.assertFalse(render(three_item_store()).endswith("\n"))


class CliTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.db = os.path.join(self._tmp.name, "inv.json")
        self.assertEqual(run_cli(["add", "A1", "Apple"], self.db)[0], 0)
        self.assertEqual(run_cli(["add", "B2", "Banana"], self.db)[0], 0)
        self.assertEqual(run_cli(["receive", "A1", "10"], self.db)[0], 0)
        self.assertEqual(run_cli(["receive", "B2", "5"], self.db)[0], 0)

    def tearDown(self):
        self._tmp.cleanup()

    def assert_error(self, result):
        rc, out, err = result
        self.assertEqual(rc, 1)
        self.assertEqual(out, "")
        self.assertTrue(err.startswith("error:"), err)

    def test_success_prints_nothing(self):
        self.assertEqual(run_cli(["reserve", "A1", "4", "o1"], self.db), (0, "", ""))
        self.assertEqual(run_cli(["release", "A1", "o1"], self.db), (0, "", ""))

    def test_report_stdout(self):
        run_cli(["reserve", "A1", "4", "o1"], self.db)
        rc, out, err = run_cli(["report"], self.db)
        self.assertEqual((rc, err), (0, ""))
        self.assertEqual(out, "\n".join([
            "SKU      NAME             ON_HAND RESERVED AVAILABLE",
            "A1       Apple                 10        4         6",
            "B2       Banana                 5        0         5",
        ]) + "\n")

    def test_ship_without_order_over_available_fails_and_saves_nothing(self):
        run_cli(["reserve", "A1", "8", "o1"], self.db)
        self.assert_error(run_cli(["ship", "A1", "3"], self.db))
        self.assertEqual(persist.load(self.db).get("A1").on_hand, 10)

    def test_ship_without_order_within_available(self):
        run_cli(["reserve", "A1", "8", "o1"], self.db)
        self.assertEqual(run_cli(["ship", "A1", "2"], self.db)[0], 0)
        self.assertEqual(persist.load(self.db).get("A1").on_hand, 8)

    def test_partial_ship_with_order(self):
        run_cli(["reserve", "A1", "4", "o1"], self.db)
        self.assertEqual(run_cli(["ship", "A1", "1", "--order", "o1"], self.db)[0], 0)
        it = persist.load(self.db).get("A1")
        self.assertEqual((it.on_hand, it.reservations), (9, {"o1": 3}))

    def test_ship_order_more_than_reserved_fails(self):
        run_cli(["reserve", "A1", "2", "o1"], self.db)
        self.assert_error(run_cli(["ship", "A1", "3", "--order", "o1"], self.db))
        self.assertEqual(persist.load(self.db).get("A1").on_hand, 10)

    def test_full_ship_with_order_clears_reservation(self):
        run_cli(["reserve", "A1", "2", "o1"], self.db)
        self.assertEqual(run_cli(["ship", "A1", "2", "--order", "o1"], self.db)[0], 0)
        it = persist.load(self.db).get("A1")
        self.assertEqual((it.on_hand, it.reservations), (8, {}))

    def test_release_unknown_order_fails(self):
        self.assert_error(run_cli(["release", "A1", "nope"], self.db))

    def test_reserve_duplicate_order_fails(self):
        run_cli(["reserve", "A1", "2", "o1"], self.db)
        self.assert_error(run_cli(["reserve", "A1", "1", "o1"], self.db))
        self.assertEqual(persist.load(self.db).get("A1").reservations, {"o1": 2})

    def test_reserve_insufficient_fails(self):
        self.assert_error(run_cli(["reserve", "A1", "11", "o1"], self.db))
        self.assertEqual(persist.load(self.db).get("A1").reservations, {})

    def test_reserve_zero_fails(self):
        self.assert_error(run_cli(["reserve", "A1", "0", "o1"], self.db))

    def test_ship_unknown_sku_fails(self):
        self.assert_error(run_cli(["ship", "ZZ", "1"], self.db))


if __name__ == "__main__":
    unittest.main()
