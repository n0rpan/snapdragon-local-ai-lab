import unittest
from decimal import Decimal

from shop.cart import Customer, Line
from shop.checkout import checkout_total
from shop.invoice import build_invoice


class CheckoutTests(unittest.TestCase):
    def test_bulk_tier_10_percent(self):
        lines = [Line("A", Decimal("250.00"), 2)]  # 500.00 -> 450.00
        self.assertEqual(checkout_total(lines, Customer("c1", False)), Decimal("450.00"))

    def test_member_then_save10(self):
        lines = [Line("B", Decimal("20.00"), 1)]  # 20 * 0.98 = 19.60, - 10 = 9.60
        self.assertEqual(checkout_total(lines, Customer("c2", True), "SAVE10"), Decimal("9.60"))

    def test_pct15(self):
        lines = [Line("C", Decimal("100.00"), 1)]
        self.assertEqual(checkout_total(lines, Customer("c3", False), "PCT15"), Decimal("85.00"))


class InvoiceTests(unittest.TestCase):
    def test_plain_invoice(self):
        inv = build_invoice([Line("A", Decimal("10.00"), 2)], Customer("c4", False))
        self.assertEqual(inv["total"], Decimal("20.00"))
        self.assertEqual(inv["discount"], Decimal("0.00"))

    def test_member_bulk_matches_checkout(self):
        lines = [Line("A", Decimal("250.00"), 2)]
        inv = build_invoice(lines, Customer("c5", True))
        self.assertEqual(inv["total"], checkout_total(lines, Customer("c5", True)))


if __name__ == "__main__":
    unittest.main()
