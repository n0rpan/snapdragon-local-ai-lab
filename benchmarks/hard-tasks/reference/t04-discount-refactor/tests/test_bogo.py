import unittest
from decimal import Decimal

from shop.cart import Customer, Line
from shop.checkout import checkout_total


class BogoTests(unittest.TestCase):
    def test_every_second_unit_free(self):
        lines = [Line("A", Decimal("10.00"), 3)]  # 30 - 10 = 20
        self.assertEqual(checkout_total(lines, Customer("c", False), "BOGO:A"), Decimal("20.00"))

    def test_bogo_before_tier(self):
        lines = [Line("A", Decimal("100.00"), 3)]  # 300 - 100 = 200 -> 5% -> 190
        self.assertEqual(checkout_total(lines, Customer("c", False), "BOGO:A"), Decimal("190.00"))


if __name__ == "__main__":
    unittest.main()
