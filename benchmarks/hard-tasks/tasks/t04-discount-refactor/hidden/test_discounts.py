"""Hidden checks for the discount refactor: oracle agreement, invoice shape, BOGO, errors, structure."""
import ast
import os
import random
import sys
import unittest
from decimal import Decimal, ROUND_HALF_UP

sys.path.insert(0, os.environ["PROJECT"])

from shop import discounts  # noqa: E402
from shop.cart import Customer, Line  # noqa: E402
from shop.checkout import checkout_total  # noqa: E402
from shop.invoice import build_invoice  # noqa: E402

D = Decimal
CENT = D("0.01")
COUPONS = [None, "SAVE10", "PCT15", "BOGO:A", "BOGO:B", "BOGO:Z"]


def oracle(lines, member, coupon):
    """Embedded copy of the original checkout rules, extended with the BOGO spec."""
    subtotal = sum((line.unit_price * line.qty for line in lines), D("0"))
    if coupon is not None and coupon.startswith("BOGO:"):
        same = [line for line in lines if line.sku == coupon[5:]]
        if same:
            free = sum(line.qty for line in same) // 2
            subtotal -= min(line.unit_price for line in same) * free
    total = subtotal
    if total >= D("500"):
        total = total * D("0.90")
    elif total >= D("200"):
        total = total * D("0.95")
    if member:
        total = total * D("0.98")
    if coupon == "SAVE10":
        total = max(total - D("10"), D("0"))
    elif coupon == "PCT15":
        total = total * D("0.85")
    return total.quantize(CENT, rounding=ROUND_HALF_UP)


def random_case(seed):
    rng = random.Random(seed)
    lines = []
    for _ in range(rng.randint(1, 4)):
        cents = rng.randint(100, 15000)
        lines.append(Line(rng.choice("ABC"), D(cents).scaleb(-2), rng.randint(1, 6)))
    return lines, Customer(f"c{seed}", rng.random() < 0.5), rng.choice(COUPONS)


class RandomAgreement(unittest.TestCase):
    def test_forty_seeded_carts(self):
        for seed in range(40):
            lines, cust, coupon = random_case(seed)
            expected = oracle(lines, cust.member, coupon)
            with self.subTest(seed=seed, coupon=coupon):
                self.assertEqual(checkout_total(lines, cust, coupon), expected)
                inv = build_invoice(lines, cust, coupon)
                self.assertEqual(inv["total"], expected)
                self.assertEqual(inv["discount"], inv["subtotal"] - inv["total"])
                self.assertEqual(
                    discounts.apply_discounts(inv["subtotal"], cust, coupon, lines=lines), expected)

    def test_tier_boundaries(self):
        for price in ["199.99", "200.00", "499.99", "500.00"]:
            for member in (False, True):
                for coupon in (None, "PCT15", "SAVE10"):
                    lines = [Line("T", D(price), 1)]
                    cust = Customer("b", member)
                    expected = oracle(lines, member, coupon)
                    with self.subTest(price=price, member=member, coupon=coupon):
                        self.assertEqual(checkout_total(lines, cust, coupon), expected)
                        self.assertEqual(build_invoice(lines, cust, coupon)["total"], expected)

    def test_half_up_rounding(self):
        self.assertEqual(checkout_total([Line("X", D("10.25"), 1)], Customer("h", True)), D("10.05"))
        self.assertEqual(checkout_total([Line("Y", D("10.10"), 1)], Customer("h", False), "PCT15"), D("8.59"))
        self.assertEqual(build_invoice([Line("X", D("10.25"), 1)], Customer("h", True))["total"], D("10.05"))


class InvoiceShape(unittest.TestCase):
    def test_lines_and_discount(self):
        lines = [Line("A", D("12.50"), 3), Line("B", D("3.33"), 7)]
        inv = build_invoice(lines, Customer("i", True), "PCT15")
        self.assertTrue({"lines", "subtotal", "discount", "total"} <= set(inv))
        self.assertEqual(inv["lines"][0], {"sku": "A", "qty": 3, "line_total": D("37.50")})
        self.assertEqual(inv["lines"][1], {"sku": "B", "qty": 7, "line_total": D("23.31")})
        self.assertEqual(inv["subtotal"], D("60.81"))
        self.assertEqual(inv["total"], D("50.65"))
        self.assertEqual(inv["discount"], D("10.16"))


class Bogo(unittest.TestCase):
    def both(self, lines, member, coupon):
        cust = Customer("g", member)
        total = checkout_total(lines, cust, coupon)
        self.assertEqual(build_invoice(lines, cust, coupon)["total"], total)
        return total

    def test_qty1_no_free_unit(self):
        self.assertEqual(self.both([Line("A", D("10.00"), 1)], False, "BOGO:A"), D("10.00"))

    def test_qty3(self):
        self.assertEqual(self.both([Line("A", D("10.00"), 3)], False, "BOGO:A"), D("20.00"))

    def test_qty5_floor(self):
        self.assertEqual(self.both([Line("A", D("10.00"), 5)], False, "BOGO:A"), D("30.00"))

    def test_sku_on_two_lines_sums_before_floor(self):
        lines = [Line("A", D("10.00"), 1), Line("A", D("10.00"), 1), Line("B", D("5.00"), 1)]
        self.assertEqual(self.both(lines, False, "BOGO:A"), D("15.00"))

    def test_two_lines_qty_1_and_2(self):
        lines = [Line("A", D("10.00"), 1), Line("A", D("10.00"), 2)]
        self.assertEqual(self.both(lines, False, "BOGO:A"), D("20.00"))

    def test_sku_not_in_cart(self):
        self.assertEqual(self.both([Line("A", D("10.00"), 4)], False, "BOGO:Z"), D("40.00"))

    def test_tier_uses_subtotal_after_bogo(self):
        # 300 - 100 = 200 -> 5% tier -> 190 (tier first would give 185)
        self.assertEqual(self.both([Line("A", D("100.00"), 3)], False, "BOGO:A"), D("190.00"))

    def test_tier_boundary_after_bogo(self):
        # 750 - 250 = 500 -> 10% -> 450
        self.assertEqual(self.both([Line("A", D("250.00"), 3)], False, "BOGO:A"), D("450.00"))

    def test_bogo_with_member(self):
        self.assertEqual(self.both([Line("A", D("100.00"), 3)], True, "BOGO:A"), D("186.20"))

    def test_apply_discounts_direct(self):
        got = discounts.apply_discounts(D("300"), Customer("g", False), "BOGO:A",
                                        lines=[Line("A", D("100"), 3)])
        self.assertEqual(got, D("190.00"))

    def test_apply_discounts_defaults(self):
        self.assertEqual(discounts.apply_discounts(D("300"), Customer("g", False)), D("285.00"))
        self.assertEqual(discounts.apply_discounts(D("300"), Customer("g", False), "BOGO:A"), D("285.00"))


class Errors(unittest.TestCase):
    def test_unknown_coupon_raises_in_both(self):
        lines = [Line("A", D("10"), 1)]
        cust = Customer("e", False)
        with self.assertRaises(ValueError):
            checkout_total(lines, cust, "NOPE")
        with self.assertRaises(ValueError):
            build_invoice(lines, cust, "NOPE")
        with self.assertRaises(ValueError):
            discounts.apply_discounts(D("10"), cust, "NOPE")


class Structure(unittest.TestCase):
    def read(self, name):
        with open(os.path.join(os.environ["PROJECT"], "shop", name), encoding="utf-8") as handle:
            return handle.read()

    def test_apply_discounts_exists(self):
        self.assertTrue(callable(getattr(discounts, "apply_discounts", None)))

    def test_callers_import_from_discounts(self):
        for name in ("checkout.py", "invoice.py"):
            tree = ast.parse(self.read(name))
            found = any(
                isinstance(node, ast.ImportFrom)
                and node.module in ("discounts", "shop.discounts")
                and any(alias.name == "apply_discounts" for alias in node.names)
                for node in ast.walk(tree))
            self.assertTrue(found, f"{name} does not import apply_discounts from shop.discounts")

    def test_no_discount_literals_left(self):
        for name in ("checkout.py", "invoice.py"):
            text = self.read(name)
            for literal in ("500", "0.02", "SAVE10", "PCT15"):
                self.assertNotIn(literal, text, f"{name} still contains {literal!r}")


if __name__ == "__main__":
    unittest.main()
