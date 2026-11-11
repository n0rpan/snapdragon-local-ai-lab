"""Invoice builder."""
from decimal import Decimal

from shop.discounts import apply_discounts


def build_invoice(lines, customer, coupon=None):
    """Return an invoice dict with per-line totals, subtotal, discount and total."""
    rows = [{"sku": line.sku, "qty": line.qty, "line_total": line.unit_price * line.qty} for line in lines]
    subtotal = sum((row["line_total"] for row in rows), Decimal("0"))
    total = apply_discounts(subtotal, customer, coupon, lines=lines)
    return {"lines": rows, "subtotal": subtotal, "discount": subtotal - total, "total": total}
