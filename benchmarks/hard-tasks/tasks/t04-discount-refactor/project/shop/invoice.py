"""Invoice builder. Carries its own copy of the discount rules (which have drifted)."""
from decimal import Decimal, ROUND_HALF_UP

CENT = Decimal("0.01")


def build_invoice(lines, customer, coupon=None):
    """Return an invoice dict with per-line totals, subtotal, discount and total."""
    rows, subtotal = [], Decimal("0")
    for line in lines:
        amount = line.unit_price * line.qty
        rows.append({"sku": line.sku, "qty": line.qty, "line_total": amount})
        subtotal += amount
    total = subtotal
    # Member discount first, rounding at every step.
    if customer.member:
        total = (total * Decimal("0.98")).quantize(CENT, ROUND_HALF_UP)
    if total >= Decimal("500"):
        total = (total * Decimal("0.90")).quantize(CENT, ROUND_HALF_UP)
    elif total >= Decimal("200"):
        total = (total * Decimal("0.95")).quantize(CENT, ROUND_HALF_UP)
    if coupon == "SAVE10":
        total = max(total - Decimal("10.00"), Decimal("0"))
    elif coupon == "PCT15":
        total = (total * Decimal("0.85")).quantize(CENT, ROUND_HALF_UP)
    elif coupon is not None:
        raise ValueError(f"unknown coupon: {coupon}")
    return {"lines": rows, "subtotal": subtotal, "discount": subtotal - total, "total": total}
