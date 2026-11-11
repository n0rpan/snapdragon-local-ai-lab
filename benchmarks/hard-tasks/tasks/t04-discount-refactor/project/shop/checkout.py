"""Checkout: the amount a customer pays for a cart. Discount rules live here (source of truth)."""
from decimal import Decimal, ROUND_HALF_UP


def checkout_total(lines, customer, coupon=None):
    """Return the payable total for `lines` after bulk, member and coupon discounts."""
    subtotal = sum((line.unit_price * line.qty for line in lines), Decimal("0"))
    total = subtotal
    if total >= Decimal("500"):
        total = total * Decimal("0.90")
    elif total >= Decimal("200"):
        total = total * Decimal("0.95")
    if customer.member:
        total = total * Decimal("0.98")
    if coupon == "SAVE10":
        total = max(total - Decimal("10.00"), Decimal("0"))
    elif coupon == "PCT15":
        total = total * Decimal("0.85")
    elif coupon is not None:
        raise ValueError(f"unknown coupon: {coupon}")
    return total.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
