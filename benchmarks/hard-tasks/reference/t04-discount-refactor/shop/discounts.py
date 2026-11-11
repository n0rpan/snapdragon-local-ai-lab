"""Discount rules: the single source of truth for checkout and invoices."""
from decimal import Decimal, ROUND_HALF_UP

CENT = Decimal("0.01")
ZERO = Decimal("0")
BULK_TIERS = ((Decimal("500"), Decimal("0.10")), (Decimal("200"), Decimal("0.05")))
MEMBER_RATE = Decimal("0.02")
FIXED_COUPONS = {"SAVE10": Decimal("10.00")}
PERCENT_COUPONS = {"PCT15": Decimal("0.15")}
BOGO_PREFIX = "BOGO:"


def _bogo_value(lines, sku):
    """Value of every second unit of `sku` (summed over its lines) given free."""
    matching = [line for line in lines if line.sku == sku]
    if not matching:
        return ZERO
    free_units = sum(line.qty for line in matching) // 2
    return min(line.unit_price for line in matching) * free_units


def apply_discounts(subtotal, customer, coupon=None, lines=()):
    """Return the payable total for `subtotal` after all discounts, rounded to cents."""
    is_bogo = coupon is not None and coupon.startswith(BOGO_PREFIX)
    if coupon is not None and not is_bogo and coupon not in FIXED_COUPONS and coupon not in PERCENT_COUPONS:
        raise ValueError(f"unknown coupon: {coupon}")
    total = Decimal(subtotal)
    if is_bogo:
        total -= _bogo_value(lines, coupon[len(BOGO_PREFIX):])
    for threshold, rate in BULK_TIERS:
        if total >= threshold:
            total *= 1 - rate
            break
    if customer.member:
        total *= 1 - MEMBER_RATE
    if coupon in FIXED_COUPONS:
        total = max(total - FIXED_COUPONS[coupon], ZERO)
    elif coupon in PERCENT_COUPONS:
        total *= 1 - PERCENT_COUPONS[coupon]
    return total.quantize(CENT, rounding=ROUND_HALF_UP)
