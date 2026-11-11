"""Checkout: the amount a customer pays for a cart."""
from decimal import Decimal

from shop.discounts import apply_discounts


def checkout_total(lines, customer, coupon=None):
    """Return the payable total for `lines` after bulk, member and coupon discounts."""
    subtotal = sum((line.unit_price * line.qty for line in lines), Decimal("0"))
    return apply_discounts(subtotal, customer, coupon, lines=lines)
