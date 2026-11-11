Refactor ticket: discount logic is duplicated and has drifted.

shop/checkout.py has the discount rules that are correct today: treat it as the source of truth. shop/invoice.py has its own copy that has drifted (it applies the member discount before the bulk tier and rounds at each step), so invoice totals disagree with checkout totals. Fix this.

1. Create shop/discounts.py with
   apply_discounts(subtotal: Decimal, customer, coupon=None, lines=()) -> Decimal
   Steps, in order. Use Decimal arithmetic and round once at the end (ROUND_HALF_UP to 0.01):
   a. Coupon 'BOGO:<SKU>': for the lines with that sku, free units = floor(total qty of those lines / 2), each valued at the lowest unit_price among those lines. Subtract that from subtotal. A sku not in the cart gives no discount.
   b. Bulk tier on the result: >= 500 gives 10% off; else >= 200 gives 5% off.
   c. If customer.member: 2% off.
   d. Coupon 'SAVE10': 10.00 off, not below 0. Coupon 'PCT15': 15% off.
   Any other coupon that is not None raises ValueError.
2. checkout_total(lines, customer, coupon=None) and build_invoice(lines, customer, coupon=None) both call apply_discounts(..., lines=lines). Keep their names and signatures.
3. checkout.py and invoice.py contain no discount rule: they import apply_discounts from shop.discounts (or .discounts) and contain no discount rate, threshold or coupon code literal.
4. build_invoice returns {"lines": [{"sku", "qty", "line_total"}], "subtotal", "discount", "total"}. line_total = unit_price * qty. subtotal = sum of line_totals. discount = subtotal - total.
5. Add unit tests for BOGO in tests/. Run: python3 -m unittest discover -s tests -t .
