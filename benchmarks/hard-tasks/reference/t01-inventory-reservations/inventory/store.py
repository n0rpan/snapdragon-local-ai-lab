"""In-memory stock store with order reservations.

Quantities must be positive ints (ValueError otherwise). Unknown skus raise KeyError.
"""
from inventory.models import Item


class InsufficientStock(Exception):
    """A requested quantity is more than is available."""


def _check_qty(qty):
    if isinstance(qty, bool) or not isinstance(qty, int) or qty <= 0:
        raise ValueError(f"qty must be a positive integer, got {qty!r}")


class Store:
    """Items keyed by sku."""

    def __init__(self, items=()):
        self._items = {item.sku: item for item in items}

    def add_item(self, sku, name):
        """Add a sku with zero stock. An existing sku raises ValueError."""
        if sku in self._items:
            raise ValueError(f"sku already exists: {sku}")
        self._items[sku] = Item(sku, name, 0)

    def get(self, sku):
        if sku not in self._items:
            raise KeyError(f"unknown sku: {sku}")
        return self._items[sku]

    def items(self):
        """All items, sorted by sku."""
        return [self._items[sku] for sku in sorted(self._items)]

    def receive(self, sku, qty):
        _check_qty(qty)
        self.get(sku).on_hand += qty

    def reserve(self, sku, qty, order_id):
        """Hold qty of sku for order_id.

        ValueError if the order already holds a reservation for sku;
        InsufficientStock if qty is more than available.
        """
        _check_qty(qty)
        item = self.get(sku)
        if order_id in item.reservations:
            raise ValueError(f"order {order_id} already reserves {sku}")
        if qty > item.available:
            raise InsufficientStock(f"only {item.available} available of {sku}")
        item.reservations[order_id] = qty

    def release(self, sku, order_id):
        """Remove the order's reservation on sku."""
        item = self.get(sku)
        if order_id not in item.reservations:
            raise KeyError(f"no reservation for order {order_id} on {sku}")
        del item.reservations[order_id]

    def ship(self, sku, qty, order_id=None):
        """Ship qty of sku, optionally against an order's reservation."""
        _check_qty(qty)
        item = self.get(sku)
        if order_id is None:
            if qty > item.available:
                raise InsufficientStock(f"only {item.available} available of {sku}")
        else:
            if order_id not in item.reservations:
                raise KeyError(f"no reservation for order {order_id} on {sku}")
            held = item.reservations[order_id]
            if qty > held:
                raise ValueError(f"order {order_id} reserves only {held} of {sku}")
            if qty == held:
                del item.reservations[order_id]
            else:
                item.reservations[order_id] = held - qty
        item.on_hand -= qty
