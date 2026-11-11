"""In-memory stock store.

Quantities must be positive ints (ValueError otherwise). Unknown skus raise KeyError.
"""
from inventory.models import Item


class InsufficientStock(Exception):
    """A requested quantity is more than is on hand."""


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

    def ship(self, sku, qty):
        _check_qty(qty)
        item = self.get(sku)
        if qty > item.on_hand:
            raise InsufficientStock(f"only {item.on_hand} on hand of {sku}")
        item.on_hand -= qty
