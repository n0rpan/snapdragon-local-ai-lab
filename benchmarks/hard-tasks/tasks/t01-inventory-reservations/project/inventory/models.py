"""Item model for the inventory package."""
from dataclasses import dataclass


@dataclass
class Item:
    """A stocked SKU with its physical quantity."""

    sku: str
    name: str
    on_hand: int
