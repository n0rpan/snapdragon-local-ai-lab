"""Item model for the inventory package."""
from dataclasses import dataclass, field


@dataclass
class Item:
    """A stocked SKU.

    on_hand is the physical quantity; reservations maps order_id -> reserved qty.
    """

    sku: str
    name: str
    on_hand: int
    reservations: dict[str, int] = field(default_factory=dict)

    @property
    def reserved(self) -> int:
        """Total quantity held by reservations."""
        return sum(self.reservations.values())

    @property
    def available(self) -> int:
        """Quantity free to reserve or to ship without an order."""
        return self.on_hand - self.reserved
