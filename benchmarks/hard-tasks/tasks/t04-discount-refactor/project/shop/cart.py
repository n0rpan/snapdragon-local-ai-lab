"""Cart and customer value types."""
from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True)
class Line:
    """One cart line: `qty` units of `sku` at `unit_price` each."""
    sku: str
    unit_price: Decimal
    qty: int


@dataclass(frozen=True)
class Customer:
    id: str
    member: bool
