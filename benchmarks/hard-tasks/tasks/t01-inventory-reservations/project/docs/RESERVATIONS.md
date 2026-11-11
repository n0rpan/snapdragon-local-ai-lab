# Reservations

Stock can be reserved for an order before it ships. Reserved stock stays in on_hand
but cannot be sold to anyone else.

## Item

- `reservations: dict[str, int]`: order_id -> reserved qty. Default: empty.
- `reserved` (read-only property): sum of the reservation quantities.
- `available` (read-only property): `on_hand - reserved`.

## Store

- `reserve(sku, qty, order_id)`: qty must be a positive int, else ValueError.
  If order_id already holds a reservation for sku, ValueError.
  If qty > available, InsufficientStock. Otherwise record `reservations[order_id] = qty`.
- `release(sku, order_id)`: remove that order's reservation on sku.
  Unknown order_id on sku: KeyError.
- `ship(sku, qty, order_id=None)`: qty must be a positive int, else ValueError.
  - Without order_id: qty must be <= available (not on_hand), else InsufficientStock.
  - With order_id: the order must hold a reservation on sku (else KeyError). qty must be
    <= that reservation's qty, else ValueError. The reservation is reduced by qty and
    removed when it reaches 0.
  - In both cases on_hand is reduced by qty.

## Persistence

- Each item is saved with `"reservations": {order_id: qty}`, keys sorted.
- Files written before this feature have no "reservations" key; they load with none.

## Report

Columns: SKU (8, left), NAME (16, left), ON_HAND (7, right), RESERVED (8, right),
AVAILABLE (9, right), separated by one space. Each line is right-stripped.
One header line, then one line per item sorted by sku.

## CLI

- `reserve SKU QTY ORDER`
- `release SKU ORDER`
- `ship SKU QTY [--order ORDER]`

Successful commands print nothing (report prints the table). Errors print
`error: <message>` to stderr, return 1, and leave the database unchanged.
