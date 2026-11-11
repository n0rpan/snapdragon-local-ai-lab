"""Plain-text stock report."""


def render(store):
    """Fixed-width table: a header line, then one line per item sorted by sku.

    Columns: SKU (8, left), NAME (16, left), ON_HAND (7, right), separated by one space.
    Each line is right-stripped. Lines are joined with newlines; no trailing newline.
    """
    lines = [f"{'SKU':<8} {'NAME':<16} {'ON_HAND':>7}"]
    for item in store.items():
        lines.append(f"{item.sku:<8} {item.name:<16} {item.on_hand:>7}")
    return "\n".join(line.rstrip() for line in lines)
