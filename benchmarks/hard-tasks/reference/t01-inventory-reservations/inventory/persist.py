"""JSON persistence for Store.

Format: {"items": [{"sku", "name", "on_hand", "reservations"}, ...]}; reservation keys are
written sorted. Files without a "reservations" key load with no reservations.
"""
import json
import os

from inventory.models import Item
from inventory.store import Store


def load(path):
    """Load a store. A missing file means an empty store."""
    if not os.path.exists(path):
        return Store()
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return Store(
        Item(d["sku"], d["name"], d["on_hand"], dict(d.get("reservations", {})))
        for d in data["items"]
    )


def save(store, path):
    data = {"items": [
        {
            "sku": item.sku,
            "name": item.name,
            "on_hand": item.on_hand,
            "reservations": dict(sorted(item.reservations.items())),
        }
        for item in store.items()
    ]}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
