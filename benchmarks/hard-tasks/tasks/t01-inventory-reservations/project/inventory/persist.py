"""JSON persistence for Store.

Format: {"items": [{"sku", "name", "on_hand"}, ...]}.
A missing file means an empty store.
"""
import json
import os

from inventory.models import Item
from inventory.store import Store


def load(path):
    if not os.path.exists(path):
        return Store()
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return Store(Item(d["sku"], d["name"], d["on_hand"]) for d in data["items"])


def save(store, path):
    data = {"items": [{"sku": i.sku, "name": i.name, "on_hand": i.on_hand} for i in store.items()]}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
