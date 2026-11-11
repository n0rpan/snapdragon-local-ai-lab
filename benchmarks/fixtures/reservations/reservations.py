from validation import normalize

class Reservations:
    def __init__(self, inventory):
        self.inventory = inventory
        self.receipts = {}

    def reserve(self, request_id, lines):
        if request_id in self.receipts:
            return self.receipts[request_id]
        items = normalize(lines)
        self.inventory.take(items)
        receipt = {"request_id": request_id, "items": items}
        self.receipts[request_id] = receipt
        return receipt
