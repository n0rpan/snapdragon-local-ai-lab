class Inventory:
    def __init__(self, stock):
        self.stock = stock

    def snapshot(self):
        return self.stock

    def take(self, items):
        for sku, quantity in items.items():
            if self.stock.get(sku, 0) < quantity:
                raise ValueError("insufficient stock")
            self.stock[sku] -= quantity
