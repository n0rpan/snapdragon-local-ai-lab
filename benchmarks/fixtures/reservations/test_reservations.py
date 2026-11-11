import unittest
from inventory import Inventory
from reservations import Reservations
from validation import normalize


class ReservationContract(unittest.TestCase):
    def setUp(self):
        self.inventory=Inventory({'apple':10,'pear':4})
        self.service=Reservations(self.inventory)

    def test_duplicate_lines_aggregate(self):
        self.assertEqual(normalize([('apple',2),('pear',1),('apple',3)]),{'apple':5,'pear':1})

    def test_line_validation(self):
        invalid=[None,'apple',{},[],[('apple',True)],[('apple',0)],
                 [('apple',-1)],[('apple',1.5)],[('',1)],[(None,1)],
                 [('apple',1,2)],['ab']]
        for lines in invalid:
            with self.subTest(lines=lines),self.assertRaises(ValueError):
                normalize(lines)

    def test_inventory_owns_input_and_snapshot(self):
        stock={'apple':10}
        inventory=Inventory(stock)
        stock['apple']=99
        self.assertEqual(inventory.snapshot(),{'apple':10})
        snapshot=inventory.snapshot();snapshot['apple']=88
        self.assertEqual(inventory.snapshot(),{'apple':10})

    def test_initial_stock_validation(self):
        for stock in ({'apple':True},{'apple':-1},{'apple':1.5},{'':1}):
            with self.subTest(stock=stock),self.assertRaises(ValueError):
                Inventory(stock)

    def test_take_is_atomic_for_each_item_order(self):
        for items in ({'apple':2,'pear':9},{'pear':9,'apple':2},
                      {'apple':2,'missing':1},{'missing':1,'apple':2}):
            with self.subTest(items=items),self.assertRaises(ValueError):
                self.inventory.take(items)
            self.assertEqual(self.inventory.snapshot(),{'apple':10,'pear':4})

    def test_take_validates_quantities_before_mutation(self):
        for invalid in (True,0,-2,1.5):
            with self.subTest(quantity=invalid),self.assertRaises(ValueError):
                self.inventory.take({'apple':1,'pear':invalid})
            self.assertEqual(self.inventory.snapshot(),{'apple':10,'pear':4})

    def test_reservation_aggregates_before_checking_stock(self):
        with self.assertRaises(ValueError):
            self.service.reserve('too-large',[('pear',3),('pear',2)])
        self.assertEqual(self.inventory.snapshot(),{'apple':10,'pear':4})

    def test_replay_uses_normalized_semantics(self):
        expected={'request_id':'r1','items':{'apple':3,'pear':1}}
        self.assertEqual(self.service.reserve('r1',[('apple',1),('pear',1),('apple',2)]),expected)
        self.assertEqual(self.service.reserve('r1',[('pear',1),('apple',3)]),expected)
        self.assertEqual(self.inventory.snapshot(),{'apple':7,'pear':3})

    def test_conflicting_replay_is_rejected(self):
        self.service.reserve('r1',[('apple',2)])
        with self.assertRaises(ValueError):
            self.service.reserve('r1',[('apple',3)])
        self.assertEqual(self.inventory.snapshot(),{'apple':8,'pear':4})

    def test_receipts_do_not_alias(self):
        receipt=self.service.reserve('r1',[('apple',2)])
        receipt['items']['apple']=999;receipt['request_id']='corrupted'
        again=self.service.reserve('r1',[('apple',2)])
        self.assertEqual(again,{'request_id':'r1','items':{'apple':2}})
        again['items']['apple']=888
        self.assertEqual(self.service.reserve('r1',[('apple',2)])['items'],{'apple':2})

    def test_failed_request_can_be_corrected(self):
        with self.assertRaises(ValueError):
            self.service.reserve('retry',[('apple',1),('pear',9)])
        self.assertEqual(self.service.reserve('retry',[('apple',2)])['items'],{'apple':2})
        self.assertEqual(self.inventory.snapshot(),{'apple':8,'pear':4})

    def test_request_id_validation(self):
        for request_id in ('',None,1,True):
            with self.subTest(request_id=request_id),self.assertRaises(ValueError):
                self.service.reserve(request_id,[('apple',1)])
        self.assertEqual(self.inventory.snapshot(),{'apple':10,'pear':4})


if __name__=='__main__':
    unittest.main(verbosity=2)
