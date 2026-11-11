import unittest

from calc import CalcError, evaluate


class PrecedenceTests(unittest.TestCase):
    def test_mul_before_add(self):
        self.assertEqual(evaluate("1+2*3"), 7)

    def test_parentheses(self):
        self.assertEqual(evaluate("(1+2)*3"), 9)

    def test_subtraction_is_left_associative(self):
        self.assertEqual(evaluate("10-4-3"), 3)

    def test_power_is_right_associative(self):
        self.assertEqual(evaluate("2^3^2"), 512)

    def test_unary_minus_binds_looser_than_power(self):
        self.assertEqual(evaluate("-2^2"), -4)

    def test_negative_exponent(self):
        self.assertEqual(evaluate("2^-1"), 0.5)

    def test_unary_chain_and_product(self):
        self.assertEqual(evaluate("--2"), 2)
        self.assertEqual(evaluate("2*-3"), -6)


class NumberTests(unittest.TestCase):
    def test_decimal_forms(self):
        self.assertAlmostEqual(evaluate(".5+3.25"), 3.75)

    def test_exponent_without_sign(self):
        self.assertEqual(evaluate("1e3"), 1000)

    def test_exponent_with_minus_sign(self):
        self.assertAlmostEqual(evaluate("1e-3+1"), 1.001)


class TypeTests(unittest.TestCase):
    def test_int_addition_stays_int(self):
        self.assertIsInstance(evaluate("2+3"), int)

    def test_int_subtraction_stays_int(self):
        self.assertIsInstance(evaluate("7-2"), int)

    def test_division_is_true_division(self):
        self.assertEqual(evaluate("7/2"), 3.5)
        self.assertIsInstance(evaluate("6/3"), float)


class ErrorTests(unittest.TestCase):
    def test_division_by_zero_pos_is_operator(self):
        with self.assertRaises(CalcError) as ctx:
            evaluate("2 / 0")
        self.assertEqual(ctx.exception.pos, 2)

    def test_unexpected_end_pos(self):
        with self.assertRaises(CalcError) as ctx:
            evaluate("2+")
        self.assertEqual(ctx.exception.pos, 2)

    def test_unexpected_character_pos(self):
        with self.assertRaises(CalcError) as ctx:
            evaluate("3 $ 4")
        self.assertEqual(ctx.exception.pos, 2)


if __name__ == "__main__":
    unittest.main()
