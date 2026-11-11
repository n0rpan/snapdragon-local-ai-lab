import os
import sys
import unittest

sys.path.insert(0, os.environ["PROJECT"])

from calc import CalcError, evaluate  # noqa: E402


class SpecCase(unittest.TestCase):
    def check(self, expr, expected):
        got = evaluate(expr)
        self.assertIs(type(got), type(expected), f"{expr!r}: got {got!r}")
        if isinstance(expected, float):
            self.assertAlmostEqual(got, expected, places=12, msg=expr)
        else:
            self.assertEqual(got, expected, msg=expr)

    def error(self, expr, pos):
        with self.assertRaises(CalcError, msg=repr(expr)) as ctx:
            evaluate(expr)
        self.assertEqual(ctx.exception.pos, pos, repr(expr))


class PrecedenceTests(SpecCase):
    def test_table(self):
        self.check("1+2*3", 7)
        self.check("2*3+4", 10)
        self.check("(1+2)*3", 9)
        self.check("10-4-3", 3)
        self.check("100/10/5", 2.0)
        self.check("2*3/4", 1.5)
        self.check("1+2^3*2", 17)
        self.check("2*3^2", 18)
        self.check("-2*3^2", -18)
        self.check("(1+2)*(3-1)", 6)


class PowerTests(SpecCase):
    def test_right_associative_chains(self):
        self.check("2^3^2", 512)
        self.check("2^2^3", 256)
        self.check("(2^3)^2", 64)

    def test_unary_and_power(self):
        self.check("-2^2", -4)
        self.check("(-2)^2", 4)
        self.check("-2^-2", -0.25)
        self.check("2*-2^2", -8)
        self.check("-(2^2)", -4)

    def test_exponent_sign_and_types(self):
        self.check("2^-1", 0.5)
        self.check("2^-2", 0.25)
        self.check("4^0.5", 2.0)
        self.check("2^10", 1024)
        self.check("2^0", 1)
        self.check("2.0^2", 4.0)


class UnaryTests(SpecCase):
    def test_chains(self):
        self.check("--2", 2)
        self.check("---2", -2)
        self.check("-+2", -2)
        self.check("+-2", -2)
        self.check("- -2", 2)
        self.check("+2", 2)
        self.check("-2.5", -2.5)

    def test_binary_and_unary_mix(self):
        self.check("2*-3", -6)
        self.check("2--3", 5)
        self.check("-(2+3)*2", -10)
        self.check("-(7-2)", -5)


class NumberFormTests(SpecCase):
    def test_scientific_forms(self):
        self.check("2.5E-2", 0.025)
        self.check("5e+2", 500.0)
        self.check("1E3", 1000.0)
        self.check(".5e1", 5.0)
        self.check("1e-3+1", 1.001)
        self.check("1e0", 1.0)
        self.check("2e-1*10", 2.0)
        self.check("1e3*2", 2000.0)

    def test_plain_forms(self):
        self.check("12", 12)
        self.check("3.25*4", 13.0)
        self.check(".5", 0.5)


class ResultTypeTests(SpecCase):
    def test_int_ops_stay_int(self):
        self.check("7-2", 5)
        self.check("2-2", 0)
        self.check("2*3", 6)
        self.check("3*(2-2)", 0)
        self.check("-(7-2)", -5)
        self.check("(2^3)", 8)

    def test_float_contamination(self):
        self.check("7-2.5", 4.5)
        self.check("2+0.5", 2.5)
        self.check("1.5+1.5", 3.0)
        self.check("1e3", 1000.0)

    def test_division_always_float(self):
        self.check("6/3", 2.0)
        self.check("7/2", 3.5)


class ErrorTests(SpecCase):
    def test_unexpected_end(self):
        self.error("2+", 2)
        self.error("(1", 2)
        self.error("2 + 3 ^", 7)
        self.error("", 0)
        self.error("   ", 3)

    def test_unexpected_token(self):
        self.error("1)", 1)
        self.error("2 2", 2)
        self.error("*2", 0)
        self.error("2 ** 3", 3)
        self.error("(1)(2)", 3)
        self.error("()", 1)
        self.error("2 * (3 + 4", 10)

    def test_unexpected_character(self):
        self.error("3 $ 4", 2)
        self.error("1e", 1)

    def test_division_by_zero_points_at_slash(self):
        self.error("2 / 0", 2)
        self.error("1/(2-2)", 1)
        self.error("4 / 2 / 0", 6)
        self.error("2^(1/0)", 4)


if __name__ == "__main__":
    unittest.main()
