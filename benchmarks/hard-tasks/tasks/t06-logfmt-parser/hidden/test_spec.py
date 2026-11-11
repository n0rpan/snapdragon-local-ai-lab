import os
import sys
import unittest

sys.path.insert(0, os.environ["PROJECT"])
from logfmt import ParseError, format, parse  # noqa: E402


class ParseCases(unittest.TestCase):
    def check(self, line, expected):
        self.assertEqual(parse(line), expected, repr(line))

    def test_empty_and_spaces(self):
        self.check('', {})
        self.check('   ', {})
        self.check('  a=1   b=2  ', {'a': '1', 'b': '2'})

    def test_unquoted_and_equals(self):
        self.check('a=b=c', {'a': 'b=c'})
        self.check('msg=héllo', {'msg': 'héllo'})
        self.check('x.y-z_9=1', {'x.y-z_9': '1'})
        self.check(r'p=C:\dir', {'p': 'C:\\dir'})

    def test_empty_values(self):
        self.check('e=', {'e': ''})
        self.check('e=""', {'e': ''})
        self.check('e= f=1', {'e': '', 'f': '1'})

    def test_quoted(self):
        self.check('msg="héllo wörld"', {'msg': 'héllo wörld'})
        self.check('msg="a=b"', {'msg': 'a=b'})
        self.check('k="="', {'k': '='})
        self.check('a="1 2  3"', {'a': '1 2  3'})
        self.check("a=\"it's\"", {'a': "it's"})
        self.check('a="x" b', {'a': 'x', 'b': True})

    def test_escapes(self):
        self.check(r'q="say \"hi\""', {'q': 'say "hi"'})
        self.check(r'p="C:\\dir"', {'p': 'C:\\dir'})
        self.check(r'q="\\"', {'q': '\\'})
        self.check(r'n="a\nb\tc"', {'n': 'a\nb\tc'})

    def test_bare_keys(self):
        self.check('bare', {'bare': True})
        self.check('x  y=2', {'x': True, 'y': '2'})
        self.check('a=1 bare b=2', {'a': '1', 'bare': True, 'b': '2'})

    def test_duplicates_last_wins(self):
        self.check('a=1 a=2 b=3 a=4', {'a': '4', 'b': '3'})


class ParseErrors(unittest.TestCase):
    def test_error_type(self):
        self.assertTrue(issubclass(ParseError, Exception))

    def assertPos(self, line, pos):
        with self.assertRaises(ParseError, msg=repr(line)) as cm:
            parse(line)
        self.assertEqual(cm.exception.pos, pos, repr(line))

    def test_key_expected(self):
        self.assertPos('=x', 0)
        self.assertPos('a=1 =x', 4)
        self.assertPos('"a"=1', 0)
        self.assertPos('clé=1', 2)

    def test_after_key(self):
        self.assertPos('a!=1', 1)
        self.assertPos('a b!', 3)
        self.assertPos('a"', 1)

    def test_unquoted_value(self):
        self.assertPos('a=b"c', 3)
        self.assertPos('a=1"', 3)

    def test_quoted_value(self):
        self.assertPos('a="abc', 2)
        self.assertPos('a="x"y', 5)
        self.assertPos('a="x"b=1', 5)
        self.assertPos('a=""x', 4)
        self.assertPos(r'a="x\q"', 4)
        self.assertPos('a="x\\', 4)
        self.assertPos(r'a="\z', 3)
        self.assertPos(r'a="x\"', 2)


class FormatCases(unittest.TestCase):
    def test_plain(self):
        self.assertEqual(format({}), '')
        self.assertEqual(format({'a': '1'}), 'a=1')
        self.assertEqual(format({'a': '1', 'b': 'two'}), 'a=1 b=two')
        self.assertEqual(format({'b': '2', 'a': '1'}), 'b=2 a=1')
        self.assertEqual(format({'u': 'héllo'}), 'u=héllo')
        self.assertEqual(format({'x': '-.'}), 'x=-.')

    def test_bare_true(self):
        self.assertEqual(format({'debug': True, 'a': '1'}), 'debug a=1')

    def test_quoting(self):
        self.assertEqual(format({'msg': 'hello world'}), 'msg="hello world"')
        self.assertEqual(format({'e': ''}), 'e=""')
        self.assertEqual(format({'eq': 'a=b'}), 'eq="a=b"')
        self.assertEqual(format({'q': 'say "hi"'}), r'q="say \"hi\""')
        self.assertEqual(format({'p': 'C:\\dir'}), r'p="C:\\dir"')
        self.assertEqual(format({'x': '\\'}), r'x="\\"')

    def test_control_escapes(self):
        self.assertEqual(format({'t': 'a\tb'}), r't="a\tb"')
        self.assertEqual(format({'n': 'a\nb'}), r'n="a\nb"')

    def test_type_error(self):
        for value in (1, False, None, 1.5, ['a']):
            with self.assertRaises(TypeError, msg=repr(value)):
                format({'a': value})

    def test_value_error(self):
        for key in ('a b', '', 'a/b', 'é'):
            with self.assertRaises(ValueError, msg=repr(key)):
                format({key: '1'})
        for value in ('a\rb', '\x00', '\x1f', 'x\x01'):
            with self.assertRaises(ValueError, msg=repr(value)):
                format({'k': value})


class RoundTripCases(unittest.TestCase):
    def test_examples(self):
        for rec in (
            {'a': True},
            {'msg': ''},
            {'k': 'é "x" \\ = \n\t'},
            {'a': '1', 'b': True, 'c': 'x y'},
            {'p': 'C:\\dir\\', 'q': '\\"'},
        ):
            self.assertEqual(parse(format(rec)), rec, repr(rec))


if __name__ == '__main__':
    unittest.main()
