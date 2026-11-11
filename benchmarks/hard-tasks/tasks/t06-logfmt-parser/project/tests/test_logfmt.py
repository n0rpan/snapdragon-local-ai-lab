import unittest

import logfmt
from logfmt import ParseError, format, parse


class SimpleTests(unittest.TestCase):
    def test_pairs(self):
        self.assertEqual(parse('a=1 b=two'), {'a': '1', 'b': 'two'})

    def test_empty(self):
        self.assertEqual(parse(''), {})

    def test_format_simple(self):
        self.assertEqual(format({'a': '1'}), 'a=1')

    def test_quoted_with_spaces(self):
        self.assertEqual(parse('msg="hello world"'), {'msg': 'hello world'})

    def test_escaped_quote(self):
        self.assertEqual(parse(r'q="say \"hi\""'), {'q': 'say "hi"'})

    def test_bare_key(self):
        self.assertEqual(parse('debug'), {'debug': True})

    def test_format_quoting(self):
        self.assertEqual(format({'msg': 'hello world'}), 'msg="hello world"')


if __name__ == '__main__':
    unittest.main()
