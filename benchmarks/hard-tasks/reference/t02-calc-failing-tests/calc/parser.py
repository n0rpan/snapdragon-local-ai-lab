"""Pratt (precedence-climbing) parser.

AST nodes are tuples:
    ("num", value, pos)
    ("neg", operand, pos) / ("pos", operand, pos)
    ("bin", op, left, right, pos)     pos is the operator's index
"""
from .errors import CalcError
from .tokenizer import tokenize

# operator -> (left binding power, right binding power); rbp < lbp means right-associative.
BINARY = {"+": (10, 11), "-": (10, 11), "*": (20, 21), "/": (20, 21), "^": (40, 39)}
# operand of unary +/-: looser than ^ (so -2^2 is -(2^2)), tighter than * and /.
UNARY_BP = 30


def parse(text):
    return _Parser(tokenize(text)).parse()


class _Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.i = 0

    def advance(self):
        tok = self.tokens[self.i]
        self.i += 1
        return tok

    def parse(self):
        node = self.expr(0)
        kind, value, pos = self.tokens[self.i]
        if kind != "END":
            raise CalcError(f"unexpected token {value!r}", pos)
        return node

    def expr(self, min_bp):
        left = self.prefix()
        while True:
            kind, op, pos = self.tokens[self.i]
            if kind != "OP" or BINARY[op][0] < min_bp:
                return left
            self.advance()
            right = self.expr(BINARY[op][1])
            left = ("bin", op, left, right, pos)

    def prefix(self):
        kind, value, pos = self.advance()
        if kind == "NUM":
            return ("num", value, pos)
        if kind == "OP" and value in "+-":
            operand = self.expr(UNARY_BP)
            return ("neg" if value == "-" else "pos", operand, pos)
        if kind == "LPAREN":
            node = self.expr(0)
            kind2, value2, pos2 = self.advance()
            if kind2 == "END":
                raise CalcError("unexpected end of input", pos2)
            if kind2 != "RPAREN":
                raise CalcError(f"expected ')' before {value2!r}", pos2)
            return node
        if kind == "END":
            raise CalcError("unexpected end of input", pos)
        raise CalcError(f"unexpected token {value!r}", pos)
