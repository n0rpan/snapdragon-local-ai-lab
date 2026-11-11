"""Evaluates the AST produced by calc.parser."""
from .errors import CalcError
from .parser import parse


def evaluate(text):
    """Return the value of expression `text` as int or float. Raises CalcError."""
    return _eval(parse(text))


def _eval(node):
    kind = node[0]
    if kind == "num":
        return node[1]
    if kind == "neg":
        return -_eval(node[1])
    if kind == "pos":
        return _eval(node[1])
    _, op, left, right, pos = node
    a, b = _eval(left), _eval(right)
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    if op == "/":
        if b == 0:
            raise CalcError("division by zero", pos)
        return a / b
    # "^": int ** non-negative int stays int; a negative int exponent gives float.
    return a ** b
