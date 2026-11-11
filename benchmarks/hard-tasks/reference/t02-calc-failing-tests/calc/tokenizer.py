"""Tokenizer: expression text -> list of (kind, value, pos) tuples.

Kinds: NUM (value is int or float), OP (+ - * / ^), LPAREN, RPAREN,
END (pos = len(text)).
"""
from .errors import CalcError

DIGITS = "0123456789"


def tokenize(text):
    tokens = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c.isspace():
            i += 1
        elif c in "+-*/^":
            tokens.append(("OP", c, i))
            i += 1
        elif c in "()":
            tokens.append(("LPAREN" if c == "(" else "RPAREN", c, i))
            i += 1
        elif c in DIGITS or (c == "." and i + 1 < n and text[i + 1] in DIGITS):
            j = _number_end(text, i)
            literal = text[i:j]
            value = float(literal) if any(ch in literal for ch in ".eE") else int(literal)
            tokens.append(("NUM", value, i))
            i = j
        else:
            raise CalcError(f"unexpected character {c!r}", i)
    tokens.append(("END", None, n))
    return tokens


def _number_end(text, i):
    """Return the index just past the number literal that starts at i."""
    n = len(text)
    j = i
    while j < n and text[j] in DIGITS:
        j += 1
    if j < n and text[j] == ".":
        j += 1
        while j < n and text[j] in DIGITS:
            j += 1
    if j < n and text[j] in "eE":
        k = j + 1
        if k < n and text[k] in "+-":
            k += 1
        if k < n and text[k] in DIGITS:
            while k < n and text[k] in DIGITS:
                k += 1
            j = k
    return j
