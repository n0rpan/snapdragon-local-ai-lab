r"""logfmt: single-line key=value records.

Spec (normative):
- A line is a sequence of pairs separated by one or more spaces (' ' only).
  Leading and trailing spaces are ignored. An empty or all-space line -> {}.
- key: one or more of [A-Za-z0-9_.-]. Where a key is expected, any other
  character -> ParseError at that character's index.
- After a key, the next character must be '=', a space, or end of line;
  anything else -> ParseError at that index.
- Forms: key=value, key="quoted", key= (value ''), bare key (value True).
- Unquoted value: a run of characters that are not ' ' and not '"'.
  A '"' inside an unquoted value -> ParseError at its index. A backslash
  is literal outside quotes.
- Quoted value: escapes \" \\ \n \t. Any other backslash, including a
  backslash at end of line -> ParseError at the backslash index.
  A missing closing quote -> ParseError with pos = index of the opening
  quote. After the closing quote the next character must be a space or
  end of line, else ParseError at that index.
- Errors: the first problem found scanning left to right is raised.
- Duplicate keys: the last one wins.
- format(record): pairs in insertion order, joined by single spaces.
  For each pair check the key first (must match the key charset, else
  ValueError), then the value. True -> bare key. Any other non-str value
  -> TypeError. A str value is quoted if it is empty or contains ' ', '=',
  '"', a backslash, or any char < 0x20. Quoted values escape backslash as
  \\ , double quote as \" , newline as \n and tab as \t; any other char
  < 0x20 -> ValueError. Values that need no quoting are written bare.
- Round trip: parse(format(r)) == r for every valid record.
"""

import string

KEY_CHARS = frozenset(string.ascii_letters + string.digits + '_.-')
ESCAPES = {'"': '"', '\\': '\\', 'n': '\n', 't': '\t'}


class ParseError(Exception):
    def __init__(self, message, pos):
        super().__init__(f"{message} at {pos}")
        self.pos = pos


def parse(line):
    out = {}
    i, n = 0, len(line)
    while True:
        while i < n and line[i] == ' ':
            i += 1
        if i >= n:
            return out
        start = i
        while i < n and line[i] in KEY_CHARS:
            i += 1
        if i == start:
            raise ParseError('expected key', i)
        key = line[start:i]
        if i < n and line[i] == '=':
            i += 1
            if i < n and line[i] == '"':
                open_pos = i
                i += 1
                chars = []
                while True:
                    if i >= n:
                        raise ParseError('missing closing quote', open_pos)
                    c = line[i]
                    if c == '"':
                        i += 1
                        break
                    if c == '\\':
                        if i + 1 >= n or line[i + 1] not in ESCAPES:
                            raise ParseError('bad escape', i)
                        chars.append(ESCAPES[line[i + 1]])
                        i += 2
                    else:
                        chars.append(c)
                        i += 1
                value = ''.join(chars)
                if i < n and line[i] != ' ':
                    raise ParseError('expected space after quoted value', i)
            else:
                vstart = i
                while i < n and line[i] != ' ' and line[i] != '"':
                    i += 1
                if i < n and line[i] == '"':
                    raise ParseError('quote in unquoted value', i)
                value = line[vstart:i]
        else:
            if i < n and line[i] != ' ':
                raise ParseError('unexpected character after key', i)
            value = True
        out[key] = value


def _escape(value):
    out = []
    for c in value:
        if c == '\\':
            out.append('\\\\')
        elif c == '"':
            out.append('\\"')
        elif c == '\n':
            out.append('\\n')
        elif c == '\t':
            out.append('\\t')
        elif ord(c) < 0x20:
            raise ValueError(f'control character {ord(c):#x} cannot be formatted')
        else:
            out.append(c)
    return ''.join(out)


def format(record):
    parts = []
    for key, value in record.items():
        if not (isinstance(key, str) and key and all(c in KEY_CHARS for c in key)):
            raise ValueError(f'invalid key: {key!r}')
        if value is True:
            parts.append(key)
            continue
        if not isinstance(value, str):
            raise TypeError(f'unsupported value type for {key!r}: {type(value).__name__}')
        if value == '' or any(c in ' ="\\' or ord(c) < 0x20 for c in value):
            parts.append(f'{key}="{_escape(value)}"')
        else:
            parts.append(f'{key}={value}')
    return ' '.join(parts)
