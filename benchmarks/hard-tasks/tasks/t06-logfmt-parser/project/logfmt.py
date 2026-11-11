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


class ParseError(Exception):
    def __init__(self, message, pos):
        super().__init__(f"{message} at {pos}")
        self.pos = pos


def parse(line):
    out = {}
    for part in line.split(' '):
        if not part:
            continue
        key, value = part.split('=', 1)
        out[key] = value
    return out


def format(record):
    return ' '.join(f'{k}={v}' for k, v in record.items())
