Bug report for project/logfmt.py:

Users report that logfmt.parse breaks on quoted values with spaces and escaped quotes, and that format() produces lines that can't be parsed back.

Rewrite or fix project/logfmt.py so that it implements the spec in its module docstring exactly, including the ParseError positions. Do not change the public names: parse(line), format(record), ParseError with attribute .pos.

Run the tests with:
cd project && python3 -m unittest discover -s tests -t .
