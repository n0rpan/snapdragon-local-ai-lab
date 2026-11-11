"""Parse a tiny Makefile-like format: one 'target: dep1 dep2' rule per line, '#' starts a comment."""


def parse_rules(text):
    rules = {}
    for line in text.splitlines():
        line = line.split('#', 1)[0].strip()
        if not line:
            continue
        target, _, deps = line.partition(':')
        rules[target.strip()] = deps.split()
    return rules
