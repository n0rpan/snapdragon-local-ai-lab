"""Exceptions raised by the planner."""


class CycleError(Exception):
    """A dependency cycle was found. `.cycle` is the path, e.g. ['a', 'b', 'a']."""

    def __init__(self, cycle):
        self.cycle = list(cycle)
        super().__init__(' -> '.join(self.cycle))


class MissingSourceError(Exception):
    """A source file has no mtime. `.name` is the missing name."""

    def __init__(self, name):
        self.name = name
        super().__init__(name)
