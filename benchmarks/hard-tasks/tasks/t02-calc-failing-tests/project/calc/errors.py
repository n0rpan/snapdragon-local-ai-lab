"""Error type for invalid expressions."""


class CalcError(Exception):
    """Invalid expression. `pos` is the 0-based index of the offending character or token,
    or len(text) when the input ends too early."""

    def __init__(self, message, pos):
        super().__init__(message)
        self.message = message
        self.pos = pos
