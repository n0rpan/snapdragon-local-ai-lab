"""Configuration wrapper over a dict of dotted keys."""

_MISSING = object()


class Config:
    """Read-only view over a dict of settings."""

    def __init__(self, data):
        self._data = dict(data)

    def get(self, key, default=_MISSING):
        """Return the value for key, or default. Raises KeyError if absent and no default is given."""
        if key in self._data:
            return self._data[key]
        if default is _MISSING:
            raise KeyError(key)
        return default

    def __getitem__(self, key):
        """Return the value for key. Raises KeyError if absent."""
        if key not in self._data:
            raise KeyError(key)
        return self._data[key]

    def require(self, key):
        """Return the value for key. Raises KeyError if absent."""
        return self[key]

    def has(self, key):
        """Return True if key is present."""
        return key in self._data
