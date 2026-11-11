"""Small config accessors shared by the components."""


def opt(cfg, key, fallback):
    """Read an optional key, returning fallback when it is absent."""
    return cfg.get(key, fallback)


def need(cfg, key):
    """Read a required key. Raises KeyError when it is absent."""
    return cfg[key]
