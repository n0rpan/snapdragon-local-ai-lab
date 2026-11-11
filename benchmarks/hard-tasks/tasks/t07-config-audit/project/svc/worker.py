"""Background worker description."""


def build_worker(cfg, database, store):
    """Return the worker wired to the database and cache."""
    return dict(queue=cfg.get('worker.queue'), db=database, cache=store)


def dump_state(cfg):
    """Debug helper: return the path for a state dump. Not used by run()."""
    return cfg.get('worker.debug_dump')
