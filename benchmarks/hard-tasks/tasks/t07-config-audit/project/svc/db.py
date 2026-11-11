"""Database pool description."""


def build_db(cfg):
    """Return the database pool settings for both hosts."""
    hosts = {}
    for name in ('primary', 'replica'):
        hosts[name] = cfg[f'db.{name}.host']
    return dict(
        hosts=hosts,
        name=cfg.get('db.name'),
        user=cfg.get('db.user'),
        port=cfg.get('db.port', 5432),
    )
