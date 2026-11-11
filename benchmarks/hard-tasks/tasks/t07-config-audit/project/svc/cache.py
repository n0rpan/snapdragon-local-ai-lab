"""Cache description."""


def build_cache(cfg):
    """Return the cache settings. A TTL is needed only when caching is enabled."""
    ttl = None
    if cfg.has('cache.enabled') and cfg['cache.enabled']:
        ttl = cfg.require('cache.ttl')
    return dict(ttl=ttl, size=cfg.get('cache.size', 128))
