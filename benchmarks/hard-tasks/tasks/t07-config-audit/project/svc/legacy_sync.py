"""Legacy file sync job. Not wired into run()."""


def sync_once(cfg):
    """Run one legacy sync pass and return its settings."""
    source = cfg['legacy.sync_dir']
    endpoint = cfg.require('legacy.endpoint')
    return dict(source=source, endpoint=endpoint)
