"""HTTP client description. Builds the object only; performs no I/O."""
from svc.util import need, opt


class Client:
    """Client settings. Holds configuration only."""

    def __init__(self, base_url, api_key, region, timeout):
        self.base_url = base_url
        self.api_key = api_key
        self.region = region
        self.timeout = timeout


def build_client(cfg):
    """Build a Client from the http.* and service.* settings."""
    return Client(
        base_url=cfg.get('http.base_url'),
        api_key=need(cfg, 'http.api_key'),
        region=cfg['service.region'],
        timeout=opt(cfg, 'http.timeout', 10),
    )
