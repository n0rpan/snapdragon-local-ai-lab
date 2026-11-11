"""Service entry point: wires the components together from one Config."""
from svc import cache, db, http, worker


def run(cfg):
    """Build every component from cfg and return the wiring."""
    name = cfg.require('service.name')
    log_level = cfg.get('log.level', 'INFO')
    region = cfg.get('service.region', 'eu')
    database = db.build_db(cfg)
    store = cache.build_cache(cfg)
    client = http.build_client(cfg)
    job = worker.build_worker(cfg, database, store)
    return dict(name=name, region=region, log_level=log_level,
                db=database, cache=store, http=client, worker=job)
