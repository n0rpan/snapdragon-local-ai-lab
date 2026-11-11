# svc

Small example service. The entry point is `svc.main.run(cfg)`, which takes a
`svc.config.Config` (a wrapper over a flat dict with dotted keys such as
`db.primary.host`) and builds the database pool, cache, HTTP client and worker.

Modules:

- `config.py`: the Config wrapper.
- `util.py`: small config accessors shared by the components.
- `db.py`, `cache.py`, `http.py`, `worker.py`: component factories.
- `main.py`: wires the components together.
