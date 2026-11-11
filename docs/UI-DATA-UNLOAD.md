# Panel data: "Unload now" per loaded model

Addendum to `docs/UI-DATA.md` (which lives on branch `claude/ui-rust-era`; link this page from its
"Controller endpoints" section when the branches meet). Source only, not deployed. Controller at
127.0.0.1:8092, same-origin rules as the other `/_local/*` routes.

## Why

The Rust Flash engine stays loaded for 600 s after its last request (other models: 120 s), holding about
27 GiB at a 20 GiB expert cache. The user must be able to free that memory at once.

## Read: `GET /_local/status` (new fields; polling never loads or wakes a model)

| field | type | meaning |
|---|---|---|
| `loaded_model` | string or null | alias of the model whose engine process is running; null when nothing is loaded |
| `resident_bytes` | integer or null | resident memory (VmRSS) of that engine process; null when nothing is loaded or it cannot be read |
| `idle_unload_seconds` | number | idle timeout of the loaded (else the selected) model: 600 for Rust Flash, else 120 |
| `idle_unload_remaining_seconds` | number or null | seconds until the automatic unload; null while a request is running, while loading, or when nothing is loaded |
| `unload.available` | boolean | true when a model is loaded and no request is running: the button is enabled |
| `unload.method`, `unload.path` | string | `POST`, `/_local/unload` |
| `state` | string | unchanged: `ready`, `loading`, `unloaded` |

At most one language model is loaded at a time, so "per loaded model" is one row: show `loaded_model`, its
`resident_bytes` (GiB, one decimal) and a countdown from `idle_unload_remaining_seconds` (re-read on each poll;
do not run a client-side clock for longer than one poll interval). When the value is null and `state` is
`ready`, show "in use".

## Act: `POST /_local/unload`

- Body: empty, or `{"model": "<alias>"}` to unload only if that alias is the loaded one (use this from a
  per-model button so a stale page cannot unload a different model).
- `200 {"state": "unloaded"}`: the engine process was stopped (SIGTERM, then SIGKILL after 10 s) and the
  inference lease released. Rust engines write their warm-up file on SIGTERM, so the next start is fast.
- `409 {"error": "A request is running."}`: try again when it finishes; nothing was stopped.
- `409 {"error": "That model is not loaded."}`: the alias in the body is not the loaded model.

Command line: `local-ai unload` (existing; same route, no alias).

## Not provided

No per-model memory when nothing is loaded (use the profile's admission numbers for an estimate), and no
"keep loaded" switch: the idle timeout is per profile (`memlat_overrides.oxe_flash.idle_unload_seconds`, 60-3600).
