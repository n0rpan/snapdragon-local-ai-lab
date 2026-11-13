# Panel and chooser data (Rust era)

2026-11-10. This change covers data and API only. The panel's look is unchanged, and a later design pass will build on the data described here. Alias ids are stable, so OpenCode and dsh configs keep working.

## What changed

- **Ornith is production, not beta.** It is the default. Display names no longer say "(beta)". The stored `release_channel` value is qualification data, and both `beta` and `production` are accepted with identical proof checks. Ornith's request defaults stay: temperature 0, `cache_prompt`, no thinking, and 2048 max tokens when the client sets none. The production qualification (classic and Rust, 13/13 OpenCode) ran with them. Error texts no longer mention a "Rust engine beta".
- **Flash-Next is no longer planning-only.** The panel's OpenCode button opens the ordinary build agent with the selected model, Flash included. Flash's launch overlay pins temperature 0 and no thinking, the only values its backend accepts. The read-only planner stays available as `local-ai plan`. The admission rules are unchanged: admitted budget plus 4 GiB of free RAM, and at least 30% battery to start.

## Controller endpoints (127.0.0.1:8092, same-origin rules as before)

`GET /_local/models` returns the chooser rows. It never loads a model. Prefix warmth is read only from entry identity headers or metadata JSON.

```json
{"object":"list","default_model":"ornith-35b-a3b-local","selected_model":"ornith-35b-a3b-local",
 "data":[{"id":"ornith-35b-a3b-local","key":"ornith","name":"Ornith 1.5 35B A3B",
          "engine":"rust|llama|stream","engine_status":"production|classic|fallback",
          "memory_gib":19.2,"battery_start_percent":null,"prefix":"warm|cold|off",
          "selected":true,"default":true,"state":"ready|loading|unloaded"}]}
```

- Rows appear in chooser order: Ornith, Qwen 3.6 MoE, Flash-Next, Dense. Only eligible profiles are listed.
- `engine_status: fallback` means a Rust override is configured but the classic engine resolved, for example on a stock boot or with `LOCAL_AI_ENABLE_OXE` unset.
- `memory_gib` is the free RAM the controller requires before it starts that model. For classic and Rust models that is the model size plus 3 GiB. For Flash it is the admitted budget plus 4 GiB.
- `prefix` is `warm` when a persistent entry exists for this exact model SHA and engine SHA. The Rust store is `--prefix-cache` (default `~/.cache/local-ai/snapdragon-engine/prefix`); the llama.cpp store is the profile's `prefix_cache.directory`.
- `/v1/models` names come from the same catalog (`model_catalog.MODELS`), so the names "(beta)" and "Flash-Next planning" are gone.

`GET /_local/storage` returns the engine caches with their sizes, without paths:
`{"data":[{"id","kind","name","bytes","model","state","deletable"}]}`.

- `kind` is one of `rust-weights`, `rust-prefix`, `flash-sidecar`, `llama-prefix` or `llama-repack`.
- `state` is one of:
  - `referenced`: some configured stock, MEMLAT or Rust variant uses the cache.
  - `in-use`: a process has it open or mapped.
  - `protected`: the Flash sidecar, which is always kept.
  - `unreferenced`: nothing uses it.

`POST /_local/storage/delete {"id": "..."}` deletes only an `unreferenced` entry. The controller re-plans the inventory at call time, requires no running request, and holds the inference lease while it deletes. If a model is loaded, the controller already owns the lease. Possible responses: 409 when the entry is referenced, in use or busy; 404 for an unknown id; 200 `{"deleted","kind","freed_bytes"}` on success. Weekly maintenance (`tools/storage_cleanup.py`) is separate and unchanged.

## CLI and panel parsing

- `local-ai status --json` adds `models`: the rows above, keyed by `ornith`, `moe` or `flash` (the dense 27B was retired in v0.1.2), with enums and numbers only. Server-supplied names are dropped.
- `local-ai storage --json` prints `{storage_available, entries}`.
- `local-ai storage-delete --id ID` deletes one entry through the controller.
- The panel parses these in `Model.js` (it replaced `Status.js` in the redesign, see `UI-PANEL-CONTRACT.md`). Allowlists throughout; all display text comes from i18n.

Code: `runtime/model_catalog.py`, `runtime/storage_view.py`, `runtime/controller.py`, `runtime/local-ai`, `runtime/launch_local_plan.py`, `omarchy_plugin/Model.js`. Tests: `tests/test_ui_data.py`, `tests/test_panel_model.py`.
