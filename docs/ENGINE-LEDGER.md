# Native request ledger

The controller can collect the Rust server's numeric request ledger in a
bounded, private log. This is source-only until the staged controller and engine
pass the consolidated qualification batch; no installed profile enables it yet.

Enable it only in an upgraded OXE profile whose pinned binary supports
`--ledger`:

```json
"environment": {
  "LOCAL_AI_ENGINE_LEDGER": "1"
}
```

Keep any existing environment entries. This is a **per-profile** setting;
setting the variable only on the controller process does not enable it. Classic
profiles and Flash on BigMoe are excluded; the Rust Flash package
(`tools/stage_flash_oxe_production.py`) resolves to an OXE profile and sets the
variable in its own environment, so its ledger is captured the same way. A profile with an explicit `--ledger` argument
keeps that argument and does not get a second controller-managed ledger.

At load time, the controller creates a pipe and passes its write descriptor to
the native child. `--ledger /proc/self/fd/N` lets the engine use its existing
ledger implementation; the controller closes its own write end after launch.
The reader drains asynchronously into:

```text
~/.local/state/local-ai/engine-ledger.jsonl
```

The directory is mode 0700 and log files are 0600, with the same ownership and
path checks as the engine diagnostic log. Rotation keeps at most three files
of 1 MiB each. Rotation belongs to the controller, so a long-lived engine cannot
continue writing into a renamed, unbounded file.

Only the known numeric counters and cache/stop enums are retained. Model alias
and pinned engine SHA are attached to each row. Prompt text, tool arguments,
generated text and unknown fields are discarded. Malformed records are counted
and discarded. The current record schema covers prompt/cache tokens, prompt and
generation time, drafts accepted, parks/unparks and checkpoint counts.

`/_local/status` exposes `engine_ledger` with active state, path, record count,
discard count and error type. Polling never creates the pipe, starts its reader
or loads a model. Cleanup closes both pipe ends after a failed launch, and joins
the reader before releasing the inference lease on unload.

If the private log cannot be prepared, the controller logs `ledger_disabled`
and starts the model without this optional capture. A disk error during capture
does not block inference; the reader continues draining and the error appears
in ledger status.

Model-free checks use a small Python child opening the inherited descriptor
with the same OS flags as Rust's create/append operation. They cover delivery,
rotation, private permissions, field filtering, failed-launch cleanup and
controller argument/descriptor wiring. A real-engine check belongs in the
03:00 batch on the exact frozen engine and staged controller.
