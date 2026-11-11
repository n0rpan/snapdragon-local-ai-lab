# Protected normal13 through a staged controller

Start and qualify the staged controller separately, then run:

```sh
python benchmarks/classic/controller_normal13.py \
  --profile /absolute/effective-profile.json --output /absolute/fresh-client \
  --controller http://127.0.0.1:18092
```

Only literal HTTP IPv4 loopback with an explicit port is accepted. Alternatively,
`--socket /short/private/controller.sock` uses an already-running bridge's
owner-only Unix socket. The default path starts its own bounded bridge in a short
private temporary directory, avoiding Unix socket pathname limits.

This driver does **not** start a native server or acquire the canonical inference
lease. Do not wrap it in a runner that takes that lease: the controller owns it.
The orchestrator owns controller unload/switch/idle/cache/RAM checks and cleanup.

The client keeps the original bwrap `--unshare-all` isolation, protected checker,
OpenCode executable pin and unchanged `inside.run_agent()` task/reader limits.
Only the private Unix socket is mounted into its namespace. The model/engine
binds are removed. A small wrapper starts the inside loopback bridge, checks the
nonwaking `/v1/models` catalog, then invokes the original task. The supplied
`model_alias` selects the actual controller profile in the private OpenCode config.
No shared network fallback exists.

The profile uses the ordinary resolved classic fields required by normal13's
private configuration preparation. Its model/engine SHA values are recorded as
**declared**, not newly verified: this client never opens model or engine files.
The existing controller's startup/identity policy supplies those checks.

One final JSON summary and `summary.json` contain actual independent13 counts,
terminal completion/tool/test evidence, task timing, protected checker hash,
source/config checks and **client/bridge** cleanup. They make no claim that the
controller's native process was unloaded, its lease released or RAM recovered.
Those remain separate actual orchestrator observations.

Model-free tests:

```sh
python -m unittest discover -s benchmarks/classic -p test_controller_normal13.py
```

## Captured synthetic requests

Pass `--capture-requests` to record the synthetic client's outbound HTTP in the
output's owner-only `requests/` directory. These files contain the system prompt,
tool schemas and fixture conversation and must stay private. Summary rows record
body hashes, sizes, tool names and the observed system-prompt date.

Each connection is capped at 4 MiB and at most 64 connections are recorded.
An incomplete or unsupported capture fails the run. Date pinning is tracked
separately as F104.
