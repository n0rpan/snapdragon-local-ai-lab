# Flash on the Rust server (opt-in switch)

**Installed 2026-11-11:** package2 `8e5240a` / `a6912dd3`, fixed14 GiB, order0, long-prompt guard on, LA/fuse off. 16K context, plain expert sidecar, 29 GiB scope, zero model swap and 1 GiB host floor. [Current qualification and rollback](FLASH-PACKAGE2.md).

The USB-C power fix is retained in the legacy controller plus two cache-size allowlists; both staged and live controller regressions passed. The UI bundle is not deployed. The original staging recipe below predates this package; its20/18 GiB examples are not the installed cache size.

## What the switch does

The alias `qwen3.8-flash-next-local` runs on the BigMoe CPU-stream backend (`bigmoe-cpu-stream`) by default. With the Flash switch on, the same alias resolves to the Rust server (`oxe-server --engine flash`) bound to loopback, using the same four pinned shards and a 16384-token context. Nothing else changes: other aliases, the MoE Rust aliases, the panel and the OpenCode configuration are untouched.

The resolution happens in `profile_policy.resolve`. A Flash profile accepts only one MEMLAT override key, `oxe_flash`. Without MEMLAT, or with the switch off, the profile resolves to the stock profile exactly (a detached copy).

## Two independent flags

| Flag | Effect |
| --- | --- |
| `LOCAL_AI_ENABLE_OXE=1` | MoE Rust aliases (`ornith-35b-a3b-local`, `qwen3.6-35b-a3b-local`). Does not affect Flash. |
| `LOCAL_AI_ENABLE_OXE_FLASH=1` | Flash alias on the Rust server. Does not affect the MoE aliases. |

## Admission

- Cache: 20 GiB (`--flash-cache-gib 20`); 18 GiB is allowed by the policy.
- Measured peak: the value from the qualification ticket, passed to the stage tool as `--measured-peak-bytes`. It must be at least the cache size plus 4522949120 bytes (the dense lower bound) and at most the budget.
- Admitted budget: the 29 GiB scope with the Flash memory policy, 28 GiB without it.
- Host floor: 1.0 GiB with the installed Flash memory policy (`flash_memory`), 1.5 GiB without it.
- Required available memory at start is the larger of measured peak + host floor and cache GiB + 4.87 GiB + host floor (the measured rule: cache + 5.87 GiB at the 1 GiB floor). With a 1 GiB floor and no session reserve that is 25.87 GiB at a 20 GiB cache and 23.87 GiB at 18 GiB. A configured `session_reserve_bytes` (0 to 4 GiB) is added on top for session checkpoints. Start is refused below that. The package reserves 512 MiB (4 checkpoints), so it needs 26.37 GiB for the 20 GiB cache and 24.37 GiB for the 18 GiB fallback.
- The controller's own cgroup `memory.max` must also be at least the measured peak when it is a number.
- While running, the classic memory guard (1.5 GiB available floor, PSI) watches the child, as it does for the MoE Rust aliases.

## Battery

- Start: refused below 30% battery.
- Run: the Flash child is stopped (SIGTERM to its process group, then SIGKILL after 10 s) at 20% battery or below. Unreadable reserve telemetry is allowed only while an external Mains/USB/UCSI supply reports online. Degraded/recovery transitions and a ten-minute outage warning are logged. A valid low battery reading always wins over external power.
- External power is not required.

## Loading and idle

- Loading deadline: at least 300 s (the server prefaults the cache).
- Idle unload: 600 s without requests (`idle_unload_seconds`, allowed 60 to 3600).

## Enable

Stage without touching live files. `--server-binary` is the frozen build output; the staged `runtime.json` names the
content-addressed install path `ENGINE_RELEASES/oxe-flash-<sha12>/oxe-server` and the binary is copied into the stage
as `engine/oxe-server`.

```
python3 -I tools/stage_flash_oxe_production.py STAGE --runtime RUNTIME_JSON --runtime-sha256 SHA \
  --unit CHAT_UNIT --unit-sha256 SHA --server-binary FROZEN_OXE_SERVER --engine-releases-root ENGINE_RELEASES \
  --releases-root RUNTIME_RELEASES --sidecar EXPERT_SIDECAR --measured-peak-bytes PEAK_FROM_TICKET \
  --fallback-measured-peak-bytes PEAK_AT_18 [--cache-gib 20] [--candidate-sha256 SHA]
```

The tool refuses unless the binary hash equals `QUALIFIED_FLASH_SERVER_SHA` (pinned in the tool after the
qualification ticket passes) or the `--candidate-sha256` given on the command line. A candidate stage is marked
`qualified: false` in `manifest.json` and in the override's `validation`, and its `install.py` exits 6 unless run with
`--allow-unqualified-candidate`.

The stage holds `runtime.json`, `local-ai-chat.service` (a content-addressed release directory plus
`Environment="LOCAL_AI_ENABLE_OXE_FLASH=1"`; an existing `LOCAL_AI_ENABLE_OXE=1` line is kept), `runtime-source/`,
`engine/oxe-server`, `manifest.json`, `install.py` and `rollback/` (exact preimages and `rollback.py`).

Install is one command, run by hand under an idle lease and only with owner approval. It checks that the live files are
still the staged preimages (exit 3 otherwise, nothing changed), copies the engine and the runtime release if absent,
replaces the two files, reloads systemd and restarts the chat service. It loads no model:

```
python3 STAGE/install.py
```

Server flags of the package: 8 decode threads, 11 prefill threads, IO thread on CPU 11, `--ctx-checkpoints 4`,
`--session-ram-bytes 0` (no parked sessions in RAM: at a 20 GiB cache there is no room for them inside the admission
rule), message-boundary prefill schedule (the engine's default for Flash),
numeric ledger through the controller (`LOCAL_AI_ENGINE_LEDGER=1`). Cancel on disconnect is the controller's upstream
close; the engine checks it after every decoded token and between prefill calls (a prefill call is one message or
4096 tokens, so a cancel during a long first prefill takes effect at the next message start).

**Sidecar and IO config.** The engine is started with `--flash-sidecar <the staged --sidecar path>` (the plain copy
`/var/lib/local-ai-models/flash/experts-r4.bin` once O-21's partition is in use; its `.json` manifest must sit next to
it). The IO thread parks by default, and the idle-QoS hook covers its idle time (see "Idle QoS on the IO core");
`--io-mode spin` is the fallback instead, which sets `OXF_IO=spins=1000000` (F-IO: one IO thread that does not park
inside a decode). An engine built without F-IO's IO code ignores the variable.

**No warm-up (2026-11-10).** The package starts without `--flash-hotset`, `--prefix-cache` and `--session-cache`:
the engine's hot-set restore (`Stream::warm`) can issue a second read into a slot whose prefetch is still pending, and
a decode that starts meanwhile can compute on a buffer being rewritten. Those flags and RAM parking are the only
callers of the restore, so with them off no prefetch-class read is ever issued. The stage tool refuses an argv that
contains them. Cost until the fix (F-IO) is qualified: after an idle unload or restart the system prompt is prefilled
again and the expert cache starts cold. Checkpoints inside a running server are unaffected.

## Roll back

One command, which restores the exact previous `runtime.json` and unit, restarts the service and loads no model:

```
python3 STAGE/rollback/rollback.py
```

Softer switch: remove the `Environment="LOCAL_AI_ENABLE_OXE_FLASH=1"` line from the installed unit, run `systemctl --user daemon-reload`, and restart the chat service. Flash returns to the BigMoe backend.

## Idle QoS on the IO core

While the Rust Flash engine is loaded, the controller starts `local-ai-flash-qos@<ENGINE_PID>.service` after the engine
reports ready, and stops it on unload and on a failed load. The root helper sets a CPU idle knob on core 8 (the IO
thread's core, `--flash-io-cpu 8`) and clears it when that pid exits. Affinity: decode uses cores 0-7, prefill 0-7 and
9-11, and core 8 is left to the IO thread. The profile key `oxe_flash.idle_qos` only switches the hook on; its values
(cpu 8, latency 100 us) are fixed in the helper, which is the only owner of the knob.

The hook is an optimisation. If the unit is missing, polkit refuses or systemctl fails, the load and unload still
proceed, the engine runs with default idle states, and `/_local/status` shows `flash_idle_qos` with `enabled`, `unit`,
`started` and `error` (the error type name). Status never runs systemctl.

Install, user part then optional root part:

```
python3 STAGE/install.py                  # the Flash switch (user)
sudo STAGE/system/install-root.sh         # helper, system unit template and polkit rule; starts no unit
```

Roll back, root part first, then the user part:

```
sudo STAGE/system/rollback-root.sh        # stops the units, resets the knob, removes the three installed files
python3 STAGE/rollback/rollback.py        # the Flash switch (user)
```

Fallback: stage with `--io-mode spin` for no hook, no `system/` directory and `OXF_IO=spins=1000000`. A default (qos)
stage refuses to run unless `packaging/flash-qos` holds the helper, the unit template and the polkit rule.

## Verify

After selecting the alias, `/_local/status` reports `backend_kind: oxe` and `execution_policy_mode: memlat` once loaded. A round trip loads the model (about 26 GiB peak), so run it only with owner approval:

```
curl -s http://127.0.0.1:8092/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-flash-next-local","messages":[{"role":"user","content":"Reply with OK."}],"max_tokens":8}'
```

## Not yet qualified

- The Rust Flash server hash, the measured peak and the sidecar identity come from the qualification ticket, which has not passed.
- Until then the stage tool refuses and the live profile stays on BigMoe.

## Cache size fallback

The package starts Flash with a 20 GiB expert cache. With `--fallback-measured-peak-bytes N` (the measured peak
at 18 GiB) the staged override also carries an 18 GiB admission: when the 20 GiB admission fails at start
(available memory below peak + 1.5 GiB, or the peak above the cgroup limit) the controller launches the same
server with `--flash-cache-gib 18` instead, and refuses only if that does not fit either. The choice is made
once per load and recorded in the launch snapshot (`oxe_flash.fallback_active`); `/_local/status` does not
expose which cache size is running yet.

## Unload

Idle unload is 600 s. To free the memory at once: `local-ai unload`, or `POST /_local/unload` (optionally
`{"model": "qwen3.8-flash-next-local"}`). Status fields for a panel button are in `docs/UI-DATA-UNLOAD.md`.

## Memory policy 29 GiB / 1 GiB (inherited)

When the Flash profile carries `flash_memory` (see `docs/FLASH-MEMORY-SCOPE.md`, staged separately by
`tools/stage_flash_memory.py`), the Rust engine starts in its own 29 GiB scope: required available memory is
the measured peak + 1 GiB, the admitted budget may be 29 GiB, a 21 GiB cache is allowed (`--cache-gib 21`), and
the memory guard watches the engine's own scope. Without the key the package keeps the 28 GiB unit limit, the
1.5 GiB floor and 18/20 GiB caches. Stage the memory policy first, then this package: the stage tool reads the
budget from the profile.

## USB-C telemetry interruption (2026-11-11 source/benchmark update)

A diagnostic arm stopped cleanly when battmgr briefly omitted the battery reserve despite USB-C/UCSI reporting external power. The shared source and lab runner now permit missing reserve only while an external Mains/USB/UCSI/Wireless supply reports online, and log degraded/recovered telemetry. Empty capacity first uses valid energy/charge ratios; it never invents a percentage. A valid reading still refuses startup below30% and stops at20% or below, even on external power. Without both reserve and external-power evidence it stops. Numeric invalid readings remain rejected. A continuous10-minute degraded interval warns for follow-up.

Five new regression tests plus the existing full suite pass (849 tests,2 skips). Native loading grace ends on the engine startup line for the pending Flash CLI screens, rather than a fixed5-second guess; floor/OOM/swap/runtime-pressure rules are unchanged. This source update has not yet replaced the installed controller package; it is kept separate from the pending UI stage.
