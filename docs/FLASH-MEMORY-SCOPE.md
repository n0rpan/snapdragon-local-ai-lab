# Flash-only memory policy: own 29 GiB scope, 1 GiB host floor

**Status: installed and live-checked, 2026-11-10.** Source4047c73, controller release `flash-memory-39cee80d75cb`. Flash chat passed in its own29GiB/swap0 scope; unload removed the scope. A subsequent Ornith chat used the unchanged28GiB/swap0 chat cgroup. Receipt: the private deployment record `flash-memory-20261110-1700/deployment.json`. Rollback: `python3 ~/.local/share/ai-stack/rollbacks/latest-flash-memory/rollback.py`.

## What is measured (Codex, Rust Flash M3, 29 GiB cap / 1 GiB floor, 2026-11-10 15:00)

| expert cache | decode | hit rate | minimum available | cgroup peak | compositor max frame |
|---|---|---|---|---|---|
| 20 GiB | 116.6 ms/token | 90.9% | 2.46 GiB | 24.88 GiB | 16.4 ms |
| 21 GiB | 112.4 ms/token | 91.8% | 1.62 GiB | 25.86 GiB | 19.3 ms |

No OOM, no model swap; memory pressure peaked briefly near 10% and was not sustained. These runs are the Rust
engine, not BigMoeOnEdge. Traces stay private; only the numbers above are for publication.

## What the patch changes

Only a profile that carries `"flash_memory": {"scope_max_bytes": 31138512896, "host_floor_bytes": 1073741824}`
is affected, and only that exact pair is accepted (anything else: the model refuses to load). The key is
refused on a model that is not Flash. Without the key every path behaves as before.

| | today | with `flash_memory` |
|---|---|---|
| Flash engine's memory limit | the chat unit's 28 GiB | its own transient scope, MemoryMax 29 GiB, swap 0 |
| MoE and other models | the chat unit's 28 GiB, 1.5 GiB floor | unchanged |
| BigMoeOnEdge guard: stop when available memory is below | 4 GiB | 1 GiB |
| BigMoeOnEdge admission | budget <= 24 GiB, available >= budget + 4 GiB | unchanged |
| BigMoeOnEdge expert cache | 8 or 12 GiB (qualified) | unchanged: a larger cache needs its own qualification |
| chat unit file | MemoryMax line | unchanged (the stage tool refuses a unit without exactly one) |

How the scope is made: the controller starts the Flash engine through `exec_in_scope.py`. That wrapper asks
the user's systemd manager for a transient scope holding only its own PID, checks in its own cgroup files that
MemoryMax and MemorySwapMax 0 apply, then execs the engine. Same PID, same parent, same environment; if the
scope cannot be made or verified it exits 78 and the engine never starts. Checked on this machine with a
trivial command: the process landed in `app.slice/local-ai-flash-<pid>.scope` with `memory.max` 31138512896
and `memory.swap.max` 0, and the scope was gone after exit.

The Rust Flash package (branch `claude/flash-production`) takes the same key: its admission then uses the
1 GiB floor and the 29 GiB budget, allows a 21 GiB cache, and its memory guard reads the engine's own scope
(`MemoryGuard(minimum_available_bytes=..., follow_process_cgroup=True)`, provided here).

## Stage (no install)

```
python3 -I tools/stage_flash_memory.py STAGE --runtime RUNTIME_JSON --runtime-sha256 SHA \
  --unit CHAT_UNIT --unit-sha256 SHA --releases-root RUNTIME_RELEASES
```

Writes `STAGE/runtime.json` (one added key on `qwen3.8-flash-next-local`), `STAGE/local-ai-chat.service`
(WorkingDirectory and ExecStart re-pointed at a new content-addressed release, every other line equal),
`STAGE/runtime-source/` and `STAGE/rollback/` (exact preimages and `rollback.py`). Copy `runtime-source/` to
the printed `release_dir`, install the two files under an idle lease, `daemon-reload`, restart the chat service.

**Roll back:** `python3 STAGE/rollback/rollback.py`. Softer: delete the `flash_memory` key from the Flash
profile; the next Flash load runs inside the unit limit with the 4 GiB guard again (no restart needed, the
profile is read at load).

**Verify after install:** `/_local/status` with Flash selected shows `"flash_memory"` with both numbers; while
Flash is loaded, `systemctl --user status 'local-ai-flash-*.scope'` shows the scope and its limit; an MoE model
loaded afterwards has no such scope.

## Scope of this qualification

- The controller launch wrapper passed the real Flash load/chat/unload check. The native scope disappeared after unload; MoE28GiB was checked separately. The base unit still says25G, overridden by `model-profiles.conf` to an effective28GiB; both are preserved.
- This policy does not automatically alter standalone Rust benchmark wrappers: they must request their own29GiB/swap0 scope and1GiB guard, and record the actual cache size. BigMoeOnEdge admission and its qualified12GiB cache remain unchanged.
- If the service environment has no UTF-8 locale, Python adds `LC_CTYPE` to the engine's environment (the
  existing `exec_no_thp.py` wrapper behaves the same way). BigMoeOnEdge's private environment sets `LANG=C.UTF-8`.
