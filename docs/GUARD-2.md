# GUARD-2: sustained host-memory stops

The O-39 decision replaces the controller's instantaneous available-RAM
floor. Idle desktop pages may use the NVMe swapfile; low `MemAvailable` alone
does not establish that the model or desktop is in trouble.

After readiness, the controller stops its native child when at least one of these conditions
remains true continuously for **at least 3 seconds**:

| Host condition | Strict threshold |
|---|---|
| Available RAM plus free swap | `MemAvailable + SwapFree < 2 GiB` |
| Full memory stalls | PSI memory `full avg10 > 20%` |
| Emergency available RAM | `MemAvailable < 256 MiB` |

The timer covers the continuous OR of these conditions. A healthy sample resets
it; crossing between unsafe conditions does not. Exact boundary values are
allowed. Attachment and readiness do not waive an observed host-floor interval.
The monitor samples every 250 ms.

GUARD-2b changes only loading PSI: until the first ready signal, `full avg10`
must be strictly above **60% for at least 5 seconds**. The two RAM/swap conditions
keep their original 3-second timers during loading. Separate timers prevent
loading PSI from shortening either host-floor timer. Readiness discards the
loading PSI timer; subsequent samples use the ordinary 20%/3-second rule.
This accounts for loader reclaim stalls when the loader is the only active task.

Native swap usage, a missing zero-swap limit, new scope/global OOM kills and PID
limit failures still stop immediately. Loading retains its bounded deadline.
This change does not alter memory caps, model selection, admission, prefill
splitting or throttling. PSI `some` and desktop swap traffic are telemetry only.

The controller explicitly selects `host_guard_version='GUARD-2'` for native
launches. Historical standalone benchmark callers retain their original guard
unless they select the new version; archived measurements remain reproducible.
Status reports `guard_version`, the actual thresholds, free swap, both PSI
averages, the timer and any active conditions. It reports no legacy 1 GiB floor
for GUARD-2.

The original model-free qualification passed **1,084 tests** (two skipped). Tests cover
strict boundaries, transient dips, the continuous timer, readiness/attachment,
and immediate no-swap/OOM stops. The earlier full-suite mock-signature failure
is retained in the private receipt alongside the corrected passing run.

Private receipt: `~/.local/state/local-ai/guard2-o39-20261111/receipt.json`.
Live qualification uses the separately pinned PREFIX-2 controller and engine;
model-free success alone does not claim deployment or live safety results.
Its 1-second traces distinguish host `pswpin` traffic from native `VmSwap` and
scope swap usage. Major file faults and ordinary streamed expert reads are not
reported as model swap-ins.
