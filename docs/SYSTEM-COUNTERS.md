# Read-only batch counters

`snapdragon_ai_lab.system_counters.snapshot()` reads DDR/LLCC residency,
swap/zswap/refault counters, PSI, memory availability, CPU policy clocks,
thermal sensors and optional inference cgroups. It does not load a model,
acquire a lease or write a system setting. The guarded runner remains responsible
for admission and process lifetime.

```python
from snapdragon_ai_lab.system_counters import snapshot, difference, cgroup_for_pid

group = cgroup_for_pid(native_pid)
before = snapshot(cgroups=[group])
# Existing guarded workload; sample this same API every 10 seconds if desired.
after = snapshot(cgroups=[group])
delta = difference(before, after)
```

For manual snapshots, use fresh output paths:

```sh
python3 tools/system_counters.py snapshot --out /private/run/before.json
python3 tools/system_counters.py snapshot --out /private/run/after.json
python3 tools/system_counters.py diff /private/run/before.json /private/run/after.json --out /private/run/delta.json
```

Outputs are mode 0600; the tool refuses to replace existing evidence. No request
text, argv, environment variables, device serials or network identifiers are read.

Residency uses each `trans_stat` row's frequency and final `time(ms)` column;
the packed frequency header is not split. The delta reports time below the maximum
state as a fraction of measured wall time and includes counter coverage. A
threshold flag requires at least one second with 90–110% residency coverage.
Counter resets, changed state sets, missing sources and malformed files are
explicitly unavailable, not zero.

PSI percentages come from the change in the cumulative stall counter divided by
wall time, rather than stale `avg10`. Cgroup `anon`/`file` values remain gauges
in the snapshots; fault, throttle and OOM values have separate counter deltas.
If a transient scope disappears, keep its last live snapshot rather than
interpreting an unavailable post-run scope as zero.

`below_max_exceeds_2percent` is a data flag, not permission to pin frequency.
O-14 requires measurements during decode and short turns, followed by its separate
bounded comparison. Idle format smokes cannot justify a clock change. This reader
never pins frequency or changes zswap, swap, memory limits or governors.

O-14 min-clock pin work is closed for the measured Flash decode: DDR and LLCC spent 100% of the interval at their maximum state. The one-off decision script was removed; raw counters and the hardware result remain. No clock settings were changed.
