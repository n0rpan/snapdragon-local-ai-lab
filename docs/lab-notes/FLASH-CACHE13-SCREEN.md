# Flash expert cache: 12 versus 13 GiB

The completed ABBA passed the unchanged exact 353-token code oracle in all four runs, preserving all 512 experts and native top 10. The larger cache raises weighted decode from **3.891 to 3.979 tok/s (+2.26%)** and reduces decoder read bytes **11.40%**. This is a screen survivor, not a production promotion. [Numeric result and provenance](../../benchmarks/results/flash-cache13-screen.json).

| Order | Cache | Complete wall | Engine load | Decode rate | Decode read MiB | Native session cache hit |
|---|---:|---:|---:|---:|---:|---:|
|a1|12 GiB|116.042 s|11.452 s|3.881|57734.1|77.8%|
|b1|13 GiB|109.275 s|6.866 s|3.992|51152.9|79.7%|
|b2|13 GiB|109.548 s|6.719 s|3.966|51152.9|79.7%|
|a2|12 GiB|111.052 s|6.634 s|3.901|57734.1|77.8%|

Mean complete screen time falls 3.64%, but the first control loads more slowly; do not attribute the whole wall difference to cache size. Prefill remains about 11.6–11.8 s, and native hit rate rises from 77.8% to 79.7%. The hit ratio is session-wide, not an exact cache-lookup counter or decode-only ratio.

All four original answers match SHA256 `460169e27739a86fbb94340a3d7ffcbace214719d46fcda2cc8bb3d0fc0a4093`. The same ba88 engine, IO8, 4 decode / 6 batch threads, 16K context and microbatch 4096 and numerical model settings remain. Only `--cache-mb` changes. CPUQuota 600%, memory high 22 / max 24 GiB and own swap 0 remain; actual unit peak was 19.555 GiB with no memory max/OOM/PID events. Admission remains 21.5 GiB plus an actual 2 GiB available reserve. Every worker exited, EOF was complete, RAM recovered and the canonical lease was released.

Plugged-in battery assistance was allowed above 20%, with power/thermal samples retained. The earlier stricter-power interruption remains separately failed. This screen does not establish a full protected-agent speedup or qualify an installed cache change. The full task must improve with all 12 tests and complete final output before promotion.
