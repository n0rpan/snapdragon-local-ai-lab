# Flash CPU quota: no useful gain at four decode threads

The completed CPU600/CPU1200/CPU1200/CPU600 screen was **flat**: request time −0.20%, decode time −0.34%, prefill time +1.11%. All four 353-token answers matched the original SHA and independent code oracle. This closes the quota-only screen without a full-task follow-up or promotion. [Numeric dataset](../benchmarks/results/flash-cpu-quota-screen.json).

| Mean seconds | CPU600 | CPU1200 |
|---|---:|---:|
| Prefill | 11.625 | 11.754 |
| Complete request | 102.480 | 102.279 |
| Decode | 90.618 | 90.305 |
| Load-to-cleanup arm | 112.410 | 110.564 |

Complete-arm times were 113.955, 110.768, 110.360 and 110.865 s. The first control loaded in 9.634 s versus about 6.4 s later, explaining most of the apparent 1.64% complete-time reduction. Do not present that as a CPU speedup. These short prompts also do not settle all larger-prompt scheduling behavior.

Only the transient service quota changed. Decode 4/batch6, IO8, cache12 GiB, OMP6/PASSIVE/FALSE, original binary, context16K/F16/FA and all routing settings stayed fixed. The unit verified exact leaf/ancestor caps before both native spawns and continuously through a 250 ms watcher. Original quotas were restored, all children reaped and watchers joined, RAM recovered and the canonical lease released. Swap and hard memory/PID events remained clear; the raw records retain soft pressure and power/thermal measurements.

The smaller classic MoE's [quota correction](CLASSIC-QUOTA1200.md) therefore does not transfer automatically to this Flash fixture. The next decode 4-versus 6 screen holds CPU1200 constant **in both conditions**, solely to leave headroom for six/eight threads and I/O workers. This is a root-selected experimental budget, not a claim that CPU1200 already improves Flash. Production stays unchanged; final protected-agent validation remains after the worthwhile short-screen queue.
