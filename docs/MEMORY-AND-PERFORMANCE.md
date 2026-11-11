# Memory and performance policy on the 32 GB Snapdragon laptop

Research-only review, 2026-11-08. Using more RAM is worthwhile when it reduces
finished-task time without changing the required answer or causing memory
pressure. The existing Flash cache comparison demonstrates that opportunity.
The current receipts do not establish a safe or faster 13/14 GiB configuration.
No runtime, profile, frozen deployment artifact, guard or model was changed for
this note; no model, build or benchmark was run.

All capacities below are binary GiB. Model files, process RSS, cgroup charges,
expert-cache capacity and host MemAvailable measure different things. A mapped
file's full byte length is not its physical residency. Conversely, admission
must not subtract all mapped bytes from an observed peak and assume them free.

## Flash: what the finished task actually used

The completed protected three-module repair passed 12 independent tests and
finished its final response. Verified task time was 1172.258 s; total wall time
was 1175.218 s, with readiness after 9.791 s. It used the demand-mapping build,
IO4 with overlap, 16K context, 4096 microbatch, 12 GiB expert cache, four decode/six
batch threads, F16 KV, THP disabled, scoped slim request, all 512 experts with
native top 10 routing, repack off and no speculation. The native empty-state
spelling was reconciled offline; original timing/memory values were preserved.

| Recorded quantity | Exact bytes | GiB |
| --- | ---: | ---: |
| Maximum sampled process RSS | 21,159,288,832 | 19.7061 |
| Final authoritative cgroup peak | 20,843,126,784 | 19.4117 |
| Admission measured peak, the larger of those | 21,159,288,832 | 19.7061 |
| Host available before this task | 27,932,102,656 | 26.0138 |
| Minimum sampled host available | 6,547,140,608 | 6.0975 |
| Host available after cleanup | 27,739,877,376 | 25.8348 |
| Conservative resident estimate | 22,776,560,128 | 21.2123 |
| Reviewed admitted budget | 23,085,449,216 | 21.5 |

The 19-minute run had no memory.high, memory.max or OOM events. Its qualification
cgroup used 22 GiB high/24 GiB maximum, swap 0 and CPU 600%. All ten recorded native
phase summaries retained exactly 12288 MiB cache budget and approximately
12286.25–12287.55 MiB resident cache. The enabled cache brake therefore did not
show a reduced budget in any captured phase. There is no evidence that these
memory caps throttled this run. CPU quota is a separate possible performance
constraint; this note does not measure its effect.

The new budget changes rounding only:

```
base = max(22,776,560,128, 21,159,288,832 + 268,435,456)
budget = ceil(base / 536,870,912) * 536,870,912 = 21.5GiB
required host available = budget + 4GiB = 25.5GiB
```

The original whole-GiB rounding required 26 GiB. The 512 MiB policy leaves the same
estimate, measured peak, 256 MiB measured-peak margin and 4 GiB host reserve intact.
It is an explicit admission-policy derivation, not another native run or a
lower measured-memory claim. The separate activation-time idle reading was
reported around 25.90 GiB; it is distinct from this task's 26.0138 GiB starting
reading. The live 4 GiB guard remains necessary for other requests and desktop
activity. One task does not bound every possible 16K conversation.

## How much additional cache might fit?

The following is arithmetic for future experiments, assuming each extra GiB of
cache adds exactly one GiB to both the old estimate and measured peak while
everything else stays constant. Allocation alignment, working-set changes and
different routing can break that assumption. These are not measured results.

| Expert cache | Projected estimate | Rounded budget | Start requirement with 4 GiB reserve | Projected minimum host available |
| --- | ---: | ---: | ---: | ---: |
| 12 GiB, qualified | 21.212 GiB | 21.5 GiB | 25.5 GiB | 6.097 GiB, measured |
| 13 GiB, hypothesis | 22.212 GiB | 22.5 GiB | 26.5 GiB | 5.097 GiB |
| 14 GiB, hypothesis | 23.212 GiB | 23.5 GiB | 27.5 GiB | 4.097 GiB |

At 25.90 GiB idle availability, neither larger cache passes the unchanged
conservative admission formula. Relative to the 4 GiB live reserve, the measured
run had only 2.0975 GiB spare at its lowest point. A linear 14 GiB projection leaves
about 100 MiB beyond the cutoff, before extra allocations or host activity; that
does not justify calling it safe. 13 GiB is the more credible bounded candidate
if the idle footprint is reduced enough or a separately reviewed estimate is
supported by new measurements. 14 GiB needs stronger evidence and more headroom.

A future comparison should start with 13 versus 12 GiB on the same finished
repair, retain the 4 GiB stop guard, and record actual final cgroup peak,
sampled RSS, minimum host available, expert-read volume/cache hits, complete
final response and independent tests. That can justify a new per-profile
estimate and admission record if it passes. This recommendation schedules no
run and promises no gain. Idle host memory also supports file caching and
desktop work; filling it without reducing task time is not an optimization.

The adapter currently qualifies only 8192/12288 MiB caches. 13/14 GiB would require
an explicit isolated policy extension and new identity-bound memory, numerical,
finished-agent, cleanup and power qualification. Increasing a config value alone
would not make the current proof apply. Larger exact caches preserve weights
and routing, but the production gate correctly requires evidence for the new
memory policy. Neither a higher cgroup maximum nor changing rounding creates
extra physical RAM or increases the actual configured cache.

There is a demonstrated reason to investigate: the earlier fixed 49-token,
246-output-token JSON pair improved decode from 1.752 to 2.447 tok/s when cache
increased 8 to 12 GiB, a 39.67% increase. Complete load-plus-request time fell
228.809 to 181.687 s, 20.59% less; expert reads fell 18,592.6 to 12,226.4 MiB and
reported cache hits rose 87.5% to 90.0%. Both exact answers passed. This was a
sequential 8K-context pair on an older build, without independently matched
power/storage-cache conditions. It establishes neither a 40% agent gain nor
linear improvement above 12 GiB. [Cache receipt](../benchmarks/results/flash-next-cache.json).

## Classic resident MoE and dense profiles

These already keep their quantized weights resident and use CPU repack. MoE
uses a 21.107 GiB file and two-token MTP with the small-batch pool fix; dense uses
a 15.411 GiB file and no speculation. Both deployed profiles use 16K context,
F16 KV, one slot, four decode/ten batch threads and batch/microbatch 256/256.
The normal chat envelope is 24 GiB high/25 GiB maximum, swap 0. Its ordinary
admission check is model-file bytes plus 3 GiB: approximately 24.107 GiB for MoE
and 18.411 GiB for dense. This check refuses a load; it does not shrink weights
or set an expert-cache capacity. [Profiles and envelope](RUNTIME-SETUP.md),
[controller](../src/snapdragon_ai_lab/runtime/controller.py).

The available historical memory records are **loaded snapshots, not peaks**:

| Model, older 4K-context/128 microbatch test | Loaded RSS | Anonymous PSS | File PSS | Host available at snapshot |
| --- | ---: | ---: | ---: | ---: |
| MoE, MTP pool limit 3 | 26.781 GiB | 20.955 GiB | 5.821 GiB | 6.557 GiB |
| Dense, four decode threads | 26.280 GiB | 15.285 GiB | 10.991 GiB | 11.843 GiB |

These include file-backed mappings alongside packed anonymous allocations.
RSS and MemAvailable overlap in their accounting of potentially reclaimable
file pages, and cgroup charges are a further distinct measurement. The snapshot
values cannot establish the current 16K agent peak or prove that a 25 GiB service
cap slowed it. No per-run peak/minimum or memory-event record is present in the
curated normal classic agent receipts. That is a measurement gap, not evidence
for reducing admission or discarding the file-backed portion of RSS.

The current normal MoE repair took 264.732 s cold and 210.208 s after full unload
and prefix restore, both with 12 protected tests passed. A separate normal
single-file MoE repair took 177.993 s. Dense's different single-file fixture took
432.888 s cold and 201.595 s after full reload/restore. The different fixtures
cannot rank model speeds. These preserve practical resident/cache benefits;
SSD expert streaming adds work when the weights already fit. [MoE](../benchmarks/results/normal-moe-opencode.json),
[MoE single file](../benchmarks/results/normal-moe-port-opencode.json),
[dense](../benchmarks/results/normal-dense-opencode.json).

More batch workspace is a finite candidate rather than an established gain.
The older dense screen held logical batch 512 and compared physical microbatch
64/256/512 across 4267 cold prompt tokens: aggregate prefill was 14.88/21.08/20.90
tok/s respectively. 512 did not beat 256 on that metric. Request totals were
400.981/331.237/324.874 s, but outputs differed, so the last difference cannot be
attributed to a faster 512 kernel. The control predates a reboot/charger change;
the screen used an older engine and six decode threads rather than the current
four. No matched finished-agent MoE 256-versus-512 evidence was found in these
receipts. Preserve repack, MTP and its pool fix in any future matched comparison;
raise logical batch as well when testing a larger physical microbatch.
[Preliminary screen](../benchmarks/results/cpu-screen-preliminary.json).

MTP already spent memory for a measured decode benefit: ordinary versus
adaptive MTP medians were 13.886/16.857 tok/s, while the same MTP weights with
pool limit 1 versus 3 gave 10.984/16.857 tok/s. The simple repair was about 133 s
with either ordinary or adaptive MTP. These are decode results with workload
and power caveats, not a universal agent-completion gain.
[MTP evidence](../benchmarks/results/before-after.json).

## Evidence identities

Private receipt locations are described generically; no user prompts, images,
auth or histories are included here. The Flash result is
`final-agent-20261108-090110-reconciled/result.json`, SHA256
`ba3e3aa0e08261cd0131632d9dcbd21d255540c8e0cec86839d7411277876d6b`.
Its unchanged original qualification is SHA256
`c62ec204515f64fc62c1869316f1251520c56b3937f34310b0c6c3a6d2f02c3f`;
the new `qualification.admission-reviewed.json` is SHA256
`8411631b9879d7d3292464584bed4a3ac8be7903a1377cd72d49e7d597727f73`.
Selected server SHA256 is
`8588e6b619f3292a412504f070e67b994162ad706ae1dfc5e09359a24f341773`,
source manifest `1fa9c70517f5543f8456afa3f69c2fad225c9bda9889d9f76c373ca2d635f2c5`,
build manifest `fa605e67eafbdb4e5965131f94e0577a68319fd67519263e729348856f2734b8`.

The historical MoE snapshot is
`llm-cpu-mtp-tuning/results/small-batch-3/result.json`, SHA256
`74dee6c2aaf3417c1098d6feeef7e6e5ed6ece16c88b331c5a217701e540ac54`;
the dense snapshot is `moe-comparison/results/dense-thread-4/result.json`,
SHA256 `2ac9bf35a089920b6992632a8358ade4787dfe226e0d038f481d1d9e956bd022`.
The evaluator records those memory values once after loading; its default
context/batch settings establish the historical 4K/128 scope above.

## Later normal MoE diagnostic: measurable soft-limit pressure

The later synthetic normal-frontend switch diagnostic partly fills the classic
measurement gap described above. Its successful MoE READY response is recorded
at cumulative elapsed **40.445 s**. The normal chat cgroup's observed
`MemoryPeak` was **25,777,524,736 bytes = 24.00719 GiB**, with
`MemoryHigh=24 GiB` and `MemoryMax=25 GiB`. This is a cgroup peak, distinct from
the historical loaded RSS snapshots. No minimum host-available measurement was
recorded for this diagnostic.

At the subsequent read, `memory.events` contained `high=48214`, `max=0`,
`oom=0`, `oom_kill=0` and `oom_group_kill=0`; `sock_throttled=157` was also
present. These are cgroup counters, not 48,214 seconds or a count of failed
requests. The high events show the soft threshold was exercised; they do not
establish how much of this request's elapsed time it cost. They do not change
the earlier Flash task's separate zero-pressure-event observation.

At **2026-11-08 08:53:37 UTC**, the current normal chat cgroup's
`memory.pressure` snapshot was:

| PSI scope | avg10 | avg60 | avg300 | Cumulative total, microseconds |
| --- | ---: | ---: | ---: | ---: |
| some | 0.00 | 0.19 | 0.72 | 4,753,136 |
| full | 0.00 | 0.19 | 0.72 | 4,753,021 |

Those totals are about 4.753 s of cumulative pressure time for the cgroup's
lifetime at that snapshot. There is no captured before/after PSI delta binding
them solely to the 40.445 s READY interval, so dividing the totals by that
request time would invent an attribution. The scopes overlap and must not be
added. Moving averages describe their sampling windows, not a measured task
speedup or a peak/minimum memory value.

The diagnostic's host-available snapshot immediately after READY was
**5,341,237,248 bytes = 4.97441 GiB**. After MoE cleanup it recovered to
**28,423,819,264 bytes = 26.47174 GiB**. The following Flash chat returned 503
only 0.299 s after the successful selection record; root observed the backend
exit in about 0.3 s. Available memory before that failed chat exceeded the
qualified 25.5 GiB threshold. This transition therefore provides no evidence
that Flash's admission budget caused the failure; root is diagnosing startup
separately. It remains an activation failure, not a failed coding-quality test.

**A 24.5 GiB soft high limit is a reasonable future measured comparison**, with
the hard 25 GiB limit and Flash's live 4 GiB reserve unchanged. It raises the
soft threshold by 536,870,912 bytes and leaves 512 MiB between high and hard max.
The new soft threshold would sit 529,149,952 bytes, about 504.6 MiB, above this
observed peak. That could reduce repeated reclaim/throttling during resident
MoE loading. The observed peak was itself constrained by the current policy,
so it is not an upper bound on what the process would use with a higher limit.
The one post-READY host snapshot also does not prove transient headroom: a
simple extra 512 MiB subtraction would leave about 4.474 GiB at that instant,
but neither its minimum nor future desktop activity is known.

A defensible future comparison of 24 and 24.5 GiB needs the same qualified
weights, repack/MTP, context, batching, prefix state, power conditions and
protected finished-agent task. Capture PSI and memory-event counters before
and after each arm, actual cgroup peak, host minimum available, cold/warm load
and first-response times, complete final response, independent tests and
cleanup. Counter deltas and finished-task time must improve together before
claiming a speed benefit. No limit was changed or experiment launched here.

Provenance: private synthetic receipt `planning-switch-diagnostic.json`, SHA256
`b89c29b133dc69fc049e20a8adef2d09dfc183a75c66f1e874248a535bf61a2a`;
read-only normal chat unit/cgroup metadata and PSI snapshot at the UTC time
above. No model output text, prompts, auth or user histories were inspected for this
appendix; only lifecycle routes, numeric counters and own-receipt metadata.
