# Fresh smaller-MoE normal-controller reference

The current daily coding profile completed the unchanged protected `atomic-reservations-v1` task twice: **488.191 s and 184.589 s**, with 12/12 independent tests and complete final responses in both. The first backend started unloaded, but **both first turns already restored 5490 cached tokens versus 60 new (98.919%)**. This is a backend-cold/prefix-restored reference followed by full unload/reload/prefix restore; it is not a cold-prefix measurement or an optimization A/B. [Allowlisted numeric evidence](../benchmarks/results/moe-normal-reference.json).

| Current normal profile session | First | Full reload |
|---|---:|---:|
| Finished protected task |488.191 s|184.589 s|
| Generated output tokens |2670|1816|
| New input tokens summed over steps |6035|3031|
| Steps / completed tools |11 /17|7 /12|
| First-turn new / cached tokens |60 /5490|60 /5490|
| Independent tests / final completion |12/12, complete|12/12, complete|

The second finished time includes the between-session unload/reset; its OpenCode process took 183.982 s. Different valid trajectories and substantially different generated/input work prevent treating the 488→184 variation as a prefix speedup, a cap effect or a model ranking. The normal controller recorded two restores and two primes without clearing or inspecting the real prefix store. Both tasks changed only the three intended implementation modules, preserved protected files and completed an actual 12-test command.

## Exact model, settings and resource differences

Model: **Unsloth Qwen3.6-35B-A3B-MTP, UD-Q4_K_M**, publisher revision `5bc3e238d916f48a861bac2f8a1990a0e9b7e98d`, 22,663,387,424 bytes, SHA256 `0b21525e972670ed59e1812e170b27c26355381f0656ecc4e25617ece7dac58b`. The publisher filename omits MTP; these are distinct weights from the ordinary non-MTP file. [Exact artifact identities](../dependencies/model-artifacts.json).

The unchanged qualified llama server SHA256 is `1bd3e439f357365c50fc8390cc2c7e4cde2f86f14fa1771a65f2323a307457b1`. Settings: decode 4/batch 10,16K context, batch/microbatch 256, CPU repack enabled,**Flash Attention OFF**, F16 K/V, native draft-MTP maximum 2, and `LLAMA_CPU_SMALL_BATCH_LIMIT=3`. The leading-prefix disk-cache budget remains 3 GiB. No profile, model or controller cap changed for this reference.

The normal backend retained **CPU 800%, MemoryHigh 24 GiB, MemoryMax 25 GiB**, zero swap and 256 tasks. Its start admission is the existing model-size+3 GiB rule: 25,884,612,896 available bytes. The isolated client had a separate CPU 200%/2 GiB scope. Flash's benchmark used CPU 600% and 21.5 GiB native admission, with different tools/system prompt, quantization, batch/microbatch, speculation and prefix behavior. These references are **not resource-matched**. Passing the same 12 tests establishes this repair's outcome, not universal quality equality.

## Memory and CPU pressure were present

Unload, client/relay reaping, RAM recovery and lease-free checks passed. Minimum sampled MemAvailable was 3,912,523,776 bytes; 28,168,478,720 bytes were available after cleanup. Normal-service max/OOM/kill event deltas were zero. That does **not** mean no pressure:

- `memory.high` increased **89,590** times: 48,751 and 40,839 within the two observed native-process windows. These are event counts, not seconds.
- The service's cumulative peak was 25,790,746,624 bytes, about 24.020 GiB, just above its 24 GiB soft limit. 15 sampled points were at/above that limit.
- Memory PSI some/full increased 9.125/9.123 s between the first/last samples; peak 10-second averages reached 19.4%. CPU PSI some/full increased 21.060/20.727 s; I/O PSI 1.211/1.198 s. These overlap and must not be added as wall partitions.
- Normal-service CPU usage increased 3357.653 core-seconds. CFS recorded 2357 throttled periods and 205.906 s of aggregate throttled accounting; that is not 205.906 s of exclusive task wall time.
- Maximum sampled thermal zone was 97.7°C, across all readable zones rather than an asserted CPU diode. Net battery percentage rose 99.567→99.986; signed power was retained without interpreting its sign as charging.

The first/second observed native windows used 2586.210/769.248 normal-service core-seconds and recorded 2552/1506 additional process major faults. Native-process read-byte deltas were 35.46/36.55 GB over those sampled windows. They include native file/mapped-page and prefix activity, not isolated weights-only I/O or physical-device throughput. Windows may miss short startup/teardown gaps; exported process identities are ordinals, not actual PIDs.

The data support investigating the soft limit and CPU/attention paths; they do not prove that raising the cap caused or would cure the session-time difference. A 24→24.5 GiB soft-limit test would need a bounded same-profile control, unchanged 25 GiB hard limit, live 2 GiB reserve and complete quality/release checks. No cap was raised here. FA off/on is a separate transfer candidate; Q8 KV is a separate numerical-quality change. Neither is enabled merely because Flash uses another configuration. [Cross-model transfer matrix](CROSS-MODEL-OPTIMIZATIONS.md).

Raw receipt SHA256: `66275a970e3ee87045f649056a0ac873e7b6ed797cf3ca9a9c5327904131b0b3`; reference harness manifest `4e3797a0f19b27742dee8df297746a1968ff512f3b6446d5592115a9493db6db`. Only allowlisted numeric/configuration/provenance fields are public. No user paths, prompts, tool payloads, cache contents or histories are included. Nothing was deployed or pushed by this curation.
