# Flash-Next prompt-batch experiments

These historical prompt/cache experiments preceded the winning demand-mapping/IO4/overlap candidate. That candidate later completed the protected full-agent repair in 1175.218 s with 12/12 tests and a complete final response, and its planning profile is now installed. Complete normal planning/image-handoff acceptance also passed separately in 238.448 s. [Installed lifecycle evidence](../benchmarks/results/flash-next-normal-lifecycle.json) The component pairs below do not independently qualify that workflow. [Final repair](../benchmarks/results/flash-next-final-agent.json), [installed policy](FLASH-NEXT-PLANNING.md)

## Wider prompt batches reduce repeated storage reads

These are two separate sequential comparisons on the Snapdragon X Elite laptop with 32 GB RAM. Each arm used a fresh CPU-only process, the same quantized weights, native expert routing, an 8 GiB logical expert cache, four compute threads and six batch threads. Each comparison used its own fixed synthetic prompt and exact answer check. Both arms in each comparison answered correctly and released memory after exit.

| Prompt | Microbatch | Prefill | Prefill time | Prefill reads | Total including load |
|---|---:|---:|---:|---:|---:|
| 350 tokens | 128 | 3.46 tok/s | 101.02 s | 52,164 MiB | 158.11 s |
| 350 tokens | 512 | 6.51 tok/s | 53.80 s | 25,661 MiB | 111.20 s |
| 922 tokens | 512 | 6.58 tok/s | 140.14 s | 51,144 MiB | 203.83 s |
| 922 tokens | 4096 | 8.95 tok/s | 102.97 s | 30,615 MiB | 169.62 s |

The 350-token pair improved prefill by 1.88× and reduced total time by 29.7%. The separate 922-token pair improved prefill by 1.36× and reduced total time by 16.8%. The latter pair verified process THP disabled in both arms. A wider batch lets more prompt tokens share streamed expert-weight reads, consistent with the observed reduction in prefill reads. This is a mechanism-supported interpretation, not a measurement that attributes every saved second to storage alone. [350-token evidence](../benchmarks/results/flash-next-prefill.json), [922-token evidence](../benchmarks/results/flash-next-wide-prefill.json)

The pairs ran sequentially without clearing global caches. OS caching, storage and thermal conditions may differ, and electrical conditions were not independently matched. Do not multiply the two ratios into a general speedup. Each answer contained only three tokens, so neither pair establishes faster sustained generation. Microbatch 4096 was subsequently retained in the passed protected full-agent repair and installed policy; these three-token screens alone did not provide that validation.

For the separate memory optimization and its tradeoff, see [process-scoped THP results](STREAMING-MEMORY.md). Disabling THP improved active host headroom; it did not measurably improve the matched three-case runtime.

## A larger expert cache improves one sustained output workload

A separate 8-versus-12 GiB logical-cache pair used microbatch 4096, 8K context, process THP disabled, and the same CPU thread settings and weights. Both arms returned the exact required 64-integer JSON array, generated 246 tokens, exited cleanly and recovered RAM. [Cache comparison](../benchmarks/results/flash-next-cache.json)

| Observation | 8 GiB cache | 12 GiB cache |
|---|---:|---:|
| Generation | 1.752 tok/s | 2.447 tok/s |
| Decode weight reads | 18,593 MiB | 12,226 MiB |
| Reported cache hit rate | 87.5% | 90.0% |
| Prompt processing | 25.38 s | 27.05 s |
| Load | 61.25 s | 52.15 s |
| Total including load | 228.81 s | 181.69 s |
| Minimum host available RAM | 12.15 GiB | 8.96 GiB |

Generation was 1.40× faster and total time fell by 20.6% in this sequential pair. The short 49-token prompt did not prefill faster. The repeated structured output favors cache reuse and is not a coding workload; the result does not predict an agent speedup. Load time varied, electrical/thermal conditions were not matched, and OS caches were not cleared. The 12 GiB setting was subsequently exercised in the bounded real repair below. Its correct artifact did not finish the complete agent workflow within the allowed window.

## Coding-client failure remains separate from model quality

The earlier 16K-context, microbatch-4096 coding test completed three valid read-tool calls, then the OpenCode client exited with SIGABRT after about 663 seconds. No source edits or completed repair were recorded. Protected tests stayed unchanged, the test processes were reaped, and RAM recovered. Recorded cgroup samples contained no OOM kill; no core was retained. Original client stdout/stderr were discarded, preventing a precise abort diagnosis from that run. [Curated failed-client observation](../benchmarks/results/flash-next-agent-client-abort.json)

Both a fast model-free replay and a replay with a 665-second delay before the first model response completed normally, including three actual completed read calls. The delayed probe took 666.86 seconds overall, exited OpenCode with code 0 and retained zero stderr bytes. This rules out a simple 660-second cutoff in that probe. It does not reproduce inference or explain the earlier abort. The subsequent actual run captured bounded stdout/stderr and retained zero stderr bytes. The earlier failure gives no finished-task time or quality ranking for the model. [Client controls](../benchmarks/results/flash-next-client-replay.json)

The initial slim-prompt control failed tool permissions and retained all 12 schemas; it remains an unsuccessful control. The corrected replay completed all three read calls, exposed four required schemas rather than twelve (4,778 versus 14,057 characters), and reduced system text from 8,633 to 7,432 characters. It did not change the user message or perform a repair. Permission policies use actions/resources such as `read:*`, `edit:*` and `shell:*`; filtering schemas is a separate operation. The subsequent actual run exercised its read, write and shell tools; the complete-workflow deadline still failed.

## Earlier bounded repair: correct code, unfinished workflow

That earlier retry used the shorter instructions and four schemas, 16K context, microbatch 4096, a 12 GiB logical cache, process THP disabled and all native experts. It completed five reads, wrote all three source modules, and ran the protected tests. A separate offline oracle observed **12/12 tests passing**, unchanged protected assertions, all three source modules changed, no unexpected files and stable source during the check. That first independent observation was 1,818.14 seconds after unit activation, while the agent was still composing its final response. [Sanitized terminal result](../benchmarks/results/flash-next-agent-deadline.json)

The run nevertheless **failed** its 1,800-second client deadline: the final response remained incomplete, so there is no verified completed-agent time or passing workflow result. Total time including load and cleanup was 1,864.24 seconds. Native model load took 60.51 seconds. The three write responses generated about 1.01–1.12 tok/s; the separate repetitive JSON result of 2.45 tok/s did not describe real coding output. Do not turn the early oracle timestamp into a completed-agent benchmark.

All owned processes and relays were cleaned up, RAM recovered, and the limited fresh-backend/stderr privacy audit passed. Minimum available host memory was 6.55 GiB. Sampled process RSS reached 24.48 GiB, while recorded cgroup peak was 22.00 GiB; mapped/shared/reclaimable pages and cgroup charging differ, so these figures are not interchangeable or a proof of required steady RAM. The larger 160-task limit recorded a peak of 114 tasks and zero limit events. The older 96-task limit is a plausible source of pressure in the earlier SIGABRT, but causation remains unproven.

That retry produced correct code but failed its bounded workflow and did not earn deployment at that stage. The later winning candidate passed a separate complete repair; this earlier failure remains unchanged and is not a successful timing baseline. The established MoE remains the ordinary coding default. Earlier interruptions, the client abort and diagnostic improvements remain in the evidence.

## Additional validation: SSD read concurrency

A separately requested validation round tested two versus four SSD read workers on
the same complete Python coding answer. Both fresh processes used 16K context,
microbatch 4096, a 12 GiB expert cache, four compute/six batch threads, process THP
disabled and unchanged full expert routing. The answers were byte-identical, with
353 generated tokens and 156 prompt tokens. Independent sandboxed assertions checked
ASCII-only ports, Unicode rejection, types, ranges and input nonmutation.
[Sanitized evidence](../benchmarks/results/flash-next-io-workers.json)

| Metric | Two read workers | Four read workers |
|---|---:|---:|
| Complete request, excluding load | 382.00 s | 298.25 s |
| Total including load and cleanup | 444.81 s | 356.67 s |
| Native prefill | 42.17 s | 22.87 s |
| Native generation | 1.040 tok/s | 1.284 tok/s |
| Native generation reads | 57,734 MiB | 57,734 MiB |
| Cache hit rate | 77.8% | 77.8% |

Request time fell 21.9%. Native prefill I/O wait fell from 31.93 to 12.72 seconds,
accounting for nearly all of the 19.30-second prefill reduction. Both runs exited
cleanly and released RAM without rebooting; minimum available RAM was 7.63/8.11 GiB.
This is one sequential pair with unmatched OS cache/order and thermal conditions.
A 46 MB Git metadata copy occurred during the first startup, so the separate load
difference is not attributed solely to read concurrency. This fixture is meaningful
code generation, but is not a completed OpenCode agent workflow.

Four read workers, demand-based startup mapping and expert-read overlap were
subsequently validated in separate pairs, retained in the passed full-agent repair
and installed planning profile. Selective ARM multirow kernels remain a separate
research path. No general speedup is inferred by multiplying these ratios.

The completed follow-up pairs are documented in
[demand-based startup mapping](FLASH-NEXT-SPLIT.md) and
[SSD read/compute overlap](FLASH-NEXT-SSD-SCHEDULING.md).
