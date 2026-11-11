# Flash prefix pair: INSTALLED_O43b with timing exceptions retained

The prefix-store pair is **INSTALLED_O43b** on the test laptop. The final live smoke passed security, chat identity and actual native configuration checks after waiting for readiness. The installed engine is `0a40d27` (`f60e9cc4`), with GUARD-2b policy source `74039b0` and controller port preflight `ae5635d`. Actual cache is 18 GiB, context is 16K, decode uses 8 threads, and the private prefix store is ON. Public prefix caching remains OFF by default.

Qualification is **accepted with the named O-43/O-43b exceptions**. Both original raw receipts remain FAIL: P1 completed disk reuse and the protected 13/13/parser task, then missed chat/cancel timing; the fresh GUARD-2b repeat missed only cancel prompt/decode timing. All five output identities, security and memory checks passed. O-43 accepted those two cancel timing fields because actual recovery was **3.41945118 s < 5 s**. Its other four cases passed the unchanged comparison.

The later installed full five-case run also retains its raw timing FAIL: the two-token chat and 4K-context predicted-time observations were about **175 ms and 118 ms** above their controls. O-43b explicitly accepted those timing-only exceptions. The final ready-wait smoke is PASS under that decision, while `raw_regression_passed` remains false. No raw result or threshold was rewritten. Native swap and OOM events stayed zero; cleanup ended unloaded with keys removed and QoS 0 (100 while loaded).

An intervening occupied-port attempt was rolled back. The installed controller now checks the native port before loading; the final ready-wait smoke is the current outcome.

For the identical 5,547-token first chat after unload/reload, cold prompt processing was **99.299845 s**, versus **5.133609 s** with 5,485 tokens restored and 62 processed anew: **19.343× for prefill only**. Disk loading took 101 ms; both arms generated the same 55-token result. This is one first-request/reload pair, not a whole-request or decode speedup claim.

GUARD-2b retains strict continuous host conditions for 3 seconds: available+SwapFree below 2 GiB, or available below 256 MiB. Ready-phase PSI full avg10 above 20% uses 3 seconds; during loading, above 60% uses 5 seconds. Native hard OOM and zero native swap remain immediate. Only `memory_policy.py` changed from the previously qualified GUARD-2 source; the native model, numerical path and argv did not change. Model-free checks passed before freezing source `74039b0`.

The [portable composite receipt](../benchmarks/results/public/FLASH-PREFIX-2.json) records the installed outcome and retained raw failures. **FP-SHORTLAT remains open for v0.1.2**, targeting short-fixture timing deltas no greater than 50 ms. The [768/1024 image work](IMAGE-LARGE-SCREEN.md) is also held for v0.1.2; neither image route is expanded by this release.

## Historical store-OFF R0 STOP

The earlier post-reboot R0 control stopped under the previous guard before its raw 4K request completed. R1 and R2 were never run in that attempt, which produced no timing comparison or output-identity result. Those STOP metrics remain unchanged below; subsequent GUARD-2 long-context controls and the O-43 composite supersede its installation decision.

R0 ran on kernel 47 (`7.3.0-47-omarchy`) with the restored secure v0.1.1 engine, preference 18 and the disk prefix store **OFF**. Authenticated properties confirmed the actual 18 GiB tier. CPU8 QoS was 100 at readiness. That historical run retained the 29 GiB scope cap, zero model swap, immediate 1 GiB host floor and previous PSI guard.

| Observation | Value | Meaning |
|---|---:|---|
| Earlier observer minimum | 1,131,393,024 bytes at 14:03:53.592443 local time | Sampled before the terminal controller guard; it is not the terminal minimum. |
| Terminal controller availability/minimum | 980,955,136 bytes | Below the 1,073,741,824-byte floor; `host_available_below_floor` caused the stop. |
| Native cgroup peak | 25,893,445,632 bytes (25.893 GB) | Charge within that scope, not whole-host memory usage. |
| Model swap / scope OOM / global OOM delta | 0 / 0 / 0 | The guard stopped the native before OOM. |
| Cleanup | Unloaded; QoS 0; keys removed | Production runtime and unit unchanged by R0. |

The run lasted 8.4815 seconds. The terminal guard was captured at 14:03:53 before cleanup. Its PSI full average was 1.26%; the observed 2.013-second window was 4.902%. The stop was the host floor. The unchanged nonstreaming endpoint reports a combined prefill/decode phase, so there is no per-token phase split to report.

The native command line disappeared during exit, producing an incidental `actual_native_argv_changed` exception. The persisted controller guard establishes the stop cause; that exit observation does not establish a configuration mismatch.

Memory pressure occurred with the prefix store OFF, so the store is not required to trigger that historical failure. The observation does not attribute all host memory use to the model. The first prefix rollout was rolled back at that point; the current installation described above uses the later accepted composite and GUARD-2b.

Historical image and earlier timing observations were captured in the kernel 40 epoch. This kernel 47 control is a memory-safety result, not a cross-epoch speed comparison. Future performance decisions still need fresh warm-up and matched controls in the new epoch.

[Whitelisted numeric result](../benchmarks/results/flash-prefix2-postboot-20261111.json) · [Image screen and epoch caveats](IMAGE-LARGE-SCREEN.md) · [Memory policy](HOST-MEMORY.md)
