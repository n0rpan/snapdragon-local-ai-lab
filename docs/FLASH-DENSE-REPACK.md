# Flash dense-layer CPU repack

The completed OFF/ON/ON/OFF code screen passed the independent oracle in all four arms, with repeatable answers and passing lifecycle/resource gates. Mean prefill was **12.324% shorter**, weighted decode **2.301% higher**, and mean request time **2.884% shorter**. Outputs differed between conditions, so request timing remains length-confounded. Retain this as a **conditional prefill lead**; it missed the standalone advance gate. Installed defaults remain unchanged. [Curated arms, formulas and disposition](../benchmarks/results/flash-dense-repack-screen.json).

| Fixed-screen aggregate | A: repack OFF | B: repack ON |
|---|---:|---:|
| Mean prefill | 11.6685 s | 10.2305 s |
| Mean request | 102.76875 s | 99.80516 s |
| Weighted decode | 3.88899 tok/s | 3.97849 tok/s |
| Tokens in each repeated answer | 353 | 355 |

Both A answers retained the original SHA; both B answers repeated their own SHA and passed the same code oracle. This is correctness on one fixed fixture, not a general quality result. Paired whole-request **speedups** (`A/B − 1`) were **3.039% and 2.899%**; the gate required both to reach 3%. These use a different denominator from the mean time reduction above. No standalone full task or promotion follows; retain only a conditional combination check after selecting the final arithmetic configuration.

The candidate enables CPU repacking for eligible resident dense weights and explicitly fences all 144 streamed expert banks plus the token/PLE tables. Their raw dimensions, strides and bytes remain unchanged. The strict loader overrides bypass CPU-extra selection for those protected objects. Raw memory is accepted only by exact ordinary-CPU or factory-derived CPU_Mapped identity with no packed trait. Name strings and a generic host flag cannot pass that fence.

Four ordinary-loader combinations exercised extras and strict overrides. The fixture checks actual Q6_K I8MM 8×8 and Q8_0 I8MM 4×8 traits and runs 46 math comparisons, eight exact row-read cases and 18 slot-restoration cases. It also checks metadata-only first-shard handling, missing siblings, substitution rejection, forced-exception restoration and raw-buffer counterexamples. All passing records use the same compiled candidate binary.

The first attempt incorrectly rejected CPU_Mapped. A corrected binary completed its cases, but mixed stdout/stderr corrupted the strict receipt. Line buffering alone left progress dots attached to a record. Both capture failures remain failed. Separating the two bounded owned streams fixed capture; no mathematical gate, expected count, compiled source or binary changed for that rerun.

The earlier complete CLI/server build passed its full native suite: all 41 original tests plus the fixture, 42 total, with all four real-hook architecture gates and zero skips. The 37 binaries, 357 resulting objects/archives, source and CMake/Ninja metadata remain bound to their receipts. The native run took 196.72 seconds with a 977,915,904-byte peak, zero memory/PID events or own swap, and all children reaped.

The [actual target allocation-only probe](../benchmarks/results/flash-dense-repack-allocation-v3.json) now passes after a cleanup-only correction. It executed and identified 497 packed objects in one 3,648,660,480-byte buffer, with all 146 protected objects kept raw. Context and scheduler reservations were 621,355,008 and 1,382,418,208 bytes; six-thread prefill CPU workspace was 195,662,336 bytes. No Session expert cache or token generation ran.

The earlier allocation-only V2 probe **failed strict shutdown memory recovery**. After all children exited, recovery-only advice on 220 verified packed-source ranges reduced the file charge from 3,990,364,160 to 368,312,320 bytes, but the last recorded sample remained above the recovery threshold at the unchanged 15-second recovery deadline. Telemetry stopped then; it does not establish that the same charge persisted until unit termination. The parent retained the inference lease until its finite unit deadline terminated it. The residual has not yet been attributed to model pages versus source/binary verification inputs. Preserve this failed receipt; neither the measured 8,470,781,952-byte allocation-only peak nor mapped address-space capacity establishes inference RAM fit. No replacement or speed credit is claimed.

The separate V3 run reused the exact unchanged helper and qualified archives. After helper destruction and reap, it advised only the verified nonexpert source prefix `[0, 4,523,024,384)`, ending before the first expert payload. This includes source gaps and read-ahead omitted by the earlier per-payload advice. The final charge was **32,538,624 bytes**, below the unchanged **284,839,936-byte** recovery threshold; own swap and memory/PID events were zero. This establishes cleanup for that allocation-only run. It does not identify every historical residual page, retroactively pass V2, or reclaim source pages during live inference.

The metadata-fix successor now passes the full **43-test** suite and separate actual A/B READY-only captures on one qualified binary. Its sealed repair changes only the Session metadata condition so that full shapes are loaded for repack **or** source reclaim; the 17 allocation/graph source hashes are unchanged. The shared experimental workspace advanced with preserved rollback, making the old `508d61b7…` full-engine receipt historical. [Current identities and evidence](../benchmarks/results/flash-reclaim-meta66-readiness.json).

Reviewed prospective admission is limited to the fixed 156-token prompt, through the full 512-token generation cap plus EOS. It includes cache/staging, plain and packed destinations, context, graph/CPU work, PLE, checkpoints, IO, stacks/tables, logs and transients:

| Bytes | A: repack OFF | B: repack ON |
|---|---:|---:|
| Packed destinations | 0 | 3,648,660,480 |
| Ordinary destinations | 4,522,949,120 | 874,288,640 |
| Complete prospective total | 22,743,365,408 | 22,744,863,520 |
| Margin below 21.5 GiB | 342,083,808 | 340,585,696 |
| Combined READY residual ceiling | 417,398,784 | 418,896,896 |

The full 256 MiB engineering contingency remains; it is not a formal allocator maximum. A single 64 MiB shared READY drift is priced once and constrained by both individual and combined ceilings. The live 2 GiB reserve, 32 MiB owned-log/output cap, power, swap/resource and recovery gates remain active. No source-RSS or packed-replacement credit is taken, and no complete protected task is admitted by this fixed-screen budget.

The launcher deadline was tightened from a refused 2,520 seconds to 2,500 seconds under the unchanged gate. The first parity attempt failed in its harness; the corrected original control and subsequent complete ABBA now pass. Failed receipts remain preserved. The ABBA unit peaked at **20,594,667,520 bytes**, with zero memory/PID events, RAM recovery and lease release; the exact prospective budgets and caps above remain unchanged. Final combined protected-task and installed lifecycle qualification remain mandatory before promotion.
