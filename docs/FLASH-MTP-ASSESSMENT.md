# Flash MTP: head integrity verified, execution not qualified

Source/header review, completed fetch and constructor allocation evidence, updated 2026-11-09. The exact head now has full-file integrity proof and 34 verified descriptors. No MTP inference or speed measurement was made. MTP remains disabled and unadmitted.

## Latest execution checkpoint

The cache8 experimental build now passes all 46 native tests and an actual head-only load. All 34 loaded tensor predicates pass. Four F32 normalization tensors have flat `[10240,1,1,1]` GGUF descriptors but are intentionally reshaped by the pinned loader to `[2560,4,1,1]`; the strict runtime contract now requires that exact loaded shape for those four names. Original file descriptors, byte counts, raw CPU placement, all 512 experts and native top-10 routing remain checked. This repairs a runtime contract mismatch, not model weights or inference arithmetic.

Two subsequent fresh NONE startup observations both stopped on the unchanged global-pageout guard before READY or generation. Neither hit its own swap, memory-limit or PID-limit gates; both reaped the child, recovered RAM and released the lease. High host swappiness (150) and file-cache growth during loading are possible contributors, not established causes. The next bounded change is to review the already tested early expert mmap RANDOM advice for this cache8 configuration. No further unchanged retry, global sysctl change or guard relaxation is scheduled. [Exact scoped results](../benchmarks/results/flash-mtp-loaded-head-startup.json).

This experimental profile uses batch capacity 4,096, with both the current and retained previous hidden-output allocations counted (336 MiB combined). The prospective fixed-screen NONE bound is 18,364,032,800 bytes, including 32 MiB of future trace growth. A fresh successful same-build startup and a complete actual MTP ledger are still required before generation. These figures do not admit a full-agent workload or a larger prompt. The earlier 12 GiB whole-head scenarios below remain historical; they are not the current cache8 allocation bound.

| Prerequisite | Evidence / result |
|---|---|
| Exact standalone head | Unsloth `MTP/mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf`, revision `084c631ac221016019d6e0a863afe92d90edd3b0`, 2,786,204,800 bytes; expected SHA256 `8087dbb39fc73f79ac069b1debe4069a44b73ada900c8bd761569d43a5a90231` |
| Necessary structural compatibility | Saved descriptors pass 34 required tensors, architecture parameters and tokenizer hashes against the installed target. This is not a CPU execution/rollback proof |
| Trained head in existing target | Absent from the four inspected target headers; changing metadata cannot create it |
| Full head payload identity | Verified complete: 2,786,204,800 bytes and exact SHA256 `8087dbb39fc73f79ac069b1debe4069a44b73ada900c8bd761569d43a5a90231`; the 34 descriptors and saved-header comparison also passed |
| Runtime and memory | Separate head context, prompt processing and target rollback allocations require new qualification within the same 21.5 GiB admission and 2 GiB actual host reserve |

[Exact publisher artifact](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/resolve/084c631ac221016019d6e0a863afe92d90edd3b0/MTP/mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf). Other Qwen heads and smaller shared variants cannot inherit this structural result.

The current demand-mapping patch applies to head loads too. The older assumption that loading the head necessarily eagerly populates its entire mmap is stale. Conversely, the head has its own context and prompt/catch-up work; it is not free shared KV or part of the 12 GiB target expert cache. See the [reconstructible source chain](FLASH-REBUILD.md).

The first fetch failed at the unchanged hard memory gate: 120 `memory.max` events, with OOM/kills, PID-limit events and own swap all zero. It did not reach completed payload verification. Its peak and anonymous/file-memory breakdown were not saved, so the precise pressure attribution remains unproven. That failed receipt is preserved. V2 added a 640 MiB soft limit inside the same 1 GiB hard cap and checkpoint/full-SHA advice only for completed ranges of its owned file. It used no global cache drop; advice does not guarantee eviction. V2 completed with zero hard events or new pageout, a 106,094,592-byte cgroup peak, the exact full-file SHA and all 34 descriptors. These fetch results establish integrity, not MTP performance. [Both attempts and descriptor evidence](../benchmarks/results/flash-mtp-head-integrity.json).

A separate bounded integrity diagnostic hashed the head output tensor's 521,472,000 bytes and obtained SHA256 `26a6e4f162f6ff10641122c93c426efa343f84d73cde6e6b6dd16959d17ecedc`, equal to the target output's previously qualified digest. The target payload did not need to be reread. Its receipt is `45d0df457a48dbc669dc2b443bab4e49bfc2306395054b358d483581d8c95857`. This proves byte identity for that slice; no alias was implemented and alias credit remains zero. It is independent of native compilation, ABI and allocation qualification.

The separate [constructor allocation diagnostic](../benchmarks/results/flash-mtp-allocation-diagnostic.json) now passes for draft widths 1, 2 and 4. It uses the exact target and head, CPU MMAP without extra buffers, and actual nextn scheduler reservations. It performs no graph computation or token generation.

| Draft width | Target context bytes | Target compute reservation | Head context bytes | Head compute reservation |
|---|---:|---:|---:|---:|
| 1 | 739,393,536 | 1,382,418,208 | 41,943,040 | 34,971,296 |
| 2 | 857,432,064 | 1,382,418,208 | 41,943,040 | 34,971,296 |
| 4 | 1,093,509,120 | 1,382,418,208 | 41,943,040 | 34,971,296 |

All three children exited normally, passed the complete numeric report checks, and released memory with no own swap or memory events. Their small diagnostic RSS does **not** bound inference memory: scheduler allocations include untouched pages, while expert cache, dense working set, CPU work buffers, hidden-output vectors, driver copies, checkpoints and metadata require separate accounting. Mapped model wrapper lengths are virtual ranges, not resident charges. The earlier helper compilation API error and truncated-return-log harness failure remain recorded; neither was relabeled a pass.

A follow-up scalar CPU planner query passed for all three widths without graph computation. At six threads it reports **195,662,336 bytes of target prefill workspace**, plus **1,769,856 bytes for the separate head**. That CPU workspace sits outside the scheduler reservations above and can remain allocated after prefill. It must be charged in the inference ledger; the small decode-only query does not replace the larger persistent charge. The linked allocation record includes all ten shape/thread queries per width.

Fetch completion does not establish memory admission. Use the full head file as a conservative upper weight charge, and count separate draft context, prompt/catch-up work, hidden/logit staging, rollback/state copies, compute workspace and startup overlap alongside the 12 GiB expert cache inside 21.5 GiB and an actual 2 GiB host reserve. No alias, shared-KV or smaller-quantization credit is established. A complete allocation bound is still missing; physical impossibility is also unproven.

The conservative whole-head scenarios now total 23,261,343,680 bytes at draft 1, 23,379,382,208 at draft 2 and 23,615,459,264 at draft 4, before all remaining driver, checkpoint, PLE, metadata and transient costs. Each exceeds the 23,085,449,216-byte admission limit. That upper-charge scenario does not prove mandatory residency or hardware impossibility, but it does not admit the current configuration. No output sharing or smaller-batch saving is credited. Further MTP work needs a separately proved memory reduction; a 156-token prompt cannot validate a proposed change in splitting prompts above 4,096 tokens.

The next prerequisites are that allocation bound and a finite CPU ABI/state/rollback probe, released only if admitted. A short integrity/state probe can check first-turn, continuation, reset and rejected-tail restoration. The requested performance screen remains no speculation versus draft 2, then draft 4 only after a winner. Count head load, prompt/catch-up, verification and cleanup in complete time. Preserve real spec-enabled policy, all 512 experts/native top 10, original independent correctness checks and failures; do not relabel MTP as `spec=none` to pass an old validator. Promote only after complete protected-task improvement and lifecycle qualification.

Original structural assessment SHA256: `5195056bcd0cc1ebaeb80335b776e9385709f8ec1d23ce5d04c2df312320c6b3` (source/header review). New successful fetch receipt SHA256: `6077f5019ee4069290516acdfe55399e8ace695ab9f03bff7efa1b62ebc4b565`; preserved failed fetch SHA256: `6d1ee8f3be7811386cf117f0ae6978952d9c6457067eedd94ab31d0098705b17`. These are fetch and structural receipts; the separate constructor receipts above establish allocation only.
