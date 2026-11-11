# Prompt reuse: measured evidence before tuning

Current checkpoint: 2026-11-09, memlat boot. These are our synthetic protected normal13 tasks, not the owner's history. No user prompts or images were reviewed. The direct native-server pairs already provide per-turn input/cache counters; the controller path still needs qualification.

| Model | New input | First-turn new input | Cached reads across later turns | Output | Turns |
|---|---:|---:|---:|---:|---:|
|Qwen MoE|6,122|5,540|50,028|849|9|
|Ornith Compact|6,066|5,540|35,825|479|7|
|Dense Qwen27B|5,983|5,540|29,818|651|6|

Each baseline/candidate pair has identical per-turn counters. SDK `input` is uncached input; cache reads are summed reuse across turns, not unique context length. The first turn accounts for90.5%,91.3% and 92.6% of all new input respectively. Every subsequent turn reports cached input. [Token-only records](../benchmarks/roofline/production-candidate-token-usage.jsonl).

The matching server source sets `stats.n_prompt_cached = n_past` **after** checking whether recurrent/SWA checkpoints can restore the common prefix. A forced full reprocess sets `n_past = 0` first. It then accumulates `n_prompt_processed` for actual evaluated prompt tokens. These are not simply an optimistic common-prefix estimate made before a failed hybrid restore. Source file `tools/server/server-context.cpp` SHA256: `e0e43f22fd2357a1e76ad6b8996b099d62615dd973de33feaa02fa0ee1a2a973` in both archived public-source builds checked locally.

These runs do not support an estimate of 45K **new** tokens. The earlier dense figure of 378 seconds at about 17 tok/s corresponds to about 6.4K tokens; applying a different model's current prefill rate to that elapsed time gives a misleading token estimate. This does not establish that every real-world session caches correctly, especially after compaction, unrelated requests or a controller-driven slot restore.

## Next observation

Fold the diagnostic into staged-controller/OpenCode qualification rather than repeat an otherwise identical standalone task. Capture only whitelisted numeric `timings`/usage fields from native SSE responses and prefix-operation durations; do not log response text, tool arguments or requests. Record new/cached tokens and prompt/decode milliseconds per turn, prefix prime/restore counts, and the cold versus restored-session time. Observe cleanup/guard/lease/RAM separately. Keep engine flags unchanged for this diagnostic.

Only after those observations should we change cache RAM, checkpoint spacing, controller prefix handling or OpenCode compaction/tool policy. These remain hypotheses, not proven fixes. Cross-session prefix reuse is valuable because the first request dominates new input, even when within-session reuse already works. Controller cache behavior is not implicated by tasks which bypass that controller.

Claude subsequently confirmed the 45K estimate was a rate-conversion error and retracted the direct-harness cache-miss theory. Its cache-keep work now targets the real controller path only. Dense raw prefill remains a relevant bottleneck; default agent work remains on the MoE.
