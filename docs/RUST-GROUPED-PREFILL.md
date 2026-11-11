# Grouped prefill: partial screen

2026-11-10. Frozen baseline `b477aec` CLI `b8795fce31a4…` versus group-2 candidate `725e751` CLI `89ae97aff7e3…` (checkout `8f1916c` changes docs only). Both precede tiled attention: these results do not describe the newer integrated main or a combined speedup.

Twelve CPUs 0–11, ubatch 256, context capacity 8192, deterministic 2048-token prompt, three timed repetitions after one warmup, same memlat boot. The existing inference lease and 28 GiB/model-swap-disabled guards cover each native arm.

| Model | Baseline tok/s | Group 2 tok/s | Change |
|---|---:|---:|---:|
| Ornith |202.31|207.46|+2.5%|
| Qwen |197.88|209.25|+5.7%|

All four arms pass, resulting first tokens match, and no model swap/OOM is observed. This timing check does not repeat the source branch's full logit/state oracle gates. The Ornith difference is small relative to run variation. No group-4 arm, full protected task, controller path or production promotion is claimed.

**Pending:** 4096-token screen. Work paused at an atomic boundary after the Qwen 2K pair so the higher-priority integrated-server gates could acquire the lease. Retain both pairs; do not rerun them just to resume 4K. If group 2 is later combined with tiled attention, measure that combination rather than multiplying these ratios.

[Paired summary](../benchmarks/results/rust-grouped-prefill-2k.json), [numeric receipts](../benchmarks/roofline/rust-grouped-prefill-screen.jsonl). Reproduce using `oxe prefill-bench --model MODEL --cpu-list 0-11 --ctx 8192 --prompt-len 2048 --ubatch 256 --reps 3`; add `--group 2` only to the candidate. The fixed source/model identities and exact timing samples are in the JSON.
