# Rust long-context attention screen

2026-11-10. **Experimental; not installed.** One frozen CLI from `dcc6bda` (attention-only branch, based on `b477aec`), SHA256 `9933eba77ffed4bd57a55313d4b807bc97b9d072d45f3c841fda11736a0dfb6d`. `OXE_ATTN=legacy|tiled` selects the path in the same binary. This isolates attention; it does not time the later integrated server, grouped prefill, or the MTP seeding fix.

Yoga Slim 7x 83ED, kernel 7.3.0-40-omarchy, memlat boot. Decode uses eight packed CPUs 0–7; prefill uses all twelve. These placements differ from the preliminary spread-core indication. Same model hashes as [RUST-M3](RUST-M3.md), native-layout weights, f16 KV, no MTP. One heavy job under the inference lease, 28 GiB cap, model swap disabled.

## Results

All rows are fixed synthetic workloads. Decode is the median of 64 timed token steps after eight warmup steps, following batched prompt setup. The initial grid has one arm per mode/context. Prefill is a 2,048-token extension on a 6,144-token prefix: one warmup, then three timed repetitions per mode. These are throughput screens, not protected-task or controller results.

| Model | Workload | Legacy tok/s | Tiled tok/s | Change |
|---|---|---:|---:|---:|
| Ornith | Decode, 8K prompt | 37.48 | 41.55 | +10.8% |
| Qwen | Decode, 8K prompt | 28.68 | 30.16 | +5.2% |
| Ornith | Prefill, 6K + 2K | 126.23 | 164.00 | +29.9% |
| Qwen | Prefill, 6K + 2K | 126.32 | 166.57 | +31.9% |
| Ornith | Decode, 1K prompt | 43.66 | 44.52 | +1.9% |
| Qwen | Decode, 1K prompt | 32.82 | 28.87 | -12.0% |
| Ornith | Decode, 4K prompt | 40.64 | 43.37 | +6.7% |
| Qwen | Decode, 4K prompt | 31.17 | 28.73 | -7.8% |
| Ornith | Decode, 16K prompt | 30.53 | 37.30 | +22.2% |
| Qwen | Decode, 16K prompt | 25.43 | 28.93 | +13.8% |

## Short-context repeats

The initial Qwen 1K/4K losses did not reproduce. Consecutive same-model arms reduced model-switch residency effects; all initial samples remain recorded. Qwen 1K uses ABBA/BAAB, four runs per arm; Qwen 4K and Ornith 2K use two per arm. Qwen 2K adds one bounded warm ABBA after a cold first arm, giving four per arm; the cold sample is retained in the combined median.

| Model | Prompt | Runs/arm | Legacy median tok/s | Tiled median tok/s | Change |
|---|---:|---:|---:|---:|---:|
| Qwen | 1K | 4 | 32.73 | 33.15 | +1.3% |
| Qwen | 4K | 2 | 30.90 | 32.29 | +4.5% |
| Ornith | 2K | 2 | 43.06 | 44.45 | +3.2% |
| Qwen | 2K | 4 | 32.02 | 32.49 | +1.5% |

Qwen's first 2K arm incurred 17,249 whole-process major faults; later arms were mostly low-fault. These counters include load and prefill, so they do not isolate decode I/O. The original grid did not capture fault counts: file residency is a plausible confound, not a proven sole cause of its negative samples. A warm repacked-file mapping is not proof that the weights are resident. Repeated short-context results are flat or modestly positive; the alleged 12% barrier penalty is not established. Sampled cluster clocks in the follow-up remain 3.4176 GHz. Timing noise and normal desktop activity remain possible.

A **4K padded-KV cutoff** for tiled decode is a reasonable conservative starting policy, not a uniquely measured crossover: benefits below it are small, while repeated Qwen 4K is +4.5%. Prefill can remain tiled. The cutoff implementation is in the later integration branch and is **not tested by this forced legacy/tiled binary**. Prompt lengths are not exact timed KV lengths: the eight warmup steps and 256-position padding add to them.

## Correctness, resources, and limits

All 44 arms finish with no cgroup max/OOM events or model swap. Peak thermal reading is 98.5°C; largest process RSS high-water mark is 19.35 GiB. Every decode pair/repetition has identical generated token IDs; prefill pairs agree on the resulting first token. Full logit/state bit-exactness at 8K and shorter fixtures was checked separately in the Claude lane (see source `M3-ATTN.md`); this timing screen does not claim to repeat that full oracle gate. The initial unscoped harness launch was refused before model execution (`own_swap_limit_not_zero`); rerunning in the existing swap-disabled transient scope fixed the harness setup.

Long-prefix prefill rises from 126.23 to 164.00 tok/s on Ornith (+29.9%), and 126.32 to 166.57 on Qwen (+31.9%). That is about 3.7–3.9 seconds less for this 2K extension. At 16K decode, the single-pair gains are +22.2% and +13.8%. Do not multiply these ratios into the historical protected-task gains or present them as installed production speed.

Next: the merged attention/rewind-fix server needs its own light native-oracle/tool/protected-task qualification. Keep the currently qualified production package pinned until the owner approves a change. Grouped prefill remains lower priority.

## Reproduce

Use the exact frozen source/model identities, the existing inference lease and memory/power guard. `MODEL` is one of the two GGUF files listed in RUST-M3; `OXE` is the CLI built from `dcc6bda`.

```sh
# Run both MODE=legacy and MODE=tiled with the same binary and IDs.
OXE_ATTN="$MODE" "$OXE" bench --model "$MODEL" --cpu-list 0-7 \
  --prompt-len 8192 --batched-prompt --warmup 8 --tokens 64 --ctx 8704
OXE_ATTN="$MODE" "$OXE" prefill-bench --model "$MODEL" --cpu-list 0-11 \
  --prefix-len 6144 --prompt-len 2048 --ubatch 256 --reps 3 --ctx 8448
```

Decode prompt lengths: 1024, 2048, 4096, 8192, 16384; context capacity is prompt + 512. Alternate order and repeat short-context pairs after model switching. Keep cold samples and report whole-process faults separately from timed decode. The CLI generates deterministic synthetic token IDs; no real chat history is used.

[Initial grid](../benchmarks/results/rust-attention-fixed-screen.json), [repeated crossover screen](../benchmarks/results/rust-attention-crossover-repeats.json), [grid receipts](../benchmarks/roofline/rust-attention-screen.jsonl), [follow-up receipts](../benchmarks/roofline/rust-attention-followup.jsonl), [metadata-only summarizer](../benchmarks/roofline/summarize_attention.py).
