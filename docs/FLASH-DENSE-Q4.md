# Flash dense-Q4: verified artifacts, substantial measured quality shift

2026-11-10. Two separate variant-A artifacts were converted from the original four-shard UD-IQ3_XXS Flash. All 144 expert tensors and the N-gram table retain their exact original payload hashes. Typed metadata, tensor names, shapes, shard membership and every other unchanged payload passed independent checks. Original files are untouched.

| Variant | Dense tensors changed | Total file bytes | Estimated non-expert bytes/token |
|---|---:|---:|---:|
| Original | 0 | 83,626,438,426 | 4.001 GB |
| Q4_K-compatible dense tensors; output head stays Q6_K | 352 | 82,546,379,552 | 3.085 GB (−22.9%) |
| Same, output head also Q4_K | 353 | 82,382,488,352 | 2.921 GB (−27.0%) |

The per-token estimate counts each dense matrix once and one token-embedding row. It excludes routed expert traffic, the sparse N-gram gathers, KV/state/activations, repeated accesses and backend expansion. It is not measured DRAM traffic, runtime RAM, or a speed claim. The model remains a mixed expert quant; it is not a uniform Q4 model.

The output head alone changes 521,472,000 → 357,580,800 bytes. Keeping it in the first variant isolates that quality sensitivity. Another 145 Q8_0 dense tensors have 320/640-element rows incompatible with 256-element Q4_K blocks. They remain Q8_0. Only if variant A proves insufficient will a separate variant B try IQ4_NL for those rows, as the owner directed.

Both conversions use the exact anchored type maps and the previously tested converter-only imatrix correction: unchanged IQ experts are copied, never requantized without importance data. A dry-run precedes conversion. The complete original verification was reused only after every file's device/inode/size/mtime/ctime still matched; candidate files were independently read in full, metadata fingerprinted, every unchanged tensor hashed, and all four files plus their directory synced. The six padding bytes added to the metadata-only first shard explain the difference between payload savings and total file savings.

The original and candidates are kept side by side under the model store, but neither candidate is selectable or installed. Conversion/checking used the canonical lease, a 6 GiB/no-swap scope, CPU200% and the existing memory/power guard. Both complete successfully with no OOM/own swap. Approximately 304/324 seconds cover artifact construction and verification, not inference.

**Completed quality screen:** 2,012 scored Wikitext positions (4 × 512-token chunks). Head Q6: mean KLD 0.1255096, top-1 agreement 89.36%, PPL 2.71264 → 2.86157. Head Q4: mean KLD 0.1357157, top-1 88.27%, PPL 2.89115. Both shifts are substantial; neither blanket variant is qualified. No imatrix was used. [Numeric quality results](../benchmarks/results/flash-dense-q4-quality-20261110.json). Pending after the user-requested pause: one-group sensitivity, a selected ~0.6–0.8 GB-saving variant targeting KLD ≤~0.03, imatrix calibration on code+text, and independent code/hard-task checks. Changed wording/token counts are expected for a lossy quant and must be reported; task correctness is still required. The engine arithmetic/noise rule applies to numeric implementation changes; a deliberately lossy Flash-fast model also needs the separately approved, measured hard-task quality/speed tradeoff. No quality pass or speedup follows from an integrity pass. The full-quality original remains the reference.

[File hashes, type-map/converter identity, resources and byte estimates](../benchmarks/results/flash-dense-q4-artifacts-20261110.json), [conversion/checking recipe](../experiments/flash-fast-20261110/README.md), [MTP numerical confirmation](FLASH-MTP-ACCEPTANCE.md).
