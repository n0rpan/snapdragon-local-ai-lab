# Flash dense-Q4 preparation

Experimental only; no model selection or production changes. `prepare.py --output /absolute/fresh/directory` reads GGUF headers and writes a complete anchored tensor-type map. It never loads model payloads. Input is the existing four-shard UD-IQ3_XXS model; its model/license/download identities are in [the header audit](../../benchmarks/results/flash-post-memlat-header-audit.json).

Variant A changes only 353 compatible non-expert Q6_K/Q8_0 tensors to Q4_K. All 144 routed expert payloads, the PLE/N-gram table, F32/BF16 tensors and 145 incompatible Q8_0 tensors remain unchanged. Those 145 have 320/640-element rows, which cannot use a 256-element Q4_K block. Their combined 421,273,600 bytes are about 34% of A's 1,243,950,080-byte storage saving. They stay unchanged for the first experiment. Only if A proves insufficient will variant B try IQ4_NL (32-element blocks) for them; Q4_0 is not the chosen exception.

The output head is a separate quality variable: `tensor-types-head-q6.txt` retains its original Q6_K, while `tensor-types.txt` converts it to Q4_K (521,472,000 → 357,580,800 bytes). Compare both against the same full-model reference and the same 2K teacher-forced corpus. Report mean KLD and top-1 agreement for each, not a fresh-BF16 quantization claim. Requantizing existing low-bit weights can lose quality.

The header traffic estimate reads each dense matrix once and only one token-embedding row. With the Q4 head, it is 4.001 → 2.921 GB/token, excluding expert traffic, KV/activations, repeated accesses and backend expansion. This is a byte model, not measured DRAM traffic or a speed prediction.

Use the [already tested converter-only importance-matrix correction](../flash-selective-q8/converter-imatrix.patch), which permits an unchanged low-bit tensor to be copied without importance data. Build `llama-quantize` from the pinned Flash fork. Dry-run first, inspect every planned type change, then use a fresh output prefix:

```sh
llama-quantize --allow-requantize --pure --keep-split --max-buffer-size 16 \
  --override-kv general.file_type=int:23 \
  --override-kv general.quantization_version=int:2 \
  --tensor-type-file tensor-types-head-q6.txt \
  ORIGINAL-00001-of-00004.gguf FRESH_OUTPUT_PREFIX Q4_K 2
```

Run conversion, hashing and evaluation serially under the inference lease. Keep the original model. Verify all descriptors and unchanged tensor payload hashes, including every expert. A successful converter exit alone is insufficient. Require the fixed353 code oracle, and report token/text differences rather than demanding the same answer from a lossy model. The standing numerical screen is mean KLD ≤0.01 and top-1 ≥97%; any separately proposed Flash-fast quality tradeoff additionally needs independent hard-task evidence. No fast profile is qualified by this preparation.

Both variant-A artifacts are now constructed and integrity-verified. [Exact hashes/bytes and pending quality work](../../docs/FLASH-DENSE-Q4.md). Use `convert.py --head q6|q4 --output /absolute/fresh/directory` under the existing leased scope/guard; it reuses the bound original verification only while all source identities still match.
