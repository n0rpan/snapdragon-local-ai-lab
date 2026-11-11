# Selective Flash dense-weight Q8 experiment

**Artifact integrity and one fixed-code ABBA screen passed; broader quality and full-task speed remain unqualified.** This candidate changes 250 Q6_K non-expert tensors to Q8_0. The other 974 payloads, including all 144 expert tensors and the PLE table, remain byte-identical. This is an additional numerical model variant, not a uniform Q8 model or a lossless optimization. It is not installed or selectable.

The preregistered A/B/B/A screen retained the original independent code tests and exact original controls. Both Q8 repeats produced the same correct 338-token answer; both original controls produced the original 353-token answer. Mean prefill fell **11.661 → 10.196 s (12.56%)** for the same 156 input tokens. Weighted decode fell **3.878 → 3.787 tok/s (2.35%)**. Request time fell 3.17%, but the changed answers and output lengths prevent an equal-work generation or finished-task speedup claim. Keep Q8 as a prefill candidate pending broader quality and the final protected task. [Complete observations and resource checks](../../benchmarks/results/flash-selective-q8-screen.json).

Both conditions used the same binary, CPU 1200%, decode 4/batch 6, IO8, cache 12 GiB, context 16K, F16 KV and microbatch 4096. CPU repacking was compiled **off** in this Flash binary, so this result establishes no I8MM repack-dispatch gain. The entire four-arm unit peaked at 19.81 GiB, below the 21.5 GiB admission gate, with zero own swap, hard memory or PID events. EOF, normal exit, reaping, RAM recovery and lease release passed in every arm. The changed tensors add 0.915 GiB on disk; the unit peak includes file cache and is not a measurement of additional resident model memory.

[Original and candidate file hashes, source revision and verification](../../benchmarks/results/flash-selective-q8-artifact.json). The original metadata-only first shard is intentional. Canonical output adds six padding bytes there; tensor payloads grow 982,635,520 bytes, approximately 0.915 GiB. Total output is 84,609,073,952 bytes. Keep the original model for rollback and allow room for both copies.

The converter is the pinned Flash llama subtree at `04ff7dfdfceee3e34b2d76372db1bc286b8fc179`, with [one converter-only correction](converter-imatrix.patch). That correction requires importance data only when a tensor's type actually changes. It permits untouched IQ2_S/IQ3_XXS expert payloads to reach the existing copy path; changing either low-bit type still refuses missing importance data. [Native tiny-file checks](../../benchmarks/results/flash-selective-converter-native-checks.json) verify that behavior. The patch does not change quantization math.

For an isolated CPU-only build of that exact subtree, apply the patch and build only `llama-quantize`, following the [pinned source/build prerequisites](../../docs/FLASH-REBUILD.md). The full transformation uses the complete anchored [1,224-tensor map](tensor-types.txt):

```sh
llama-quantize --allow-requantize --pure --keep-split --max-buffer-size 16 \
  --override-kv general.file_type=int:23 \
  --override-kv general.quantization_version=int:2 \
  --tensor-type-file tensor-types.txt \
  ORIGINAL-00001-of-00004.gguf FRESH_OUTPUT_PREFIX Q8_0 2
```

Replace the input with the exact first original shard from the hash record and use a fresh output prefix. File type 23 preserves the original overall IQ3_XXS declaration; most model bytes still belong to unchanged experts. `COPY` is unsuitable because it suppresses the intended type changes.

Our guarded run used CPU200%, memory.high3 GiB/max4 GiB, swap0, one inference lease, external power, the battery cutoff and actual host reserve. It flushed all four outputs and their directory, independently checked every typed metadata value/descriptor and all unchanged payload hashes, and collected new full-file hashes. The earlier power-stopped attempt remains [recorded](../../benchmarks/results/flash-selective-conversion-power-stop.json). A successful converter exit alone does not establish those checks or authorize deployment. Follow the experiment plan before running quality/performance comparisons.

Only the map and patch are included here. Original llama.cpp remains MIT-licensed; model weights and their licenses remain separate. This experimental recipe does not claim finished bootstrap support.
