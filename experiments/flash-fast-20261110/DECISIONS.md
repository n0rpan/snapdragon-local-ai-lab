# Experiment decisions before conversion

1. Read-only header plan: 1,224 tensors, 144 expert tensors. Variant A converts 353 compatible dense tensors; 145 row-incompatible Q8_0 tensors stay unchanged. The first planner assertion rejected their incompatible shapes before any payload was read or written; the failed directory is retained.
2. Run A with the original Q6_K output head first, then A with the head changed to Q4_K. This isolates output-head sensitivity. Both use the same fixed 2K-token corpus and original full Flash reference.
3. Only consider IQ4_NL for the remaining 145 tensors if A's measured improvement is insufficient. Their 421.3 MB is not a license to silently fall back to Q4_0.
4. Reuse the converter-only imatrix correction already validated by the earlier selective-Q8 experiment. Record its patch and binary identity. No importance data is fabricated; changing an IQ expert still requires valid importance data.
5. Require a complete dry-run type map, then verify unchanged payloads and metadata independently. Preserve originals. Numerical quality and completed-task quality remain separate measurements; no profile promotion follows automatically.
