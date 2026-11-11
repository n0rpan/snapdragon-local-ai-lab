# Image attention precision, 2026-11-11

**The 32/32 chunk setting with f32 attention passes the image-quality gates for seeds 42, 43 and 44. The selected configuration has not been installed in the image pipeline or Studio; separate integration remains pending.** [Portable evidence](../benchmarks/results/image-attention-fimgattn1-20261111.json).

The F-IMG-ATTN-1 candidate is commit `ff4b489`, oxi `30bfecd3…`, and oxi-block `d2048ac3…`. It keeps int8 linears and selects float attention explicitly. The binary default remains int8 attention.

## Six saved-input block arms

The probe uses the same saved single block 0 input, 12 threads and 1,536 tokens. Errors below are **dimensionless relative RMSE against the sd.cpp-arithmetic-parity reference**. Delta is the error in the block update, `out - in`; attention, MLP and z retain their receipt metric names. All median timings are **seconds**, directly from `median_total` and `median_attention`, over seven repetitions per arm. The three precisions are interleaved within one process for each chunk setting.

| Chunks | Attention | Delta | Attention error | MLP error | z error | Median total s | Median attention s |
|---|---|---:|---:|---:|---:|---:|---:|
| 32/32 | int8 | 0.010990841 | 0.012501849 | 0.000248717 | 0.009072424 | 0.273428599 | 0.024826306 |
| 32/32 | f16 | 0.001590252 | 0.000353279 | 0.000248717 | 0.001370879 | 0.286238786 | 0.058264841 |
| 32/32 | f32 | 0.001219179 | 0.000140658 | 0.000248717 | 0.001064160 | 0.283345475 | 0.057721085 |
| 64/256 | int8 | 0.017122820 | 0.014214529 | 0.013488973 | 0.014712547 | 0.212997130 | 0.023667124 |
| 64/256 | f16 | 0.014056030 | 0.006914857 | 0.013488973 | 0.012302861 | 0.248351357 | 0.055834961 |
| 64/256 | f32 | 0.014059347 | 0.006914427 | 0.013488973 | 0.012304694 | 0.260164447 | 0.061165330 |

A0 sanity passed: every lean result matched its tapped result, and both int8 settings reproduced their earlier update errors against sd.cpp within the frozen tolerance. A1 passed: at 32/32, f32 attention error is 0.000140658 and update error 0.001219179, below 0.0025 and 0.0050 respectively. The int8 attention error was 0.012501849 at that setting.

The frozen choice rule selected **32/32/f32**: minimize update error, then choose the smallest median total among settings within 10% of that minimum. The f16/f32 update-error ratios were 1.30436 at 32/32 and 0.99976 at 64/256; these ratios were report-only. Per-block timings are indicative, not an image-speed comparison.

## Three-seed image quality

R3 measured an image-level one-ULP noise score of **33.81200360 dB**. The frozen quality gate is `min(30, N_dB - 6)` = **27.81200360 dB**. The selected 32/32/f32 setting used 512×512 images, four steps, 12 threads and each seed's saved handoff.

| Seed | PSNR vs sd.cpp dB | Gate dB | Native process wall s | Result |
|---|---:|---:|---:|---|
| 42 | 31.46671267 | 27.81200360 | 42.394 | PASS |
| 43 | 32.07772473 | 27.81200360 | 43.496 | PASS |
| 44 | 34.08030388 | 27.81200360 | 44.927 | PASS |

All three seeds passed. Seed 42 was also 32.62825579 dB versus the parity image. The later two seeds were more than 1 dB above the gate, so the frozen follow-up rule did not require extra seed-specific noise pairs.

The native walls **42.394/43.496/44.927 seconds** are correctness-run costs. They include native process overhead, while the smaller per-step/model timers in the JSON do not represent the whole process. The saved handoff skips live prompt text encoding. There is **no matched image-speed claim**, and these numbers are not an end-to-end Studio latency measurement.

## Guard state and remaining integration

All five native attention-candidate processes—the two block probes and three image renders—completed with clear memory and power guards in 8 GiB scopes, zero model swap, no OOM and no timeout. The two R3 reference arms also had clear guards; their earlier image-quality failure remains historical evidence. The portable JSON retains each scope peak, host minimum available memory, native wall and executable hash.

R3's former 64/256 int8 fast image scored 22.75074563 dB and failed the noise gate. The new attention setting passes that quality gate across all three seeds. This advances the qualified engine configuration; it does not install or qualify a live image pipeline or Studio flow. Separate integration must bind the selected setting and validate the complete prompt-to-image path.

The earlier [image parity record](IMAGE-PARITY.md), including its unmodified R3 section and previous failed gates, is retained.
