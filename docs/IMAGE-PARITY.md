# Image arithmetic-parity gate, 2026-11-11

> **Attention quality follow-up:** F-IMG-ATTN-1 now passes the frozen noise-based image gate for seeds 42/43/44 with 32/32 chunks and f32 attention. See [Image attention](IMAGE-ATTENTION.md) and [portable evidence](../benchmarks/results/image-attention-fimgattn1-20261111.json). Image pipeline and Studio integration remain pending; the earlier parity and R3 evidence below is preserved unchanged.

The frozen abdb931 Rust reference still fails the whole-model gate: 1.7863% relative RMS error at seed 42, step 0, versus a 0.5% limit. The later R2 self-noise control passes its predeclared GA′ rule; image deployment remains unqualified. The original GA failure is retained below. [Numeric evidence](../benchmarks/results/image-parity-bisect-20261111.json).

The isolated sd.cpp stage-tap build preserves the input to single block 0 byte for byte. Teacher-forced block comparisons locate the first local-update threshold crossing in double block 1: 0.21398% against the 0.20% diagnostic threshold. Double blocks 2 and 3 also cross. All single blocks stay at or below 0.03728%; the final projection, fed its own reference input, differs by 0.000872%.

Embedding and time/modulation parity are around 1e-7–1e-6 relative RMS. This narrows further source review to the double-stream stack and propagation through it. It does not yet distinguish all remaining arithmetic noise from a port defect, and is not a fast-image quality pass.

An extra single-block 1 byte comparison initially stopped the wrapper, beyond the ticket's required block 0 check. Its difference is only 2.686e-6 relative RMS. The original receipt is retained, and the completed capture was reused for the Rust block comparison after correcting that extra stop condition. No sd.cpp capture was repeated. Both native arms completed without OOM or model swap. Their 80.35 s / 102.73 s correctness-run wall times are not speed comparisons.

## R2 self-noise controls

The [same frozen arithmetic](../benchmarks/results/image-parity-noise-20261111.json) reproduces the original 1.786299% parity error. Moving the input embeddings by at most one ULP changes the Rust output by **1.685859%**. Repeating sd.cpp with the **same binary** produces identical output values (0% relative difference). The earlier 1.266555% sd.cpp difference used the separate stage-tap build and is not same-binary run-to-run noise.

R2 passes its frozen rule: parity error 1.786299% is below twice the measured self-noise, **3.371719%**. The contrary-result branch does not apply because the Rust perturbation exceeds 0.3%. This establishes that the whole-forward difference lies within the predeclared sensitivity budget; it does not qualify the fast image path. Part B next checks full-image parity against the unchanged 40 dB gate before measuring the per-seed fast-image noise floor. These correctness runs make no speed comparison.

## Full-image parity remains blocked

The conditional [Part B seed-42 render](../benchmarks/results/image-parity-full-20261111.json) reaches **35.579846 dB** versus sd.cpp, below the unchanged **40 dB** end-to-end parity gate. The render completed without OOM, model swap or a guard failure. The script stopped there: the f32 noise-floor image, fast 64/256 image and seeds 43/44 were not run. Passing the step-0 sensitivity control therefore does not establish full-image parity or qualify deployment. The 493.93 s render is a correctness-run cost, not a matched performance measurement.

## R3 image-level noise gate

The [perturbed full-image control](../benchmarks/results/image-noise-fast-gate-20261111.json) gives **33.8120 dB** between parity images with and without the one-ULP input perturbation. The unperturbed parity-versus-sd.cpp score, 35.5798 dB, satisfies R3's frozen `PORT OK` branch (`P0 >= N_dB - 1`). The second parity-versus-sd.cpp sample is 32.6300 dB. This supersedes the estimated 40 dB port threshold as the current decision rule; the earlier failed result remains recorded.

The fast 64/256 seed-42 image reproduces **22.75075 dB**, below the frozen noise-based gate `min(30, N_dB - 6)` = **27.8120 dB**. It therefore fails. Seeds 43/44 were not run, and the fast image path remains unqualified. Both native arms completed without OOM or model swap. Next is the separate fast-attention fix; no image default changed.
