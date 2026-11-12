# Flash cache, queue depth and IO placement

2026-11-10. These are **timing-prototype screens**, not correct Flash inference. M0 uses real weights/kernels and recorded routes with dummy activations. Nothing here establishes a completed-task speedup or changes production.

| Change | Fixed replay result | Full protected task | Decision |
|---|---|---|---|
| Cache 16 → 18 → 20 GiB | 122.14 → 105.17 → 96.02 ms/token; one screen per size | Not run; prototype cannot generate a valid answer | 20 GiB is the largest fully verified cache in this wave |
| Cache 22 GiB | Internal footprint refusal at the original 26 GiB estimate; explicit 28 GiB admission then hit the host floor | Not run | Stopped, not a timing result |
| Cache 22 GiB after desktop reclaim | Host available fell to 1.26 GiB while loading; guard stopped it | Not run | 23/24 GiB arms skipped, as required |
| QD32 → QD4 | Fresh ABBA 95.068 → 94.876 ms/token, −0.20% elapsed | Not run | No reliable gain; retain QD32 |
| kcryptd mask `fff` → `e00` | ABBA 95.448 → 96.041 ms/token, +0.62% elapsed | Not run | No gain; original mask restored |
| Spread+IO3 → packed+IO8 | 97.45 → 95.60 ms/token in the isolated screen | Not run | Preliminary only; no promoted placement change |

Cache screens use the original frozen `f3ff4781…` executable. The QD/crypto confirmation uses isolated `4f13026f…`, rebuilt from the same clean BF97 commit with seven tests passing. The shared M0 worktree had advanced to M1 and replaced its executable; the SHA preflight refused the mismatch before running. Exact-binary groups are kept separate. The apparent initial QD gain disappeared when the first QD32 control's 104.20 ms drift was replaced by a matched ABBA. Do not report that initial 6% figure as a win.

All isolated arms use a 20 GiB cache, native routing replay, eight compute cores and one IO core outside compute, with a 28 GiB hard scope, model swap disabled and the existing MemoryGuard/power checks. Peak sampled process RSS is about 24.46 GiB. Temporary workqueue affinity was applied under the inference lease and restored in `finally`, with per-arm restoration proofs. No persistent dm-crypt/workqueue or production RAM policy changed.

The larger-cache attempt proactively reclaimed only desktop app/session cgroups. Their combined anonymous resident memory fell from about 1.37 to 0.26 GiB, with about 0.28 GiB disk swap-out and 0.85 GiB extra available RAM. That still did not make 22 GiB safe under the standing 1.5 GiB host floor. There was no model swap or OOM kill. This is not proof that 20 GiB is the physical maximum: 21 GiB was not tested. The 29.5 GiB test scopes were never reached because the prerequisite 22 GiB run failed.

After release, a two-minute desktop recovery sample recorded 22.34 MiB swap-in, mean 0.186 MiB/s, peak one-second 6.94 MiB/s, and memory PSI full avg10 at zero. Five original compositor probes failed because the instance was unspecified, so their timings are excluded; a corrected `hyprctl -i 0 -j version` succeeded. No human interactive-responsiveness claim follows from that IPC check. Current swap is the NVMe file plus zswap; zram is disabled.

One completed QD confirmation arm failed its final guard assertion before the old harness serialized metadata. Its exact failure reason was not retained. It is excluded, the failure and native result remain on disk, and a fresh four-arm ABBA supplies the reported comparison. Receipt handling was corrected without relaxing the RAM/PSI/swap limits.

Next: measure merged-head MTP acceptance at depths 1/2/3 on the fixed353 and realistic-code workloads. Then evaluate the dense-only Q4 candidate, first retaining the Q6_K output head and then changing it, with expert payloads unchanged and independent KLD/top-1/code checks. The >20 tok/s serving target remains unproven.

[Numeric receipts and per-token timings](../benchmarks/results/flash-m0-tuning-20261110.json), helpers and reproduction notes (lab note omitted from this release), [original authoritative M0](FLASH-M0.md), dense-Q4 preparation (lab note omitted from this release).
