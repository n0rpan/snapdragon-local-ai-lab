# 768/1024 image screen: HOLD

One seed per resolution was screened on 2026-11-11, before the kernel update (kernel 40 epoch). Neither resolution qualified for deployment. Studio's installed 512 px VAE path remains in place; 768/1024 routing stays on sd.cpp.

The 768/1024 work is carried into v0.1.2. Both resolutions remain on HOLD until their quality gates pass; the observations below authorize no route expansion or release speed claim.

| Resolution | sd.cpp A, seconds | Oxi B, seconds | SD self-noise N, dB | Derived gate G, dB | Oxi vs SD P, dB | Decision |
|---|---:|---:|---:|---:|---:|---|
| 1024 | 369.5724 | 121.8908 | 37.5917 | 30.0000 | 27.0579 | MARGINAL_HOLD |
| 768 | 198.0234 | 63.3458 | 35.4068 | 29.4068 | 28.8446 | HOLD |

N is the RGB-normalized PSNR between the SD reference and its opt-in one-ULP control at the live first-block image/text inputs. The frozen resolution-specific gate is `G = min(30, N − 6)` dB. P is the Oxi result against that SD reference. Both P values missed their derived gate. The older 512 px gate was not reused.

These are **one-seed screens**, with no general speed claim. The 1024 A image and completed timing were reused from the preceding run, so that row is explicitly a **cross-run screen**. The 768 A and B were fresh observations in the later run. Neither row is an interleaved, repeated timing qualification. B covers the full cold path through PNG, including conditioning handover and the int8 VAE. The corrected adapter captured numeric SD diagnostics separately from progress output.

The guarded worker used a 22 GiB cap and zero model swap. Its cgroup peak was 18,387,447,808 bytes, host availability stayed at least 14,122,385,408 bytes, and scope OOM events and the global OOM delta were zero. Those safety observations do not turn the quality HOLD into a PASS.

These kernel 40 observations remain historical after the reboot to kernel 47. Future performance comparisons need a fresh warm-up and matched control in the new epoch. No routes, production configuration or model selection were expanded by this screen.

[Whitelisted numeric results](../benchmarks/results/image-large-screen-20261111.json) · [Installed 512 px VAE result](IMAGE-VAE.md) · [Results policy](RESULTS.md#2-method)
