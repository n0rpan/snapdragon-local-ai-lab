# Image Studio: qualified oxi512 path

IMG-SHIP-1 qualified and installed the oxi path for **FLUX.2 klein Q8 at 512×512**, CPU-only with 12 threads. All five gates passed: completion, routing, end-to-end time, image quality, and no OOM/swap. The separate installed smoke passed on 2026-11-11. [Portable numeric results](../benchmarks/results/image-ship-20261111.json) contain the measurements and immutable artifact identities. The [public IMG-SHIP-1 receipt](../benchmarks/results/public/IMG-SHIP-1.json) records the exported qualification.

## Behavior

Image Studio's `auto` engine selects oxi for the qualified 512×512 Klein Q8 path. Other model/size combinations, including 1024 px, use sd.cpp. The `sdcpp` selector provides a soft fallback. An oxi fallback is visible through the job's engine note; qualification and the installed smoke had no such note.

The fast path uses chunks 32/32, f32 attention, f32 VAE convolution, sd.cpp initial noise and stored sigmas. Prompt conditioning is obtained through the pinned sd.cpp probe and can be reused. The installed configuration sets `oxi.threads=12` and `klein.threads=12`, with Klein flash attention off. Existing q4/q8 settings and the chat controller's profiles, engines and defaults are preserved.

## Measurements

Same-run Studio sequence: A42 B42 B43 A43 C44. A used sd.cpp; each B started with its conditioning cache cleared; C reused conditioning. These are end-to-end wall times, not isolated kernel rates. OS/file caches were not explicitly cleared.

| Path | Individual wall times | Reported result |
| --- | --- | --- |
| sd.cpp A | 75.394 s, 80.311 s | mean 77.8525 s |
| oxi cold conditioning B | 52.628 s, 50.435 s | mean 51.5315 s |
| oxi warm conditioning C | 45.944 s | one observation |

Cold B took 33.81% less wall time than A (B/A 0.6619). PSNR against retained sd.cpp references was 31.4667 dB at seed 42, 32.0777 dB at seed 43 and 34.0803 dB at seed 44, above the frozen 27.812003603374173 dB floor. The same-run sd.cpp A outputs were pixel-identical to their retained references.

Maximum sampled cgroup memory for A/B/C was 15.737 / 14.758 / 12.752 GiB. These charges include page cache and should not be interpreted as process RSS measurements. No OOM or swap occurred.

The installed new-prompt seed 7 smoke completed in 47.420708 s against a 66.99095 s limit (1.3×cold B mean). It produced 512×512 output on oxi, observed the canonical inference lease, and had a clear active MemoryGuard. Its cgroup peak was9, 116, 315, 648 bytes under the 22 GiB production limit; swap and OOM counters stayed zero. This smoke uses a different prompt and cache/cgroup history, so its memory and time do not form another paired comparison.

## Artifact pins and installation boundary

| Artifact | Source | SHA256 |
| --- | --- | --- |
| Studio | f4d0451 | `32a8eeb001ea2d8255b4004a4bc361f928a9abf826204c000188f59f1b20cad0` |
| oxi | 339f185 | `b15437d8396bdf486058dd83ae9f87ab1915bfddf1c048664cde083b6da77976` |
| sd.cpp conditioning probe | qualified probe build | `4a47a2659438a7f6677bfeec2832ea8d37521135ec72cfd354fd4179dd20c6d6` |

Installation uses immutable copies, a Studio-only systemd drop-in and JSON editing that preserves other image settings. CPU quota is 1200%, memory limit 22 GiB, swap limit zero, and the engine setting is auto. The chat controller, its CLI/UI, model profiles and defaults remain unchanged.

The guarded full rollback checks current component hashes and original backups before stopping Studio, then restores exact prior settings/drop-in presence and reloads/restarts. Immutable asset copies remain inert. This avoids deleting or replacing a later user's configuration blindly.

## Conditioning cache policy

The approved lab setting retains up to 16 conditioning entries in directories with mode 0700, accessible only to the owner. Conditioning data can encode prompt information and is private application state. Public installations keep persistence off by default. The lab opts in explicitly with `oxi.prompt_cache: true` in `images/runtime-settings.json`; without that opt-in, each job uses a private temporary cache that is removed afterwards. The 45.944 s warm observation requires a retained matching conditioning entry with persistence enabled. Prompt text, embeddings, output images, user paths and private receipt paths are omitted from the portable numeric result.
