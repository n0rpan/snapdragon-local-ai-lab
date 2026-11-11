**Installed on MEMLAT, 2026-11-10.** The measurements below are the pre-install qualification, retained unchanged. [Live verification and rollback](RUST-DEPLOYMENT.md).

# Integrated Rust server: qualification for promotion

2026-11-10. **Qualified candidate; not staged or installed.** Source `707a0ec` (documentation-only successor to build `3f7b35f`); server SHA256 `649ac80670e58d5a4806355543f67a8bce0ce6034cf63e380dbf0c2a3f3ccf69`. CLI SHA256 `10b298f70d3cf5b9ff3097f48c236aa7114ad57577666645e385ed6df6ca513d`. All changes after the earlier `789e8784…` candidate are integrated: tiled attention, MTP rewind/seeding fixes and prefix-cache tools. Both proposed profiles use **MTP off**, `OXE_ATTN_TILED_MIN_KV=0`, decode CPUs 0–7, prefill 0–11, context 16K, ubatch 256 and four checkpoints.

## Upgrade isolated against the earlier Rust build

Three trials per arm/model, same boot, same three native warmups followed by a fresh native process per task. Persistent prefix stores are off here. Every arm matches 128 greedy tokens on each of three private split-last native llama.cpp references, passes the same 13 independent protected-task checks and real OpenCode tools, and unloads cleanly.

| Model | Rust 789e median | Integrated 649a median | Less task time |
|---|---:|---:|---:|
| Ornith | 60.06 s | 52.13 s | 13.2% |
| Qwen | 65.53 s | 57.91 s | 11.6% |

Token counts match across the isolated trials: Ornith 6,086 new prompt / 661 output tokens, Qwen 6,184 / 670. The candidate threshold is verified in the actual process environment; executable hashes and command lines are observed. One initial lease race was refused before inference, retained separately and excluded. Other hardware jobs were paused for the measured suite.

## Fair production comparison: cold with cold, warm with warm

Each engine/model uses an isolated prefix store. Session one populates it; session two runs in a **fresh native process**. Same three short model warmups before each timed task. Thus “cold” means prefix-cold, not SSD-cold weights. One task sample per engine/state/model; these are not three-trial medians.

| Model | Production CPP, prefix cold | Rust, prefix cold | Production CPP, prefix warm | Rust, prefix warm |
|---|---:|---:|---:|---:|
| Ornith | 64.59 s | 52.83 s | 30.24 s | 23.42 s |
| Qwen | 112.90 s | 59.21 s | 70.54 s | 29.33 s |

Controls use the **actual installed resolved profiles**, with experimental aliases and fresh cache directories: Ornith CPP `acd47ff1`, MTP off; Qwen fallback CPP `1bd3e439`, MTP depth 2. Rust is MTP off. The Qwen control lacks the final patched persistent-repack path and takes ~35–37 s to start. It is deliberately different from the earlier shared-patched-engine research control; do not splice their ratios or attribute the whole gain solely to attention. All four CPP sessions pass 13/13, with one cold prime and one warm restore per model. Native/KleidiAI trajectories differ: CPP output counts are Ornith 522/522 and Qwen 684/555; Rust 661/661 and 670/670. Finished-task time, not seconds/output-token, is the comparison.

## Prefix and memory

Rust restores **5,376 tokens** on each second session, processing only 168 new tokens on the first request. First-request prompt processing falls from **30.64 → 1.22 s Ornith** and **31.25 → 1.22 s Qwen**. First useful SSE delta follows at ~1.23 s; this backend observation excludes controller/model startup. Prefix-cache cold/warm request-body hashes match. Prefix files remain local, owner-checked, directory 0700/files 0600; no prefix payload is exported or backed up.

For CPP, cold priming happens **before** the visible request, so its small native `prompt_ms` alone is misleading. Prefix preparation plus first native prompt work is Ornith **34.92 → 1.11 s**, Qwen **39.81 → 2.90 s**, excluding model startup. Whole task times above include startup, preparation and tools. The serial sum of measured startup + preparation + backend first-useful-delta stages is warm Ornith 4.14 → 1.49 s and warm Qwen 38.30 → 1.49 s. This is a component estimate of controller first response, not an independently timed client-wire TTFT. Ornith's warm prompt-work component alone is slightly higher in Rust (1.22 versus 1.11 s); the first-response sum and whole task improve. Primer token counts are not separately captured; the preparation wall timer includes primer/save or restore.

| Model | CPP peak task RSS | Rust peak task RSS | CPP prefix disk | Rust prefix disk |
|---|---:|---:|---:|---:|
| Ornith | 16.22 GiB | 15.90 GiB | 358.96 MiB | 167.83 MiB |
| Qwen | 27.76 GiB | 20.89 GiB | 369.87 MiB | 167.83 MiB |

RSS is sampled during the timed prefix sessions, not a claim about every possible load/context. Component RSS peaks occur at different instants and must not be added together. Cgroup memory can omit shared file pages charged elsewhere. Across all 20 measured arms: zero model swap, zero cgroup OOM kills, no kernel OOM/panic in the experiment window, native reaped, guards/readers joined and lease released. Host swap remains permitted. One repair task does not establish broad model quality.

## Package and rollback

`claude/oxe-production` now pins this server, threshold 0 and `--prefix-cache default` for both aliases; all **302 package Python tests pass**, plus pure resolution against the real runtime. The external llama.cpp `/slots` PrefixStore is disabled only in Rust mode; Rust owns its persistent store. Stock boot and flag-off classic paths remain available. Model identities and the 28 GiB/model-swap-disabled policy are unchanged. **No stage directory, live config, service restart or default change has been made.** the owner's approval remains required. Existing one-command classic rollback is retained.

[Three-trial metadata and timings](../benchmarks/results/rust-integrated-upgrade-qualification.json), [production prefix fairness](../benchmarks/results/production-prefix-fairness.json), [numeric receipts](../benchmarks/roofline/rust-integrated-controller.jsonl), [upgrade summarizer](../benchmarks/roofline/summarize_integrated_upgrade.py), [fairness summarizer](../benchmarks/roofline/summarize_prefix_fairness.py). Earlier results remain historical in [the original qualification](RUST-SERVER-QUALIFICATION.md).
