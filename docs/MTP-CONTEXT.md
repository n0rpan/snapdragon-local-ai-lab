# MTP context policy with VL15 — installed 2026-11-11

Both MoE profiles use Rust `bbc2990`, server SHA256 `921741ee568c2ba4063991e35fa272d7ce453938e41e117433c85ba56f1f6a7a`, VL15 and MTP depth 1 without `--mtp-max-ctx`. Ornith remains default and uses the grafted Qwen head; selectable Qwen3.6 uses its own head. Both allocate 32K with decode cores 0–7, prefill 0–11, ubatch 256, prefill mask 1021, eight checkpoints, 2 GiB parked-session budget and persistent session/prefix caches. MoE memory limits remain 28 GiB with model swap disabled.

The old Ornith 2K cutoff came from a VL0 screen. With VL15, it suppressed a useful MTP path on real OpenCode turns. We repeated the context decision on the installed binary, without rebuilding.

## Matched decode screen

One MTP-capable server per model, with a per-request depth override, restores the same prompt checkpoint before each arm. Depth 0 keeps the head loaded and follows its state, matching how a context cutoff works in production. A 16-token warm-up is discarded at each context; A/B/B/A then generate 256 tokens each. Every measured arm re-evaluates one prompt token. All token-id hashes match. Values below average the two arms at each depth.

| Model | Prompt tokens | Depth 0 ms/token | Depth 1 ms/token | Less decode time |
|---|---:|---:|---:|---:|
| Ornith | 4,096 | 22.416 | 17.602 | 21.48% |
| Ornith | 8,180 | 24.203 | 17.920 | 25.96% |
| Ornith | 15,980 | 27.354 | 22.610 | 17.34% |
| Ornith | 32,000 | 32.925 | 29.015 | 11.88% |
| Qwen | 2,048 | 30.183 | 24.413 | 19.11% |
| Qwen | 8,180 | 32.610 | 25.073 | 23.11% |
| Qwen | 15,980 | 35.245 | 28.859 | 18.12% |

The rule was fixed before testing: use the first context where depth 1 is more than 3% slower in **both** paired comparisons; if none, omit the cutoff. Neither model reached that condition. Contexts beyond the original 8K native fixture repeat its synthetic token corpus. Qwen was measured through 16K with a 32K allocation; this is not a measured 32K Qwen speed result. These two-arm-per-depth screens decide configuration and are not three-warm-run release headlines. No guard stop, model swap or OOM occurred.

## Final controller checks and installation

| Final profile | Matched fixed 2K request, median of 3 per arm | Protected task | Highest task RSS |
|---|---|---|---:|
| Ornith | 10.261 → 9.819 s (4.31% less time) | 13/13 twice, 54.50 / 31.99 s | 18.92 GiB |
| Qwen | 12.115 → 8.361 s (30.99% less time) | 13/13, 44.45 s | 22.66 GiB |

Ornith's control is the already-installed bbc2990 profile with the 2K cutoff; Qwen's control is its previously-installed 707a0ec profile. The Qwen request improvement therefore includes the candidate-2 engine/prefill changes, not MTP alone. All paired request token hashes match. The two Ornith task runs share one server; they are not the earlier two-fresh-session cold/warm protocol. The earlier Qwen MTP-off 22.01 GiB task peak remains recorded separately from the final MTP-on 22.66 GiB peak.

Persistent session reuse (2304/2304), prefill and decode cancellation, active-MTP cancellation and retry passed. Installed-unit capture and verification both start unloaded with disk caches retained; all six cases passed without changing timing thresholds. Native process SHA, argv, VL15, MTP depth 1 and absent cutoff were checked before unloading.

## Rollback and evidence

Each command uses a SHA-bound guard and preserves the other model profiles:

```sh
python3 ~/.local/share/ai-stack/rollbacks/latest-ornith-candidate2/rollback/rollback.py
python3 ~/.local/share/ai-stack/rollbacks/latest-qwen-candidate2/rollback/rollback.py
```

The Ornith rollback restores the previous 2K cutoff. The Qwen fallback restores its previous 707a0ec/16K profile and matching OpenCode limit. Flash has its own refreshed fallback plan. All rollback guards were checked after the final installation.

[Numeric screen and controller extract](../benchmarks/results/mtp-context-vl15-20261111.json). Private source receipts are cited there and in `runq/QC-CTX2.result.md`, `QC-CTX2-Q.result.md`, `G0prime.result.md` and `G2prime.result.md`. The clean full-session B/C/C/B replay remains pending; no session-wide percentage is inferred from decode-only gains.
