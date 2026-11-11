# Flash: opt-in 18 GiB cache

The test laptop now prefers 18 GiB on the secure v0.1.1 engine. Public defaults remain opt-in. Both native PSI pass-splitting thresholds are explicitly zero; the standard 1 GiB host floor, sustained-pressure checks, 29 GiB scope and zero-model-swap policy remain active. The F127 swap-tolerant guard and 20 GiB preference remain parked. Other models, engine binaries and the controller unit are unchanged.

A fixed cold-context request used 5,547 prompt tokens, zero cached tokens and one generated token in every arm:

| Run order | Cache | Prefill |
|---|---:|---:|
| 1 | 14 GiB | 92.2355 s |
| 2 | 18 GiB | 96.8423 s |
| 3 | 18 GiB | 90.3032 s |
| 4 | 14 GiB | 93.5347 s |

The first pair failed the 3% gate. A single reverse-order follow-up was declared before running it because the sampled second arm had 4–7% lower cluster frequencies. The first 14 GiB arm has only partial thermal coverage. The original failed pair is retained; no samples were discarded. Across all four samples, geometric means are **92.8829 s at 14 GiB** and **93.5156 s at 18 GiB**, a **0.6812% increase**, within the unchanged 3% gate. All four guard, token-count and no-swap/OOM checks passed. This is a non-regression qualification, not evidence of a speed gain; the earlier 2.47% decode difference also remains noise-level.

The installed live check confirmed actual tier 18, a successful chat, 401/403/400 security checks, standard guards, no model swap/OOM, key cleanup, unload and QoS reset. Two verifier-only errors (field name and startup readiness) were corrected; both safely restored secure 14 GiB before the final passing attempt. See the [numeric evidence](../benchmarks/results/public/o32-cache18.json).

One-command cache rollback: `~/.local/share/ai-stack/rollbacks/latest-flash-cache18/rollback-live.py`. It restores the secure 14 GiB runtime without changing the controller or engines. The older full-pair guard intentionally refuses the newer 18 GiB runtime until this rollback has restored its expected preimage.
