# Flash MTP acceptance: promising, awaiting numerical confirmation

**Gate amendment from the owner, 2026-11-10:** bit identity is a bug detector, not an unconditional gate for numeric changes. Compare verify-versus-single-token KLD with the production engine's own batched-versus-single noise on the same model and positions, and require the task checks to pass. Claude's source diagnosis points to single-query flash attention splitting KV at 512 positions while multi-query verification uses a sequential path. Hardware confirmation over positions 100–350 is pending; about 1e-3 is a guide, not a substituted measured threshold. The cold-load and next-turn KV-reuse issues remain separate. The observations below retain the initial stop and do not yet qualify a gain.

2026-11-10. The merged-head/exact-verification branch builds as a static CPU binary (`5e920099…`), with repack and KleidiAI off. Merge, config, parameter, mapping-release, file-reader and three synthetic streaming gate suites pass. The Q4_K_M head's complete SHA256 is `8087dbb39fc73f79ac069b1debe4069a44b73ada900c8bd761569d43a5a90231`.

| Fixed-code workload | MTP off | Depth 2, `--spec-exact` |
|---|---:|---:|
| Emitted tokens | 353 | 358 |
| Original answer identity | Pass | **Fail** |
| Independent Python code checks | Pass | Pass |
| Position 1 proposed / accepted | — | 128 / 121 (94.53%) |
| Position 2 proposed / accepted | — | 128 / 110 (85.94%) |
| Draft tokens accepted, all positions | — | 231 / 256 |
| Target verify calls | 353 | 128 |
| Request seconds | 94.23 | 77.74 |
| Prefill seconds | 11.55 | 12.42 |
| Effective decode tok/s, including draft overhead | 4.27 | 5.48, **unqualified** |
| Cold load seconds | 53.18 | 53.16 |
| Own swap / OOM kill | 0 / 0 | 0 / 0 |

Acceptance by position is **prefix survival divided by proposals at that position**, not an independent or conditional per-position probability. Counts include an accepted EOS: 128 target calls plus 231 accepted drafts confirm 359 tokens, of which 358 are emitted. This is about 2.80 confirmed tokens per verify. The numbers are preliminary because the exact output diverges.

**No speedup is qualified.** OFF reproduces the original 353-token answer SHA `460169e2…0a4093`. Depth 2 gives `cf550de9…e6fab8`; its first text difference is character 1139, a changed docstring near the end. The initial strict-identity screen stopped here. Under the amended rule, a measured benign numerical difference may pass; a real state or length bug may not. Other draft depths and the realistic-code workload stopped at this first identity failure, preserving the stop rule. A faster different trajectory is not an equal-work or full-agent win.

Both arms use the same frozen executable, original target weights, greedy temperature 0, seed 42, decode 4 / batch 6, IO8, 12 GiB expert cache, F16 KV, 16K context and the existing memory/power guard under the inference lease. The test scope is capped at 28 GiB with model swap disabled. Loader file-cache pressure touches the hard cap, but there is no OOM or own swap. The roughly 53-second cold load is reported separately and is not a fast-start result. The historic 11-second load and M0's prototype rates are different conditions, not substituted controls.

Claude's offline lane is investigating verify-batch exactness (including HC-expanded columns), recurrent/QSA/PLE rollback and final KV accounting. The MTP summary reports `n_past=156` versus 509 in OFF, which is another diagnostic to check, not an established root cause. Resume depths 1/2/3 and the realistic workload after the KLD/noise confirmation and any real accounting fix, retaining output hashes as diagnostics rather than enforcing identical wording. Meanwhile dense-Q4 artifact preparation proceeds. No production/model selection changes.

[Numeric observations, build identities and resource records](../benchmarks/results/flash-mtp-acceptance-20261110.json), [small counter patch and harness](../experiments/flash-mtp-20261110/README.md), [IO/cache experiments](FLASH-M0-TUNING.md).

## Corrected-branch checkpoint before user pause

Same-window reanalysis over 190 common-history rows (k100..289) gives verify KLD 0.000291699 versus production batch/single 0.000581669, passing the amended gate. Corrected branch da35f88/fork bdbf68e plus observational counters is built. Fresh no-dump OFF:4.439 effective tok/s,90.753s request; depth1:5.264 effective tok/s,80.277s request,174/184 drafts accepted (94.57%), code oracle pass. These are single fixed-code measurements, not a full-agent or deployed speedup. Depth2 was cleanly aborted at the owner’s pause request; depth3 and the complete LRU-code workload have not run. [Saved partial receipts](../benchmarks/results/flash-mtp-fixed-partial-20261110.json).
