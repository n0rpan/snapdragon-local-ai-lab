# Ornith candidate 2 installed on 2026-11-11

Ornith uses Rust source `bbc2990`, server SHA256
`921741ee568c2ba4063991e35fa272d7ce453938e41e117433c85ba56f1f6a7a`.
The model is APEX Compact, with 32K context, eight decode CPUs, twelve prefill
CPUs, prefill mask 1021 and verify-lane mask 15. The Qwen MTP sidecar drafts one
token throughout the available context after the VL15 context retune. Sessions use eight checkpoints and a 2 GiB RAM budget.

The parser accepts Python True/False literals for typed boolean tool parameters
in lenient mode. The exact failing OpenCode edit and its streamed form pass the
pinned parser test. Both real OpenCode task runs passed all thirteen independent
checks, in 57.71 and 35.45 seconds on the same running server. Their captured
system prompts show Wed Nov 11 2026. Full request bodies stay in the private
local receipts. Date pinning is deferred as F104.

| Measurement | Before | After | Scope |
|---|---:|---:|---|
| Fixed 2K-token controller request, three per arm | 13.267 s | 10.164 s | Same boot, a219347 versus bbc2990, identical generated tokens, −23.4% time |
| Short-prompt MTP decode, VL0 versus VL15 | 23.387 ms/token | 18.384 ms/token | ABBA, mean of each pair, identical hashes, −21.4% time |
| 4K MTP decode, VL0 versus VL15 | 26.467 ms/token | 22.639 ms/token | Forced MTP-on screen, −14.5% time |
| 8K MTP decode, VL0 versus VL15 | 24.314 ms/token | 19.433 ms/token | Forced MTP-on screen, −20.1% time |

The 4K and 8K rows above are the earlier forced-MTP-on screen. The original
2K production cutoff has since been removed: [VL15 context retune and both installed MoE profiles](MTP-CONTEXT.md). The twelve-versus-ten-thread screen favored
twelve for the prefill-heavy session estimate by 2.44%, while ten decoded faster.

Native prefill/state/logit oracles at 2K and 8K, three graft fixtures, multi-turn
exactness, idle-unload session reuse and cancellation passed. The native model
arms used a 28 GiB / zero-swap scope. The qualification's highest process VmHWM
was about 19.04 GiB. Cgroup charges report page ownership and can be lower than
RSS when file pages were already charged elsewhere.

The first live regression attempts found a short-prefill timing difference
between isolated and installed contexts. The approved history check measured
144.38 ms on a fresh server and 186.02 ms after a 2,304-token request, within the
helper's existing tolerance. Back-to-back old/new captures through the installed
unit and a new-unit verification passed. The final verification took 6.77 s;
its tool-followup prefill was 147.65 ms against a 136.53 ms installed reference.
All existing thresholds passed. The engine ledger was active while loaded.

The guarded rollback is
`python3 ~/.local/share/ai-stack/rollbacks/latest-ornith-candidate2/rollback/rollback.py`.
It restores the previous Ornith profile and retains the installed Rust Flash
profile. The prior engine file is retained.

[Public-safe qualification extract](../benchmarks/results/ornith-bbc2990-qualification.json)
records the numeric and identity gates. The private runq files G0prime.result.md
and G1prime.result.md hold local evidence paths and reproduce settings. These
measurements supplement the historical stock-to-Rust chain. The final two-boot
public-launch comparison remains scheduled after optimization.
