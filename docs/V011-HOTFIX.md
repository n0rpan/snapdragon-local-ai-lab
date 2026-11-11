# v0.1.1 authentication and sampling hotfix

The controller and Rust engines were installed together for Ornith, Qwen3.6 and Flash. Every final installed security and five-case regression check passed against the original references. One full, unmeasured smoke pass preceded each warm comparison; no timing waiver was needed on the final run. Earlier failed timing and verifier attempts remain in the private lab notebook.

The native server now requires the controller's private per-launch bearer on every route, including health. Foreign Host/Origin requests return 403 and responses carry no wildcard CORS header. Authenticated `temperature: 1, min_p: 2` requests return 400 without killing the server. Streaming and nonstreaming chat/completion probes verified valid subsequent requests on the same process. Keys stay in owner-only runtime files and are removed on unload; keys, their paths, headers and conversation bodies are absent from the exported evidence.

| Model | Protected checks | OpenCode task timer | Harness wrapper wall time |
|---|---:|---:|---:|
| Ornith | 13/13 | 48.55 s | 49.77 s |
| Qwen3.6 | 13/13 | 42.89 s | 43.80 s |
| Flash, 14 GiB | 13/13 | 183.04 s | 183.81 s |

These are qualification costs, not matched performance gains. Installed checks covered chat, tool call, tool follow-up, 4K context and cancellation. Qwen's previous six-case reference remains intact; its separate disk-session-resume case was not repeated for this emergency fix. All three models showed zero model swap/OOM, clean guards, key removal and unload; CPU 8 QoS returned to zero.

Internal source freezes: engine `25a6c74`, controller `165dac4`. Built binary SHA-256: MoE `71d927e4ca0887c8adfee2ca375586b55ad7f6258b4cc2c6e32196630ea5d84e`; Flash `b565c250e4b3339ee589dfe0f416d1d2000e7d1d54d7522ec4b6bec5bc4a89ff`. See the [numeric evidence](../benchmarks/results/public/v011-hotfix.json).

The subsequent [18 GiB cache policy](FLASH-CACHE18.md) has its own rollback to secure 14 GiB. After that cache rollback, the guarded full-pair rollback remains available at `~/.local/share/ai-stack/rollbacks/latest-v011-hotfix/rollback-live.sh`.

The full-pair rollback restores the earlier release, including its known B1/B2 behavior; it is a recovery option, not the preferred cache-policy rollback.
