**Current status: integrated Rust 649ac806 is installed on MEMLAT.** Older measurements below retain their historical context. [Live deployment and rollback](RUST-DEPLOYMENT.md).

# Standalone Rust server: promotion handoff

2026-11-10. **Candidate only; production remains llama.cpp.** Integrated source `b477aec`, qualified server SHA `789e8784a8ea7cfeff97e9939b512001364418206ea47521fe23b4b57fb41835`. Two repeats add timing-only diagnostics (SHA `a8263c22fbf2…`); no math/layout changes, and all three native-reference prompts match again in every Rust arm.

## One-screen decision

| Protected OpenCode task | llama.cpp median, n=3 | Rust median, n=3 | Less task time |
|---|---:|---:|---:|
| Ornith, MTP off |77.70 s|60.42 s|22.2%|
| Qwen, MTP depth 2 |75.31 s|67.42 s|10.5%|

All arms pass 13/13 independent checks, real tool calls, zero OOM/model swap, and clean native/guard/lease cleanup. Rust also matches 128 greedy tokens on each of three split-last native llama.cpp references per model. Reference uses native layout/KleidiAI off, not the numerically different production KleidiAI path. Both models retain their exact GGUF identities listed in RUST-M3.

**Recommend Ornith Rust/MTP off as default, Qwen Rust/MTP off selectable; retain llama.cpp as fallback.** Qwen-off isolation passes the same checks in 64.72 s (one sample, not a median). MTP raises decode but its prompt cost loses time here. Emitted text/tool-input traces differ on/off despite equal token counts; Claude is investigating sampling versus exactness. Fixed-oracle checks pass, so this is not yet a confirmed exactness bug. No live configuration/default change has been made.

## Where task time goes

Phase medians below use the two instrumented repeats; finished-task medians above use all three. Phase medians need not sum to the median task.

| Arm | Evaluated prompt tokens | Cache reuse tokens | Prefill | Decode | Reported decode tok/s | Requests |
|---|---:|---:|---:|---:|---:|---:|
| Ornith llama.cpp |6,160|36,851|40.27 s|30.80 s|25.45|7|
| Ornith Rust |6,086|36,662|40.25 s|19.16 s|34.50|7|
| Qwen llama.cpp/MTP2 |6,872*|46,243*|46.64 s|18.91 s|29.30|8|
| Qwen Rust/MTP2 |6,184|42,865|45.56 s|20.84 s|32.15|8|

*Rounded medians of integer counts. Evaluated tokens include any re-prefills; cache counts are logical reuse summed across requests, not unique tokens. CPP timing excludes the first output interval in its reported decode rate; emitted-token/time is also in the JSON. No major cache-reuse failure appears.

Rust checkpoint copies total 0.15 s Ornith / 0.09 s Qwen and are included in prompt/decode timers. Qwen MTP seeding totals 3.88 s, included in prefill. Backend HTTP/template/tokenizer/serialization residual is 0.05–0.14 s for the entire task, not tens of seconds. Process startup is 0.27 s Rust versus 4.02 s Ornith CPP / 8.78 s Qwen CPP; client/tool residual is about0.84–0.92 s. Actual process RSS high-water marks: Rust 15.97 GiB Ornith / 21.72 GiB Qwen, CPP 16.25 / 23.45 GiB. Cgroup peaks can be lower because file pages were already charged elsewhere.

Qwen-off isolation: 6,184 evaluated /42,865 reused /670 generated /8 requests; 38.96 s prefill, 24.62 s decode, 27.22 emitted tok/s, 0.416 s checkpoints. Matching-count MTP-on repeat: 45.50 s prefill, 20.71 s decode, 3.89 s seeding, 67.42 s task versus 64.72 s off. Counts match; text/tool-input trace differs, so do not call this an identical-trace speed A/B.

## Sampling and comparison limits

OpenCode sent **temperature 0**; **top_p, top_k and seed were absent** (engine defaults). The original observer omitted these fields. A no-model mock captured the same first request and its entire-body SHA matches the measured Qwen request (`55d1ed0a…befdb55`), proving those fields without another inference run. The mock intentionally fails the protected task and is labelled metadata-only; it is not a model failure. All Qwen client config files are unchanged.

Same warm-up protocol (three 128-token synthetic requests), then fresh processes, 16K context, 8 decode cores 0–7 and 12 prefill cores 0–11, ubatch 256, 28 GiB cap, zero service swap. External prefix-file restore is disabled in both arms; in-session cache reuse remains on. OS page cache is uncontrolled. First requests in paired diagnostic runs have matching body hashes. Tool results/trajectories differ; total seconds/output-token is not a decode benchmark because prefill/startup are substantial.

The original unmatched 89.43 s CPP arm is excluded. The historical 64.0 s direct run generated 479 tokens versus 957 in the first new CPP arm, and used a different direct path/first prompt (5,540 versus5,544 initial tokens); it is not the new control. Earlier provisional 32.4%/27.6% figures are replaced by the 22.2% median. Sample output counts: Ornith CPP 957/791/791 versus Rust 746/661/661; Qwen CPP 718/537/586 versus Rust 535/670/670. One protected repair task does not establish broad quality superiority.

[Per-arm and per-request metadata](../benchmarks/results/rust-integrated-server-qualification.json), [sampling capture](../benchmarks/results/rust-integrated-sampling.json), [summarizer](../benchmarks/roofline/summarize_integrated_server.py). Raw synthetic output/oracles stay private. Lightweight repeats completed in about 13 minutes of serial wall time (plus the separate off diagnostic and build), under the lease.

## Follow-up: faster MTP seeding (not promoted)

A separate `c73a628` Qwen off/on pair passes all three native greedy fixtures and 13/13 protected checks. Task time is flat: 77.93 / 77.92 s, with 6,184 new prompt tokens, 42,865 reused tokens and 670 output tokens in each arm. Seeding falls from the earlier ~3.88 s to 1.02 s, but this pair does not demonstrate an overall speed gain. Both arms are slower than the earlier b477 off sample; the cause is unresolved. Every request sends temperature 0 with top_p/top_k/seed absent. Request bodies match on the first turn and differ starting at turn two, so equal counts do not prove identical text. Zero model swap/OOM and clean cleanup. [Metadata and per-request timings](../benchmarks/results/rust-mtp-seed-fast-single-pair.json). The production package remains pinned to the earlier qualified binary.

Inspecting the retained client event logs narrows the new pair's trace difference: assistant text events match, and the tool-name/input multiset matches. Only two read-tool events exchange order (`test_port.py` / `parse_port.py`). Request-body hashes still differ because they include IDs and ordering. This removes the evidence of differing textual content in this pair; it does not prove token-stream exactness.

## Opt-in deployment (MEMLAT boot only)

The package now pins integrated `707a0ec` (binary built at `3f7b35f`), SHA `649ac80670e58d5a4806355543f67a8bce0ce6034cf63e380dbf0c2a3f3ccf69`. Three trials/arm against the earlier 789e build: Ornith 60.06 → 52.13 s, Qwen 65.53 → 57.91 s. Separate fresh-process prefix pairs reduce first-request prompt time to 1.22 s from 30.64/31.25 s. All 16 arms pass native fixtures and protected checks; no OOM/model swap. Original qualification above remains historical. [Upgrade report](RUST-SERVER-UPGRADE.md).

Installed on MEMLAT on 2026-11-10. The same aliases `ornith-35b-a3b-local` and `qwen3.6-35b-a3b-local` resolve to the Rust server only when **both** MEMLAT is detected and the controller runs with `LOCAL_AI_ENABLE_OXE=1`. The stock boot resolves the stock classic profile exactly (detached copy); MEMLAT without the variable keeps the existing classic MEMLAT overrides. Other aliases, the panel and OpenCode configuration are unchanged.

Each alias gets one nested `memlat_overrides.oxe` entry (`server_binary`, `server_sha256`, `oxe_args`, optional `threads`, `environment`, informational `validation`). When active, it sets `backend_kind: oxe`, the pinned `649ac80670e…` server, `extra_args: []`, forces the external PrefixStore off, and replaces the classic environment with `OXE_MODEL_SHA256=<profile sha256>` and `OXE_ATTN_TILED_MIN_KV=0` (`GGML_*`/`LLAMA_*` are rejected in the override and stripped from the inherited child environment). Arguments: `--oxe-cpus 0-7 --oxe-prefill-cpus 0-11 --ctx-size 16384 --ubatch-size 256 --ctx-checkpoints 4 --reasoning off --spec-type none --prefix-cache default`. The external llama.cpp PrefixStore stays off; Rust's private persistent prefix store is enabled. The controller verifies only the Rust engine hash for the beta alias in this mode. Model identity, readiness and RAM admission stay stock.

**Enable.** Install the qualified binary as an owned regular file, then stage without touching live files:

```
python3 -I tools/stage_oxe_production.py STAGE --runtime RUNTIME_JSON --runtime-sha256 SHA \
  --unit CHAT_UNIT --unit-sha256 SHA --server-binary INSTALLED_OXE_SERVER --releases-root RUNTIME_RELEASES
```

It hashes the binary (must be `649ac806…`), checks both model SHAs, and writes `runtime.json`, `local-ai-chat.service` (new content-addressed release directory plus `Environment="LOCAL_AI_ENABLE_OXE=1"`), `runtime-source/` and `rollback/` (exact preimages plus `rollback.py`). Copy `runtime-source/` to the printed `release_dir`, install the two files under an idle lease, `daemon-reload`, and restart the chat service.

**Roll back.** `python3 STAGE_ROLLBACK_DIR/rollback.py` restores the exact previous `runtime.json` and unit (previous controller release, no enable variable, classic engines) and restarts the service. A softer switch is removing the `LOCAL_AI_ENABLE_OXE=1` line and restarting: the aliases return to classic MEMLAT.

**Verify.** `/_local/status` after selecting an alias reports `backend_kind: oxe` and `execution_policy_mode: memlat`; one round trip:

```
curl -s http://127.0.0.1:8092/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"ornith-35b-a3b-local","messages":[{"role":"user","content":"Reply with OK."}],"max_tokens":8}'
```
