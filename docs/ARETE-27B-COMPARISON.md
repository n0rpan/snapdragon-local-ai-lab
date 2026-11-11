# Arete27B comparison (retired)

**2026-11-09:** The owner retired Arete. Its alias/selector/download entry is removed; the17,203,416,064-byte model file was deleted under the approved cleanup plan. Historical test receipts and metadata remain. The text below records the original comparison plan and results, not an available profile. See deletion manifest.

Requested by the user on 2026-11-08. **The download is complete and its exact size/SHA256 are verified.** Local runtime and quality qualification are pending; the existing dense Qwen27B remains unchanged. If Arete is comparable on our coding/chat checks and lifecycle tests, keep both as separate selectable on-demand profiles. Ordinary coding remains the smaller MoE.

## Selected artifact

[Nous Research describes Arete27B](https://huggingface.co/NousResearch/Arete-27B) as a supervised fine-tune of Qwen3.8-27B targeting particular value-framing/refusal behavior, with no architectural change. Its reported general/agentic capability retention is publisher evidence, not our local result. The compatible [GGUF publisher](https://huggingface.co/bartowski/NousResearch_Arete-27B-GGUF) identifies the original model and provides Q4_K_M.

| Field | Selected value |
|---|---|
| Quantized repository | `bartowski/NousResearch_Arete-27B-GGUF` |
| Immutable quant revision | `9762c8502a79bd2b8634735d75f426b9febe092c` |
| Exact file | `NousResearch_Arete-27B-Q4_K_M.gguf` |
| Bytes |17,203,416,064|
| Expected payload SHA256 | `6f186a1fe0c4f741acb3a35b5df96bc2fc7279651f48500246e56079418f091b` |
| License |Apache-2.0 for the model; separate from the MIT wrapper |
| Payload verified locally |Yes—full incoming-stream and completed-cache-file SHA256 checks |

The current DevQuasar dense Qwen file is 16,547,400,064 bytes, SHA256 `9f27cff95dd7cfcb721aa7a581504fdba72d1b9cad7aaf58b4b332f7d38184f3`. Both use the Q4_K_M family, but different publishers/calibration and a roughly 4% file-size difference remain comparison limits. Do not claim identical tensor quantization from the label alone. No vision projector or speculative head is assumed compatible or enabled by this text/coding comparison.

## Download and comparison gates

1. Schedule transfer outside the active model benchmark. Use the pinned HTTPS revision, owner-only resumable partial/checkpoint, actual size/SHA verification and atomic final publication. Streaming I/O/hash stays bounded; canonical lease, external power, battery reserve and actual 2 GiB available-memory guard apply. Keep the existing model and never expose a partial file as a ready model.
2. Inspect actual GGUF architecture/tokenizer/template/context/quant descriptors after integrity verification. Use the same qualified classic CPU engine as dense Qwen,16K context,4 decode/10 batch threads, batch/microbatch 256, F16 KV and thinking off. Record effective settings and native support; do not turn on an untrained/unverified MTP path.
3. Compare the same fixed functional, structured-output/tool and chat fixtures, plus the protected normal single-file and three-module repairs. Require complete responses and the original independent tests; preserve failures/timeouts. Similarity means matching mandatory functional/protocol success and comparable results on the fixed suite, not merely producing more tokens or matching a vendor score. Broader quality equivalence is not established by this small suite.
4. Report load, prefill, decode, output-token counts and complete cold/reload task time with memory/power/thermal conditions. Investigate regressions over 20% rather than silently calling them similar; output variation and the quantizer difference remain explicit. Do not count cached prefix priming as free cold startup.
5. Before making it selectable, verify unload/reload, actual idle release, switching to both Qwen profiles and image handoff. Use a separate model/template/runtime-bound prefix namespace. Add a distinct Arete alias to the controller/OpenCode options/panel only after evidence passes, preserving the existing Qwen alias and ordinary MoE default. Models switch on demand; availability does not mean simultaneous residency.

Exact artifacts, outcome counts, controlled timings and any promotion will be added to the MIT repository. Weights/proprietary binaries/user data are not bundled. No push is authorized in this goal.

## Prepared execution

The bounded resumable downloader passed eleven offline tests, including resource-stop, slow-response and termination checkpoint cases. Its successful frozen source manifest is `b604bcbf8fc07aebe50b9281110ca8e2e18388d5fbea53034bd0bffe9897b37e`. Transfer started after the normal MoE reference finished and unloaded. Two attempts stopped on background swap page-ins with zero own swap and no new global swap-outs; the final download-only guard permits those existing-page reads while retaining the zero-own-swap, no-new-global-swap-out, actual 2 GiB RAM and power checks. Model benchmark policy is unchanged. All partials were rehashed on resume. Both source versions from an initial source-edit/launch overlap are retained; no performance claim uses this transfer. [Verified download receipt](../benchmarks/results/arete27b-download.json).

The model-free outer/nested sandbox oracle preflight passed. The first smoke attempt stopped before model loading because a sanitized public example path differed from the installed engine path; a separate harness binds the actual installed profile while preserving engine hash and numeric settings.

Both models subsequently passed the same short-answer/load/unload smoke: Arete loaded in 34.418 s and dense Qwen in 86.910 s, each answering correctly in about 3 s. These are single prepared-cache samples with only three output tokens, not a quality or representative throughput comparison. The diagnostic used CPU 600% and a 20 GiB soft limit, whereas installed classic settings are 800% and 24 GiB. [Loading smoke receipts](../benchmarks/results/arete27b-loading-smokes.json).

The cache preparation targets only the selected verified GGUF. It requests release of clean pages under the inference lease before loading into a new memory group, without reading/writing weights or changing host VM settings. Observed cache/free-memory changes and zero new swap-outs in the passing smokes support the mechanism; advice acceptance alone does not prove complete eviction. This remains a diagnostic loading policy, not a production-wide promotion.

The subsequent Arete code/chat suite stopped during loading on the sustained-PSI guard (peak 6.56%, only 24,576 bytes of new global swap-outs), before a completed answer suite. Keep that failure. Correct short answers establish compatibility only; coding/chat similarity and normal-agent/lifecycle qualification remain pending. The latest user priority puts CPU quota/thread ABBA screening next; remaining Arete comparisons stay in the same goal.

Use a roughly 15% finished-task time band when both models pass the mandatory checks, with a reverse-order repeat if the choice is ambiguous. Report a different quality/speed tradeoff directly when it falls outside that band. Keep cold and prefix-restored results separate, and preserve failed or unfinished runs.
