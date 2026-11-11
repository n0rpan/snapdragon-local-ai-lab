# Ornith additional MoE candidate

Ornith is downloaded and full-file checksum-verified alongside the current Qwen MoE. [Download and header evidence](../benchmarks/results/ornith-moe-download.json). It is not yet a qualified or selectable runtime profile. Keep the existing model and default while comparing real finished-task behavior; publisher scores do not establish local quality or speed.

## Reproducible candidate

| Field | Value |
|---|---|
| Publisher | [ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) |
| Quant repository | [mudler/Ornith-1.5-35B-A3B-APEX-MTP-GGUF](https://huggingface.co/mudler/Ornith-1.5-35B-A3B-APEX-MTP-GGUF) |
| Revision | `4c326713c52d13fdb40983a2a09a46cca8e88739` |
| File | `Ornith-1.5-35B-A3B-APEX-MTP-Compact.gguf` |
| Exact bytes | 17,437,861,152 |
| Expected SHA256 | `a7373f724cf3693e8a6a5f8209ae9a06ff8071ae2af60659bdbdf8ff6f6c10df` |

Compact is the first candidate because it leaves more room for repacking, context and the desktop. The file is approximately 16.24 GiB; this is its storage size, not a measured runtime footprint. Other APEX tiers are different numerical artifacts, not equivalent speed settings.

The complete bounded header/tensor inventory confirms GGUF3, architecture `qwen35moe`, 41 blocks, one NextN/MTP layer, 256 routed experts and native top 8. All 753 tensor descriptors and 44 metadata entries were parsed without reading tensor payload. Types are F32 (310 tensors), Q3_K (90), Q4_K (221), Q6_K (121) and Q8_0 (11); numerical type support in source is not proof of binary loading or quality. The embedded template is hash-recorded. Native loading, MTP behavior and quality remain separate gates. This model's native routing differs from Flash's 512/top-10 contract; no routing is changed in either model.

The quant card declares Apache-2.0, while the embedded model license identifies MIT and links the original publisher license. Preserve both provenance notices and verify the upstream terms when packaging. Repository MIT licensing applies to our code and does not replace model/vendor licenses. Weights stay outside Git.

## Acceptance and real-world testing

1. Verify the complete file's pinned byte count and SHA256, retaining the download receipt and all existing models. A resumable partial download is not a model qualification.
2. Inspect the complete bounded GGUF header/tensor table, the embedded template and NextN compatibility. Run a bounded CPU load, fixed synthetic answer and unload; check actual RAM, own swap, recovery and lease release.
3. Run the unchanged independent coding checks, synthetic chat and tool-call protocol fixtures. Record every failed attempt, repair, generated token and complete elapsed time. Different models may produce different valid programs; answer-hash identity across models is not required.
4. Compare against the qualified Qwen MoE with the same resource policy and context, recording each model's exact template, thinking mode and speculation settings. Check whether the MTP head helps this model rather than inheriting Qwen's acceptance rate. Advance a useful candidate to the protected agent task and lifecycle checks.
5. If mandatory checks pass, expose both names through the local controller/panel for real-world selection. Keep ordinary OpenCode's tested default until comparison supports changing it. Heavy inference stays serialized; storing both files does not mean loading both into RAM.

The current production reference is Unsloth Qwen3.6-35B-A3B-MTP UD-Q4_K_M, SHA256 `0b21525e972670ed59e1812e170b27c26355381f0656ecc4e25617ece7dac58b`. Its exact source revision and bytes are in the [model registry](../dependencies/model-artifacts.json). New artifact registration and fresh-install availability follow qualification; the public installer must not silently mark this candidate ready.

Cleanup is limited to verified redundant temporary payloads and obsolete build material. Preserve source pins, benchmark/failure receipts, rollback requirements and all useful model alternatives. The first cleanup removed one redundant 335,304,388-byte VAE download-test copy after both it and the retained copy matched the pinned checksum.

The first complete transfer matched SHA256 but failed its strict resource receipt because kernel-accounted peak exceeded the nominal 1 GiB hard cap (1,105,596,416 bytes), with no OOM, own swap or PID failure. That receipt stays failed. An unchanged-code, local-only full-file reverification with a stricter 512 MiB soft limit passed at 537,563,136 bytes peak; no second network transfer was needed.

## First native smoke: loads, strict reply check fails

The same qualified CPU engine loaded Ornith in 11.554 s and returned a complete three-token response, then reaped, recovered RAM and released the lease. Its response included a terminal period, failing the strict expected-READY checker for an instruction ending `READY.`. [Original failed smoke](../benchmarks/results/ornith-moe-first-smoke.json). The failure and all costs remain recorded; this is not a passed smoke or runtime qualification. A separately versioned instruction explicitly prohibiting punctuation will be used for both models, with the same strict checker. The original coding/protocol/protected oracles remain unchanged. The three-token timing is not a representative speed or MTP-acceptance result.

## Matched V2 readiness/lifecycle checkpoint

Both fresh V2 runs passed the same strict no-punctuation readiness check and cleanup gates. Ornith loaded in 11.024 s and the reference Qwen MoE in 35.005 s. [Matched smoke receipts](../benchmarks/results/ornith-moe-v2-smokes.json). Loading/storage/cache state and file size differ, so this is not a causal load-speed claim. Both retain their model files; coding, chat, tool-call and protected-agent quality remain required before a selectable rollout.

## Matched V3 code, chat and tools: both pass

Both models now pass the same two independent coding checks and five functional checks, including a structured tool call and handling its result. Complete screen time was 124.017 s for Qwen and 92.656 s for Ornith. Qwen needed one parser repair; Ornith passed both coding cases on its first answer. This is a small model comparison, not a causal runtime improvement or a general quality ranking. Both unloaded cleanly, recovered RAM and released the inference lease. [Curated results](../benchmarks/results/ornith-moe-v3-screen.json).

Both earlier V2 screens returned correct final tool-result values inside Markdown fences and failed strict raw JSON parsing. Those failures remain preserved. V3 adds the same native API JSON Schema to that final response for both models; it constrains field types without supplying answer values. No client stripping or answer repair occurs. The actual OpenCode agent task does not receive this forced format. The separate protected normal 13-case repair subsequently passed before any selectable rollout.

## Protected OpenCode task: pass

Ornith completed the unchanged normal repair in **128.250 s from load through cleanup** (115.663 s in the agent). All 13 independent checks passed; it used six completed tool calls, edited only the intended implementation, ran the tests and left the protected test file unchanged. The native server and descendants exited, RAM recovered and the lease was released. No forced JSON response format was added to OpenCode. [Protected task evidence](../benchmarks/results/ornith-moe-normal13.json).

This establishes eligibility for a separately reviewed selectable alternative, not an installed profile, default change or broad quality ranking. There is no new matched Qwen agent run, so these seconds must not be compared causally with historical Qwen tasks. The optional atomic-reservations stress task was not run. Installed model switching, panel routing, idle unload and rollback still need verification.
