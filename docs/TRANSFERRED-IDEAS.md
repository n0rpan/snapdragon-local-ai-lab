# Ideas from other engines

The deployed stack uses llama.cpp and stable-diffusion.cpp; an additional Flash-Next experiment uses BigMoeOnEdge's CPU streamer. Reviewing another engine does not mean its kernels, models or performance transfer to Snapdragon.

## Sushi

Primary sources reviewed on 2026-11-07: [repository](https://github.com/beamivalice/sushi), [performance methodology and baselines](https://github.com/beamivalice/sushi/blob/main/docs/perf-baselines.md).

Sushi targets Apple Silicon. Its low-bit formats are paired with specialized kernels. Its performance notes describe packed-reader changes, fused operations, interleaved comparisons and output-equivalence checks. These are implementation and measurement lessons, not a portable speed multiplier.

Applied here:

- Keep exact engine, model and patch identities beside measurements.
- Screen kernel/configuration changes before running expensive agent tasks.
- Check correctness independently and count complete task time.
- Repeat finalists under matched power conditions; use interleaved comparisons where practical.
- Preserve fallback paths and report failures.

Completed here: the KleidiAI/Q4_0 candidate improved a standalone prefill case but did not improve the full warm agent task, so those weights were not promoted. The affinity screen was flat and stopped. On the harder three-module task, the existing MoE passed cold and after reload; both dense variants timed out, and the successful plan/build/review split was slower. See [measured comparisons](BEFORE-AFTER.md).

The additional Flash-Next trial targets Snapdragon with 32 GB RAM. Unsloth V3.1 UD-IQ3_XXS has now passed fixed short-answer checks, native HTTP/streaming/tool calls and RAM recovery using a bounded CPU expert cache. A matched prompt-batch experiment improved prefill, and a scoped hugepage experiment recovered substantial memory headroom. Full coding-agent qualification remains open; see the current evidence and limits. Transferable Sushi ideas include importance-aware precision allocation, protection of sensitive shared weights, a complete memory budget, and quality checks across code, math and language. Its STQ/EXL3 bytes require their own reader and kernels; matching a bit count or converting them again to GGUF would not reproduce Sushi's quality.

The initial trial preserves native expert routing and does not omit or substitute experts. Compare complete correct-answer and coding-task time, cold and warm prefill, SSD reads, peak RAM and clean unload before promoting it. Apple and phone measurements are context, not acceptance evidence for this machine. [Pinned Sushi format](https://github.com/beamivalice/sushi/blob/9f8801bda5600b4a1f926f683e82b9a4fbb203d1/docs/pack-format.md), [Unsloth model guide](https://unsloth.ai/docs/models/qwen3.8-next), [streaming runtime](https://github.com/Helldez/BigMoeOnEdge/tree/930b2c9ebd8c2cface6566bd9f11075ac07adfe0).

## oMLX

Primary source: [oMLX](https://github.com/jundot/omlx), reviewed at commit `2165e75726ba83d9651dbaced61c97728da77bba`.

Useful concepts include cached prompt reuse, memory-aware model lifecycle, architecture-specific attention/recurrent kernels and speculative-head prompt priming. Apple Metal and ANE implementations are hardware-specific. The current llama.cpp stack already contains fused gated-delta support and MTP prompt processing; verify actual dispatch and timings before duplicating existing features.

The deployed controller saves only a leading instruction/tool prefix, before any user request or tool result is processed. Real isolated OpenCode repairs through the normal installed endpoint verified reuse across complete unload/reload. The actual idle timer, model switching, cache clearing, image handoff and RAM recovery also passed. Flash-Next's separate runtime has resident prefix reuse but no compatible disk slot interface; the existing cross-session cache result must not be transferred to it.

No source from Sushi or oMLX is included in this repository.
