# Rust production deployment

Current follow-up: non-thinking history reuse (lab note omitted from this release) is installed for both MoE aliases (2026-11-12). The original deployment below is retained as historical evidence.

2026-11-10, explicitly approved by the owner. Package `claude/oxe-production` at `9d9ccbe`; its runtime source is byte-identical in the merged main release. Server SHA256 `649ac80670e58d5a4806355543f67a8bce0ce6034cf63e380dbf0c2a3f3ccf69`, integrated engine `707a0ec` / build `3f7b35f`. Both GGUF payload SHA256 values were recomputed before installation and match the qualification.

On MEMLAT with `LOCAL_AI_ENABLE_OXE=1`, Ornith is the everyday default and Qwen3.6 is selectable. Both use 8 decode CPUs 0–7, 12 prefill CPUs 0–11, 16K context, microbatch 256, four checkpoints, MTP off, tiled attention threshold 0 and `--prefix-cache default`. Stock boots retain classic resolution; removing the enable flag on MEMLAT also restores the classic engines. The panel/API aliases stay unchanged.

## Live checks

- One chat per alias passes; actual native executable hash/argv/environment match the package.
- Ornith's real offline OpenCode run passes 13/13 independent checks with 7 tool calls, in 55.39 s. This is a post-install functional check, not a new matched performance A/B.
- `/_local/status` reports `backend_kind: oxe`, `execution_policy_mode: memlat`, selected default Ornith, and an active ready MemoryGuard with no failure. Effective service limits are MemoryHigh=infinity, MemoryMax=28 GiB, MemorySwapMax=0 and CPUQuota=1200%. Model swap remains zero.
- Dynamic `memory.low` is **not deployed**; it remains a separately qualified follow-up, as clarified by Claude/the owner.

The first verifier incorrectly expected state `running` instead of the controller's `ready`. The required rollback ran immediately. Only that verifier condition was corrected; the identical package was reapplied and all checks above passed. The failed verifier receipt is retained locally.

## Rollback

```sh
python3 ~/.local/share/ai-stack/rollbacks/latest-oxe-production/rollback.py
```

The link currently resolves to `$AI_STACK/rollbacks/oxe-production-20261110T045739Z`. It restores exact previous runtime.json and chat-unit preimages, stops the Rust controller/native process, reloads systemd and starts the previous controller with classic profiles. Old executables remain installed. Drop-ins retain the existing 28 GiB/no-model-swap policy.

Runtime release: `$AI_STACK/runtime-releases/oxe-production-561d427d9df2`. Frozen server: `~/.local/share/ai-stack/releases/oxe-649ac80670e5/oxe-server`. Config/unit: `~/.config/local-ai/llm/runtime.json` and `~/.config/systemd/user/local-ai-chat.service`. [Numeric verification/identities](../benchmarks/results/rust-production-deployment-20261110.json), [matched cold/warm qualification](RUST-SERVER-UPGRADE.md).

Flash experiments pause for deployment; the default remains configured as Rust even when unloaded to release the inference lease for those experiments. Requests load the chosen model on demand.
