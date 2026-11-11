# App and memory settings audit

This source/settings audit checked the installed laptop's model/profile metadata and a narrow OpenCode configuration allowlist on 2026-11-08. It started no model, read no chat/auth/session content and changed no live settings. Separate records establish classic lifecycle acceptance and the protected Flash repair. The new Flash settings, controls and UI described here are installed; complete normal planning/image-handoff acceptance passed separately in 238.448 s. [Installed lifecycle evidence](../benchmarks/results/flash-next-normal-lifecycle.json)

| Action | Intended model | Configuration |
|---|---|---|
| Ordinary OpenCode / build | Qwen3.6-35B-A3B MTP, UD-Q4_K_M | Explicit `local-qwen/qwen3.6-35b-a3b-local`; 16K context, 12K input, 4096 output |
| Local AI planning launch | Qwen3.8-Flash-Next, Unsloth V3.1 UD-IQ3_XXS | Named `local-plan` pins Flash; 16K context, 12K input, 2048 output; launch overlay permits file reads |
| Panel OpenCode with Dense/MoE selected | Selected explicit coding model | Per-launch model override; ordinary default agent/policies retained; global MoE unchanged |
| Selected chat | Dense, MoE or qualified Flash | Dynamic `qwen-local` follows the panel selection; selecting alone stays lazy |
| Dense alternative | Qwen3.8-27B, Q4_K_M | Explicit `qwen3.8-27b-local`; 16K context, 12K input, 4096 output |
| Images | Z-Image-Turbo Q8_0; Qwen3-4B-Instruct-2507 Q4_K_M text encoder; compatible VAE | Q8 draft 256/8 and final 1024/8; exclusive handoff |

The installed global OpenCode model is explicit MoE; title generation remains disabled. `local-plan` describes the supported Local AI launch. Manual agent selection inside another unrestricted client does not apply the launch overlay. Both classic profiles retain four decode/ten batch threads, F16 KV and CPU repacking; MoE uses its qualified two-token MTP policy. Exact model/binary identities and reconstruction details are in [RUNTIME-SETUP.md](RUNTIME-SETUP.md), [DEPENDENCIES.md](DEPENDENCIES.md) and the model-specific benchmark receipts.

## Spend RAM where it has measured value

Flash already selects the larger tested 12 GiB expert cache. The isolated 8-to-12 GiB comparison generated the same 246-token JSON answer in each arm: generation rose 1.752→2.447 tok/s, decode reads fell 18,593→12,226 MiB and complete elapsed fell 228.81→181.69 s. Minimum available RAM fell 12.15→8.96 GiB. These are sequential structured-output observations; they do not predict the same percentage for a coding agent. [Full comparison and limits](FLASH-NEXT-PREFILL.md)

The installed Flash admission budget is 21.5 GiB, above its conservative 21.212 GiB estimate and measured 19.706 GiB peak. Start additionally requires 4 GiB available; the live watcher stops below that reserve. This budget admits measured memory use; it is not a requested smaller cache or a per-child RSS ceiling. The existing chat unit's 25 GiB hard limit applies to its whole process group. Keep the 12 GiB cache and demand-mapped ngram table; reducing cache merely to show more free RAM would discard a measured benefit.

The classic Dense/MoE engines load their complete weight files rather than using Flash's SSD expert-cache design. Native `--cache-ram 0` disables a separate llama prompt-cache facility; it does not cap weight or KV residency. The controller's 3 GiB prefix-store limit controls disk usage and its saved instruction/tool prefix, not model RAM. More unused RAM alone does not make these already resident weights faster. Larger context, more concurrent slots or an extra resident engine would consume RAM but need their own finished-task evidence; none is enabled by this audit.

No RAM-saving knob was added. A cache larger than 12 GiB remains a finite follow-up experiment requiring separate memory admission and exact task/time verification. It cannot safely be exposed as an unrestricted slider: the current qualified policy binds cache size, measured peaks, 4 GiB reserve and native/source identities. The user preference favors a larger cache if a measured complete-task improvement fits that envelope.

## Concrete source corrections

The portable OpenCode example now pins ordinary coding to explicit MoE, names the exact Dense/MoE quantizations and matches the installed explicit classic 4096-output limits. The portable dynamic selected-model alias is limited to 2048 output so it can safely route to Flash; the installed dynamic alias now also advertises 2048. Supported planning and ordinary coding already use their explicit models and correct limits. A separate [Flash planning fragment](../examples/flash-next-opencode.fragment.json) preserves that global default and caps Flash output at 2048. All example runtime profiles remain disabled until locally qualified.

The manual runtime recipe now copies all seven imported chat modules, the image worker and the planning launcher. Omitting the new helper modules would break even classic controller imports. The optional panel installer places the planning helper beside its existing regular control command with backup/rollback support. Focused example and installer tests cover these contracts.

The installed reviewed UI patch provides a full-width Flash selector, a `Plan (read-only)` action label when Flash is selected, a qualified status that supersedes the old testing-pending download marker, and clearer CPU/failure text. The patch leaves model/cache/runtime policy unchanged. This patch is installed. Complete normal-workflow validation passed separately; the source audit itself launched no model.

The reviewed controls fix makes the panel's Dense/MoE selection apply to its OpenCode launch while preserving ordinary agent policies and the global MoE default. The existing launcher consumes a fixed `--model dense|moe` enum itself; it forwards no unsupported model flag to OpenCode. Its no-option planning mode retains the verified read-only overlay. An isolated pinned actual-client probe confirmed Dense and MoE request their respective aliases from a global-MoE base, and repeated the planning read/deny-write proof. These control files are installed by the deployment transaction. The isolated probe is distinct from the passed complete normal planning/image-handoff run.
