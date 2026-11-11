# Qwen tomorrow package — staged only

`tools/stage_qwen_night.py` prepares a Qwen-only candidate from pinned live
preimages. It never installs, starts a service, loads a model or changes a
default. The accepted native binaries are the frozen `a219347` night build or
its `e831814` fallback. Qualification on Qwen is still required.

The memlat Rust profile gets 32K context, 8 decode CPUs (0–7), 12 prefill CPUs
(0–11), ubatch 256, 8 checkpoints, 2 GiB parked-session budget, private disk
sessions, numeric ledger capture and mask 13. Speculation is explicitly **off**:
Qwen does not inherit Ornith's graft or its compiled speculation default. Mask
13 must pass Qwen's own fixture and speed checks. Memory limits stay at 28 GiB
with no model swap. The idle and output defaults stay at 600 seconds / 16384.

Only Qwen's OpenCode entry changes: context 32768, input 24576, output 16384.
The engine still clamps output to the remaining context. Ornith, Flash,
`qwen-local`, selected model and default model remain unchanged. Classic/stock
boot launch settings stay unchanged; use the rollback's 16K provider settings
when testing the stock fallback.

The stage contains `runtime.json`, `local-ai-chat.service`, `opencode.json`,
`runtime-source/`, a hash-checked engine copy and `rollback/rollback.py`. Configs
can contain credentials: the stage is outside git, directory 0700, files 0600.
No private sessions may be placed on the public-weight partition.

Run the helper with absolute input paths and their actual SHA256 values:

```sh
python3 tools/stage_qwen_night.py /private/staged/qwen-next \
  --runtime /path/to/runtime.json --runtime-sha256 ACTUAL_SHA \
  --unit /path/to/local-ai-chat.service --unit-sha256 ACTUAL_SHA \
  --opencode /path/to/opencode.json --opencode-sha256 ACTUAL_SHA \
  --server /path/to/frozen/oxe-server --releases-root /private/releases \
  --sessions /encrypted/private/qwen-sessions
```

Regenerate the package **after tonight's Ornith install**. The recorded preimages
are deliberately invalidated by later live changes; do not overwrite a newer
Ornith or Flash configuration with this snapshot.

Tomorrow's gate: Qwen fixtures, assembled tool calls, cancel, session resume,
one protected task, 32K RAM/no swap and a matched performance check. Keep the
candidate `qualified:false` until then. Promotion needs its separate approval.
When approved, copy the engine and runtime-source to the manifest's release
directories, verify hashes, then use the existing preimage installer around a
service stop/reload/start. The one-command rollback is:

```sh
python3 /private/staged/qwen-next/rollback/rollback.py
```

The rollback restores the exact three preimages and restarts the controller;
it loads no model and deletes no weights or sessions.
