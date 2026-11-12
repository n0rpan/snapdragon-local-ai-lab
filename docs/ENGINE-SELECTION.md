# Engine selection

The Rust engines (`oxe-server`) are the default. A profile runs on Rust when it has a top-level
`rust_engine` block (`{"oxe": {...}}` for the MoE/dense profiles, `{"oxe_flash": {...}}` for Flash);
the [example MoE profile](../examples/runtime/llm-runtime.json) has one. The block applies with or
without MEMLAT. The older location, `memlat_overrides.oxe` / `memlat_overrides.oxe_flash`, is still
read; there the Rust engine applies only on a MEMLAT boot (the name is historical).

| Switch (chat unit environment) | Default | Effect |
| --- | --- | --- |
| `LOCAL_AI_ENABLE_OXE` | on | `0` runs MoE/dense profiles on the classic llama-server fallback |
| `LOCAL_AI_ENABLE_OXE_FLASH` | on | `0` runs Flash on the classic BigMoe streamed backend |
| `LOCAL_AI_IMAGE_NPU` | off | `1` enables NPU images (needs your own Hexagon SDK / QAIRT build) |
| `LOCAL_AI_IMAGE_WATCHDOG_KILL_CLIENT` | off | only the exact value `1` lets the NPU image watchdog stop a client that keeps the image slot past its hold limit; any other value or unset is off |

Unset, `1` or any other value means on; only `0`, `off`, `false` or `no` switches a Rust engine off.
The classic stack is therefore an explicit opt-in. If a Rust binary is missing, a load fails with
`The Rust engine binary is missing: <path>` and names the switch for the fallback; it never falls
back silently. At start the controller logs one line per profile (`local-ai: <alias>: engine rust`,
or `engine llama.cpp (classic fallback: ...)` with the reason), and `/_local/status` reports
`engine` and `engine_note` for the selected profile.

Defaults when the unit sets no path: `LOCAL_LLM_CONFIG` is `$XDG_CONFIG_HOME/local-ai/llm/runtime.json`
and `LOCAL_AI_LEASE` is `$XDG_STATE_HOME/local-ai/inference.lock`.
