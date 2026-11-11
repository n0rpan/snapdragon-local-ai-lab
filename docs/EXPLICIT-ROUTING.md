# Explicit session routing — off by default

The controller has an opt-in `local-auto` alias. It performs no inference to
choose a model. Launch the controller with `LOCAL_AI_ENABLE_ROUTING=1` only
after the routing/deployment decision; this change does not enable it in any
unit, runtime config or installed release.

```json
{"model":"local-auto","local_route":"think-hard","messages":[{"role":"user","content":"Your request"}]}
```

`local_route:"code"` resolves to Ornith; `"think-hard"` resolves to Flash.
Clients choose once at session launch and retain that choice on subsequent
requests. Missing or unknown choices are rejected. No prompt heuristics or
logprob router are enabled. A concrete model alias, including the panel's
`qwen-local` selection, always wins over `local_route`. The controller removes
this extension before forwarding to the engine.

Resolved requests use the existing FIFO, eligibility checks, model hashes,
memory admission, inference lease and idle-unload behaviour. A different model
unloads the existing child before loading; two large engines never coexist.
The selected/default model on disk does not change. Idle unload stays 600 s.
Status and model-list polling stay nonwaking; `local-auto` is listed only when
the flag is enabled and the code target is eligible.

With the flag enabled, successful loads and backend load failures append
`model_load` / `model_switch` records to the existing private rotating numeric
ledger. `switch_ms` measures stop/load/readiness time inside `Manager.begin`,
excluding FIFO wait. Records contain model aliases, elapsed time and success,
with no request text. Existing engine generation records are unchanged.

This source is not deployed. Enabling routing awaits O-1 and the deployment
decision; tracked as F-C2 in the handoff FOLLOW-UPS.md. Rollback is the previous
controller release, or leave `LOCAL_AI_ENABLE_ROUTING` unset.
