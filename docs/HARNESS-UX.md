# UI proposal: harnesses, warm prefixes, clear-cache controls

For the UI redesign (MASTER-PLAN section 10: "a model chooser with engine, warm/cold prefix
state"; "a storage view with safe cache delete"). Scope: how a user sees which harness is
ready, prepares one, and clears caches. Visual design is Fable's; this fixes the content,
the states, the wording and the data behind them.

## 1. What the user needs to know

One question: *if I start pi (or dsh, or OpenCode) now, does it answer at once or do I wait?*
The wait has two parts the user cannot tell apart and should not have to:

- the model is not loaded (10 to 60 s);
- this harness's system prompt and tools were never evaluated by this engine version
  (12 to 73 s, once).

So the UI shows one state per harness and one action.

## 2. The harness strip

A row of chips under the model chooser, one per harness that has ever talked to the local
model, most recently used first. No chip for a harness that was never used: nothing can be
prepared for it yet.

| Chip state | Shown as | Meaning | Data |
|---|---|---|---|
| Ready | filled dot, "pi · ready" | the prefix is stored for the selected model and engine | a prefix with `warm: true` |
| Needs warm-up | hollow dot, "dsh · first start about 45 s" | seen before, but not on this engine version or model, or the cache was cleared | all prefixes `warm: false` |
| Preparing | spinner, "dsh · preparing…" | a prepare request is running | the button's own request is open |

Tooltip or expanded row: last used ("12 min ago"), tools and prompt size ("23 tools"), and
after the latest request "reused 5376 tokens, evaluated 212".

The estimate in "first start about N s" is `tokens / 124` rounded up to 5 s, with tokens
taken from the last request's `last_cache_n + last_prompt_n`; without that, show
"first start is slower" and no number.

Unknown clients get no chip unless the user opts in to learning them; they are listed as
candidates (numbers only) so the panel can offer "Learn". The data and the calls are in
`OPENAI-API.md` section 7.

## 3. One action: Prepare

A chip in "Needs warm-up" has one button, **Prepare**.

- Click: `POST /_local/harness/prepare {"harness": "dsh"}`. The chip goes to "Preparing".
- The call loads the selected model if needed and evaluates the stored system prompt and
  tools once. It goes through the same queue as any request, so it never interrupts a
  running answer; if one is running, the chip says "waiting for the current answer".
- Success: chip "Ready". The response lists `prompt_n` and seconds per prefix; show
  "ready in 43 s" for a few seconds.
- 409 "No recorded prefix": the harness has to be run once first. The chip does not exist in
  that case, so this only appears after "Forget".
- Failure: chip back to "Needs warm-up" with "could not prepare, try again".

Say before the click what it costs when the model is not loaded: "Loads Ornith (about 20 GiB
RAM) and prepares dsh." No confirmation dialog: the action is safe and can be undone with
"Unload now".

No automatic preparation at login or at controller start (HARNESSES.md section 4.2). After
an engine update the chips all read "Needs warm-up"; a single "Prepare all" link next to the
strip covers that case.

## 4. Clear-cache controls

One entry, "Clear caches…", in the storage view. It opens one small sheet with three rows,
each with its size and its consequence in plain words. Only the first is ticked by default.

| Row | Does | Consequence shown | Call |
|---|---|---|---|
| Stored prompts (N MiB on disk) | deletes the prefix store | "Each harness is slower on its next first start (12 to 75 s)" | `POST /_local/cache/clear` |
| Paused conversations (N MiB in memory) | unloads the model | "Open conversations are evaluated again from the start" | `POST /_local/unload` |
| Harness list | forgets seen harnesses and their saved system prompts | "Harness chips disappear until each harness is used again" | `POST /_local/harness/forget` |

The button is disabled while an answer is running, with "an answer is running" beside it
(the controller answers 409 in that case). After clearing, the strip updates at once: all
chips "Needs warm-up", or gone.

Per-harness: the chip's menu has "Forget this harness" (`{"harness": "pi"}`), nothing else.

## 5. Data

All from `GET /_local/status` (polling it never loads a model):

```
"harnesses": {
  "pi":  {"warm": 1, "last_used": 1794315000.1,
          "prefixes": [{"key": "c713f2bb9235", "model": "ornith-35b-a3b-local", "warm": true,
                        "system_chars": 2301, "tools": 4, "requests": 9,
                        "last_used": 1794315000.1, "last_cache_n": 1554, "last_prompt_n": 20}]},
  "dsh": {"warm": 0, ...}
},
"request_compat": {"null_temperature_dropped": 3}
```

- `warm` is computed for the selected model's identity (model SHA, engine binary SHA,
  context size). It is **expected** warm: the controller saw a successful request with this
  prefix on this identity and no cache clear since. The engine can still have evicted the
  entry (LRU at 4 GiB). In that case the next request is slow once and the chip was wrong
  once. An exact state needs the engine to report its store entries (proposal in
  HARNESSES.md section 4.1); the UI contract above does not change when that arrives.
- A harness can have several prefixes (OpenCode build and subagent, pi in a project with its
  own instructions). The chip is "Ready" when the most recently used prefix is warm; the
  expanded row lists the others.
- Sizes for the clear sheet: prefix store bytes from the storage data API (lab
  `docs/UI-DATA.md`); parked-session bytes are not exposed by the engine yet [?], so show
  the row without a size until they are.
- `request_compat` counts requests the controller had to rewrite for the engine. It belongs
  in a diagnostics view, not in the strip.

## 6. Kept out on purpose

- No per-prefix controls, no pinning, no budget slider. The 4 GiB store holds every harness
  on two models; a control nobody needs to touch is noise.
- No token counts on the chip face. They live in the tooltip.
- No "auto-prepare" switch in the first version. If the measurements of ticket 026 show that
  first starts hurt in practice, add a single setting "Prepare my harnesses after updates".
