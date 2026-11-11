# Local AI panel: front-end contract (branch claude/ui-redesign)

2026-11-10. Design: handoff `UI-REDESIGN.md`. This file fixes the data shapes and the
component boundaries so the pieces can be written independently. Nothing here is installed.

Stack: the existing Omarchy shell plugin `snapdragon.local-ai` (Quickshell QML on `qs.Ui` /
`qs.Commons`), fed by the `local-ai` command. No new app, no package.

## 1. Rules for every file

- Every `Text` sets `textFormat: Text.PlainText`. No server string reaches the screen: the
  view model holds enums, numbers and allowlisted keys only; all words come from `i18n/en.json`
  through `strings.t(key)` (plural: `strings.tn(key, n)`), numbers through
  `Commons.I18n.formatNumber(value, decimals)`.
- Colours: only the `foreground`, `accent` and `urgent` properties handed down from `Panel.qml`
  (`foreground` = `bar ? bar.foreground : Commons.Color.foreground`, `accent` =
  `Commons.Color.accent`, `urgent` = `Commons.Color.urgent`). Dim text = `opacity: 0.6`.
  Surfaces = the foreground at alpha (`Qt.rgba(fg.r, fg.g, fg.b, 0.06)`). No hex colours.
- Sizes: `Style.font.caption | bodySmall | body | subtitle | title | heading | display`,
  spacing `Style.space(px)`, radius `Style.cornerRadius`. No bare pixel numbers except 1 px
  hairlines.
- Motion: `Style.duration(ms)` only, at most 180 ms, `Easing.OutCubic`. The only looping
  animation is the loading/preparing spinner or pulse.
- State is never shown by colour alone: a filled dot, a hollow ring or a spinner plus words.
- Component properties shared by all view components: `property var vm`,
  `property var strings`, `property color foreground`, `property color accent`,
  `property color urgent`, `property string face`, `property bool busy` (an action is running).
- Components emit signals; only `Panel.qml` runs processes.

## 2. Data commands (all through `~/.local/bin/local-ai`)

| Command | Exists in production CLI | Notes |
|---|---|---|
| `status --json` | yes | the extra keys of section 3 are optional |
| `load`, `unload`, `dense`, `moe`, `ornith`, `flash`, `chat`, `code`, `images`, `clear-cache` | yes | unchanged |
| `ledger --json --limit N` | new on this branch | reads the request ledger; prints `{"ledger_available": false, "rows": []}` when there is none |
| `storage --json` | new (ported from claude/ui-rust-era) | `{"storage_available": bool, "entries": [...]}` |
| `storage-delete --id ID` | new (ported) | exit 0 on success |
| `harness-prepare --harness NAME` | new | POST `/_local/harness/prepare`; exit 0 on 200 |
| `harness-forget [--harness NAME]` | new | POST `/_local/harness/forget` |

`status --json` asks the controller for `/_local/models`, and `storage --json` for
`/_local/storage`, only when the controller's status carries `"capabilities": [...]` naming
`models` / `storage`. A controller without such a route forwards an unknown GET to the loaded
engine, so a route is never probed on the chance that it exists. No controller sends
`capabilities` yet (handoff UI-REDESIGN.md, missing item 1).

An old CLI rejects the new subcommands with exit code 2; the panel treats any non-zero exit of a
data command as "not available" and hides or empties that block. It never shows an error for it.

## 3. `status --json`: keys

Existing keys (unchanged): `chat_state` (`unloaded|ready|loading|offline`), `active_requests`,
`idle_seconds`, `compute_busy`, `model`, `selected_model` (`dense|moe|ornith|flash|null`),
`prefix_cache_enabled`, `backend`, `npu_status`, `image_state` (`idle|generating|offline`),
`image_model` (`q4|q8|null`), `available_gib`, `flash_available`, `ornith_available`,
`execution_policy_mode`, the `flash_download_*` and `flash_trial_*` keys.

New optional keys (the CLI adds each only when it has the value):

| Key | Type | Source |
|---|---|---|
| `total_gib` | number | `/proc/meminfo` MemTotal |
| `queued_requests` | int 0..128 | controller status |
| `context_size` | int | controller status |
| `loaded_model` | `dense|moe|ornith|flash|null` | controller `loaded_model` (branch claude/flash-production) |
| `resident_gib` | number or null | controller `resident_bytes` |
| `idle_remaining_seconds` | number or null | controller `idle_unload_remaining_seconds` |
| `has_error` | bool | controller `last_error` is non-empty (the text is never passed on) |
| `models` | list | controller `/_local/models` rows (claude/ui-rust-era): `{key, engine, engine_status, memory_gib, battery_start_percent, prefix, selected, default, state}` |
| `harnesses` | list, most recent first, at most 12 | controller status `harnesses` (claude/harnesses), flattened: `{name, warm (bool: most recent prefix warm), last_used (unix s), tools, cache_n, prompt_n, prefixes (count)}`; `name` matches `[A-Za-z0-9._-]{1,32}`; the group `other` is dropped |

## 4. `ledger --json`: rows

`{"ledger_available": true, "rows": [...]}`, newest first, at most `--limit` (default 50, max
200). Each row holds numbers and allowlisted labels only:

`t` (unix s), `model` (`dense|moe|ornith|flash|null`), `harness` (`[A-Za-z0-9._-]{1,32}` or
null; falls back to the client product name without version), `status` (int), `stream` (bool),
`queue_wait_ms`, `admit_ms`, `ttfb_ms`, `wall_ms`, `cache_n`, `prompt_n`, `prompt_ms`,
`predicted_n`, `predicted_ms`, `finish` (`stop|length|tool_calls|null`).

Source: `~/.local/state/local-ai/ledger/requests-YYYYMMDD.jsonl` (writer:
`runtime/request_ledger.py` on branch claude/controller-queue; directory from
`LOCAL_AI_LEDGER_DIR`). The CLI reads the tail of today's file (and yesterday's when today's has
fewer rows than the limit), at most 512 KiB per file. A missing directory means
`ledger_available: false`.

## 5. `Model.js` (pragma library; replaces `Status.js`)

```
Model.parse(text, loadingSince)   -> vm
Model.parseLedger(text)           -> ledger
Model.parseStorage(text)          -> storage
Model.countdown(seconds)          -> "9:42" ("" when seconds < 0)
Model.gib(bytes)                  -> number, one decimal
Model.harnessEstimate(tokens)     -> seconds: ceil(tokens / 124 / 5) * 5, or -1 when tokens <= 0
```

Every parser returns a complete fallback object on bad input (not a string, too long, not an
object). Limits: status text 16384 chars, ledger text 262144, storage text 65536.

### vm

```
{
  service:   "offline" | "unloaded" | "loading" | "ready",
  phase:     "offline" | "sleeping" | "loading" | "ready" | "answering",
  active: int, queued: int,
  selected: key or "",            // key = "ornith" | "moe" | "flash" | "dense"
  loaded:   key or "",            // loaded_model, else `model` when service is ready/loading
  backend: "cpu" | "npu" | "gpu",
  ram: { totalGib, availableGib, modelGib, otherGib },   // -1 = unknown; otherGib = total - available - model, floored at 0
  idle: { totalSeconds, remainingSeconds },              // remainingSeconds -1 = unknown
  canUnload: bool,                // service ready, active 0, images not generating
  unloadBlocked: "" | "nothing" | "loading" | "answering" | "images",
  models: [ { key, group: "named" | "heavy", available: bool, selected, loaded, isDefault,
              state: "loaded" | "loading" | "cold",
              engine: "rust" | "llama" | "stream" | "", prefix: "warm" | "cold" | "off" | "",
              needGib, fits: "yes" | "no" | "unknown", batteryStart } ],
  harnesses: [ { name, state: "ready" | "warmup", estimateSeconds, tools, lastUsed,
                 cachedTokens, evaluatedTokens, prefixes } ],
  hasHarnessData: bool,
  images: "idle" | "generating" | "offline", imageModel: "" | "Q4" | "Q8",
  computeBusy: bool,              // another task holds the compute slot
  hasError: bool,
  prefixCache: bool,
  flash: { state, percent, doneBytes, totalBytes, verified, shards, waiting,
           trialActive, trialKind, trialElapsed }       // as Status.js computed them
}
```

- Chooser order and groups: `ornith` (named, default), `moe` (named), `flash` (heavy). `dense`
  is listed (named, last) only when it is the selected or loaded model. `ornith` and `flash`
  have `available: false` when `ornith_available` / `flash_available` is not true; such a row is
  shown disabled.
- `needGib` comes from `models[].memory_gib`; without it -1. `fits` is `unknown` when `needGib`
  or `availableGib` is unknown; otherwise `yes` when
  `needGib <= availableGib + max(0, modelGib)` (the loaded model is released first) or when the
  row is the loaded model.
- `phase`: `offline` when service is offline; `loading`; `answering` when ready and active > 0;
  `ready`; `sleeping` when unloaded.
- `harnesses[].state` is `ready` when `warm` is true, else `warmup`. `estimateSeconds` =
  `harnessEstimate(cache_n + prompt_n)`. "Preparing" is panel-local state, not in the vm.

### ledger

```
{ available: bool,
  rows: [ { t, model, harness, ok: bool (status 200..299), status,
            queueMs, loadMs, ttftMs, wallMs, tokS, cacheN, promptN, cacheShare, predictedN,
            finish } ],
  summary: { count, failed, medianTtftMs, medianTokS, cacheShare, lastT } }
```

- `loadMs` = `admit_ms` when it is at least 1000, else 0 (a model load happened).
- `ttftMs` = `queue_wait_ms + admit_ms + ttfb_ms` (missing parts count 0; -1 when `ttfb_ms` is
  missing).
- `tokS` = `predicted_n / predicted_ms * 1000` when both are > 0, else -1.
- `cacheShare` = `cache_n / (cache_n + prompt_n)` in 0..1, or -1 when the sum is 0.
- Summary medians are over rows with a value; `cacheShare` in the summary is token-weighted.

### storage

As `Status.js` `parseStorage` on claude/ui-rust-era:
`{ available, entries: [ { id, kind, state, bytes, deletable, name, model } ], totalBytes, freeableBytes }`.

## 6. Components (all in `src/snapdragon_ai_lab/omarchy_plugin/`)

| File | Role | Signals |
|---|---|---|
| `Panel.qml` | bar button, polling, processes, view switch, launch row | none |
| `NowCard.qml` | what is loaded, RAM, countdown, Unload now | `unloadRequested()`, `loadRequested()` |
| `RamBar.qml` | one stacked bar: model, other, free, marker for the selected model's need | none |
| `ModelsView.qml` | chooser rows (named, heavy), harness strip | `selectRequested(string key)`, `prepareRequested(string name)`, `prepareAllRequested()`, `forgetRequested(string name)` |
| `HarnessChip.qml` | one chip; `markState`: ready, warmup (with Prepare), preparing | `prepareRequested()`, `forgetRequested()` |
| `ActivityView.qml` | ledger summary and rows | none |
| `StorageView.qml` | storage rows, delete, Clear caches sheet | `deleteRequested(string id)`, `clearRequested(bool prompts, bool conversations, bool harnesses)` |

`Panel.qml` properties handed to the views: `vm`, `ledger`, `storage`, `preparing` (list of
harness names with a prepare running), `strings`, colours, `face`, `busy`.

Polling: `status` every 5 s while `pollEnabled` (1 s while the panel is open and a countdown or
a load is shown); `ledger` and `storage` only while their view is open (ledger every 5 s,
storage on open and after an action). Polling never loads a model.

## 7. Checks

- `tests/test_panel_model.py`: `Model.js` under node (fallbacks, allowlists, derivations).
- `tests/test_local_ai_ui_data.py`: the `local-ai` data commands against a fake controller.
- `tests/test_panel_preview.py`: NowCard, ModelsView, ActivityView and StorageView load and lay
  out offscreen (`qml6`) with stand-ins for Omarchy's kit in `tests/panel_preview/qs`, without
  QML warnings; every `Text` is plain text; no hex colours. `tests/panel_preview/render.sh
  <status.json> <out.png>` writes a picture.
- Not covered: `Panel.qml` itself (it needs Quickshell's `Process` and the shell's panel
  classes; qmllint syntax only) and the look under the real kit and theme.
