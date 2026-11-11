# Agent harnesses beyond OpenCode: pi and dsh

Status 2026-11-10, branch `claude/harnesses`. Tags: **[M]** measured (offline, in a recorded
request or by the engine's own tokenizer), **[S]** read in documentation or source,
**[E]** estimate, **[?]** not known yet.

No model ran for anything in this document. The harnesses talked to a scripted mock
(`tools/harness_probe.py mock`) through the recorder (`tools/session_replay.py record`);
token counts come from `oxe-server --engine stub` (release `oxe-p1-b9ad2eb05c33`, Ornith
tokenizer and chat template, no weights). Real-model numbers are the job of tickets 025 and
026 (`handoff/runq/`).

Setup guides: [HARNESS-PI.md](HARNESS-PI.md), [HARNESS-DSH.md](HARNESS-DSH.md).
UI proposal: [HARNESS-UX.md](HARNESS-UX.md).

Any other OpenAI-compatible client (Aider, Continue, Cline, Zed, Open WebUI, …): see
`OPENAI-API.md`.

## 1. What each harness sends

| | OpenCode 2.2.6 (reference) | pi 1.5.1 | dsh 0.2.2-rc.1 (headless profile) |
|---|---|---|---|
| What it is | OpenCode CLI | `@earendil-works/pi-coding-agent` (mise `aqua:earendil-works/pi`) [S] | DeepSeek Harness; its LLM layer is `@deepseek-ai/dsh-llm-pi-ai`, the same pi-ai request builder as pi [S] |
| System prompt | 8610 chars [M] | 2301 chars, one `system` message [M] | 2380 chars, one `system` message [M] |
| Tools | 12 (build), 10 (subagent) [M] | 4: read, bash, edit, write [M] | 23 [M] (the web profile has 31 [S, earlier note]) |
| Prefix before the first user message | 5490 tokens (build), 4515 (subagent) [M] | 1507 tokens [M] | 5393 tokens [M] |
| Same prefix in another project directory | [?] not probed | first 1497 of 1507 tokens: the working directory is the last line of the system prompt [M] | all 5393 tokens: directory and policy text come in a `user` message after the prompt [M] |
| `temperature`, `top_p`, `seed` | `temperature: 0` from the provider `body` [M] | none sent [M] | none sent [M]; cannot be configured [S] |
| Length limit | `max_tokens: 2048` [M] | `max_completion_tokens: 4096` [M] | `max_tokens: 4096` [M] |
| Streaming | `stream: true` with usage [M] | `stream: true`, `stream_options.include_usage` [M] | same [M] |
| Other body keys | `cache_prompt`, `chat_template_kwargs` [M] | `store: false` (ignored by oxe) [M] | none [M]; `chat_template_kwargs` only when the model entry declares reasoning efforts [M] |
| Tool calls | OpenAI `tool_calls` | OpenAI `tool_calls`; echoed with `content: null`, compact JSON arguments, the server's call id [M] | same shape [M] |
| Tool results | `tool` messages | `tool` message, string content, no `name` [M] | same [M] |
| User message content | string | array of `text` parts [M] | string [M] |
| History | re-sent; 2 of 24 build-agent follow-ups rewrote or restarted it [M] | re-sent verbatim, also across `--continue` [M] | re-sent verbatim, also across `--session-id` [M] |
| Side requests | title (disabled in the live config) | none seen [M] | one title request per session (481-char system prompt, no tools), sent in parallel with the first tool round [M] |
| Later `system` messages | none | none [M] | none: the runtime context is a `user` message at position 2 and stays there [M] |

What this means for the five speed conditions:

1. **Prefix reuse.** Both prefixes are stable token for token inside a session [M]. dsh's is
   also stable across project directories. pi's differs only in its last 10 tokens, and the
   engine stores prefixes cut down to the 256-token prefill grid, so the stored entry (1280
   tokens) is shared by all directories [M for the token counts, E for the store behaviour].
   A project `AGENTS.md` goes into pi's system prompt [S]; where exactly is [?], so a project
   with one may get its own entry.
2. **Temperature 0.** Neither harness sends a temperature. The controller sets 0 when the key
   is missing (`oxe_runtime.request_defaults`), so both decode greedily and speculative
   decoding applies. No per-client override is needed. A client that sends
   `"temperature": null` used to slip through and sample at the engine's 0.8: fixed here.
3. **Stable rendering.** Inside one tool loop, turn N+1's prompt extends turn N's prompt plus
   reply exactly, for both harnesses [M: 100% of the previous tokens reusable]. At every
   **new user message** it does not, for any harness, OpenCode included. See section 2.
4. **Tool-call format.** Both use plain OpenAI `tool_calls`; the engine renders them as Qwen
   XML and parses the model's XML back. Nothing to adapt [M: the stub server accepted every
   recorded request and rendered it].
5. **Concurrent requests.** dsh sends its title request while the first agent request is
   running [M]. The controller queue (FIFO) handles it; without the queue it would be a 503.
   With parked sessions on, the title request does not cost the agent its conversation state.

## 2. The finding that matters most: `<think>` blocks at each new user message

The Qwen3.6 / Ornith chat template writes an empty `<think>\n\n</think>\n\n` block in front of
every assistant turn of the *current* tool loop when thinking is off, and drops those blocks
from all turns *before the newest user message*. So when the user (or the harness) adds a
user message, the rendering of the whole previous tool loop changes and its tokens cannot be
reused. The engine then goes back to an earlier checkpoint.

Measured:

- pi and dsh probes: the follow-up prompt after a second user message shares only the tokens
  up to the first assistant turn of the previous loop (pi 1525 of 1632, dsh 5618 of 5832) [M].
- The recorded real OpenCode session `ornith-m2-20261110` (oxe 649ac806, build agent, 24
  follow-up turns): the engine evaluated **16 300 prompt tokens in 131 s** (124 tok/s) [M,
  from the recorded timings]. A perfect cache needs 11 692 of them with today's rendering
  [M, offline]. One turn went back to the system-prompt anchor and evaluated 4114 tokens.
  OpenCode triggers this more often than expected because it injects user messages in the
  middle of a tool loop (the "Instructions from: AGENTS.md" note after a read) [M].

The template has a switch for exactly this: `chat_template_kwargs.preserve_thinking = true`
keeps the block on every assistant turn. With it:

| Recording | Prompt tokens a perfect cache must evaluate, follow-up turns | with `preserve_thinking` |
|---|---|---|
| pi probe (5 follow-ups) | 217 | 122 [M] |
| dsh probe (5 follow-ups) | 512 | 310 [M] |
| OpenCode M2, build agent (24 follow-ups) | 11 692 | 5 887 [M] |

Every pi and dsh follow-up prompt then extends the previous prompt and reply completely [M].
For the OpenCode session this is about half the unavoidable prefill, roughly 47 s of the 131 s
at the recorded rate [E]; the real saving can be larger because the engine currently does
worse than the perfect cache.

It changes what the model reads: earlier turns keep four extra tokens each (`<think>`, two
newlines, `</think>`), the same form the model already sees inside a tool loop. That is the
template's own documented variant, but it is a different prompt, so it needs the usual
quality and exactness gates before it becomes a default.

What is in this branch: `runtime.json` profile key
`"request_defaults": {"preserve_thinking": true}`, **off unless set**, applied by
`oxe_runtime.request_defaults` (a value sent by the client wins). `session_replay.py replay
--template-kwarg preserve_thinking=true` runs the A/B arm; `harness_probe.py analyze
--template-kwarg ...` gives the offline number. Ticket 026 measures it on Ornith.

## 3. Compatibility fixes in this branch

All in the lab repo; the engine is unchanged.

| Change | File | Triggered by pi or dsh today? |
|---|---|---|
| `temperature: null` / `cache_prompt: null` count as missing, so the defaults 0 / true apply (the engine would otherwise sample at 0.8 and stop speculating) | `oxe_runtime.py`, `harness_compat.py` | no [M]; clients built on other SDKs do send nulls |
| A `system`/`developer` message after the conversation started becomes a `user` message in `<system-reminder>` tags. The template answers such a request with HTTP 400 "System message must be at the beginning" [M on the stub] | `harness_compat.py` | no [M] |
| `tool_choice` naming one function becomes `required` with only that tool. The engine answers the object form with HTTP 400 [M on the stub] | `harness_compat.py` | no [M] |
| Image, audio or file content parts become a text note. The engine answers them with HTTP 400 [M on the stub] | `harness_compat.py` | no with a text-only model entry [M] |
| Opt-in `preserve_thinking` request default (section 2) | `oxe_runtime.py` | all harnesses |
| Harness registry: which prefixes were seen, expected warm state, primers; `/_local/status` gains `harnesses` and `request_compat`; `POST /_local/harness/prepare` and `/_local/harness/forget` | `harness_registry.py`, `controller.py` | all harnesses |

The rewrites are deterministic and local to one message, so a longer conversation renders
its earlier part identically (tested). They apply only to requests for the Rust engine.

Not fixed, by decision: a request that ends with an assistant message (prefill) is still an
HTTP 400 from the engine. Neither harness sends one [M].

Tests: `tests/test_harness_compat.py` (compat, registry, prepare), `tests/test_harness_probe.py`
(mock and analysis), one added case in `tests/test_session_replay.py`.

## 4. Cache strategy for several harnesses

### 4.1 The persistent prefix store (disk)

One entry per distinct system-prompt-plus-tools token sequence, per engine identity (model
SHA, server binary SHA, state layout, ubatch). Entry size on Ornith, from the five entries in
the live store [M]: about **61 MiB + 20.5 KB per token** (1792 tokens: 97 MiB; 4352: 147 MiB;
5376: 167 MiB). Prefixes shorter than 1024 tokens are not stored.

| Prefix | Stored tokens (256 grid) | Entry size | Cold build at 124 tok/s |
|---|---|---|---|
| OpenCode build | 5376 | 167 MiB [M] | 44 s [E] |
| OpenCode subagent | 4352 | 147 MiB [M] | 36 s [E] |
| dsh headless | 5376 | 167 MiB [E] | 43 s [E] |
| dsh web profile (about 9.1K tokens) | 8960 | 236 MiB [E] | 73 s [E] |
| pi | 1280 | 86 MiB [E] | 12 s [E] |
| titles (OpenCode 429, dsh 120 tokens) | not stored | 0 | under 4 s |

All three harnesses on one model need about 0.6 to 0.8 GiB. Two Rust-engine models (Ornith,
Qwen3.6) need 1.2 to 1.6 GiB. After an engine promotion every old entry is dead weight until
the LRU removes it (two such entries, 316 MiB, are in the live store now [M]), so the peak is
about twice that.

**The 4 GiB default is right.** It holds two models times three harnesses plus one stale
generation, with room for a few per-project pi variants. Raising it buys nothing measurable;
lowering it below 2 GiB would start evicting live harness prefixes after a promotion. Disk is
not the constraint (378 GiB free [M]).

Two engine changes would make the budget behave better; neither is needed for correctness
(proposals, not implemented):

- Evict entries of another server binary for the same model before any entry of the current
  identity.
- Report the store entry used or written in `timings` and list the entries in `/props`, so
  the controller's "warm" is exact instead of expected (section 5 of HARNESS-UX.md).

### 4.2 Should warm prefixes be pre-built at controller start? No.

- The controller starts without a model. Pre-building means loading 16 to 21 GiB and taking
  the inference lease at login, 12 to 73 s of prefill per prefix, for a harness that may not
  be used that day.
- The store is persistent. A prefix is built once per engine binary, model and harness
  version, then loaded from disk. Controller restarts do not lose it.

Pre-build instead at the two moments it pays:

1. **On request**: `POST /_local/harness/prepare {"harness": "dsh"}` (the "Prepare" button).
   It sends a 1-token primer with the recorded system prompt and tools through the normal
   queue. It needs one earlier real request from that harness, which is where the primer
   comes from.
2. **After an engine or model promotion**, as the last step of the promotion procedure, for
   every harness in the registry: the promotion is what invalidates all entries, and the
   model is already loaded for the smoke test.

### 4.3 Parked sessions across a harness switch (RAM)

Two harnesses share almost no prompt prefix, so switching harness replaces the live
conversation. With `--session-ram-bytes` set (2 GiB in production) the engine first parks the
live session (state plus checkpoints) and restores it as a byte copy when that harness
continues. A parked session costs about the same as a store entry of its length [E, same
state format]: on Ornith 8K tokens 225 MiB, 16K 390 MiB, 32K 700 MiB. Measured on Qwen3.6:
32 183 tokens parked in 817 MiB and resumed in 42 ms [M, ticket 008]. 2 GiB holds about five
16K-token sessions or two full 32K ones, with LRU eviction. Keep 2 GiB: three harnesses
alternating is covered, and the memory admission rule (cache + dense + 3 GiB) has no room to
give away.

The gap is the idle unload. After 120 s without a request the model process exits and every
parked session is gone; the next request from any harness prefills its whole conversation
from the stored prefix. Working with two harnesses makes pauses in each of them longer.
`--session-cache DIR` writes parked sessions to disk on eviction and shutdown (8 GiB, 24 h
TTL by default) and is off in production. Recommendation: qualify it and turn it on. It is a
production argument change, so it is a decision for the owner after ticket 026 shows the numbers.

### 4.4 Budget summary

| Store | Where | Default | Recommendation |
|---|---|---|---|
| Prefix store | disk, `~/.cache/local-ai/snapdragon-engine/prefix` | 4 GiB | keep |
| Parked sessions | RAM | 2 GiB (production) | keep |
| Session cache | disk | off | turn on after qualification, 8 GiB, 24 h |
| Harness registry | disk, `~/.local/state/local-ai/harness-prefixes` | 24 prefixes, about 30 KB each | new in this branch |

## 5. Reproducing the probe

```
oxe-server --engine stub --model <Ornith gguf> --port 18199 --reasoning off &
python3 tools/harness_probe.py mock --port 18210 &
python3 tools/session_replay.py record --listen 127.0.0.1:18301 --upstream http://127.0.0.1:18210 --out <dir outside git>
# run the harness with an isolated HOME and its base URL set to http://127.0.0.1:18301/v1
python3 tools/harness_probe.py analyze <dir>/recording.jsonl [<second session>/recording.jsonl] \
    --server http://127.0.0.1:18199 [--template-kwarg preserve_thinking=true] [--out report.json]
```

`analyze` on two recordings of the same harness in different directories also reports
whether the stored prefix survives a change of project. The stub server and the mock are CPU
jobs: run them in DEV windows only.

## 6. Open points

- The `User-Agent` of pi and dsh is [?] (the probe did not keep the headers). Harness
  detection uses the first words of the system prompt, which is measured.
- dsh: the lean preset from the earlier overlay had no effect in the headless profile (still
  23 tools) [M]; why is [?]. The web profile was not probed, because the running dsh web
  service was off limits.
- pi with a project `AGENTS.md`, and pi with thinking enabled, were not probed.
- Reply reconstruction in the offline analysis assumes the model writes tool calls in the
  template's canonical form. Ticket 026 measures the real `cache_n`.
