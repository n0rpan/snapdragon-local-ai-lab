# pi with the local models

pi 1.5.1 (`@earendil-works/pi-coding-agent`). Background and measurements:
[HARNESSES.md](HARNESSES.md). Tags as there: [M] measured offline, [S] from pi's docs, [?] open.

## 1. Configure

pi reads custom providers from `<agent-dir>/models.json`. The agent dir is `~/.pi/agent`
unless `PI_CODING_AGENT_DIR` is set [S, pi `docs/configuration.md`].

Copy [`examples/harnesses/pi/models.json`](../examples/harnesses/pi/models.json) there, or
merge its `local-ai` provider into the file you already have:

| Key | Value | Why |
|---|---|---|
| `baseUrl` | `http://127.0.0.1:8092/v1` | the Local AI controller |
| `api` | `openai-completions` | chat completions with OpenAI tool calls [M] |
| `apiKey` | any non-empty string | the controller ignores it |
| model `id` | `ornith-35b-a3b-local` (default), `qwen3.6-35b-a3b-local` | the controller's aliases; an unknown id is a 404 |
| `contextWindow` | 32768 for Ornith, 16384 for Qwen3.6 | must match the server's context, or pi compacts too late and the server answers 400 |
| `maxTokens` | 4096 | sent as `max_completion_tokens`, which the engine accepts [M] |
| `reasoning` | `false` | the local profiles run with thinking off |
| `input` | `["text"]` | the engine is text-only |

Nothing else is required. This exact file shape was used in the probe [M].

## 2. Run

```
pi --provider local-ai --model ornith-35b-a3b-local            # interactive
pi --provider local-ai --model ornith-35b-a3b-local -p "task"  # print mode
pi --provider local-ai --model ornith-35b-a3b-local -c -p "…"  # continue the last session of this directory
```

## 3. Recommended parameters

- **Temperature**: leave it unset. pi sends none [M] and the controller makes it 0, which is
  what speculative decoding needs. If you want it pinned on the pi side as well, a model
  entry accepts `"samplingParams": {"temperature": 0}` [S, `docs/models-configuration.md`;
  not probed]. Do not set a non-zero temperature for agent work: it switches speculation off.
- **Thinking**: keep `reasoning: false`.
- **Tools**: the four built-ins (read, bash, edit, write) are a 1507-token prefix [M], the
  smallest of the three harnesses. Extensions and extra tools grow it; every change to the
  tool list is a new stored prefix (about 12 s to build once [E]).
- **System prompt stability**: the working directory is the last line of pi's system prompt
  [M]. The stored prefix still covers all directories. Content you add through
  `APPEND_SYSTEM.md` or a project `AGENTS.md` becomes part of the prompt [S]; keep volatile
  text (dates, git status) out of those files, or each change costs a prefix rebuild.
- **Optional compat keys** [S, not probed]: `"compat": {"supportsStore": false,
  "maxTokensField": "max_tokens"}` stops pi from sending `store` and switches the length
  field. Neither changes behaviour on this server.

## 4. What to expect

- The first request ever (per engine version) builds the prefix: about 12 s [E]. After that
  it is loaded from disk.
- Inside a task, each step evaluates only the new tool result [M offline].
- A new prompt in the same session re-evaluates the previous task's tool loop unless the
  profile has `preserve_thinking` on (HARNESSES.md section 2).
- Switching between pi and another harness keeps both conversations while the model stays
  loaded (parked sessions). After the idle unload (120 s) the conversation is prefilled again
  from the stored prefix.

## 5. Check that it works

`curl -s http://127.0.0.1:8092/_local/status` shows `harnesses.pi` after the first request,
with `last_cache_n` (tokens reused) and `last_prompt_n` (tokens evaluated) of the latest one.
On the second step of a task `last_prompt_n` should be small (tens to hundreds).
