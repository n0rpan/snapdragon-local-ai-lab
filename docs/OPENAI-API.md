# The local OpenAI-compatible API

Every local model is served on one loopback address in the OpenAI wire format, so any
harness or client library that can talk to "an OpenAI-compatible server" can use it.
OpenCode, pi and dsh have their own pages (`HARNESSES.md`, `HARNESS-PI.md`,
`HARNESS-DSH.md`); this page is for everything else.

Tags: [M] measured here without a model (stub engine), [S] from the harness's own
documentation or source, not run here, [?] open.

## 1. Connection

| Setting | Value |
|---|---|
| Base URL | `http://127.0.0.1:8092/v1` |
| API key | none by default. Clients that insist on one can send any text (`local`). |
| Model id | `local-default`, or a full name from the table below |
| Context | Ornith and Qwen3.6: 131 072 tokens; Flash: 16 384 (prompt and reply together; `context_size` in `GET /_local/status`) |
| Requests at a time | one. Further requests wait in a queue for up to 120 s, then get `503` with `Retry-After: 5`. |

The server listens on `127.0.0.1` only. There is no setting to change that (section 6).

**Port 8092 is the only supported client entry.** The engine ports behind it (8093 and 8094)
are internal: they skip the API key, the CORS rule, the queue and the ledger. From v0.1.1 the
controller starts the Rust engine with a random token that is new on every launch (a 0600
file in the user's runtime directory) and only the controller sends it. The engine also refuses
browser requests (any `Origin` header or a foreign `Host`). Do not point a client at 8093 or 8094.

Check it from a shell:

```
curl http://127.0.0.1:8092/v1/models
curl http://127.0.0.1:8092/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model": "local-default", "messages": [{"role": "user", "content": "Hello"}]}'
```

With the official Python client:

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8092/v1", api_key="local")
reply = client.chat.completions.create(model="local-default",
                                       messages=[{"role": "user", "content": "Hello"}])
```

The first request after a pause loads the model (10 to 60 s). Listing models never loads
one.

## 2. Model ids

| Id | Model | Notes |
|---|---|---|
| `local-default` | the model selected in the panel (Ornith today) | follows the selection |
| `local-heavy` | Flash (`qwen3.8-flash-next-local`) | the "think hard" model; slower; listed only while Flash is qualified |
| `ornith-35b-a3b-local` | Ornith 1.5 35B A3B | |
| `qwen3.6-35b-a3b-local` | Qwen 3.6 35B A3B | |
| `qwen3.8-flash-next-local` | Qwen 3.8 Flash-Next | same as `local-heavy` |
| `qwen-local` | same as `local-default` | older name, kept for existing OpenCode configs |

- `GET /v1/models` lists them in this order. Alias rows carry `alias_of` (the full name
  they point at); `qwen-local` also carries `legacy: true`.
- A `provider/` prefix in the litellm style is accepted: `openai/ornith-35b-a3b-local`,
  `openai/local-default`.
- The reply's `model` field is the full name of the model that answered.
- An unknown id gets `404` with `code: "model_not_found"`. Some harnesses send a second,
  fixed model name for side requests (titles, summaries). Either configure that model to
  the same id, or set `"openai_api": {"unknown_model": "default"}` in `runtime.json` to
  serve unknown ids with the selected model.
- **Only one model is loaded at a time.** A request for another id unloads the current
  model and loads the other (10 to 60 s, and the cached conversation is gone). Give a
  harness one id for all its roles.
- More aliases: `"openai_api": {"model_aliases": {"local-fast": "qwen3.6-35b-a3b-local"}}`.
  A `null` target removes a built-in alias. An alias cannot replace a full name.

Aliases are fixed names. Choosing a model per request (routing, escalation to Flash) is a
separate piece of work in the controller and is not part of this table.

### Flash limits

Flash is qualified for temperature 0, at most 2048 output tokens and thinking off. For
generic clients the controller brings a request into that range instead of refusing it:
`max_completion_tokens` becomes `max_tokens`, a limit above 2048 becomes 2048, another
temperature becomes 0, `reasoning_effort` is dropped. Each change is counted in
`request_compat` of `GET /_local/status`. Not yet run against the real Flash engine [?].

## 3. What is supported

Measured through the controller in front of `oxe-server --engine stub` (the Rust engine's
real request and reply code, no model) with the requests the `openai` Python client sends
[M]. Reproduce: section 8.

| Endpoint | Feature | Status | Detail |
|---|---|---|---|
| `/v1/models` | list | supported | |
| `/v1/models/{id}` | retrieve | supported | new |
| `/v1/chat/completions` | non-streaming | supported | |
| `/v1/chat/completions` | streaming (SSE, `[DONE]`) | supported | |
| `/v1/chat/completions` | `usage` block | supported | with `prompt_tokens_details.cached_tokens` |
| `/v1/chat/completions` | `stream_options.include_usage` | supported | final chunk: `choices: []` plus `usage` |
| `/v1/chat/completions` | `max_tokens`, `max_completion_tokens` | supported | |
| `/v1/chat/completions` | `stop` | supported | |
| `/v1/chat/completions` | `temperature`, `top_p`, `seed` | supported | default temperature is 0 when none is sent |
| `/v1/chat/completions` | `logprobs`, `top_logprobs` | supported | |
| `/v1/chat/completions` | `tools`, `tool_choice` auto / none / required / named function | supported | tool calls come back as `tool_calls`, also streamed; `finish_reason: "tool_calls"` |
| `/v1/chat/completions` | `parallel_tool_calls`, tool results in the history | supported | |
| `/v1/chat/completions` | `system` and `developer` roles, text content parts | supported | |
| `/v1/chat/completions` | system message after the first turn | supported | |
| `/v1/chat/completions` | `n` > 1 | partial | one choice is returned |
| `/v1/chat/completions` | `frequency_penalty`, `presence_penalty`, `logit_bias` | partial | accepted and ignored |
| `/v1/chat/completions` | image, audio, file parts | partial | replaced by a short text note; the models read text only |
| `/v1/chat/completions` | `response_format` `json_object` / `json_schema` | **missing** | `400`. Use a tool with the schema instead. |
| `/v1/completions` | non-streaming, streaming, `logprobs`, `echo`, `stop` | supported | new: translated to the engine's native route |
| `/v1/completions` | several prompts in one request | missing | `400` |
| `/v1/completions` | `suffix` (insert / fill-in-the-middle) | missing | `400` |
| `/v1/embeddings` | | **missing** | `404`, `code: "unsupported_endpoint"`. No embedding model is installed. |
| `/v1/responses` | | **missing** | `404`, same code |
| `/v1/rerank` | | missing | `404`, same code |
| errors | OpenAI shape `{"error": {"message", "type", "param", "code"}}` on every route outside `/_local` | supported | new |
| auth | optional API key | supported | new, off by default (section 5) |
| CORS | browser apps on another origin | supported, opt-in | new, off by default (section 5) |
| bind address | loopback only | fixed | |

The same table for the controller on `main` before this branch [M]: `/v1/completions`
answered `404`, `/v1/models/{id}` `503`, image parts `400`, errors were partly a bare
string (`{"error": "…"}`), unknown models had no `code`, and `/v1/embeddings` first loaded
the model and then answered `404`.

Unsupported endpoints are answered by the controller without loading a model. The status is
`404`, not `501`, because client libraries retry every 5xx.

Scope: the translation of `/v1/completions`, the leniency for `n`, images and late system
messages, and harness learning apply to the Rust engine models (Ornith and Qwen3.6 in
production). The llama.cpp fallback profiles are passed through to llama-server, which has
its own OpenAI routes. Flash accepts `/v1/chat/completions` only.

Extra fields in replies that OpenAI does not have: `timings` (llama.cpp's block, with
`cache_n` and `prompt_n`), and `reasoning_content` on a message when thinking is on.
Thinking is off unless the request sends `"chat_template_kwargs": {"enable_thinking": true}`.

## 4. Harnesses

None of the harnesses in this section was run here; the settings are from their
documentation [S]. What was verified is the wire format they use (section 3).

What they need from the server:

| Harness | Calls | Needs beyond plain chat | Works |
|---|---|---|---|
| Aider | `/v1/chat/completions`, streaming, through litellm | nothing; edits are plain text, no tools | yes |
| Continue (VS Code, JetBrains) | chat, streaming; tools in agent mode; `/v1/completions` for autocomplete | fill-in-the-middle for good autocomplete | chat, edit, agent: yes. Autocomplete: not recommended |
| Cline, Roo Code | chat, streaming, `include_usage` | a large context: the system prompt alone is over 10 000 tokens | yes on Ornith (32K) |
| Zed | chat, streaming, tools | | yes |
| LibreChat | `/v1/models`, chat, streaming; a title request per conversation | reach the host from Docker | yes with host networking |
| Open WebUI | `/v1/models`, chat, streaming; title, tag and follow-up requests | reach the host from Docker; CORS only for "direct connections" | yes with host networking |
| openai SDKs, LangChain, LlamaIndex | chat, completions | `json_schema` for structured output by default | yes with function calling as the structured-output method |
| Codex CLI | `/v1/responses` | the Responses API | **no** |
| Claude Code | Anthropic Messages API | another wire format | **no** |

Rules that hold for all of them:

- One model id for every role (main, weak, title, summary, autocomplete).
- Set the context window to the model's (131 072 for Ornith and Qwen3.6, 16 384 for Flash). A harness that assumes more
  sends prompts the server answers with `400` (`type: "exceed_context_size_error"`).
- Background requests (titles, tags, autocomplete) wait in the same queue as the answer
  the user is waiting for. Turn them off where the harness allows it.
- Do not put a date, time or random id at the *start* of a system prompt. The prefix cache
  reuses a prompt up to the first changed token (section 7).

### Aider

```
export OPENAI_API_BASE=http://127.0.0.1:8092/v1
export OPENAI_API_KEY=local
aider --model openai/ornith-35b-a3b-local --no-show-model-warnings
```

litellm strips the `openai/` prefix; the server also accepts it. Aider does not know the
model, so tell it the context size in `.aider.model.metadata.json`:

```json
{"openai/ornith-35b-a3b-local": {"max_input_tokens": 32768, "max_output_tokens": 8192,
  "input_cost_per_token": 0, "output_cost_per_token": 0, "litellm_provider": "openai", "mode": "chat"}}
```

Aider uses the main model as its weak model for unknown models; if you set `--weak-model`,
give it the same id.

### Continue

`~/.continue/config.yaml`:

```yaml
models:
  - name: Local Ornith
    provider: openai
    model: ornith-35b-a3b-local
    apiBase: http://127.0.0.1:8092/v1
    apiKey: local
    roles: [chat, edit, apply]
    capabilities: [tool_use]
    defaultCompletionOptions:
      contextLength: 32768
```

Leave the `autocomplete` role to another provider or off: it fires on every pause in
typing, each request waits behind the chat, and the server has no fill-in-the-middle.

### Cline and Roo Code

API provider "OpenAI Compatible"; base URL `http://127.0.0.1:8092/v1`; API key `local`;
model id `ornith-35b-a3b-local`; context window 32768; no image support; no computer use.
Their system prompt is long, so the first request on a fresh engine is slow once
(section 7) and the room left for the conversation is about 20 000 tokens.

### Zed

`settings.json`:

```json
"language_models": {
  "openai_compatible": {
    "Local AI": {
      "api_url": "http://127.0.0.1:8092/v1",
      "available_models": [{
        "name": "ornith-35b-a3b-local", "display_name": "Ornith (local)",
        "max_tokens": 32768, "max_output_tokens": 8192,
        "capabilities": {"tools": true, "images": false, "parallel_tool_calls": false, "prompt_cache_key": false}
      }]
    }
  }
}
```

Zed asks for the API key in its provider settings; any text works.

### LibreChat

`librechat.yaml`:

```yaml
endpoints:
  custom:
    - name: "Local AI"
      apiKey: "local"
      baseURL: "http://127.0.0.1:8092/v1"
      models: {default: ["local-default"], fetch: true}
      titleConvo: true
      titleModel: "current_model"
```

LibreChat calls the API from its server, not from the browser. In Docker the container
must share the host's network (`network_mode: host`), because the API listens on the
host's loopback address only.

### Open WebUI

Admin settings, Connections, OpenAI API: URL `http://127.0.0.1:8092/v1`, key `local`. In
Docker run it with `--network=host`. In Admin settings, Interface, set the task model to
"current model" or turn title, tag and follow-up generation off. Its document search uses
its own built-in embedding model, not this API. "Direct connections" (the browser calls the
API itself) need the origin in `cors_origins` (section 5).

### Client libraries

- LangChain: `ChatOpenAI(base_url=…, api_key="local", model="local-default")`. For
  structured output pass `method="function_calling"`; the default (`json_schema`) is
  answered with `400`.
- The `openai` client retries `503` on its own, which is the right behaviour while another
  request is running.

## 5. Optional settings

All off by default. They live under `"openai_api"` in `runtime.json`
(`~/.config/local-ai/llm/runtime.json`), read on every request, no restart needed.

```json
"openai_api": {
  "api_key_file": "/path/to/local-ai/llm/api-key",
  "cors_origins": ["http://localhost:3000"],
  "model_aliases": {"local-fast": "qwen3.6-35b-a3b-local"},
  "unknown_model": "default"
}
```

- **API key.** `api_key_file` (first line of the file; keep it 0600) or the environment
  variable `LOCAL_AI_API_KEY` of the controller service. With a key set, every API route
  needs `Authorization: Bearer <key>` (`x-api-key` and `api-key` headers also work) and
  answers `401` with `code: "invalid_api_key"` otherwise. Not covered: `/_local/*` (the
  panel's control routes, which the controller answers itself) and `/health`. The built-in
  chat pages do not send a key, so they stop working while one is set. A key file that is
  configured but unusable (missing, unreadable, not an absolute path) closes the API rather
  than opening it, and a `runtime.json` that is briefly unreadable does not drop the key.
  The key is never passed on to an engine.
- **Scope.** The API key and CORS settings cover port 8092 only. The engine ports are
  internal (section 1): v0.1.0 engines answer any local process and any web page there, so
  update to v0.1.1, where the engine needs the controller's per-launch token and refuses
  browser requests.
- **CORS.** Without `cors_origins` a browser page on another origin is refused (`403`), as
  before: any web page you visit could otherwise use the model. List exact origins
  (`scheme://host:port`). `"*"` is honoured only together with an API key. Allowed origins
  get preflight answers, `Access-Control-Allow-Origin` and private-network access. The
  control routes (`/_local/*`) stay same-origin whatever is configured.

## 6. Bind address

`127.0.0.1` only, fixed in the controller; the engines behind it also bind to loopback, and
for the Rust engine the controller refuses profile arguments that would change that.
Consequences:

- Other machines cannot reach the API. Use an SSH tunnel (`ssh -L 8092:127.0.0.1:8092 host`).
- Docker containers cannot reach it through `host.docker.internal`; use host networking.

A LAN bind is not offered. If it is ever wanted it should require an API key and be a
decision of its own.

## 7. Caching for a harness the controller does not know

The Rust engine stores every system-prompt-plus-tools prefix of at least 1024 tokens on
disk, for any client, keyed by the token sequence. Nothing has to be registered. Measured
with two made-up harnesses (3 900 and 2 100 prompt tokens) on the stub engine, which runs
the real cache code [M] (`tools/openai_api_check.py cache`):

| Request | Reused tokens | Evaluated tokens |
|---|---|---|
| A, first request ever | 0 | 3882 |
| A, next request in the session | 3840 | 71 |
| A, new session, engine still loaded | 3840 | 42 |
| B (another unknown harness), first request | 0 | 2103 |
| A again after B | 3840 | 38 |
| A, new session after an engine restart (idle unload) | 3840 | 39 |
| B after the restart | 2048 | 54 |
| A with a different *last* line in the system prompt | 3840 | 47 |
| A with a different *first* line in the system prompt | 0 | 3883 |

So an unknown harness pays for its prefix **once per engine version**: the first request
after the harness is first used, after an engine or model update, or after "Clear caches".
Every later request, session, harness switch and restart reuses 97 to 99 % of the prefix
without the controller knowing the harness. At the 124 tokens/s cold prefill rate used in
`HARNESSES.md` section 4.1, that one time is about 31 s for 3 900 tokens and about 90 s for
a Cline-sized prompt of 11 000 tokens [E].

### What learning adds

Learning does not make any request faster that the store did not already make fast. It
adds three things:

1. **Prepare.** The controller keeps the harness's system prompt and tool list, so it can
   rebuild the stored prefix on request (a button, or "Prepare all" after an engine
   update) instead of inside the user's first request. Measured [M]: first request after
   an emptied store without Prepare, 0 tokens reused and 3876 evaluated; with Prepare
   3840 reused and 36 evaluated, and Prepare itself evaluated 3886. The cost moves; it
   does not shrink.
2. **A chip.** The harness shows up in the panel with a name and a ready / needs warm-up
   state, like OpenCode, pi and dsh.
3. **A diagnosis.** A client whose system prompt changes on every request is shown as
   `stable: false`. That is the case caching cannot help, and the usual reason a harness
   feels slow.

### How it works

Off by default. Two ways to opt in:

- all unknown clients: `POST /_local/harness/learning {"enabled": true}` (writes
  `"harness_learning": {"enabled": true}` to `runtime.json`);
- one client: `POST /_local/harness/learn {"harness": "learned-1a2b3c4d"}` for a candidate
  from the list.

With learning off, an unknown client is only counted in memory by the controller: sizes,
request counts, a fingerprint, the product name from its `User-Agent`. No text, and the
controller writes nothing to disk. This is about the controller's registry. The engine's
own prefix store is separate and works for every client regardless of this setting: it
holds the evaluated state of the system-prompt-plus-tools tokens (and their token ids), in
the private cache directory, which is what makes the table above possible.

A client is learned when the same system prompt and tool list arrived `min_requests` times
(default 3), byte for byte, and they are at least `min_prefix_chars` long (default 3000;
shorter prefixes are below what the engine stores). Then the controller writes a primer:
the leading system messages and the tool schemas, nothing else. It is the O-9 rule
unchanged: never a user turn, tool result or reply; file mode 0600 in a 0700 directory
(`~/.local/state/local-ai/harness-prefixes`).

- Name: `learned-<8 hex>`, a hash of the first 256 characters of the system prompt and the
  tool names. A project path or date further down the prompt does not change the name.
- Label: the `User-Agent` product name, changeable with `POST /_local/harness/label`.
- At most 4 prefixes per learned client (for example one per project directory), least
  recently used dropped first; 24 prefixes in the registry overall, as before.
- "Forget" deletes the primers. With `"ignore": true` the client is also never learned
  again until it is approved by name.
- Known harnesses (OpenCode, pi, dsh) are unchanged: recorded from their first request.
- Rust engine models only, like the rest of the registry.

### Endpoints (for the panel)

Polling `GET` routes never load a model. `POST` bodies are JSON of at most 1024 bytes.
These routes are same-origin only and are not covered by the API key.

| Call | Body | Reply |
|---|---|---|
| `GET /_local/harnesses` | | `{"harnesses": {...}, "harness_learning": {...}, "request_compat": {...}}`; the same three keys are in `GET /_local/status` |
| `POST /_local/harness/learning` | `{"enabled": true, "min_requests": 3}` (`min_requests` optional, 1 to 100) | the settings now in force; `400` on a bad body |
| `POST /_local/harness/learn` | `{"harness": "learned-…"}` | `{"approved": "learned-…"}`; `404` if that client was not seen since the controller started |
| `POST /_local/harness/label` | `{"harness": "learned-…", "label": "Aider"}` | `{"labelled": true}`; `400` otherwise. Letters, digits, space and `._+-`, 40 characters. |
| `POST /_local/harness/prepare` | `{"harness": "learned-…"}` or `{"harness": "*"}` (every harness that is not warm), optional `"model"` | per prefix `cache_n`, `prompt_n`, `seconds`; `409` when there is nothing to prepare. Loads the model. |
| `POST /_local/harness/forget` | `{"harness": "learned-…", "ignore": true}`; no `harness` forgets all | `{"forgotten": …, "ignored": …}` |

`harnesses` (existing shape; learned clients add two fields):

```
"learned-1a2b3c4d": {"learned": true, "label": "Zed", "warm": 1, "last_used": 1794345000.1,
                     "prefixes": [{"key": "c713f2bb9235", "model": "ornith-35b-a3b-local", "warm": true,
                                   "system_chars": 9120, "tools": 18, "requests": 14,
                                   "last_used": 1794345000.1, "last_cache_n": 3840, "last_prompt_n": 52}]}
```

`harness_learning`:

```
{"enabled": false, "min_requests": 3, "min_prefix_chars": 3000, "persistent": true, "ignored": [],
 "candidates": [{"harness": "learned-9f8e7d6c", "agent": "Aider", "requests": 12,
                 "distinct_prefixes": 1, "consistent_requests": 12, "stable": true,
                 "system_chars": 5200, "tools": 0, "prefix_chars": 5200, "large_enough": true,
                 "approved": false, "ignored": false, "first_seen": …, "last_used": …,
                 "last_cache_n": 1280, "last_prompt_n": 96}]}
```

Suggested chips: a learned harness is a normal chip with its `label` (fall back to the
name). A candidate with `stable` and `large_enough` is a dashed chip "Unknown client
(Aider) · Learn"; with `stable: false` it reads "prompt changes every request, cannot be
cached" and has no button. Candidates are kept in memory only, so they are gone after a
controller restart until the client talks again.

## 8. Reproducing the checks

```
# the engine's real request code without a model (DEV windows only)
oxe-server --engine stub --model <any gguf of the model family> --alias ornith-35b-a3b-local --port 18193 &
# the controller in front of it, with model start replaced by a marker
PYTHONPATH=src python3 tests/openai_api_stack.py --backend-port 18193 --port 18292 &

python3 tools/openai_api_check.py audit --base-url http://127.0.0.1:18292 --model local-default --markdown
python3 tools/openai_api_check.py sdk   --base-url http://127.0.0.1:18292 --model local-default   # needs `pip install openai` in a venv
python3 tools/openai_api_check.py cache --server <oxe-server> --gguf <gguf>

PYTHONPATH=src python3 -m unittest tests.test_openai_api
OXE_STUB_SERVER=<oxe-server> OXE_STUB_GGUF=<gguf> PYTHONPATH=src python3 -m unittest tests.test_openai_api.Live
```

`audit` also runs against the live controller (`--base-url http://127.0.0.1:8092`); it then
loads a model and sends about 45 short requests.

## 9. Open points

- `response_format` needs constrained sampling in the engine. Until then: tools.
- `/v1/responses` is what Codex CLI and newer OpenAI agent SDK defaults need. Not built.
- `/v1/embeddings`: no embedding model is installed. A small one on its own port would be
  a separate service, not a job for the LLM engine.
- `local-heavy` through a generic client has not been run against the real Flash engine.
- The harness notes in section 4 are from documentation; none was run against this server.
- `User-Agent` labels are whatever the client sends; several harnesses share a library's
  default (`OpenAI/Python`), which is why the label can be changed.
