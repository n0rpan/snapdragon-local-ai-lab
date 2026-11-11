# dsh (DeepSeek Harness) with the local models

dsh 0.2.2-rc.1. Background and measurements: [HARNESSES.md](HARNESSES.md). Tags as there:
[M] measured offline, [S] from the package's types and docs, [?] open.

## 1. Configure

dsh keeps profiles in `<DSH_HOME>/profiles/<name>/` (`DSH_HOME` defaults to `~/.dsh`) [S].
A profile is `package.json` (the bundles), `cordis.yml` and your overrides in
`cordis.patch.yml`. Provider routes belong to the `llm-pi-ai` row.

1. Put the rows of
   [`examples/harnesses/dsh/cordis.patch.yml`](../examples/harnesses/dsh/cordis.patch.yml)
   into the profile's `cordis.patch.yml`. **A patch row replaces the whole config of its id**
   [S, M]: keep your other providers (for example `openrouter` with its `apiKeyEnv`) inside
   the same `llm-pi-ai` row, or they disappear from the profile.
2. `export LOCAL_AI_API_KEY=local` in the environment dsh starts from. dsh wants a key
   variable; the controller ignores the value.
3. Check the result without starting anything: `dsh --profile <name> --dump-config`.

Do **not** run `--from-default-profile` on a profile you have edited: it rewrites the profile
from the shipped template every time, `--dump-config` included, and discards the patch [M].
Use it once to create a new profile, for example the headless one used in the probe:
`dsh --profile local --from-default-profile headless --dump-config`, with
[`examples/harnesses/dsh/package.json`](../examples/harnesses/dsh/package.json).

| Key | Value | Why |
|---|---|---|
| `api` | `openai-completions` | chat completions with OpenAI tool calls [M] |
| `baseURL` | `http://127.0.0.1:8092/v1` | the Local AI controller |
| `timeoutMs`, `streamIdleTimeoutMs` | 1800000 | a cold long prompt prefills for minutes; a queued request waits |
| `compat.maxTokensField` | `max_tokens` | [M] |
| `compat.supportsStore`, `supportsDeveloperRole` | `false` | no `store` key, no `developer` role [M] |
| model `contextWindow` | 32768 for Ornith, 16384 for Qwen3.6 | the server's real context. The earlier overlay said 65536; with that dsh compacts too late and the server answers 400 |
| model `reasoningEfforts` | `false` | thinking off. With `false`, dsh sends no `chat_template_kwargs` at all [M]; the controller adds `enable_thinking: false` itself, so `thinkingFormat` and `chatTemplateKwargs` are not needed |
| `agent-default-model` | `local-ai` / `ornith-35b-a3b-local` | otherwise new sessions start on the shipped default model [M] |

## 2. Run

```
dsh --profile local "task"                        # headless: answer on stdout
dsh --profile local --session-id <id> "next"      # same session; the id is the directory name under <DSH_HOME>/sessions/<project>/
```

A profile with the web-app bundle does not take a task on the command line [M]; use the web
UI's `/model` to pick `local-ai` there.

## 3. Recommended parameters

- **Use Ornith (32K).** The headless prefix is 5393 tokens [M], the web profile's about 9.1K
  [S, earlier note]. On the 16K Qwen3.6 profile the web prefix leaves little room.
- **Temperature**: nothing to set. dsh sends none [M] and has no way to send one [S]; the
  controller makes it 0.
- **Keep the tool set fixed.** 23 tools are 20 KB of schema and 5393 tokens. Building that
  prefix takes about 43 s once per engine version [E]; afterwards it loads from disk. Each
  different tool list (another profile or preset) is another stored prefix of about 170 MiB.
- **Lean preset**: the preset rows of the earlier overlay composed but did not change the
  request in the headless profile (still 23 tools) [M]. Until that is understood, do not
  count on it to shrink the prefix.
- **Title request**: dsh asks for a session title in parallel with the first step [M]. The
  controller queues it. It is one short extra request per session.

## 4. What to expect

- The directory and policy text is a `user` message after your prompt, not part of the
  system prompt [M], so every project and every session shares the same stored prefix.
- Inside a task, each step evaluates only the new tool result [M offline].
- A new prompt in the same session re-evaluates the previous task's tool loop unless the
  profile has `preserve_thinking` on (HARNESSES.md section 2).

## 5. Check that it works

`curl -s http://127.0.0.1:8092/_local/status` shows `harnesses.dsh` after the first request.
From the second session on, the first request should report `last_cache_n` of 5376 (the
stored prefix) and a `last_prompt_n` of a few hundred.
