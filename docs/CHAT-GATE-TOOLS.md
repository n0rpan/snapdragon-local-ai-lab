# CHAT gate tools: closed-loop session and fused-attention numeric reference

This page covers the two tools the v0.1.2 CHAT gate needs and that the older tools cannot provide.
`tools/long_session.py replay` stays what it is: a fixed, scripted history. It produces a cost curve,
not a proof of correctness. Its scripted assistant and tool turns are edits from the engine's point of view.

## 1. Closed-loop session runner (`tools/closed_loop_session.py`)

Each request carries the model's actual previous reply, the way OpenCode sends it:

- the returned `content`;
- the returned `tool_calls`, with arguments byte for byte as streamed;
- with `--reasoning-echo send` (the default), the returned `reasoning_content`;
- one seeded synthetic tool result per tool call.

With `--reasoning-echo drop`, no reasoning is sent back. This is what OpenCode 2.2.x was measured to do, so
the engine's ECHO must restore the reasoning itself.

Other behaviour:

- **Thinking** is `chat_template_kwargs.enable_thinking`, set with `--thinking` or `--no-thinking`.
  `preserve_thinking` is always true.
- **Compaction** follows the OpenCode 2.2.x layout:
  1. When the measured context reaches `--compact-at` (default: the target), the runner sends one summary
     request: [system, history to the cut, the instruction]. The instruction carries the markers
     `runtime/cache_turn.py` recognises.
  2. The history then becomes [system, checkpoint(summary), tail]. The tail is the last user turns within
     `--keep-tokens` (default 8192).
  3. The run then continues for `--post-compaction-turns` more user turns (default 3).
- **Requests** go to the loopback controller with `X-Local-AI-Origin: lab` and a unique `X-Session-Id` per
  run. They use temperature 0 and a fixed request seed. User and tool text is seeded with `--seed`.

The per-request receipt holds:

- prompt, cached and completion tokens;
- C3-eligible tokens (previous prompt + previous completion − 1) and the hit against them;
- `resume_src` and `cache_src`;
- the ECHO fields (`echo_n`, `echo_tokens`, `echo_fixed`, `roundtrip_miss`, `echo_miss_class`, `echo_misses`);
- the ledger's `miss_class`, `miss_basis`, `previous_reply` and `turn_kind`;
- TTFT, prefill ms, decode tok/s and the finish reason;
- tool-call count and validity.

It holds no prompt or reply text.

The ledger fields are read only. They come from `--ledger-dir`, matched by `origin=lab`, the message count
and the request start time. All of the following must hold to pass (exit 0):

| gate | rule |
|---|---|
| `hit_ratio_eligible` | at least 99% of C3-eligible tokens hit, over the append requests |
| `c3_per_request` | no append request is short of C3. A lost reply is small next to the history, so the aggregate ratio alone would pass a broken ECHO |
| `unknown_miss_class` | 0 ledger rows with `unknown`. Every request's ledger row must be found, so `--ledger-dir` is required |
| `tool_json_valid` | 100% of tool calls are a JSON object for a known tool, with at least one call |
| `compaction` | at least one compaction; the first post-compaction request reused the static prefix (cached > 0); neither request is `unknown` |
| `context_reached` | the peak prompt reached at least 90% of the target |
| `run_complete` | no HTTP or stream error |

The compaction request and the first post-compaction request are not C3-eligible. The ledger classifies them
as `compaction`. Every later request is.

`--dry-run` runs the whole loop against a built-in fake OpenAI server on 127.0.0.1. The fake streams
reasoning and tool calls, reports ECHO-like cache numbers and writes a fake ledger in a temporary directory.
No real endpoint is contacted.

### Commands

Run these under the inference lease, against the installed controller, one model at a time:

```sh
L=~/.local/state/local-ai/ledger; S=<result dir>
for M in ornith-35b-a3b-local qwen3.6-35b-a3b-local; do
  for T in 32K 64K 128K; do
    python3 tools/closed_loop_session.py --model $M --target-context $T --no-thinking \
      --max-tokens 256 --ledger-dir $L --receipt $S/closed-loop-$M-$T-off.json
    python3 tools/closed_loop_session.py --model $M --target-context $T --thinking \
      --max-tokens 1024 --ledger-dir $L --receipt $S/closed-loop-$M-$T-on.json
  done
done
```

Variants:

- `--reasoning-echo drop` is the OpenCode-faithful thinking-on arm, in which ECHO restores the reasoning.
  Run it once per model at 64K.
- `--no-compaction` runs a pure growth arm with no compaction step.

Every arm stops by itself: after the compaction plus 3 user turns, or at the target with `--no-compaction`.
The served context must exceed the target by one more turn, because one user message plus its tool output
can add up to ~18K tokens. Use a served context of at least the target + 24K. On a profile capped at
exactly 128K, run `--target-context 104K` instead of 128K.

Rough request counts: 32K ~15-25, 64K ~30-45, 128K ~60-90 requests per arm, plus 4-6 after compaction.

## 2. Fused-attention numeric reference (engine `oxe fused-ref`)

This is the reference recipe for the long-context fused-attention gate. It lives in the engine repository:
`FUSED-REF.md` there has the commands, the cost and the gate.

Its declared scope, to be quoted in the receipt:

> f32-attention reference with production weights (fused vs tiled attention over the same stored f16 KV
> and the same saved prefix state; weights, matmuls and recurrent layers are the production CPU path);
> NOT a full-model f32 reference.
