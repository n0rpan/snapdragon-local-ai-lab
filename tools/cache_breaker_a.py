"""Breaker A root cause, offline (CACHE-PERFECT P0).

For each consecutive pair of chat turns of one conversation, the expected prefix is
prompt(N-1) + completion(N-1), with token ids from a `session_replay tokens` file, and the actual
prefix is prompt(N). The first divergent token is located and attributed to one of:
- `prev-generation-prompt`: the divergence falls in the previous generation prompt (the template's
  assistant prefix, e.g. the think-block opener), so nothing after the last message was reusable;
- `history-render-changed-<role>`: the client sent the previous messages unchanged, but the template
  rendered an earlier message differently (role from the message spans of the previous prompt);
- `client-history-changed`: the client sent different earlier messages;
- `completion`: the previous completion itself was re-rendered; its field-level cause is then
  attributed (reasoning, content, tool-call name, arguments key order / whitespace / value).
Pairs that are not one conversation (different first user message) are counted apart and skipped.

Message spans: each non-tool message opens one `<|im_start|>` span, consecutive tool results share a
span, and the generation prompt is the last span. The delimiter id is checked against that model:
`--im-start-id` (default 248045, the Qwen3.5/3.6-family <|im_start|>) is accepted only if every
previous prompt ends with the same generation-prompt tail; the result reports that check.

Input is private (recording and token ids). Output is counts and categories only: no prompt text,
no message content, no token ids, no file names. --out writes the same counts as JSON (0600).
"""
import argparse
import bisect
import collections
import json
import os
import re
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
from session_replay import CHAT_PATHS, UsageError, agent_info, load_recording  # noqa: E402
from cache_contract import first_divergence  # noqa: E402

DEFAULT_IM_START = 248045


def _ordered_equal(a, b):
    """Equal values and the same key order at every object level."""
    if isinstance(a, dict) and isinstance(b, dict):
        return list(a) == list(b) and all(_ordered_equal(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_ordered_equal(x, y) for x, y in zip(a, b))
    return a == b and type(a) is type(b)


def _args_text(value):
    return value if isinstance(value, str) else json.dumps(value)


def argument_category(generated, client):
    if generated == client:
        return None
    try:
        a, b = json.loads(generated or '{}'), json.loads(client or '{}')
    except ValueError:
        return 'arguments-not-json'
    if a == b:
        return 'arguments-whitespace-or-escape' if _ordered_equal(a, b) else 'arguments-key-order'
    return 'arguments-value'


def completion_field_categories(response, client_message):
    """Where the client's copy of the assistant message differs from what the engine generated."""
    response = response or {}
    cats = []
    if response.get('reasoning') and not (client_message.get('reasoning_content') or client_message.get('reasoning')):
        cats.append('reasoning-dropped-by-client')
    gen_text, cli_text = response.get('content') or '', client_message.get('content') or ''
    if gen_text != cli_text:
        same = re.sub(r'\s+', '', gen_text) == re.sub(r'\s+', '', cli_text)
        cats.append('content-whitespace' if same else 'content-text')
    gen_calls, cli_calls = response.get('tool_calls') or [], client_message.get('tool_calls') or []
    if len(gen_calls) != len(cli_calls):
        cats.append('tool-call-count')
    for g, c in zip(gen_calls, cli_calls):
        if (g.get('name') or '') != (c.get('function') or {}).get('name', ''):
            cats.append('tool-name')
        cat = argument_category(_args_text(g.get('arguments')), _args_text((c.get('function') or {}).get('arguments')))
        if cat:
            cats.append(cat)
    return cats


def span_sources(messages):
    """(role, source message index) of each <|im_start|> span the template emits for these messages;
    adjacent tool results share one span, which points at the first of them."""
    spans, in_tools = [], False
    for i, m in enumerate(messages):
        role = m.get('role') if isinstance(m, dict) else None
        if role == 'tool':
            if not in_tools:
                spans.append(('tool', i))
            in_tools = True
            continue
        in_tools = False
        spans.append((role or 'unknown', i))
    return spans


def span_roles(messages):
    """The role of each <|im_start|> span the template emits for these messages (tool results share one)."""
    return [role for role, _ in span_sources(messages)]


def generation_tail_uniform(rows, im_start, tail_len=4):
    tails = collections.Counter()
    for r in rows:
        ids = r['prompt_ids']
        starts = [i for i, t in enumerate(ids) if t == im_start]
        if starts:
            tails[tuple(ids[starts[-1]:starts[-1] + tail_len])] += 1
    return len(tails) == 1 and sum(tails.values()) == len(rows) and len(rows) > 0


def history_category(prev_turn, cur_turn, prev_prompt, cur_prompt, d, im_start, gen_start):
    old, new = prev_turn['body']['messages'], cur_turn['body']['messages']
    first = next((i for i, m in enumerate(old) if isinstance(m, dict) and m.get('role') != 'system'), None)
    if first is not None and (len(new) <= first or new[first] != old[first]):
        return 'other-conversation'
    if new[:len(old)] != old:
        return 'client-history-changed'
    if im_start is None or gen_start is None:
        return 'history-render-changed-unlocated'
    starts = [i for i, t in enumerate(prev_prompt) if t == im_start]
    spans = span_sources(old)
    if len(starts) != len(spans) + 1:
        return 'history-render-changed-unmapped'
    k = bisect.bisect_right(starts, d) - 1
    if k < 0:
        return 'history-render-changed-system'
    role, src = spans[k] if k < len(spans) else ('generation', None)
    # the span index differs from the message index once tool results are coalesced: use the source
    if role == 'assistant' and isinstance(old[src], dict) and old[src].get('tool_calls'):
        role = 'assistant-tool-call'
    return 'history-render-changed-' + role


def opener_category(prev, cur, prev_turn, cur_turn, im_start, gen_start):
    """The previous generation prompt ends with an opener token (e.g. the think-block opener). Does the
    template render the previous assistant message with the same opener? If not, the completion is
    re-prefilled: the cache stops at the generation prompt."""
    tail = prev['prompt_ids'][gen_start:]
    if len(tail) < 4:
        return 'prev-generation-prompt:opener-unknown'
    opener = tail[3]
    k = len(prev_turn['body']['messages'])
    starts = [i for i, t in enumerate(cur['prompt_ids']) if t == im_start]
    span = len(span_roles(cur_turn['body']['messages'][:k + 1])) - 1
    if span < 0 or span >= len(starts) or starts[span] + 3 >= len(cur['prompt_ids']):
        return 'prev-generation-prompt:opener-unlocated'
    rendered = cur['prompt_ids'][starts[span] + 3]
    return 'prev-generation-prompt:opener-' + ('same' if rendered == opener else 'differs')


def analyse(turns, token_rows, im_start):
    turns_by_seq = {t['seq']: t for t in turns}
    rows = sorted((r for r in token_rows if r['seq'] in turns_by_seq), key=lambda r: r['seq'])
    served = [r for r in rows if turns_by_seq[r['seq']].get('status') == 200
              and not turns_by_seq[r['seq']].get('aborted')
              and (turns_by_seq[r['seq']].get('path') or '').split('?')[0] in CHAT_PATHS]
    uniform = generation_tail_uniform(served, im_start) if im_start is not None else None
    by_agent = collections.defaultdict(list)
    for r in served:
        by_agent[agent_info(turns_by_seq[r['seq']]['body'])[0]].append(r)
    result = collections.Counter()
    cause = collections.Counter()
    region = collections.Counter()
    quartile = collections.Counter()
    lost = 0
    for agent_rows in by_agent.values():
        for prev, cur in zip(agent_rows, agent_rows[1:]):
            prev_turn, cur_turn = turns_by_seq[prev['seq']], turns_by_seq[cur['seq']]
            expected = prev['prompt_ids'] + prev['completion_ids']
            actual = cur['prompt_ids']
            if actual[:len(expected)] == expected:
                result['clean'] += 1
                continue
            d = min(first_divergence(expected, actual), len(actual))
            prompt_len = len(prev['prompt_ids'])
            gen_start = None
            if im_start is not None:
                starts = [i for i, t in enumerate(prev['prompt_ids']) if t == im_start]
                gen_start = starts[-1] if starts else None
            cat = history_category(prev_turn, cur_turn, prev['prompt_ids'], actual, d, im_start, gen_start)
            if cat == 'other-conversation':
                result['other-conversation'] += 1
                continue
            result['divergent'] += 1
            lost += len(expected) - d
            if d >= prompt_len:
                region['completion'] += 1
                width = max(1, len(prev['completion_ids']))
                quartile['q%d' % min(3, (4 * (d - prompt_len)) // width)] += 1
                k = len(prev_turn['body']['messages'])
                client = cur_turn['body']['messages'][k] if k < len(cur_turn['body']['messages']) else {}
                if cur_turn['body']['messages'][:k] != prev_turn['body']['messages'] or client.get('role') != 'assistant':
                    cause['client-history-changed'] += 1
                    continue
                cats = completion_field_categories(prev_turn.get('response'), client)
                if prev.get('completion_len_matches') is False:
                    cats = ['reconstruction-suspect'] + cats
                cause[cats[0] if cats else 'completion-template-render-only'] += 1
            elif gen_start is not None and d >= gen_start:
                region['generation-prompt'] += 1
                cause[opener_category(prev, cur, prev_turn, cur_turn, im_start, gen_start)] += 1
            else:
                region['history'] += 1
                cause[cat] += 1
    out = {'turns_served': len(served), 'agents': len(by_agent), 'pairs': sum(max(0, len(v) - 1) for v in by_agent.values()),
           'by_result': dict(result), 'divergent_region': dict(region), 'divergent_cause': dict(cause),
           'completion_offset_quartile': dict(quartile), 'lost_tokens_total': lost,
           'reconstruction_length_matches': sum(bool(r.get('completion_len_matches')) for r in served),
           'delimiter': {'id': im_start, 'generation_tail_uniform': uniform}}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('recording')
    ap.add_argument('tokens', help='session_replay tokens JSONL for the same recording')
    ap.add_argument('--im-start-id', type=int, default=DEFAULT_IM_START, help='message delimiter token id')
    ap.add_argument('--out', help='also write the counts as JSON (owner-only)')
    args = ap.parse_args(argv)
    try:
        _header, turns, sha = load_recording(args.recording)
        token_rows = [json.loads(l) for l in Path(args.tokens).read_text().splitlines() if l.strip()]
        result = analyse(turns, token_rows, args.im_start_id)
        result['recording_sha256'] = sha
        text = json.dumps(result, indent=1, sort_keys=True)
        if args.out:
            fd = os.open(args.out, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
            with os.fdopen(fd, 'w') as f:
                f.write(text + '\n')
        print(text)
        return 0
    except (OSError, ValueError, KeyError, UsageError) as exc:
        ap.exit(2, f'cache_breaker_a: {exc}\n')


if __name__ == '__main__':
    raise SystemExit(main())
