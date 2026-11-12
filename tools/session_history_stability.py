#!/usr/bin/env python3
"""Offline counterfactual prompt-prefix attribution; never infer measured cache_n.

Compare tokenizer rows from the SAME recording under two template policies.
Only append-only, otherwise unchanged message histories within an agent are
eligible. Count previously supplied INPUT tokens invalidated by re-rendering,
excluding the old generation prompt and reconstructed completion. This is a
lower-bound accounting of input reuse, not a cache/checkpoint simulator or a
prediction of server timings. Output contains numbers and reasons only.
"""
import argparse
import json
import os
from pathlib import Path

from session_replay import CHAT_PATHS, UsageError, agent_info, load_recording


def lcp(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


def ordered(value):
    # Schema property order can change rendered tokens. Do not canonicalize it
    # away, as the native cache also compares exact token prefixes.
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


def attribute_history(turns, token_rows):
    by_seq = {r['seq']: r for r in token_rows}
    if len(by_seq) != len(token_rows):
        raise UsageError('duplicate token-row sequence')
    previous = {}
    rows = []
    for turn in sorted(turns, key=lambda t: t['seq']):
        seq = turn['seq']
        if (turn.get('status') != 200 or turn.get('aborted') or seq not in by_seq
                or turn.get('path', '').split('?')[0] not in CHAT_PATHS):
            continue
        body = turn.get('body') or {}
        messages = body.get('messages')
        if not isinstance(messages, list):
            raise UsageError('chat recording has no messages')
        token_row = by_seq[seq]
        ids, boundary = token_row['prompt_ids'], token_row.get('generation_prompt_start')
        if type(boundary) is not int or not 0 <= boundary < len(ids):
            raise UsageError('token rows need an exact generation-prompt boundary')
        agent = agent_info(body)[0]
        before = previous.get(agent)
        row = {'seq': seq, 'eligible': False, 'reason': 'first-agent-input',
               'prompt_tokens': len(ids), 'rerendered_input_tokens': 0}
        if before is not None:
            old_body, old_ids, old_boundary = before
            old_messages = old_body['messages']
            if (ordered(body.get('tools')) != ordered(old_body.get('tools'))
                    or ordered(body.get('tool_choice')) != ordered(old_body.get('tool_choice'))):
                row['reason'] = 'tool-schema-or-choice-changed'
            elif (len(messages) < len(old_messages)
                  or ordered(messages[:len(old_messages)]) != ordered(old_messages)):
                row['reason'] = 'history-content-or-order-changed'
            else:
                common = lcp(old_ids, ids)
                invalid = max(0, old_boundary - common)
                row.update(eligible=True, reason='append-only-history',
                           previous_input_tokens=old_boundary, shared_prefix_tokens=common,
                           rerendered_input_tokens=invalid)
        previous[agent] = (body, ids, boundary)
        rows.append(row)
    return {'turns': rows, 'eligible_turns': sum(r['eligible'] for r in rows),
            'rerendered_input_tokens': sum(r['rerendered_input_tokens'] for r in rows)}


def compare(turns, baseline, preserved, minimum=.8):
    a, b = attribute_history(turns, baseline), attribute_history(turns, preserved)
    if [(r['seq'], r['eligible'], r['reason']) for r in a['turns']] != [
            (r['seq'], r['eligible'], r['reason']) for r in b['turns']]:
        raise UsageError('counterfactual policies did not cover identical histories')
    before, after = a['rerendered_input_tokens'], b['rerendered_input_tokens']
    if not before or not a['eligible_turns']:
        raise UsageError('no baseline re-rendering to price')
    reduction = 1 - after / before
    return {'passed': reduction >= minimum, 'minimum_reduction': minimum, 'reduction': reduction,
            'kind': 'offline previously supplied input invalidated by template rendering; not measured cache reuse',
            'baseline': a, 'preserved': b}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('recording')
    ap.add_argument('--baseline', required=True)
    ap.add_argument('--preserved', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--minimum-reduction', type=float, default=.8)
    args = ap.parse_args()
    try:
        if not 0 <= args.minimum_reduction <= 1:
            raise UsageError('minimum reduction must be in [0,1]')
        _, turns, recording_sha = load_recording(args.recording)
        read = lambda p: [json.loads(line) for line in Path(p).read_text().splitlines() if line.strip()]
        result = compare(turns, read(args.baseline), read(args.preserved), args.minimum_reduction)
        result['recording_sha256'] = recording_sha
        fd = os.open(args.out, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, 'w') as f:
            json.dump(result, f, indent=2)
            f.write('\n')
        print(json.dumps({k: result[k] for k in ('passed', 'reduction', 'minimum_reduction')}))
        return 0 if result['passed'] else 1
    except (OSError, ValueError, KeyError, UsageError) as exc:
        ap.exit(2, f'session_history_stability: {exc}\n')


if __name__ == '__main__':
    raise SystemExit(main())
