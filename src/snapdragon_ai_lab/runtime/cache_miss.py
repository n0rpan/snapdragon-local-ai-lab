"""Per-turn cache-miss classifier for the request ledger (CACHE-PERFECT P0).

Pure functions over numeric ledger fields. A turn is judged against its predecessor in the same
session (same agent and model): the backend must reuse at least the previous prompt plus the
previous completion, minus one token (the C3 invariant). Anything shorter is a miss, and its
class says which part of the history changed. Numbers and enum labels only: no prompt text, no
token ids, no message content ever reaches this module.

Classes: append-ok, retry, history-edit, reasoning-drop, tool-rerender, compaction, template-drift,
unknown. `unknown` means an unexplained shortfall; the caller logs a WARN for it.

Predecessor (M2): the last COMPLETE chat turn of the same (session, model, agent, origin); error,
incomplete and non-chat rows are neither judged nor used as a predecessor. C3 eligibility: the
previous completion counts only when the returned assistant reply is present right after the
previous input prefix (`previous_reply: present`, from request_ledger's reply digest). An exact
repeat of the prompt, or a prompt whose reply was edited or left out, is a `retry`: its
expectation is the previous prompt minus one token, not prompt plus completion.

Basis (P1, strongest first): `history` when the root history fields decide (keyed message hashes:
history_change, previous_reply); `label` when the controller marked the turn as a compaction or the
first turn after one (cache_turn.py); `echo` when the engine's ECHO report names why an earlier
assistant turn was not spliced (content or tool-call edit, different reasoning, unknown layout);
`divergence` when the row names the first divergent message (first_divergent_message and its
role) and that role is not assistant; `inferred` when it is an assistant message (a hash change
is not proof of a reasoning drop), else `counts`: a heuristic over the counts the
ledger holds today (message count, last role, completion size, finish reason). A count-only label
is a suspicion, not a proof; the offline Breaker A tool gives the exact cause.
"""

CLASSES = ('append-ok', 'retry', 'history-edit', 'reasoning-drop', 'tool-rerender', 'compaction',
           'template-drift', 'unknown')
SLACK = 1          # C3 allows one token of slack
WHOLE_TOL = 2      # a re-render of the whole previous completion matches within this many tokens
WHOLE_FRAC = 0.05  # or within 5% of the completion size
ROLES = frozenset({'system', 'user', 'assistant', 'tool'})
TURN_KINDS = frozenset({'chat', 'compaction', 'post-compaction'})
CHAT_PATHS = frozenset({'/v1/chat/completions', '/chat/completions'})
HISTORY = frozenset({'initial', 'same', 'append', 'edited', 'truncated', 'over_limit'})
REPLY = frozenset({'present', 'edited', 'absent', 'unknown'})
COMPLETE = frozenset({'complete', 'closed-after-done'})  # a reply fully generated: its cache is valid
COMPLETENESS = COMPLETE | {'incomplete', 'errored'}
# engine ECHO miss class -> ledger miss_class (no-record says nothing about the cause: fall through)
ECHO_CLASS = {'content-mismatch': 'history-edit', 'tool-mismatch': 'history-edit',
              'reasoning-mismatch': 'reasoning-drop', 'layout': 'template-drift', 'no-record': None}


def _num(v):
    return v if type(v) in (int, float) and v == v and v not in (float('inf'), float('-inf')) else None


def _enum(row, key, allowed):
    return row.get(key) if isinstance(row.get(key), str) and row.get(key) in allowed else None


def _int(row, key):
    return row.get(key) if type(row.get(key)) is int else None


def complete(row):
    """True for a finished chat row whose reply was fully generated (status 200, no transport error
    except a client that closed after the final event). Old rows without `completeness` fall back
    to status and error."""
    path = row.get('path')
    if isinstance(path, str) and path.split('?')[0] not in CHAT_PATHS:
        return False
    if 'completeness' in row:
        return row.get('completeness') in COMPLETE
    return row.get('status', 200) == 200 and not row.get('error')


def summarize(row):
    """The fields the classifier needs from one finished ledger row, or None when the row cannot be
    judged nor serve as a predecessor (non-chat path, HTTP error, incomplete reply, no counts)."""
    cache, new = _num(row.get('cache_n')), _num(row.get('prompt_n'))
    if cache is None or new is None or not complete(row):
        return None
    predicted = _num(row.get('predicted_n')) or 0
    index = _int(row, 'first_divergent_message')
    return {
        'id': row.get('id') if isinstance(row.get('id'), str) else None,
        'ctx': int(cache + new), 'cache': int(cache), 'predicted': int(predicted),
        'n_messages': _int(row, 'n_messages'),
        'last_role': row.get('last_role'), 'finish_reason': row.get('finish_reason'),
        'session': row.get('session'), 'origin': row.get('origin'),
        'agent': row.get('agent'), 'model': row.get('model'), 'template': row.get('template_sha'),
        'div_role': _enum(row, 'first_divergent_role', ROLES),
        'div_index': index if index is not None else _int(row, 'first_divergent_index'),
        'history': _enum(row, 'history_change', HISTORY), 'reply': _enum(row, 'previous_reply', REPLY),
        'prev_id': row.get('previous_request_id') if isinstance(row.get('previous_request_id'), str) else None,
        'turn_kind': _enum(row, 'turn_kind', TURN_KINDS),
        'tools_hash': row.get('tools_hash') if isinstance(row.get('tools_hash'), str) else None,
        'echo_miss': _enum(row, 'echo_miss_class', ECHO_CLASS),
    }


def same_session(prev, cur):
    """One chain is one (session, model, agent, origin); session and origin are compared when both
    rows carry them (old rows have neither)."""
    if prev is None or cur is None or prev['agent'] != cur['agent'] or prev['model'] != cur['model']:
        return False
    return all(prev.get(k) is None or cur.get(k) is None or prev[k] == cur[k] for k in ('session', 'origin'))


def continuation(prev, cur):
    """(eligible, basis): does this prompt carry the previous reply right after the previous input
    prefix? Only then can the previous completion's tokens be reused (C3)."""
    if cur.get('reply') in ('present', 'unknown') or cur.get('history') in ('edited', 'truncated'):
        return True, 'history'
    if cur.get('reply') in ('edited', 'absent') or cur.get('history') == 'same':
        return False, 'history'
    if cur.get('echo_miss') in ('content-mismatch', 'tool-mismatch'):
        return False, 'echo'  # the client's assistant turn is not what the model wrote
    return True, 'counts'


def classify(prev, cur):
    """(miss_class, miss_tokens, basis) for `cur` after `prev`; (None, None, None) when there is no
    predecessor to judge against (first turn of a session, or another session, agent or model)."""
    if not same_session(prev, cur) or prev['ctx'] is None or cur['cache'] is None:
        return None, None, None
    expected = prev['ctx'] + prev['predicted'] - SLACK
    eligible, basis = continuation(prev, cur)
    if cur['cache'] >= expected:
        return 'append-ok', 0, basis
    if not eligible:
        # retry / regenerate / edited reply: only the previous prompt (its real LCP) can be reused
        expected = prev['ctx'] - SLACK
        if cur['cache'] >= expected:
            return 'retry', 0, basis
    miss = expected - cur['cache']
    # The checkpoint message stays in the history for the rest of the session, so every later turn
    # is 'post-compaction' too: only the first one after the cut is explained by the compaction.
    # A later miss goes through the other rules (and can be `unknown`, which is the alarm).
    kind = cur.get('turn_kind')
    if kind == 'compaction' or (kind == 'post-compaction' and prev.get('turn_kind') != 'post-compaction'):
        return 'compaction', miss, 'label'
    if prev.get('template') and cur.get('template') and prev['template'] != cur['template']:
        return 'template-drift', miss, 'counts'
    if ECHO_CLASS.get(cur.get('echo_miss')):
        return ECHO_CLASS[cur['echo_miss']], miss, 'echo'
    if cur.get('history') == 'truncated':
        return 'compaction', miss, 'history'
    if cur['n_messages'] is not None and prev['n_messages'] is not None and cur['n_messages'] < prev['n_messages']:
        return 'compaction', miss, 'counts'
    if cur['div_role'] is not None:
        # the first divergent message of this turn, compared with the turn before it
        role, index = cur['div_role'], cur['div_index']
        if role == 'tool':
            return 'tool-rerender', miss, 'divergence'
        if role == 'assistant':
            # a changed assistant hash is no proof of a dropped reasoning block: inferred only
            last = prev['n_messages'] is not None and index is not None and index >= prev['n_messages'] - 1
            if last and prev['finish_reason'] == 'tool_calls':
                return 'tool-rerender', miss, 'inferred'
            return 'reasoning-drop', miss, 'inferred'
        return 'history-edit', miss, 'divergence'
    if not eligible:
        return 'unknown', miss, basis  # the previous prompt itself was lost: no client-side cause
    whole = prev['predicted']
    if whole and abs(miss - whole) <= max(WHOLE_TOL, int(whole * WHOLE_FRAC)):
        return ('tool-rerender' if prev['finish_reason'] == 'tool_calls' else 'reasoning-drop'), miss, 'counts'
    if cur['last_role'] == 'tool':
        return 'tool-rerender', miss, 'counts'
    if cur['last_role'] == 'user' and cur['n_messages'] is not None and prev['n_messages'] is not None:
        if cur['n_messages'] == prev['n_messages'] + 1:
            return 'reasoning-drop', miss, 'counts'
        if cur['n_messages'] == prev['n_messages'] and cur.get('history') is None:
            return 'history-edit', miss, 'counts'
    return 'unknown', miss, 'counts'
