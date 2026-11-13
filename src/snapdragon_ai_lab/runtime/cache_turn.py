"""CACHE-PERFECT P1 turn labels for chat requests to the Rust engine (COMPACT-HIT, design sec. 1).

Pure functions over a chat payload. Output is enum labels and short hashes only: no prompt text
leaves this module.

- `turn_kind`: 'compaction' for the client's summary request (OpenCode sends the history to the
  cut plus a fixed instruction as the last user message), 'post-compaction' for the first turns
  after it (a user message carrying the summary checkpoint), else 'chat'. Recognised from the
  instruction's markers; the instruction's hash is kept so an unknown wording can be added later.
- `compact_layout` (post-compaction only): 'ok' when the history is [static prefix, summary,
  tail], i.e. the summary is the first message after the system message, so the anchor
  checkpoint (end of system + tools) still hits; 'summary-moved' otherwise.
- `tools_hash`: 12 hex of the tool list (unkeyed here; the ledger keys it with its install salt), so a compaction request that drops or changes the tools
  (the whole prefix then misses) is told apart from a real miss.
- `session_hint`: the value forwarded to the engine as `X-Oxe-Session`. A hint only: the engine
  finds its echo records by token chain, never by this id.
"""
import hashlib
import json

TURN_KINDS = ('chat', 'compaction', 'post-compaction')
LAYOUTS = ('ok', 'summary-moved')
SESSION_HEADER = 'X-Oxe-Session'
TURN_KIND_HEADER = 'X-Oxe-Turn-Kind'

# OpenCode 2.2.x (strings of the installed client): the summary instruction and the checkpoint
# message that replaces the compacted history afterwards. Matched as substrings.
COMPACTION_MARKERS = (
    'summarize the conversation above into a structured summary',
    'Summarize only what the user and the assistant said and did.',
)
CHECKPOINT_MARKERS = (
    '<conversation-checkpoint>',
    'The following is a summary and serialized record of earlier conversation.',
)


def _text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return ''.join(p.get('text', '') for p in content if isinstance(p, dict) and isinstance(p.get('text'), str))
    return ''


def _messages(payload):
    msgs = payload.get('messages') if isinstance(payload, dict) else None
    return [m for m in msgs if isinstance(m, dict)] if isinstance(msgs, list) else []


def _short(text):
    return hashlib.sha256(text.encode('utf-8', 'replace')).hexdigest()[:12]


def tools_hash(payload):
    tools = payload.get('tools') if isinstance(payload, dict) else None
    if not isinstance(tools, list) or not tools:
        return None
    return _short(json.dumps(tools, sort_keys=True, separators=(',', ':'), ensure_ascii=False))


def label(payload):
    """{'turn_kind', 'compact_layout', 'instruction_hash', 'tools_hash'} for a chat payload."""
    msgs = _messages(payload)
    out = {'turn_kind': 'chat', 'compact_layout': None, 'instruction_hash': None, 'tools_hash': tools_hash(payload)}
    users = [(i, m) for i, m in enumerate(msgs) if m.get('role') == 'user']
    if not users:
        return out
    last_i, last = users[-1]
    last_text = _text(last.get('content'))
    if last_i == len(msgs) - 1 and any(k in last_text for k in COMPACTION_MARKERS):
        out['turn_kind'] = 'compaction'
        out['instruction_hash'] = _short(last_text)
        return out
    checkpoint = next((i for i, m in users if any(k in _text(m.get('content')) for k in CHECKPOINT_MARKERS)), None)
    if checkpoint is not None:
        out['turn_kind'] = 'post-compaction'
        first_non_system = next((i for i, m in enumerate(msgs) if m.get('role') not in ('system', 'developer')), None)
        out['compact_layout'] = 'ok' if checkpoint == first_non_system else 'summary-moved'
    return out


def backend_headers(session, turn):
    """Headers the controller adds to a chat request for the Rust engine."""
    out = {}
    if isinstance(session, str) and 1 <= len(session) <= 128:
        out[SESSION_HEADER] = session
    if isinstance(turn, dict) and turn.get('turn_kind') in TURN_KINDS:
        out[TURN_KIND_HEADER] = turn['turn_kind']
    return out
