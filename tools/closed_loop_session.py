#!/usr/bin/env python3
"""Closed-loop agentic session runner: the CHAT-GATE proof of ECHO, thinking and compaction.

Unlike `long_session.py replay` (a fixed, scripted history: a cost curve, not a correctness proof),
every request here carries the model's ACTUAL previous reply, as OpenCode sends it:

- the assistant message with the returned `content`, the returned `tool_calls` (arguments exactly
  as streamed) and, with `--reasoning-echo send` (default), the returned `reasoning_content`
  (`--reasoning-echo drop` sends no reasoning, as OpenCode 2.2.x measured: the engine's ECHO must
  restore it);
- one synthetic tool result per tool call (seeded, sized like long_session's tool outputs);
- then the next synthetic user message once the model answers without a tool call (or after
  `--max-hops` tool rounds).

Thinking is `chat_template_kwargs.enable_thinking` (`--thinking/--no-thinking`; preserve_thinking
is always true, the C1 rule). Compaction mimics OpenCode 2.2.x: when the last measured context
reaches `--compact-at`, one summary request [system, history to the cut, instruction] is sent
(the instruction carries the markers `runtime/cache_turn.py` recognises), then the history becomes
[system, checkpoint(summary), tail] (the tail is the last user turns within `--keep-tokens`).

Per-request receipts: prompt / cached tokens, C3-eligible tokens and hit, resume_src, cache_src,
the ECHO fields (echo_n, echo_tokens, roundtrip_miss, echo_miss_class, ...), miss_class from the
controller's request ledger (`--ledger-dir`, matched by origin=lab and message count), TTFT,
prefill ms, decode tok/s, finish reason, tool-call JSON validity. No prompt or reply text is
written, only sizes and enum labels.

Pass criteria (all must hold): hit >= 99% of C3-eligible tokens over the append turns, and no
append request short of C3 (cached >= previous prompt + previous completion - 1); 0 ledger
rows with miss_class `unknown` (the ledger must have been read); 100% valid tool-call JSON with
at least one tool call; every compaction request and first post-compaction turn reached the
server, and the post-compaction turn reused the static prefix (cached > 0).

O86 (numbers only, `--thinking-metrics`, default on): per turn with reasoning, the engine's own
`/apply-template` and `/tokenize` are called in memory on the request's preceding context and the
reasoning text. Stored: token counts and, for n=3..8, the share of the reasoning's n-gram windows
that also occur in the preceding context (never the current reply). Not a gate.

`--dry-run` runs the whole loop against a built-in fake OpenAI server on 127.0.0.1 (reasoning,
tool calls, ECHO-like cache numbers, a fake ledger in a temporary directory); no real endpoint is
contacted. Python 3.10+ standard library only; loopback endpoints only; sends
`X-Local-AI-Origin: lab`. It does not take the inference lease: run it under the backend's lease.
"""
import argparse
import datetime as dt
import functools
import hashlib
import http.client
import json
import math
import os
import random
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import long_session as ls  # noqa: E402  (sibling tool: synthetic text, tools, endpoint check)

VERSION = 1
TOOL = 'closed_loop_session'
RECEIPT_SCHEMA = 'closed-loop-session/v1'
HIT_MIN = 0.99
SLACK = 1  # C3: one token of slack
DEFAULT_MAX_HOPS = 4
DEFAULT_KEEP = 8192
DEFAULT_POST_TURNS = 3
O86_NS = (3, 4, 5, 6, 7, 8)
HELPER_TIMEOUT = 60.0
HELPER_PATHS = ('/tokenize', '/apply-template')  # root paths on the controller/engine (no /v1)
REQUEST_SEED = ls.REQUEST_SEED
TOOL_NAMES = frozenset(t['function']['name'] for t in ls.TOOLS)
# OpenCode 2.2.x wording (cache_turn.py COMPACTION_MARKERS / CHECKPOINT_MARKERS are substrings)
COMPACTION_INSTRUCTION = (
    'You MUST summarize the conversation above into a structured summary with the sections '
    'Objective, Requirements, Decisions, Work State, Next Move, Relevant Files and Important '
    'Context. Summarize only what the user and the assistant said and did. Do not continue the '
    'task or call tools.')
CHECKPOINT_HEAD = ('<conversation-checkpoint>\nThe following is a summary and serialized record of '
                   'earlier conversation.\n')
CHECKPOINT_TAIL = '\n</conversation-checkpoint>'
TIMING_KEYS = ('cache_n', 'prompt_n', 'prompt_ms', 'predicted_n', 'predicted_ms',
               'predicted_per_second', 'resume_ms', 'preload_ms', 'echo_blocks', 'echo_n',
               'echo_fixed', 'echo_tokens', 'roundtrip_miss')
TIMING_ENUMS = ('resume_src', 'cache_src', 'prefix_reason', 'session_reason', 'echo_miss_class')
LEDGER_KEYS = ('miss_class', 'miss_tokens', 'miss_basis', 'previous_reply', 'history_change',
               'turn_kind', 'compact_layout', 'echo_miss_class', 'resume_src', 'completeness')


class UsageError(Exception):
    pass


class AbortRun(Exception):
    pass


def _num(v):
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) else None


def _round(v):
    return None if v is None else round(float(v), 3)


def _utc():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')


# ---------------------------------------------------------------- synthetic client side

class Script:
    """Seeded synthetic user messages and tool results (no real data)."""

    def __init__(self, seed):
        self.rng = random.Random(seed)

    def user(self):
        tok = ls.lognormal(self.rng, ls.USER_TOKENS_P50, ls.USER_TOKENS_P90, ls.USER_TOKENS_MIN, 12000)
        return {'role': 'user', 'content': ls._user_text(self.rng, int(round(tok * ls.CHARS_PER_TOKEN)))}

    def tool_result(self, call_id, name):
        if name in ('edit', 'write'):
            text = 'Edit applied successfully.\n'
        else:
            kind = {'read': 'read', 'grep': 'grep', 'glob': 'glob'}.get(name, 'bash_log')
            size = int(round(ls.lognormal(self.rng, ls.TOOL_BYTES_P50, ls.TOOL_BYTES_P90,
                                          ls.TOOL_BYTES_MIN, 24576)))
            text = ls._tool_output(self.rng, kind, size)
        return {'role': 'tool', 'tool_call_id': call_id, 'content': text}

    def invalid_tool_result(self, call_id):
        return {'role': 'tool', 'tool_call_id': call_id,
                'content': 'error: the tool call arguments are not a valid JSON object'}


def assistant_message(reply, echo_reasoning):
    """The assistant turn as the client sends it back: actual content, tool calls, reasoning."""
    msg = {'role': 'assistant', 'content': reply['content']}
    if reply['tool_calls']:
        msg['tool_calls'] = [{'id': c['id'], 'type': 'function',
                              'function': {'name': c['name'], 'arguments': c['arguments']}}
                             for c in reply['tool_calls']]
    if echo_reasoning and reply['reasoning']:
        msg['reasoning_content'] = reply['reasoning']
    return msg


def tool_call_valid(call):
    if call['name'] not in TOOL_NAMES:
        return False
    try:
        args = json.loads(call['arguments'])
    except ValueError:
        return False
    return isinstance(args, dict)


def est_tokens(messages):
    return sum(ls.msg_tokens(m) + math.ceil(len(m.get('reasoning_content') or '') / ls.CHARS_PER_TOKEN)
               for m in messages)


def split_tail(history, keep_tokens):
    """Index where the kept tail starts: the earliest user message (after the system message)
    whose suffix fits in keep_tokens (estimated). len(history) when nothing fits."""
    cut = len(history)
    for i in range(len(history) - 1, 0, -1):
        if history[i].get('role') == 'user':
            if est_tokens(history[i:]) > keep_tokens:
                break
            cut = i
    return cut


# ---------------------------------------------------------------- HTTP

def build_body(model, messages, max_tokens, thinking):
    return {
        'model': model, 'messages': messages, 'tools': ls.TOOLS, 'tool_choice': 'auto',
        'temperature': 0, 'top_p': 1, 'max_tokens': max_tokens, 'seed': REQUEST_SEED,
        'stream': True, 'stream_options': {'include_usage': True},
        'chat_template_kwargs': {'enable_thinking': bool(thinking), 'preserve_thinking': True},
    }


def send(host, port, path, body, timeout, session_id, api_key=None):
    """POST one streamed request; return the parsed reply, usage, timings and timing numbers."""
    payload = json.dumps(body).encode('utf-8')
    headers = {'Content-Type': 'application/json', 'Content-Length': str(len(payload)),
               'X-Local-AI-Origin': 'lab', 'X-Session-Id': session_id}
    if api_key:
        headers['Authorization'] = 'Bearer ' + api_key
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    content, reasoning, calls = [], [], {}
    usage, timings, finish, ttft = {}, {}, None, None
    t0 = time.monotonic()
    try:
        conn.request('POST', path, body=payload, headers=headers)
        resp = conn.getresponse()
        if resp.status != 200:
            raise AbortRun('HTTP {} from endpoint'.format(resp.status))
        if (resp.getheader('Content-Type', '') or '').startswith('application/json'):
            doc = json.loads(resp.read())
            ttft = (time.monotonic() - t0) * 1000.0
            usage, timings = doc.get('usage') or {}, doc.get('timings') or {}
            choice = (doc.get('choices') or [{}])[0]
            msg = choice.get('message') or {}
            content.append(msg.get('content') or '')
            reasoning.append(msg.get('reasoning_content') or msg.get('reasoning') or '')
            for i, c in enumerate(msg.get('tool_calls') or []):
                fn = c.get('function') or {}
                calls[i] = {'id': c.get('id'), 'name': fn.get('name') or '', 'arguments': fn.get('arguments') or ''}
            finish = choice.get('finish_reason')
        else:
            while True:
                raw = resp.readline()
                if not raw:
                    break
                line = raw.decode('utf-8', 'replace').strip()
                if not line.startswith('data:'):
                    continue
                data = line[5:].strip()
                if data == '[DONE]':
                    break
                try:
                    chunk = json.loads(data)
                except ValueError:
                    raise AbortRun('malformed stream chunk')
                if not isinstance(chunk, dict) or 'error' in chunk:
                    raise AbortRun('error object in stream')
                if isinstance(chunk.get('usage'), dict):
                    usage = chunk['usage']
                if isinstance(chunk.get('timings'), dict):
                    timings.update(chunk['timings'])
                for choice in chunk.get('choices') or []:
                    delta = choice.get('delta') or {}
                    piece = delta.get('content')
                    think = delta.get('reasoning_content') or delta.get('reasoning')
                    if ttft is None and (piece or think or delta.get('tool_calls')):
                        ttft = (time.monotonic() - t0) * 1000.0
                    if piece:
                        content.append(piece)
                    if think:
                        reasoning.append(think)
                    for tc in delta.get('tool_calls') or []:
                        slot = calls.setdefault(tc.get('index', len(calls)), {'id': None, 'name': '', 'arguments': ''})
                        if tc.get('id'):
                            slot['id'] = tc['id']
                        fn = tc.get('function') or {}
                        if fn.get('name'):
                            slot['name'] += fn['name']
                        if fn.get('arguments'):
                            slot['arguments'] += fn['arguments']
                    if choice.get('finish_reason'):
                        finish = choice['finish_reason']
    except AbortRun:
        raise
    except (OSError, http.client.HTTPException, ValueError) as exc:
        raise AbortRun('connection or response error: {}'.format(type(exc).__name__))
    finally:
        conn.close()
    wall = (time.monotonic() - t0) * 1000.0
    tool_calls = [calls[k] for k in sorted(calls)]
    return {'content': ''.join(content), 'reasoning': ''.join(reasoning), 'tool_calls': tool_calls,
            'usage': usage, 'timings': timings, 'finish_reason': finish, 'ttft_ms': ttft, 'wall_ms': wall}


# ---------------------------------------------------------------- O86 thinking metrics

class HelperUnavailable(Exception):
    pass


def helper_post(host, port, path, body, timeout, credential=None):
    """POST a model-helper request (/tokenize, /apply-template); returns the decoded JSON object."""
    payload = json.dumps(body).encode('utf-8')
    headers = {'Content-Type': 'application/json', 'Content-Length': str(len(payload)),
               'X-Local-AI-Origin': 'lab'}
    if credential:
        headers['Authorization'] = 'Bearer ' + credential
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    try:
        conn.request('POST', path, body=payload, headers=headers)
        resp = conn.getresponse()
        status, data = resp.status, resp.read()
    except (OSError, http.client.HTTPException, ValueError):
        raise HelperUnavailable('connection_error')
    finally:
        conn.close()
    if status != 200:
        raise HelperUnavailable('http_{}'.format(status))
    try:
        doc = json.loads(data)
    except ValueError:
        raise HelperUnavailable('malformed')
    if not isinstance(doc, dict):
        raise HelperUnavailable('malformed')
    return doc


def _token_ids(doc):
    ids = doc.get('tokens')
    if not isinstance(ids, list) or not all(isinstance(t, int) and not isinstance(t, bool) for t in ids):
        raise HelperUnavailable('malformed')
    return ids


def ngram_copy_shares(reasoning_ids, context_ids, ns=O86_NS):
    """Per n: windows of the reasoning token ids, how many occur in the context's n-gram set, and
    the share (None when the reasoning is shorter than n). Token ids only; no text."""
    out = {}
    for n in ns:
        windows = len(reasoning_ids) - n + 1
        if windows <= 0:
            out['copy_windows_n{}'.format(n)] = 0
            out['copy_hits_n{}'.format(n)] = 0
            out['copy_share_n{}'.format(n)] = None
            continue
        grams = {tuple(context_ids[i:i + n]) for i in range(len(context_ids) - n + 1)}
        hits = sum(1 for i in range(windows) if tuple(reasoning_ids[i:i + n]) in grams)
        del grams
        out['copy_windows_n{}'.format(n)] = windows
        out['copy_hits_n{}'.format(n)] = hits
        out['copy_share_n{}'.format(n)] = round(hits / windows, 4)
    return out


def thinking_metrics(host, port, body, reasoning, timeout=HELPER_TIMEOUT, credential=None):
    """O86 numbers against the engine's ECHO-spliced preceding context, never the current reply.
    Token ids and reasoning stay in memory; only counts and shares are returned. A raw template
    render is not a substitute for the effective context when ECHO restores stripped reasoning."""
    if not reasoning:
        return {'status': 'no_reasoning', 'reasoning_tokens': 0}
    template_req = {k: body[k] for k in ('messages', 'tools', 'tool_choice', 'chat_template_kwargs') if k in body}
    template_req.update(return_tokens=True, echo=True)
    try:
        view = helper_post(host, port, '/apply-template', template_req, timeout, credential)
        context_ids = _token_ids({'tokens': view.get('echo_tokens')})
        del view
        reasoning_ids = _token_ids(helper_post(host, port, '/tokenize',
                                               {'content': reasoning, 'add_special': False, 'parse_special': True},
                                               timeout, credential))
    except HelperUnavailable as exc:
        return {'status': str(exc)}
    out = {'status': 'ok', 'reasoning_tokens': len(reasoning_ids), 'context_tokens': len(context_ids)}
    out.update(ngram_copy_shares(reasoning_ids, context_ids))
    return out


def _thinking(args, endpoint, credential, body, resp):
    if not args.thinking_metrics:
        return None
    return thinking_metrics(endpoint[0], endpoint[1], body, resp['reasoning'], HELPER_TIMEOUT, credential)


def thinking_summary(rows):
    """Run-level O86 numbers (no gate): totals and token-weighted copy shares over measured turns."""
    ok = [r['thinking'] for r in rows if (r.get('thinking') or {}).get('status') == 'ok']
    out = {'measured_turns': len(ok), 'reasoning_tokens_total': sum(t['reasoning_tokens'] for t in ok),
           'statuses': sorted({(r.get('thinking') or {}).get('status') for r in rows if r.get('thinking')} - {None})}
    for n in O86_NS:
        windows = sum(t['copy_windows_n{}'.format(n)] for t in ok)
        hits = sum(t['copy_hits_n{}'.format(n)] for t in ok)
        out['copy_share_n{}'.format(n)] = round(hits / windows, 4) if windows else None
    return out


# ---------------------------------------------------------------- ledger

class Ledger:
    """Read-only matcher for the controller's request ledger rows of this run (origin=lab, same
    message count, written after the request started; after the first match, the same session)."""

    def __init__(self, directory):
        self.dir = Path(directory) if directory else None
        self.seen = set()
        self.session = None

    def _rows(self):
        if self.dir is None:
            return []
        rows = []
        for path in sorted(self.dir.glob('requests-*.jsonl'))[-2:]:
            try:
                with open(path, 'rb') as f:
                    size = os.fstat(f.fileno()).st_size
                    f.seek(max(0, size - 1048576))
                    lines = f.read().splitlines()
            except OSError:
                continue
            for line in lines:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if isinstance(row, dict):
                    rows.append(row)
        return rows

    def match(self, started, n_messages, wait_s=3.0):
        if self.dir is None:
            return None
        deadline = time.monotonic() + wait_s
        while True:
            for row in self._rows():
                rid = row.get('id')
                if (rid in self.seen or row.get('origin') != 'lab' or row.get('n_messages') != n_messages
                        or not isinstance(row.get('t'), (int, float)) or row['t'] < started - 1.0):
                    continue
                if self.session is not None and row.get('session') != self.session:
                    continue
                self.seen.add(rid)
                if self.session is None:
                    self.session = row.get('session')
                return {k: row[k] for k in LEDGER_KEYS if k in row and isinstance(row[k], (str, int, float))}
            if time.monotonic() >= deadline:
                return {}
            time.sleep(0.1)


# ---------------------------------------------------------------- the loop

def _turn_row(idx, kind, user_turn, hop, messages, resp, prev, eligible_basis):
    usage, timings = resp['usage'], resp['timings']
    prompt = _num(usage.get('prompt_tokens'))
    if prompt is None and _num(timings.get('prompt_n')) is not None:
        prompt = _num(timings.get('prompt_n')) + (_num(timings.get('cache_n')) or 0)
    cached = _num((usage.get('prompt_tokens_details') or {}).get('cached_tokens'))
    if cached is None:
        cached = _num(timings.get('cache_n'))
    completion = _num(usage.get('completion_tokens'))
    if completion is None:
        completion = _num(timings.get('predicted_n'))
    eligible = None
    if eligible_basis == 'c3' and prev is not None and prev['prompt_tokens'] is not None and prompt is not None:
        # NOT clamped to the new prompt: a re-render that shrinks the history (reasoning or
        # completion tokens lost) gives a shorter prompt, and C3 must then FAIL, not move the target
        eligible = prev['prompt_tokens'] + (prev['completion_tokens'] or 0) - SLACK
    decode = _num(timings.get('predicted_per_second'))
    if decode is None and completion and resp['ttft_ms'] is not None and resp['wall_ms'] > resp['ttft_ms']:
        decode = completion / ((resp['wall_ms'] - resp['ttft_ms']) / 1000.0)
    row = {
        'request': idx + 1, 'kind': kind, 'user_turn': user_turn, 'hop': hop,
        'n_messages': len(messages), 'estimated_prompt_tokens': est_tokens(messages),
        'prompt_tokens': prompt, 'cached_tokens': cached, 'completion_tokens': completion,
        'eligible_basis': eligible_basis, 'eligible_tokens': eligible,
        'eligible_hit_tokens': (min(cached, eligible) if eligible is not None and cached is not None else None),
        'ttft_ms': _round(resp['ttft_ms']), 'prefill_ms': _round(_num(timings.get('prompt_ms'))),
        'decode_tok_s': _round(decode), 'wall_ms': _round(resp['wall_ms']),
        'finish_reason': resp['finish_reason'],
        'content_chars': len(resp['content']), 'reasoning_chars': len(resp['reasoning']),
        'tool_calls': len(resp['tool_calls']),
        'tool_calls_valid': sum(1 for c in resp['tool_calls'] if tool_call_valid(c)),
        'timings': {k: timings[k] for k in TIMING_KEYS if _num(timings.get(k)) is not None},
    }
    for k in TIMING_ENUMS:
        if isinstance(timings.get(k), str) and len(timings[k]) <= 32:
            row['timings'][k] = timings[k]
    if isinstance(timings.get('echo_misses'), dict):
        row['timings']['echo_misses'] = {str(k)[:32]: v for k, v in timings['echo_misses'].items() if _num(v) is not None}
    return row


def run_session(args, endpoint, ledger, err):
    host, port, path = endpoint
    credential = os.environ.get(args.key_env) if args.key_env else None
    if args.key_env and not credential:
        raise UsageError('environment variable {} is not set'.format(args.key_env))
    script = Script(args.seed)
    history = [{'role': 'system', 'content': ls.SYSTEM_PROMPT}]
    rows = []
    status = 'complete'
    error = None
    compactions = 0
    post_turns_left = None
    prev = None           # previous row (C3 predecessor)
    kind_next = 'chat'
    measured_ctx = 0
    user_turn = 0
    try:
        while len(rows) < args.max_requests:
            # user turn boundary: compact, stop, or add the next user message
            if (args.compaction and compactions < args.max_compactions and measured_ctx >= args.compact_at
                    and history[-1].get('role') != 'user'):
                cut = split_tail(history, args.keep_tokens)
                summary_req = history[:cut] + [{'role': 'user', 'content': COMPACTION_INSTRUCTION}]
                body = build_body(args.model, summary_req, args.summary_max_tokens, args.thinking)
                started = time.time()
                resp = send(host, port, path, body, args.timeout, args.session_id, credential)
                row = _turn_row(len(rows), 'compaction', user_turn, 0, summary_req, resp, prev, 'compaction')
                row['ledger'] = ledger.match(started, len(summary_req))
                thinking = _thinking(args, endpoint, credential, body, resp)
                if thinking is not None:
                    row['thinking'] = thinking
                row['kept_tail_messages'] = len(history) - cut
                rows.append(row)
                _log(err, row)
                checkpoint = {'role': 'user', 'content': CHECKPOINT_HEAD + (resp['content'] or resp['reasoning'] or '(empty summary)') + CHECKPOINT_TAIL}
                history = [history[0], checkpoint] + history[cut:]
                compactions += 1
                post_turns_left = args.post_compaction_turns
                prev = None  # the next request starts a new chain: not C3-eligible
                kind_next = 'post-compaction'
                measured_ctx = 0
                continue
            if post_turns_left is not None:
                if post_turns_left <= 0:
                    break
                post_turns_left -= 1
            elif not args.compaction and measured_ctx >= args.target:
                break
            user_turn += 1
            history.append(script.user())
            for hop in range(args.max_hops + 1):
                if len(rows) >= args.max_requests:
                    break
                body = build_body(args.model, list(history), args.max_tokens, args.thinking)
                started = time.time()
                resp = send(host, port, path, body, args.timeout, args.session_id, credential)
                basis = 'c3' if prev is not None else ('anchor' if kind_next == 'post-compaction' else 'first')
                row = _turn_row(len(rows), kind_next, user_turn, hop, history, resp, prev, basis)
                row['ledger'] = ledger.match(started, len(history))
                thinking = _thinking(args, endpoint, credential, body, resp)
                if thinking is not None:
                    row['thinking'] = thinking
                rows.append(row)
                _log(err, row)
                kind_next = 'chat'
                prev = row
                measured_ctx = (row['prompt_tokens'] or 0) + (row['completion_tokens'] or 0)
                history.append(assistant_message(resp, args.reasoning_echo == 'send'))
                if not resp['tool_calls'] or hop == args.max_hops:
                    if resp['tool_calls']:
                        # the client still answers every call before the next user message
                        for i, c in enumerate(resp['tool_calls']):
                            history.append(script.tool_result(c['id'] or 'call_{}_{}'.format(len(rows), i), c['name']))
                    break
                for i, c in enumerate(resp['tool_calls']):
                    cid = c['id'] or 'call_{}_{}'.format(len(rows), i)
                    if not c['id']:
                        history[-1]['tool_calls'][i]['id'] = cid
                    history.append(script.tool_result(cid, c['name']) if tool_call_valid(c) else script.invalid_tool_result(cid))
                if args.compaction and compactions < args.max_compactions and measured_ctx >= args.compact_at:
                    break  # compact before the next request (OpenCode checks per request)
    except AbortRun as exc:
        status, error = 'aborted_http_error', str(exc)
    return rows, status, error, compactions


def _log(err, row):
    err.write('req {} {} turn {} hop {} prompt={} cached={} eligible={} ttft_ms={} resume={} echo_n={} miss={}\n'.format(
        row['request'], row['kind'], row['user_turn'], row['hop'], row['prompt_tokens'], row['cached_tokens'],
        row['eligible_tokens'], row['ttft_ms'], row['timings'].get('resume_src'), row['timings'].get('echo_n'),
        (row.get('ledger') or {}).get('miss_class')))


def verdict(rows, ledger_dir, compactions, compaction_on, target, status):
    c3 = [r for r in rows if r['eligible_tokens'] is not None and r['eligible_tokens'] > 0]
    elig = sum(r['eligible_tokens'] for r in c3)
    hit = sum(r['eligible_hit_tokens'] for r in c3)
    ratio = hit / elig if elig else None
    short_all = [{'request': r['request'], 'kind': r['kind'], 'short': r['eligible_tokens'] - r['eligible_hit_tokens'],
                   'miss_class': (r.get('ledger') or {}).get('miss_class'),
                   'echo_miss_class': r['timings'].get('echo_miss_class')}
                  for r in c3 if r['eligible_hit_tokens'] < r['eligible_tokens']]
    calls = sum(r['tool_calls'] for r in rows if r['kind'] != 'compaction')
    valid = sum(r['tool_calls_valid'] for r in rows if r['kind'] != 'compaction')
    ledger_rows = [r['ledger'] for r in rows if r.get('ledger')]
    unknown = sum(1 for lr in ledger_rows if lr.get('miss_class') == 'unknown')
    classes = {}
    for lr in ledger_rows:
        if lr.get('miss_class'):
            classes[lr['miss_class']] = classes.get(lr['miss_class'], 0) + 1
    ledger_ok = ledger_dir is not None and len(ledger_rows) == len(rows)
    comp_rows = [r for r in rows if r['kind'] == 'compaction']
    post_rows = [r for r in rows if r['kind'] == 'post-compaction']
    comp_ok = (not compaction_on) or (compactions >= 1 and len(post_rows) == compactions
                                      and all((r['cached_tokens'] or 0) > 0 for r in post_rows)
                                      and all((r.get('ledger') or {}).get('miss_class') != 'unknown' for r in comp_rows + post_rows))
    peak = max((r['prompt_tokens'] or 0) for r in rows) if rows else 0
    gates = {
        'hit_ratio_eligible': {'value': None if ratio is None else round(ratio, 6), 'min': HIT_MIN,
                               'pass': ratio is not None and ratio >= HIT_MIN},
        # the per-request form of the same invariant: the aggregate ratio alone would pass a broken
        # ECHO (a lost reply is small next to the history), so every append request must meet C3
        'c3_per_request': {'short_requests': len(short_all), 'append_requests': len(c3),
                           'pass': bool(c3) and not short_all},
        'unknown_miss_class': {'value': unknown, 'max': 0, 'ledger_rows': len(ledger_rows),
                               'ledger_read': ledger_ok, 'pass': ledger_ok and unknown == 0},
        'tool_json_valid': {'valid': valid, 'calls': calls, 'pass': calls > 0 and valid == calls},
        'compaction': {'enabled': compaction_on, 'compactions': compactions, 'post_compaction_turns': len(post_rows),
                       'post_compaction_cached': [r['cached_tokens'] for r in post_rows], 'pass': comp_ok},
        'context_reached': {'peak_prompt_tokens': peak, 'target': target, 'pass': peak >= 0.9 * target},
        'run_complete': {'status': status, 'pass': status == 'complete'},
    }
    return {
        'passed': all(g['pass'] for g in gates.values()),
        'gates': gates,
        'eligible_tokens': elig, 'eligible_hit_tokens': hit, 'c3_requests': len(c3),
        'shortfalls': short_all[:64], 'miss_classes': classes,
        'reasoning_requests': sum(1 for r in rows if r['reasoning_chars'] > 0),
        'echo_n_total': sum(r['timings'].get('echo_n', 0) for r in rows),
        'roundtrip_miss_requests': sum(1 for r in rows if r['timings'].get('roundtrip_miss')),
        'ttft_ms_p50': _round(ls.percentile([r['ttft_ms'] for r in rows], 50)),
        'ttft_ms_p90': _round(ls.percentile([r['ttft_ms'] for r in rows], 90)),
        'decode_tok_s_p50': _round(ls.percentile([r['decode_tok_s'] for r in rows], 50)),
    }


# ---------------------------------------------------------------- fake server (dry run, tests)

class FakeServer(ThreadingHTTPServer):
    """OpenAI-shaped streaming fake with reasoning, tool calls and ECHO-like cache numbers.

    Tokens are chars/4 per message (+ tools). A request's cached tokens are the tokens of its
    longest message prefix shared with the previous request plus that request's generated reply
    (the reply counts as matched when content and tool calls are what the fake produced, whatever
    the reasoning: the engine's ECHO restores it). With `ledger_dir`, one controller-like ledger
    row per request (origin lab, n_messages, miss_class append-ok / compaction / unknown).
    `break_echo` makes replies never match (the C3 failure case); `bad_json_every` emits invalid
    tool-call arguments on every N-th tool call. `shrink_rerender` re-renders each echoed reply
    without its reasoning (the reply still matches, but its reasoning tokens are gone from the
    prompt and the cache: the lost-reasoning case C3 must catch even when the prompt shrinks).
    """
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, ledger_dir=None, tool_every=2, break_echo=False, bad_json_every=0, reasoning_chars=240,
                 shrink_rerender=False, no_tokenize=False):
        super().__init__(('127.0.0.1', 0), _FakeHandler)
        self.ledger_dir = Path(ledger_dir) if ledger_dir else None
        self.no_tokenize = no_tokenize  # helper routes answer 404 (an engine without them)
        self.helper_requests = []       # /tokenize and /apply-template bodies, kept apart from requests
        self.tool_every, self.break_echo, self.bad_json_every = tool_every, break_echo, bad_json_every
        self.reasoning_chars = reasoning_chars
        self.shrink_rerender = shrink_rerender
        self.lock = threading.Lock()
        self.requests = []
        self.headers_seen = []
        self.prev = None   # (message keys, generated reply key) of the previous request
        self.gen = {}      # generated reply key -> (completion tokens, reasoning tokens)
        self.last_kind = None
        self.calls = 0
        self.url = 'http://127.0.0.1:{}/v1'.format(self.server_address[1])
        self._thread = threading.Thread(target=self.serve_forever, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self.shutdown()
        self.server_close()


def _fake_tok(text):
    return max(1, math.ceil(len(text) / 4))


def _msg_key(m):
    calls = [(c['function']['name'], c['function']['arguments']) for c in m.get('tool_calls') or []]
    return json.dumps([m.get('role'), m.get('content') or '', calls, m.get('tool_call_id')], sort_keys=True)


def _reply_key(content, calls):
    return json.dumps(['assistant', content, [(c['name'], c['arguments']) for c in calls], None], sort_keys=True)


class _FakeHandler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.0'

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get('Content-Length', '0'))))
        srv = self.server
        if self.path in HELPER_PATHS:
            return self._helper(body)
        with srv.lock:
            idx = len(srv.requests)
            srv.requests.append(body)
            srv.headers_seen.append({k: v for k, v in self.headers.items()})
            msgs = body['messages']
            keys = [_msg_key(m) for m in msgs]
            toks = [_fake_tok(k) for k in keys]
            tools_tok = _fake_tok(json.dumps(body.get('tools') or []))
            last = msgs[-1].get('content') or ''
            compaction = 'summarize the conversation above into a structured summary' in last
            thinking = (body.get('chat_template_kwargs') or {}).get('enable_thinking')
            # the generated reply (deterministic from the request index)
            reasoning = ('Considering the next step. ' * 64)[:srv.reasoning_chars] if thinking else ''
            calls = []
            if not compaction and srv.tool_every and idx % srv.tool_every == 0:
                srv.calls += 1
                args = json.dumps({'filePath': 'src/fake_{}.py'.format(idx)})
                if srv.bad_json_every and srv.calls % srv.bad_json_every == 0:
                    args = '{"filePath": "src/broken'
                calls = [{'id': 'call_f{}'.format(idx), 'name': 'read', 'arguments': args}]
                content = ''
            else:
                content = 'Summary: objective, decisions and next move.' if compaction else 'Done with step {}.'.format(idx)
            completion = _fake_tok(reasoning) + _fake_tok(_reply_key(content, calls))
            # tokens: an assistant message the fake generated counts as its generated tokens
            # (ECHO splices them, reasoning included); with break_echo it is re-rendered instead
            def t_of(i):
                if msgs[i].get('role') == 'assistant' and keys[i] in srv.gen and not srv.break_echo:
                    comp, think = srv.gen[keys[i]]
                    return comp - think if srv.shrink_rerender else comp
                return toks[i]
            prompt = tools_tok + sum(t_of(i) for i in range(len(keys)))
            cached = 0
            if srv.prev is not None:
                pkeys, preply = srv.prev
                cached = tools_tok
                for i, k in enumerate(keys):
                    if i < len(pkeys) and k == pkeys[i]:
                        cached += t_of(i)
                    elif i == len(pkeys) and k == preply and not srv.break_echo:
                        cached += t_of(i)  # the previous reply, still in the engine's cache
                    else:
                        break
            cached = max(0, min(cached, prompt - 1))
            srv.gen[_reply_key(content, calls)] = (completion, _fake_tok(reasoning) if reasoning else 0)
            srv.prev = (keys, _reply_key(content, calls))
            if srv.ledger_dir is not None:
                post = any('<conversation-checkpoint>' in (m.get('content') or '') for m in msgs if m.get('role') == 'user')
                kind = 'compaction' if compaction else ('post-compaction' if post else 'chat')
                row = {'id': 'f{:05d}'.format(idx), 't': round(time.time(), 3), 'origin': 'lab', 'session': 'fakesession',
                       'n_messages': len(msgs), 'turn_kind': kind, 'cache_n': cached, 'prompt_n': prompt - cached}
                if idx > 0:
                    first_post = kind == 'post-compaction' and srv.last_kind != 'post-compaction'
                    row['miss_class'] = ('compaction' if kind == 'compaction' or first_post
                                         else 'unknown' if srv.break_echo else 'append-ok')
                srv.last_kind = kind
                srv.ledger_dir.mkdir(parents=True, exist_ok=True)
                with open(srv.ledger_dir / 'requests-{}.jsonl'.format(time.strftime('%Y%m%d')), 'a') as f:
                    f.write(json.dumps(row) + '\n')
        usage = {'prompt_tokens': prompt, 'completion_tokens': completion,
                 'prompt_tokens_details': {'cached_tokens': cached}}
        timings = {'cache_n': cached, 'prompt_n': prompt - cached, 'prompt_ms': 5.0, 'predicted_n': completion,
                   'predicted_ms': 10.0, 'predicted_per_second': 30.0, 'resume_src': 'live' if cached else 'cold',
                   'resume_ms': 0.1, 'echo_blocks': 1 if idx else 0, 'echo_n': 1 if (idx and not srv.break_echo) else 0,
                   'echo_fixed': 0, 'echo_tokens': 0, 'roundtrip_miss': 0,
                   'echo_miss_class': 'content-mismatch' if srv.break_echo and idx else None}
        if timings['echo_miss_class'] is None:
            del timings['echo_miss_class']
        chunks = [{'choices': [{'delta': {'role': 'assistant'}, 'index': 0}]}]
        if reasoning:
            half = len(reasoning) // 2
            chunks += [{'choices': [{'delta': {'reasoning_content': reasoning[:half]}, 'index': 0}]},
                       {'choices': [{'delta': {'reasoning_content': reasoning[half:]}, 'index': 0}]}]
        if content:
            chunks.append({'choices': [{'delta': {'content': content}, 'index': 0}]})
        for i, c in enumerate(calls):
            a = c['arguments']
            chunks.append({'choices': [{'delta': {'tool_calls': [{'index': i, 'id': c['id'], 'type': 'function',
                                                                  'function': {'name': c['name'], 'arguments': a[:5]}}]}, 'index': 0}]})
            chunks.append({'choices': [{'delta': {'tool_calls': [{'index': i, 'function': {'arguments': a[5:]}}]}, 'index': 0}]})
        chunks.append({'choices': [{'delta': {}, 'index': 0, 'finish_reason': 'tool_calls' if calls else 'stop'}]})
        chunks.append({'choices': [], 'usage': usage, 'timings': timings})
        payload = b''.join(b'data: ' + json.dumps(c).encode() + b'\n\n' for c in chunks) + b'data: [DONE]\n\n'
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


    def _helper(self, body):
        srv = self.server
        with srv.lock:
            srv.helper_requests.append({'path': self.path, 'body': body})
        if srv.no_tokenize:
            status, doc = 404, {'error': 'not found'}
        elif self.path == '/tokenize':
            status, doc = 200, {'tokens': _fake_ids(body.get('content') or '')}
        else:
            prompt = _fake_render(body.get('messages') or [])
            status, doc = 200, {'prompt': prompt, 'tokens': _fake_ids(prompt), 'echo_tokens': _fake_ids(prompt)}
        payload = json.dumps(doc).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@functools.lru_cache(maxsize=65536)
def _word_id(word):
    return int(hashlib.sha1(word.encode('utf-8')).hexdigest()[:6], 16)


def _fake_ids(text):
    """Stand-in tokenizer: one id per whitespace-separated word (stable across calls)."""
    return [_word_id(w) for w in text.split()]


def _fake_render(messages):
    """Stand-in chat template: role, content and any echoed reasoning, in order."""
    return ''.join('<|{}|>{}{}\n'.format(m.get('role'), m.get('reasoning_content') or '', m.get('content') or '')
                   for m in messages)


# ---------------------------------------------------------------- CLI

def run(args, out=None, err=None):
    out, err = out or sys.stdout, err or sys.stderr
    if args.max_requests < 1 or args.max_hops < 0:
        raise UsageError('--max-requests >= 1 and --max-hops >= 0 required')
    args.target = ls.parse_size(args.target_context)
    args.compact_at = ls.parse_size(args.compact_at) if args.compact_at else args.target
    args.keep_tokens = ls.parse_size(args.keep_tokens)
    if args.session_id is None:
        args.session_id = 'lab-closed-loop-{}-{}-{}'.format(args.seed, args.target, os.urandom(4).hex())
    receipt = {
        'schema': RECEIPT_SCHEMA, 'tool': {'name': TOOL, 'version': VERSION},
        'started_utc': _utc(), 'dry_run': bool(args.dry_run), 'model': args.model,
        'params': {'seed': args.seed, 'target_context': args.target, 'thinking': args.thinking,
                   'reasoning_echo': args.reasoning_echo, 'compaction': args.compaction,
                   'compact_at': args.compact_at, 'keep_tokens': args.keep_tokens,
                   'post_compaction_turns': args.post_compaction_turns, 'max_compactions': args.max_compactions,
                   'max_tokens': args.max_tokens, 'summary_max_tokens': args.summary_max_tokens,
                   'thinking_metrics': bool(args.thinking_metrics),
                   'max_hops': args.max_hops, 'max_requests': args.max_requests, 'request_seed': REQUEST_SEED,
                   'temperature': 0, 'preserve_thinking': True},
        'session_id_sha256_12': hashlib.sha256(args.session_id.encode()).hexdigest()[:12],
        'scope': ('closed loop: each request carries the actual previous reply (content, tool_calls'
                  + (', reasoning_content' if args.reasoning_echo == 'send' else '; reasoning dropped as OpenCode')
                  + ') and synthetic tool results; synthetic user text; OpenCode 2.2.x compaction layout'),
    }
    if args.dry_run:
        with tempfile.TemporaryDirectory(prefix='closed-loop-dry-') as tmp, FakeServer(ledger_dir=tmp) as srv:
            endpoint = ls.parse_endpoint(srv.url)
            rows, status, error, comps = run_session(args, endpoint, Ledger(tmp), err)
            receipt['endpoint'] = {'fake': True}
            receipt.update(_finish(rows, status, error, comps, args, ledger_dir=tmp))
    else:
        if not args.receipt:
            raise UsageError('--receipt is required for a live run')
        endpoint = ls.parse_endpoint(args.base_url)
        receipt['endpoint'] = {'host': endpoint[0], 'port': endpoint[1], 'path': endpoint[2]}
        rows, status, error, comps = run_session(args, endpoint, Ledger(args.ledger_dir), err)
        receipt.update(_finish(rows, status, error, comps, args, ledger_dir=args.ledger_dir))
    receipt['finished_utc'] = _utc()
    if args.receipt:
        ls.write_json_atomic(args.receipt, receipt)
    summary = {'passed': receipt['verdict']['passed'], 'status': receipt['status'],
               'requests': len(receipt['requests']),
               'gates': {k: v['pass'] for k, v in receipt['verdict']['gates'].items()},
               'hit_ratio_eligible': receipt['verdict']['gates']['hit_ratio_eligible']['value'],
               'receipt': str(args.receipt) if args.receipt else None, 'dry_run': bool(args.dry_run)}
    out.write(json.dumps(summary, indent=2) + '\n')
    return 0 if receipt['verdict']['passed'] else 1


def _finish(rows, status, error, comps, args, ledger_dir):
    return {'status': status, 'error': error, 'requests': rows,
            'verdict': verdict(rows, ledger_dir, comps, args.compaction, args.target, status),
            'thinking_summary': thinking_summary(rows)}


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--base-url', default=ls.DEFAULT_BASE_URL, help='http://127.0.0.1:PORT/v1 (loopback only)')
    p.add_argument('--model', required=True, help='model name the endpoint expects')
    p.add_argument('--receipt', help='JSON receipt to write (required unless --dry-run)')
    p.add_argument('--seed', type=int, default=1, help='synthetic user/tool text seed')
    p.add_argument('--target-context', default='32K', help='context to grow to (32K, 64K, 128K)')
    p.add_argument('--thinking', action=argparse.BooleanOptionalAction, default=False,
                   help='chat_template_kwargs.enable_thinking')
    p.add_argument('--reasoning-echo', choices=('send', 'drop'), default='send',
                   help='send the returned reasoning_content back (default) or drop it as OpenCode does')
    p.add_argument('--compaction', action=argparse.BooleanOptionalAction, default=True,
                   help='compact once the measured context reaches --compact-at (OpenCode 2.2.x layout)')
    p.add_argument('--compact-at', default=None, help='compaction threshold in tokens (default: the target)')
    p.add_argument('--keep-tokens', default=str(DEFAULT_KEEP), help='verbatim tail kept after compaction (estimated)')
    p.add_argument('--post-compaction-turns', type=int, default=DEFAULT_POST_TURNS,
                   help='user turns after the (last) compaction before the run ends')
    p.add_argument('--max-compactions', type=int, default=1)
    p.add_argument('--thinking-metrics', action=argparse.BooleanOptionalAction, default=True,
                   help='O86 numbers per turn: reasoning tokens and n=3..8 copy share vs the preceding context')
    p.add_argument('--max-tokens', type=int, default=256, help='per chat request')
    p.add_argument('--summary-max-tokens', type=int, default=1024, help='for the compaction request')
    p.add_argument('--max-hops', type=int, default=DEFAULT_MAX_HOPS, help='tool rounds per user turn')
    p.add_argument('--max-requests', type=int, default=400)
    p.add_argument('--timeout', type=float, default=1800.0, help='per-request seconds')
    p.add_argument('--session-id', default=None, help='X-Session-Id (default: unique per run)')
    p.add_argument('--ledger-dir', default=None,
                   help="controller ledger directory to read miss_class from (read only), e.g. "
                        "~/.local/state/local-ai/ledger; without it the unknown-miss gate cannot pass")
    p.add_argument('--key-env', help='name of an environment variable holding an API key')
    p.add_argument('--dry-run', action='store_true',
                   help='run the loop against a built-in fake server on 127.0.0.1 (no real endpoint)')
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except (UsageError, ls.UsageError, OSError) as exc:
        sys.stderr.write('closed_loop_session: {}\n'.format(exc))
        return 2


if __name__ == '__main__':
    sys.exit(main())
