"""Always-on request ledger for the local inference proxy: one JSON line per inference request.

Numbers, hashes and enum-like labels only, never prompt or reply content:
- queue wait (FIFO admission), admit time (includes a model load), time to the backend's first
  byte, wall time, HTTP status;
- labels: client product token from User-Agent (e.g. "opencode/2.2.6"), keyed (install-salt HMAC) hash of the
  system prompt (agent identity), message/tool counts, last role, stream flag, max_tokens,
  temperature;
- the backend's own numbers from the end of its response (llama-server/oxe `timings`:
  cache_n, prompt_n, prompt_ms, predicted_n, predicted_ms, draft counters; finish_reason),
  and the same counts under plain names: ctx_tokens (= cached + new), cached_tokens, new_tokens,
  output_tokens, prefill_ms, decode_ms (ttfb_ms is the time to first byte, the TTFT);
- REAL-1 origin: `origin` is 'lab' only when the request carries `X-Local-AI-Origin: lab` (the
  lab's replay/gate tools send it), 'remote' for a request on a LAN share listener (lan_share.py;
  `client_addr` is then a keyed hash of the client address, never the address), otherwise 'organic'. `session` is a keyed hash (HMAC with a
  per-install salt file, owner-only) of the client's session header, else of the first user
  message, so it is never reversible; `gap_ms` is the time since the previous request of the same
  session finished; `tool_bytes` lists the byte size of each tool-result message (sizes only).
- CACHE-PERFECT M2: `completeness` (complete | closed-after-done | incomplete | errored) and, for a
  chat turn after a complete predecessor of the same (session, model, agent, origin),
  `previous_reply` (present | edited | absent | unknown): whether the returned assistant reply
  (a keyed digest held in memory, never written) is the message after the previous input prefix.

Files: LOCAL_AI_LEDGER_DIR (default ~/.local/state/local-ai/ledger)/requests-YYYYMMDD.jsonl,
directory 0700, files 0600, kept LOCAL_AI_LEDGER_DAYS (14) days. LOCAL_AI_LEDGER=0 turns it
off. A ledger failure never affects the request.
"""
from collections import OrderedDict
import hashlib
import hmac
import json
import logging
import math
import os
from pathlib import Path
import re
import secrets
import stat
import threading
import time

TAIL_BYTES = 16384
_UA = re.compile(r'^[A-Za-z0-9._-]{1,32}(/[A-Za-z0-9._+-]{1,24})?')
_FINISH = re.compile(rb'"finish_reason"\s*:\s*"([a-z_]{1,24})"')
ORIGIN_HEADER = 'X-Local-AI-Origin'
ECHO_MISS_CLASSES = frozenset({'no-record', 'content-mismatch', 'tool-mismatch', 'reasoning-mismatch', 'layout'})
LAB_ORIGIN = 'lab'
REMOTE_ORIGIN = 'remote'
_CLIENT_HASH = re.compile(r'[0-9a-f]{8,32}')
SESSION_HEADERS = ('X-Session-Id', 'X-Claude-Code-Session-Id', 'Session-Id')
TOOL_BYTES_MAX = 128
GAP_SESSIONS = 512
_lock = threading.Lock()
_pruned_day = [None]
_salt = [None]
_gap_lock = threading.Lock()
_last_end = OrderedDict()  # session hash -> monotonic time its last request finished
_log = logging.getLogger('local_ai.ledger')
HISTORY_SESSIONS = 256
HISTORY_MESSAGES = 4096
HISTORY_HASH_BUDGET = 32768  # <= 512 KiB of digests, plus bounded Python bookkeeping
_history_lock = threading.Lock()
_history = OrderedDict()  # (session, model, agent, origin) -> last complete chat turn: id, digests, counts


def _message_hash(message):
    # Keyed and order-preserving: tool argument strings and part order affect rendering.
    digest = hmac.new(_install_salt(), b'ledger-message-v1\0', hashlib.sha256)
    for chunk in json.JSONEncoder(ensure_ascii=False, separators=(',', ':'), allow_nan=False).iterencode(message):
        digest.update(chunk.encode('utf-8', 'replace'))
    return digest.digest()[:16]


_history_seq = [0]
CHAT_PATHS = frozenset({'/v1/chat/completions', '/chat/completions'})
REPLY_MAX = 1 << 20  # bytes of reply text kept for the reply digest; beyond it the digest is unknown
_ROLES = frozenset({'system', 'user', 'assistant', 'tool'})


def _canon_args(value):
    """Tool arguments as a JSON value (key order and escapes do not matter, as in the engine's ECHO)."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def reply_digest(content, calls):
    """Keyed digest of an assistant turn: trimmed text plus (name, arguments) of each tool call.
    Reasoning is left out (the engine's ECHO report judges it)."""
    digest = hmac.new(_install_salt(), b'ledger-reply-v1\0', hashlib.sha256)
    digest.update(json.dumps([content.strip(), [[n or '', _canon_args(a or '')] for n, a in calls]],
                             ensure_ascii=False).encode('utf-8', 'replace'))
    return digest.digest()[:16]


def message_reply_digest(message):
    calls = [((c.get('function') or {}).get('name'), (c.get('function') or {}).get('arguments'))
             for c in message.get('tool_calls') or [] if isinstance(c, dict)]
    return reply_digest(_text(message.get('content')), calls)


def is_comment_only(chunk):
    """True when every non-empty line of `chunk` is an SSE comment (starts with ':')."""
    lines = [line for line in chunk.split(b'\n') if line.strip()]
    return bool(lines) and all(line.lstrip().startswith(b':') for line in lines)


class ReplyCollector:
    """The returned assistant turn of one chat response (SSE deltas or a JSON body), kept in memory
    only until finish(), which turns it into a digest. Never written anywhere."""

    def __init__(self):
        self.buf, self.body, self.text, self.calls, self.size, self.mode, self.broken = b'', b'', [], {}, 0, None, False

    def feed(self, chunk):
        if self.broken:
            return
        if self.mode is None and chunk.strip():
            self.mode = 'json' if chunk.lstrip()[:1] == b'{' else 'sse'
        if self.mode == 'json':
            self.body += chunk
            self.broken = len(self.body) > REPLY_MAX
            return
        *lines, self.buf = (self.buf + chunk).split(b'\n')
        for line in lines:
            if line.startswith(b'data:') and line[5:].strip() not in (b'', b'[DONE]'):
                try:
                    self._choice(json.loads(line[5:]), 'delta')
                except (ValueError, AttributeError, TypeError, IndexError):
                    self.broken = True
        self.broken = self.broken or self.size > REPLY_MAX or len(self.buf) > REPLY_MAX

    def _choice(self, obj, key):
        choices = obj.get('choices') if isinstance(obj, dict) else None
        if not choices:
            return
        part = choices[0].get(key) or {}
        if isinstance(part.get('content'), str):
            self.text.append(part['content'])
            self.size += len(part['content'])
        for call in part.get('tool_calls') or []:
            slot = self.calls.setdefault(call.get('index', len(self.calls)) if key == 'delta' else len(self.calls), ['', ''])
            fn = call.get('function') or {}
            slot[0] += fn.get('name') or ''
            args = fn.get('arguments')
            slot[1] += args if isinstance(args, str) else json.dumps(args) if args is not None else ''
            self.size += len(slot[1])

    def digest(self):
        if self.broken:
            return None
        if self.mode == 'json':
            try:
                self._choice(json.loads(self.body), 'message')
            except (ValueError, AttributeError, TypeError, IndexError):
                return None
        if not self.text and not self.calls:
            return None
        return reply_digest(''.join(self.text), [tuple(self.calls[k]) for k in sorted(self.calls)])


def history_start(entry, payload):
    """Compare request histories without keeping any text or claiming token-level equality.

    Newly appended messages (including the previous generated assistant) are not rewrites.
    Cache loss despite an append is reported separately from an edited earlier message.
    The predecessor is the last COMPLETE chat turn of the same (session, model, agent, origin):
    a request only becomes the predecessor in history_finish, so error, incomplete and non-chat
    rows in between are skipped (CACHE-PERFECT M2).
    """
    if not entry.session or not isinstance(payload, dict) or not isinstance(payload.get('messages'), list):
        return
    if entry.row.get('path', '').split('?')[0] not in CHAT_PATHS:
        return
    messages = payload['messages']
    row = entry.row
    key = (entry.session, row.get('model'), row.get('agent'), row.get('origin'))
    if len(messages) > HISTORY_MESSAGES:
        row['history_change'] = 'over_limit'
        with _history_lock:
            _history.pop(key, None)
        return
    hashes = tuple(_message_hash(m) for m in messages)
    roles = tuple(m.get('role') if isinstance(m, dict) and m.get('role') in _ROLES else None for m in messages)
    with _history_lock:
        _history_seq[0] += 1
        entry.history = {'seq': _history_seq[0], 'hashes': hashes, 'roles': roles}
        previous = _history.get(key)
        common = 0
        if previous is None:
            row['history_change'] = 'initial'
        else:
            old = previous['hashes']
            common = next((i for i, (a, b) in enumerate(zip(old, hashes)) if a != b), min(len(old), len(hashes)))
            row['previous_request_id'] = previous['id']
            row['history_change'] = ('edited' if common < min(len(old), len(hashes)) else
                                     'truncated' if len(hashes) < len(old) else
                                     'same' if len(hashes) == len(old) else 'append')
            if row['history_change'] in ('edited', 'truncated'):
                row['first_divergent_message'] = common
                row['first_divergent_message_hash'] = hashes[common].hex() if common < len(hashes) else None
                row['previous_divergent_message_hash'] = old[common].hex()
                row['first_divergent_role'] = previous['roles'][common]
            if previous.get('expected_cache') is not None:
                row['previous_expected_cache'] = previous['expected_cache']
            entry.prev_turn = previous.get('turn')
        row['history_common_messages'] = common
    if previous is not None and row['history_change'] in ('same', 'append'):
        # C3 eligibility: is the returned reply the message right after the previous input prefix?
        nxt = messages[len(previous['hashes'])] if row['history_change'] == 'append' else None
        if not isinstance(nxt, dict) or nxt.get('role') != 'assistant':
            row['previous_reply'] = 'absent'
        elif previous.get('reply') is None:
            row['previous_reply'] = 'unknown'
        else:
            row['previous_reply'] = 'present' if hmac.compare_digest(message_reply_digest(nxt), previous['reply']) else 'edited'
    entry.history_key = key


def completeness(status, error, row):
    """complete | closed-after-done | incomplete | errored. The backend's timings come with its last
    event, so their presence says the reply was fully generated; a client that closed after that is
    a transport label and the cache state is still valid."""
    done = _num(row.get('predicted_n')) is not None and _num(row.get('cache_n')) is not None
    if status != 200 or (error and error != 'client_disconnected'):
        return 'errored'
    if not done:
        return 'incomplete'
    return 'closed-after-done' if error else 'complete'


def history_finish(entry):
    row = entry.row
    row['completeness'] = completeness(row.get('status'), row.get('error'), row)
    expected = row.get('previous_expected_cache')
    if expected is not None and _num(row.get('cache_n')) is not None:
        row['cache_shortfall_tokens'] = max(0, expected - row['cache_n'])
    if entry.history is None or row['completeness'] not in ('complete', 'closed-after-done'):
        return
    turn = classify_turn(entry, row)
    counts = [row.get(k) for k in ('cache_n', 'prompt_n', 'predicted_n')]
    record = {'id': row['id'], 'seq': entry.history['seq'], 'hashes': entry.history['hashes'],
              'roles': entry.history['roles'], 'turn': turn,
              'reply': entry.reply.digest() if entry.reply is not None else None,
              'expected_cache': sum(counts) if all(_num(n) is not None and n >= 0 for n in counts) else None}
    with _history_lock:
        previous = _history.get(entry.history_key)
        # A slower overlapping request must not overwrite the next request's history.
        if previous is None or previous['seq'] < record['seq']:
            _history[entry.history_key] = record
            _history.move_to_end(entry.history_key)
        while (len(_history) > HISTORY_SESSIONS or
               sum(len(item['hashes']) for item in _history.values()) > HISTORY_HASH_BUDGET):
            _history.popitem(last=False)


def enabled():
    return os.environ.get('LOCAL_AI_LEDGER', '1') != '0'


def ledger_dir():
    return Path(os.environ.get('LOCAL_AI_LEDGER_DIR', str(Path.home() / '.local/state/local-ai/ledger')))


def _text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return ''.join(p.get('text', '') for p in content if isinstance(p, dict) and isinstance(p.get('text'), str))
    return ''


def _num(v):
    return v if type(v) in (int, float) and math.isfinite(v) else None


def origin_of(value):
    """'lab' only for the explicit marker; anything else (or nothing) is organic traffic."""
    return LAB_ORIGIN if isinstance(value, str) and value.strip().lower() == LAB_ORIGIN else 'organic'


def _install_salt():
    """32 random bytes in an owner-only file under the ledger dir; created once."""
    with _lock:
        if _salt[0] is None:
            d = ledger_dir()
            d.mkdir(parents=True, exist_ok=True)
            os.chmod(d, 0o700)
            path = d / '.salt'
            try:
                fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
            except FileNotFoundError:
                fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
                os.write(fd, secrets.token_bytes(32))
                os.close(fd)
                fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
            try:
                info = os.fstat(fd)
                value = os.read(fd, 64)
                if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or
                        info.st_mode & 0o077 or len(value) != 32):
                    raise ValueError('invalid_private_ledger_salt')
                _salt[0] = value
            finally:
                os.close(fd)
        return _salt[0]


def keyed_short(domain, data):
    """12 hex of HMAC(install salt, domain NUL data): equal inputs match within this install, but
    a ledger file alone cannot be checked against a guessed prompt or tool list. None when the
    install salt is missing or not private."""
    try:
        salt = _install_salt()
    except (OSError, ValueError):  # no private salt: no hash at all, never an unkeyed one
        return None
    return hmac.new(salt, domain + b'\0' + data, hashlib.sha256).hexdigest()[:12]


def session_key(headers, payload):
    """Keyed hash naming one client session, or None. Uses an explicit session header when the
    client sends one, else the first user message (constant across a session's turns)."""
    raw = None
    for name in SESSION_HEADERS:
        value = headers.get(name) if headers is not None else None
        if isinstance(value, str) and 1 <= len(value) <= 200:
            raw = 'h:' + value
            break
    if raw is None and isinstance(payload, dict):
        msgs = payload.get('messages') if isinstance(payload.get('messages'), list) else []
        first = next((x for x in msgs if isinstance(x, dict) and x.get('role') == 'user'), None)
        if first is not None:
            raw = 'm:' + _text(first.get('content'))
    if raw is None:
        return None
    return hmac.new(_install_salt(), raw.encode('utf-8', 'replace'), hashlib.sha256).hexdigest()[:16]


class Entry:
    def __init__(self, path, payload, user_agent=None, model_alias=None, headers=None, origin=None, client=None):
        self.t0 = time.monotonic()
        self.row = {'id': secrets.token_hex(6), 't': round(time.time(), 3), 'path': path,
                    'model': model_alias if isinstance(model_alias, str) else None,
                    'origin': 'organic'}
        m = _UA.match(user_agent or '')
        self.row['client'] = m.group(0) if m else None
        self.session = None
        self.history_key = self.history = self.prev_turn = None
        self.reply = ReplyCollector() if path.split('?')[0] in CHAT_PATHS else None
        try:
            self.row['origin'] = origin_of(headers.get(ORIGIN_HEADER) if headers is not None else None)
        except Exception:  # the marker is optional; a bad header must never fail a request
            pass
        if origin == REMOTE_ORIGIN:  # set by the controller only, before history_start keys on it
            self.row['origin'] = REMOTE_ORIGIN
            if isinstance(client, str) and _CLIENT_HASH.fullmatch(client):
                self.row['client_addr'] = client
        if isinstance(payload, dict):
            msgs = payload.get('messages') if isinstance(payload.get('messages'), list) else []
            system = next((_text(x.get('content')) for x in msgs if isinstance(x, dict) and x.get('role') == 'system'), None)
            self.row.update({
                'agent': keyed_short(b'ledger-agent-v1', system.encode('utf-8', 'replace')) if system is not None else None,
                'n_messages': len(msgs),
                'last_role': msgs[-1].get('role') if msgs and isinstance(msgs[-1], dict) and isinstance(msgs[-1].get('role'), str) else None,
                'n_tools': len(payload['tools']) if isinstance(payload.get('tools'), list) else 0,
                'stream': bool(payload.get('stream')),
                'max_tokens': _num(payload.get('max_tokens', payload.get('n_predict'))),
                'temperature': _num(payload.get('temperature')),
            })
            try:
                tools = [x for x in msgs if isinstance(x, dict) and x.get('role') == 'tool']
                self.row['n_tool_msgs'] = len(tools)
                self.row['tool_bytes'] = [len(_text(x.get('content')).encode('utf-8', 'replace')) for x in tools[:TOOL_BYTES_MAX]]
            except Exception:
                pass
        try:
            self.session = session_key(headers, payload)
            self.row['session'] = self.session
            with _gap_lock:
                prev = _last_end.get(self.session) if self.session else None
            self.row['gap_ms'] = round((self.t0 - prev) * 1e3, 1) if prev is not None else None
        except Exception:
            self.row['session'] = None
            self.row['gap_ms'] = None
        try:
            history_start(self, payload)
        except Exception:
            pass  # advisory telemetry must never fail inference
        self.admitted = None
        self.first_byte = None
        self.tail = b''

    def label(self, harness=None, prefix_key=None):
        """Harness registry labels (O-9): the detected harness name and the registry's prefix
        key (a hash), both safe to log."""
        if isinstance(harness, str) and re.fullmatch(r'[A-Za-z0-9._-]{1,32}', harness):
            self.row['harness'] = harness
        if isinstance(prefix_key, str) and re.fullmatch(r'[0-9a-f]{8,64}', prefix_key):
            self.row['prefix_key'] = prefix_key[:16]

    def label_turn(self, turn):
        """CACHE-PERFECT turn labels (cache_turn.py): enum labels and 12-hex hashes only."""
        if not isinstance(turn, dict):
            return
        try:
            try:
                from .cache_turn import TURN_KINDS, LAYOUTS
            except ImportError:
                from cache_turn import TURN_KINDS, LAYOUTS
            if turn.get('turn_kind') in TURN_KINDS:
                self.row['turn_kind'] = turn['turn_kind']
            if turn.get('compact_layout') in LAYOUTS:
                self.row['compact_layout'] = turn['compact_layout']
            for key in ('instruction_hash', 'tools_hash'):
                if isinstance(turn.get(key), str) and re.fullmatch(r'[0-9a-f]{12}', turn[key]):
                    # cache_turn's hashes are unkeyed; key them before they reach the file (audit S4).
                    self.row[key] = keyed_short(b'ledger-turn-v1', turn[key].encode())
        except Exception:
            pass

    def admitted_now(self, queue_wait_s=None, begin_s=None):
        self.admitted = time.monotonic()
        self.row['queue_wait_ms'] = round(queue_wait_s * 1e3, 1) if queue_wait_s is not None else None
        self.row['admit_ms'] = round(begin_s * 1e3, 1) if begin_s is not None else None

    def observe(self, chunk):
        # v0.1.3 LONG-APPEND: the engine's SSE keepalive comments (`: prefill ...`) are not the
        # reply's first byte: ttfb stays the time to the first data
        if self.first_byte is None and not is_comment_only(chunk):
            self.first_byte = time.monotonic()
        self.tail = (self.tail + chunk)[-TAIL_BYTES:]
        if self.reply is not None:
            try:
                self.reply.feed(chunk)
            except Exception:  # advisory: the reply digest must never fail a request
                self.reply = None

    def finish(self, status, error=None):
        now = time.monotonic()
        r = self.row
        r['status'] = status
        r['error'] = error
        r['wall_ms'] = round((now - self.t0) * 1e3, 1)
        if self.admitted is not None and self.first_byte is not None:
            r['ttfb_ms'] = round((self.first_byte - self.admitted) * 1e3, 1)
        r.update(parse_tail(self.tail))
        derive_counts(r)
        try:
            history_finish(self)
        except Exception:
            pass
        if self.session:
            with _gap_lock:
                _last_end[self.session] = now
                _last_end.move_to_end(self.session)
                while len(_last_end) > GAP_SESSIONS:
                    _last_end.popitem(last=False)
        write(r)
        return r


def derive_counts(r):
    """Plain-name token and time counts from the backend's timings; no tokenisation here."""
    cache, new = _num(r.get('cache_n')), _num(r.get('prompt_n'))
    if cache is not None and new is not None:
        r['cached_tokens'], r['new_tokens'], r['ctx_tokens'] = cache, new, cache + new
    if _num(r.get('predicted_n')) is not None:
        r['output_tokens'] = r['predicted_n']
    if _num(r.get('prompt_ms')) is not None:
        r['prefill_ms'] = r['prompt_ms']
    if _num(r.get('predicted_ms')) is not None:
        r['decode_ms'] = r['predicted_ms']
    return r


def classify_turn(entry, row):
    """Cache-miss class of one complete chat turn against its predecessor (history_start picked it:
    same session, model, agent and origin). Adds miss_class, miss_tokens and miss_basis when judged;
    `unknown` logs a WARN with counts only. Returns this turn's summary (the next turn's
    predecessor). Never raises: a classifier fault must not cost the request its ledger row."""
    try:
        try:
            from .cache_miss import classify, summarize
        except ImportError:
            from cache_miss import classify, summarize
        cur = summarize(row)
        if cur is None:
            return None
    except Exception:
        return None
    try:
        cls, miss, basis = classify(entry.prev_turn, cur)
        if cls is None:
            return cur
        row['miss_class'], row['miss_tokens'], row['miss_basis'] = cls, miss, basis
        if cls == 'unknown':
            _log.warning('unclassified cache miss: %s tokens short, n_messages=%s, last_role=%s',
                         miss, row.get('n_messages'), row.get('last_role'))
    except Exception:  # the classifier is diagnostic; the request is already served
        pass
    return cur


def parse_tail(tail):
    """Backend numbers from the end of a response (SSE stream or JSON body)."""
    out = {}
    i = tail.rfind(b'"timings"')
    if i >= 0:
        j = tail.find(b'{', i)
        try:
            t, _ = json.JSONDecoder().raw_decode(tail[j:].decode('utf-8', 'replace'))
            for k in ('cache_n', 'prompt_n', 'prompt_ms', 'predicted_n', 'predicted_ms', 'draft_n', 'draft_n_accepted',
                      'prefix_lookup_ms', 'prefix_load_ms', 'session_lookup_ms', 'session_load_ms',
                      # CACHE-PERFECT ECHO (engine echo.rs): counts and the C2 first divergent offset
                      'echo_blocks', 'echo_n', 'echo_fixed', 'echo_tokens', 'roundtrip_miss',
                      # CACHE-PERFECT P2b: restore cost and a preloaded state's load time
                      'resume_ms', 'preload_ms'):
                if _num(t.get(k)) is not None:
                    out[k] = t[k]
            # Native reasons are bounded enums; never accept paths or raw backend errors.
            try:
                from .engine_ledger import ENUMS
            except ImportError:
                from engine_ledger import ENUMS
            for key in ('cache_src', 'prefix_reason', 'session_reason', 'resume_src', 'preload_src'):
                if isinstance(t.get(key), str) and t[key] in ENUMS[key]:
                    out[key] = t[key]
            if isinstance(t.get('echo_miss_class'), str) and t['echo_miss_class'] in ECHO_MISS_CLASSES:
                out['echo_miss_class'] = t['echo_miss_class']
        except (ValueError, AttributeError):
            pass
    f = _FINISH.findall(tail)
    if f:
        out['finish_reason'] = f[-1].decode()
    return out


_SESSION_HASH = re.compile(r'[0-9a-f]{16}')


def last_session(model_alias, tail_bytes=262144):
    """The newest ledger row's session hash for `model_alias` (today's file, then yesterday's), or
    None. Reads only the files' tails; used by the state preload after an explicit load."""
    try:
        paths = sorted(ledger_dir().glob('requests-*.jsonl'), reverse=True)[:2]
    except OSError:
        return None
    for path in paths:
        try:
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
            with os.fdopen(fd, 'rb') as f:
                size = os.fstat(f.fileno()).st_size
                f.seek(max(0, size - tail_bytes))
                lines = f.read(tail_bytes).splitlines()
        except OSError:
            continue
        for line in reversed(lines):
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if (isinstance(row, dict) and row.get('model') == model_alias and isinstance(row.get('session'), str)
                    and _SESSION_HASH.fullmatch(row['session'])):
                return row['session']
    return None


def retention_days():
    """LOCAL_AI_LEDGER_DAYS as a finite positive number of days; anything else is the default 14."""
    try:
        value = float(os.environ.get('LOCAL_AI_LEDGER_DAYS', '14'))
    except ValueError:
        return 14.0
    return value if math.isfinite(value) and value > 0 else 14.0


def write(row):
    if not enabled():
        return
    try:
        d = ledger_dir()
        day = time.strftime('%Y%m%d')
        with _lock:
            d.mkdir(parents=True, exist_ok=True)
            os.chmod(d, 0o700)
            if _pruned_day[0] != day:
                _pruned_day[0] = day
                keep = retention_days() * 86400
                for p in d.glob('requests-*.jsonl'):
                    try:
                        if time.time() - p.stat().st_mtime > keep:
                            p.unlink()
                    except OSError:
                        pass
            fd = os.open(d / f'requests-{day}.jsonl', os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, 'a') as f:
                f.write(json.dumps(row, separators=(',', ':')) + '\n')
    except OSError:
        pass


MEDIAN_KEYS = ('queue_wait_ms', 'admit_ms', 'ttfb_ms', 'wall_ms')


def _median(values):
    values = sorted(values)
    if not values:
        return None
    mid = len(values) // 2
    return values[mid] if len(values) % 2 else round((values[mid - 1] + values[mid]) / 2, 3)


def recent(limit=50, tail_bytes=1024 * 1024):
    """Newest rows first (today's file, then yesterday's), and medians over the answered
    ones. For `GET /_local/ledger`; the rows are what `write()` stored: numbers and labels."""
    limit = min(200, max(1, limit if type(limit) is int else 50))
    directory = ledger_dir()
    if not directory.is_dir():
        return {'object': 'list', 'available': False, 'rows': [], 'medians': {}}
    rows = []
    for stamp in (time.strftime('%Y%m%d'), time.strftime('%Y%m%d', time.localtime(time.time() - 86400))):
        try:
            with open(directory / f'requests-{stamp}.jsonl', 'rb') as stream:
                size = stream.seek(0, os.SEEK_END)
                stream.seek(max(0, size - tail_bytes))
                lines = stream.read().splitlines()
                if size > tail_bytes:
                    lines = lines[1:]  # the first line may be cut
        except OSError:
            continue
        for line in reversed(lines):
            if len(rows) >= limit:
                break
            try:
                row = json.loads(line)
            except (ValueError, RecursionError):
                continue
            if isinstance(row, dict):
                rows.append(row)
        if len(rows) >= limit:
            break
    answered = [r for r in rows if r.get('status') == 200]
    medians = {key: _median([r[key] for r in answered if _num(r.get(key)) is not None]) for key in MEDIAN_KEYS}
    for name, count, ms in (('prompt_tokens_per_second', 'prompt_n', 'prompt_ms'),
                            ('predicted_tokens_per_second', 'predicted_n', 'predicted_ms')):
        medians[name] = _median([round(r[count] * 1000 / r[ms], 2) for r in answered
                                 if _num(r.get(count)) and _num(r.get(ms)) and r[ms] > 0 and r[count] >= 8])
    return {'object': 'list', 'available': True, 'rows': rows, 'medians': medians, 'answered': len(answered)}
