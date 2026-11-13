#!/usr/bin/env python3
"""v0.1.3 TOOL-OUTPUT CAP: large tool results keep head, tail and error lines; the rest is one
exact 'read more' call away. Behind LOCAL_AI_TOOL_OUTPUT_CAP=on; default off (byte-identical
passthrough).

Why: tool results are 44% of appended tokens, results over ~2K tokens hold 47-80% of the result
tokens, and every token appended at depth costs prefill (APPEND-SOURCES-ANALYSIS.md).

Where: the controller applies `cap_request` to a local (never a LAN-shared) chat request before
any engine shaping. OpenCode 2.2.7 has no plugin hook that can rewrite a tool result (its
`tool.before/after` hooks only observe Code Mode calls), and its own `tool_output` limit keeps only
the head (or only the tail for shell) of results over 2000 lines / 50 KB. The read-more tool is this
file run as a stdio MCP server (`python3 tool_output_cap.py mcp`), which the OpenCode launch overlay
registers with `codemode: false` so it is a direct tool; OpenCode names it `<server>_read_tool_output`.

Rules (each one is a test in tests/test_tool_output_cap.py):
- Never truncate when the request offers no read-more tool: the marker must name a tool the agent
  really has. The tool name is taken from the request's own `tools`.
- Only the tool results after the last assistant message (new ones) are candidates. An older one is
  rewritten only when this store already holds its render, which is then reused byte for byte, so
  the prompt prefix the engine has cached never changes; history first seen uncapped stays uncapped.
- Per-tool policy: on for shell/search/list tools, off for read/edit/write and everything unknown;
  never for a `read` with an explicit offset/limit, a call with `no_truncate`, the read-more tool's
  own results, or a result under the threshold.
- The full output is stored before the request is rewritten (0700 dir, 0600 files, atomic write,
  TTL and size cap, LRU by use); any store error leaves the result untouched (fail open).
- Same output + same settings -> same bytes (no clock, no session, no randomness in the render).
"""
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time

ENV = 'LOCAL_AI_TOOL_OUTPUT_CAP'
SERVER_NAME = 'local_ai'
TOOL = 'read_tool_output'
DEFAULTS = {'tokens': 2000, 'head': 40, 'tail': 60, 'errors': 20, 'line_chars': 400,
            'ttl_secs': 7 * 86400, 'budget_bytes': 256 * 1024 * 1024, 'read_limit': 200}
# Tools capped by default (OpenCode 2.2.7 names and common aliases); everything else is off.
CAP_TOOLS = frozenset({'bash', 'shell', 'exec', 'exec_command', 'run', 'terminal', 'grep', 'rg', 'ripgrep',
                       'glob', 'find', 'list', 'ls', 'codesearch', 'search'})
NEVER_TOOLS = frozenset({'read', 'edit', 'write', 'patch', 'apply_patch', 'multiedit', 'todowrite', 'todoread',
                         'task', 'subagent', 'question', 'skill'})
ERROR_LINE = re.compile(r'error|fail|fatal|panic|traceback|exception|warning|warn\b|assert|denied|'
                        r'not found|undefined|segmentation|abort|exit (?:code|status)|\bE\d{3,4}\b', re.I)
ID_RE = re.compile(r'^[0-9a-f]{12,64}$')
SESSION_HEADERS = ('x-opencode-session-id', 'x-opencode-session', 'x-session-id', 'x-session-affinity')


def _int_env(environ, name, default, low, high):
    try:
        value = int(environ.get(name, default))
    except (TypeError, ValueError):
        return default
    return value if low <= value <= high else default


def settings(environ=None):
    """The cap's settings from the environment; `enabled` is False unless the flag is exactly on."""
    environ = os.environ if environ is None else environ
    flag = str(environ.get(ENV, 'off')).strip().lower()
    state = Path(environ.get('XDG_STATE_HOME') or Path.home() / '.local/state')
    policy = {}
    for item in str(environ.get(ENV + '_TOOLS', '')).split(','):
        name, _, value = item.strip().partition('=')
        if name and value in ('on', 'off'):
            policy[name.strip().lower()] = value == 'on'
    return {
        'enabled': flag in ('on', '1', 'true'),
        'tokens': _int_env(environ, ENV + '_TOKENS', DEFAULTS['tokens'], 256, 1_000_000),
        'head': _int_env(environ, ENV + '_HEAD_LINES', DEFAULTS['head'], 1, 10_000),
        'tail': _int_env(environ, ENV + '_TAIL_LINES', DEFAULTS['tail'], 1, 10_000),
        'errors': _int_env(environ, ENV + '_ERROR_LINES', DEFAULTS['errors'], 0, 1_000),
        'line_chars': DEFAULTS['line_chars'],
        'ttl_secs': _int_env(environ, ENV + '_TTL_SECS', DEFAULTS['ttl_secs'], 60, 10 * 365 * 86400),
        'budget_bytes': _int_env(environ, ENV + '_BUDGET_BYTES', DEFAULTS['budget_bytes'], 1 << 20, 1 << 40),
        'store': Path(environ.get(ENV + '_STORE') or state / 'local-ai/tool-output'),
        'policy': policy,
    }


def estimate_tokens(text):
    """Conservative: ~3 UTF-8 bytes per token (chars/4 undercounts code and logs by 2-3x)."""
    return -(-len(text.encode('utf-8')) // 3)


def tool_enabled(name, cfg):
    name = (name or '').lower()
    if name.endswith(TOOL):
        return False  # the read-more tool's own results: never (no loops, the agent asked for them)
    if name in cfg['policy']:
        return cfg['policy'][name]
    if name in NEVER_TOOLS:
        return False
    return name in CAP_TOOLS


def _explicit_full_read(name, arguments):
    if not isinstance(arguments, dict):
        return False
    if arguments.get('no_truncate') or arguments.get('noTruncate'):
        return True
    return (name or '').lower() == 'read' and any(k in arguments for k in ('offset', 'limit', 'range'))


def _clip(line, number, cap):
    if len(line) <= cap:
        return line
    return '%s<<line %d: +%d chars; read it with offset=%d limit=1>>' % (line[:cap], number, len(line) - cap, number)


def render(text, digest, tool_name, cfg):
    """The capped text for `text` (deterministic: depends on the text, the tool name and cfg only)."""
    lines = text.split('\n')
    total = len(lines)
    head, tail, cap = cfg['head'], cfg['tail'], cfg['line_chars']
    budget = cfg['tokens'] * 3  # bytes of kept content, ~the threshold
    # shrink head/tail (keep their 2:3 ratio) until the kept lines fit the budget
    while head + tail > 2:
        kept = lines[:head] + lines[max(head, total - tail):]
        if sum(min(len(x), cap) + 1 for x in kept) <= budget:
            break
        head, tail = max(1, head * 3 // 4), max(1, tail * 3 // 4)
    if head + tail >= total:
        head, tail = total, 0
    first_hidden, last_hidden = head + 1, total - tail
    errors = []
    if cfg['errors'] and last_hidden >= first_hidden:
        for number in range(first_hidden, last_hidden + 1):
            if ERROR_LINE.search(lines[number - 1]):
                errors.append(number)
                if len(errors) >= cfg['errors']:
                    break
    hidden_lines = last_hidden - first_hidden + 1
    hidden_text = '\n'.join(lines[first_hidden - 1:last_hidden])
    ident = digest[:12]
    out = [_clip(lines[i], i + 1, cap) for i in range(head)]
    if hidden_lines <= 0:  # few, very long lines: nothing hidden by range, the long lines are clipped
        out.append('<<truncated: long lines clipped (~%d tokens hidden). Read more with the %s tool: id=%s '
                   'offset=1 limit=%d, or all=true. Do not re-run the command.>>'
                   % (estimate_tokens(text) - estimate_tokens('\n'.join(out)), tool_name, ident, total))
        return '\n'.join(out)
    out.append('<<truncated: lines %d-%d of %d hidden (%d lines, ~%d tokens)%s. Read more with the %s tool: '
               'id=%s offset=%d limit=%d, or all=true for the whole output. Do not re-run the command.>>'
               % (first_hidden, last_hidden, total, hidden_lines, estimate_tokens(hidden_text),
                  '; %d error/warning lines from them follow' % len(errors) if errors else '',
                  tool_name, ident, first_hidden, min(DEFAULTS['read_limit'], hidden_lines)))
    for number in errors:
        out.append('L%d: %s' % (number, _clip(lines[number - 1], number, cap)))
    if errors:
        out.append('<<last %d lines:>>' % tail)
    out.extend(_clip(lines[i], i + 1, cap) for i in range(total - tail, total))
    return '\n'.join(out)


# ---------------------------------------------------------------------------------------------- store

class Store:
    """Per session: <root>/<session16>/<sha256 of output>-<call16>.raw (exact bytes) + .cap (render).

    The key includes the tool call id, so only the very same tool message reuses a render (prefix
    stability); the same output from another call (say one with `no_truncate`) is judged afresh."""

    def __init__(self, root, ttl_secs, budget_bytes):
        self.root, self.ttl, self.budget = Path(root), ttl_secs, budget_bytes

    def _session_dir(self, session):
        return self.root / hashlib.sha256((session or 'default').encode()).hexdigest()[:16]

    def lookup(self, session, key):
        path = self._session_dir(session) / (key + '.cap')
        raw = path.with_suffix('.raw')
        try:
            if not raw.is_file():
                return None
            rendered = path.read_text(encoding='utf-8')
            now = time.time()
            os.utime(path, (now, now)); os.utime(raw, (now, now))  # LRU: in use, keep it
            return rendered
        except OSError:
            return None

    def _write(self, path, data):
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix='.tmp-')
        try:
            with os.fdopen(fd, 'wb') as stream:
                stream.write(data); stream.flush(); os.fsync(stream.fileno())
            os.chmod(tmp, 0o600)
            os.replace(tmp, path)
        except BaseException:
            try: os.unlink(tmp)
            except OSError: pass
            raise

    def put(self, session, key, text, rendered):
        old = os.umask(0o077)
        try:
            self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
            os.chmod(self.root, 0o700)
            folder = self._session_dir(session)
            folder.mkdir(exist_ok=True, mode=0o700)
            os.chmod(folder, 0o700)
            raw = folder / (key + '.raw')
            self._write(raw, text.encode('utf-8'))           # the exact output first ...
            self._write(raw.with_suffix('.cap'), rendered.encode('utf-8'))  # ... its render last
        finally:
            os.umask(old)

    def gc(self, now=None):
        """Drop entries older than the TTL, then the least recently used until under the budget."""
        now = time.time() if now is None else now
        entries = []
        try:
            folders = [p for p in self.root.iterdir() if p.is_dir()]
        except OSError:
            return 0
        removed = 0
        for folder in folders:
            for raw in folder.glob('*.raw'):
                try:
                    st = raw.stat()
                    cap = raw.with_suffix('.cap')
                    size = st.st_size + (cap.stat().st_size if cap.exists() else 0)
                except OSError:
                    continue
                if now - st.st_mtime > self.ttl:
                    removed += self._drop(raw)
                else:
                    entries.append((st.st_mtime, size, raw))
            for tmp in folder.glob('.tmp-*'):
                try:
                    if now - tmp.stat().st_mtime > 3600: tmp.unlink()
                except OSError:
                    pass
        used = sum(size for _, size, _ in entries)
        for _, size, raw in sorted(entries):
            if used <= self.budget:
                break
            removed += self._drop(raw); used -= size
        return removed

    @staticmethod
    def _drop(raw):
        for path in (raw.with_suffix('.cap'), raw):  # render first: a .cap never outlives its .raw
            try: path.unlink()
            except FileNotFoundError: pass
            except OSError: return 0
        return 1

    def find(self, ident):
        """The exact stored output for a marker id (any session of this user), or None."""
        if not isinstance(ident, str) or not ID_RE.match(ident):
            return None
        try:
            for folder in sorted(self.root.iterdir()):
                for raw in sorted(folder.glob(ident + '*.raw')):
                    return raw.read_bytes().decode('utf-8')
        except OSError:
            return None
        return None


def read_more(store, ident, offset=None, limit=None, all_lines=False):
    """(exact text, footer or None) for lines offset..offset+limit-1 (1-based), or the whole output."""
    text = store.find(ident)
    if text is None:
        raise KeyError('No stored tool output with id %s (expired or never stored); re-run the command.' % ident)
    if all_lines:
        return text, None
    lines = text.split('\n')
    total = len(lines)
    offset = 1 if offset is None else int(offset)
    limit = DEFAULTS['read_limit'] if limit is None else int(limit)
    if offset < 1 or limit < 1:
        raise ValueError('offset and limit must be >= 1.')
    if offset > total:
        raise ValueError('offset %d is past the end; the output has %d lines.' % (offset, total))
    end = min(total, offset + limit - 1)
    footer = '<<lines %d-%d of %d%s>>' % (offset, end, total,
                                           '; next: offset=%d' % (end + 1) if end < total else '; end of output')
    return '\n'.join(lines[offset - 1:end]), footer


# ------------------------------------------------------------------------------------ request transform

def _read_tool_name(payload):
    for tool in payload.get('tools') or ():
        name = (tool.get('function') or {}).get('name') if isinstance(tool, dict) else None
        if isinstance(name, str) and (name == TOOL or name.endswith('_' + TOOL)):
            return name
    return None


def _calls_by_id(messages):
    calls = {}
    for message in messages:
        if isinstance(message, dict) and message.get('role') == 'assistant':
            for call in message.get('tool_calls') or ():
                if not isinstance(call, dict):
                    continue
                function = call.get('function') or {}
                arguments = function.get('arguments')
                if isinstance(arguments, str):
                    try: arguments = json.loads(arguments)
                    except ValueError: arguments = None
                calls[call.get('id')] = (function.get('name'), arguments)
    return calls


def session_of(headers):
    if headers is None:
        return 'default'
    for name in SESSION_HEADERS:
        value = headers.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()[:256]
    return 'default'


def cap_request(payload, headers=None, cfg=None, store=None):
    """(payload, stats). The payload is the same object when nothing changed (flag off included)."""
    cfg = settings() if cfg is None else cfg
    stats = {'capped': 0, 'reused': 0, 'hidden_tokens': 0, 'reads': 0}
    if not cfg['enabled'] or not isinstance(payload, dict):
        return payload, stats
    messages = payload.get('messages')
    if not isinstance(messages, list):
        return payload, stats
    reader = _read_tool_name(payload)
    if reader is None:
        return payload, stats  # no read-more tool on offer: a marker would strand the agent
    store = Store(cfg['store'], cfg['ttl_secs'], cfg['budget_bytes']) if store is None else store
    session = session_of(headers)
    last_assistant = max((i for i, m in enumerate(messages) if isinstance(m, dict) and m.get('role') == 'assistant'),
                         default=-1)
    calls = _calls_by_id(messages)
    out = None
    wrote = False
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or message.get('role') != 'tool':
            continue
        name, arguments = calls.get(message.get('tool_call_id'), (None, None))
        if index > last_assistant and (name or '').endswith(TOOL):
            stats['reads'] += 1
        content = message.get('content')
        if not isinstance(content, str) or not content:
            continue
        digest = hashlib.sha256(content.encode('utf-8')).hexdigest()
        key = digest + '-' + hashlib.sha256(str(message.get('tool_call_id') or '').encode()).hexdigest()[:16]
        rendered = store.lookup(session, key)
        if rendered is not None:
            stats['reused'] += 1
        elif (index > last_assistant and tool_enabled(name, cfg) and not _explicit_full_read(name, arguments)
              and estimate_tokens(content) > cfg['tokens']):
            rendered = render(content, digest, reader, cfg)
            if len(rendered) >= len(content):
                continue
            try:
                store.put(session, key, content, rendered)
            except OSError:
                continue  # fail open: never hide what could not be stored
            wrote = True
            stats['capped'] += 1
            stats['hidden_tokens'] += estimate_tokens(content) - estimate_tokens(rendered)
        else:
            continue
        if out is None:
            out = list(messages)
        out[index] = {**message, 'content': rendered}
    if wrote:
        try: store.gc()
        except OSError: pass
    if out is None:
        return payload, stats
    return {**payload, 'messages': out}, stats


# ------------------------------------------------------------------------------- MCP stdio server

TOOL_SPEC = {
    'name': TOOL,
    'description': ('Read the hidden part of a truncated tool output exactly, without re-running the command. '
                    'Use the id from the <<truncated ...>> marker; give offset (1-based line) and limit, '
                    'or all=true for the whole output.'),
    'inputSchema': {'type': 'object', 'properties': {
        'id': {'type': 'string', 'description': 'The id from the truncation marker.'},
        'offset': {'type': 'integer', 'minimum': 1, 'description': 'First line to return (1-based).'},
        'limit': {'type': 'integer', 'minimum': 1, 'description': 'Number of lines (default 200).'},
        'all': {'type': 'boolean', 'description': 'Return the whole output.'}},
        'required': ['id']},
}


def handle(message, store):
    """One JSON-RPC message -> reply dict or None (notifications)."""
    method, ident = message.get('method'), message.get('id')
    if ident is None:
        return None
    if method == 'initialize':
        version = (message.get('params') or {}).get('protocolVersion') or '2025-06-18'
        result = {'protocolVersion': version, 'capabilities': {'tools': {}},
                  'serverInfo': {'name': SERVER_NAME, 'version': '0.1.3'}}
    elif method == 'ping':
        result = {}
    elif method == 'tools/list':
        result = {'tools': [TOOL_SPEC]}
    elif method == 'tools/call':
        params = message.get('params') or {}
        args = params.get('arguments') or {}
        if params.get('name') != TOOL:
            return {'jsonrpc': '2.0', 'id': ident, 'error': {'code': -32602, 'message': 'Unknown tool.'}}
        try:
            text, footer = read_more(store, args.get('id'), args.get('offset'), args.get('limit'),
                                     bool(args.get('all')))
            content = [{'type': 'text', 'text': text}] + ([{'type': 'text', 'text': footer}] if footer else [])
            result = {'content': content, 'isError': False}
        except (KeyError, ValueError, TypeError) as error:
            result = {'content': [{'type': 'text', 'text': str(error).strip('\'"')}], 'isError': True}
    else:
        return {'jsonrpc': '2.0', 'id': ident, 'error': {'code': -32601, 'message': 'Method not found.'}}
    return {'jsonrpc': '2.0', 'id': ident, 'result': result}


def serve(stdin=None, stdout=None, environ=None):
    stdin, stdout = stdin or sys.stdin, stdout or sys.stdout
    cfg = settings(environ)
    store = Store(cfg['store'], cfg['ttl_secs'], cfg['budget_bytes'])
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except ValueError:
            reply = {'jsonrpc': '2.0', 'id': None, 'error': {'code': -32700, 'message': 'Parse error.'}}
        else:
            reply = handle(message, store) if isinstance(message, dict) else None
        if reply is not None:
            stdout.write(json.dumps(reply, separators=(',', ':')) + '\n'); stdout.flush()


def mcp_overlay_entry(script, environ=None):
    """The OpenCode 2.2.7 `mcp.servers` entry for the read-more server (direct tool, not Code Mode)."""
    environ = os.environ if environ is None else environ
    env = {k: environ[k] for k in (ENV + '_STORE', 'XDG_STATE_HOME') if environ.get(k)}
    entry = {'type': 'local', 'command': [sys.executable or 'python3', str(script), 'mcp'], 'codemode': False}
    if env:
        entry['environment'] = env
    return entry


if __name__ == '__main__':
    if sys.argv[1:] == ['mcp']:
        serve()
    else:
        sys.stderr.write('usage: tool_output_cap.py mcp\n'); sys.exit(2)
