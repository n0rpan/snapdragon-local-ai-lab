"""Which harness prefixes (system prompt + tools) the local engine has seen, and primers to
re-warm them.

The Rust engine keeps one persistent prefix-store entry per distinct system-prompt-plus-
tools token sequence. This registry is the controller's view of the same thing, so a UI
can show "pi: warm" and offer "prepare dsh" without loading a model to ask:

- `observe()` at request start: the harness name, a key over the leading system messages,
  tools and template kwargs, and sizes. Nothing else of the request is kept.
- `complete()` when the reply ended with HTTP 200: the reused / evaluated token counts
  (`timings.cache_n` / `prompt_n`). After a successful request the engine holds this
  prefix, so the entry is `warm` for the engine identity that served it.
- `primer()`: a 1-token request with the same system messages and tools and a fixed user
  line. It makes the engine build (or load) and store that prefix. It holds no user turn.

Disk: `index.json` (metadata only) and `primers/<key>.json` (the system prompt and tool
schemas, the same text whose token ids the engine's prefix store holds), files 0600 in a
0700 directory. At most `max_entries` prefixes, least recently used dropped first.

Unknown clients (harness `other`) are different: nothing of theirs is written to disk unless
learning is on (runtime.json `harness_learning.enabled`, off by default) or that one client
was approved (`approve()`). Until then the registry only counts them in memory (`candidates`:
sizes, counts and a fingerprint, no text). A client is learned once the same system prompt
and tool list arrived `min_requests` times, byte for byte; a prompt that changes on every
request is never learned, and the candidate row says so (`stable: false`). A learned client
gets the name `learned-<8 hex>` (a hash of the first 256 characters of its system prompt and
its tool names), a label from its User-Agent that the user can change, a primer under the
same rule as above (system prompt and tool schemas only, 0600), and at most
`LEARNED_PREFIXES` prefixes.

A prefix is `warm` only while the engine identity (model + server binary + context size)
matches the one that served it and the store was not cleared since; the engine can still
have evicted it (LRU), so `warm` is "expected warm": the next request's `cache_n` is the
proof.
"""
import collections
import hashlib
import json
import os
from pathlib import Path
import re
import time

SCHEMA = 1
TEMPLATE_KEYS = ('chat_template_kwargs', 'parallel_tool_calls')
PRIMER_USER = 'Initialize this coding prefix. Reply READY. Do not call tools.'
MAX_PRIMER_BYTES = 2 * 1024**2
LEARNED = 'learned-'
LEARNED_PREFIXES = 4      # prefixes kept per learned client (one per project directory, say)
MAX_CANDIDATES = 16
CANDIDATE_KEYS = 8
LEARNING_DEFAULTS = {'enabled': False, 'min_requests': 3, 'min_prefix_chars': 3000}
_LABEL = re.compile(r'[^A-Za-z0-9 ._+-]')
_TIMINGS = re.compile(rb'"(cache_n|prompt_n)"\s*:\s*(\d+)')


def leading_system(messages):
    out = []
    for message in messages if isinstance(messages, list) else []:
        if not isinstance(message, dict) or message.get('role') not in ('system', 'developer'):
            break
        out.append(message)
    return out


def prefix_parts(payload):
    """(systems, tools, template kwargs) of a chat body, or None when it has no system prompt."""
    if not isinstance(payload, dict):
        return None
    systems = leading_system(payload.get('messages'))
    if not systems:
        return None
    tools = payload.get('tools') if isinstance(payload.get('tools'), list) else []
    template = {key: payload[key] for key in TEMPLATE_KEYS if key in payload}
    return systems, tools, template


def prefix_key(payload):
    parts = prefix_parts(payload)
    if parts is None:
        return None
    try:
        # order-preserving: the chat template renders tools and keys in the order given
        encoded = json.dumps(parts, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    except (TypeError, ValueError):
        return None
    return hashlib.sha256(encoded.encode()).hexdigest()


def learning_settings(config):
    """`harness_learning` of runtime.json (or of a resolved profile), with defaults."""
    raw = config.get('harness_learning') if isinstance(config, dict) else None
    raw = raw if isinstance(raw, dict) else {}
    out = dict(LEARNING_DEFAULTS)
    out['enabled'] = raw.get('enabled') is True
    for key, low, high in (('min_requests', 1, 100), ('min_prefix_chars', 0, 10**6)):
        if type(raw.get(key)) is int and low <= raw[key] <= high:
            out[key] = raw[key]
    return out


def _system_text(systems):
    parts = []
    for message in systems:
        content = message.get('content')
        if isinstance(content, list):
            content = ''.join(p.get('text', '') for p in content if isinstance(p, dict) and isinstance(p.get('text'), str))
        parts.append(content if isinstance(content, str) else '')
    return '\n'.join(parts)


def learned_id(systems, tools):
    """Stable name for an unknown client: the start of its system prompt and its tool names.
    A project path or a date further down the prompt does not change it."""
    names = sorted(str(t['function'].get('name')) for t in tools
                   if isinstance(t, dict) and isinstance(t.get('function'), dict))
    head = _system_text(systems).lstrip()[:256]
    return LEARNED + hashlib.sha256(json.dumps([head, names], ensure_ascii=False).encode()).hexdigest()[:8]


def clean_label(text, limit=40):
    """A display label: printable ASCII subset, bounded. '' when nothing usable is left."""
    if not isinstance(text, str):
        return ''
    return _LABEL.sub('', text).strip()[:limit]


def agent_label(user_agent):
    """Product name of a User-Agent ('OpenAI/Python 1.2' -> 'OpenAI/Python' is cut to 'OpenAI')."""
    if not isinstance(user_agent, str):
        return ''
    return clean_label(user_agent.strip().split('/', 1)[0].split(' ', 1)[0], 24)


def engine_identity(config):
    """What must stay the same for a stored prefix to be usable (mirrors the engine's rule)."""
    if not isinstance(config, dict):
        return ''
    return '%s:%s:%s' % (config.get('sha256', ''), config.get('server_sha256', ''), config.get('context_size', ''))


def timings_from_tail(tail):
    """`cache_n` / `prompt_n` from the last bytes of a reply (JSON or SSE). Missing -> None."""
    found = {}
    for key, value in _TIMINGS.findall(tail or b''):
        found[key.decode()] = int(value)
    return found.get('cache_n'), found.get('prompt_n')


class HarnessRegistry:
    def __init__(self, directory=None, max_entries=24, clock=time.time):
        self.directory = Path(directory) if directory is not None else None
        self.max_entries = max_entries
        self.clock = clock
        self.entries = {}
        self.approved, self.ignored, self.labels = set(), set(), {}
        self.candidates = collections.OrderedDict()  # memory only: unknown clients, no text
        self._load()

    # ------------------------------------------------------------ persistence
    def _load(self):
        if self.directory is None:
            return
        try:
            value = json.loads((self.directory / 'index.json').read_text())
            if value.get('schema') == SCHEMA and isinstance(value.get('entries'), dict):
                self.entries = {k: v for k, v in value['entries'].items()
                                if re.fullmatch('[0-9a-f]{64}', k) and isinstance(v, dict)}
                # unknown clients are never kept without the opt-in (older index files had them)
                for key in [k for k, v in self.entries.items() if v.get('harness') == 'other']:
                    self._drop(key)
                name = re.compile(LEARNED + '[0-9a-f]{8}')
                self.approved = {x for x in value.get('approved', []) if isinstance(x, str) and name.fullmatch(x)}
                self.ignored = {x for x in value.get('ignored', []) if isinstance(x, str) and name.fullmatch(x)}
                labels = value.get('labels') if isinstance(value.get('labels'), dict) else {}
                self.labels = {k: clean_label(v) for k, v in labels.items()
                               if isinstance(k, str) and name.fullmatch(k) and clean_label(v)}
        except (OSError, ValueError, AttributeError, TypeError):
            self.entries = {}

    def _write(self, path, value):
        tmp = path.with_name(path.name + '.pending')
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, ensure_ascii=False)
        os.replace(tmp, path)

    def _save(self):
        if self.directory is None:
            return
        try:
            (self.directory / 'primers').mkdir(mode=0o700, parents=True, exist_ok=True)
            os.chmod(self.directory, 0o700)
            self._write(self.directory / 'index.json', {
                'schema': SCHEMA, 'entries': self.entries, 'approved': sorted(self.approved),
                'ignored': sorted(self.ignored), 'labels': self.labels})
        except OSError:
            pass  # the registry is advisory: never fail a request over it

    def _primer_path(self, key):
        return self.directory / 'primers' / (key + '.json')

    # ------------------------------------------------------------ recording
    def _candidate(self, name, key, now, sizes, user_agent):
        row = self.candidates.pop(name, None)
        if row is None:
            row = {'first_seen': now, 'requests': 0, 'keys': collections.OrderedDict(),
                   'last_cache_n': None, 'last_prompt_n': None}
        row['keys'][key] = row['keys'].pop(key, 0) + 1
        while len(row['keys']) > CANDIDATE_KEYS:
            row['keys'].popitem(last=False)
        row.update(sizes, last_used=now, requests=row['requests'] + 1, last_key=key)
        if agent_label(user_agent):
            row['agent'] = agent_label(user_agent)
        self.candidates[name] = row
        while len(self.candidates) > MAX_CANDIDATES:
            self.candidates.popitem(last=False)
        return row

    def observe(self, payload, harness, config, learning=None, user_agent=None):
        """Note a chat request before it runs; returns its prefix key (None: no system prompt).

        `learning`: the `harness_learning` settings (see `learning_settings`); None is off.
        An unknown client (`harness == 'other'`) is only counted in memory until it is
        learned; its key is still returned so `complete()` can note the reuse numbers."""
        key = prefix_key(payload)
        if key is None:
            return None
        systems, tools, template = prefix_parts(payload)
        now = self.clock()
        system_chars = sum(len(m['content']) if isinstance(m.get('content'), str) else len(json.dumps(m.get('content')))
                           for m in systems)
        sizes = {'system_chars': system_chars, 'tools': len(tools),
                 'tools_chars': len(json.dumps(tools, ensure_ascii=False)) if tools else 0}
        entry = self.entries.get(key)
        fresh = entry is None
        if entry is not None and str(entry.get('harness', '')).startswith(LEARNED):
            harness = entry['harness']  # a learned client keeps its name
        elif entry is not None and harness == 'other':
            harness = entry.get('harness', 'other')  # the same prefix as a known harness: its row stays
        elif harness == 'other':
            name = learned_id(systems, tools)
            settings = learning if isinstance(learning, dict) else LEARNING_DEFAULTS
            row = self._candidate(name, key, now, sizes, user_agent)
            known = name in self.approved or any(e.get('harness') == name for e in self.entries.values())
            allowed = name not in self.ignored and (known or settings.get('enabled') is True)
            needed = 1 if known else settings.get('min_requests', LEARNING_DEFAULTS['min_requests'])
            large = system_chars + sizes['tools_chars'] >= settings.get('min_prefix_chars', 0)
            if not (allowed and large and row['keys'][key] >= needed):
                return key
            del self.candidates[name]
            self.approved.discard(name)
            if row.get('agent') and name not in self.labels:
                self.labels[name] = row['agent']
            harness = name
            entry = {'first_seen': row['first_seen'], 'requests': row['keys'][key] - 1, 'warm_identity': None,
                     'learned': True}
        if entry is None:
            entry = {'first_seen': now, 'requests': 0, 'warm_identity': None}
        entry.update({'harness': harness, 'model': config.get('model_alias') if isinstance(config, dict) else None,
                      **sizes, 'last_used': now, 'requests': entry.get('requests', 0) + 1})
        self.entries[key] = entry
        if fresh and self.directory is not None:
            primer = {'messages': systems + [{'role': 'user', 'content': PRIMER_USER}], 'tools': tools, **template}
            try:
                text = json.dumps(primer, ensure_ascii=False, allow_nan=False)
                if len(text.encode()) <= MAX_PRIMER_BYTES:
                    (self.directory / 'primers').mkdir(mode=0o700, parents=True, exist_ok=True)
                    self._write(self._primer_path(key), primer)
            except (OSError, TypeError, ValueError):
                pass
        self._evict(harness)
        self._save()
        return key

    def complete(self, key, config, cache_n=None, prompt_n=None):
        """The request for `key` returned 200: the engine now holds this prefix."""
        entry = self.entries.get(key)
        if entry is None:
            for row in self.candidates.values():
                if row.get('last_key') == key:
                    row['last_cache_n'], row['last_prompt_n'] = cache_n, prompt_n
            return
        entry['warm_identity'] = engine_identity(config)
        entry['last_cache_n'], entry['last_prompt_n'] = cache_n, prompt_n
        entry['last_ok'] = self.clock()
        self._save()

    def cleared(self):
        """The prefix store was cleared: nothing is warm, the primers stay."""
        for entry in self.entries.values():
            entry['warm_identity'] = None
        self._save()

    def forget(self, harness=None, ignore=False):
        """Drop entries and primers (all, or one harness). `ignore`: also never learn this
        unknown client again (until `approve()`)."""
        for key in [k for k, v in self.entries.items() if harness is None or v.get('harness') == harness]:
            self._drop(key)
        if harness is None:
            self.candidates.clear(); self.approved.clear(); self.labels.clear()
            if not ignore:
                self.ignored.clear()
        else:
            self.candidates.pop(harness, None)
            self.approved.discard(harness)
            self.labels.pop(harness, None)
            if ignore and re.fullmatch(LEARNED + '[0-9a-f]{8}', harness):
                self.ignored.add(harness)
        self._save()

    def approve(self, harness):
        """Opt in for one unknown client: it is learned at its next request. Returns False
        for a name that is not a known candidate or learned client."""
        if not isinstance(harness, str) or not re.fullmatch(LEARNED + '[0-9a-f]{8}', harness):
            return False
        if harness not in self.candidates and harness not in self.ignored:
            return any(e.get('harness') == harness for e in self.entries.values())
        self.ignored.discard(harness)
        self.approved.add(harness)
        self._save()
        return True

    def set_label(self, harness, label):
        label = clean_label(label)
        if not label or not any(e.get('harness') == harness for e in self.entries.values()) \
                or not str(harness).startswith(LEARNED):
            return False
        self.labels[harness] = label
        self._save()
        return True

    def _drop(self, key):
        del self.entries[key]
        if self.directory is not None:
            try: self._primer_path(key).unlink(missing_ok=True)
            except OSError: pass

    def _evict(self, harness=None):
        if isinstance(harness, str) and harness.startswith(LEARNED):
            own = sorted((k for k, v in self.entries.items() if v.get('harness') == harness),
                         key=lambda k: self.entries[k].get('last_used', 0))
            for key in own[:-LEARNED_PREFIXES]:
                self._drop(key)
        while len(self.entries) > self.max_entries:
            self._drop(min(self.entries, key=lambda k: self.entries[k].get('last_used', 0)))

    # ------------------------------------------------------------ queries
    def harness_names(self, config=None, cold_only=False):
        """Harnesses with a stored primer, most recently used first; `cold_only`: those whose
        newest prefix is not expected warm for `config`."""
        identity = engine_identity(config) if config else None
        newest = {}
        for key, entry in self.entries.items():
            name = entry.get('harness', 'other')
            if name not in newest or entry.get('last_used', 0) > newest[name].get('last_used', 0):
                newest[name] = entry
        names = sorted(newest, key=lambda n: -newest[n].get('last_used', 0))
        return [n for n in names if not cold_only or newest[n].get('warm_identity') != identity or not identity]

    def learning_status(self, learning=None):
        """The opt-in state and the unknown clients seen since start (numbers only)."""
        settings = dict(LEARNING_DEFAULTS, **(learning if isinstance(learning, dict) else {}))
        rows = []
        for name, row in reversed(self.candidates.items()):
            best = max(row['keys'].values())
            size = row['system_chars'] + row['tools_chars']
            rows.append({'harness': name, 'agent': row.get('agent', ''), 'requests': row['requests'],
                         'distinct_prefixes': len(row['keys']), 'consistent_requests': best,
                         'stable': best >= 2 or row['requests'] == 1,
                         'system_chars': row['system_chars'], 'tools': row['tools'], 'prefix_chars': size,
                         'large_enough': size >= settings['min_prefix_chars'],
                         'approved': name in self.approved, 'ignored': name in self.ignored,
                         'first_seen': row['first_seen'], 'last_used': row['last_used'],
                         'last_cache_n': row['last_cache_n'], 'last_prompt_n': row['last_prompt_n']})
        return {**settings, 'persistent': self.directory is not None, 'candidates': rows,
                'ignored': sorted(self.ignored)}

    def primers(self, harness, model_alias, limit=3):
        """Primer bodies for a harness, most recently used first: [(key, body)]."""
        if self.directory is None:
            return []
        keys = sorted((k for k, v in self.entries.items() if v.get('harness') == harness),
                      key=lambda k: -self.entries[k].get('last_used', 0))[:limit]
        out = []
        for key in keys:
            try:
                body = json.loads(self._primer_path(key).read_text())
            except (OSError, ValueError, AttributeError):
                continue
            if not isinstance(body, dict) or prefix_key(body) != key:
                continue  # edited or damaged primer: it would warm a different prefix
            out.append((key, {**body, 'model': model_alias, 'max_tokens': 1, 'temperature': 0, 'stream': False,
                              'cache_prompt': True, 'tool_choice': 'none'}))
        return out

    def status(self, config=None):
        """Per harness: prefixes seen and whether each is expected warm for `config`."""
        identity = engine_identity(config) if config else None
        harnesses = {}
        for key, entry in sorted(self.entries.items(), key=lambda kv: -kv[1].get('last_used', 0)):
            name = entry.get('harness', 'other')
            row = harnesses.setdefault(name, {'prefixes': [], 'warm': 0, 'last_used': 0})
            if str(name).startswith(LEARNED):
                row.update(learned=True, label=self.labels.get(name, ''))
            warm = bool(identity) and entry.get('warm_identity') == identity
            row['prefixes'].append({'key': key[:12], 'model': entry.get('model'), 'warm': warm,
                                    'system_chars': entry.get('system_chars'), 'tools': entry.get('tools'),
                                    'requests': entry.get('requests'), 'last_used': entry.get('last_used'),
                                    'last_cache_n': entry.get('last_cache_n'),
                                    'last_prompt_n': entry.get('last_prompt_n')})
            row['warm'] += warm
            row['last_used'] = max(row['last_used'], entry.get('last_used', 0))
        return harnesses
