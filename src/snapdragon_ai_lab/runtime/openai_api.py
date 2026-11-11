"""The controller's OpenAI-compatible surface, so that any harness can use the local models.

Pure functions over request/response values (the only I/O is reading an API-key file). The
controller calls them through narrow hooks; nothing here starts a model or keeps request
content.

- Model names: `local-default` (the selected model), `local-heavy` (Flash) and further
  aliases from runtime.json `openai_api.model_aliases`, besides every profile's full name.
  A litellm-style `openai/<name>` prefix is accepted. Static names only: choosing a model
  per request (routing) is not done here.
- Errors on non-`/_local` routes always have the OpenAI shape
  `{"error": {"message", "type", "param", "code"}}`.
- Optional API key (off by default): env `LOCAL_AI_API_KEY` or runtime.json
  `openai_api.api_key_file`. When set, the API routes need `Authorization: Bearer <key>`.
- Opt-in CORS: runtime.json `openai_api.cors_origins` (exact origins; `*` only together
  with an API key). Without it cross-origin browser requests stay refused.
- `/v1/completions` for the Rust engine, which only has llama-server's native
  `/completion`: request and response (JSON and SSE) are translated.
- Flash (BigMoe) accepts only temperature 0, `max_tokens` <= 2048 and no thinking; generic
  clients send other values, so they are brought into that range and counted.
"""
import hmac
import json
import os
import re
import secrets
import time

DEFAULT_ALIAS = 'local-default'
LEGACY_DEFAULT = 'qwen-local'
DEFAULT_NAMES = (None, '', DEFAULT_ALIAS, LEGACY_DEFAULT)
RESERVED_NAMES = ('local-auto',)  # the explicit router's name (explicit_route.py); never an alias
BUILTIN_ALIASES = {'local-heavy': 'qwen3.8-flash-next-local'}
ALIAS_RE = re.compile(r'[A-Za-z0-9][A-Za-z0-9._:-]{0,63}')
PROVIDER_PREFIXES = ('openai/', 'openai_compatible/', 'local/')
KEY_ENV = 'LOCAL_AI_API_KEY'
STATIC_SUFFIXES = ('.js', '.css', '.svg', '.ico', '.woff', '.woff2', '.html', '.png', '.json.gz')
# Endpoints the text-only engines (Rust engine, Flash) do not have. Answered by the
# controller without waking a model.
UNSUPPORTED = {
    '/v1/embeddings': 'embeddings', '/embeddings': 'embeddings', '/embedding': 'embeddings',
    '/v1/rerank': 'reranking', '/rerank': 'reranking', '/v1/reranking': 'reranking', '/reranking': 'reranking',
    '/v1/responses': 'the Responses API', '/infill': 'infill',
}
FLASH_MAX_TOKENS = 2048


# ------------------------------------------------------------------ settings
def settings(value):
    raw = value.get('openai_api') if isinstance(value, dict) else None
    return raw if isinstance(raw, dict) else {}


def alias_table(value):
    """{alias: profile name}: the built-in table, changed by `openai_api.model_aliases`
    (a null target removes an alias). Aliases never shadow a profile or a default name."""
    table = dict(BUILTIN_ALIASES)
    extra = settings(value).get('model_aliases')
    if isinstance(extra, dict):
        for name, target in extra.items():
            if not isinstance(name, str) or not ALIAS_RE.fullmatch(name):
                continue
            if target is None:
                table.pop(name, None)
            elif isinstance(target, str) and target:
                table[name] = target
    profiles = value.get('profiles', {}) if isinstance(value, dict) else {}
    base = value.get('model_alias') if isinstance(value, dict) else None
    return {name: target for name, target in table.items()
            if name not in profiles and name != base and name not in DEFAULT_NAMES
            and name not in RESERVED_NAMES}


def resolve_model(name, value):
    """Profile name for a requested model id; None means "the selected model".

    Unknown names are returned unchanged (the caller reports them), unless
    `openai_api.unknown_model` is "default"."""
    if name in DEFAULT_NAMES:
        return None
    if not isinstance(name, str):
        return name
    profiles = value.get('profiles', {}) if isinstance(value, dict) else {}
    base = value.get('model_alias', LEGACY_DEFAULT) if isinstance(value, dict) else LEGACY_DEFAULT
    table = alias_table(value)
    candidates = [name] + [name[len(p):] for p in PROVIDER_PREFIXES if name.startswith(p)]
    for candidate in candidates:
        if candidate in DEFAULT_NAMES:
            return None
        if candidate in profiles or candidate == base:
            return candidate
        if candidate in table:
            return table[candidate]
    if settings(value).get('unknown_model') == 'default':
        return None
    return name


def model_rows(value, rows, selected):
    """/v1/models rows: `local-default`, the aliases, every full name, then the legacy
    `qwen-local`. `rows` are the eligible profile rows the controller already built."""
    by_id = {row['id']: row for row in rows}
    out = []

    def alias_row(name, target, label):
        return {'id': name, 'object': 'model', 'created': 0, 'owned_by': 'local',
                'name': label, 'alias_of': target}

    chosen = by_id.get(selected)
    if chosen is not None:
        out.append(alias_row(DEFAULT_ALIAS, selected, 'Default: ' + str(chosen.get('name', selected))))
    for name, target in alias_table(value).items():
        if target in by_id and target not in DEFAULT_NAMES:
            out.append(alias_row(name, target, '%s: %s' % (name.rsplit('-', 1)[-1].capitalize(),
                                                          by_id[target].get('name', target))))
    out.extend(row for row in rows if row['id'] != LEGACY_DEFAULT)
    if LEGACY_DEFAULT in by_id:
        out.append({**by_id[LEGACY_DEFAULT], 'alias_of': selected, 'legacy': True})
    return out


# ------------------------------------------------------------------ errors
ERROR_TYPES = {400: 'invalid_request_error', 401: 'authentication_error', 403: 'permission_error',
               404: 'not_found_error', 409: 'conflict_error', 411: 'invalid_request_error',
               413: 'invalid_request_error', 429: 'rate_limit_error', 501: 'not_supported_error',
               503: 'unavailable_error'}


def error_body(status, message, kind=None, code=None, param=None):
    return {'error': {'message': str(message), 'type': kind or ERROR_TYPES.get(status, 'server_error'),
                      'param': param, 'code': code}}


def normalize_error(status, value):
    """Give any `{"error": ...}` reply the OpenAI shape; other values pass unchanged."""
    if not isinstance(value, dict) or 'error' not in value or status < 400:
        return value
    error = value['error']
    if isinstance(error, dict):
        filled = {'message': str(error.get('message', '')),
                  'type': error.get('type') or ERROR_TYPES.get(status, 'server_error'),
                  'param': error.get('param'), 'code': error.get('code')}
        return {**value, 'error': {**error, **filled}}
    code = 'model_not_found' if status == 404 and 'model' in str(error).lower() else None
    return {**value, 'error': error_body(status, error, code=code)['error']}


# ------------------------------------------------------------------ API key
def api_key(value, environ=None):
    """The configured key, or None (no key: every loopback client is accepted)."""
    environ = os.environ if environ is None else environ
    key = environ.get(KEY_ENV, '').strip()
    if key:
        return key
    path = settings(value).get('api_key_file')
    if path is not None and not (isinstance(path, str) and os.path.isabs(path)):
        return secrets.token_hex(32)  # configured but unusable (relative, "~"): closed, not open
    if path is not None:
        try:
            with open(path, encoding='utf-8') as stream:
                key = stream.readline(4097).strip()
        except (OSError, UnicodeError):
            # A configured but unreadable key file must not open the API.
            return secrets.token_hex(32)
        return key or secrets.token_hex(32)
    return None


def presented_key(headers):
    value = headers.get('Authorization')
    if isinstance(value, str) and value[:7].lower() == 'bearer ':
        return value[7:].strip()
    for name in ('x-api-key', 'api-key'):
        value = headers.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def key_required(method, path):
    """Whether a route needs the key when one is set: everything except the control routes
    (`/_local/*`, answered by the controller itself) and `/health`. That includes the
    built-in chat page, which would otherwise load a model for a keyless visitor."""
    return not (path == '/health' or path.startswith('/_local/'))


def authorized(headers, key):
    given = presented_key(headers)
    return given is not None and hmac.compare_digest(given.encode(), key.encode())


# ------------------------------------------------------------------ CORS
def cors_origin(origins, value, key_set):
    """The origin to allow for this request, or None. `origins`: the request's Origin
    header values."""
    if len(origins) != 1:
        return None
    allowed = settings(value).get('cors_origins')
    if not isinstance(allowed, list):
        return None
    origin = origins[0]
    if origin in allowed and origin != '*' and origin != 'null':
        return origin
    if '*' in allowed and key_set:  # any site may call, but only with the key
        return origin
    return None


def cors_headers(origin, preflight=False, request_headers=None, private_network=False):
    headers = [('Access-Control-Allow-Origin', origin), ('Vary', 'Origin')]
    if preflight:
        headers += [('Access-Control-Allow-Methods', 'GET, POST, OPTIONS'),
                    ('Access-Control-Allow-Headers', request_headers or 'authorization, content-type'),
                    ('Access-Control-Max-Age', '600')]
        if private_network:
            headers.append(('Access-Control-Allow-Private-Network', 'true'))
    return headers


# ------------------------------------------------------------------ request fixes
def single_choice(payload, notes):
    """The engines return one choice. `n` > 1 is served as one (counted), not refused:
    clients that send it read `choices[0]`."""
    n = payload.get('n')
    if 'n' in payload and (type(n) is not int or n != 1):
        payload = {k: v for k, v in payload.items() if k != 'n'}
        if type(n) is int and n > 1:
            notes.append('n_reduced_to_1')
    return payload


def flash_request(payload, notes):
    """Bring a generic chat body into what Flash is qualified for (temperature 0,
    max_tokens <= 2048, no thinking). Each change is counted in `notes`."""
    result = dict(payload)
    if 'max_completion_tokens' in result:
        limit = result.pop('max_completion_tokens')
        if 'max_tokens' not in result and limit is not None:
            result['max_tokens'] = limit
        notes.append('flash_max_completion_tokens_renamed')
    limit = result.get('max_tokens')
    if limit is None and 'max_tokens' in result:
        del result['max_tokens']
    elif type(limit) is int and limit > FLASH_MAX_TOKENS:
        result['max_tokens'] = FLASH_MAX_TOKENS
        notes.append('flash_max_tokens_capped')
    temperature = result.get('temperature')
    if 'temperature' in result and temperature != 0:
        result['temperature'] = 0
        if temperature is not None:
            notes.append('flash_temperature_forced_0')
    if result.get('reasoning_effort', 'none') != 'none':
        del result['reasoning_effort']
        notes.append('flash_reasoning_effort_dropped')
    return single_choice(result, notes)


# ------------------------------------------------------------------ /v1/completions shim
class Unsupported(ValueError):
    pass


def completion_request(payload):
    """OpenAI `/v1/completions` body -> (native `/completion` body, meta).

    Raises Unsupported for what cannot be honoured (several prompts, `suffix`)."""
    if not isinstance(payload, dict):
        raise Unsupported('The request body must be a JSON object.')
    prompt = payload.get('prompt', '')
    if isinstance(prompt, list) and prompt and all(isinstance(p, (str, list)) for p in prompt):
        if len(prompt) != 1:
            raise Unsupported('One prompt per request: send a string, or a list with one prompt.')
        prompt = prompt[0]
    if not isinstance(prompt, (str, list)):
        raise Unsupported('"prompt" must be a string or a list.')
    if payload.get('suffix') not in (None, ''):
        raise Unsupported('"suffix" (insert mode) is not supported by this model server.')
    native = {k: v for k, v in payload.items()
              if k not in ('prompt', 'max_tokens', 'max_completion_tokens', 'logprobs', 'echo', 'suffix',
                           'n', 'best_of', 'stream_options', 'user')}
    native['prompt'] = prompt
    for key in ('max_tokens', 'max_completion_tokens'):
        if type(payload.get(key)) is int and 'n_predict' not in native:
            native['n_predict'] = payload[key]
    logprobs = payload.get('logprobs')
    if type(logprobs) is int and logprobs > 0:
        native['n_probs'] = min(logprobs, 20)
    options = payload.get('stream_options')
    meta = {'id': 'cmpl-' + secrets.token_hex(12), 'created': int(time.time()),
            'model': payload.get('model'), 'stream': payload.get('stream') is True,
            'echo': prompt if payload.get('echo') is True and isinstance(prompt, str) else '',
            'include_usage': isinstance(options, dict) and options.get('include_usage') is True,
            'logprobs': 'n_probs' in native}
    return native, meta


def _finish(native):
    return 'stop' if native.get('stop_type') in ('eos', 'word') else 'length'


def _usage(native):
    prompt, done = native.get('tokens_evaluated'), native.get('tokens_predicted')
    prompt, done = (prompt if type(prompt) is int else 0), (done if type(done) is int else 0)
    usage = {'prompt_tokens': prompt, 'completion_tokens': done, 'total_tokens': prompt + done}
    timings = native.get('timings')
    if isinstance(timings, dict) and 'cache_n' in timings:
        usage['prompt_tokens_details'] = {'cached_tokens': timings['cache_n']}
    return usage


def _logprobs(native, offset=0):
    rows = native.get('completion_probabilities')
    if not isinstance(rows, list):
        return None
    out = {'tokens': [], 'token_logprobs': [], 'top_logprobs': [], 'text_offset': []}
    for row in rows:
        if not isinstance(row, dict):
            continue
        token = row.get('token') if isinstance(row.get('token'), str) else ''
        out['tokens'].append(token)
        out['token_logprobs'].append(row.get('logprob'))
        tops = row.get('top_logprobs') if isinstance(row.get('top_logprobs'), list) else []
        out['top_logprobs'].append({str(t.get('token', '')): t.get('logprob') for t in tops if isinstance(t, dict)})
        out['text_offset'].append(offset)
        offset += len(token)
    return out


def _chunk(meta, text, finish, logprobs=None):
    return {'id': meta['id'], 'object': 'text_completion', 'created': meta['created'], 'model': meta['model'],
            'choices': [{'text': text, 'index': 0, 'logprobs': logprobs, 'finish_reason': finish}]}


def completion_response(native, meta):
    """Native `/completion` reply (not streamed) -> OpenAI `text_completion`."""
    if not isinstance(native, dict) or 'content' not in native:
        return native  # an error object: already in the OpenAI shape
    content = native.get('content')
    text = meta['echo'] + (content if isinstance(content, str) else '')
    value = _chunk(meta, text, _finish(native),
                   _logprobs(native, len(meta['echo'])) if meta['logprobs'] else None)
    value['model'] = native.get('model', meta['model'])
    value['usage'] = _usage(native)
    if isinstance(native.get('timings'), dict):
        value['timings'] = native['timings']
    return value


class CompletionStream:
    """Translate the native `/completion` SSE stream into OpenAI `text_completion` chunks."""

    def __init__(self, meta):
        self.meta = meta
        self.buffer = b''
        self.done = False
        self.first = True

    def _frame(self, value):
        return b'data: ' + json.dumps(value, ensure_ascii=False).encode() + b'\n\n'

    def _event(self, raw):
        # Anything this translation cannot handle is passed on as the engine sent it; an
        # exception here would cut the HTTP response in the middle.
        try:
            return self._translate(raw)
        except (TypeError, ValueError, KeyError, AttributeError, UnicodeError):
            return raw + b'\n\n'

    def _translate(self, raw):
        if not raw.startswith(b'data:'):
            return raw + b'\n\n' if raw.strip() else b''
        data = raw[5:].strip()
        if data == b'[DONE]':
            return b''
        try:
            native = json.loads(data)
        except ValueError:
            return raw + b'\n\n'
        if not isinstance(native, dict) or 'content' not in native:
            return raw + b'\n\n'  # an error event
        text = native.get('content')
        text = text if isinstance(text, str) else ''
        if self.first:
            text, self.first = self.meta['echo'] + text, False
        if native.get('stop') is not True:
            return self._frame(_chunk(self.meta, text, None,
                                      _logprobs(native) if self.meta['logprobs'] else None))
        self.done = True
        last = _chunk(self.meta, text, _finish(native))
        out = b''
        if self.meta['include_usage']:
            out += self._frame(last)
            last = {**_chunk(self.meta, '', None), 'choices': [], 'usage': _usage(native)}
        if isinstance(native.get('timings'), dict):
            last['timings'] = native['timings']
        return out + self._frame(last) + b'data: [DONE]\n\n'

    def feed(self, chunk):
        self.buffer += chunk
        out = b''
        while b'\n\n' in self.buffer:
            raw, self.buffer = self.buffer.split(b'\n\n', 1)
            out += self._event(raw)
        return out

    def finish(self):
        out = self._event(self.buffer) if self.buffer.strip() else b''
        self.buffer = b''
        return out
