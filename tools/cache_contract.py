"""CACHE-PERFECT P0 correctness contract, model-free (no weights, no engine, no install).

- C1 prefix-monotone rendering: tokens(render(H[..k])) is a prefix of tokens(render(H[..k+1])).
- C2 round trip: tokens(render(parse(completion))) must equal the generated token ids.
- C3 per-turn invariant: cache_n(N) >= cache_n(N-1) + prompt_n(N-1) + predicted_n(N-1) - 1,
  unless the client edited history.
- GGUF header reader: the chat template and tokenizer shape from the header only. It stops after the
  metadata; no tensor data is read and no model is loaded.

The checks take the renderer, parser and tokenizer as arguments, so the same functions run on the
engine's real template (through a tokenizer-only server) and on a toy stand-in in the tests.

P1 template hook: `oxe-server -m MODEL.gguf --engine stub --port P` reads only the GGUF metadata
(tokenizer + chat template, no weights) and answers `POST /apply-template` with
`"return_tokens": true` (the prompt ids exactly as a chat request tokenizes them) and
`"echo": true` (the ids after the engine's ECHO splice). `server_renderer` turns that into the
`render` argument of the checks; `c1` on the command line runs C1 over a history file.
Output of the CLI is counts and enum labels, never prompt text or token ids.
"""
import json
import os
import struct
import sys
import urllib.request

SLACK = 1
GGUF_MAGIC = b'GGUF'
_SCALAR = {0: ('B', 1), 1: ('b', 1), 2: ('H', 2), 3: ('h', 2), 4: ('I', 4), 5: ('i', 4), 6: ('f', 4),
           7: ('?', 1), 10: ('Q', 8), 11: ('q', 8), 12: ('d', 8)}
_STRING, _ARRAY = 8, 9
MAX_STRING = 1 << 22  # a chat template is kilobytes; anything near 4 MiB is refused
MAX_KV = 1 << 16      # metadata keys in one header (real files hold a few dozen)
MAX_DEPTH = 2         # nested arrays: an array of arrays at most


class ContractError(Exception):
    pass


def first_divergence(a, b):
    """Index of the first position where a and b differ, or None when one is a prefix of the other
    and both lengths are equal (no divergence)."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def render_deterministic(render, history):
    """The same history renders to byte-identical prompt tokens on two calls."""
    return render(history) == render(history)


def prefix_monotone_violations(render, history):
    """Check C1 over the growing prefixes of `history` (a list of messages). Returns the list of
    (k, first_divergent_token_index) for every step where the shorter render is not a prefix of the
    longer one. Empty means C1 holds for this history."""
    out = []
    previous = render(history[:1]) if history else []
    for k in range(2, len(history) + 1):
        current = render(history[:k])
        if current[:len(previous)] != previous:
            out.append((k, first_divergence(previous, current[:len(previous)])))
        previous = current
    return out


def roundtrip(parse, render, completion, generated_ids):
    """Check C2 for one completion: render(parse(text)) must equal the ids the engine generated.
    Returns (ok, first_divergent_offset_or_None)."""
    rendered = render(parse(completion))
    if rendered == list(generated_ids):
        return True, None
    return False, first_divergence(rendered, list(generated_ids))


def turn_invariant_ok(prev, cur, edited=False):
    """Check C3. `prev` and `cur` carry cache_n, prompt_n and predicted_n (predicted_n optional)."""
    if edited:
        return True
    expected = prev['cache_n'] + prev['prompt_n'] + (prev.get('predicted_n') or 0) - SLACK
    return cur['cache_n'] >= expected


# ---------------------------------------------------------------- template hook (oxe-server --engine stub)

def template_tokens(url, body, key=None, timeout=30):
    """POST /apply-template with return_tokens; returns the decoded JSON reply."""
    req = dict(body)
    req['return_tokens'] = True
    data = json.dumps(req).encode()
    headers = {'Content-Type': 'application/json', 'X-Local-AI-Origin': 'lab'}  # REAL-1: not organic use
    if key:
        headers['Authorization'] = 'Bearer ' + key
    r = urllib.request.Request(url.rstrip('/') + '/apply-template', data=data, headers=headers, method='POST')
    with urllib.request.urlopen(r, timeout=timeout) as resp:
        out = json.loads(resp.read())
    if not isinstance(out.get('tokens'), list):
        raise ContractError('template hook returned no tokens (engine without return_tokens?)')
    return out


def server_renderer(url, tools=None, kwargs=None, add_generation_prompt=False, key=None):
    """`render(history) -> token ids` through the engine's real template and tokenizer. The history
    form (no generation prompt) is the default: that is what C1 compares."""
    def render(history):
        body = {'messages': history, 'add_generation_prompt': add_generation_prompt}
        if tools:
            body['tools'] = tools
        if kwargs:
            body['chat_template_kwargs'] = kwargs
        return template_tokens(url, body, key)['tokens']
    return render


def c1_report(render, history):
    """Counts only: messages checked, C1 violations, and the first violating message index."""
    bad = prefix_monotone_violations(render, history)
    return {'messages': len(history), 'c1_violations': len(bad), 'first_violation_at': bad[0][0] if bad else None}


# ---------------------------------------------------------------- GGUF header (metadata only)

def _read(f, n):
    data = f.read(n)
    if len(data) != n:
        raise ContractError('truncated GGUF header')
    return data


def _left(f):
    """Bytes from the read position to the end of the file (every count is checked against it)."""
    return f.size - f.tell()


def _string(f):
    (length,) = struct.unpack('<Q', _read(f, 8))
    if length > MAX_STRING or length > _left(f):
        raise ContractError('GGUF string over the header limit')
    return _read(f, length).decode('utf-8', 'replace')


def _skip_value(f, kind, depth=0):
    if kind in _SCALAR:
        _read(f, _SCALAR[kind][1])
        return None
    if kind == _STRING:
        _string(f)
        return None
    if kind == _ARRAY:
        (elem, count) = struct.unpack('<IQ', _read(f, 12))
        size = _SCALAR[elem][1] if elem in _SCALAR else 8 if elem == _STRING else 12
        if depth >= MAX_DEPTH or count * size > _left(f):  # each element needs at least `size` bytes
            raise ContractError('GGUF array beyond the file or nesting limit')
        if elem in _SCALAR:
            f.seek(count * size, 1)
        else:
            for _ in range(count):
                _skip_value(f, elem, depth + 1)
        return count
    raise ContractError(f'unknown GGUF value type {kind}')


class _Sized:
    """A file with its size, so every seek and count can be bounded."""

    def __init__(self, f):
        self.f, self.size = f, os.fstat(f.fileno()).st_size
        self.read, self.seek, self.tell = f.read, f.seek, f.tell


def read_gguf_metadata(path, keys):
    """Values of the named metadata keys from a GGUF header. Strings come back as text, numbers as
    numbers, arrays as their element count. Stops once every key is found."""
    wanted = set(keys)
    found = {}
    with open(path, 'rb') as raw:
        f = _Sized(raw)
        if _read(f, 4) != GGUF_MAGIC:
            raise ContractError('not a GGUF file')
        (version,) = struct.unpack('<I', _read(f, 4))
        if version not in (2, 3):
            raise ContractError(f'unsupported GGUF version {version}')
        _tensors, kv_count = struct.unpack('<QQ', _read(f, 16))
        if kv_count > MAX_KV or kv_count * 13 > _left(f):  # key length + one char + value type
            raise ContractError('GGUF metadata count beyond the file')
        for _ in range(kv_count):
            key = _string(f)
            (kind,) = struct.unpack('<I', _read(f, 4))
            if key in wanted:
                if kind == _STRING:
                    found[key] = _string(f)
                elif kind in _SCALAR:
                    fmt, size = _SCALAR[kind]
                    found[key] = struct.unpack('<' + fmt, _read(f, size))[0]
                else:
                    found[key] = _skip_value(f, kind)
            else:
                _skip_value(f, kind)
            if len(found) == len(wanted):
                break
    return found


TEMPLATE_KEYS = ('general.architecture', 'tokenizer.chat_template', 'tokenizer.ggml.model',
                 'tokenizer.ggml.tokens', 'tokenizer.ggml.merges')


def template_shape(path):
    """Counts and flags only: the template's length, whether it keeps earlier reasoning under the
    preserve_thinking flag, and whether it drops reasoning before the last user query (the pattern
    that breaks prefix-monotone rendering, C1)."""
    meta = read_gguf_metadata(path, TEMPLATE_KEYS)
    template = meta.get('tokenizer.chat_template') or ''
    return {
        'architecture': meta.get('general.architecture'),
        'tokenizer_model': meta.get('tokenizer.ggml.model'),
        'vocab_size': meta.get('tokenizer.ggml.tokens'),
        'merges': meta.get('tokenizer.ggml.merges'),
        'template_chars': len(template),
        'has_preserve_thinking': 'preserve_thinking' in template,
        'drops_reasoning_before_last_query': 'last_query_index' in template,
    }


USAGE = ('usage: cache_contract.py <model.gguf>\n'
         '       cache_contract.py c1 <engine-url> <history.json> [<chat_template_kwargs JSON>]\n'
         '  history.json: {"messages": [...], "tools": [...]} (tools optional); engine-url of an\n'
         '  `oxe-server --engine stub` (no weights). Prints counts only.')


def main(argv):
    if len(argv) >= 4 and argv[1] == 'c1':
        try:
            with open(argv[3], encoding='utf-8') as f:
                hist = json.load(f)
            kwargs = json.loads(argv[4]) if len(argv) > 4 else {'enable_thinking': False, 'preserve_thinking': True}
            render = server_renderer(argv[2], hist.get('tools'), kwargs)
            print(c1_report(render, hist['messages']))
        except (OSError, ValueError, KeyError, ContractError) as exc:
            print(f'cache_contract: {type(exc).__name__}', file=sys.stderr)
            return 2
        return 0
    if len(argv) != 2:
        print(USAGE, file=sys.stderr)
        return 2
    try:
        print(template_shape(argv[1]))
    except (OSError, ContractError) as exc:
        print(f'cache_contract: {exc}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))
