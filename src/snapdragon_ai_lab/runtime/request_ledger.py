"""Always-on request ledger for the local inference proxy: one JSON line per inference request.

Numbers, hashes and enum-like labels only, never prompt or reply content:
- queue wait (FIFO admission), admit time (includes a model load), time to the backend's first
  byte, wall time, HTTP status;
- labels: client product token from User-Agent (e.g. "opencode/2.2.6"), SHA-256 prefix of the
  system prompt (agent identity), message/tool counts, last role, stream flag, max_tokens,
  temperature;
- the backend's own numbers from the end of its response (llama-server/oxe `timings`:
  cache_n, prompt_n, prompt_ms, predicted_n, predicted_ms, draft counters; finish_reason).

Files: LOCAL_AI_LEDGER_DIR (default ~/.local/state/local-ai/ledger)/requests-YYYYMMDD.jsonl,
directory 0700, files 0600, kept LOCAL_AI_LEDGER_DAYS (14) days. LOCAL_AI_LEDGER=0 turns it
off. A ledger failure never affects the request.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import threading
import time

TAIL_BYTES = 16384
_UA = re.compile(r'^[A-Za-z0-9._-]{1,32}(/[A-Za-z0-9._+-]{1,24})?')
_FINISH = re.compile(rb'"finish_reason"\s*:\s*"([a-z_]{1,24})"')
_lock = threading.Lock()
_pruned_day = [None]


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
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None


class Entry:
    def __init__(self, path, payload, user_agent=None, model_alias=None):
        self.t0 = time.monotonic()
        self.row = {'id': secrets.token_hex(6), 't': round(time.time(), 3), 'path': path,
                    'model': model_alias if isinstance(model_alias, str) else None}
        m = _UA.match(user_agent or '')
        self.row['client'] = m.group(0) if m else None
        if isinstance(payload, dict):
            msgs = payload.get('messages') if isinstance(payload.get('messages'), list) else []
            system = next((_text(x.get('content')) for x in msgs if isinstance(x, dict) and x.get('role') == 'system'), None)
            self.row.update({
                'agent': hashlib.sha256(system.encode()).hexdigest()[:12] if system is not None else None,
                'n_messages': len(msgs),
                'last_role': msgs[-1].get('role') if msgs and isinstance(msgs[-1], dict) and isinstance(msgs[-1].get('role'), str) else None,
                'n_tools': len(payload['tools']) if isinstance(payload.get('tools'), list) else 0,
                'stream': bool(payload.get('stream')),
                'max_tokens': _num(payload.get('max_tokens', payload.get('n_predict'))),
                'temperature': _num(payload.get('temperature')),
            })
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

    def admitted_now(self, queue_wait_s=None, begin_s=None):
        self.admitted = time.monotonic()
        self.row['queue_wait_ms'] = round(queue_wait_s * 1e3, 1) if queue_wait_s is not None else None
        self.row['admit_ms'] = round(begin_s * 1e3, 1) if begin_s is not None else None

    def observe(self, chunk):
        if self.first_byte is None:
            self.first_byte = time.monotonic()
        self.tail = (self.tail + chunk)[-TAIL_BYTES:]

    def finish(self, status, error=None):
        now = time.monotonic()
        r = self.row
        r['status'] = status
        r['error'] = error
        r['wall_ms'] = round((now - self.t0) * 1e3, 1)
        if self.admitted is not None and self.first_byte is not None:
            r['ttfb_ms'] = round((self.first_byte - self.admitted) * 1e3, 1)
        r.update(parse_tail(self.tail))
        write(r)
        return r


def parse_tail(tail):
    """Backend numbers from the end of a response (SSE stream or JSON body)."""
    out = {}
    i = tail.rfind(b'"timings"')
    if i >= 0:
        j = tail.find(b'{', i)
        try:
            t, _ = json.JSONDecoder().raw_decode(tail[j:].decode('utf-8', 'replace'))
            for k in ('cache_n', 'prompt_n', 'prompt_ms', 'predicted_n', 'predicted_ms', 'draft_n', 'draft_n_accepted'):
                if _num(t.get(k)) is not None:
                    out[k] = t[k]
        except (ValueError, AttributeError):
            pass
    f = _FINISH.findall(tail)
    if f:
        out['finish_reason'] = f[-1].decode()
    return out


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
                keep = float(os.environ.get('LOCAL_AI_LEDGER_DAYS', '14')) * 86400
                for p in d.glob('requests-*.jsonl'):
                    try:
                        if time.time() - p.stat().st_mtime > keep:
                            p.unlink()
                    except OSError:
                        pass
            fd = os.open(d / f'requests-{day}.jsonl', os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
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
