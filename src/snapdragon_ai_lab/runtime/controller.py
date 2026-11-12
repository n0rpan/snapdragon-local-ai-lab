#!/usr/bin/env python3
"""Local llama.cpp proxy with an exclusive inference lease and idle unloading.

HTTP request/response bodies are not logged. Native stderr goes to a bounded,
private diagnostic log. Settings live in runtime.json; inference shares a file
lock with image jobs.

One exception to "nothing of a request is kept": for requests to the Rust engine the
harness registry saves the leading system messages and tool definitions (never user
turns, tool results or replies) as primers under LOCAL_AI_HARNESS_STATE, mode 0600, so
a harness prefix can be rebuilt on request. Set LOCAL_AI_HARNESS_STATE='' to keep the
registry in memory only.
"""
import collections
from contextlib import contextmanager
import errno
import fcntl
import gzip
import hashlib
import html
import http.client
import io
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import os
from pathlib import Path
import re
import resource
import shutil
import signal
import socket
import stat
import pwd
import subprocess
import sys
import threading
import traceback
import tempfile
import time
from urllib.parse import parse_qs, urlsplit
if __package__:
    from .prefix_plan import make_plan
    from .prefix_store import PrefixStore
    from . import bigmoe_runtime, classic_beta, memory_policy, model_catalog, profile_policy, oxe_runtime, engine_ledger, explicit_route, storage_view
    from .engine_diagnostics import EngineLogWriter, EngineStderrReader
    from .client_disconnect import ClientDisconnect
    from . import harness_compat, openai_api, request_ledger, image_slot, image_npu, cache_turn, state_cache
    from .harness_registry import HarnessRegistry, learning_settings, timings_from_tail
else:
    from prefix_plan import make_plan
    from prefix_store import PrefixStore
    import bigmoe_runtime
    import classic_beta
    import memory_policy
    import model_catalog
    import profile_policy
    import oxe_runtime
    import engine_ledger
    import explicit_route
    from engine_diagnostics import EngineLogWriter, EngineStderrReader
    from client_disconnect import ClientDisconnect
    import harness_compat
    import openai_api
    import request_ledger
    import cache_turn
    import state_cache
    import image_slot
    import image_npu
    from harness_registry import HarnessRegistry, learning_settings, timings_from_tail
    import storage_view

ROOT = Path(__file__).resolve().parent
# M4b: XDG defaults (the old ~/Work/local-llm and ~/Work/prompt-studio fallbacks were private-lab
# paths). The installed unit sets both explicitly, so this only affects a unit without them.
CONFIG = Path(os.environ.get('LOCAL_LLM_CONFIG', str(
    Path(os.environ.get('XDG_CONFIG_HOME') or Path.home() / '.config') / 'local-ai/llm/runtime.json')))
LEASE = Path(os.environ.get('LOCAL_AI_LEASE', str(
    Path(os.environ.get('XDG_STATE_HOME') or Path.home() / '.local/state') / 'local-ai/inference.lock')))


def under_test_runner():
    return 'unittest' in sys.modules or 'pytest' in sys.modules


# The real per-user production paths, from the password database so a patched HOME/XDG cannot
# hide them: runtime config, controller state and lease, and the installed Omarchy panel.
_REAL_HOME = Path(pwd.getpwuid(os.getuid()).pw_dir)
REAL_PRODUCTION_PATHS = (_REAL_HOME / '.config/local-ai', _REAL_HOME / '.local/state/local-ai',
                         _REAL_HOME / '.config/omarchy', _REAL_HOME / '.local/bin/local-ai',
                         _REAL_HOME / 'Work/local-llm', _REAL_HOME / 'Work/prompt-studio')
if under_test_runner():
    # A test that forgets to patch CONFIG/LEASE must never reach production state: without an
    # explicit environment path the defaults point into a private per-process temp directory.
    _TEST_ROOT = Path(tempfile.mkdtemp(prefix='local-ai-controller-test-'))
    if 'LOCAL_LLM_CONFIG' not in os.environ:
        CONFIG = _TEST_ROOT / 'runtime.json'
    if 'LOCAL_AI_LEASE' not in os.environ:
        LEASE = _TEST_ROOT / 'inference.lock'


def guard_production_path(path):
    """Refuse to write the real runtime config or take the real lease from a test run."""
    if not under_test_runner():
        return
    resolved = Path(path).resolve()
    for real in REAL_PRODUCTION_PATHS:
        real = real.resolve()
        if resolved == real or resolved.is_relative_to(real):
            raise RuntimeError(f'test run refused to touch production path {resolved}')
BACKEND_PORT = int(os.environ.get('LOCAL_LLM_BACKEND_PORT', '8093'))
FLASH_BACKEND_PORT = int(os.environ.get('LOCAL_FLASH_BACKEND_PORT', '8094'))
if BACKEND_PORT == FLASH_BACKEND_PORT:
    raise ValueError('Classic and Flash backends require distinct private ports.')
PROXY_PORT = int(os.environ.get('LOCAL_LLM_PORT', '8092'))
IDLE_SECONDS = float(os.environ.get('LOCAL_LLM_IDLE_SECONDS', '600'))
# Overlapping inference requests (OpenCode title/subagent calls) wait in FIFO order for
# the single backend slot instead of failing with 503; 503 only after this many seconds.
QUEUE_WAIT_SECONDS = float(os.environ.get('LOCAL_LLM_QUEUE_WAIT_SECONDS', '120'))
START_SECONDS = float(os.environ.get('LOCAL_LLM_START_SECONDS', '180'))
# cleanup(): SIGTERM grace on the first stop, then a bounded wait after each SIGKILL. The manager
# lock is held meanwhile, so neither may be unbounded (a D-state engine is retried by the reaper).
TERM_WAIT_SECONDS = 10.0
KILL_WAIT_SECONDS = 1.0
# REAL-1: a request is lab traffic only when it says so (same rule as request_ledger.origin_of).
ORIGIN_HEADER = 'X-Local-AI-Origin'
LAB_ORIGIN = 'lab'


# S14: where an unexpected failure's traceback goes, named in every generic error reply.
JOURNAL_HINT = 'Details: journalctl --user -u local-ai-chat.'


def unexpected_cause(error):
    """'TypeName: first line of the message', bounded; never request data (begin() has none)."""
    text = str(error).splitlines()[0] if str(error) else ''
    return (type(error).__name__ + (': ' + text if text else ''))[:200]


def request_origin(headers):
    value = headers.get(ORIGIN_HEADER) if headers is not None else None
    return LAB_ORIGIN if isinstance(value, str) and value.strip().lower() == LAB_ORIGIN else 'organic'


def engine_summary(config):
    """M4: the engine a resolved profile will use, and why it is not the Rust one when Rust is
    configured. Metadata only (one stat of the Rust binary)."""
    kind = config.get('backend_kind') or 'llama'
    engine = {'oxe': 'rust', 'llama': 'llama.cpp', 'bigmoe-cpu-stream': 'bigmoe'}.get(kind, 'unknown')
    note = None
    if engine == 'rust':
        note = oxe_runtime.ineligible_reason(config)
    else:
        source, rust = profile_policy.rust_overrides(config)
        if rust:
            flash = kind == profile_policy.FLASH_KIND
            switch = profile_policy.OXE_FLASH_ENABLE_ENV if flash else profile_policy.OXE_ENABLE_ENV
            if not profile_policy.switch_on(switch):
                note = f'classic fallback: {switch}=0'
            elif source == 'memlat_overrides':
                note = 'classic fallback: Rust is configured under memlat_overrides and MEMLAT is not active on this boot'
            else:
                note = 'classic fallback: Rust configuration did not apply'
    return {'engine': engine, 'engine_note': note}


HOP_HEADERS = {'connection', 'keep-alive', 'proxy-authenticate', 'proxy-authorization',
               'te', 'trailer', 'transfer-encoding', 'upgrade', 'content-length'}
MODEL_HELPER_PATHS = {'/tokenize', '/detokenize', '/apply-template'}
INFERENCE_PATHS = {'/completion', '/completions', '/chat/completions', '/infill',
                   '/embedding', '/embeddings', '/rerank', '/reranking',
                   '/v1/completions', '/v1/chat/completions', '/v1/embeddings',
                   '/v1/rerank', '/v1/reranking', '/v1/responses'}

FLASH_CHAT_HTML = b'''<!doctype html><html><meta charset="utf-8"><title>Local Flash chat</title>
<style>body{font:16px system-ui;max-width:760px;margin:5vh auto;padding:16px;background:#171717;color:#eee}textarea{width:100%;min-height:90px}pre{white-space:pre-wrap}button{margin-top:8px}</style>
<body data-model="{MODEL}"><h1>Local Flash chat</h1><p>This page keeps messages in memory until it closes. The model loads when you send a message.</p>
<pre id="transcript"></pre><textarea id="prompt" aria-label="Message"></textarea><button id="send">Send</button><p id="status"></p>
<script>const messages=[];const transcript=document.getElementById('transcript');const status=document.getElementById('status');const button=document.getElementById('send');const input=document.getElementById('prompt');
button.onclick=async()=>{const text=input.value.trim();if(!text)return;button.disabled=true;status.textContent='Loading or answering...';messages.push({role:'user',content:text});transcript.textContent+='You: '+text+'\\n\\n';input.value='';try{const reply=await fetch('/v1/chat/completions',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({model:document.body.dataset.model,messages,stream:false,max_tokens:512,think:false})});const data=await reply.json();if(!reply.ok)throw new Error('The local backend could not answer.');const answer=data.choices?.[0]?.message?.content;if(typeof answer!=='string')throw new Error('Invalid local answer.');messages.push({role:'assistant',content:answer});transcript.textContent+='Assistant: '+answer+'\\n\\n';status.textContent='Ready';}catch(error){messages.pop();status.textContent=error.message;}finally{button.disabled=false;}};
</script></body></html>'''


def canonical_target(raw):
    """The request target the controller routes and forwards: one leading '/', no trailing '/'
    (except the root), query kept, fragment dropped. None for anything else ('//x' would parse
    as a host name, absolute-form and '*' are not served)."""
    if not isinstance(raw, str) or not raw.startswith('/') or raw.startswith('//'):
        return None
    parts = urlsplit(raw)
    if parts.scheme or parts.netloc:
        return None
    path = parts.path.rstrip('/') or '/'
    if path.startswith('//'):
        return None
    return path + ('?' + parts.query if parts.query else '')


def available_memory_bytes():
    try:
        for line in Path('/proc/meminfo').read_text().splitlines():
            if line.startswith('MemAvailable:'):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        pass
    return None


def total_memory_bytes():
    try:
        for line in Path('/proc/meminfo').read_text().splitlines():
            if line.startswith('MemTotal:'):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        pass
    return None


def swap_bytes():
    """(SwapTotal, SwapFree) in bytes; None for a field that is missing or unreadable (= no swap)."""
    found = {}
    try:
        for line in Path('/proc/meminfo').read_text().splitlines():
            name, _, rest = line.partition(':')
            if name in ('SwapTotal', 'SwapFree'):
                found[name] = int(rest.split()[0]) * 1024
    except (OSError, ValueError, IndexError):
        return None, None
    return found.get('SwapTotal'), found.get('SwapFree')


def cgroup_memory_max():
    """The controller's own cgroup memory limit in bytes, or None when unlimited or unreadable."""
    try:
        value = (memory_policy.current_cgroup() / 'memory.max').read_text().strip()
    except (OSError, ValueError):
        return None
    return int(value) if value.isdecimal() else None


def idle_seconds(config):
    config = config or {}
    if config.get('backend_kind') == oxe_runtime.KIND and oxe_runtime.is_flash(config):
        return config['oxe_flash'].get('idle_unload_seconds', IDLE_SECONDS)
    return IDLE_SECONDS


def resident_bytes(pid, root=Path('/proc')):
    """Resident memory of one process from its status file, or None."""
    try:
        for line in (Path(root) / str(pid) / 'status').read_text().splitlines():
            if line.startswith('VmRSS:'):
                fields = line.split()
                if len(fields) == 3 and fields[1].isdecimal() and fields[2] == 'kB':
                    return int(fields[1]) * 1024
    except (OSError, UnicodeError):
        pass
    return None


class Unavailable(Exception):
    pass


def open_lease_append():
    """N6: the lease for appending, created 0600, never through a symlink (a planted link is refused)."""
    try:
        return os.fdopen(os.open(LEASE, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW, 0o600), 'a')
    except OSError as error:
        if error.errno == errno.ELOOP:
            raise Unavailable(f'The inference lease {LEASE} is a symbolic link; refusing to use it. '
                              'Remove the link and retry.') from None
        raise


def listening_inodes(port, proc_root=Path('/proc')):
    """socket:[inode] names of loopback/any-address TCP listeners on `port`."""
    inodes = set()
    for name in ('tcp', 'tcp6'):
        try:
            rows = (proc_root / 'net' / name).read_text().splitlines()[1:]
        except OSError:
            continue
        for row in rows:
            fields = row.split()
            if len(fields) >= 10 and fields[3] == '0A':
                try:
                    address, number = fields[1].rsplit(':', 1)
                    loopback = {'00000000', '0100007F'} if name == 'tcp' else {
                        '00000000000000000000000000000000',
                        '0000000000000000FFFF00000100007F'}
                    if address in loopback and int(number, 16) == port:
                        inodes.add('socket:[' + fields[9] + ']')
                except (ValueError, IndexError):
                    continue
    return inodes


def port_listener_in_group(port, pgid, proc_root=Path('/proc')):
    """S6: True when a process in process group `pgid` holds the listener on `port`, False when
    another process does, None when the listener cannot be seen (no claim either way)."""
    inodes = listening_inodes(port, proc_root)
    if not inodes:
        return None
    deadline = time.monotonic() + .5
    other = False
    for process in proc_root.glob('[0-9]*'):
        if time.monotonic() >= deadline:
            return None
        try:
            stat_text = (process / 'stat').read_text()
            group = int(stat_text.rsplit(')', 1)[1].split()[2])
            for descriptor in (process / 'fd').iterdir():
                if os.readlink(descriptor) in inodes:
                    if group == pgid:
                        return True
                    other = True
        except (OSError, ValueError, IndexError):
            continue
    return False if other else None


def listening_process(port, proc_root=Path('/proc')):
    """Best-effort bounded diagnostic; never read argv, environment or socket data."""
    deadline = time.monotonic() + .25
    inodes = listening_inodes(port, proc_root)
    if not inodes:
        return 'another process'
    for process in proc_root.glob('[0-9]*'):
        try:
            for descriptor in (process / 'fd').iterdir():
                if time.monotonic() >= deadline:
                    return 'another process'
                if os.readlink(descriptor) in inodes:
                    name = (process / 'comm').read_text().strip()
                    return re.sub(r'[^A-Za-z0-9_.+-]', '_', name)[:64] or 'another process'
        except (OSError, ValueError):
            continue
    return 'another process'


def require_backend_port(port, setting):
    """Refuse a busy loopback port before allocating/loading model weights.

    This is a preflight, not a reservation: the native bind remains authoritative
    if another program takes the port between this check and launch.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind(('127.0.0.1', port))
    except OSError as error:
        if error.errno == errno.EADDRINUSE:
            raise Unavailable(f'Local engine port {port} is in use by {listening_process(port)}. '
                              f'Close that app or set {setting} to an unused port and restart the chat service.') from None
        raise Unavailable(f'Local engine port {port} is unavailable (error {error.errno}). '
                          f'Check {setting} and restart the chat service.') from None


# Control routes this controller answers itself (status `capabilities`); the panel asks
# for a route only when it is listed.
PRELOAD_TIMEOUT = 5.0  # seconds; the engine answers 202 at once and loads in the background
_log = logging.getLogger('local_ai.controller')
CAPABILITIES = ('models', 'storage', 'harness', 'harness_learning', 'ledger', 'active', 'openai_aliases', 'state_cache')


# Opt-in native CPU repack cache (patches/llama-repack-cache.patch). The
# controller only supplies identity and surfaces ERROR lines; it never scans or
# prunes the cache. After an engine or model update, prune explicitly with the
# patch's tool, e.g. `tools/repack_cache.py prune --dir <cache-dir> --keep-latest`.
REPACK_CACHE_ENV = 'GGML_CPU_REPACK_CACHE'
REPACK_ERROR_PREFIX = b'ggml-repack-cache: ERROR'
REPACK_ERROR_LOG = Path(os.environ.get('LOCAL_AI_REPACK_ERROR_LOG', str(
    Path(os.environ.get('XDG_STATE_HOME', str(Path.home() / '.local/state'))) / 'local-ai/repack-cache-errors.log')))
REPACK_ERROR_LINE_CHARS = 512
# Harness prefix registry (which system-prompt + tools prefixes were seen, and primers to
# re-warm them). LOCAL_AI_HARNESS_STATE='' keeps it in memory only (no primers, no prepare).
HARNESS_STATE = os.environ.get('LOCAL_AI_HARNESS_STATE', str(
    Path(os.environ.get('XDG_STATE_HOME', str(Path.home() / '.local/state'))) / 'local-ai/harness-prefixes'))
REPACK_ERROR_LINES_PER_LOAD = 64
REPACK_ERROR_LOG_BYTES = 256 * 1024
REPACK_ERROR_LOG_KEEP = 2
SHA256_RE = re.compile('[0-9a-f]{64}')
ENGINE_DIAGNOSTIC_LOG = Path(os.environ.get('XDG_STATE_HOME', str(
    Path.home() / '.local/state'))) / 'local-ai/engine-stderr.log'
ENGINE_LEDGER_LOG = ENGINE_DIAGNOSTIC_LOG.with_name('engine-ledger.jsonl')


def repack_cache_enabled(environment):
    return REPACK_CACHE_ENV in environment


def repack_cache_identity(config):
    """Exact model/engine identity for the native repack cache, or None."""
    model_sha, engine_sha = config.get('sha256'), config.get('server_sha256')
    if not all(isinstance(value, str) and SHA256_RE.fullmatch(value) for value in (model_sha, engine_sha)):
        return None
    return {'GGML_CPU_REPACK_CACHE_MODEL_SHA256': model_sha, 'GGML_CPU_REPACK_CACHE_ENGINE_ID': engine_sha}


class RepackErrorReader:
    """Drain stderr; optionally tee it, while keeping only repack ERROR lines here.

    Without a tee, other stderr text is read and discarded. Runs until EOF of
    the child's pipe, so it remains the sole reader when a tee writer is used.
    """

    def __init__(self, stream, *, log_path=None, label='', notify=None, tee_writer=None):
        self.stream = stream
        self.log_path = Path(log_path or REPACK_ERROR_LOG)
        self.label = re.sub(r'[^A-Za-z0-9._-]', '_', str(label))[:64]
        self.notify = notify or notify_repack_errors
        self.tee_writer = tee_writer
        self.count = 0
        self.notified = False
        self.thread = threading.Thread(target=self.run, name='repack-cache-stderr', daemon=True)

    def start(self):
        self.thread.start()
        return self

    def join(self, timeout=5):
        self.thread.join(timeout)
        return not self.thread.is_alive()

    def run(self):
        at_line_start = True
        try:
            while chunk := self.stream.readline(8192):
                if self.tee_writer is not None:
                    self.tee_writer.write(chunk)
                candidate = at_line_start and chunk.startswith(REPACK_ERROR_PREFIX)
                at_line_start = chunk.endswith(b'\n')
                if candidate and self.count < REPACK_ERROR_LINES_PER_LOAD:
                    self.record(chunk)
        except (OSError, ValueError):
            pass
        finally:
            try: self.stream.close()
            except (OSError, ValueError): pass

    def record(self, raw):
        text = raw.decode('utf-8', 'replace').rstrip('\r\n')
        text = ''.join(c if c.isprintable() else '?' for c in text)[:REPACK_ERROR_LINE_CHARS]
        self.count += 1
        try:
            write_repack_error(self.log_path, f'{time.strftime("%Y-%m-%dT%H:%M:%S%z")} {self.label} {text}\n')
        except OSError:
            pass
        if not self.notified:
            self.notified = True
            self.notify(self.log_path)


def disable_repack_cache(environment, alias):
    """Load normally without the cache; record and announce it once."""
    for key in [key for key in environment if key.startswith(REPACK_CACHE_ENV)]:
        del environment[key]
    label = re.sub(r'[^A-Za-z0-9._-]', '_', str(alias))[:64]
    try:
        write_repack_error(REPACK_ERROR_LOG, f'{time.strftime("%Y-%m-%dT%H:%M:%S%z")} {label} '
                           f'local-ai: ERROR repack-cache-disabled missing-identity alias={label}\n')
    except OSError:
        pass
    notify_repack_errors(REPACK_ERROR_LOG)


def write_repack_error(path, line):
    path = Path(path)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    data = line.encode()
    try:
        size = os.lstat(path).st_size
    except FileNotFoundError:
        size = 0
    if size and size + len(data) > REPACK_ERROR_LOG_BYTES:
        for index in range(REPACK_ERROR_LOG_KEEP, 1, -1):
            older = path.with_name(f'{path.name}.{index - 1}')
            if older.exists(): os.replace(older, path.with_name(f'{path.name}.{index}'))
        os.replace(path, path.with_name(path.name + '.1'))
    descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        os.write(descriptor, data)
    finally:
        os.close(descriptor)


def notify_repack_errors(log_path):
    command = shutil.which('notify-send')
    if command is None: return
    try:
        subprocess.run([command, '--app-name=Local AI', 'Local AI: repack cache error',
                        f'The CPU repack cache reported an error during model load. Details: {log_path}'],
                       stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       timeout=5, check=False)
    except (OSError, subprocess.SubprocessError):
        pass


def request_defaults(target, payload, path):
    """Backend request defaults for the resolved target (qwen-local is already resolved).

    The classic beta's 2048 output default belongs to its llama.cpp engine only; an
    oxe-backed Ornith gets the oxe defaults (per-profile default_max_tokens) instead.
    Returns payload itself when no backend defaults apply.
    """
    if target.get('backend_kind') == oxe_runtime.KIND:
        return oxe_runtime.request_defaults(payload, path, target)
    if classic_beta.is_beta(target):
        return classic_beta.request_defaults(payload, path, alias=target['model_alias'])
    return payload


class Manager:
    def __init__(self):
        self.mu = threading.RLock()
        self.process = None
        self.lease = None
        self.active = 0
        self.inferences = 0
        self.last_used = 0
        self.state = 'unloaded'
        self.cache = {}
        self.flash_cache_status = None
        self.loaded_config = None
        self.state_engine = False  # the loaded oxe engine owns the disk state store (--state-cache)
        self.state_preload = False  # ... and serves POST /state-cache/preload (P2b)
        self.launches = 0  # engines that reached ready; an explicit load preloads only its own launch
        self.session_hints = {}  # model alias -> last X-Oxe-Session hint forwarded to it
        self.prefix_store = None
        # FIFO admission of inference requests (see begin_queued)
        self.queue_cv = threading.Condition(threading.Lock())
        self.queue_waiters = collections.deque()
        self.queue_reserved = False
        self.backend_runtime = None
        self.classic_memory_guard = None
        self.flash_power_guard = None
        self.repack_reader = None
        self.engine_reader = None
        self.engine_log_writer = None
        self.engine_ledger = None
        # Optional Flash idle-QoS hook (root unit, see oxe_runtime.flash_idle_qos_unit); best effort.
        self.flash_idle_qos_pid = None
        self.flash_idle_qos_unit = None
        self.flash_idle_qos_started = False
        self.flash_idle_qos_error = None
        self.engine_death_lock = threading.Lock()
        self.last_memory_guard = None
        # M1: {reason, kind, alias, origin, t, detail?}; cleared on a successful start and on
        # every new explicit load. Lab-origin entries never reach the user-facing last_error.
        self.last_failure_info = None
        self.activity_origin = 'organic'
        self.stop_pending = None  # S1: set while an engine's exit is unproven (lease kept)
        self._start_attempt = False
        self.current_prefix_key = None
        self.prefix_stats = {'restores': 0, 'primes': 0, 'failures': 0}
        self.harness_registry = HarnessRegistry(HARNESS_STATE or None)
        self.compat_stats = collections.Counter()
        self.answering_since = None
        self.image_active = False
        self.image_broker = None
        self.image_broker_error = None
        self.harness_mu = threading.Lock()  # registry only; never held across a backend call
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.reaper, daemon=True)
        self.thread.start()
        if os.environ.get('LOCAL_AI_IMAGE_NPU') == '1':
            path = Path(os.environ.get('LOCAL_AI_IMAGE_SOCKET', str(Path.home() / '.local/state/local-ai/image-slot/permit.sock')))
            try:
                # Explicit opt-in: only the exact value "1" kills the stale client; anything else is off.
                kill_client = os.environ.get('LOCAL_AI_IMAGE_WATCHDOG_KILL_CLIENT') == '1'
                self.image_broker = image_slot.Broker(path, self.reserve_image, kill_client=kill_client).start()
            except (OSError, image_slot.SlotUnavailable):
                self.image_broker_error = 'Image slot could not be started; NPU images remain unavailable.'

    @contextmanager
    def reserve_image(self, profile, peer_pid):
        """Delegate the existing lease while keeping the chat model resident.

        The image worker inherits the lease descriptor and control socket. Chat
        requests use the existing FIFO until that worker has finished. Images
        do not jump ahead of queued chat requests or unload a resident model.
        """
        if os.environ.get('LOCAL_AI_IMAGE_NPU') != '1' or profile != 'klein-q8-512-npu':
            raise Unavailable('NPU image slots are not enabled.')
        owned_lease = None
        activated = False
        reserved = False
        with self.queue_cv:
            if self.queue_reserved or self.queue_waiters or self.inferences:
                raise Unavailable('Chat is busy; the image slot is unavailable.')
            self.queue_reserved = True
            reserved = True
        try:
            with self.mu:
                if self.active or self.image_active:
                    raise Unavailable('A model operation is already active.')
                if self.process is not None and self.process.poll() is not None:
                    raise Unavailable('The resident engine needs recovery before an image.')
                lease = self.lease
                if lease is None:
                    if self.process is not None:
                        raise Unavailable('Resident model has no canonical lease.')
                    guard_production_path(LEASE)
                    LEASE.parent.mkdir(parents=True, exist_ok=True)
                    guard_production_path(LEASE)
                    fd = os.open(LEASE, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
                    owned_lease = lease = os.fdopen(fd, 'a+')
                    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                resident = resident_bytes(self.process.pid) if self.process is not None else None
                if self.process is not None and resident is None:
                    raise Unavailable('Resident-model memory telemetry is unavailable.')
                choice = image_npu.choose(enabled=True, ready=True,
                    latched=(Path.home() / '.local/state/local-ai/NPU-STOP').exists(),
                    mem_available=available_memory_bytes(), mem_total=total_memory_bytes(),
                    resident_model_bytes=resident, swap_free=swap_bytes()[1])
                metadata = {'backend': choice.backend, 'reason': choice.reason,
                            'required_bytes': choice.required_bytes, 'chat_retained': self.process is not None}
                self.image_active = True
                self.active += 1
                self.inferences += 1
                activated = True
            with self.queue_cv:
                self.queue_reserved = False
                reserved = False
                self.queue_cv.notify_all()
            yield lease.fileno(), metadata
        finally:
            if owned_lease is not None:
                owned_lease.close()
            if activated:
                with self.mu:
                    self.image_active = False
                self.end(inference=True, touch=True)
            with self.queue_cv:
                if reserved:
                    self.queue_reserved = False
                self.queue_cv.notify_all()

    def _engine_writer(self):
        """Create the fixed-path sink only when an engine is actually starting."""
        if self.engine_log_writer is None:
            self.engine_log_writer = EngineLogWriter(ENGINE_DIAGNOSTIC_LOG)
        return self.engine_log_writer

    def record_engine_death(self, process, config=None, guard_failure=None):
        """Record a native child exit once, without including request data."""
        if process is None:
            return False
        with self.engine_death_lock:
            exit_code = process.poll()
            if exit_code is None or vars(process).get('_local_ai_engine_death_logged', False):
                return False
            try:
                process._local_ai_engine_death_logged = True
            except (AttributeError, TypeError):
                pass
            if guard_failure is None and process is self.process:
                if self.backend_runtime is not None:
                    guard_failure = self.backend_runtime.guard_failure()
                elif self.classic_memory_guard is not None:
                    guard_failure = self.classic_memory_guard.failure()
            config = config or (self.loaded_config if process is self.process else None) or {}
            event = {
                'event': 'engine_died',
                'pid': getattr(process, 'pid', None),
                'exit_code': exit_code,
                'model_alias': config.get('model_alias'),
                'engine_sha256': config.get('server_sha256'),
            }
            if guard_failure:
                event['guard_failure'] = str(guard_failure)
            writer = self.engine_log_writer
            if writer is not None:
                writer.write(json.dumps(event, sort_keys=True) + '\n')
            return True

    def config(self, alias=None):
        try:
            value = json.loads(CONFIG.read_text())
            if not isinstance(value, dict): raise ValueError()
            base_alias = value.get('model_alias', 'qwen-local')
            profiles = value.get('profiles', {})
            selected = value.get('selected_model', base_alias)
            # local-default / local-heavy / openai/<name>: static names only (openai_api.py)
            alias = openai_api.resolve_model(alias, value)
            requested = selected if alias in (None, '', 'qwen-local') else alias
            if requested != base_alias and requested not in profiles:
                if requested in classic_beta.RETIRED_ALIASES:
                    raise Unavailable('This local model has been retired.')
                if profiles: raise Unavailable('Unknown local model. Choose one listed in /v1/models.')
                requested = base_alias
            selected_profile = profiles.get(requested, {})
            merged = {**value, **selected_profile, 'model_alias': requested}
            # Unlike legacy shared defaults, execution overrides are per model.
            if requested != base_alias and 'memlat_overrides' not in selected_profile:
                merged.pop('memlat_overrides', None)
            return profile_policy.resolve(merged)
        except (OSError, ValueError):
            raise Unavailable('Local model setup is still being prepared.')

    def eligible(self, config):
        if classic_beta.is_retired(config): return False
        if config.get('backend_kind') == oxe_runtime.KIND:
            return oxe_runtime.eligible(config)
        if classic_beta.is_beta(config):
            return classic_beta.eligible(config)
        if config.get('backend_kind') not in (None, 'llama', bigmoe_runtime.KIND): return False
        if config.get('backend_kind') == bigmoe_runtime.KIND:
            return bigmoe_runtime.eligible(config)
        return bool(config.get('ready'))

    def models(self):
        value = json.loads(CONFIG.read_text())
        aliases = [value.get('model_alias', 'qwen-local'), *value.get('profiles', {})]
        if value.get('profiles'): aliases.insert(0, 'qwen-local')
        rows = [{'id': alias, 'object': 'model', 'created': 0, 'owned_by': 'local',
                 'name': model_catalog.SELECTED_ALIAS_NAME if alias == 'qwen-local' else model_catalog.display_name(alias, self.config(alias))} for alias in dict.fromkeys(aliases)
                if self.eligible(self.config(alias))]
        models = openai_api.model_rows(value, rows, value.get('selected_model', value.get('model_alias', 'qwen-local')))
        if (os.environ.get(explicit_route.ENABLE_ENV) == '1'
                and explicit_route.TARGETS['code'] in {entry['id'] for entry in rows}):
            models.append({'id': explicit_route.ALIAS, 'object': 'model', 'created': 0,
                           'owned_by': 'local', 'name': 'Local explicit route (code / think-hard)'})
        return models

    def catalog(self):
        """Chooser rows for eligible profiles; metadata only, never loads a model."""
        value = json.loads(CONFIG.read_text())
        if not isinstance(value, dict): raise ValueError('runtime_object_required')
        base = value.get('model_alias', 'qwen-local')
        profiles = value.get('profiles') if isinstance(value.get('profiles'), dict) else {}
        selected = self.config().get('model_alias')
        # Unlocked snapshot: status polling must not wait for a model load.
        loaded, state = self.loaded_config, self.state
        rows = []
        for alias in dict.fromkeys([base, *profiles]):
            try: config = self.config(alias)
            except Unavailable: continue
            if not self.eligible(config): continue
            raw = profiles.get(alias, value if alias == base else {})
            rows.append(model_catalog.row(alias, raw, config, selected=selected,
                                          loaded_alias=loaded.get('model_alias') if loaded else None, state=state,
                                          flash_cache=(getattr(self, 'flash_cache_status', None) if loaded and
                                              loaded.get('model_alias') == alias else None)))
        rows.sort(key=model_catalog.sort_key)
        return {'object': 'list', 'default_model': model_catalog.DEFAULT_ALIAS,
                'selected_model': selected, 'data': rows}

    def flash_qualification(self):
        """M3: Flash's derived status (model_catalog.qualification of the resolved installed
        profile), or None when there is no Flash profile or it is not on the Rust engine."""
        try:
            config = self.config(model_catalog.FLASH_ALIAS)
        except Unavailable:
            return None
        if config.get('model_alias') != model_catalog.FLASH_ALIAS or not oxe_runtime.is_flash(config):
            return None
        return model_catalog.qualification(config)

    def flash_cache_setting(self, value=None):
        """Opt-in Flash preference only; config reads never start or contact a model."""
        if value is None:
            value = json.loads(CONFIG.read_text())
        profile = value.get('profiles', {}).get('qwen3.8-flash-next-local', {})
        flash = profile_policy.rust_overrides(profile)[1].get('oxe_flash', {})
        if not isinstance(flash, dict) or flash.get('cache_pref') is None:
            return {'enabled': False}
        if not oxe_runtime.is_flash(self.config('qwen3.8-flash-next-local')):
            return {'enabled': False}
        pref = oxe_runtime.flash_cache_pref
        # S2: no 20/18 on a host without swap (the MemTotal rule would page the desktop to nowhere).
        return {'enabled': True, 'pref': pref.read_pref(os.environ, flash), 'choices': pref.offered_prefs(swap_bytes()[0]),
                'desktop_floor_gib': pref.desktop_floor(flash),
                'editable': not bool(os.environ.get(pref.ENV, '').strip())}

    def set_flash_cache_pref(self, preference):
        """Persist one enabled preference while idle; never enable a profile or change its guards."""
        if not isinstance(preference, str) or preference not in oxe_runtime.flash_cache_pref.PREFS:
            raise ValueError('Choose auto, 20, 18, 16 or 14.')
        if not self.mu.acquire(timeout=2):
            raise Unavailable('A model is loading; change the preference when it is idle.')
        temporary = None
        try:
            if self.active or self.queue_length():
                raise Unavailable('A request is running or queued.')
            value = json.loads(CONFIG.read_text())
            setting = self.flash_cache_setting(value)
            if not setting['enabled']:
                raise Unavailable('Flash cache preferences are not enabled for this profile.')
            if not setting['editable']:
                raise Unavailable('The environment overrides this preference; remove that override first.')
            if preference not in setting['choices']:
                raise ValueError('20 and 18 GiB need swap on this host; choose auto, 16 or 14.')
            profile = value['profiles']['qwen3.8-flash-next-local']
            flash = profile[profile_policy.rust_overrides(profile)[0]]['oxe_flash']
            if flash['cache_pref'] == preference:
                return setting
            guard_production_path(CONFIG)
            flash['cache_pref'] = preference
            with tempfile.NamedTemporaryFile(mode='w', prefix='.flash-cache-', suffix='.tmp',
                                             dir=CONFIG.parent, delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(json.dumps(value, indent=2) + '\n')
                stream.flush(); os.fsync(stream.fileno())
            if self.loaded_config and self.loaded_config.get('model_alias') == 'qwen3.8-flash-next-local':
                self.cleanup()
            guard_production_path(CONFIG)
            os.replace(temporary, CONFIG)
            temporary = None
            return {**setting, 'pref': preference}
        finally:
            try:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
            finally:
                self.mu.release()

    @contextmanager
    def held_lease(self):
        """The canonical inference lease for a cache mutation, held for the whole block: the
        loaded model's (this controller already owns it) or taken here, non-blocking. Call with
        mu held. Unavailable when another image, chat or setup job holds it."""
        if self.lease is not None:
            yield  # a loaded model already owns the lease
            return
        guard_production_path(LEASE)
        LEASE.parent.mkdir(parents=True, exist_ok=True)
        with open_lease_append() as lease:
            try:
                fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise Unavailable('Another image, chat or setup job is running.') from None
            yield

    def delete_storage(self, entry_id):
        """Delete one unreferenced cache under the inference lease."""
        with self.mu:
            if self.active: raise Unavailable('A request is running.')
            guard_production_path(CONFIG)
            document = json.loads(CONFIG.read_text())
            # With a loaded model the lease is already ours; in-use checks still apply.
            with self.held_lease():
                return storage_view.delete(document, entry_id)

    def select(self, alias):
        if not isinstance(alias, str): raise Unavailable('Choose a local model.')
        if not self.mu.acquire(blocking=False): raise Unavailable('A model is loading or a request is running.')
        try:
            if self.active: raise Unavailable('A request is running; model selection was not changed.')
            chosen_config = self.config(alias)
            if classic_beta.is_retired(chosen_config):
                raise Unavailable('This local model has been retired.')
            if classic_beta.is_beta(chosen_config) and not self.eligible(chosen_config):
                raise Unavailable('This profile needs its reviewed quality and current model identity proof.')
            if chosen_config.get('backend_kind') == oxe_runtime.KIND and not self.eligible(chosen_config):
                raise Unavailable(oxe_runtime.ineligible_reason(chosen_config) or 'The Rust engine is not enabled and ready.')
            if chosen_config.get('backend_kind') == bigmoe_runtime.KIND and not self.eligible(chosen_config):
                raise Unavailable('Flash profile has not passed native qualification and RAM admission.')
            chosen = chosen_config['model_alias']
            value = json.loads(CONFIG.read_text())
            if chosen != value.get('model_alias') and chosen not in value.get('profiles', {}):
                raise Unavailable('Selected model is not a registered local profile.')
            guard_production_path(CONFIG)
            if self.loaded_config and self.loaded_config['model_alias'] != chosen: self.cleanup()
            value['selected_model'] = chosen
            guard_production_path(CONFIG)
            temporary = CONFIG.with_suffix('.select.tmp')
            temporary.write_text(json.dumps(value, indent=2) + '\n'); temporary.replace(CONFIG)
            self.cache.pop('/props', None)
            return chosen
        finally: self.mu.release()

    def _flash_idle_qos(self, verb, config, pid):
        """The only place that runs systemctl for the Flash idle-QoS hook; returns (unit, started, error).

        Best effort and never raises: a denied, missing or slow unit is reported in /_local/status and
        never fails or blocks a model load or unload. unit is None when the profile has no idle_qos.
        """
        unit = None
        try:
            unit = oxe_runtime.flash_idle_qos_unit(config, pid)
            command = oxe_runtime.flash_idle_qos_command(config, pid, verb)
            if command is None:
                return None, False, None
            done = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                  stderr=subprocess.DEVNULL, timeout=5, check=False)
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            return unit, False, type(error).__name__
        if done.returncode != 0:
            return unit, False, 'CalledProcessError'
        return unit, True, None

    def cleanup(self):
        """Stop the loaded engine and release everything it held.

        S1 (Codex review): every teardown step runs even when an earlier one raises, and the
        outermost finally always makes sure the child's process group is dead and then releases
        the lease and the loaded state. The first error is re-raised after that.
        """
        errors = []
        state = {'guard_failure': None, 'process': self.process}

        def step(function):
            try:
                function()
            except Exception as error:  # keep going; re-raised after the lease is released
                errors.append(error)

        try:
            runtime = self.backend_runtime

            def read_guard_failure():
                guard_failure = runtime.guard_failure() if runtime is not None else None
                if self.classic_memory_guard is not None:
                    guard_failure = self.classic_memory_guard.failure() or guard_failure
                state['guard_failure'] = guard_failure
            step(read_guard_failure)
            guard_failure = state['guard_failure']
            process, self.process = self.process, None

            def stop_process():
                died_before_cleanup = process.poll() is not None
                if not died_before_cleanup:
                    retry = self.stop_pending is not None
                    if not retry:  # first attempt: the engine gets its SIGTERM grace
                        try:
                            os.killpg(process.pid, signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                    try:
                        if retry:
                            raise subprocess.TimeoutExpired('engine', 0)
                        process.wait(timeout=TERM_WAIT_SECONDS)
                    except subprocess.TimeoutExpired:
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        # Bounded (Codex review 3): an engine stuck in D-state may not die at once;
                        # a timeout leaves it 'stopping' and the reaper retries.
                        process.wait(timeout=KILL_WAIT_SECONDS)
                else:
                    process.wait()
                state['process'] = None  # reaped; the finally below has nothing left to kill
                if died_before_cleanup or guard_failure:
                    self.record_engine_death(process, self.loaded_config, guard_failure)
            if process is not None:
                step(stop_process)
                step(lambda: oxe_runtime.remove_api_key(getattr(process, '_local_ai_api_key_file', None)))

            def stop_idle_qos():
                if self.flash_idle_qos_pid is not None:
                    pid, self.flash_idle_qos_pid = self.flash_idle_qos_pid, None
                    _, _, self.flash_idle_qos_error = self._flash_idle_qos('stop', self.loaded_config or {}, pid)
                    self.flash_idle_qos_unit, self.flash_idle_qos_started = None, False
            step(stop_idle_qos)
            reader, self.repack_reader = self.repack_reader, None
            if reader is not None:
                # The child (and its process group) is gone, so stderr reaches EOF.
                step(lambda: reader.join(timeout=5))
            engine_reader, self.engine_reader = self.engine_reader, None
            if engine_reader is not None:
                step(lambda: engine_reader.stop(timeout=5))
            ledger, self.engine_ledger = self.engine_ledger, None
            if ledger is not None:
                step(lambda: ledger.stop(timeout=5))
            alias = (self.loaded_config or {}).get('model_alias')
            reasons = []
            if runtime is not None:
                def stop_runtime_guard():
                    try:
                        reasons.append(runtime.guard_failure() or guard_failure)
                    finally:
                        runtime.stop_guard()
                step(stop_runtime_guard)
            guard, self.classic_memory_guard = self.classic_memory_guard, None
            if guard is not None:
                def stop_classic_guard():
                    guard.stop()
                    reasons.append(guard.failure() or guard_failure)
                    self.last_memory_guard = guard.metadata()
                step(stop_classic_guard)
            flash_guard, self.flash_power_guard = self.flash_power_guard, None
            if flash_guard is not None:
                def stop_flash_guard():
                    flash_guard.stop()
                    reasons.append(flash_guard.failure)
                step(stop_flash_guard)
            stop_reason = next((reason for reason in reversed(reasons) if reason), None)
            if stop_reason:
                step(lambda: self.record_failure(stop_reason, kind='guard', alias=alias))
        finally:
            # The engine must be gone before the lease is released, or another job could start
            # beside it. Codex review (round 2): a failed signal or a wait timeout is not proof of
            # exit; then the controller keeps the process, the lease and the loaded config, latches
            # state 'stopping' and retries from the reaper and from every new begin().
            leftover = state['process']
            try:
                if leftover is not None and leftover.poll() is None:
                    try:
                        os.killpg(leftover.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    leftover.wait(timeout=KILL_WAIT_SECONDS)
            except Exception as error:
                errors.append(error)
            unconfirmed = False
            try:
                unconfirmed = leftover is not None and leftover.poll() is None
            except Exception as error:  # cannot even ask: not proof of exit either
                errors.append(error)
                unconfirmed = True
            if unconfirmed:
                self.process = leftover
                self.state = 'stopping'
                self.stop_pending = (f'engine pid {getattr(leftover, "pid", "?")} has not exited: '
                                     + (unexpected_cause(errors[-1]) if errors else 'still running'))
                print(f'local-ai: {self.stop_pending}; lease kept, retrying', file=sys.stderr, flush=True)
            else:
                self.stop_pending = None
                self.process = None
                if self.lease is not None:
                    try:
                        self.lease.close()
                    except Exception as error:
                        errors.append(error)
                    finally:
                        self.lease = None
                self.state = 'unloaded'
                self.flash_cache_status = None
                self.loaded_config = None
                self.state_engine = False
                self.state_preload = False
                self.current_prefix_key = None
                self.prefix_store = None
                self.backend_runtime = None
        if errors:
            raise errors[0]

    def record_failure(self, reason, *, kind, alias=None, origin=None, detail=None):
        """M1: remember why the last start failed or a guard stopped a model; call with self.mu held."""
        if not reason:
            return
        info = {'reason': str(reason), 'kind': kind, 'alias': alias,
                'origin': origin or self.activity_origin, 't': int(time.time())}
        if detail and str(detail) != info['reason']:
            info['detail'] = str(detail)
        self.last_failure_info = info

    def clear_failure(self):
        self.last_failure_info = None

    @property
    def last_failure(self):
        """The user-facing error string: None for lab-origin failures (REAL-1 header)."""
        info = self.last_failure_info
        if not info or info.get('origin') == LAB_ORIGIN:
            return None
        return info['reason']

    def reaper(self):
        while not self.stop_event.wait(1):
            try:
                self.reap_once()
            except Exception as error:  # S1: one bad pass must not end idle unload for good
                try:
                    print(f'local-ai: reaper pass failed: {type(error).__name__}', file=sys.stderr, flush=True)
                except Exception:
                    pass

    def reap_once(self):
        with self.mu:
            if self.stop_pending is not None:
                self.cleanup()  # S1: retry an unproven stop; the lease stays held until it succeeds
            elif self.process is not None:
                if ((self.classic_memory_guard is not None and self.classic_memory_guard.failure())
                        or (self.flash_power_guard is not None and self.flash_power_guard.failure)
                        or self.process.poll() is not None):
                    self.cleanup()
                elif self.active == 0 and time.monotonic() - self.last_used >= idle_seconds(self.loaded_config):
                    self.cleanup()

    def queue_length(self):
        with self.queue_cv:
            return len(self.queue_waiters)

    def begin_queued(self, *, inference=False, allow_start=False, model_alias=None, wait_seconds=None, origin=None):
        """begin(), but an inference request that finds another one running waits for its
        turn (first come, first served) for up to QUEUE_WAIT_SECONDS before failing."""
        if not inference:
            return self.begin(inference=False, allow_start=allow_start, model_alias=model_alias, origin=origin)
        entered = time.monotonic()
        wait = QUEUE_WAIT_SECONDS if wait_seconds is None else wait_seconds
        deadline = time.monotonic() + wait
        ticket = object()
        with self.queue_cv:
            self.queue_waiters.append(ticket)
            try:
                while not (self.queue_waiters[0] is ticket and not self.queue_reserved and self.inferences == 0):
                    left = deadline - time.monotonic()
                    if left <= 0:
                        raise Unavailable('A language-model request is already running and the wait '
                                          f'for a turn exceeded {wait:.0f} s; retry when it finishes.')
                    self.queue_cv.wait(min(left, 0.5))
                self.queue_reserved = True
            finally:
                self.queue_waiters.remove(ticket)
                self.queue_cv.notify_all()
        admitted = time.monotonic()
        try:
            process = self.begin(inference=True, allow_start=allow_start, model_alias=model_alias, origin=origin)
            # For the request ledger: (seconds queued, seconds in begin incl. a model load).
            # Only the admitted request reads it; the next admission needs this one to end.
            self.admission_times = (admitted - entered, time.monotonic() - admitted)
            self.answering_since = time.monotonic()
            return process
        finally:
            with self.queue_cv:
                self.queue_reserved = False
                self.queue_cv.notify_all()

    def begin(self, *, inference=False, allow_start=False, model_alias=None, origin=None, explicit_load=False):
        """Admit a request, loading the model when needed (see _begin).

        M1: a new explicit load clears the previous failure; a successful start clears it
        too. Only a failure after a start was really attempted is recorded, with its origin.
        """
        with self.mu:
            origin = origin or 'organic'
            if explicit_load:
                self.clear_failure()
            previous = self.last_failure_info
            self._start_attempt = False
            try:
                process = self._begin(inference=inference, allow_start=allow_start, model_alias=model_alias,
                                      origin=origin)
            except Exception as error:
                if not isinstance(error, Unavailable):
                    # S14: an unexpected error keeps its cause in the journal and in last_failure.
                    traceback.print_exc(file=sys.stderr)
                if self._start_attempt:
                    guard = self.last_failure_info if self.last_failure_info is not previous else None
                    reason = str(error) if isinstance(error, Unavailable) else 'Local model could not be started.'
                    detail = (guard or {}).get('reason')
                    if not isinstance(error, Unavailable):
                        detail = unexpected_cause(error)
                    self.record_failure(reason, kind='start', alias=model_alias, origin=origin, detail=detail)
                raise
            finally:
                self._start_attempt = False
            self.activity_origin = origin
            return process

    def _begin(self, *, inference=False, allow_start=False, model_alias=None, origin='organic'):
        with self.mu:
            if self.image_active:
                raise Unavailable('An image is running; the resident chat model is retained.')
            if self.stop_pending is not None:
                try:
                    self.cleanup()
                except Exception:
                    pass
                if self.stop_pending is not None:
                    raise Unavailable('The previous local engine has not exited yet; its memory and the '
                                      'inference lease stay reserved. Retry shortly.')
            if inference and self.inferences:
                raise Unavailable('A language-model request is already running; retry when it finishes.')
            if (self.classic_memory_guard is not None and self.classic_memory_guard.failure()
                    or self.process is not None and self.process.poll() is not None):
                self.cleanup()
            config = self.config(model_alias) if model_alias is not None or self.process is None else self.loaded_config
            transition = None
            if (os.environ.get(explicit_route.ENABLE_ENV) == '1'
                    and (self.process is None or config['model_alias'] != self.loaded_config['model_alias'])):
                transition = (self.loaded_config['model_alias'] if self.loaded_config else None,
                              config['model_alias'], time.monotonic())
            if classic_beta.is_retired(config):
                raise Unavailable('This local model has been retired.')
            # Retain the child's execution snapshot, but an explicit request for
            # a proof-gated profile must still respect fresh proof revocation.
            if model_alias is not None and classic_beta.is_beta(config) and not self.eligible(config):
                raise Unavailable('Model proof or identity changed; requalification is required.')
            if self.process is not None and config['model_alias'] == self.loaded_config['model_alias']:
                config = self.loaded_config
            if classic_beta.is_beta(config) and not self.eligible(config):
                raise Unavailable('Model proof or identity changed; requalification is required.')
            if self.process is not None and config['model_alias'] != self.loaded_config['model_alias']:
                if self.active: raise Unavailable('Another model request is running; retry when it finishes.')
                self.cleanup()
            if self.process is None:
                if config.get('backend_kind') not in (None, 'llama', bigmoe_runtime.KIND, oxe_runtime.KIND):
                    raise Unavailable('Unknown local backend kind.')
                if config.get('backend_kind') == oxe_runtime.KIND and not self.eligible(config):
                    raise Unavailable(oxe_runtime.ineligible_reason(config) or 'The Rust engine is not enabled and ready.')
                if not allow_start:
                    raise Unavailable('Model is sleeping; submitting a message will load it.')
                self._start_attempt = True
                self.activity_origin = origin  # a guard stop during loading belongs to this request
                if not config.get('ready'):
                    raise Unavailable('Qwen setup and validation are still running. Check /_local/status.')
                # Check before model files, RAM admission, scopes or token files.
                # A loaded engine is reused above and never probes its own port.
                streamed = config.get('backend_kind') == bigmoe_runtime.KIND
                require_backend_port(FLASH_BACKEND_PORT if streamed else BACKEND_PORT,
                                     'LOCAL_FLASH_BACKEND_PORT' if streamed else 'LOCAL_LLM_BACKEND_PORT')
                model = Path(config['model_path'])
                if not model.is_file():
                    raise Unavailable('The verified local model is not available yet.')
                if config.get('expected_bytes') is not None and model.stat().st_size != config['expected_bytes']:
                    raise Unavailable('The local model size does not match its verified profile.')
                available = available_memory_bytes()
                runtime = None
                flash_oxe = config.get('backend_kind') == oxe_runtime.KIND and oxe_runtime.is_flash(config)
                flash_swap_guard = None
                if config.get('backend_kind') == bigmoe_runtime.KIND:
                    try:
                        runtime = bigmoe_runtime.Runtime(config, FLASH_BACKEND_PORT)
                    except (ValueError, OSError, KeyError, TypeError):
                        raise Unavailable('Flash profile is unqualified or has insufficient admitted RAM.') from None
                    # Do not manufacture a second lock for a streamed backend.
                    guard_production_path(LEASE)
                    descriptor = os.open(LEASE, os.O_RDWR | os.O_NOFOLLOW)
                    lease = os.fdopen(descriptor, 'r+')
                else:
                    if flash_oxe:
                        if not oxe_runtime.flash_files_ready(config):
                            raise Unavailable('Flash model shards or the expert sidecar are missing or changed.')
                        if oxe_runtime.flash_power_failure(bigmoe_runtime.power_snapshot(), starting=True):
                            raise Unavailable('Flash needs at least 30% battery to start and stops at 20%.')
                        try:
                            # A smaller-cache fallback, when configured, replaces the launch snapshot.
                            # With a flash_memory policy the engine's limit is its own scope's.
                            scoped = isinstance(config.get('flash_memory'), dict)
                            pref_enabled = oxe_runtime.flash_cache_pref.enabled(os.environ, config['oxe_flash'])
                            mem_total = total_memory_bytes() if pref_enabled else None
                            # S2: the MemTotal rule for 20/18 also needs swap for what must leave RAM.
                            swap_free = swap_bytes()[1] if pref_enabled else None
                            config = oxe_runtime.flash_select(
                                config, available, oxe_runtime.flash_limits(config)[1] if scoped else cgroup_memory_max(),
                                env=os.environ, mem_total=mem_total, swap_free=swap_free)
                            # F127: opt-in swap-tolerant guard, only for a preferred 20/18 on the MemTotal rule.
                            flash_swap_guard = oxe_runtime.flash_swap_tolerant_guard(
                                config, available, mem_total, env=os.environ, swap_free=swap_free)
                        except ValueError:
                            if oxe_runtime.flash_cache_pref.enabled(os.environ, config['oxe_flash']):
                                raise Unavailable('Flash cache preference cannot fit the current native memory budget; '
                                                  'the host floor and runtime pressure guards remain active.') from None
                            raise Unavailable(f'Flash (Rust) needs about {oxe_runtime.flash_smallest_required_available(config) / 1024**3:.1f} '
                                              'GiB of available RAM; close other large apps and retry.') from None
                    else:
                        required = model.stat().st_size + 3 * 1024**3
                        if available is not None and available < required:
                            raise Unavailable(f'This model needs about {required / 1024**3:.1f} GiB of available RAM; '
                                              f'{available / 1024**3:.1f} GiB is available. Close other large apps and retry.')
                    guard_production_path(LEASE)
                    LEASE.parent.mkdir(parents=True, exist_ok=True)
                    guard_production_path(LEASE)
                    # N6: never follow a symlink planted at the lease path (as at the other sites).
                    lease = open_lease_append()
                try:
                    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    lease.close()
                    raise Unavailable('Another image, chat or setup job is running. Please try again when it finishes.')
                self.lease = lease
                self.state = 'loading'
                self.loaded_config = config
                self.backend_runtime = runtime
                self.cache.pop('/props', None)
                try:
                    # An alias switched to Rust is pinned by its Rust engine
                    # hash, never by the classic Ornith engine allowlist.
                    if config.get('backend_kind') == oxe_runtime.KIND:
                        try:
                            oxe_runtime.verify_engine(config)
                        except FileNotFoundError:
                            # M4: a missing Rust binary fails loudly and names the fallback switch.
                            raise Unavailable(oxe_runtime.ineligible_reason(config)
                                              or 'The Rust engine binary is missing.') from None
                    elif classic_beta.is_beta(config):
                        classic_beta.verify_engine(config)
                    args = []
                    if config.get('backend_kind') == oxe_runtime.KIND:
                        try:
                            args = oxe_runtime.command(config, BACKEND_PORT)
                        except ValueError as error:
                            if str(error) != 'state_cache_root_override_invalid': raise
                            raise Unavailable(f'{state_cache.ROOT_ENV} (or runtime.json cache.root) must be an absolute '
                                              'path inside your home; the model was not started.') from None
                        self.state_engine = '--state-cache' in args
                        self.state_preload = (self.state_engine and
                                              oxe_runtime.engine_has_state_preload(config.get('server_binary')))
                    elif runtime is None:
                        args = [str(config['server_binary']), '--model', str(model),
                            '--alias', config.get('model_alias', 'qwen-local'),
                            '--host', '127.0.0.1', '--port', str(BACKEND_PORT),
                            '--ctx-size', str(int(config.get('context_size', 8192))),
                            '--threads', str(int(config.get('threads', 10))),
                            '--n-gpu-layers', str(int(config.get('gpu_layers', 0))),
                            *config.get('extra_args', [])]
                    cache_config = config.get('prefix_cache', {})
                    # Flash only: its own memory scope and host floor. Every other model stays
                    # inside this unit's limit with the default floor.
                    try:
                        flash_memory = memory_policy.flash_policy(config)
                    except ValueError:
                        raise Unavailable('The Flash memory policy in this profile is not a qualified one.') from None
                    if flash_memory is not None and runtime is None and not flash_oxe:
                        raise Unavailable('A Flash memory policy is set on a model that is not Flash.')
                    if runtime is not None:
                        runtime.require_memory(available_memory_bytes(), lease_owned=self.lease is not None)
                        args = runtime.prepare(lease_owned=self.lease is not None)
                        if flash_memory is not None:
                            args = memory_policy.scope_command(args, flash_memory['scope_max_bytes'])
                    elif flash_oxe and flash_memory is not None:
                        args = memory_policy.scope_command(args, flash_memory['scope_max_bytes'])
                    elif cache_config.get('enabled'):
                        self.prefix_store = PrefixStore(cache_config['directory'],
                            cache_config.get('max_bytes', 3 * 1024**3))
                        args += ['--slots', '--slot-save-path', str(self.prefix_store.root)]
                    environment = os.environ.copy()
                    if runtime is not None:
                        environment = runtime.environment(environment)
                    else:
                        if config.get('backend_kind') == oxe_runtime.KIND:
                            environment = oxe_runtime.launch_environment(environment)
                        environment.update(config.get('environment', {}))
                    backend_key = backend_key_file = None
                    if runtime is None and config.get('backend_kind') == oxe_runtime.KIND:
                        # B1: per-launch bearer token; set after the profile environment so no
                        # profile can replace or drop it. The proxy and status calls send it.
                        try:
                            backend_key, backend_key_file = oxe_runtime.new_api_key(BACKEND_PORT)
                        except OSError:
                            raise Unavailable('Local model cannot start: no private runtime directory for its access token.') from None
                        environment[oxe_runtime.API_KEY_ENV] = str(backend_key_file)
                    repack_cache = (runtime is None and config.get('backend_kind') != oxe_runtime.KIND
                                    and repack_cache_enabled(environment))
                    if repack_cache:
                        identity = repack_cache_identity(config)
                        if identity is None:
                            # Never let the native cache fall back to its weaker
                            # default identity: load without it, while still capturing engine stderr.
                            disable_repack_cache(environment, config.get('model_alias', ''))
                            repack_cache = False
                        else:
                            environment.update(identity)
                    if runtime is None:
                        try:
                            scoped_flash = ({'minimum_available_bytes': flash_memory['host_floor_bytes'],
                                             'follow_process_cgroup': True}
                                            if flash_oxe and flash_memory is not None else {})
                            # O-39 replaces the old host floor/F127 stop mode;
                            # admission and the engine's prefill settings are unchanged.
                            self.classic_memory_guard = memory_policy.MemoryGuard(
                                loading_timeout_seconds=min(300.0, START_SECONDS),
                                host_guard_version=memory_policy.GUARD_2_VERSION, **scoped_flash)
                        except memory_policy.MemoryPolicyError as error:
                            self.record_failure(error, kind='start', alias=config.get('model_alias'), origin=origin)
                            self.last_memory_guard = {'reason': str(error)}
                            raise Unavailable('Local model cannot load under the current memory conditions.') from None
                    engine_writer = self._engine_writer()
                    launch_options = {}
                    if engine_ledger.enabled(config, args):
                        try:
                            self.engine_ledger = engine_ledger.NativeLedger(ENGINE_LEDGER_LOG, config)
                        except (OSError, ValueError):
                            engine_writer.write('{"event":"ledger_disabled","reason":"log_unavailable"}\n')
                        else:
                            args += self.engine_ledger.arguments
                            launch_options['pass_fds'] = self.engine_ledger.pass_fds
                    try:
                        self.process = subprocess.Popen(args, stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE, start_new_session=True, env=environment, umask=0o077,
                            **launch_options)
                    except BaseException:
                        oxe_runtime.remove_api_key(backend_key_file)
                        raise
                    self.process._local_ai_engine_config = config
                    self.process._local_ai_api_key = backend_key
                    self.process._local_ai_api_key_file = backend_key_file
                    if self.engine_ledger is not None:
                        self.engine_ledger.start()
                    if repack_cache:
                        self.repack_reader = RepackErrorReader(self.process.stderr,
                            label=config.get('model_alias', ''), tee_writer=engine_writer).start()
                    else:
                        self.engine_reader = EngineStderrReader(self.process.stderr, engine_writer)
                        self.engine_reader.start()
                    if runtime is not None:
                        if flash_memory is not None:
                            runtime.start_guard(self.process, available_memory_bytes, lease_owned=self.lease is not None,
                                                reserve_bytes=flash_memory['host_floor_bytes'])
                        else:
                            runtime.start_guard(self.process, available_memory_bytes, lease_owned=self.lease is not None)
                    else:
                        self.classic_memory_guard.start(self.process)
                        if flash_oxe:
                            self.flash_power_guard = oxe_runtime.FlashPowerGuard(self.process)
                            self.flash_power_guard.start()
                    # The Rust Flash server prefaults a 20 GiB cache before it reports ready.
                    deadline = time.monotonic() + (max(START_SECONDS, 300.0) if flash_oxe else START_SECONDS)
                    while time.monotonic() < deadline:
                        if self.classic_memory_guard is not None and self.classic_memory_guard.failure():
                            raise Unavailable('Local model stopped to preserve system memory.')
                        if self.flash_power_guard is not None and self.flash_power_guard.failure:
                            raise Unavailable('Flash stopped to preserve power.')
                        if runtime is not None and runtime.guard_failure():
                            raise Unavailable('Flash stopped to preserve power or available memory.')
                        if self.process.poll() is not None:
                            raise Unavailable('Local model backend exited during loading. Setup diagnostics are in the LLM project.')
                        if runtime is not None:
                            if runtime.readiness(lease_owned=self.lease is not None).ready:
                                runtime.verify_child(self.process)
                                break
                        else:
                            connection = http.client.HTTPConnection('127.0.0.1', BACKEND_PORT, timeout=2)
                            try:
                                connection.request('GET', '/health', headers=oxe_runtime.backend_headers(backend_key))
                                response = connection.getresponse()
                                health_body = response.read()
                                classic = config.get('backend_kind') != oxe_runtime.KIND
                                if (response.status == 200 and classic
                                        and port_listener_in_group(BACKEND_PORT, self.process.pid) is False):
                                    # S6: a 200 from a process outside our child's group is not our engine
                                    # (oxe is pinned by its per-launch bearer token instead).
                                    raise Unavailable(f'Local engine port {BACKEND_PORT} is answered by '
                                                      f'{listening_process(BACKEND_PORT)}, not the engine this '
                                                      'controller started. Close that app or set '
                                                      'LOCAL_LLM_BACKEND_PORT to an unused port.')
                                if response.status == 200 and (classic or oxe_runtime.health_ready(health_body)):
                                    self.classic_memory_guard.mark_ready()
                                    if self.classic_memory_guard.failure():
                                        raise Unavailable('Local model stopped to preserve system memory.')
                                    break
                            except (OSError, http.client.HTTPException):
                                pass
                            finally:
                                connection.close()
                        time.sleep(.25)
                    else:
                        raise Unavailable('Local model loading timed out; no generation was started.')
                    self.last_used = time.monotonic()
                    self.state = 'ready'
                    self.launches += 1
                    self.clear_failure()  # M1: a successful start supersedes any older failure
                    if transition is not None:
                        engine_ledger.record_transition(ENGINE_LEDGER_LOG, transition[0], transition[1],
                                                        (time.monotonic() - transition[2]) * 1000, success=True)
                    # Idle-QoS hook: only now, with the engine reporting ready. The stop is in cleanup().
                    self.flash_idle_qos_pid = self.process.pid
                    self.flash_idle_qos_unit, self.flash_idle_qos_started, self.flash_idle_qos_error = \
                        self._flash_idle_qos('start', config, self.process.pid)
                    # Preserve llama.cpp's complete model metadata for its UI,
                    # including architecture/context fields; polling stays cheap
                    # after idle unload and never reopens the model.
                    if runtime is not None:
                        self.active += 1
                        if inference: self.inferences += 1
                        return self.process
                    if flash_oxe:
                        # Capture actual native selection once after readiness, under the existing lease.
                        # Status/catalog polling only reads this snapshot and never calls the backend.
                        props_connection = http.client.HTTPConnection('127.0.0.1', BACKEND_PORT, timeout=3)
                        try:
                            props_connection.request('GET', '/props', headers=oxe_runtime.backend_headers(backend_key))
                            props_response = props_connection.getresponse()
                            props_body = props_response.read(65537)
                            if props_response.status == 200 and len(props_body) <= 65536:
                                self.flash_cache_status = oxe_runtime.flash_cache_pref.status_from_props(json.loads(props_body))
                        except (OSError, ValueError, http.client.HTTPException):
                            pass
                        finally:
                            props_connection.close()
                        # F127: keep swap-tolerant mode only when the engine itself reports it; else the 1 GiB floor.
                        if (self.classic_memory_guard is not None
                                and self.classic_memory_guard.mode() == memory_policy.SWAP_TOLERANT_MODE
                                and (self.flash_cache_status or {}).get('reason') != 'preferred-swap-tolerant'):
                            self.classic_memory_guard.leave_swap_tolerant()
                    metadata_connection = http.client.HTTPConnection('127.0.0.1', BACKEND_PORT, timeout=3)
                    try:
                        metadata_connection.request('GET', '/v1/models', headers=oxe_runtime.backend_headers(backend_key))
                        metadata_response = metadata_connection.getresponse()
                        metadata = metadata_response.read()
                        parsed = json.loads(metadata)
                        if metadata_response.status == 200 and isinstance(parsed.get('data'), list):
                            self.cache['/v1/models'] = ('application/json', metadata)
                    except (OSError, ValueError, http.client.HTTPException):
                        pass
                    finally:
                        metadata_connection.close()
                except Exception:
                    self.cleanup()
                    if transition is not None:
                        engine_ledger.record_transition(ENGINE_LEDGER_LOG, transition[0], transition[1],
                                                        (time.monotonic() - transition[2]) * 1000, success=False)
                    raise
            self.active += 1
            if inference: self.inferences += 1
            return self.process

    def backend_key(self):
        """The loaded oxe engine's per-launch bearer token, or None."""
        return getattr(self.process, '_local_ai_api_key', None)

    def backend_json(self, path, payload):
        connection = http.client.HTTPConnection('127.0.0.1', BACKEND_PORT, timeout=900)
        try:
            connection.request('POST', path, body=json.dumps(payload).encode(),
                               headers={'Content-Type': 'application/json', **oxe_runtime.backend_headers(self.backend_key())})
            response = connection.getresponse()
            data = response.read(4 * 1024**2 + 1)
            if response.status != 200 or len(data) > 4 * 1024**2:
                raise ValueError('Prefix operation failed')
            value = json.loads(data)
            if not isinstance(value, dict): raise ValueError('Invalid prefix response')
            return value
        finally: connection.close()

    def prepare_prefix(self, path, payload):
        # begin(inference=True) already excludes competing inference. Do not
        # hold mu here: status must remain responsive during a cold primer.
        if self.backend_runtime is not None: return
        store = self.prefix_store
        if store is None: return
        plan = None
        try:
            profile = dict(self.loaded_config)
            if not profile.get('server_sha256'):
                digest = hashlib.sha256()
                with open(profile['server_binary'], 'rb') as stream:
                    for chunk in iter(lambda: stream.read(1024**2), b''): digest.update(chunk)
                profile['server_sha256'] = digest.hexdigest()
            plan = make_plan(payload, profile) if (path in ('/chat/completions', '/v1/chat/completions')
                and isinstance(payload, dict) and payload.get('cache_prompt') is not False
                and payload.get('id_slot', 0) == 0) else None
            if plan is None:
                self.current_prefix_key = None
                return
            if self.current_prefix_key == plan.key: return
            self.current_prefix_key = None
            filename = store.lookup(plan)
            if filename:
                restored = self.backend_json('/slots/0?action=restore', {'filename': filename})
                if restored.get('n_restored', 0) <= 0:
                    store.discard(plan.key)
                    raise ValueError('Prefix was not restored')
                self.prefix_stats['restores'] += 1
            else:
                self.backend_json('/v1/chat/completions', {**plan.primer, 'id_slot': 0})
                filename = store.pending(plan)
                self.backend_json('/slots/0?action=save', {'filename': filename})
                store.commit(plan)
                self.prefix_stats['primes'] += 1
            self.current_prefix_key = plan.key
        except (OSError, ValueError, KeyError, TypeError, http.client.HTTPException):
            # A cache failure must not log context or prevent ordinary inference.
            self.current_prefix_key = None
            self.prefix_stats['failures'] += 1
            if plan is not None:
                try: store.path(plan.key, 'pending').unlink(missing_ok=True)
                except OSError: pass

    def active_block(self, loading=False):
        """What is happening now, for a progress line. No lock: plain attribute reads.

        phase: idle, queued (requests wait, none admitted), loading, answering. Token
        progress needs the engine to report it and is not available yet (None)."""
        since = getattr(self, 'answering_since', None)
        queued = self.queue_length()
        if loading or self.state == 'loading': phase = 'loading'
        elif self.inferences and since is not None: phase = 'answering'
        elif queued: phase = 'queued'
        else: phase = 'idle'
        loaded = self.loaded_config
        return {'phase': phase, 'model': loaded.get('model_alias') if loaded else None,
                'seconds': round(time.monotonic() - since, 1) if phase == 'answering' else None,
                'queued_requests': queued, 'prompt_tokens_done': None, 'prompt_tokens_total': None,
                'tokens_out': None, 'tokens_per_second': None}

    def harness_learning(self):
        """The opt-in as it is in runtime.json now (a loaded model keeps an older snapshot)."""
        try: return learning_settings(json.loads(CONFIG.read_text()))
        except (OSError, ValueError): return learning_settings(None)

    def harness_status(self, config):
        learning = self.harness_learning()
        with self.harness_mu:
            return {'harnesses': self.harness_registry.status(config), 'request_compat': dict(self.compat_stats),
                    'harness_learning': self.harness_registry.learning_status(learning)}

    def set_harness_learning(self, enabled, min_requests=None):
        """Persist the opt-in (runtime.json `harness_learning`); nothing else is changed."""
        if type(enabled) is not bool or (min_requests is not None and (type(min_requests) is not int or not 1 <= min_requests <= 100)):
            raise Unavailable('Send {"enabled": true or false} and optionally "min_requests" (1-100).')
        # runtime.json is rewritten under mu (like select); mu is held for a whole model load.
        if not self.mu.acquire(timeout=2): raise Unavailable('A model is loading; change the setting when it is ready.')
        try:
            try:
                value = json.loads(CONFIG.read_text())
                if not isinstance(value, dict): raise ValueError()
                current = value.get('harness_learning') if isinstance(value.get('harness_learning'), dict) else {}
                value['harness_learning'] = {**current, 'enabled': enabled,
                                             **({'min_requests': min_requests} if min_requests is not None else {})}
                guard_production_path(CONFIG)
                temporary = CONFIG.with_suffix('.learning.tmp')
                temporary.write_text(json.dumps(value, indent=2) + '\n'); temporary.replace(CONFIG)
            except (OSError, ValueError):
                raise Unavailable('The harness learning setting could not be saved.')
            return learning_settings(value)
        finally: self.mu.release()

    # ---- CACHE-PERFECT P2: disk state cache (state_cache.py) ----
    def cache_settings(self):
        return state_cache.settings(self.runtime_value())

    def runtime_value(self):
        try:
            value = json.loads(CONFIG.read_text())
            return value if isinstance(value, dict) else None
        except (OSError, ValueError): return None

    def state_root(self, runtime=None):
        """The disk state store root: LOCAL_AI_STATE_CACHE_ROOT / runtime.json cache.root when set
        (ValueError when that override is invalid), else the default under the home."""
        return state_cache.root(runtime=self.runtime_value() if runtime is None else runtime)

    def state_engine_live(self):
        """True while a loaded oxe engine owns the store: budget and clear go to it, never to Python."""
        return bool(self.state_engine and self.process is not None and self.process.poll() is None)

    def state_engine_json(self, method, path, payload=None):
        connection = http.client.HTTPConnection('127.0.0.1', BACKEND_PORT, timeout=30)
        try:
            connection.request(method, path, body=None if payload is None else json.dumps(payload).encode(),
                               headers={'Content-Type': 'application/json', **oxe_runtime.backend_headers(self.backend_key())})
            response = connection.getresponse()
            data = response.read(1024**2 + 1)
            value = json.loads(data) if response.status == 200 and len(data) <= 1024**2 else None
            if not isinstance(value, dict): raise ValueError('state cache engine call failed')
            return value
        finally: connection.close()

    def preload_state(self, alias):
        """CACHE-PERFECT P2b: after an explicit load, ask the engine to read its newest stored
        conversation for `alias` back from disk (POST /state-cache/preload, with the last session
        hint when known). Background thread, short timeout; the engine cancels it itself when a
        request arrives. Only an oxe engine with the state-cache flags and the route. Returns
        True when a preload was sent off."""
        with self.mu:
            if not (self.state_preload and self.state_engine_live() and self.loaded_config
                    and self.loaded_config.get('model_alias') == alias):
                return False
            headers = {'Content-Type': 'application/json', **oxe_runtime.backend_headers(self.backend_key())}
        hint = self.session_hints.get(alias)
        if hint is None:
            try: hint = request_ledger.last_session(alias)
            except Exception: hint = None
        body = json.dumps({'session': hint} if isinstance(hint, str) and re.fullmatch(r'[0-9a-f]{16}', hint) else {}).encode()

        def send():
            connection = http.client.HTTPConnection('127.0.0.1', BACKEND_PORT, timeout=PRELOAD_TIMEOUT)
            try:
                connection.request('POST', oxe_runtime.STATE_PRELOAD_ROUTE, body=body, headers=headers)
                response = connection.getresponse()
                reply = response.read(4096)
                _log.info('state preload for %s: HTTP %s %s', alias, response.status, reply[:200])
            except (OSError, http.client.HTTPException) as error:
                _log.warning('state preload for %s failed: %s', alias, type(error).__name__)
            finally:
                connection.close()
        threading.Thread(target=send, name='state-preload', daemon=True).start()
        return True

    def cache_usage(self):
        """GET /_local/cache: a read-only walk of the store (no engine needed); the loaded
        engine's own figures are added under `engine` when it owns the store."""
        s = self.cache_settings()
        value = state_cache.usage(self.state_root(), s['disk_budget_bytes'], s['ttl_secs'])
        value['invalid_settings'] = s['invalid']
        if self.state_engine_live():
            try:
                engine = self.state_engine_json('GET', '/state-cache')
                value['engine'] = {k: engine[k] for k in ('budget_bytes', 'used_bytes') if type(engine.get(k)) is int}
            except (OSError, ValueError, http.client.HTTPException):
                value['engine'] = None
        return value

    def set_cache_budget(self, budget):
        """Persist runtime.json `cache.disk_budget_bytes`, then evict down to it at once: by the
        loaded engine when it owns the store, else in Python. Never touches RAM state."""
        if not state_cache.valid_budget(budget):
            raise ValueError('Send {"bytes": N}: 0 (disk tier off) or a whole number of at least 4000000000.')
        if not self.mu.acquire(timeout=2): raise Unavailable('A model is loading; change the budget when it is ready.')
        try:
            try:
                value = json.loads(CONFIG.read_text())
                if not isinstance(value, dict): raise ValueError()
                value['cache'] = {**(value.get('cache') if isinstance(value.get('cache'), dict) else {}), 'disk_budget_bytes': budget}
                guard_production_path(CONFIG)
                temporary = CONFIG.with_suffix('.cache.tmp')
                temporary.write_text(json.dumps(value, indent=2) + '\n'); temporary.replace(CONFIG)
            except (OSError, ValueError):
                raise Unavailable('The cache budget could not be saved.')
            ttl = state_cache.settings(value)['ttl_secs']
            if self.state_engine_live():
                try: engine = self.state_engine_json('POST', '/state-cache/budget', {'bytes': budget})
                except (OSError, ValueError, http.client.HTTPException):
                    raise Unavailable('The budget was saved; the engine did not apply it yet (it will at the next load).')
                return {'budget_bytes': budget, 'used_bytes': engine.get('used_bytes'),
                        'freed_bytes': engine.get('freed_bytes'), 'applied_by': 'engine'}
            path = self.state_root(value)
            try:
                with self.held_lease():  # no engine may open the store between plan and delete
                    freed = state_cache.gc(path, budget, ttl)
            except Unavailable:
                raise Unavailable('The budget was saved; another job is running, so saved state is trimmed later '
                                  '(at the next load or budget change).') from None
            return {'budget_bytes': budget, 'used_bytes': state_cache.usage(path, budget, ttl)['used_bytes'],
                    'freed_bytes': freed, 'applied_by': 'controller'}
        finally: self.mu.release()

    def clear_state_cache(self, scope='all', model=None, session=None):
        """Called with mu held and no request running. Returns {'scope', 'freed_bytes'}."""
        if self.state_engine_live():
            if scope == 'model':
                # the engine's clear has no model key: it cannot be told which model to drop
                raise Unavailable('Unload the model to clear one model\'s saved state.')
            body = {'scope': scope, **({'session': session} if scope == 'session' else {})}
            engine = self.state_engine_json('POST', '/state-cache/clear', body)
            return {'scope': scope, 'freed_bytes': engine.get('freed_bytes'), 'applied_by': 'engine'}
        runtime = self.runtime_value()
        with self.held_lease():  # across planning and deletion; mu alone does not stop another process
            freed = state_cache.clear(self.state_root(runtime), scope, model, session)
            if scope == 'all':
                # as the engine's clear-all does while it runs: the legacy Rust prefix store too
                freed += self.clear_legacy_prefix(runtime)
        return {'scope': scope, 'freed_bytes': freed, 'applied_by': 'controller'}

    def clear_legacy_prefix(self, runtime):
        """Unlink the entries (<sha>.bin with an OXEPFX01 identity, regular, ours) of the Rust
        --prefix-cache stores of the non-Flash oxe profiles. Lease held by the caller."""
        dirs = {state_cache.store_dir('prefix', runtime=runtime)}
        if state_cache.override(runtime) is None:
            profiles = list((runtime or {}).get('profiles', {}).values()) + [runtime or {}]
            for profile in profiles:
                if (isinstance(profile, dict) and profile.get('backend_kind') == oxe_runtime.KIND
                        and not oxe_runtime.is_flash(profile)):
                    directory = model_catalog.rust_prefix_directory(profile)
                    if directory is not None:
                        dirs.add(Path(directory))
        freed = 0
        for directory in dirs:
            for paths in model_catalog.prefix_identities(directory, 'rust').values():
                for path in paths:
                    try:
                        info = os.lstat(path)
                        if stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid():
                            os.unlink(path)
                            freed += info.st_size
                    except OSError:
                        pass
        return freed

    def prepare_harness(self, harness, model_alias=None):
        """Build or load the stored prefixes of one harness (explicit user action: this
        loads the model). One 1-token primer per known prefix, through the normal queue."""
        if not isinstance(harness, str) or not harness:
            raise Unavailable('Name the harness to prepare.')
        target = self.config(model_alias)
        if target.get('backend_kind') != oxe_runtime.KIND:
            raise Unavailable('Harness prefixes are prepared on the Rust engine models only.')
        if harness == '*':
            # "Prepare all" (after an engine update): every harness whose newest prefix is
            # not expected warm for this model, one after the other through the queue.
            with self.harness_mu:
                names = self.harness_registry.harness_names(target, cold_only=True)
            done = []
            for name in names:
                try: done.append(self.prepare_harness(name, target['model_alias']))
                except Unavailable: pass  # no usable primer for this one
            return {'harness': '*', 'model': target['model_alias'], 'harnesses': done,
                    'prepared': [row for item in done for row in item['prepared']]}
        with self.harness_mu:
            primers = self.harness_registry.primers(harness, target['model_alias'])
        if not primers:
            raise Unavailable('No recorded prefix for this harness yet; run it once against the local model.')
        prepared = []
        for key, body in primers:
            self.begin_queued(inference=True, allow_start=True, model_alias=target['model_alias'])
            try:
                started = time.monotonic()
                reply = self.backend_json('/v1/chat/completions', oxe_runtime.request_defaults(body, '/v1/chat/completions', target))
                timings = reply.get('timings') if isinstance(reply.get('timings'), dict) else {}
                with self.harness_mu:
                    self.harness_registry.complete(key, self.loaded_config or target, timings.get('cache_n'), timings.get('prompt_n'))
                prepared.append({'key': key[:12], 'cache_n': timings.get('cache_n'), 'prompt_n': timings.get('prompt_n'),
                                 'seconds': round(time.monotonic() - started, 2)})
            finally:
                self.end(inference=True)
        return {'harness': harness, 'model': target['model_alias'], 'prepared': prepared}

    def end(self, *, inference=False, touch=False):
        with self.mu:
            self.active -= 1
            if inference:
                self.inferences -= 1
                self.answering_since = None
            if inference or touch:
                self.last_used = time.monotonic()
        if inference:
            with self.queue_cv:
                self.queue_cv.notify_all()

    def unload_info(self):
        """What an 'Unload now' control needs; call with self.mu held. No model access."""
        loaded = self.process is not None and self.loaded_config is not None
        remaining = None
        if loaded and self.state == 'ready' and self.active == 0:
            remaining = max(0.0, idle_seconds(self.loaded_config) - (time.monotonic() - self.last_used))
        return {'loaded_model': self.loaded_config.get('model_alias') if loaded else None,
                'resident_bytes': resident_bytes(self.process.pid) if loaded else None,
                'idle_unload_remaining_seconds': None if remaining is None else round(remaining, 1),
                'unload': {'method': 'POST', 'path': '/_local/unload', 'available': loaded and self.active == 0}}

    def status(self):
        # Avoid blocking status requests while a large model is being loaded.
        if not self.mu.acquire(timeout=.05):
            if self.stop_pending is not None:  # a stop retry holds the lock; say so, promptly
                return {'state': 'stopping', 'stop_pending': self.stop_pending, 'active_requests': 0,
                        'capabilities': list(CAPABILITIES), 'active': self.active_block(loading=True)}
            return {'state': 'loading', 'active_requests': 0, 'capabilities': list(CAPABILITIES),
                    'active': self.active_block(loading=True)}
        try:
            selection_error = None
            try:
                selected_config = self.config()
            except Unavailable:
                selected_config = {}
                selection_error = 'invalid_or_unavailable_selection'
            config = self.loaded_config or selected_config
            setup = {}
            try:
                setup = json.loads((CONFIG.parent / 'download-status.json').read_text())
            except (OSError, ValueError):
                pass
            expected = config.get('expected_bytes', setup.get('expected_bytes'))
            downloaded = 0
            try:
                model = Path(config['model_path'])
                partial = model.with_suffix(model.suffix + '.part')
                downloaded = model.stat().st_size if model.is_file() else partial.stat().st_size if partial.is_file() else 0
                if expected is None:
                    expected = json.loads((CONFIG.parent / 'model.json').read_text()).get('expected_bytes')
            except (OSError, ValueError, KeyError):
                pass
            return {**({'flash_cache': dict(self.flash_cache_status)} if self.flash_cache_status else {}),
                    **({'stop_pending': self.stop_pending} if self.stop_pending is not None else {}),
                    'state': self.state, 'validated': self.eligible(config),
                    'download_state': setup.get('state', 'preparing'),
                    'downloaded_bytes': downloaded, 'expected_bytes': expected,
                    'model': config.get('model_alias', 'qwen3.8-27b-local'),
                    'selected_model': selected_config.get('model_alias'),
                    'selection_error': selection_error,
                    'backend': config.get('backend', 'cpu'),
                    'context_size': config.get('context_size', 8192),
                    'threads': config.get('threads'),
                    'execution_policy_mode': config.get('execution_policy_mode', 'stock'),
                    'memory_policy': self.classic_memory_guard.metadata() if self.classic_memory_guard is not None else self.last_memory_guard,
                    'validation': config.get('validation', {}),
                    'flash_memory': config.get('flash_memory'),
                    'gpu_layers': config.get('gpu_layers', 0),
                    'active_requests': self.inferences, 'queued_requests': self.queue_length(), 'idle_unload_seconds': idle_seconds(config),
                    'image_active': self.image_active, 'image_broker_error': self.image_broker_error,
                    **self.unload_info(),
                    'prefix_cache': {'enabled': self.backend_runtime is None and config.get('prefix_cache', {}).get('enabled') is True, **self.prefix_stats},
                    'backend_kind': config.get('backend_kind', 'llama'),
                    **engine_summary(config),
                    'flash_qualification': self.flash_qualification(),
                    'engine_ledger': self.engine_ledger.metadata() if self.engine_ledger is not None else {'enabled': False},
                    **self.harness_status(config),
                    'flash_idle_qos': {'enabled': isinstance(config.get('oxe_flash'), dict) and 'idle_qos' in config['oxe_flash'],
                                       'unit': self.flash_idle_qos_unit, 'started': self.flash_idle_qos_started,
                                       'error': self.flash_idle_qos_error},
                    'capabilities': list(CAPABILITIES),
                    'active': self.active_block(),
                    'last_error': self.last_failure,
                    'last_failure': dict(self.last_failure_info) if self.last_failure_info else None,
                    'stop_pending': self.stop_pending,
                    'image_slot_stale': len(getattr(self.image_broker, 'stale', None) or ()),
                    'chat_url': 'http://127.0.0.1:' + str(PROXY_PORT),
                    'api_base': 'http://127.0.0.1:' + str(PROXY_PORT) + '/v1'}
        finally:
            self.mu.release()

    def close(self):
        self.stop_event.set()
        if self.image_broker is not None:
            self.image_broker.close()
        with self.mu: self.cleanup()
        self.thread.join(timeout=2)


class _BadFlashChat(Exception):
    """A Flash chat body the runtime could not convert (answered 400 after admission)."""


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    def log_message(self, *args):
        pass

    @property
    def manager(self): return self.server.manager

    def reply(self, code, body, content_type='application/json'):
        self.send_response(code)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        if code in (429, 503): self.send_header('Retry-After', '5')
        for name, value in self.cors_reply_headers(): self.send_header(name, value)
        self.end_headers()
        if self.command != 'HEAD':
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                self.close_connection = True

    def json_reply(self, code, value):
        # Every error outside the control routes has the OpenAI shape, for any client library.
        if code >= 400 and not urlsplit(self.path).path.startswith('/_local/'):
            value = openai_api.normalize_error(code, value)
        self.reply(code, json.dumps(value).encode())

    def api_settings(self):
        """runtime.json as a dict: API key, CORS and alias settings. While the file is
        unreadable the last good copy is used, so a configured key never lapses."""
        try:
            value = json.loads(CONFIG.read_text())
            if not isinstance(value, dict): raise ValueError()
            self.server.api_settings_last = value
            return value
        except (OSError, ValueError):
            return getattr(self.server, 'api_settings_last', {})

    def cors_reply_headers(self):
        origin = getattr(self, 'cors_allowed_origin', None)
        return openai_api.cors_headers(origin) if origin else []

    def do_HEAD(self): self.forward()
    def do_GET(self): self.forward()
    def do_POST(self): self.forward()
    def do_OPTIONS(self): self.forward()
    def do_DELETE(self): self.forward()

    def browser_origin_allowed(self):
        origins = self.headers.get_all('Origin', [])
        sites = self.headers.get_all('Sec-Fetch-Site', [])
        if len(origins) > 1 or len(sites) > 1:
            return False
        # A deliberate top-level click may come from the separate Image Studio.
        navigation = {'Sec-Fetch-Mode':'navigate', 'Sec-Fetch-Dest':'document',
                      'Sec-Fetch-User':'?1'}
        if (not origins and self.command in ('GET', 'HEAD') and self.path == '/'
                and all(self.headers.get_all(key, []) == [value]
                        for key, value in navigation.items())):
            return True
        if sites and sites[0] not in ('same-origin', 'none'):
            return False
        allowed = {f'http://127.0.0.1:{self.server.server_port}',
                   f'http://localhost:{self.server.server_port}',
                   f'http://[::1]:{self.server.server_port}'}
        return not origins or origins[0] in allowed

    def host_allowed(self):
        """F80: only loopback names with the bound port. A DNS-rebinding page reaches 127.0.0.1
        under its own host name; without this check its same-origin GETs (no Origin header) could
        read keyless routes. One Host header, compared case-insensitively."""
        hosts = self.headers.get_all('Host', [])
        if len(hosts) != 1:
            return False
        port = self.server.server_port
        return hosts[0].strip().lower() in {f'127.0.0.1:{port}', f'localhost:{port}', f'[::1]:{port}'}

    def forward(self):
        self.cors_allowed_origin = None  # the handler persists across keep-alive requests
        if not self.host_allowed():
            # before any work: no settings read, no admission, no backend
            self.close_connection = True
            return self.json_reply(403, {'error': 'Host not allowed: use 127.0.0.1, localhost or [::1] with this port.'})
        canonical = canonical_target(self.path)
        if canonical is None:
            self.close_connection = True
            return self.json_reply(404, {'error': {'message': 'Unknown route.', 'type': 'invalid_request_error'}})
        # S4: route and forward the same path. The Rust engine trims trailing slashes, so
        # '/v1/chat/completions/' must be an inference request here too (admission, ledger,
        # defaults, disconnect watcher), not a pass-through.
        self.path = canonical
        path = urlsplit(self.path).path
        origins = self.headers.get_all('Origin', [])
        api_value = self.api_settings()
        api_key = openai_api.api_key(api_value)
        if origins and not path.startswith('/_local/'):
            # Opt-in CORS for the API routes (openai_api.cors_origins); the control routes
            # stay same-origin only.
            self.cors_allowed_origin = openai_api.cors_origin(origins, api_value, api_key is not None)
        if self.cors_allowed_origin is None and not self.browser_origin_allowed():
            self.close_connection = True
            return self.json_reply(403, {'error': 'Cross-origin local AI request refused.'})
        if self.cors_allowed_origin is not None and self.command == 'OPTIONS':
            self.send_response(204)
            for name, value in openai_api.cors_headers(
                    self.cors_allowed_origin, preflight=True,
                    request_headers=self.headers.get('Access-Control-Request-Headers'),
                    private_network=self.headers.get('Access-Control-Request-Private-Network') == 'true'):
                self.send_header(name, value)
            return self.end_headers()
        if api_key is not None and openai_api.key_required(self.command, path) and not openai_api.authorized(self.headers, api_key):
            self.close_connection = True
            return self.json_reply(401, {'error': {
                'message': 'Missing or wrong API key. Send "Authorization: Bearer <key>".',
                'type': 'authentication_error', 'code': 'invalid_api_key'}})
        inference = self.command == 'POST' and path in INFERENCE_PATHS
        helper = self.command == 'POST' and path in MODEL_HELPER_PATHS
        if path == '/_local/status':
            return self.json_reply(200, self.manager.status())
        if path == '/_local/flash/cache' and self.command in ('GET', 'HEAD'):
            try: return self.json_reply(200, self.manager.flash_cache_setting())
            except (Unavailable, OSError, ValueError, KeyError, TypeError, AttributeError):
                return self.json_reply(503, {'error': 'Flash cache settings are unavailable.'})
        if path == '/_local/models' and self.command in ('GET', 'HEAD'):
            try: return self.json_reply(200, self.manager.catalog())
            except (Unavailable, OSError, ValueError, KeyError, TypeError):
                return self.json_reply(503, {'error': 'Model setup and validation are pending.'})
        if path == '/_local/storage' and self.command in ('GET', 'HEAD'):
            try:
                rows = storage_view.inventory(json.loads(CONFIG.read_text()))
            except (OSError, ValueError, KeyError, TypeError, AttributeError):
                return self.json_reply(503, {'error': 'Storage status is unavailable.'})
            return self.json_reply(200, {'object': 'list', 'data': storage_view.public(rows)})
        if self.command == 'POST' and path.startswith('/slots/'):
            self.close_connection = True
            return self.json_reply(403, {'error': 'Slot persistence is controller-owned.'})
        control = {}
        if path == '/_local/ledger' and self.command in ('GET', 'HEAD'):
            wanted = parse_qs(urlsplit(self.path).query).get('limit', ['50'])[-1]
            return self.json_reply(200, request_ledger.recent(int(wanted) if wanted.isdecimal() and len(wanted) < 5 else 50))
        if path == '/_local/harnesses' and self.command in ('GET', 'HEAD'):
            try: config = self.manager.loaded_config or self.manager.config()
            except Unavailable: config = {}
            return self.json_reply(200, self.manager.harness_status(config))
        if path == '/_local/cache' and self.command in ('GET', 'HEAD'):
            try: return self.json_reply(200, self.manager.cache_usage())
            except (OSError, ValueError, KeyError, TypeError):
                return self.json_reply(503, {'error': 'Cache usage is unavailable.'})
        if path in ('/_local/load', '/_local/unload', '/_local/select', '/_local/cache/clear', '/_local/cache/budget', '/_local/harness/prepare', '/_local/harness/forget',
                    '/_local/harness/learning', '/_local/harness/learn', '/_local/harness/label',
                    '/_local/storage/delete', '/_local/flash/cache') and self.command == 'POST':
            # Consume control bodies so a persistent HTTP connection remains
            # usable. These actions accept no prompt or image data.
            try:
                control_length = int(self.headers.get('Content-Length', '0'))
            except ValueError:
                self.close_connection = True
                return self.json_reply(400, {'error': 'Invalid content length.'})
            if self.headers.get('Transfer-Encoding') or not 0 <= control_length <= 1024:
                self.close_connection = True
                return self.json_reply(413, {'error': 'Control request is too large.'})
            if control_length:
                try:
                    control = json.loads(self.rfile.read(control_length))
                    if not isinstance(control, dict): raise ValueError()
                except ValueError: return self.json_reply(400, {'error': 'Invalid control request.'})
        if path == '/_local/flash/cache' and self.command == 'POST':
            if set(control) != {'pref'}:
                return self.json_reply(400, {'error': 'Send only a cache preference.'})
            try: result = self.manager.set_flash_cache_pref(control.get('pref'))
            except Unavailable as error: return self.json_reply(409, {'error': str(error)})
            except ValueError as error: return self.json_reply(400, {'error': str(error)})
            except (OSError, KeyError, TypeError, AttributeError):
                return self.json_reply(503, {'error': 'Flash cache preference could not be saved.'})
            return self.json_reply(200, result)
        if path == '/_local/cache/budget' and self.command == 'POST':
            if set(control) != {'bytes'}:
                return self.json_reply(400, {'error': 'Send only {"bytes": N}.'})
            try: return self.json_reply(200, self.manager.set_cache_budget(control.get('bytes')))
            except ValueError as error: return self.json_reply(400, {'error': str(error)})
            except Unavailable as error: return self.json_reply(409, {'error': str(error)})
            except (OSError, KeyError, TypeError): return self.json_reply(503, {'error': 'The cache budget could not be applied.'})
        if path == '/_local/cache/clear' and self.command == 'POST':
            # No body: the prefix store as before plus all saved state. With a body, only the
            # disk state cache in that scope (all also clears the prefix store).
            scope = control.get('scope', 'all')
            if (not set(control) <= {'scope', 'model', 'session'} or scope not in state_cache.SCOPES
                    or (control.get('model') is not None and not (isinstance(control['model'], str) and state_cache.MODEL.fullmatch(control['model'])))
                    or (scope == 'model' and control.get('model') is None)
                    or (scope == 'session' and not (isinstance(control.get('session'), str) and state_cache.SID.fullmatch(control['session'])))):
                return self.json_reply(400, {'error': 'Send {"scope": "all"}, {"scope": "model", "model": key} or {"scope": "session", "session": id}.'})
            with self.manager.mu:
                if self.manager.active:
                    return self.json_reply(409, {'error': 'A request is running.'})
                if scope == 'all':
                    if self.manager.config().get('backend_kind') == bigmoe_runtime.KIND:
                        return self.json_reply(409, {'error': 'Unload Flash to clear its resident cache.'})
                    try:
                        cache_config = self.manager.config().get('prefix_cache', {})
                        if cache_config.get('enabled'):
                            PrefixStore(cache_config['directory'], cache_config.get('max_bytes', 3 * 1024**3)).clear()
                        self.manager.current_prefix_key = None
                        with self.manager.harness_mu: self.manager.harness_registry.cleared()
                    except (OSError, ValueError, KeyError):
                        return self.json_reply(503, {'error': 'Prefix cache could not be cleared.'})
                try: state = self.manager.clear_state_cache(scope, control.get('model'), control.get('session'))
                except Unavailable as error: return self.json_reply(409, {'error': str(error)})
                except (OSError, ValueError, http.client.HTTPException):
                    return self.json_reply(503, {'error': 'Saved state could not be cleared.'})
            return self.json_reply(200, {'cleared': True, 'state_cache': state})
        if path == '/_local/storage/delete' and self.command == 'POST':
            try: result = self.manager.delete_storage(control.get('id'))
            except Unavailable as error: return self.json_reply(409, {'error': str(error)})
            except LookupError: return self.json_reply(404, {'error': 'Unknown storage entry.'})
            except PermissionError: return self.json_reply(409, {'error': 'This cache is referenced or in use and was kept.'})
            except (OSError, ValueError, KeyError, TypeError): return self.json_reply(400, {'error': 'Storage entry could not be deleted.'})
            return self.json_reply(200, result)
        if path == '/_local/harness/forget' and self.command == 'POST':
            harness = control.get('harness')
            if harness is not None and not isinstance(harness, str):
                return self.json_reply(400, {'error': 'Invalid harness name.'})
            with self.manager.harness_mu: self.manager.harness_registry.forget(harness, ignore=control.get('ignore') is True)
            return self.json_reply(200, {'forgotten': harness or 'all', 'ignored': control.get('ignore') is True})
        if path == '/_local/harness/learning' and self.command == 'POST':
            try: return self.json_reply(200, self.manager.set_harness_learning(control.get('enabled'), control.get('min_requests')))
            except Unavailable as error: return self.json_reply(400, {'error': str(error)})
        if path == '/_local/harness/learn' and self.command == 'POST':
            with self.manager.harness_mu: approved = self.manager.harness_registry.approve(control.get('harness'))
            if not approved: return self.json_reply(404, {'error': 'No such unknown client; it must have sent a request since the controller started.'})
            return self.json_reply(200, {'approved': control.get('harness')})
        if path == '/_local/harness/label' and self.command == 'POST':
            with self.manager.harness_mu: labelled = self.manager.harness_registry.set_label(control.get('harness'), control.get('label'))
            if not labelled: return self.json_reply(400, {'error': 'Only a learned harness can be labelled, with a short plain-text label.'})
            return self.json_reply(200, {'harness': control.get('harness'), 'labelled': True})
        if path == '/_local/harness/prepare' and self.command == 'POST':
            try: result = self.manager.prepare_harness(control.get('harness'), control.get('model'))
            except Unavailable as error: return self.json_reply(409, {'error': str(error)})
            except Exception: return self.json_reply(503, {'error': 'Harness prefix could not be prepared.'})
            return self.json_reply(200, result)
        if path == '/_local/select' and self.command == 'POST':
            try: alias = self.manager.select(control.get('model'))
            except Unavailable as error: return self.json_reply(409, {'error': str(error)})
            return self.json_reply(200, {'selected_model': alias, 'state': self.manager.state})
        if path == '/_local/load' and self.command == 'POST':
            launches = self.manager.launches
            try:
                self.manager.begin(allow_start=True, model_alias=control.get('model'),
                                   origin=request_origin(self.headers), explicit_load=True)
            except Unavailable as error:
                return self.json_reply(503, {'error': str(error)})
            except Exception as error:
                return self.json_reply(503, {'error': 'Local model could not be started '
                                             f'({unexpected_cause(error)}). {JOURNAL_HINT}'})
            launched = self.manager.launches != launches
            alias = (self.manager.loaded_config or {}).get('model_alias')
            self.manager.end(touch=True)
            self.json_reply(200, {'state': 'ready'})
            if launched:  # an explicit load that started an engine: warm it from disk (P2b)
                try: self.manager.preload_state(alias)
                except Exception: pass  # best effort; the load already answered
            return
        if path == '/_local/unload' and self.command == 'POST':
            with self.manager.mu:
                if self.manager.active:
                    return self.json_reply(409, {'error': 'A request is running.'})
                # Optional {"model": alias}: unload only if that model is the loaded one.
                wanted, loaded = control.get('model'), self.manager.loaded_config
                if wanted is not None and (loaded is None or self.manager.process is None
                                           or loaded.get('model_alias') != wanted):
                    return self.json_reply(409, {'error': 'That model is not loaded.'})
                self.manager.cleanup()
            return self.json_reply(200, {'state': 'unloaded'})
        if self.command in ('GET', 'HEAD') and path.startswith(('/v1/models/', '/models/')):
            wanted = path.split('/models/', 1)[1]
            try:
                rows = self.manager.models()
                target = openai_api.resolve_model(wanted, api_value)
            except (Unavailable, OSError, ValueError, KeyError, TypeError):
                return self.json_reply(503, {'error': 'Model setup and validation are pending.'})
            for row in rows:
                if row['id'] == wanted or (row['id'] == target and wanted.startswith(openai_api.PROVIDER_PREFIXES)):
                    return self.json_reply(200, row)
            return self.json_reply(404, {'error': {'message': 'Unknown local model. Choose one listed in /v1/models.',
                                                   'type': 'invalid_request_error', 'param': 'model', 'code': 'model_not_found'}})
        if path.startswith('/_local/'):
            # Every control route is answered above; nothing under /_local reaches an engine.
            return self.json_reply(404, {'error': 'Unknown local control route.'})
        # Polling these routes must never repeatedly reload an idle model.
        if self.command in ('GET', 'HEAD') and path in ('/health', '/v1/models', '/models'):
            if path != '/health':
                try:
                    if self.manager.config().get('profiles'):
                        # Qualification is independent of the backend lifecycle lock.
                        models = self.manager.models()
                        if models:
                            return self.json_reply(200, {'object': 'list', 'data': models})
                        return self.json_reply(503, {'error': 'Model setup and validation are pending.'})
                except (Unavailable, OSError, ValueError, KeyError, TypeError):
                    return self.json_reply(503, {'error': 'Model setup and validation are pending.'})
            status = self.manager.status()
            if not status.get('validated'):
                return self.json_reply(503, {'error': 'Model setup and validation are pending.'})
            if path == '/health': return self.json_reply(200, {'status': 'ok', 'sleeping': status['state'] == 'unloaded'})
            if '/v1/models' in self.manager.cache:
                content_type, data = self.manager.cache['/v1/models']
                return self.reply(200, data, content_type)
            return self.json_reply(200, {'object': 'list', 'data': [
                {'id': status['model'], 'object': 'model', 'created': 0, 'owned_by': 'local'}]})
        length = self.headers.get('Content-Length', '0')
        try: length = int(length)
        except ValueError: return self.json_reply(400, {'error': 'Invalid content length.'})
        if length < 0 or length > 16 * 1024 * 1024:
            self.close_connection = True
            return self.json_reply(413, {'error': 'Request is too large.'})
        if self.headers.get('Transfer-Encoding'):
            self.close_connection = True
            return self.json_reply(411, {'error': 'Use a content length for request bodies.'})
        body = self.rfile.read(length) if length else None
        model_alias = None
        payload = None
        rewritten = False
        if (inference or helper) and body:
            try:
                raw = body
                if self.headers.get('Content-Encoding', '').lower() == 'gzip':
                    with gzip.GzipFile(fileobj=io.BytesIO(body)) as stream: raw = stream.read(16 * 1024**2 + 1)
                    if len(raw) > 16 * 1024**2: raise ValueError()
                payload = json.loads(raw)
                routed = explicit_route.rewrite(payload, enabled=os.environ.get(explicit_route.ENABLE_ENV) == '1')
                if routed is not payload:
                    payload = routed
                    body = json.dumps(payload).encode(); rewritten = True
                if isinstance(payload, dict) and 'model' in payload:
                    if not isinstance(payload['model'], str): raise ValueError()
                    model_alias = self.manager.config(payload['model'])['model_alias']
                    payload['model'] = model_alias
                    body = json.dumps(payload).encode(); rewritten = True
            except (ValueError, OSError, EOFError): return self.json_reply(400, {'error': 'Invalid model request.'})
            except Unavailable as error:
                return self.json_reply(404, {'error': {'message': str(error), 'type': 'invalid_request_error',
                                                       'param': 'model', 'code': 'model_not_found'}})
        try:
            target = self.manager.config(model_alias) if model_alias is not None else (self.manager.loaded_config or self.manager.config())
        except Unavailable as error:
            return self.json_reply(503, {'error': {'message': str(error), 'type': 'local_backend_busy'}})
        harness_key = None
        harness = None
        turn = None  # CACHE-PERFECT turn labels (cache_turn.py), oxe chat requests only
        backend_path = self.path
        completion_meta = None
        text_only = target.get('backend_kind') in (oxe_runtime.KIND, bigmoe_runtime.KIND)
        if text_only and self.command == 'POST' and path in openai_api.UNSUPPORTED:
            # Answered here: no model is loaded for an endpoint the engine does not have.
            # 404, not 501: client libraries retry every 5xx.
            return self.json_reply(404, {'error': {
                'message': 'This local model server does not provide %s (%s).' % (openai_api.UNSUPPORTED[path], path),
                'type': 'invalid_request_error', 'code': 'unsupported_endpoint'}})
        if inference and isinstance(payload, dict):
            if target.get('backend_kind') == oxe_runtime.KIND:
                payload, compat_notes = harness_compat.normalize(payload, path)
                compat_notes = list(compat_notes)
                payload = openai_api.single_choice(payload, compat_notes)
                payload = oxe_runtime.request_defaults(payload, path, target)
                if path == '/v1/completions':
                    # The Rust engine has llama-server's native /completion only.
                    try: payload, completion_meta = openai_api.completion_request(payload)
                    except openai_api.Unsupported as error:
                        return self.json_reply(400, {'error': {'message': str(error), 'type': 'invalid_request_error'}})
                    backend_path = '/completion'
                body = json.dumps(payload).encode(); rewritten = True
                harness = None
                if path in harness_compat.CHAT_PATHS:
                    harness = harness_compat.identify(payload, self.headers.get('User-Agent'))
                    try:
                        turn = cache_turn.label(payload)
                    except Exception:  # labels are diagnostic; never fail the request
                        turn = None
                with self.manager.harness_mu:
                    self.manager.compat_stats.update(compat_notes)
                    if harness is not None:
                        harness_key = self.manager.harness_registry.observe(
                            payload, harness, target, learning=learning_settings(api_value),
                            user_agent=self.headers.get('User-Agent'))
            elif target.get('backend_kind') == bigmoe_runtime.KIND and path in harness_compat.CHAT_PATHS:
                compat_notes = []
                payload = openai_api.flash_request(payload, compat_notes)
                if compat_notes:
                    with self.manager.harness_mu: self.manager.compat_stats.update(compat_notes)
            else:
                defaulted = request_defaults(target, payload, path)
                if defaulted is not payload:
                    payload = defaulted
                    body = json.dumps(payload).encode(); rewritten = True
        streamed = target.get('backend_kind') == bigmoe_runtime.KIND
        if streamed:
            if path == '/' and self.command in ('GET', 'HEAD'):
                if not self.manager.eligible(target):
                    return self.json_reply(503, {'error': 'Flash profile is not qualified.'})
                page = FLASH_CHAT_HTML.replace(b'{MODEL}', html.escape(target['model_alias'], quote=True).encode())
                return self.reply(200, page, 'text/html; charset=utf-8')
            if self.command != 'POST' or self.path != '/v1/chat/completions':
                self.close_connection = True
                return self.json_reply(403, {'error': 'This Flash route is not exposed by Local AI.'})
            try:
                payload = bigmoe_runtime.normalize_chat(target, payload, FLASH_BACKEND_PORT)
                body = json.dumps(payload).encode(); rewritten = True
            except (ValueError, OSError, TypeError, KeyError):
                return self.json_reply(400, {'error': 'Invalid Flash chat request or unqualified reasoning settings.'})
        elif self.command in ('GET', 'HEAD') and path != '/' and path in self.manager.cache:
            content_type, data = self.manager.cache[path]
            return self.reply(200, data, content_type)
        request_process = None
        request_engine_config = None
        # One ledger line per inference request: numbers and labels only (request_ledger.py).
        led = None
        if inference:
            led = request_ledger.Entry(path, payload, self.headers.get('User-Agent'), target.get('model_alias'),
                                       headers=self.headers)
            led.label(harness, harness_key)
            led.label_turn(turn)
        try:
            request_process = self.manager.begin_queued(inference=inference,
                allow_start=inference or helper or (self.command in ('GET', 'HEAD') and path == '/'), model_alias=model_alias,
                origin=request_origin(self.headers))
            if led is not None:
                led.admitted_now(*(getattr(self.manager, 'admission_times', None) or (None, None)))
        except Unavailable as error:
            if led is not None: led.finish(503, 'busy' if 'running' in str(error) else 'unavailable')
            if path == '/':
                # Deliberately simple setup status; the chat UI comes from llama.cpp.
                page = b'''<!doctype html><meta charset="utf-8"><title>Local Qwen</title>
<style>body{font:17px system-ui;max-width:650px;margin:12vh auto;padding:20px;background:#171717;color:#eee}a{color:#9dcaff}</style>
<h1>Local Qwen</h1><p>{MESSAGE}</p>
<p><a href="/">Try opening chat again</a> &middot; <a href="/_local/status">Setup status</a> &middot; <a href="http://127.0.0.1:8090">Image studio</a></p>'''
                page = page.replace(b'{MESSAGE}', html.escape(str(error)).encode())
                return self.reply(503, page, 'text/html; charset=utf-8')
            return self.json_reply(503, {'error': {'message': str(error), 'type': 'local_backend_busy'}})
        except Exception as error:
            if led is not None: led.finish(503, 'start_failed')
            return self.json_reply(503, {'error': {'message': 'Local backend could not be started '
                                                   f'({unexpected_cause(error)}). {JOURNAL_HINT}',
                                                   'type': 'local_backend_error'}})
        # F80: from admission to the response's own try/finally, any exception must release the
        # slot, or every later request waits in the FIFO until its 503 (stuck queue)
        try:
            request_engine_config = (getattr(request_process, '_local_ai_engine_config', None)
                                     or self.manager.loaded_config)
            runtime = self.manager.backend_runtime
            if runtime is not None:
                try:
                    if payload.get('model') != runtime.spec.backend_model_id:
                        payload = runtime.chat_body(payload)
                    body = json.dumps(payload).encode(); rewritten = True
                except (ValueError, TypeError, KeyError):
                    raise _BadFlashChat() from None
            elif inference:
                self.manager.prepare_prefix(path, payload)
            backend_port = runtime.spec.port if runtime is not None else BACKEND_PORT
            connection = http.client.HTTPConnection('127.0.0.1', backend_port, timeout=1800 if runtime is not None else 900)
            headers = {k: v for k, v in self.headers.items()
                       if k.lower() not in HOP_HEADERS | {'host', 'accept-encoding'}
                       and (runtime is None or k.lower() not in {'authorization', 'x-api-key', 'api-key', 'cookie', 'origin'})
                       # the local API key is checked here and never handed to an engine
                       and (api_key is None or k.lower() not in {'authorization', 'x-api-key', 'api-key'})}
            if rewritten:
                headers = {k: v for k, v in headers.items() if k.lower() != 'content-encoding'}
            headers['Host'] = '127.0.0.1:' + str(backend_port)
            if runtime is None:
                # B1: the backend port is internal. Browser context headers never reach it (the
                # engine refuses any Origin or cross-site fetch); the controller already applied
                # its own Host/Origin/CORS rules. An oxe engine gets its per-launch token instead of
                # whatever credentials the client sent.
                headers = {k: v for k, v in headers.items()
                           if k.lower() != 'origin' and not k.lower().startswith('sec-fetch-')
                           and k.lower() not in ('x-oxe-session', 'x-oxe-turn-kind')}
                if turn is not None:
                    # CACHE-PERFECT: session hint (the ledger's keyed hash, never a cache key) and turn kind
                    try:
                        hint = led.session if led is not None else request_ledger.session_key(self.headers, payload)
                    except Exception:
                        hint = None
                    headers.update(cache_turn.backend_headers(hint, turn))
                    hints = getattr(self.manager, 'session_hints', None)  # for the next explicit load's preload
                    if isinstance(hints, dict) and isinstance(hint, str) and isinstance(target.get('model_alias'), str):
                        hints[target['model_alias']] = hint
                backend_key = getattr(request_process, '_local_ai_api_key', None)
                if backend_key:
                    headers = {k: v for k, v in headers.items()
                               if k.lower() not in {'authorization', 'x-api-key', 'api-key'}}
                    headers.update(oxe_runtime.backend_headers(backend_key))
            if runtime is not None:
                if self.headers.get('Origin') is not None:
                    headers['Origin'] = f'http://127.0.0.1:{backend_port}'
                headers.update(runtime.headers())
                headers['Content-Type'] = 'application/json'
        except _BadFlashChat:
            # sibling clauses: an exception raised while replying here is not caught by the next
            # clause, so the slot is released exactly once on every path
            self.manager.end(inference=inference, touch=helper)
            if led is not None: led.finish(400, 'bad_request')
            return self.json_reply(400, {'error': 'Invalid Flash chat request or unqualified reasoning settings.'})
        except Exception:
            self.manager.end(inference=inference, touch=helper)
            if led is not None: led.finish(500, 'local_controller_error')
            self.close_connection = True
            return self.json_reply(500, {'error': {'message': 'Local controller error before the request reached the engine.',
                                                   'type': 'local_controller_error'}})
        if self.command in ('GET', 'HEAD') and not path.startswith(('/v1/', '/props', '/slots', '/metrics', '/health', '/models')):
            # llama.cpp embeds its UI as gzip-only assets. Decode them below so
            # both browser clients and the in-memory cache get ordinary bytes.
            headers['Accept-Encoding'] = 'gzip'
        started_response = False
        client_write_failed = False
        watcher = None
        backend_status = None
        request_error = None

        def client_gone():
            return client_write_failed or (watcher is not None and watcher.disconnected.is_set())

        def backend_failed():
            nonlocal request_error
            native_died = request_process is not None and request_process.poll() is not None
            if native_died:
                self.manager.record_engine_death(request_process, request_engine_config)
            request_error = 'backend_died'
            if not started_response:
                self.json_reply(502, {'error': {'message': 'Local backend connection closed.', 'type': 'backend_died'}})
            else:
                self.close_connection = True

        def write_client(data):
            nonlocal client_write_failed
            try:
                self.wfile.write(data)
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                client_write_failed = True
                raise

        def reply_client(status, data, content_type):
            nonlocal client_write_failed
            if watcher is not None:
                watcher.stop()
            try:
                return self.reply(status, data, content_type)
            except (BrokenPipeError, ConnectionResetError):
                client_write_failed = True
                raise

        try:
            connection.request(self.command, backend_path, body=body, headers=headers)
            if inference and connection.sock is not None:
                watcher = ClientDisconnect(self.connection, connection.sock).start()
            response = connection.getresponse()
            backend_status = response.status
            if inference and response.status >= 400: self.manager.current_prefix_key = None
            if response.status != 200: harness_key = None
            tail = b''
            content_type = response.getheader('Content-Type', 'application/octet-stream')
            # Static app resources and /props are safe to cache; never cache slots or user responses.
            cacheable = self.command == 'GET' and response.status == 200 and (
                path == '/props' or path.endswith(('.js', '.css', '.svg', '.ico', '.woff', '.woff2', '.html')))
            if response.getheader('Content-Encoding', '').lower() == 'gzip' and self.command != 'HEAD':
                try:
                    data = gzip.decompress(response.read())
                except (OSError, EOFError, ValueError):  # N6: a corrupt gzip body is a 502, not a crash
                    return reply_client(502, json.dumps({'error': {'message': 'The local backend sent a corrupt '
                                        'compressed response.', 'type': 'local_backend_error'}}).encode(), 'application/json')
                if cacheable: self.manager.cache[path] = (content_type, data)
                return reply_client(response.status, data, content_type)
            if cacheable:
                data = response.read()
                self.manager.cache[path] = (content_type, data)
                return reply_client(response.status, data, content_type)
            shim = None
            if completion_meta is not None and response.status == 200:
                if not completion_meta['stream']:
                    data = response.read()
                    if led is not None: led.observe(data)
                    try: data = json.dumps(openai_api.completion_response(json.loads(data), completion_meta)).encode()
                    except ValueError: pass
                    return reply_client(200, data, 'application/json')
                shim = openai_api.CompletionStream(completion_meta)
            try:
                self.send_response(response.status)
                for k, v in response.getheaders():
                    # CORS is decided here (openai_api.cors_origins), never by a backend.
                    if k.lower().startswith('access-control-'): continue
                    if k.lower() not in HOP_HEADERS and (runtime is None or k.lower() not in {'authorization', 'x-api-key', 'api-key', 'set-cookie'}): self.send_header(k, v)
                for k, v in self.cors_reply_headers(): self.send_header(k, v)
                self.send_header('Cache-Control', 'no-store')
                self.send_header('Transfer-Encoding', 'chunked')
                self.end_headers()
            except (BrokenPipeError, ConnectionResetError):
                client_write_failed = True
                raise
            started_response = True
            if self.command != 'HEAD':
                while chunk := response.read1(65536):
                    if harness_key is not None: tail = (tail + chunk)[-4096:]
                    if led is not None: led.observe(chunk)
                    if shim is not None:
                        chunk = shim.feed(chunk)
                        if not chunk: continue
                    write_client(('%x\r\n' % len(chunk)).encode() + chunk + b'\r\n')
                if shim is not None and (chunk := shim.finish()):
                    write_client(('%x\r\n' % len(chunk)).encode() + chunk + b'\r\n')
                if watcher is not None:
                    watcher.stop()
                if request_process is not None and request_process.poll() is not None:
                    # Headers already escaped; a closed, incomplete response is
                    # honest, whereas a final chunk would disguise engine death.
                    request_error = 'backend_died'
                    self.close_connection = True
                else:
                    write_client(b'0\r\n\r\n')
                    if harness_key is not None:
                        with self.manager.harness_mu:
                            self.manager.harness_registry.complete(harness_key, target, *timings_from_tail(tail))
        except (BrokenPipeError, ConnectionResetError):
            if client_gone():
                self.close_connection = True
            else:
                backend_failed()
        except (OSError, http.client.HTTPException) as error:
            if client_gone():
                self.close_connection = True
            else:
                if inference: self.manager.current_prefix_key = None
                if isinstance(error, http.client.RemoteDisconnected) or (
                        request_process is not None and request_process.poll() is not None):
                    backend_failed()
                elif not started_response:
                    self.json_reply(503, {'error': {'message': 'Local backend connection failed.', 'type': 'local_backend_error'}})
                else:
                    self.close_connection = True
        finally:
            try:
                # Stop and join before end() admits another FIFO request. The watcher
                # owns only this request's socket; no later request can be cancelled.
                if watcher is not None:
                    watcher.stop()
                if client_gone():
                    request_error = 'client_disconnected'
                    self.close_connection = True
                if request_process is not None and request_process.poll() is not None:
                    self.manager.record_engine_death(request_process, request_engine_config)
                connection.close()
                if led is not None:
                    led.finish(backend_status if backend_status is not None else 503, request_error)
                if request_error and self.manager.engine_log_writer is not None:
                    self.manager.engine_log_writer.write(json.dumps({
                        'event': 'request_end', 'error': request_error,
                        'status': backend_status,
                        'model_alias': (request_engine_config or {}).get('model_alias'),
                    }, sort_keys=True) + '\n')
            finally:
                # F80: release the slot even if the bookkeeping above raises
                self.manager.end(inference=inference, touch=helper)


class Server(ThreadingHTTPServer):
    daemon_threads = True


def log_effective_engines(manager, stream=None):
    """M4: one startup line per profile naming the engine it will really use, so a missing
    switch or binary is visible in the journal instead of silently running the classic stack."""
    stream = stream or sys.stderr
    try:
        value = json.loads(CONFIG.read_text())
        aliases = [value.get('model_alias', 'qwen-local'), *value.get('profiles', {})]
    except (OSError, ValueError, AttributeError, TypeError):
        print(f'local-ai: no readable runtime config at {CONFIG}', file=stream, flush=True)
        return
    for alias in dict.fromkeys(aliases):
        try:
            summary = engine_summary(manager.config(alias))
            note = f" ({summary['engine_note']})" if summary.get('engine_note') else ''
            print(f"local-ai: {alias}: engine {summary['engine']}{note}", file=stream, flush=True)
        except Exception as error:
            print(f'local-ai: {alias}: profile not usable: {error}', file=stream, flush=True)


def main():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    manager = Manager()
    log_effective_engines(manager)
    server = Server(('127.0.0.1', PROXY_PORT), Handler)
    server.manager = manager
    # Raising out of the serve loop guarantees child cleanup on service stop.
    def stop(*_): raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM, stop)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally:
        server.server_close()
        manager.close()


if __name__ == '__main__': main()
