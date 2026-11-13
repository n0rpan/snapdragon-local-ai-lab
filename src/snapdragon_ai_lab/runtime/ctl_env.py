"""Controller settings: where its files live, its ports and time limits, and its two errors.

Everything here comes from the environment once, at import. Tests replace single values on the
`controller` module; the split modules therefore read them as `ctl.NAME` (ctl_seams.py), never
from this module directly.
"""
import logging
import math
import os
from pathlib import Path
import pwd
import sys
import tempfile
if __package__:
    from .ctl_seams import ctl
else:
    from ctl_seams import ctl


def xdg_dir(name, fallback):
    """Audit N8: an XDG base directory, used only when it is set and absolute (the XDG spec: a
    relative value is invalid and ignored). An empty XDG_STATE_HOME gave relative log paths."""
    value = os.environ.get(name)
    return Path(value) if value and os.path.isabs(value) else Path.home() / fallback


def env_seconds(name, default):
    """Audit N8: a positive, finite number of seconds from the environment, else the default
    (NaN, inf, zero or a negative idle time disabled or broke the idle unload)."""
    try:
        value = float(os.environ.get(name, default))
    except ValueError:
        return float(default)
    return value if math.isfinite(value) and value > 0 else float(default)


# M4b: XDG defaults (the old ~/Work/local-llm and ~/Work/prompt-studio fallbacks were private-lab
# paths). The installed unit sets both explicitly, so this only affects a unit without them.
CONFIG = Path(os.environ.get('LOCAL_LLM_CONFIG', str(
    xdg_dir('XDG_CONFIG_HOME', '.config') / 'local-ai/llm/runtime.json')))
LEASE = Path(os.environ.get('LOCAL_AI_LEASE', str(
    xdg_dir('XDG_STATE_HOME', '.local/state') / 'local-ai/inference.lock')))


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
    for real in ctl.REAL_PRODUCTION_PATHS:
        real = real.resolve()
        if resolved == real or resolved.is_relative_to(real):
            raise RuntimeError(f'test run refused to touch production path {resolved}')


BACKEND_PORT = int(os.environ.get('LOCAL_LLM_BACKEND_PORT', '8093'))
FLASH_BACKEND_PORT = int(os.environ.get('LOCAL_FLASH_BACKEND_PORT', '8094'))
if BACKEND_PORT == FLASH_BACKEND_PORT:
    raise ValueError('Classic and Flash backends require distinct private ports.')
PROXY_PORT = int(os.environ.get('LOCAL_LLM_PORT', '8092'))
IDLE_SECONDS = env_seconds('LOCAL_LLM_IDLE_SECONDS', 600)

# Overlapping inference requests (OpenCode title/subagent calls) wait in FIFO order for
# the single backend slot instead of failing with 503; 503 only after this many seconds.
QUEUE_WAIT_SECONDS = env_seconds('LOCAL_LLM_QUEUE_WAIT_SECONDS', 120)
START_SECONDS = env_seconds('LOCAL_LLM_START_SECONDS', 180)
# v0.1.3 LONG-APPEND: socket read timeout (seconds without a byte) of a proxied engine request. A
# streamed oxe request sends an SSE comment (`: prefill done/total Ns`) every 15 s while it waits
# or prefills, so this bounds a dead engine, never a long prefill; a non-streamed one sends nothing
# until it ends, so it must cover the longest prefill (a 28.5K append at ~83K: ~950-1200 s). It is
# never shorter than the OpenCode overlay's waits (launch_local_plan: 3600 s), so the controller
# never gives up first and the client never sees a 503 it would retry with a second full prefill.
BACKEND_READ_TIMEOUT_SECONDS = float(os.environ.get('LOCAL_LLM_BACKEND_READ_TIMEOUT', '3600'))

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


class Unavailable(Exception):
    pass


class StopPending(Unavailable):
    """Audit S5: a control action had to stop the engine and could not prove its exit (503)."""


class ModelBusy(Unavailable):
    """O-102 M3/M4: a shared request found no model, or another one, loaded (409). Raised inside
    Manager.begin under `mu`, so no load, switch or unload can slip in between check and admission."""


_log = logging.getLogger('local_ai.controller')
# Control routes this controller answers itself (status `capabilities`); the panel asks
# for a route only when it is listed.
CAPABILITIES = ('models', 'storage', 'harness', 'harness_learning', 'ledger', 'active', 'openai_aliases', 'state_cache',
                'context_modes', 'preload')

# Harness prefix registry (which system-prompt + tools prefixes were seen, and primers to
# re-warm them). LOCAL_AI_HARNESS_STATE='' keeps it in memory only (no primers, no prepare).
HARNESS_STATE = os.environ.get('LOCAL_AI_HARNESS_STATE', str(
    xdg_dir('XDG_STATE_HOME', '.local/state') / 'local-ai/harness-prefixes'))
ENGINE_DIAGNOSTIC_LOG = xdg_dir('XDG_STATE_HOME', '.local/state') / 'local-ai/engine-stderr.log'
ENGINE_LEDGER_LOG = ENGINE_DIAGNOSTIC_LOG.with_name('engine-ledger.jsonl')
