"""Suite-wide isolation: every lab test runs with a private HOME and XDG roots.

Imported first by `unittest discover` (tests/test_000_isolation.py sorts first) and by
tests/conftest.py under pytest, before any module reads a default path. PATH is kept.
The controller additionally refuses to write the real config or lease under a test runner
(controller.guard_production_path), and tests/test_zzz_production_sentinel.py checks that the
real files did not change during the run.
"""
import os
import pwd
import tempfile
from pathlib import Path

REAL_HOME = Path(pwd.getpwuid(os.getuid()).pw_dir)
SENTINELS = (REAL_HOME / '.config/local-ai/llm/runtime.json',
             REAL_HOME / '.config/omarchy/plugins/snapdragon.local-ai/Model.js',
             REAL_HOME / '.config/omarchy/shell.json')
ROOT = None
BEFORE = {}


def stamp(path):
    try:
        info = path.stat()
        return (info.st_mtime_ns, info.st_size)
    except OSError:
        return None


def isolate():
    global ROOT
    if ROOT is not None:
        return ROOT
    BEFORE.update({str(path): stamp(path) for path in SENTINELS})
    ROOT = Path(tempfile.mkdtemp(prefix='local-ai-lab-tests-'))
    for name in ('home', 'config', 'state', 'data', 'cache', 'runtime'):
        (ROOT / name).mkdir(mode=0o700)
    os.environ.update(HOME=str(ROOT / 'home'), XDG_CONFIG_HOME=str(ROOT / 'config'),
                      XDG_STATE_HOME=str(ROOT / 'state'), XDG_DATA_HOME=str(ROOT / 'data'),
                      XDG_CACHE_HOME=str(ROOT / 'cache'), XDG_RUNTIME_DIR=str(ROOT / 'runtime'))
    for name in ('LOCAL_LLM_CONFIG', 'LOCAL_AI_LEASE'):
        os.environ.pop(name, None)
    return ROOT


def socket_tmpdir():
    """N7: a short parent for temporary dirs that hold AF_UNIX sockets (sun_path is 108 bytes,
    so a deep TMPDIR breaks bind). /tmp when usable, else the default temp dir."""
    return '/tmp' if os.path.isdir('/tmp') and os.access('/tmp', os.W_OK | os.X_OK) else None
