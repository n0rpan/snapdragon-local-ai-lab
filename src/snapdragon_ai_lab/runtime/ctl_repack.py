"""The opt-in llama.cpp CPU repack cache: its identity, the stderr reader that keeps its ERROR
lines, the bounded error log and the desktop notification.
"""
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading
import time
if __package__:
    from .ctl_env import xdg_dir
    from .ctl_seams import ctl
else:
    from ctl_env import xdg_dir
    from ctl_seams import ctl


# Opt-in native CPU repack cache (patches/llama-repack-cache.patch). The
# controller only supplies identity and surfaces ERROR lines; it never scans or
# prunes the cache. After an engine or model update, prune explicitly with the
# patch's tool, e.g. `tools/repack_cache.py prune --dir <cache-dir> --keep-latest`.
REPACK_CACHE_ENV = 'GGML_CPU_REPACK_CACHE'
REPACK_ERROR_PREFIX = b'ggml-repack-cache: ERROR'
REPACK_ERROR_LOG = Path(os.environ.get('LOCAL_AI_REPACK_ERROR_LOG', str(
    xdg_dir('XDG_STATE_HOME', '.local/state') / 'local-ai/repack-cache-errors.log')))
REPACK_ERROR_LINE_CHARS = 512
REPACK_ERROR_LINES_PER_LOAD = 64
REPACK_ERROR_LOG_BYTES = 256 * 1024
REPACK_ERROR_LOG_KEEP = 2
SHA256_RE = re.compile('[0-9a-f]{64}')


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
        self.log_path = Path(log_path or ctl.REPACK_ERROR_LOG)
        self.label = re.sub(r'[^A-Za-z0-9._-]', '_', str(label))[:64]
        self.notify = notify or ctl.notify_repack_errors
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
                if candidate and self.count < ctl.REPACK_ERROR_LINES_PER_LOAD:
                    self.record(chunk)
        except (OSError, ValueError):
            pass
        finally:
            try: self.stream.close()
            except (OSError, ValueError): pass

    def record(self, raw):
        text = raw.decode('utf-8', 'replace').rstrip('\r\n')
        text = ''.join(c if c.isprintable() else '?' for c in text)[:ctl.REPACK_ERROR_LINE_CHARS]
        self.count += 1
        try:
            ctl.write_repack_error(self.log_path, f'{time.strftime("%Y-%m-%dT%H:%M:%S%z")} {self.label} {text}\n')
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
        ctl.write_repack_error(ctl.REPACK_ERROR_LOG, f'{time.strftime("%Y-%m-%dT%H:%M:%S%z")} {label} '
                           f'local-ai: ERROR repack-cache-disabled missing-identity alias={label}\n')
    except OSError:
        pass
    ctl.notify_repack_errors(ctl.REPACK_ERROR_LOG)


def write_repack_error(path, line):
    path = Path(path)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    data = line.encode()
    try:
        size = os.lstat(path).st_size
    except FileNotFoundError:
        size = 0
    if size and size + len(data) > ctl.REPACK_ERROR_LOG_BYTES:
        for index in range(ctl.REPACK_ERROR_LOG_KEEP, 1, -1):
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
