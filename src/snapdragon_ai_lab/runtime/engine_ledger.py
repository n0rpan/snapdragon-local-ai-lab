"""Opt-in numeric ledger capture through an inherited pipe; never loads a model."""
import json
import math
import os

try:
    from .engine_diagnostics import EngineLogWriter, EngineStderrReader
except ImportError:
    from engine_diagnostics import EngineLogWriter, EngineStderrReader


ENABLE_ENV = 'LOCAL_AI_ENGINE_LEDGER'
NUMBERS = frozenset({'t', 'n_prompt', 'cache_n', 'prompt_n', 'prompt_ms',
                     'predicted_n', 'predicted_ms', 'draft_n', 'draft_n_accepted',
                     'parks', 'unparks', 'park_ms', 'unpark_ms', 'park_evictions',
                     'checkpoints', 'parked', 'parked_mib'})
ENUMS = {'cache_src': {'none', 'live', 'parked', 'parked-checkpoint', 'checkpoint', 'session', 'store'},
         'stop': {'none', 'eos', 'word', 'limit'}}


def record_transition(path, previous, target, elapsed_ms, *, success):
    """Controller's actual stop/load/readiness cost, excluding queue wait.

    The caller opts in with explicit routing. No prompt, request or arbitrary
    metadata is accepted. Same bounded owner-only storage as native records;
    diagnostic failures cannot prevent inference.
    """
    if (previous is not None and (not isinstance(previous, str) or len(previous) > 256)
            or not isinstance(target, str) or len(target) > 256
            or type(elapsed_ms) not in (int, float) or not math.isfinite(elapsed_ms) or elapsed_ms < 0):
        return False
    writer = EngineLogWriter(path)
    writer.write(json.dumps({'event': 'model_load' if previous is None else 'model_switch',
                             'from_model': previous, 'model_alias': target,
                             'switch_ms': elapsed_ms, 'success': bool(success)}, sort_keys=True) + '\n')
    return writer.error is None


def enabled(config, command):
    """Per-profile capability opt-in, not an inherited global toggle.

    An explicitly supplied ledger argument remains the profile's own choice.
    Checking metadata must not create a directory, file, pipe or thread.
    """
    return (config.get('backend_kind') == 'oxe'
            and config.get('environment', {}).get(ENABLE_ENV) == '1'
            and not any(arg.split('=', 1)[0] == '--ledger' for arg in command))


class NativeLedger:
    """Bridge a child's --ledger to the existing bounded private log writer.

    The child opens its inherited descriptor through /proc/self/fd. No growing
    file is held open by the engine, so rotation remains controller-owned.
    """
    def __init__(self, path, config):
        self.writer = EngineLogWriter(path)
        if self.writer.error is not None:
            raise OSError('native_ledger_log_unavailable') from self.writer.error
        self.context = {'event': 'generation', 'model_alias': config['model_alias'],
                        'engine_sha256': config['server_sha256']}
        self.records = 0
        self.dropped = 0
        read_fd, self.write_fd = os.pipe()
        self.reader = EngineStderrReader(os.fdopen(read_fd, 'rb', buffering=0), self)

    @property
    def arguments(self):
        return ['--ledger', f'/proc/self/fd/{self.write_fd}']

    @property
    def pass_fds(self):
        return (self.write_fd,)

    def start(self):
        """Call only after Popen inherited write_fd; close the parent's copy."""
        self._close_write_fd()
        self.reader.start()

    def _close_write_fd(self):
        if self.write_fd is not None:
            os.close(self.write_fd)
            self.write_fd = None

    def stop(self, timeout=5):
        """Also closes both ends when launch failed before start()."""
        self._close_write_fd()
        return self.reader.stop(timeout=timeout)

    def write(self, line):
        """Called by the pipe reader; discard unknown fields and raw content."""
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError('ledger_object_required')
            safe = {key: value for key, value in row.items()
                    if key in NUMBERS and type(value) in (int, float)
                    and math.isfinite(value)}
            if not {'n_prompt', 'predicted_n', 'prompt_ms'} <= safe.keys():
                raise ValueError('ledger_counts_required')
            for key, choices in ENUMS.items():
                if isinstance(row.get(key), str) and row[key] in choices:
                    safe[key] = row[key]
            self.writer.write(json.dumps({**self.context, **safe}, sort_keys=True) + '\n')
            self.records += 1
        except (ValueError, TypeError, RecursionError, OverflowError):
            self.dropped += 1

    def metadata(self):
        error = self.writer.error or self.reader.error
        return {'enabled': True, 'path': str(self.writer.path), 'records': self.records,
                'dropped': self.dropped, 'error': type(error).__name__ if error else None}
