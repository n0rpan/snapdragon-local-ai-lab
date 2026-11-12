"""Read-only high-rate devfreq trace and explicit, reversible experiment votes.

This module is not part of inference admission or production power policy.
Callers of temporary_max_vote own the inference/build leases and stop their
child before exiting its context. No frequency is changed by import or tracing.
"""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import threading
import time

NODES = ('ddr', 'llcc', 'ddr-qos')
SYSFS = Path('/sys/class/devfreq')


def policies(root=SYSFS):
    result = {}
    for node in NODES:
        row = {'units': 'Hz' if node != 'ddr-qos' else 'driver_value'}
        for field in ('governor', 'cur_freq', 'min_freq', 'max_freq', 'available_frequencies'):
            try:
                text = (root / node / field).read_text().strip()
                row[field] = (text if field == 'governor' else
                              [int(x) for x in text.split()] if field == 'available_frequencies' else int(text))
            except (OSError, ValueError) as exc:
                row[field] = None
                row[field + '_error'] = type(exc).__name__
        result[node] = row
    return result


def _write_vote(path, value):
    # Only the two approved runtime knobs; no shell interpolation or arbitrary
    # file writes through sudo. Tests inject their own in-memory writer.
    if path not in {SYSFS / n / 'min_freq' for n in ('ddr', 'llcc')}:
        raise ValueError('unapproved_devfreq_path')
    result = subprocess.run(['sudo', '-n', 'tee', str(path)], input=str(int(value)) + '\n',
                            text=True, capture_output=True, timeout=5)
    if result.returncode:
        raise RuntimeError('devfreq_write_failed:' + path.parent.name)


@contextmanager
def temporary_max_vote(*, root=SYSFS, writer=_write_vote):
    """Save both original minima, apply maxima, restore on every exit.

    The result reports readbacks; an accepted minimum is not proof the remote
    governor honoured it. Use the cur_freq trace during the actual operation.
    No persistent file/governor/max_freq change is made.
    """
    before = policies(root)
    intended = []
    record = {'before': before, 'applied': {}, 'restored': {}, 'restore_errors': []}
    try:
        # Validate everything before the first write.
        for node in ('ddr', 'llcc'):
            row = before[node]
            old, target = row['min_freq'], row['max_freq']
            if type(old) is not int or type(target) is not int or old < 0 or target <= 0 or target < old:
                raise ValueError('invalid_devfreq_limits:' + node)
            intended.append((root / node / 'min_freq', old, target))
        touched = []
        try:
            for path, old, target in intended:
                # A failed/partial write must still receive a restore attempt.
                touched.append((path, old))
                writer(path, target)
                record['applied'][path.parent.name] = {
                    'requested': target, 'readback': int(path.read_text().strip())}
            yield record
        finally:
            for path, old in reversed(touched):
                try:
                    writer(path, old)
                    got = int(path.read_text().strip())
                    record['restored'][path.parent.name] = got
                    if got != old:
                        raise RuntimeError('restore_readback_mismatch')
                except BaseException as exc:
                    record['restore_errors'].append({'node': path.parent.name, 'error': type(exc).__name__})
            if record['restore_errors']:
                raise RuntimeError('devfreq_restore_failed:' + json.dumps(record['restore_errors']))
    finally:
        record['after'] = policies(root)


class Sampler:
    """One small polling thread, persistent fds, monotonic deadlines, 0600 JSONL.

    Driver-reported frequency is evidence about its vote/state, not an
    independent measurement of physical memory clocks. Sampling is itself a
    small CPU workload, so keep it identical in the compared arms.
    """
    def __init__(self, out, *, interval_ms=10, root=SYSFS):
        if not 5 <= interval_ms <= 1000:
            raise ValueError('interval_ms_out_of_range')
        self.out, self.root = Path(out), root
        self.period_ns = int(interval_ms * 1_000_000)
        self.stop_event = threading.Event()
        self.thread = self.stream = None
        self.fds = {}
        self.count = 0
        self.error = None

    def start(self):
        fd = os.open(self.out, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        self.stream = os.fdopen(fd, 'w')
        self.stream.write(json.dumps({'event': 'header', 'period_ns': self.period_ns,
                                      'policies': policies(self.root)}) + '\n')
        for node in NODES:
            try:
                self.fds[node] = os.open(self.root / node / 'cur_freq', os.O_RDONLY | os.O_CLOEXEC)
            except OSError:
                self.fds[node] = None
        self.thread = threading.Thread(target=self._run, name='devfreq-10ms', daemon=True)
        self.thread.start()
        return self

    def _run(self):
        deadline = time.monotonic_ns()
        try:
            while not self.stop_event.is_set():
                begin = time.monotonic_ns()
                row = {'event': 'sample', 'monotonic_ns': begin, 'wall_ns': time.time_ns(), 'cur_value': {}}
                for node, fd in self.fds.items():
                    try:
                        if fd is None:
                            raise OSError('missing_cur_freq')
                        os.lseek(fd, 0, os.SEEK_SET)
                        row['cur_value'][node] = int(os.read(fd, 128).strip())
                    except (OSError, ValueError):
                        row['cur_value'][node] = None
                row['read_duration_ns'] = time.monotonic_ns() - begin
                self.stream.write(json.dumps(row, separators=(',', ':')) + '\n')
                self.count += 1
                deadline += self.period_ns
                if deadline < time.monotonic_ns():
                    deadline = time.monotonic_ns() + self.period_ns
                self.stop_event.wait(max(0, deadline - time.monotonic_ns()) / 1e9)
        except BaseException as exc:
            self.error = type(exc).__name__

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=3)
            if self.thread.is_alive():
                raise RuntimeError('devfreq_sampler_not_stopped')
        for fd in self.fds.values():
            if fd is not None:
                os.close(fd)
        self.fds.clear()
        if self.stream:
            self.stream.write(json.dumps({'event': 'end', 'samples': self.count, 'error': self.error}) + '\n')
            self.stream.close()
            self.stream = None
        if self.error:
            raise RuntimeError('devfreq_sampler_failed:' + self.error)
        return self.count

    def __enter__(self):
        return self.start()

    def __exit__(self, *_):
        self.stop()
