"""Host memory stops for one controller-owned native process; no lease or prompt access."""
from collections import deque
import math
import os
from pathlib import Path
import sys
import threading
import time

MIN_AVAILABLE_BYTES = 1536 * 1024**2
PSI_FULL_AVG10_PERCENT = 10.0
PSI_SUSTAINED_SECONDS = 10.0
POLL_SECONDS = .25
TERM_GRACE_SECONDS = 3.0
MAX_LOADING_SECONDS = 300.0
# Flash-only policy (profile key ``flash_memory``): the engine runs in its own scope with
# this limit and a lower host floor. Only the measured pair is accepted; every other model
# keeps the controller unit's limit and MIN_AVAILABLE_BYTES.
FLASH_SCOPE_MAX_BYTES = 29 * 1024**3
FLASH_HOST_FLOOR_BYTES = 1024**3


class MemoryPolicyError(RuntimeError):
    pass


def current_cgroup():
    rows = [line[3:] for line in Path('/proc/self/cgroup').read_text().splitlines()
            if line.startswith('0::')]
    if len(rows) != 1 or not rows[0].startswith('/') or '..' in Path(rows[0]).parts:
        raise ValueError('invalid_current_cgroup')
    return Path('/sys/fs/cgroup') / rows[0].lstrip('/')


def flash_policy(config):
    """None without ``flash_memory``; else the validated Flash scope limit and host floor."""
    value = config.get('flash_memory') if isinstance(config, dict) else None
    if value is None:
        return None
    wanted = {'scope_max_bytes': FLASH_SCOPE_MAX_BYTES, 'host_floor_bytes': FLASH_HOST_FLOOR_BYTES}
    if not isinstance(value, dict) or set(value) != set(wanted) or any(
            type(value[key]) is not int or value[key] != number for key, number in wanted.items()):
        raise ValueError('flash_memory_policy_unqualified')
    return dict(wanted)


def scope_command(args, scope_max_bytes, *, runtime_dir=None):
    """Launch ``args`` through exec_in_scope.py: same PID and environment, own memory limit."""
    if not isinstance(args, list) or not args or not all(isinstance(a, str) for a in args):
        raise ValueError('scope_command_requires_arguments')
    wrapper = Path(__file__).resolve().with_name('exec_in_scope.py')
    directory = runtime_dir or '/run/user/%d' % os.getuid()
    return [sys.executable, '-I', '-B', str(wrapper), '--memory-max', str(scope_max_bytes),
            '--runtime-dir', directory, '--', *args]


def process_cgroup(pid, root=Path('/proc')):
    rows = [line[3:] for line in (Path(root) / str(pid) / 'cgroup').read_text().splitlines()
            if line.startswith('0::')]
    if len(rows) != 1 or not rows[0].startswith('/') or '..' in Path(rows[0]).parts:
        raise ValueError('invalid_process_cgroup')
    return Path('/sys/fs/cgroup') / rows[0].lstrip('/')


def _unsigned(value):
    if not value.isascii() or not value.isdecimal():
        raise ValueError('invalid_memory_counter')
    return int(value)


def parse_available_bytes(text):
    available = [line.split()[1:] for line in text.splitlines()
                 if line.startswith('MemAvailable:')]
    if len(available) != 1 or len(available[0]) != 2 or available[0][1] != 'kB':
        raise ValueError('invalid_available_memory')
    return _unsigned(available[0][0]) * 1024


def parse_psi_full(text):
    full = [line.split()[1:] for line in text.splitlines()
            if line.startswith('full ')]
    if len(full) != 1:
        raise ValueError('missing_full_memory_pressure')
    fields = dict(item.split('=', 1) for item in full[0])
    pressure = float(fields['avg10'])
    if not math.isfinite(pressure) or not 0 <= pressure <= 100:
        raise ValueError('invalid_memory_pressure')
    return pressure, _unsigned(fields['total'])


def parse_psi_full_avg10(text):
    return parse_psi_full(text)[0]


def parse_events(text):
    events = {key: _unsigned(value) for key, value in
              (line.split() for line in text.splitlines())}
    if not {'oom', 'oom_kill'} <= events.keys():
        raise ValueError('missing_oom_counters')
    return events


def read_sample(cgroup):
    """Read only host/cgroup numeric metadata; global swap/pageouts are permitted."""
    cgroup = Path(cgroup)
    pids = dict(line.split() for line in (cgroup / 'pids.events').read_text().splitlines())
    pressure, total = parse_psi_full(Path('/proc/pressure/memory').read_text())
    return {'available_bytes': parse_available_bytes(Path('/proc/meminfo').read_text()),
            'psi_full_avg10': pressure, 'psi_full_total_us': total,
            'swap_current_bytes': _unsigned((cgroup / 'memory.swap.current').read_text().strip()),
            'swap_max': (cgroup / 'memory.swap.max').read_text().strip(),
            'memory_high': (cgroup / 'memory.high').read_text().strip(),
            'memory_max': (cgroup / 'memory.max').read_text().strip(),
            'events': parse_events((cgroup / 'memory.events').read_text()),
            'pids_max_events': _unsigned(pids['max'])}


def event_baseline(sample):
    return {**{key: sample['events'].get(key, 0) for key in ('oom', 'oom_kill', 'oom_group_kill')},
            'pids_max': sample.get('pids_max_events', 0)}


def evaluate_sample(sample, now, pressure_since, baseline, minimum_available=MIN_AVAILABLE_BYTES):
    """Pure policy: return (current stop reason, next sampled high-PSI start).

    Call every250ms. A sample below10% resets the timer. High/max memory events
    and global swap/pageout activity never produce a stop reason.
    """
    if sample['psi_full_avg10'] >= PSI_FULL_AVG10_PERCENT:
        if pressure_since is None:
            pressure_since = now
    else:
        pressure_since = None
    reason = None
    if sample['swap_max'] != '0':
        reason = 'own_swap_limit_not_zero'
    elif sample['swap_current_bytes'] != 0:
        reason = 'own_swap_nonzero'
    elif any(sample['events'].get(key, 0) > baseline.get(key, 0)
             for key in ('oom', 'oom_kill', 'oom_group_kill')):
        reason = 'cgroup_oom'
    elif sample.get('pids_max_events', 0) > baseline.get('pids_max', 0):
        reason = 'cgroup_pids_limit'
    elif sample['available_bytes'] < minimum_available:
        reason = 'host_available_below_floor'
    elif pressure_since is not None and now - pressure_since >= PSI_SUSTAINED_SECONDS:
        reason = 'host_memory_pressure_sustained'
    return reason, pressure_since


class MemoryGuard:
    """Construct before Popen, attach once afterward, stop before normal cleanup.

    The constructor records OOM baselines before the existing native launch. The
    guard only signals that Popen's PID. It never waits, closes a lease, kills a
    process group, starts a model, or reads backend output. The controller keeps
    its lease until its ordinary cleanup has terminated/reaped the native child.
    Optional loading_timeout_seconds exempts only loading PSI until mark_ready()
    or a bounded deadline (maximum 300 seconds); all hard limits remain active.
    An avg10 stop also needs >=10% actual full stalls in a recent >=10s
    counter window sampled after attach/readiness; loading history is excluded.
    """
    def __init__(self, *, reader=None, clock=time.monotonic, loading_timeout_seconds=None,
                 minimum_available_bytes=None, follow_process_cgroup=False):
        # Flash only: a lower host floor, and cgroup counters read from the scope the child
        # moves into (exec_in_scope.py) instead of the controller's unit.
        if minimum_available_bytes is not None and (
                type(minimum_available_bytes) is not int
                or not FLASH_HOST_FLOOR_BYTES <= minimum_available_bytes <= MIN_AVAILABLE_BYTES):
            raise ValueError('minimum_available_must_be_between_flash_floor_and_default')
        self._minimum_policy = MIN_AVAILABLE_BYTES if minimum_available_bytes is None else minimum_available_bytes
        self._follow = follow_process_cgroup is True and reader is None
        self._followed = None
        if loading_timeout_seconds is not None:
            if (isinstance(loading_timeout_seconds, bool) or
                    not isinstance(loading_timeout_seconds, (int, float)) or
                    not math.isfinite(loading_timeout_seconds) or
                    not 0 < loading_timeout_seconds <= MAX_LOADING_SECONDS):
                raise ValueError('loading_timeout_must_be_positive_and_at_most_300_seconds')
        self._loading_timeout_seconds = loading_timeout_seconds
        self._loading = loading_timeout_seconds is not None
        self._loading_deadline = clock() + loading_timeout_seconds if self._loading else None
        cgroup = current_cgroup() if reader is None else None
        self._home_cgroup = cgroup
        self._reader = reader or (lambda: read_sample(cgroup))
        self._clock = clock
        self._stop = threading.Event()
        self._signals = threading.Lock()
        self._metadata = threading.Lock()
        self._thread = None
        self._process = None
        self._pressure_since = None
        self._psi_previous = None
        self._psi_window = deque(maxlen=64)
        self._reset_pressure_window()
        self._failure = None
        self._last = None
        self._minimum_available = None
        try:
            initial = self._reader()
            self._initial_events = event_baseline(initial)
            self._observe(initial)
        except (OSError, ValueError, KeyError, TypeError):
            self._failure = 'memory_metadata_unavailable'
        if self._failure:
            raise MemoryPolicyError(self._failure)

    def _reset_pressure_window(self):
        self._psi_window.clear()
        self._psi_window_seconds = 0.
        self._psi_window_percent = None
        self._psi_corroborated = False

    def _corroborate_pressure(self, sample, now):
        total = sample['psi_full_total_us']
        if type(total) is not int or not 0 <= total <= 2**64 - 1 or not math.isfinite(now):
            raise ValueError('invalid_pressure_counter')
        if self._psi_previous is not None:
            at, previous = self._psi_previous
            if now < at or total < previous:
                raise ValueError('pressure_clock_or_counter_regressed')
        self._psi_previous = (now, total)
        if self._loading:
            self._reset_pressure_window()
            return False
        # Keep the last sample at/before the 10s boundary. Divide by its real
        # elapsed interval; never assume regular polling or interpolate stalls.
        while len(self._psi_window) > 1 and self._psi_window[1][0] <= now - PSI_SUSTAINED_SECONDS:
            self._psi_window.popleft()
        if self._psi_window and self._psi_window[-1][0] == now:
            self._psi_window[-1] = (now, total)
        else:
            if len(self._psi_window) == self._psi_window.maxlen:
                raise ValueError('pressure_sampling_bound_exceeded')
            self._psi_window.append((now, total))
        at, previous = self._psi_window[0]
        self._psi_window_seconds = now - at
        self._psi_window_percent = 100 * (total - previous) / (1e6 * (now - at)) if now > at else None
        self._psi_corroborated = (self._psi_window_seconds >= PSI_SUSTAINED_SECONDS and
                                  self._psi_window_percent >= PSI_FULL_AVG10_PERCENT)
        return self._psi_corroborated

    def _observe(self, sample):
        now = self._clock()
        with self._metadata:
            self._last = sample
            available = sample['available_bytes']
            self._minimum_available = available if self._minimum_available is None else min(self._minimum_available, available)
            reason, self._pressure_since = evaluate_sample(sample, now, self._pressure_since, self._initial_events,
                                                           self._minimum_policy)
            corroborated = self._corroborate_pressure(sample, now)
            if reason == 'host_memory_pressure_sustained' and not corroborated:
                reason = None
            if self._loading:
                # Only loading PSI is exempt: never waive own limits, OOM,
                # metadata failures or the host's hard available-memory floor.
                self._pressure_since = None
                if reason == 'host_memory_pressure_sustained':
                    reason = None
                if reason is None and now >= self._loading_deadline:
                    reason = 'model_loading_timeout'
            if reason and self._failure is None:
                self._failure = reason
            return self._failure

    def mark_ready(self):
        """End explicit loading grace once readiness is observed by the caller.

        Idempotent: later readiness notifications cannot reset runtime PSI.
        Readiness cannot revive an already failed guard or an expired deadline.
        """
        with self._metadata:
            if self._loading:
                if self._clock() >= self._loading_deadline:
                    self._failure = self._failure or 'model_loading_timeout'
                self._loading = False
                self._pressure_since = None
                self._reset_pressure_window()

    def failure(self):
        with self._metadata:
            return self._failure

    def metadata(self):
        with self._metadata:
            sample = self._last or {}
            return {'reason': self._failure,
                    'phase': 'loading' if self._loading else 'ready',
                    'loading_deadline_monotonic': self._loading_deadline,
                    'minimum_available_bytes': self._minimum_available,
                    'available_bytes': sample.get('available_bytes'),
                    'psi_full_avg10': sample.get('psi_full_avg10'),
                    'psi_full_total_us': sample.get('psi_full_total_us'),
                    'psi_window_seconds': self._psi_window_seconds,
                    'psi_window_full_percent': self._psi_window_percent,
                    'psi_stop_corroborated': self._psi_corroborated,
                    'psi_high_seconds': 0 if self._pressure_since is None else max(0, self._clock() - self._pressure_since),
                    'memory_high': sample.get('memory_high'), 'memory_max': sample.get('memory_max'),
                    'swap_current_bytes': sample.get('swap_current_bytes'),
                    'memory_events': dict(sample.get('events', {})),
                    'pids_max_events': sample.get('pids_max_events'),
                    'cgroup': None if self._followed is None else self._followed.name,
                    'policy': {'minimum_available_bytes': self._minimum_policy,
                               'loading_timeout_seconds': self._loading_timeout_seconds,
                               'psi_full_avg10_percent': PSI_FULL_AVG10_PERCENT,
                               'psi_continuous_seconds': PSI_SUSTAINED_SECONDS,
                               'psi_actual_window_min_seconds': PSI_SUSTAINED_SECONDS,
                               'psi_actual_window_min_percent': PSI_FULL_AVG10_PERCENT,
                               'host_swap_activity_allowed': True,
                               'high_and_max_events_are_telemetry': True}}

    def start(self, process):
        if self._thread is not None or self._stop.is_set():
            raise ValueError('guard_can_attach_only_once')
        if not isinstance(process.pid, int) or process.pid <= 0:
            raise ValueError('owned_native_process_required')
        self._process = process
        if self._follow:
            self._reader = self._read_followed
        # Count continuous PSI only while this attached monitor is sampling.
        with self._metadata:
            self._pressure_since = None
            self._psi_previous = None
            self._reset_pressure_window()
            if self._loading:
                self._loading_deadline = self._clock() + self._loading_timeout_seconds
        self._thread = threading.Thread(target=self._run, name='native-memory-guard', daemon=True)
        self._thread.start()

    def _read_followed(self):
        """Sample the child's own cgroup once it has left the controller's unit.

        A fresh scope starts its event counters at zero, so the baseline taken from the
        controller's unit is replaced when the move is first seen. Before the move, and if
        the child's cgroup can no longer be read (it exited), the unit is sampled.
        """
        try:
            cgroup = process_cgroup(self._process.pid)
        except (OSError, ValueError):
            cgroup = self._followed or self._home_cgroup
        if cgroup != self._home_cgroup and cgroup != self._followed:
            sample = read_sample(cgroup)
            with self._metadata:
                self._followed = cgroup
                self._initial_events = {key: 0 for key in self._initial_events}
            return sample
        return read_sample(cgroup)

    def _signal(self, *, kill=False):
        with self._signals:
            if self._stop.is_set() or self._process.poll() is not None:
                return
            try:
                if kill:
                    self._process.kill()
                else:
                    self._process.terminate()
            except ProcessLookupError:
                pass

    def _run(self):
        while not self._stop.is_set():
            # Preserve an OOM/PID reason even if the kernel exited the child
            # between samples. Cgroup metadata belongs to the live controller.
            exited = self._process.poll() is not None
            try:
                reason = self._observe(self._reader())
            except (OSError, ValueError, KeyError, TypeError):
                with self._metadata:
                    self._failure = self._failure or 'memory_metadata_unavailable'
                    reason = self._failure
            if exited:
                return
            if reason:
                self._signal()
                if not self._stop.wait(TERM_GRACE_SECONDS):
                    self._signal(kill=True)
                return
            self._stop.wait(POLL_SECONDS)

    def stop(self):
        self._stop.set()
        # Drain an in-progress signal; no future signal can pass the stopped check.
        with self._signals:
            pass
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=1)
