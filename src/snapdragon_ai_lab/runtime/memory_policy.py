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
# O-39: host pressure is a stop signal, never admission/prefill throttling.
# The controller opts in explicitly so historical standalone benchmark callers
# retain their original policy until their own qualification is updated.
GUARD_2_VERSION = 'GUARD-2'
HOST_MEMORY_AND_SWAP_FLOOR_BYTES = 2 * 1024**3
HOST_EMERGENCY_AVAILABLE_BYTES = 256 * 1024**2
HOST_FULL_AVG10_PERCENT = 20.0
HOST_SUSTAINED_SECONDS = 3.0
GUARD_2_REVISION = 'GUARD-2b'
LOADING_FULL_AVG10_PERCENT = 60.0
LOADING_PSI_SUSTAINED_SECONDS = 5.0
# Flash-only policy (profile key ``flash_memory``): the engine runs in its own scope with
# this limit and a lower host floor. Only the measured pair is accepted; every other model
# keeps the controller unit's limit and MIN_AVAILABLE_BYTES.
FLASH_SCOPE_MAX_BYTES = 29 * 1024**3
FLASH_HOST_FLOOR_BYTES = 1024**3
# F127 (O-30): opt-in ``swap-tolerant`` mode, only for a preferred 20/18 GiB Flash cache admitted on the
# MemTotal rule. Idle desktop pages are expected to go to swap, so MemAvailable is the wrong signal there.
# The 1 GiB MemAvailable floor is replaced by: engine swap 0, scope + global OOM, sustained PSI full/some
# (after a load grace window) and a last-resort floor on MemAvailable + SwapFree. Defaults come from 064:
# reclaim/load peaked at ~27 % PSI (inside the grace window); loaded full avg10 stayed > 10 % for < 7.1 s
# (some > 10 % for 7.1 s total, max 16.47 %); the desktop stayed usable. Every other mode is unchanged.
SWAP_TOLERANT_DEFAULTS = {
    'psi_full_avg10_percent': 10.0,   # X: full avg10 at/above this ...
    'psi_full_seconds': 20.0,         # Y: ... for this long (plus real stalls >= X over the same window)
    'psi_some_avg10_percent': 30.0,   # Z: some avg10 at/above this ...
    'psi_some_seconds': 30.0,         # W: ... for this long (same counter corroboration)
    'grace_seconds': 60.0,            # PSI stops ignored until max(ready, attach + grace)
    # S3 (v0.1.1): the avg10 track resets on one sample below X, so oscillating thrash (25 % / 9 %) never
    # trips it. A second, counter-only stop cannot be reset by a dip: real stall time (the PSI ``total``
    # counter) over the last ``psi_counter_window_seconds`` >= X (full) or Z (some) percent of wall time.
    # 60 s at X = 10 % means >= 6 s of full stalls per minute (the kernel's avg60 sense).
    'psi_counter_window_seconds': 60.0,
    'last_resort_floor_bytes': 1024**3,  # MemAvailable + SwapFree below this: no RAM and no swap left
}
_SWAP_TOLERANT_BOUNDS = {
    'psi_full_avg10_percent': (1.0, 100.0), 'psi_full_seconds': (1.0, 300.0),
    'psi_some_avg10_percent': (1.0, 100.0), 'psi_some_seconds': (1.0, 300.0),
    'grace_seconds': (0.0, 300.0), 'last_resort_floor_bytes': (512 * 1024**2, 8 * 1024**3),
    'psi_counter_window_seconds': (30.0, 600.0),
}
SWAP_TOLERANT_MODE = 'swap-tolerant'


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


def swap_tolerant_policy(value):
    """The profile's ``oxe_flash.swap_tolerant_guard`` -> None (off, the public default) or a full threshold dict.

    ``true`` uses SWAP_TOLERANT_DEFAULTS; ``{"enabled": true, <threshold>: <number>, ...}`` overrides single values.
    Anything else (unknown keys, bools as numbers, out-of-bounds values) raises ValueError instead of guessing.
    """
    if value is None or value is False:
        return None
    if value is True:
        return dict(SWAP_TOLERANT_DEFAULTS)
    if not isinstance(value, dict) or type(value.get('enabled')) is not bool or \
            set(value) - {'enabled', *SWAP_TOLERANT_DEFAULTS}:
        raise ValueError('swap_tolerant_guard_invalid')
    result = dict(SWAP_TOLERANT_DEFAULTS)
    for key, (low, high) in _SWAP_TOLERANT_BOUNDS.items():
        if key not in value:
            continue
        number = value[key]
        if key == 'last_resort_floor_bytes':
            if type(number) is not int or not low <= number <= high:
                raise ValueError('swap_tolerant_guard_invalid')
        elif type(number) not in (int, float) or not math.isfinite(number) or not low <= number <= high:
            raise ValueError('swap_tolerant_guard_invalid')
        result[key] = number if key == 'last_resort_floor_bytes' else float(number)
    return result if value['enabled'] else None


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


def parse_psi(text, kind):
    rows = [line.split()[1:] for line in text.splitlines()
            if line.startswith(kind + ' ')]
    if len(rows) != 1:
        raise ValueError('missing_%s_memory_pressure' % kind)
    fields = dict(item.split('=', 1) for item in rows[0])
    pressure = float(fields['avg10'])
    if not math.isfinite(pressure) or not 0 <= pressure <= 100:
        raise ValueError('invalid_memory_pressure')
    return pressure, _unsigned(fields['total'])


def parse_psi_full(text):
    return parse_psi(text, 'full')


def parse_swap_free_bytes(text):
    rows = [line.split()[1:] for line in text.splitlines() if line.startswith('SwapFree:')]
    if len(rows) != 1 or len(rows[0]) != 2 or rows[0][1] != 'kB':
        raise ValueError('invalid_swap_free')
    return _unsigned(rows[0][0]) * 1024


def parse_host_oom_kill(text):
    """/proc/vmstat ``oom_kill``: global OOM kills since boot (any cgroup, any process)."""
    rows = [line.split()[1:] for line in text.splitlines() if line.startswith('oom_kill ')]
    if len(rows) != 1 or len(rows[0]) != 1:
        raise ValueError('missing_host_oom_kill')
    return _unsigned(rows[0][0])


def parse_psi_full_avg10(text):
    return parse_psi_full(text)[0]


def parse_events(text):
    events = {key: _unsigned(value) for key, value in
              (line.split() for line in text.splitlines())}
    if not {'oom', 'oom_kill'} <= events.keys():
        raise ValueError('missing_oom_counters')
    return events


def read_sample(cgroup, extended=False):
    """Read only host/cgroup numeric metadata; global swap/pageouts are permitted.

    ``extended`` (swap-tolerant mode) adds PSI some, SwapFree and the global oom_kill counter."""
    cgroup = Path(cgroup)
    pids = dict(line.split() for line in (cgroup / 'pids.events').read_text().splitlines())
    psi_text = Path('/proc/pressure/memory').read_text()
    meminfo = Path('/proc/meminfo').read_text()
    pressure, total = parse_psi_full(psi_text)
    extra = {}
    if extended:
        some, some_total = parse_psi(psi_text, 'some')
        extra = {'psi_some_avg10': some, 'psi_some_total_us': some_total,
                 'swap_free_bytes': parse_swap_free_bytes(meminfo),
                 'host_oom_kill': parse_host_oom_kill(Path('/proc/vmstat').read_text())}
    return {**extra, 'available_bytes': parse_available_bytes(meminfo),
            'psi_full_avg10': pressure, 'psi_full_total_us': total,
            'swap_current_bytes': _unsigned((cgroup / 'memory.swap.current').read_text().strip()),
            'swap_max': (cgroup / 'memory.swap.max').read_text().strip(),
            'memory_high': (cgroup / 'memory.high').read_text().strip(),
            'memory_max': (cgroup / 'memory.max').read_text().strip(),
            'events': parse_events((cgroup / 'memory.events').read_text()),
            'pids_max_events': _unsigned(pids['max'])}


def event_baseline(sample):
    baseline = {**{key: sample['events'].get(key, 0) for key in ('oom', 'oom_kill', 'oom_group_kill')},
                'pids_max': sample.get('pids_max_events', 0)}
    if 'host_oom_kill' in sample:
        baseline['host_oom_kill'] = sample['host_oom_kill']
    return baseline


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


def evaluate_guard2(sample, now, pressure_since, baseline):
    """O-39: continuous unsafe host state for >=3s; own limits fail immediately.

    A continuous OR of the three host conditions starts the timer. A healthy
    sample resets it. Loading/readiness does not waive or reset these checks.
    PSI some, host swapping and high/max cgroup events are telemetry only.
    """
    for key in ('available_bytes', 'swap_free_bytes', 'host_oom_kill'):
        if type(sample[key]) is not int or sample[key] < 0:
            raise ValueError('invalid_host_memory_counter')
    full = sample['psi_full_avg10']
    if type(full) not in (int, float) or not math.isfinite(full) or not 0 <= full <= 100:
        raise ValueError('invalid_memory_pressure')
    if not math.isfinite(now) or (pressure_since is not None and now < pressure_since):
        raise ValueError('pressure_clock_regressed')
    conditions = []
    if sample['available_bytes'] + sample['swap_free_bytes'] < HOST_MEMORY_AND_SWAP_FLOOR_BYTES:
        conditions.append('host_memory_and_swap_exhausted')
    if full > HOST_FULL_AVG10_PERCENT:
        conditions.append('host_memory_pressure_full_sustained')
    if sample['available_bytes'] < HOST_EMERGENCY_AVAILABLE_BYTES:
        conditions.append('host_available_emergency')
    pressure_since = (now if pressure_since is None else pressure_since) if conditions else None
    # Baselines are captured before launch; host OOM remains global across a
    # child's move into its own no-swap scope.
    reason = None
    if sample['swap_max'] != '0':
        reason = 'own_swap_limit_not_zero'
    elif sample['swap_current_bytes'] != 0:
        reason = 'own_swap_nonzero'
    elif any(sample['events'].get(key, 0) > baseline.get(key, 0)
             for key in ('oom', 'oom_kill', 'oom_group_kill')):
        reason = 'cgroup_oom'
    elif sample['host_oom_kill'] > baseline['host_oom_kill']:
        reason = 'host_oom_kill'
    elif sample.get('pids_max_events', 0) > baseline.get('pids_max', 0):
        reason = 'cgroup_pids_limit'
    elif pressure_since is not None and now - pressure_since >= HOST_SUSTAINED_SECONDS:
        reason = conditions[0]
    return reason, pressure_since, conditions


def evaluate_guard2b(sample, now, pressure_since, loading_pressure_since, baseline, loading):
    """GUARD-2b: only loading PSI uses >60%/5s; host floors stay <.../3s.

    Load-time reclaim can stall the loader without stalling an active desktop.
    Separate timers keep a catastrophic loading PSI sample from extending or
    shortening the unchanged host-floor interval. No admission/throttle path.
    """
    full = sample['psi_full_avg10']
    if type(full) not in (int, float) or not math.isfinite(full) or not 0 <= full <= 100:
        raise ValueError('invalid_memory_pressure')
    if not loading:
        reason, since, conditions = evaluate_guard2(sample, now, pressure_since, baseline)
        return reason, since, conditions, None
    # Retain the exact native-limit/OOM and host-floor rules; evaluate only
    # loading PSI on its own threshold and timer.
    reason, since, conditions = evaluate_guard2({**sample, 'psi_full_avg10': 0.}, now,
                                              pressure_since, baseline)
    if loading_pressure_since is not None and now < loading_pressure_since:
        raise ValueError('pressure_clock_regressed')
    if full > LOADING_FULL_AVG10_PERCENT:
        loading_pressure_since = now if loading_pressure_since is None else loading_pressure_since
        conditions.append('host_memory_pressure_full_loading_sustained')
        if reason is None and now - loading_pressure_since >= LOADING_PSI_SUSTAINED_SECONDS:
            reason = 'host_memory_pressure_full_loading_sustained'
    else:
        loading_pressure_since = None
    return reason, since, conditions, loading_pressure_since


class PressureTrack:
    """One sustained-PSI condition, two ways to trip (either is enough):

    1. avg10 >= percent continuously for ``seconds`` (a sample below resets it), corroborated by real
       stalls from the ``total`` counter: >= percent over a window of >= ``seconds``;
    2. (S3) counter only: real stalls >= percent over a window of >= ``long_seconds``. A dip in avg10
       cannot reset this one, so oscillating thrash still trips it.
    Divides by the real elapsed interval; never assumes regular polling."""
    def __init__(self, percent, seconds, long_seconds=None):
        if long_seconds is not None and long_seconds < seconds:
            raise ValueError('pressure_long_window_shorter_than_short_window')
        self.percent, self.seconds, self.long_seconds = percent, seconds, long_seconds
        self.reset()

    def reset(self):
        self.since = None
        self.previous = None
        self.window = deque()
        self.window_seconds = 0.
        self.window_percent = None
        self.long_window = deque()
        self.long_window_seconds = 0.
        self.long_window_percent = None
        self.tripped_by = None

    @staticmethod
    def _slide(window, span, now, total):
        # Keep the last sample at/before the window boundary, so the measured span is >= ``span``.
        while len(window) > 1 and window[1][0] <= now - span:
            window.popleft()
        if window and window[-1][0] == now:
            window[-1] = (now, total)
        else:
            window.append((now, total))
        at, first = window[0]
        return now - at, (100 * (total - first) / (1e6 * (now - at)) if now > at else None)

    def update(self, avg10, total, now):
        if type(total) is not int or not 0 <= total <= 2**64 - 1 or not math.isfinite(now) or \
                type(avg10) not in (int, float) or not math.isfinite(avg10):
            raise ValueError('invalid_pressure_counter')
        if self.previous is not None and (now < self.previous[0] or total < self.previous[1]):
            raise ValueError('pressure_clock_or_counter_regressed')
        self.previous = (now, total)
        if avg10 >= self.percent:
            self.since = now if self.since is None else self.since
        else:
            self.since = None
        self.window_seconds, self.window_percent = self._slide(self.window, self.seconds, now, total)
        short = (self.since is not None and now - self.since >= self.seconds
                 and self.window_seconds >= self.seconds and self.window_percent >= self.percent)
        long = False
        if self.long_seconds is not None:
            self.long_window_seconds, self.long_window_percent = self._slide(
                self.long_window, self.long_seconds, now, total)
            long = (self.long_window_seconds >= self.long_seconds
                    and self.long_window_percent is not None and self.long_window_percent >= self.percent)
        self.tripped_by = 'avg10' if short else 'counter' if long else None
        return short or long

    def high_seconds(self, now):
        return 0 if self.since is None else max(0, now - self.since)


def evaluate_swap_tolerant(sample, now, baseline, policy, tracks, in_grace):
    """Pure swap-tolerant policy (F127). Returns the stop reason or None. ``tracks`` = {'full','some'}
    PressureTrack; reset (not evaluated) while ``in_grace``. MemAvailable alone never stops."""
    full = tracks['full'].update(sample['psi_full_avg10'], sample['psi_full_total_us'], now)
    some = tracks['some'].update(sample['psi_some_avg10'], sample['psi_some_total_us'], now)
    if in_grace:
        tracks['full'].reset()
        tracks['some'].reset()
        full = some = False
    available = sample['available_bytes'] + sample['swap_free_bytes']
    if type(sample['host_oom_kill']) is not int or 'host_oom_kill' not in baseline:
        raise ValueError('missing_host_oom_kill')
    if sample['swap_max'] != '0':
        return 'own_swap_limit_not_zero'
    if sample['swap_current_bytes'] != 0:
        return 'own_swap_nonzero'
    if any(sample['events'].get(key, 0) > baseline.get(key, 0)
           for key in ('oom', 'oom_kill', 'oom_group_kill')):
        return 'cgroup_oom'
    if sample['host_oom_kill'] > baseline['host_oom_kill']:
        return 'host_oom_kill'
    if sample.get('pids_max_events', 0) > baseline.get('pids_max', 0):
        return 'cgroup_pids_limit'
    if available < policy['last_resort_floor_bytes']:
        return 'host_memory_and_swap_exhausted'
    if full:
        return 'host_memory_pressure_full_sustained'
    if some:
        return 'host_memory_pressure_some_sustained'
    return None


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

    ``swap_tolerant`` (a swap_tolerant_policy() dict; Flash pref 20/18 on the MemTotal rule only) switches
    to evaluate_swap_tolerant(): no MemAvailable floor; PSI stops wait for max(readiness, attach + grace).
    leave_swap_tolerant() returns to the standard floor if the engine did not load swap-tolerant.

    The controller explicitly selects host_guard_version='GUARD-2' (O-39).
    That replaces both legacy host-stop modes with evaluate_guard2(), including
    during loading. Standalone historical harnesses keep their legacy defaults.
    No mode changes admission, splits prefill passes or throttles the engine.
    """
    def __init__(self, *, reader=None, clock=time.monotonic, loading_timeout_seconds=None,
                 minimum_available_bytes=None, follow_process_cgroup=False, swap_tolerant=None,
                 host_guard_version=None):
        if host_guard_version not in (None, GUARD_2_VERSION):
            raise ValueError('unknown_host_guard_version')
        if host_guard_version is not None and swap_tolerant is not None:
            raise ValueError('host_guard_policies_are_mutually_exclusive')
        self._guard2 = host_guard_version == GUARD_2_VERSION
        self._host_conditions = []
        self._loading_pressure_since = None
        if swap_tolerant is not None and (not isinstance(swap_tolerant, dict) or
                                          swap_tolerant_policy({'enabled': True, **swap_tolerant}) != swap_tolerant):
            raise ValueError('swap_tolerant_guard_invalid')
        self._swap = None if swap_tolerant is None else dict(swap_tolerant)
        self._tracks = None if self._swap is None else {
            'full': PressureTrack(self._swap['psi_full_avg10_percent'], self._swap['psi_full_seconds'],
                                  max(self._swap['psi_full_seconds'], self._swap['psi_counter_window_seconds'])),
            'some': PressureTrack(self._swap['psi_some_avg10_percent'], self._swap['psi_some_seconds'],
                                  max(self._swap['psi_some_seconds'], self._swap['psi_counter_window_seconds']))}
        self._attached_at = None
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
        self._extended = self._swap is not None or self._guard2
        self._reader = reader or (lambda: self._read_cgroup(cgroup))
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
            if self._guard2:
                reason, self._pressure_since, self._host_conditions, self._loading_pressure_since = evaluate_guard2b(
                    sample, now, self._pressure_since, self._loading_pressure_since,
                    self._initial_events, self._loading)
                if reason is None and self._loading and now >= self._loading_deadline:
                    reason = 'model_loading_timeout'
                if reason and self._failure is None:
                    self._failure = reason
                return self._failure
            if self._swap is not None:
                in_grace = (self._loading or self._attached_at is None
                            or now < self._attached_at + self._swap['grace_seconds'])
                reason = evaluate_swap_tolerant(sample, now, self._initial_events, self._swap,
                                                self._tracks, in_grace)
                if self._loading and reason is None and now >= self._loading_deadline:
                    reason = 'model_loading_timeout'
                if reason and self._failure is None:
                    self._failure = reason
                return self._failure
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
                self._loading_pressure_since = None
                self._host_conditions = [condition for condition in self._host_conditions
                                        if condition != 'host_memory_pressure_full_loading_sustained']
                if not self._guard2:
                    self._pressure_since = None
                self._reset_pressure_window()

    def failure(self):
        with self._metadata:
            return self._failure

    def mode(self):
        with self._metadata:
            return 'standard' if self._swap is None else SWAP_TOLERANT_MODE

    def leave_swap_tolerant(self):
        """Back to the standard floor/PSI policy (the engine did not report preferred-swap-tolerant).
        One-way: a guard never enters swap-tolerant mode after construction."""
        with self._metadata:
            if self._swap is None:
                return
            self._swap = None
            self._tracks = None
            self._pressure_since = None
            self._psi_previous = None
            self._reset_pressure_window()

    def metadata(self):
        with self._metadata:
            sample = self._last or {}
            swap = None
            if self._swap is not None:
                now = self._clock()
                swap = {**self._swap,
                        'grace_ends_monotonic': None if self._attached_at is None
                        else self._attached_at + self._swap['grace_seconds'],
                        'psi_some_avg10': sample.get('psi_some_avg10'),
                        'swap_free_bytes': sample.get('swap_free_bytes'),
                        'host_oom_kill': sample.get('host_oom_kill'),
                        'host_oom_kill_baseline': self._initial_events.get('host_oom_kill'),
                        **{f'psi_{name}_{field}': value for name, track in self._tracks.items()
                           for field, value in (('high_seconds', track.high_seconds(now)),
                                                ('window_seconds', track.window_seconds),
                                                ('window_percent', track.window_percent),
                                                ('counter_window_seconds', track.long_window_seconds),
                                                ('counter_window_percent', track.long_window_percent),
                                                ('tripped_by', track.tripped_by))}}
            return {'reason': self._failure,
                    'guard_version': GUARD_2_VERSION if self._guard2 else None,
                    'guard_revision': GUARD_2_REVISION if self._guard2 else None,
                    'mode': 'standard' if self._swap is None else SWAP_TOLERANT_MODE,
                    'swap_tolerant': swap,
                    'phase': 'loading' if self._loading else 'ready',
                    'loading_deadline_monotonic': self._loading_deadline,
                    'minimum_available_bytes': self._minimum_available,
                    'available_bytes': sample.get('available_bytes'),
                    'swap_free_bytes': sample.get('swap_free_bytes'),
                    'psi_some_avg10': sample.get('psi_some_avg10'),
                    'psi_some_total_us': sample.get('psi_some_total_us'),
                    'host_oom_kill': sample.get('host_oom_kill'),
                    'host_high_seconds': 0 if self._pressure_since is None else max(0, self._clock() - self._pressure_since),
                    'loading_psi_high_seconds': 0 if self._loading_pressure_since is None else max(0, self._clock() - self._loading_pressure_since),
                    'host_trigger_conditions': list(self._host_conditions),
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
                    # F127-7: no MemAvailable floor is enforced in swap-tolerant mode (its last-resort floor
                    # on MemAvailable + SwapFree is under swap_tolerant), so do not report one.
                    'policy': {'minimum_available_bytes': None if self._swap is not None or self._guard2 else self._minimum_policy,
                               'loading_timeout_seconds': self._loading_timeout_seconds,
                               'psi_full_avg10_percent': HOST_FULL_AVG10_PERCENT if self._guard2 else PSI_FULL_AVG10_PERCENT,
                               **({'host_memory_and_swap_floor_bytes': HOST_MEMORY_AND_SWAP_FLOOR_BYTES,
                                   'host_emergency_available_bytes': HOST_EMERGENCY_AVAILABLE_BYTES,
                                   'host_continuous_seconds': HOST_SUSTAINED_SECONDS,
                                   'loading_psi_full_avg10_percent': LOADING_FULL_AVG10_PERCENT,
                                   'loading_psi_continuous_seconds': LOADING_PSI_SUSTAINED_SECONDS,
                                   'phase_psi_full_avg10_percent': LOADING_FULL_AVG10_PERCENT if self._loading else HOST_FULL_AVG10_PERCENT,
                                   'phase_psi_continuous_seconds': LOADING_PSI_SUSTAINED_SECONDS if self._loading else HOST_SUSTAINED_SECONDS,
                                   'threshold_comparisons': 'strict',
                                   'stop_only': True} if self._guard2 else {}),
                               'psi_continuous_seconds': HOST_SUSTAINED_SECONDS if self._guard2 else PSI_SUSTAINED_SECONDS,
                               'psi_actual_window_min_seconds': None if self._guard2 else PSI_SUSTAINED_SECONDS,
                               'psi_actual_window_min_percent': None if self._guard2 else PSI_FULL_AVG10_PERCENT,
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
        # Legacy PSI starts at attachment. GUARD-2 preserves the host interval
        # already observed at construction; attachment/readiness never waive it.
        with self._metadata:
            if not self._guard2:
                self._pressure_since = None
            self._psi_previous = None
            self._reset_pressure_window()
            self._attached_at = self._clock()
            if self._tracks is not None:
                for track in self._tracks.values():
                    track.reset()
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
            sample = self._read_cgroup(cgroup)
            with self._metadata:
                self._followed = cgroup
                # Host counters (global oom_kill) keep their pre-launch baseline.
                self._initial_events = {key: value if key == 'host_oom_kill' else 0
                                        for key, value in self._initial_events.items()}
            return sample
        return self._read_cgroup(cgroup)

    def _read_cgroup(self, cgroup):
        # Fixed at construction: a guard that left swap-tolerant mode keeps reading the extra fields.
        return read_sample(cgroup, extended=True) if self._extended else read_sample(cgroup)

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
