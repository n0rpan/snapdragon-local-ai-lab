"""Read-only, prompt-free telemetry for a guarded inference runner.

This module owns no lease, model process, governor, or memory policy. Callers
snapshot around their existing guarded run and may sample the same API every10s.
"""
from datetime import datetime, timezone
import errno
import math
from pathlib import Path, PurePosixPath
import re
import time


VM_PREFIXES = ('pswp', 'zswp', 'allocstall', 'workingset_refault', 'pgmajfault', 'oom_kill')
DEVFREQ = ('ddr', 'llcc', 'ddr-qos')


def parse_pairs(text):
    return {key: int(value) for key, value in (line.split() for line in text.splitlines() if line.strip())}


def parse_trans_stat(text):
    """Use each row's frequency and last time(ms) column, not its packed header."""
    if 'time(ms)' not in text:
        raise ValueError('unknown_residency_time_unit')
    states, current = {}, None
    for line in text.splitlines():
        match = re.fullmatch(r'\s*(\*)?(\d+):\s*(\d+(?:\s+\d+)*)\s*', line)
        if match:
            frequency = int(match[2])
            if frequency in states:
                raise ValueError('duplicate_residency_state')
            states[frequency] = int(match[3].split()[-1])
            if match[1]:
                current = frequency
    if not states:
        raise ValueError('residency_states_missing')
    return {'time_unit': 'ms', 'time_ms_by_freq_hz': {str(k): v for k, v in states.items()},
            'max_freq_hz': max(states), 'current_freq_hz': current}


def parse_psi(text):
    result = {}
    for line in text.splitlines():
        kind, *values = line.split()
        if kind not in ('some', 'full'):
            continue
        fields = dict(value.split('=', 1) for value in values)
        result[kind] = {key: int(value) if key == 'total' else float(value)
                        for key, value in fields.items() if key in ('total', 'avg10', 'avg60', 'avg300')}
        if 'total' not in result[kind] or any(not math.isfinite(v) or v < 0 for v in result[kind].values()):
            raise ValueError('invalid_pressure_counter')
    if not result:
        raise ValueError('pressure_counters_missing')
    return result


def _unavailable(reason):
    return {'available': False, 'reason': reason}


def _read(path, parser=str.strip):
    try:
        with path.open() as stream:
            text = stream.read(65537)
        if len(text) > 65536:
            return _unavailable('source_exceeds_64k')
        return {'available': True, 'value': parser(text)}
    except OSError as error:
        return _unavailable(errno.errorcode.get(error.errno, 'OSERROR'))
    except (ValueError, TypeError, OverflowError):
        return _unavailable('unrecognized_counter_format')


def _cgroup_path(value):
    path = PurePosixPath(value)
    if not path.is_absolute() or '..' in path.parts or not path.is_relative_to('/sys/fs/cgroup'):
        raise ValueError('cgroup_path_required')
    return path


def cgroup_for_pid(pid, root=Path('/')):
    """Resolve only an explicitly supplied PID; never inspect argv or environment."""
    if type(pid) is not int or pid < 1:
        raise ValueError('positive_pid_required')
    for line in (Path(root) / f'proc/{pid}/cgroup').read_text().splitlines():
        if line.startswith('0::'):
            tail = PurePosixPath(line[3:])
            if not tail.is_absolute() or '..' in tail.parts:
                raise ValueError('invalid_process_cgroup')
            return str(_cgroup_path('/sys/fs/cgroup' + str(tail)))
    raise ValueError('unified_cgroup_missing')


def snapshot(root=Path('/'), cgroups=()):
    """All filesystem accesses are reads. Missing counters remain unavailable."""
    root = Path(root)
    started = time.monotonic()
    def read(relative, parser=str.strip):
        return _read(root / str(relative).lstrip('/'), parser)
    def integer(text):
        return int(text.strip())
    def meminfo(text):
        wanted = {'MemAvailable', 'SwapTotal', 'SwapFree', 'AnonPages', 'Cached'}
        return {key: int(rest.split()[0]) * 1024 for key, rest in
                (line.split(':', 1) for line in text.splitlines()) if key in wanted}
    result = {
        'schema': 1, 'utc': datetime.now(timezone.utc).isoformat(), 'monotonic_s': started,
        'vmstat': read('proc/vmstat', lambda text: {k: v for k, v in parse_pairs(text).items()
                                                 if k.startswith(VM_PREFIXES)}),
        'memory_bytes': read('proc/meminfo', meminfo),
        'psi': {kind: read(f'proc/pressure/{kind}', parse_psi) for kind in ('memory', 'cpu', 'io')},
        'devfreq': {name: read(f'sys/class/devfreq/{name}/trans_stat', parse_trans_stat) for name in DEVFREQ},
        'zswap': {name: read(f'sys/module/zswap/parameters/{name}')
                  for name in ('enabled', 'max_pool_percent', 'shrinker_enabled', 'compressor')},
        'cpu_policies': {}, 'thermal': {}, 'cgroups': {},
    }
    for policy in sorted((root / 'sys/devices/system/cpu/cpufreq').glob('policy*')):
        result['cpu_policies'][policy.name] = {
            name: _read(policy / name, integer) for name in ('cpuinfo_avg_freq', 'scaling_cur_freq')}
    for zone in sorted((root / 'sys/class/thermal').glob('thermal_zone*')):
        result['thermal'][zone.name] = {'type': _read(zone / 'type'), 'temp_millic': _read(zone / 'temp', integer)}
    for raw in cgroups:
        group = _cgroup_path(raw)
        result['cgroups'][str(group)] = {
            name: read(str(group / name), parse_pairs)
            for name in ('cpu.stat', 'memory.events', 'memory.stat')}
        result['cgroups'][str(group)].update({
            name: read(str(group / name), integer)
            for name in ('memory.current', 'memory.swap.current', 'memory.peak')})
    result['read_wall_s'] = time.monotonic() - started
    return result


def _counter_delta(before, after):
    if not before.get('available') or not after.get('available'):
        return _unavailable('source_unavailable')
    a, b = before['value'], after['value']
    common = a.keys() & b.keys()
    reset = sorted(k for k in common if b[k] < a[k])
    return {'available': True, 'delta': {k: b[k] - a[k] for k in sorted(common) if k not in reset},
            'reset': reset, 'missing_before': sorted(b.keys() - a.keys()),
            'missing_after': sorted(a.keys() - b.keys())}


def difference(before, after):
    """Counter deltas, actual PSI stall percentage, and O-14 residency evidence."""
    elapsed = after['monotonic_s'] - before['monotonic_s']
    if elapsed <= 0:
        raise ValueError('positive_snapshot_interval_required')
    result = {'schema': 1, 'elapsed_s': elapsed,
              'vmstat': _counter_delta(before['vmstat'], after['vmstat']),
              'psi': {}, 'devfreq': {}, 'cgroups': {}}
    for name in ('memory', 'cpu', 'io'):
        a, b = before['psi'][name], after['psi'][name]
        if not a.get('available') or not b.get('available'):
            result['psi'][name] = _unavailable('source_unavailable')
            continue
        values = {}
        for kind in a['value'].keys() & b['value'].keys():
            delta = b['value'][kind]['total'] - a['value'][kind]['total']
            values[kind] = ({'total_us': delta, 'stall_percent': 100 * delta / (elapsed * 1e6)}
                            if delta >= 0 else _unavailable('counter_reset'))
        result['psi'][name] = {'available': True, 'value': values}
    for name in DEVFREQ:
        a, b = before['devfreq'][name], after['devfreq'][name]
        if not a.get('available') or not b.get('available'):
            result['devfreq'][name] = _unavailable('source_unavailable')
            continue
        x, y = a['value']['time_ms_by_freq_hz'], b['value']['time_ms_by_freq_hz']
        if x.keys() != y.keys() or any(y[k] < x[k] for k in x):
            result['devfreq'][name] = _unavailable('counter_reset_or_state_change')
            continue
        delta = {k: y[k] - x[k] for k in x}
        maximum = max(map(int, delta))
        below_ms = sum(v for k, v in delta.items() if int(k) < maximum)
        fraction = below_ms / (elapsed * 1000)
        coverage = sum(delta.values()) / (elapsed * 1000)
        usable = elapsed >= 1 and 0.9 <= coverage <= 1.1
        result['devfreq'][name] = {'available': True, 'time_ms_by_freq_hz': delta,
                                  'max_freq_hz': maximum, 'below_max_fraction': fraction,
                                  'coverage_fraction': coverage, 'usable_for_o14': usable,
                                  'below_max_exceeds_2percent': usable and fraction > 0.02}
    for group in before.get('cgroups', {}).keys() | after.get('cgroups', {}).keys():
        a, b = before.get('cgroups', {}).get(group), after.get('cgroups', {}).get(group)
        if a is None or b is None:
            result['cgroups'][group] = _unavailable('cgroup_missing_at_one_snapshot')
        else:
            result['cgroups'][group] = {name: _counter_delta(a[name], b[name])
                                        for name in ('cpu.stat', 'memory.events')}
            stats = []
            for frame in (a, b):
                entry = frame['memory.stat']
                stats.append({'available': True, 'value': {
                    key: value for key, value in entry['value'].items()
                    if key.startswith(('pgfault', 'pgmajfault', 'workingset_refault', 'pgscan', 'pgsteal'))}}
                    if entry.get('available') else entry)
            result['cgroups'][group]['memory_stat_counters'] = _counter_delta(*stats)
    vm = result['vmstat']
    result['swapin_pages_per_s'] = vm.get('delta', {}).get('pswpin', 0) / elapsed if (
        vm.get('available') and 'pswpin' in vm.get('delta', {})) else None
    return result
