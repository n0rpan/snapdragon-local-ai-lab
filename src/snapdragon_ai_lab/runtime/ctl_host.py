"""What the controller reads from the host: memory, swap, cgroup limit, a process's resident
size, temperatures and CPU frequencies, and who listens on a port. Numbers and names only.
"""
import errno
import os
from pathlib import Path
import re
import socket
import time
if __package__:
    from .ctl_env import Unavailable
    from .ctl_seams import ctl
    from . import memory_policy
else:
    from ctl_env import Unavailable
    from ctl_seams import ctl
    import memory_policy


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


THERMAL_ROOT = Path('/sys/class/thermal')
CPUFREQ_ROOT = Path('/sys/devices/system/cpu/cpufreq')
MAX_THERMAL_ZONES = 128
MAX_CPU_POLICIES = 16


def _sysfs_int(path):
    try:
        with open(path, 'rb') as stream:
            text = stream.read(32).strip()
        return int(text) if text.lstrip(b'-').isdigit() else None
    except (OSError, ValueError):
        return None


def thermal_sample(thermal_root=None, cpufreq_root=None):
    """ST-4: one sample for a ledger row, numbers only: the hottest thermal zone in degrees C
    (None when none is readable) and each CPU frequency policy's current MHz, in policy order."""
    thermal_root = ctl.THERMAL_ROOT if thermal_root is None else Path(thermal_root)
    cpufreq_root = ctl.CPUFREQ_ROOT if cpufreq_root is None else Path(cpufreq_root)
    hottest = None
    try:
        zones = sorted(d for d in os.listdir(thermal_root) if d.startswith('thermal_zone'))[:ctl.MAX_THERMAL_ZONES]
    except OSError:
        zones = []
    for zone in zones:
        value = _sysfs_int(thermal_root / zone / 'temp')
        if value is not None and -40000 <= value <= 150000 and (hottest is None or value > hottest):
            hottest = value
    clusters = []
    try:
        policies = [d for d in os.listdir(cpufreq_root) if d.startswith('policy') and d[6:].isdigit()]
    except OSError:
        policies = []
    for policy in sorted(policies, key=lambda d: int(d[6:]))[:ctl.MAX_CPU_POLICIES]:
        khz = _sysfs_int(cpufreq_root / policy / 'scaling_cur_freq')
        if khz is not None and 0 < khz < 10**8:
            clusters.append(khz // 1000)
    return {'thermal_max_c': None if hottest is None else round(hottest / 1000, 1), 'cluster_mhz': clusters}


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
    inodes = ctl.listening_inodes(port, proc_root)
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
    inodes = ctl.listening_inodes(port, proc_root)
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
            raise Unavailable(f'Local engine port {port} is in use by {ctl.listening_process(port)}. '
                              f'Close that app or set {setting} to an unused port and restart the chat service.') from None
        raise Unavailable(f'Local engine port {port} is unavailable (error {error.errno}). '
                          f'Check {setting} and restart the chat service.') from None
