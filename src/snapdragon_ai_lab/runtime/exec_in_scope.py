"""Flash-only exec wrapper: move this process into its own memory-limited scope, exec.

The controller's unit keeps its own MemoryMax for every other model. A Flash
child starts through this wrapper, which asks the user's systemd manager for a
transient scope holding only this PID (MemoryMax as given, MemorySwapMax 0),
verifies from its own cgroup files that both limits apply, then execs the
command with the environment it was given, unchanged. The PID and parent stay
the controller's child. Any failure exits 78 before the command runs.
No shell, no background process, no unit or configuration file is written.
"""
import argparse
import os
from pathlib import Path
import subprocess
import select
import time

GIB = 1024 ** 3
BUSCTL = '/usr/bin/busctl'
MINIMUM, MAXIMUM = 8 * GIB, 29 * GIB


def own_cgroup(root=Path('/proc/self/cgroup')):
    rows = [line[3:] for line in Path(root).read_text().splitlines() if line.startswith('0::')]
    if len(rows) != 1 or not rows[0].startswith('/') or '..' in Path(rows[0]).parts:
        raise ValueError('invalid_own_cgroup')
    return Path('/sys/fs/cgroup') / rows[0].lstrip('/')


def scope_call(name, pid, memory_max):
    """busctl arguments for StartTransientUnit of a scope that adopts one PID."""
    return [BUSCTL, '--user', '--quiet', 'call', 'org.freedesktop.systemd1', '/org/freedesktop/systemd1',
            'org.freedesktop.systemd1.Manager', 'StartTransientUnit', 'ssa(sv)a(sa(sv))', name, 'fail',
            '4', 'Description', 's', 'Local AI Flash engine (own memory limit)',
            'PIDs', 'au', '1', str(pid), 'MemoryMax', 't', str(memory_max), 'MemorySwapMax', 't', '0', '0']


def limits_apply(cgroup, name, memory_max):
    try:
        return (cgroup.name == name and (cgroup / 'memory.max').read_text().strip() == str(memory_max)
                and (cgroup / 'memory.swap.max').read_text().strip() == '0')
    except OSError:
        return False


def enter_scope(memory_max, runtime_dir, *, run=subprocess.run, cgroup=own_cgroup, clock=time.monotonic,
                sleep=time.sleep, timeout=5.0):
    """Return True once this process is alone in a scope with exactly these limits."""
    name = 'local-ai-flash-%d.scope' % os.getpid()
    # Only the bus lookup sees the session variables; the command's environment is not changed.
    bus_environment = {'PATH': '/usr/bin:/bin', 'XDG_RUNTIME_DIR': runtime_dir}
    try:
        done = run(scope_call(name, os.getpid(), memory_max), env=bus_environment, stdin=subprocess.DEVNULL,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return False
    if done.returncode != 0:
        return False
    deadline = clock() + timeout
    while True:
        try:
            if limits_apply(cgroup(), name, memory_max):
                return True
        except (OSError, ValueError):
            pass
        if clock() >= deadline:
            return False
        sleep(.02)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--memory-max', type=int, required=True)
    parser.add_argument('--runtime-dir', required=True)
    parser.add_argument('--ready-fd', type=int)
    parser.add_argument('--go-fd', type=int)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    runtime_dir = Path(args.runtime_dir)
    if ((args.ready_fd is None) != (args.go_fd is None)
            or not command or not Path(command[0]).is_absolute() or not MINIMUM <= args.memory_max <= MAXIMUM
            or not runtime_dir.is_absolute() or runtime_dir != Path('/run/user') / str(os.getuid())):
        return 78
    if not enter_scope(args.memory_max, str(runtime_dir)):
        return 78
    if args.ready_fd is not None:
        # Optional checked-worker handshake: no native executable runs before
        # its parent has identified this scope and attached its memory guard.
        try:
            os.write(args.ready_fd, b'R')
            os.close(args.ready_fd)
            ready, _, _ = select.select([args.go_fd], [], [], 8)
            if not ready or os.read(args.go_fd, 1) != b'G':
                return 78
        finally:
            os.close(args.go_fd)
    os.execv(command[0], command)
    return 70


if __name__ == '__main__':
    raise SystemExit(main())
