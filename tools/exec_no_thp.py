"""Private candidate exec wrapper: verify pinned executable, disable own THP, exec.

Only this wrapper process and its exec replacement change policy. No shell,
preexec_fn, background process, global sysfs or production configuration writes.
"""
import argparse
import ctypes
import hashlib
import os
from pathlib import Path
import re
import stat


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command or re.fullmatch('[0-9a-f]{64}', args.sha256) is None:
        return 78
    path = Path(command[0])
    if not path.is_absolute() or path.resolve() != path:
        return 78
    info = path.lstat()
    if (not stat.S_ISREG(info.st_mode) or info.st_uid not in (0, os.getuid())
            or not os.access(path, os.X_OK)):
        return 78
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while data := stream.read(1024 * 1024):
            digest.update(data)
    if digest.hexdigest() != args.sha256:
        return 78
    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong]
    libc.prctl.restype = ctypes.c_int
    if libc.prctl(41, 1, 0, 0, 0) != 0 or libc.prctl(42, 0, 0, 0, 0) != 1:
        return 70
    os.execv(str(path), command)
    return 70


if __name__ == '__main__':
    raise SystemExit(main())
