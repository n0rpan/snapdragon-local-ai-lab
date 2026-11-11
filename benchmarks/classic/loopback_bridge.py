#!/usr/bin/env python3
"""Bridge a namespace-local TCP port to one owner-only Unix socket."""
import argparse
import errno
import itertools
import os
from pathlib import Path
import signal
import socket
import stat
import threading

STOP = threading.Event()
CHUNK = 64 * 1024
ACTIVE = set()
ACTIVE_LOCK = threading.Lock()
MAX_CONNECTIONS = 8


def request_stop(*_):
    STOP.set()
    with ACTIVE_LOCK:
        for sock in tuple(ACTIVE):
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass


def destination(host, port):
    if host != "127.0.0.1" or not 1 <= port <= 65535:
        raise ValueError("host destination must be 127.0.0.1 and port must be 1..65535")
    return host, port


def owned_directory(path):
    parent = Path(path).absolute().parent
    info = parent.stat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise ValueError("Unix socket parent must be an owner-owned mode-700 directory")
    return parent


def host_listener(path):
    path = Path(path).absolute()
    owned_directory(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to replace existing Unix socket path: {path}")
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    old_umask = os.umask(0o077)
    try:
        listener.bind(str(path))
    except BaseException:
        listener.close()
        raise
    finally:
        os.umask(old_umask)
    try:
        os.chmod(path, 0o600)
        listener.listen(8)
        listener.settimeout(0.5)
    except BaseException:
        listener.close()
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise
    return listener, path, path.stat().st_ino


def unix_client(path):
    info = Path(path).lstat()
    if not stat.S_ISSOCK(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise ValueError("Unix endpoint must be an owner-owned mode-600 socket")
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.connect(str(path))
    return client


def relay(left, right, capture=None):
    """Copy both directions; propagate EOF as a TCP/Unix half-close."""
    failures = []
    captured = 0

    def pump(source, target):
        nonlocal captured
        try:
            while True:
                data = source.recv(CHUNK)
                if not data:
                    try:
                        target.shutdown(socket.SHUT_WR)
                    except OSError:
                        pass
                    return
                if source is left and capture is not None:
                    captured += len(data)
                    if captured > 4 * 1024**2:
                        raise OSError(errno.EFBIG, "private request capture exceeded 4 MiB")
                    capture.write(data)
                target.sendall(data)
        except OSError as exc:
            failures.append(exc)
            for sock in (left, right):
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass

    with ACTIVE_LOCK:
        ACTIVE.update((left, right))
    workers = [threading.Thread(target=pump, args=(left, right), daemon=True),
               threading.Thread(target=pump, args=(right, left), daemon=True)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    with ACTIVE_LOCK:
        ACTIVE.discard(left)
        ACTIVE.discard(right)
    if failures:
        raise failures[0]


def host_mode(path, host, port, capture_dir=None):
    target = destination(host, port)
    captures = itertools.count()
    if capture_dir is not None:
        capture_dir = Path(capture_dir)
        info = capture_dir.lstat()
        if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid()
                or info.st_mode & 0o077 or capture_dir.resolve() != capture_dir):
            raise ValueError("request capture requires an absolute owner-only directory")
    listener, path, inode = host_listener(path)
    print("BRIDGE_READY", flush=True)
    def handle(incoming):
        outgoing = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            outgoing.settimeout(10)
            outgoing.connect(target)
            outgoing.settimeout(None)
            with outgoing:
                if capture_dir is None:
                    relay(incoming, outgoing)
                else:
                    number = next(captures)
                    if number >= 64:
                        raise OSError(errno.EFBIG, "private request capture exceeded 64 connections")
                    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW
                    fd = os.open(capture_dir / f"request-{number:03d}.http", flags, 0o600)
                    with os.fdopen(fd, "wb") as capture:
                        relay(incoming, outgoing, capture)
        except OSError as exc:
            print(f"bridge connection failed: {exc}", flush=True)
    try:
        serve(listener, handle)
    finally:
        listener.close()
        try:
            info = path.lstat()
            if stat.S_ISSOCK(info.st_mode) and info.st_ino == inode:
                path.unlink()
        except FileNotFoundError:
            pass


def inside_mode(path, port):
    if not 1 <= port <= 65535:
        raise ValueError("listen port must be 1..65535")
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", port))
    listener.listen(8)
    listener.settimeout(0.5)
    print("BRIDGE_READY", flush=True)
    def handle(incoming):
        outgoing = unix_client(path)
        with outgoing:
            relay(incoming, outgoing)
    try:
        serve(listener, handle)
    finally:
        listener.close()


def serve(listener, handler):
    """Accept concurrently with a hard connection cap; join workers on stop."""
    workers = set()
    lock = threading.Lock()
    def run(conn):
        try:
            with conn:
                with ACTIVE_LOCK:
                    ACTIVE.add(conn)
                handler(conn)
        except OSError as exc:
            print(f"bridge connection failed: {exc}", flush=True)
        finally:
            with ACTIVE_LOCK:
                ACTIVE.discard(conn)
            with lock:
                workers.discard(threading.current_thread())
    while not STOP.is_set():
        try:
            conn, _ = listener.accept()
        except socket.timeout:
            continue
        with lock:
            if len(workers) >= MAX_CONNECTIONS:
                conn.close()
                continue
            worker = threading.Thread(target=run, args=(conn,), daemon=True)
            workers.add(worker)
            worker.start()
    with ACTIVE_LOCK:
        active = tuple(ACTIVE)
    for conn in active:
        try:
            conn.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
    with lock:
        pending = tuple(workers)
    for worker in pending:
        worker.join(timeout=3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    host = modes.add_parser("host", help="Unix listener forwarding only to IPv4 loopback")
    host.add_argument("--socket", type=Path, required=True)
    host.add_argument("--target-host", default="127.0.0.1")
    host.add_argument("--target-port", type=int, required=True)
    host.add_argument("--capture-dir", type=Path)
    inside = modes.add_parser("inside", help="127.0.0.1 listener forwarding to one Unix socket")
    inside.add_argument("--socket", type=Path, required=True)
    inside.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    try:
        if args.mode == "host":
            host_mode(args.socket, args.target_host, args.target_port, args.capture_dir)
        else:
            inside_mode(args.socket, args.port)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
