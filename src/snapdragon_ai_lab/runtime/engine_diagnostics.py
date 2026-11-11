"""Bounded, private capture of a local engine's stderr pipe."""

from __future__ import annotations

import os
import select
import stat
import threading
from pathlib import Path
from typing import BinaryIO


_MAX_WRITE_BYTES = 64 * 1024
_MAX_INPUT_BYTES = 15 * 1024
_READ_BYTES = 4096
_MAX_LINE_BYTES = 8192
_POLL_SECONDS = 0.2
_MAX_STOP_DRAIN_BYTES = 256 * 1024


def _safe_text(data: bytes) -> bytes:
    """Decode with replacement and render controls visibly before writing."""
    text = data.decode("utf-8", errors="replace")
    out: list[str] = []
    for char in text:
        code = ord(char)
        if char in "\n\t":
            out.append(char)
        elif code < 32 or code == 127 or 0x80 <= code < 0xA0:
            out.append(f"\\x{code:02x}")
        else:
            out.append(char)
    encoded = "".join(out).encode("utf-8", errors="replace")
    if len(encoded) > _MAX_WRITE_BYTES:
        encoded = encoded[:_MAX_WRITE_BYTES]
        while True:
            try:
                encoded.decode("utf-8")
                break
            except UnicodeDecodeError:
                encoded = encoded[:-1]
        encoded += b"...[write truncated]\n"
    return encoded


class EngineLogWriter:
    """Append bounded diagnostic records to a private rotating log.

    Errors opening, rotating, or writing the log are kept in ``error`` and
    swallowed so diagnostic storage cannot fail an inference request.
    """

    def __init__(self, path: str | os.PathLike[str], max_bytes: int = 1024 * 1024,
                 backups: int = 2):
        if max_bytes < 1 or backups < 0:
            raise ValueError("max_bytes must be positive and backups nonnegative")
        self.path = Path(path)
        self.max_bytes = int(max_bytes)
        self.backups = int(backups)
        self._lock = threading.Lock()
        self._error_lock = threading.Lock()
        self._error: BaseException | None = None
        try:
            self._prepare_parent()
        except BaseException as exc:
            self._remember_error(exc)

    @property
    def error(self) -> BaseException | None:
        """The first filesystem or path-safety error, if any."""
        return self._error

    def _remember_error(self, exc: BaseException) -> None:
        with self._error_lock:
            if self._error is None:
                self._error = exc

    def _prepare_parent(self) -> None:
        if not self.path.is_absolute():
            raise ValueError("diagnostic log path must be absolute")
        if ".." in self.path.parts:
            raise PermissionError("diagnostic log path cannot traverse parent directories")
        parent = self.path.parent
        current = Path(parent.anchor)
        parts = parent.parts[1:]
        if not parts:
            raise PermissionError("diagnostic log cannot be placed directly in a filesystem root")
        euid = os.geteuid()
        for index, part in enumerate(parts):
            current = current / part
            try:
                info = current.lstat()
            except FileNotFoundError:
                current.mkdir(mode=0o700)
                info = current.lstat()
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                raise OSError(f"unsafe diagnostic directory: {current}")
            is_leaf = index == len(parts) - 1
            if is_leaf:
                if info.st_uid != euid:
                    raise PermissionError(f"diagnostic directory is not user-owned: {current}")
                os.chmod(current, 0o700, follow_symlinks=False)
            elif info.st_uid not in (0, euid):
                raise PermissionError(f"foreign diagnostic path component: {current}")
            elif info.st_mode & 0o022 and not (info.st_uid == 0 and info.st_mode & stat.S_ISVTX):
                raise PermissionError(f"writable diagnostic path component: {current}")

    def _check_regular_owned(self, path: Path) -> os.stat_result | None:
        try:
            info = path.lstat()
        except FileNotFoundError:
            return None
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise OSError(f"unsafe diagnostic log file: {path}")
        if info.st_uid != os.geteuid():
            raise PermissionError(f"diagnostic log file is not user-owned: {path}")
        if info.st_nlink != 1:
            raise PermissionError(f"diagnostic log file has unexpected hard links: {path}")
        return info

    def _rotate(self) -> None:
        self._check_regular_owned(self.path)
        for index in range(self.backups, 1, -1):
            target = self.path.with_name(self.path.name + f".{index}")
            self._check_regular_owned(target)
            source = self.path.with_name(self.path.name + f".{index - 1}")
            if index == self.backups:
                try:
                    target.unlink()
                except FileNotFoundError:
                    pass
            if self._check_regular_owned(source) is not None:
                os.replace(source, target)
        if self.backups:
            first = self.path.with_name(self.path.name + ".1")
            self._check_regular_owned(first)
        if self._check_regular_owned(self.path) is not None and self.backups:
            os.replace(self.path, self.path.with_name(self.path.name + ".1"))
        elif self.backups == 0:
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass

    def write(self, bytes_or_str: bytes | str) -> None:
        """Write one bounded record; filesystem errors never escape."""
        try:
            truncated = len(bytes_or_str) > _MAX_INPUT_BYTES
            if isinstance(bytes_or_str, str):
                bounded = bytes_or_str[:_MAX_INPUT_BYTES]
                data = bounded.encode("utf-8", errors="replace")
            else:
                data = memoryview(bytes_or_str)[:_MAX_INPUT_BYTES].tobytes()
            data = _safe_text(data)
            if truncated:
                data += b"...[input truncated]\n"
            if len(data) > self.max_bytes:
                data = data[:self.max_bytes].decode("utf-8", errors="ignore").encode("utf-8")
            if not data:
                return
            with self._lock:
                if self._error is not None:
                    return
                self._prepare_parent()
                info = self._check_regular_owned(self.path)
                if info is not None and info.st_mode & 0o777 != 0o600:
                    os.chmod(self.path, 0o600, follow_symlinks=False)
                    info = self._check_regular_owned(self.path)
                if info is not None and info.st_size + len(data) > self.max_bytes:
                    self._rotate()
                flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
                if hasattr(os, "O_NOFOLLOW"):
                    flags |= os.O_NOFOLLOW
                fd = os.open(self.path, flags, 0o600)
                try:
                    opened = os.fstat(fd)
                    if not stat.S_ISREG(opened.st_mode) or opened.st_uid != os.geteuid():
                        raise PermissionError("opened diagnostic log is unsafe")
                    os.fchmod(fd, 0o600)
                    remaining = memoryview(data)
                    while remaining:
                        written = os.write(fd, remaining)
                        if written <= 0:
                            raise OSError("short write to diagnostic log")
                        remaining = remaining[written:]
                finally:
                    os.close(fd)
        except BaseException as exc:
            self._remember_error(exc)


class EngineStderrReader:
    """Drain an owned stderr pipe without blocking on read or close."""

    def __init__(self, stream: BinaryIO, writer: EngineLogWriter):
        self.stream = stream
        self.writer = writer
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._error: BaseException | None = None

    @property
    def error(self) -> BaseException | None:
        return self._error

    def start(self) -> None:
        with self._lock:
            if self._thread is not None:
                return
            self._thread = threading.Thread(target=self._run, name="engine-stderr-reader", daemon=True)
            self._thread.start()

    def stop(self, timeout: float = 1.0) -> bool:
        """Request shutdown, then join for at most ``timeout`` seconds.

        The reader thread closes its owned stream itself after leaving select,
        avoiding cross-thread close/read races and descriptor reuse hazards.
        """
        self._stop.set()
        with self._lock:
            thread = self._thread
        if thread is None or thread.ident is None:
            try:
                self.stream.close()
            except BaseException as exc:
                self._remember_error(exc)
            return True
        thread.join(max(0.0, timeout))
        return not thread.is_alive()

    def _remember_error(self, exc: BaseException) -> None:
        if self._error is None:
            self._error = exc

    def _emit(self, line: bytes, truncated: bool = False) -> None:
        if truncated:
            line += b"...[line truncated]"
        self.writer.write(line + b"\n")

    def _consume(self, data: bytes, pending: bytearray) -> None:
        parts = data.split(b"\n")
        for index, part in enumerate(parts):
            while part:
                room = _MAX_LINE_BYTES - len(pending)
                take = min(room, len(part))
                pending.extend(part[:take])
                part = part[take:]
                if len(pending) == _MAX_LINE_BYTES:
                    self._emit(bytes(pending), truncated=True)
                    pending.clear()
            if index < len(parts) - 1:
                self._emit(bytes(pending))
                pending.clear()

    def _run(self) -> None:
        pending = bytearray()
        stop_drain_remaining = _MAX_STOP_DRAIN_BYTES
        try:
            fd = self.stream.fileno()
            while True:
                timeout = 0.0 if self._stop.is_set() else _POLL_SECONDS
                readable, _, _ = select.select([fd], [], [], timeout)
                if not readable:
                    if self._stop.is_set():
                        break
                    continue
                read_size = _READ_BYTES
                if self._stop.is_set():
                    if stop_drain_remaining <= 0:
                        break
                    read_size = min(read_size, stop_drain_remaining)
                try:
                    chunk = os.read(fd, read_size)
                except InterruptedError:
                    continue
                if not chunk:
                    if pending:
                        self._emit(bytes(pending))
                        pending.clear()
                    break
                if self._stop.is_set():
                    stop_drain_remaining -= len(chunk)
                self._consume(chunk, pending)
        except BaseException as exc:
            self._remember_error(exc)
        finally:
            if pending:
                self._emit(bytes(pending))
                pending.clear()
            try:
                self.stream.close()
            except BaseException as exc:
                self._remember_error(exc)

