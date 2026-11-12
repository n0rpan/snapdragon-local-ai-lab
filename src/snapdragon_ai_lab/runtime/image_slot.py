"""Controller-owned image slots with an inherited canonical lease descriptor.

The controller keeps chat resident and queues its requests while an image runs.
Its reservation context stays open until EVERY duplicate of the client's socket
has closed. The worker inherits both descriptors, so a Studio/controller crash
does not release the hardware lease under a surviving image process.
"""
import array
import json
import os
from pathlib import Path
import signal
import socket
import stat
import sys
import struct
import threading
import time

MAX_MESSAGE = 4096
# S7: longest an image job may hold the chat slot after handoff before the controller takes it
# back (a hung worker must not block chat forever). The NPU image path finishes in minutes.
MAX_HOLD_SECONDS = float(os.environ.get('LOCAL_AI_IMAGE_SLOT_MAX_SECONDS', '1800'))


class SlotUnavailable(RuntimeError):
    pass


def socket_peer_inode(sock):
    """Inode of the other end of a connected AF_UNIX socket (NETLINK_SOCK_DIAG, UNIX_DIAG_PEER),
    or None when the kernel cannot tell (no unix_diag): the caller must then fail closed."""
    try:
        own = os.fstat(sock.fileno()).st_ino
        with socket.socket(socket.AF_NETLINK, socket.SOCK_DGRAM, 4) as diag:  # NETLINK_SOCK_DIAG
            diag.settimeout(2)
            # unix_diag_req: family, protocol, pad, states (all), ino, show=UDIAG_SHOW_PEER, cookie
            request = struct.pack('=BBHIII2I', socket.AF_UNIX, 0, 0, 0xffffffff, own, 4, 0xffffffff, 0xffffffff)
            diag.send(struct.pack('=IHHII', 16 + len(request), 20, 1, 1, 0) + request)  # SOCK_DIAG_BY_FAMILY
            data = diag.recv(8192)
    except OSError:
        return None
    if len(data) < 32 or struct.unpack('=IH', data[:6])[1] != 20:
        return None
    if struct.unpack('=I', data[20:24])[0] != own:
        return None
    offset = 32
    while offset + 8 <= len(data):
        length, kind = struct.unpack('=HH', data[offset:offset + 4])
        if kind == 2:  # UNIX_DIAG_PEER
            return struct.unpack('=I', data[offset + 4:offset + 8])[0]
        if length < 4:
            break
        offset += (length + 3) & ~3
    return None


def socket_holders(peer_inode, proc_root=Path('/proc')):
    """Every visible process holding the image client's end of the handoff socket, wherever it
    has been reparented. This is the job's own identity (one socket per handoff), never the
    shared lease inode, so resident chat and unrelated lease readers are never included."""
    target = f'socket:[{peer_inode}]'
    holders = set()
    for entry in proc_root.glob('[0-9]*'):
        try:
            for descriptor in (entry / 'fd').iterdir():
                try:
                    if os.readlink(descriptor) == target:
                        holders.add(int(entry.name))
                        break
                except OSError:
                    continue
        except (OSError, ValueError):
            continue
    return holders


def kill_holder(pid, peer_inode, proc_root=Path('/proc')):
    """SIGKILL `pid` only if it still holds the job's socket, through a pidfd opened BEFORE the
    check, so a recycled pid can never be signalled."""
    pidfd = os.pidfd_open(pid)
    try:
        if pid not in socket_holders_of(pid, peer_inode, proc_root):
            return False
        signal.pidfd_send_signal(pidfd, signal.SIGKILL)
        return True
    finally:
        os.close(pidfd)


def socket_holders_of(pid, peer_inode, proc_root=Path('/proc')):
    target = f'socket:[{peer_inode}]'
    try:
        for descriptor in (proc_root / str(pid) / 'fd').iterdir():
            try:
                if os.readlink(descriptor) == target:
                    return {pid}
            except OSError:
                continue
    except OSError:
        pass
    return set()


class ImageLease:
    def __init__(self, connection, descriptor, metadata):
        self.connection = connection
        self.descriptor = descriptor
        self.metadata = metadata

    def fileno(self):
        return self.descriptor

    def pass_fds(self):
        return (self.descriptor, self.connection.fileno())

    def close(self):
        if self.descriptor is not None:
            os.close(self.descriptor)
            self.descriptor = None
        if self.connection is not None:
            # No shutdown(): a child may still hold its duplicate during cleanup.
            self.connection.close()
            self.connection = None


def _receive(connection):
    payload, ancillary, flags, _ = connection.recvmsg(
        MAX_MESSAGE, socket.CMSG_SPACE(4 * 8), socket.MSG_CMSG_CLOEXEC)
    descriptors = []
    for level, kind, data in ancillary:
        if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
            fds = array.array('i')
            fds.frombytes(data[:len(data) - len(data) % fds.itemsize])
            descriptors.extend(fds)
    try:
        if flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC) or not payload:
            raise SlotUnavailable('invalid image slot response')
        value = json.loads(payload)
        if not isinstance(value, dict):
            raise SlotUnavailable('invalid image slot message')
        return value, descriptors
    except BaseException:
        for fd in descriptors:
            os.close(fd)
        raise


def connect(path, profile, timeout=10):
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    descriptor = None
    try:
        connection.settimeout(timeout)
        connection.connect(str(path))
        _, uid, _ = struct.unpack('3i', connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
        if uid != os.getuid():
            raise SlotUnavailable('image slot server is not the installing user')
        connection.sendall(json.dumps({'version': 1, 'profile': profile}).encode())
        value, descriptors = _receive(connection)
        if value.get('ok') is not True or len(descriptors) != 1:
            for fd in descriptors:
                os.close(fd)
            raise SlotUnavailable(value.get('error', 'image slot is unavailable'))
        descriptor = descriptors[0]
        os.set_inheritable(descriptor, False)
        connection.settimeout(None)
        return ImageLease(connection, descriptor, value.get('metadata', {}))
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
        connection.close()
        raise


class Broker:
    """`reserve(profile, peer_pid)` yields (lease fd, nonprivate metadata).

    The supplied context must block model inference and retain the model/lease
    until it exits. No command, prompt, file path or arbitrary argv crosses IPC.
    """
    def __init__(self, path, reserve, profiles=('klein-q8-512-npu',), max_hold_seconds=None,
                 holders=None, kill=None, retry_seconds=60.0, kill_client=False):
        self.path = Path(path)
        self.max_hold_seconds = MAX_HOLD_SECONDS if max_hold_seconds is None else float(max_hold_seconds)
        # S7 (Codex review 2): the reservation is released ONLY on EOF of the handoff socket, i.e.
        # once every copy of the client's end (Studio and any worker, even reparented) is closed.
        # Past max_hold_seconds the job is marked stale and its socket holders are stopped; the
        # client process itself only with kill_client=True (an install decision, off by default).
        self.holders = holders or (lambda connection: (lambda peer: None if peer is None else socket_holders(peer))(
            socket_peer_inode(connection)))
        self.kill = kill or (lambda pid, connection: kill_holder(pid, socket_peer_inode(connection)))
        self.retry_seconds = float(retry_seconds)
        self.kill_client = kill_client
        self.forced_releases = 0
        self.stale = {}  # peer pid -> monotonic time the hold went past max_hold_seconds
        self.reserve = reserve
        self.profiles = frozenset(profiles)
        self.stop = threading.Event()
        self.listener = None
        self.thread = None
        self.identity = None

    def start(self):
        parent = self.path.parent
        parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        info = parent.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
            raise SlotUnavailable('image slot directory is not owner-only')
        # bind refuses an existing socket/file; never unlink an unknown preimage.
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        try:
            listener.bind(str(self.path))
            self.identity = (self.path.stat().st_dev, self.path.stat().st_ino)
            self.path.chmod(0o600)
            listener.listen(4)
            listener.settimeout(0.2)
        except BaseException:
            listener.close()
            self._unlink_owned()
            raise
        self.listener = listener
        self.thread = threading.Thread(target=self._serve, daemon=True, name='image-slots')
        self.thread.start()
        return self

    def _serve(self):
        while not self.stop.is_set():
            try:
                connection, _ = self.listener.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            threading.Thread(target=self._handle, args=(connection,), daemon=True,
                             name='image-slot-client').start()

    def _handle(self, connection):
        with connection:
            sent = False
            try:
                connection.settimeout(10)
                pid, uid, _ = struct.unpack('3i', connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
                if uid != os.getuid():
                    raise SlotUnavailable('image slot peer is not the installing user')
                request, descriptors = _receive(connection)
                for fd in descriptors:
                    os.close(fd)
                if descriptors or set(request) != {'version', 'profile'} or type(request['version']) is not int or request['version'] != 1:
                    raise SlotUnavailable('invalid image slot request')
                profile = request['profile']
                if not isinstance(profile, str) or profile not in self.profiles:
                    raise SlotUnavailable('image profile is not qualified')
                with self.reserve(profile, pid) as (lease_fd, metadata):
                    data = json.dumps({'ok': True, 'metadata': metadata}).encode()
                    if len(data) > MAX_MESSAGE:
                        raise SlotUnavailable('image slot metadata is too large')
                    connection.sendmsg([data], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array('i', [lease_fd]))])
                    sent = True
                    # EOF only after Studio AND its inherited worker copies close, or the watchdog:
                    # after max_hold_seconds the reservation is released (forced) so chat resumes.
                    deadline = time.monotonic() + self.max_hold_seconds
                    stale = False
                    try:
                        while True:
                            left = deadline - time.monotonic()
                            if left <= 0:
                                if not stale:
                                    stale = True
                                    self.stale[pid] = time.monotonic()
                                    print(f'image slot: job of pid {pid} held the chat slot past '
                                          f'{self.max_hold_seconds:.0f} s; stopping its workers', file=sys.stderr, flush=True)
                                self._stop_job(connection, pid)
                                deadline = time.monotonic() + self.retry_seconds
                                continue
                            connection.settimeout(min(left, 60.0))
                            try:
                                if not connection.recv(1):
                                    break  # EOF: every copy of the client's end is closed
                            except socket.timeout:
                                continue
                    finally:
                        self.stale.pop(pid, None)
                    if stale:
                        self.forced_releases += 1
            except Exception:
                if not sent:
                    try:
                        connection.sendall(b'{"ok":false,"error":"image slot is unavailable"}')
                    except OSError:
                        pass

    def _stop_job(self, connection, peer_pid):
        """SIGKILL the processes holding this job's end of the handoff socket (not this process,
        not the client unless kill_client). Never releases anything itself: release needs EOF."""
        try:
            holders = self.holders(connection)
        except OSError:
            holders = None
        if holders is None:
            print('image slot: cannot identify the stale job\'s processes; the chat slot stays reserved',
                  file=sys.stderr, flush=True)
            return
        for target in sorted(holders - {os.getpid()} - (set() if self.kill_client else {peer_pid})):
            try:
                self.kill(target, connection)
            except ProcessLookupError:
                pass
            except OSError as error:
                print(f'image slot: could not stop stale holder {target}: {error}', file=sys.stderr, flush=True)

    def _unlink_owned(self):
        try:
            info = self.path.lstat()
            if self.identity == (info.st_dev, info.st_ino):
                self.path.unlink()
        except FileNotFoundError:
            pass

    def close(self):
        self.stop.set()
        if self.listener is not None:
            self.listener.close()
        if self.thread is not None:
            self.thread.join(timeout=2)
        self._unlink_owned()
