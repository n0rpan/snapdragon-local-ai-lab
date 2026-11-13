"""Checked, per-process NPU image worker. Never changes permissions or recovers CDSP.

Studio passes BOTH the controller's lease descriptor and its permit socket to
this worker and every descendant. A surviving native child therefore continues
to exclude chat, even if its Python parent dies. Only this invocation's verified
memory scope may be signalled. Proprietary runtime assets remain local files.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import select
import socket
import stat
import struct
import subprocess
import sys
import tempfile
import threading
import time

if __package__:
    from . import image_npu, memory_policy
else:
    import image_npu
    import memory_policy

NO_CONDITIONING, NEEDS_REFERENCE, FALLBACK, RECOVERY_REQUIRED, FAULT = 3, 4, 69, 70, 71
# Native text encoder on the NPU (oxi cond-native): fp16 linears, three sessions, 6 GiB weight
# budget, released before the process exits, so it never overlaps the DiT (one at a time under the
# same permit). Admission stays the DiT's qualified figure: the encoder phase is smaller
# (about 5.1 GiB weights + 0.75 GiB buffers) and runs strictly before it.
TE_NPU_ARGS = ('--te-backend', 'npu', '--te-npu', 'device', '--te-npu-sessions', '3',
               '--te-npu-weights', 'fp16', '--te-npu-budget-mib', '6144', '--te-npu-kc', '0', '--threads', '12')
STATE = Path.home() / '.local/state/local-ai'
SCOPE_BYTES = 28 * 1024**3


def load_profile(path):
    """Owner-controlled, pinned deployment manifest; no executable from a request."""
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                or info.st_mode & 0o077 or info.st_size > 65536):
            raise ValueError('unsafe NPU profile')
        value = json.loads(os.read(fd, 65537))
    finally:
        os.close(fd)
    keys = {'version', 'binary', 'wrapper', 'kit', 'runtime', 'runtime_sha256', 'files', 'links'}
    if not isinstance(value, dict) or set(value) != keys or type(value['version']) is not int or value['version'] != 1:
        raise ValueError('invalid NPU profile')
    for key in ('binary', 'wrapper', 'kit', 'runtime'):
        if not isinstance(value[key], str) or not Path(value[key]).is_absolute():
            raise ValueError('NPU package paths must be absolute')
    if not isinstance(value['files'], dict) or not isinstance(value['links'], dict):
        raise ValueError('invalid NPU package inventory')
    if not all(value[key] in value['files'] for key in ('binary', 'wrapper')):
        raise ValueError('NPU executable is not pinned')
    required = [Path(value['runtime']) / 'monitor-lib/libcdsprpc.so.1.0.0',
                Path(value['kit']) / 'build/SHA256SUMS',
                Path(value['kit']) / 'build/oxn_bench',
                Path(value['kit']) / 'build/liboxn_skel_v73.so']
    if any(str(p) not in value['files'] for p in required):
        raise ValueError('NPU runtime is not pinned')
    if value['files'][str(required[0])] != value['runtime_sha256']:
        raise ValueError('NPU runtime hash disagrees with inventory')
    for name, digest in value['files'].items():
        if not Path(name).is_absolute():
            raise ValueError('relative NPU package file')
        image_npu.verify_file(name, digest)
    for name, target in value['links'].items():
        p = Path(name)
        if (not p.is_absolute() or not isinstance(target, str) or not p.is_symlink()
                or p.lstat().st_uid != os.getuid() or os.readlink(p) != target
                or str(p.resolve()) not in value['files']):
            raise ValueError('NPU runtime link changed')
    return value


def verify_permit(lease_fd, socket_fd, lease_path):
    """Prove the inherited descriptor owns the canonical lock, without unlocking it."""
    expected = Path(lease_path).stat(follow_symlinks=False)
    actual = os.fstat(lease_fd)
    if (not stat.S_ISREG(expected.st_mode) or expected.st_uid != os.getuid()
            or expected.st_mode & 0o077 or (expected.st_dev, expected.st_ino) != (actual.st_dev, actual.st_ino)):
        raise ValueError('invalid canonical image lease')
    # Trying EX on an independent open distinguishes a live flock from an fd to
    # the right inode. No flock operation is made on the delegated description.
    with Path(lease_path).open('a+') as probe:
        try:
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            fcntl.flock(probe, fcntl.LOCK_UN)
            raise ValueError('canonical image lease is not held')
    # An fd to the right inode is insufficient if a DIFFERENT open description
    # owns the flock. Reasserting EX succeeds only on the delegated description.
    fcntl.flock(lease_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    with socket.socket(fileno=os.dup(socket_fd)) as connection:
        _, uid, _ = struct.unpack('3i', connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
        if uid != os.getuid() or connection.getsockopt(socket.SOL_SOCKET, socket.SO_TYPE) != socket.SOCK_SEQPACKET:
            raise ValueError('invalid image permit socket')


class OwnedScope:
    """Popen adapter for MemoryGuard, including timeout's separate process group."""
    def __init__(self, process, cgroup):
        self.process, self.pid, self.cgroup = process, process.pid, cgroup
        self.name = 'local-ai-flash-%d.scope' % self.pid
        if (cgroup.name != self.name or (cgroup / 'memory.max').read_text().strip() != str(SCOPE_BYTES)
                or (cgroup / 'memory.swap.max').read_text().strip() != '0'):
            raise ValueError('worker did not enter its own no-swap scope')
        self.identity = cgroup.stat().st_ino

    def populated(self):
        try:
            if self.cgroup.stat().st_ino != self.identity:
                raise RuntimeError('image scope was replaced')
            return 'populated 1' in (self.cgroup / 'cgroup.events').read_text().splitlines()
        except FileNotFoundError:
            return False

    def poll(self):
        rc = self.process.poll()
        return None if rc is None or self.populated() else rc

    def _signal(self, name):
        if self.populated():
            done = subprocess.run(['/usr/bin/systemctl', '--user', 'kill', '--kill-whom=all',
                                   '--signal=' + name, self.name], stdin=subprocess.DEVNULL,
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=3)
            if done.returncode and self.populated():
                raise RuntimeError('could not stop owned image scope')

    def terminate(self):
        self._signal('TERM')

    def kill(self):
        self._signal('KILL')

    def cleanup(self):
        for operation, seconds in ((self.terminate, 5), (self.kill, 5)):
            if not self.populated():
                break
            operation()
            deadline = time.monotonic() + seconds
            while self.populated() and time.monotonic() < deadline:
                time.sleep(.05)
        self.process.wait(timeout=2)
        if self.populated():
            raise RuntimeError('image scope remains populated; recovery required')


def command(profile, native_args):
    """All device knobs are fixed here, after rejecting caller overrides."""
    if not native_args or native_args[0] not in ('generate', 'cond-native'):
        raise ValueError('only qualified image generation is supported')
    forbidden = ('--npu', '--backend', '--chunk1', '--chunk2', '--attn-prec', '--threads', '--vae-conv',
                 '--te-backend', '--te-npu', '--te-io-threads')
    if any(a.split('=', 1)[0] in forbidden or a.startswith(('--npu-', '--te-npu-')) for a in native_args):
        raise ValueError('caller cannot override qualified image arithmetic')
    if native_args[0] == 'cond-native':
        # the cache namespace must name the arithmetic this worker pins
        if native_args.count('--te-arith') != 1 or native_args[native_args.index('--te-arith') + 1:][:1] != ['npu-fp16']:
            raise ValueError('text encoder namespace is not the qualified NPU arithmetic')
        for flag, expected in (('--width', '512'), ('--height', '512'), ('--steps', '4'), ('--max-tokens', '512')):
            if native_args.count(flag) != 1 or native_args[native_args.index(flag) + 1:native_args.index(flag) + 2] != [expected]:
                raise ValueError('image shape is not qualified')
        return [profile['wrapper'], 'cmd', profile['binary'], *native_args, *TE_NPU_ARGS]
    for flag, expected in (('--width', '512'), ('--height', '512'), ('--steps', '4')):
        if native_args.count(flag) != 1 or native_args[native_args.index(flag) + 1:native_args.index(flag) + 2] != [expected]:
            raise ValueError('image shape is not qualified')
    return [profile['wrapper'], 'cmd', profile['binary'], *native_args,
            '--backend', 'fast', '--chunk1', '32', '--chunk2', '32', '--attn-prec', 'f32',
            '--threads', '12', '--vae-conv', 'f32', '--npu', 'device', '--npu-sessions', '4',
            '--npu-budget-mib', '8192', '--npu-kc', '0', '--npu-weights', 'fp16']


def latch(reason, state=STATE):
    state.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(state / 'NPU-STOP', os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'w') as output:
        output.write(json.dumps({'reason': reason, 'time': time.time(), 'source': 'image-worker'}) + '\n')


def result_code(wrapper_rc, text, failure=None):
    if failure:
        return FAULT
    if wrapper_rc == 0 and 'VERDICT: DEVICE CLEAN' in text:
        return 0
    if (wrapper_rc == 3 and 'VERDICT: CLEAN ERROR' in text
            and 'oxi: no cached conditioning for key ' in text and 'cmd exit 3,' in text):
        return NO_CONDITIONING
    if (wrapper_rc == 4 and 'VERDICT: CLEAN ERROR' in text
            and "oxi: the prompt needs sd.cpp's text encoder" in text and 'cmd exit 4,' in text):
        return NEEDS_REFERENCE
    return FAULT


def te_dma_bytes(text):
    """Actual DMA bytes the native text encoder allocated (its `oxi: te: released` line), else None."""
    for line in text.splitlines():
        if line.startswith('oxi: te: released; dma_allocated_bytes '):
            value = line.split()[4]
            return int(value) if value.isdigit() else None
    return None


def execute(profile, native_args, pass_fds, *, state=STATE, cancelled=None):
    """Returns only once every native descendant is gone, or refuses CPU recovery."""
    cancelled = cancelled or threading.Event()
    state.mkdir(mode=0o700, parents=True, exist_ok=True)
    logs = state / 'image-npu'
    logs.mkdir(mode=0o700, exist_ok=True)
    info = logs.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise ValueError('unsafe image diagnostics directory')
    run = Path(tempfile.mkdtemp(prefix='run-', dir=logs))
    environment = image_npu.clean_environment(os.environ)
    environment.update(OXN_RT=profile['runtime'], OXN_KIT=profile['kit'],
                       OXN_RT_SHA256=profile['runtime_sha256'], OXN_ATTENDED='1',
                       OXN_LEASE_HELD='1', OXN_LOGDIR=str(run / 'device'), OXN_CMD_TIMEOUT='120',
                       OXI_NPU_STATS='1')
    # Preserve the qualified prompt-conditioning cache; strip all other inherited knobs.
    if 'OXI_COND_CACHE' in os.environ:
        environment['OXI_COND_CACHE'] = os.environ['OXI_COND_CACHE']
    guard = None
    process, owned, failure, rc = None, None, None, RECOVERY_REQUIRED
    te_dma = None
    ready_r, ready_w = os.pipe2(os.O_CLOEXEC)
    go_r, go_w = os.pipe2(os.O_CLOEXEC)
    activation_fds = [ready_r, ready_w, go_r, go_w]
    native_released = False
    started = time.monotonic()
    try:
        with (run / 'worker.log').open('w') as output, (run / 'memory.jsonl').open('w') as trace:
            scoped = memory_policy.scope_command(command(profile, native_args), SCOPE_BYTES)
            separator = scoped.index('--')
            scoped[separator:separator] = ['--ready-fd', str(ready_w), '--go-fd', str(go_r)]
            process = subprocess.Popen(scoped,
                                       env=environment, stdin=subprocess.DEVNULL, stdout=output,
                                       stderr=subprocess.STDOUT, pass_fds=(*pass_fds, ready_w, go_r))
            os.close(ready_w); activation_fds.remove(ready_w)
            os.close(go_r); activation_fds.remove(go_r)
            readable, _, _ = select.select([ready_r], [], [], 8)
            if not readable or os.read(ready_r, 1) != b'R':
                raise RuntimeError('image scope creation timed out')
            owned = OwnedScope(process, memory_policy.process_cgroup(process.pid))
            # The native child is still behind the handshake. Baseline its NEW
            # swap-free scope, rather than the caller's unrelated service limits.
            guard = memory_policy.MemoryGuard(host_guard_version=memory_policy.GUARD_2_VERSION,
                loading_timeout_seconds=120,
                reader=lambda: memory_policy.read_sample(owned.cgroup, extended=True))
            guard.start(owned)
            if guard.failure():
                raise RuntimeError('memory guard refused image startup')
            os.write(go_w, b'G')
            native_released = True
            os.close(go_w); activation_fds.remove(go_w)
            next_sample, cursor = 0., 0
            while owned.poll() is None:
                if cancelled.is_set():
                    failure = 'image cancelled'
                    break
                if time.monotonic() - started > 140:
                    failure = 'image watchdog timeout'
                    break
                if guard.failure():
                    failure = guard.failure()
                    break
                # Forward only progress from our engine, never request text/argv.
                log = run / 'device/cmd.log'
                if log.exists():
                    with log.open(errors='replace') as source:
                        source.seek(cursor)
                        while line := source.readline():
                            if line.startswith(('forward:', 'vae:', 'oxi: npu:', 'oxi: te:')):
                                print(line[:2048].rstrip(), flush=True)
                                if line.startswith(('forward:', 'oxi: te: released')):
                                    guard.mark_ready()
                        cursor = source.tell()
                now = time.monotonic()
                if now >= next_sample:
                    trace.write(json.dumps({'elapsed': now - started, 'guard': guard.metadata()}) + '\n')
                    trace.flush()
                    next_sample = now + 1
                time.sleep(.05)
            owned.cleanup()
            text = (run / 'worker.log').read_text(errors='replace') + (
                (run / 'device/cmd.log').read_text(errors='replace') if (run / 'device/cmd.log').exists() else '')
            rc = result_code(process.returncode, text, failure)
            te_dma = te_dma_bytes(text) if native_args[0] == 'cond-native' else None
            if rc == FAULT:
                latch(failure or 'NPU image did not finish DEVICE CLEAN', state)
    except BaseException as error:
        failure = type(error).__name__
        if process is None:
            rc = FALLBACK  # No process, therefore no device contact.
        else:
            try:
                if owned is not None:
                    owned.cleanup()
                else:
                    # Scope creation failed before native exec. The wrapper is never
                    # allowed to run unless exec_in_scope verified the new scope.
                    process.kill()
                    process.wait(timeout=5)
                rc = FAULT if native_released else FALLBACK
            except BaseException:
                rc = RECOVERY_REQUIRED
            if native_released or rc == RECOVERY_REQUIRED:
                latch('image worker failed: ' + failure, state)
    finally:
        for fd in activation_fds:
            os.close(fd)
        if guard is not None:
            guard.stop()
        receipt = {'backend': 'oxi-npu', 'returncode': rc, 'failure': failure,
                   'elapsed_seconds': time.monotonic() - started,
                   'guard': None if guard is None else guard.metadata(),
                   'phase': native_args[0] if native_args else None,
                   # generate: the DiT's qualified figure; cond-native: the encoder's own counter
                   'dma_allocated_bytes': (te_dma if native_args[:1] == ['cond-native']
                                           else image_npu.QUALIFIED_DMA_BYTES),
                   'scope': None if owned is None else owned.name}
        (run / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
        print('image-npu: result ' + json.dumps({'returncode': rc, 'receipt': str(run / 'receipt.json')}), flush=True)
    return rc


def main(argv=None):
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', required=True)
    parser.add_argument('--lease-fd', type=int, required=True)
    parser.add_argument('--slot-fd', type=int, required=True)
    parser.add_argument('--prompt-fd', type=int)
    parser.add_argument('native_args', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    cancelled = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: cancelled.set())
    try:
        try:
            verify_permit(args.lease_fd, args.slot_fd, STATE / 'inference.lock')
        except (OSError, ValueError):
            print('image-npu: invalid controller permit; refusing all compute', flush=True)
            return RECOVERY_REQUIRED
        profile = load_profile(args.profile)
        if (STATE / 'NPU-STOP').exists():
            return FALLBACK
        native_args = args.native_args[1:] if args.native_args[:1] == ['--'] else args.native_args
        command(profile, native_args)  # Fail before creating the native process.
        fds = (args.lease_fd, args.slot_fd) + (() if args.prompt_fd is None else (args.prompt_fd,))
        return execute(profile, native_args, fds, cancelled=cancelled)
    except (OSError, ValueError, TypeError, KeyError):
        print('image-npu: package/permit preflight refused; CPU fallback', flush=True)
        return FALLBACK


if __name__ == '__main__':
    raise SystemExit(main())
