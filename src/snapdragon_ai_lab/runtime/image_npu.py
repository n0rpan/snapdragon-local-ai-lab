"""Opt-in NPU image policy. No device access or permission changes at import time.

The controller supplies resident-model accounting before granting an image slot.
Fallback never requests a chat unload. Kernel/runtime execution belongs to the
separate checked worker, not to this model-free policy module.
"""
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import stat

GIB = 1024 ** 3
# C4 + W1: four sessions, fp16, no K split. These are OUR allocation counters.
QUALIFIED_DMA_BYTES = 9_261_023_232
MEASURED_HOST_GROWTH_BYTES = 10_586_181_632  # max of 10 fresh processes, 1 Hz sampling
HOST_MARGIN_BYTES = GIB
REQUIRED_HOST_BYTES = max(QUALIFIED_DMA_BYTES, MEASURED_HOST_GROWTH_BYTES) + HOST_MARGIN_BYTES
DESKTOP_RESIDENT_BYTES = 5 * GIB // 2


@dataclass(frozen=True)
class Choice:
    backend: str
    reason: str
    required_bytes: int = REQUIRED_HOST_BYTES


def choose(*, enabled=False, ready=False, latched=False, width=512, height=512,
           model='klein-q8', mem_available=None, mem_total=None,
           resident_model_bytes=None, swap_free=None):
    """Choose a backend, never evicting the resident model to make it fit.

    Resident mode uses a physical budget (including active desktop RAM); idle
    desktop pages may swap. A missing resident budget fails closed to CPU.
    This is only admission. GUARD-2b stays active throughout execution.
    """
    if enabled is not True:
        return Choice('cpu', 'NPU images are opt-in')
    if not ready:
        return Choice('cpu', 'qualified NPU package is unavailable')
    if latched:
        return Choice('cpu', 'NPU is latched off after an error')
    if model != 'klein-q8' or (width, height) != (512, 512):
        return Choice('cpu', 'NPU image gate covers klein Q8 at 512 pixels')
    numbers = [mem_available, swap_free]
    if any(type(n) is not int or n < 0 for n in numbers):
        return Choice('cpu', 'host memory telemetry is unavailable')
    if mem_available + swap_free < 2 * GIB:
        return Choice('cpu', 'host reserve cannot clear GUARD-2')
    if resident_model_bytes is None:
        budget = mem_available
    else:
        if (type(mem_total) is not int or mem_total <= 0
                or type(resident_model_bytes) is not int or resident_model_bytes < 0):
            return Choice('cpu', 'resident-model budget is unavailable')
        budget = mem_total - resident_model_bytes - DESKTOP_RESIDENT_BYTES
        if swap_free < max(0, REQUIRED_HOST_BYTES - mem_available) + GIB:
            return Choice('cpu', 'insufficient swap headroom for idle desktop pages')
    if budget < REQUIRED_HOST_BYTES:
        return Choice('cpu', 'NPU working set cannot fit next to the resident workload')
    return Choice('npu', 'qualified per-process fp16 NPU backend')


def verify_file(path, expected_sha256):
    """Verify pinned regular files without following a substituted final symlink."""
    if (not isinstance(expected_sha256, str) or len(expected_sha256) != 64
            or any(c not in '0123456789abcdef' for c in expected_sha256)):
        raise ValueError('invalid pinned file hash')
    path = Path(path)
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_uid != os.getuid() or before.st_mode & 0o022:
            raise ValueError('unsafe NPU package file')
        digest = hashlib.sha256()
        while data := os.read(descriptor, 1024 * 1024):
            digest.update(data)
        after = os.fstat(descriptor)
        if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise ValueError('NPU package changed during verification')
        if digest.hexdigest() != expected_sha256:
            raise ValueError('NPU package hash mismatch')
    finally:
        os.close(descriptor)


def clean_environment(environment):
    """Do not inherit experimental knobs into the qualified invocation."""
    prefixes = ('OXI_', 'OXN_', 'OXE_', 'GGML_', 'LLAMA_', 'DSPQUEUE_', 'ADSP_', 'DSP_LIBRARY_')
    return {key: value for key, value in environment.items()
            if not key.startswith(prefixes) and key not in ('FASTRPC_LIB', 'LD_PRELOAD', 'LD_LIBRARY_PATH')}
