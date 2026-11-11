"""Portable Linux wrapper checks: no models, services or global policy changes."""
import ctypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

WRAPPER = Path(__file__).resolve().parents[1] / 'tools/exec_no_thp.py'
TESTED_SHA = '29b29012985908d660b919778b5629dc34c6abc9115b6c07f96f9cdc368f9457'


def sha256(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


@unittest.skipUnless(sys.platform == 'linux' and Path('/proc/self/status').is_file(), 'Linux proc/prctl required')
class ProcessTHPWrapper(unittest.TestCase):
    def test_exact_reviewed_bytes_and_import_do_not_execute_or_change_policy(self):
        self.assertEqual(sha256(WRAPPER), TESTED_SHA)
        spec = importlib.util.spec_from_file_location('optional_exec_no_thp', WRAPPER)
        module = importlib.util.module_from_spec(spec)
        with patch('ctypes.CDLL', side_effect=AssertionError('import must not call prctl')), \
             patch('os.execv', side_effect=AssertionError('import must not exec')):
            spec.loader.exec_module(module)
        self.assertTrue(callable(module.main))

    def test_inheritance_and_process_identity_with_only_16mib_probe(self):
        libc = ctypes.CDLL(None)
        libc.prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong]
        before = libc.prctl(42, 0, 0, 0, 0)  # Read only in this test parent.
        if before < 0:
            self.skipTest('Kernel does not expose process THP control')
        executable = Path(sys.executable).resolve()
        code = '''import os,json,mmap
from pathlib import Path
memory=mmap.mmap(-1,16*1024*1024,flags=mmap.MAP_PRIVATE|mmap.MAP_ANONYMOUS)
for offset in range(0,16*1024*1024,mmap.PAGESIZE):memory[offset]=1
status={k:v.strip() for k,s,v in (line.partition(':') for line in Path('/proc/self/status').read_text().splitlines()) if s}
rollup={k:v.strip() for k,s,v in (line.partition(':') for line in Path('/proc/self/smaps_rollup').read_text().splitlines()) if s}
print(json.dumps({'pid':os.getpid(),'ppid':os.getppid(),'pgid':os.getpgrp(),'THP_enabled':int(status['THP_enabled']),'AnonHugePages':rollup['AnonHugePages'],'allocated_bytes':16*1024*1024}))
'''
        child = subprocess.Popen([sys.executable, '-I', '-B', str(WRAPPER), '--sha256', sha256(executable),
                                  '--', str(executable), '-I', '-c', code],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
        try:
            stdout, stderr = child.communicate(timeout=10)
        finally:
            if child.poll() is None:
                child.kill(); child.wait(timeout=5)
        self.assertEqual(child.returncode, 0, stderr)
        value = json.loads(stdout)
        self.assertEqual(value['pid'], child.pid); self.assertEqual(value['pgid'], child.pid)
        self.assertEqual(value['ppid'], os.getpid()); self.assertEqual(value['THP_enabled'], 0)
        self.assertEqual(value['AnonHugePages'], '0 kB')
        self.assertEqual(value['allocated_bytes'], 16 * 1024**2)
        self.assertEqual(libc.prctl(42, 0, 0, 0, 0), before)

    def test_hash_mismatch_refuses_before_executing_target(self):
        executable = Path(sys.executable).resolve()
        with tempfile.TemporaryDirectory() as temporary:
            marker = Path(temporary) / 'target-ran'
            code = 'from pathlib import Path;Path(' + repr(str(marker)) + ').write_text("unexpected")'
            completed = subprocess.run([sys.executable, '-I', '-B', str(WRAPPER), '--sha256', '0' * 64,
                                        '--', str(executable), '-I', '-c', code],
                                       capture_output=True, text=True, timeout=10)
            self.assertEqual(completed.returncode, 78)
            self.assertFalse(marker.exists())


if __name__ == '__main__':
    unittest.main()
