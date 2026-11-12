"""No device, model, sudo or live service is used by these tests."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import Mock, patch

from snapdragon_ai_lab.runtime import exec_in_scope, image_npu_worker as worker


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def profile(self):
        runtime, kit = self.root / 'runtime', self.root / 'kit'
        paths = [runtime / 'monitor-lib/libcdsprpc.so.1.0.0', kit / 'build/SHA256SUMS',
                 kit / 'build/oxn_bench', kit / 'build/liboxn_skel_v73.so',
                 self.root / 'oxi', self.root / 'wrapper']
        files = {}
        for p in paths:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b'owned fixture ' + p.name.encode())
            p.chmod(0o600)
            files[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
        value = dict(version=1, binary=str(paths[-2]), wrapper=str(paths[-1]), kit=str(kit),
                     runtime=str(runtime), runtime_sha256=files[str(paths[0])], files=files, links={})
        path = self.root / 'profile.json'
        path.write_text(json.dumps(value))
        path.chmod(0o600)
        return path, value

    def test_package_tamper_or_missing_runtime_pin_refused(self):
        path, value = self.profile()
        self.assertEqual(worker.load_profile(path), value)
        Path(value['binary']).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            worker.load_profile(path)
        path, value = self.profile()
        del value['files'][str(Path(value['kit']) / 'build/liboxn_skel_v73.so')]
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, 'not pinned'):
            worker.load_profile(path)
        path.chmod(0o644)
        with self.assertRaisesRegex(ValueError, 'unsafe'):
            worker.load_profile(path)

    def test_permit_requires_the_locked_open_description(self):
        path = self.root / 'inference.lock'
        owner = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
        other = os.open(path, os.O_RDWR)
        left, right = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        try:
            with self.assertRaisesRegex(ValueError, 'not held'):
                worker.verify_permit(owner, left.fileno(), path)
            fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
            worker.verify_permit(owner, left.fileno(), path)
            with self.assertRaises(BlockingIOError):
                worker.verify_permit(other, left.fileno(), path)
            with self.assertRaises(BlockingIOError):
                fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(owner); os.close(other); left.close(); right.close()

    def test_only_qualified_arithmetic_and_shape(self):
        p = {'binary': '/owned/oxi', 'wrapper': '/owned/check'}
        base = ['generate', '--width', '512', '--height', '512', '--steps', '4']
        result = worker.command(p, base)
        self.assertIn('--npu', result)
        self.assertEqual(result[result.index('--npu-sessions') + 1], '4')
        self.assertEqual(result[result.index('--vae-conv') + 1], 'f32')
        for extra in (['--npu', 'device'], ['--npu-kc=64'], ['--backend', 'ref']):
            with self.assertRaises(ValueError):
                worker.command(p, base + extra)
        with self.assertRaises(ValueError):
            worker.command(p, ['generate', '--width', '768', '--height', '512', '--steps', '4'])

    def test_clean_cache_miss_is_not_a_fault_and_unclean_is_never_success(self):
        clean = 'VERDICT: DEVICE CLEAN'
        self.assertEqual(worker.result_code(0, clean), 0)
        self.assertEqual(worker.result_code(0, ''), worker.FAULT)
        self.assertEqual(worker.result_code(0, clean, 'memory stop'), worker.FAULT)
        miss = 'VERDICT: CLEAN ERROR\noxi: no cached conditioning for key abc\ncmd exit 3,'
        self.assertEqual(worker.result_code(3, miss), worker.NO_CONDITIONING)
        self.assertEqual(worker.result_code(3, 'VERDICT: CLEAN ERROR'), worker.FAULT)

    def test_scope_cannot_target_the_controller_or_another_native_process(self):
        process = Mock(pid=12345)
        cg = self.root / 'controller.service'; cg.mkdir()
        with self.assertRaisesRegex(ValueError, 'own no-swap'):
            worker.OwnedScope(process, cg)
        cg = self.root / 'local-ai-flash-12345.scope'; cg.mkdir()
        (cg / 'memory.max').write_text(str(worker.SCOPE_BYTES))
        (cg / 'memory.swap.max').write_text('0')
        (cg / 'cgroup.events').write_text('populated 1\n')
        owned = worker.OwnedScope(process, cg)
        process.poll.return_value = 0
        self.assertIsNone(owned.poll(), 'wrapper exit must not hide an orphan descendant')
        with patch.object(worker.subprocess, 'run', return_value=Mock(returncode=0)) as run:
            owned.kill()
        self.assertEqual(run.call_args.args[0][-1], 'local-ai-flash-12345.scope')
        (cg / 'cgroup.events').write_text('populated 0\n')
        self.assertEqual(owned.poll(), 0)

    def test_scope_handshake_refuses_native_exec_without_go(self):
        ready_r, ready_w = os.pipe()
        go_r, go_w = os.pipe()
        os.close(go_w)  # Parent disappearance must fail closed, before native exec.
        try:
            with patch.object(exec_in_scope, 'enter_scope', return_value=True), \
                    patch.object(exec_in_scope.os, 'execv') as execute:
                rc = exec_in_scope.main(['--memory-max', str(worker.SCOPE_BYTES),
                    '--runtime-dir', '/run/user/%d' % os.getuid(), '--ready-fd', str(ready_w),
                    '--go-fd', str(go_r), '--', '/bin/true'])
            self.assertEqual(rc, 78)
            self.assertEqual(os.read(ready_r, 1), b'R')
            execute.assert_not_called()
        finally:
            os.close(ready_r)

    def test_ready_timeout_preserves_fallback_receipt_before_device_contact(self):
        profile = dict(binary='/owned/oxi', wrapper='/owned/wrapper', runtime='/unused',
                       kit='/unused', runtime_sha256='0'*64)
        native = ['generate', '--width', '512', '--height', '512', '--steps', '4']
        process = Mock(pid=12345)
        with patch.object(worker.subprocess, 'Popen', return_value=process), \
                patch.object(worker.select, 'select', return_value=([], [], [])), \
                patch('builtins.print'):
            rc = worker.execute(profile, native, (), state=self.root)
        self.assertEqual(rc, worker.FALLBACK)
        process.kill.assert_called_once()
        process.wait.assert_called_once_with(timeout=5)
        paths = list((self.root / 'image-npu').glob('run-*/receipt.json'))
        self.assertEqual(len(paths), 1)
        row = json.loads(paths[0].read_text())
        self.assertEqual(row['returncode'], worker.FALLBACK)
        self.assertIsNone(row['guard'])
        self.assertFalse((self.root / 'NPU-STOP').exists())


if __name__ == '__main__':
    unittest.main()
