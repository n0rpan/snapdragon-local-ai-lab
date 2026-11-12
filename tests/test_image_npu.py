import hashlib
import os
from pathlib import Path
import tempfile
import unittest

from snapdragon_ai_lab.runtime import image_npu as npu


class ImageNpuPolicyTests(unittest.TestCase):
    def choose(self, **values):
        args = dict(enabled=True, ready=True, mem_available=20*npu.GIB, swap_free=10*npu.GIB)
        args.update(values)
        return npu.choose(**args)

    def test_off_and_fault_latch_always_fall_back(self):
        self.assertEqual(npu.choose().backend, 'cpu')
        self.assertEqual(self.choose(enabled='true').backend, 'cpu')
        self.assertEqual(self.choose(latched=True).backend, 'cpu')
        self.assertEqual(self.choose(ready=False).backend, 'cpu')

    def test_gate_is_only_the_qualified_model_and_size(self):
        self.assertEqual(self.choose().backend, 'npu')
        self.assertEqual(self.choose(width=768, height=768).backend, 'cpu')
        self.assertEqual(self.choose(model='klein-q4').backend, 'cpu')

    def test_dma_and_host_growth_are_both_budgeted(self):
        self.assertGreater(npu.REQUIRED_HOST_BYTES, npu.QUALIFIED_DMA_BYTES)
        self.assertGreater(npu.QUALIFIED_DMA_BYTES, int(7.7*npu.GIB))
        self.assertEqual(self.choose(mem_available=npu.REQUIRED_HOST_BYTES-1).backend, 'cpu')
        self.assertEqual(self.choose(mem_available=npu.REQUIRED_HOST_BYTES).backend, 'npu')

    def test_resident_chat_is_retained_and_budgeted(self):
        self.assertEqual(self.choose(mem_total=31*npu.GIB, resident_model_bytes=15*npu.GIB,
                                     mem_available=4*npu.GIB).backend, 'npu')
        self.assertEqual(self.choose(mem_total=31*npu.GIB, resident_model_bytes=19*npu.GIB).backend, 'cpu')
        self.assertEqual(self.choose(resident_model_bytes=15*npu.GIB).backend, 'cpu')
        self.assertEqual(self.choose(mem_total=31*npu.GIB, resident_model_bytes=15*npu.GIB,
                                     mem_available=4*npu.GIB, swap_free=0).backend, 'cpu')

    def test_missing_telemetry_and_oom_risk_refuse_npu(self):
        for value in [None, -1, True]:
            self.assertEqual(self.choose(mem_available=value).backend, 'cpu')
        self.assertEqual(self.choose(mem_available=npu.GIB//2, swap_free=npu.GIB).backend, 'cpu')

    def test_inherited_experiment_knobs_are_removed(self):
        environment = {'PATH': '/bin', 'OXI_NPU_CPU_ROWS': '25', 'OXN_ATTENDED': '0',
                       'DSPQUEUE_QUEUE_MONITOR': '1', 'FASTRPC_LIB': '/tmp/lib.so',
                       'LD_PRELOAD': '/tmp/hook.so', 'GGML_X': '1'}
        self.assertEqual(npu.clean_environment(environment), {'PATH': '/bin'})


class PackageIdentityTests(unittest.TestCase):
    def test_tampered_or_symlinked_or_writable_package_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'package'
            path.write_bytes(b'qualified fixture')
            path.chmod(0o600)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            npu.verify_file(path, digest)
            path.write_bytes(b'changed fixture')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                npu.verify_file(path, digest)
            path.chmod(0o666)
            with self.assertRaisesRegex(ValueError, 'unsafe'):
                npu.verify_file(path, digest)
            alias = Path(directory)/'alias'
            alias.symlink_to(path)
            with self.assertRaises(OSError):
                npu.verify_file(alias, digest)


if __name__ == '__main__':
    unittest.main()
