import importlib.util
from pathlib import Path
import tempfile
import unittest
import subprocess
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('memlat_shutdown', Path(__file__).parents[1]/'tools/memlat_shutdown.py')
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


class ShutdownTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.driver = self.root/'bus/scmi_protocol/drivers'/helper.DRIVER
        self.driver.mkdir(parents=True)

    def device(self, number=3, protocol='0x80', name='qcom-generic-ext', attached=True):
        device = self.root/'bus/scmi_protocol/devices'/f'scmi_dev.{number}'
        device.mkdir(parents=True)
        (device/'protocol_id').write_text(protocol)
        (device/'name').write_text(name)
        if attached:
            (device/'driver').symlink_to(self.driver)
        return device

    def test_stock_boot_is_noop(self):
        self.device(protocol='0x13')
        self.assertEqual(helper.control('unbind', self.root)['state'], 'absent')

    def test_manual_stop_or_restart_never_unbinds(self):
        self.device()
        for state in ('running', 'degraded', 'starting', '', 'offline'):
            result = helper.control('shutdown', self.root, state=lambda: state,
                                    write=lambda *_: self.fail('manual stop wrote sysfs'))
            self.assertFalse(result['changed'])

    def test_unavailable_pid1_never_unbinds(self):
        self.device()
        def unavailable():
            raise OSError('fixture unavailable')
        result = helper.control('shutdown', self.root, state=unavailable,
                                write=lambda *_: self.fail('missing state wrote sysfs'))
        self.assertFalse(result['changed'])
        self.assertEqual(result['state'], 'query-failed')

    def test_shutdown_stopping_uses_exact_unbind(self):
        device = self.device()
        writes = []
        def write(path, value):
            writes.append((path.name, value))
            (device/'driver').unlink()
        result = helper.control('shutdown', self.root, state=lambda: 'stopping',
                                jobs=lambda: [('reboot.target', 'start')], write=write)
        self.assertTrue(result['changed'])
        self.assertEqual(writes, [('unbind', 'scmi_dev.3')])

    def test_soft_reboot_or_non_shutdown_job_never_unbinds(self):
        self.device()
        for queued in ([], [('soft-reboot.target', 'start')],
                       [('systemd-soft-reboot.service', 'start')],
                       [('reboot.target', 'start'), ('soft-reboot.target', 'start')],
                       [('reboot.target', 'stop')], [('rescue.target', 'start')]):
            result = helper.control('shutdown', self.root, state=lambda: 'stopping',
                                    jobs=lambda: queued, write=lambda *_: self.fail('unsafe unbind'))
            self.assertFalse(result['changed'])

    def test_all_normal_kernel_shutdown_targets_accepted(self):
        for unit in helper.KERNEL_SHUTDOWN_UNITS:
            self.assertIn(unit.split('.')[0].removeprefix('systemd-'), ('reboot', 'poweroff', 'halt', 'kexec'))
            device = self.device(attached=True)
            def write(path, value):
                self.assertEqual(path, self.driver/'unbind')
                (device/'driver').unlink()
            result = helper.control('shutdown', self.root, state=lambda: 'stopping',
                                    jobs=lambda: [(unit, 'start')], write=write)
            self.assertTrue(result['changed'])
            for p in device.iterdir():
                p.unlink()
            device.rmdir()

    def test_failed_job_query_never_unbinds(self):
        self.device()
        def failed():
            raise subprocess.CalledProcessError(1, 'busctl')
        result = helper.control('shutdown', self.root, state=lambda: 'stopping', jobs=failed,
                                write=lambda *_: self.fail('failed query unbound client'))
        self.assertFalse(result['changed'])
        self.assertEqual(result['state'], 'query-failed')

    def test_typed_job_parser_refuses_malformed_data(self):
        valid = '{"type":"a(usssoo)","data":[[[4,"reboot.target","start","waiting","/job/4","/unit/reboot"]]]}'
        with patch.object(helper.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, valid)):
            self.assertEqual(helper.system_jobs(), [('reboot.target', 'start')])
        for body in ('[]', 'null', '{}', '{"type":"a(usssoo)","data":[[[]]]}', 'not json'):
            with patch.object(helper.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, body)):
                with self.assertRaises(ValueError):
                    helper.system_jobs()

    def test_exact_client_only_and_bind_cycle(self):
        device = self.device()
        self.device(4, protocol='0x13')
        writes = []
        def write(path, value):
            writes.append((path.name, value))
            if path.name == 'unbind':
                (device/'driver').unlink()
            else:
                (device/'driver').symlink_to(self.driver)
        self.assertTrue(helper.control('unbind', self.root, write=write)['changed'])
        self.assertFalse(helper.control('unbind', self.root, write=write)['changed'])
        self.assertTrue(helper.control('bind', self.root, write=write)['changed'])
        self.assertEqual(writes, [('unbind', 'scmi_dev.3'), ('bind', 'scmi_dev.3')])

    def test_multiple_matches_refused(self):
        self.device(); self.device(5)
        with self.assertRaisesRegex(RuntimeError, 'ambiguous'):
            helper.control('unbind', self.root)

    def test_other_driver_refused(self):
        device = self.device(attached=False)
        other = self.root/'other'; other.mkdir()
        (device/'driver').symlink_to(other)
        with self.assertRaisesRegex(RuntimeError, 'different driver'):
            helper.control('unbind', self.root)

    def test_unbind_timeout_is_not_reported_as_success(self):
        self.device()
        clock = iter((0, 6))
        with self.assertRaisesRegex(RuntimeError, 'requested state'):
            helper.control('unbind', self.root, write=lambda *_: None, clock=lambda: next(clock), sleep=lambda _: None)


if __name__ == '__main__':
    unittest.main()
