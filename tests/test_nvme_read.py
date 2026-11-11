import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1]/'benchmarks/roofline/nvme_read.py'
spec = importlib.util.spec_from_file_location('roofline_nvme', SOURCE)
nvme = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nvme)


class NVMeReadTests(unittest.TestCase):
    def test_energy_only_battery_has_cutoff_and_cannot_imply_external_power(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            supply = root/'sys/class/power_supply/bat'
            supply.mkdir(parents=True)
            for name, value in {'type': 'Battery', 'energy_now': '190',
                                'energy_full': '1000', 'online': '1'}.items():
                (supply/name).write_text(value)
            usb = supply.parent/'usb'; usb.mkdir()
            (usb/'type').write_text('USB'); (usb/'online').write_text('0')
            with patch.object(nvme, 'Path', side_effect=lambda p: root/str(p).lstrip('/')):
                values = nvme.sensors()
            self.assertEqual(values['bat/capacity'], 19)
            self.assertNotIn('bat/online', values)
            self.assertFalse(any(v for k,v in values.items() if k.endswith('/online')))
            (supply/'capacity').write_text('21')
            (usb/'online').write_text('1')
            with patch.object(nvme, 'Path', side_effect=lambda p: root/str(p).lstrip('/')):
                values = nvme.sensors()
            self.assertEqual(values['bat/capacity'], 21)
            self.assertEqual(values['usb/online'], 1)

    def test_every_arm_is_readonly_existing_file_no_implicit_cache_invalidation(self):
        arms = list(nvme.matrix())
        self.assertEqual(len(arms), 120)
        for arm in arms:
            cmd = nvme.command('/usr/bin/fio', Path('/owned/weights.gguf'), 50*1024**3, 'io_uring', 2, arm)
            for required in ['--readonly', '--rw=randread', '--allow_file_create=0',
                             '--create_on_open=0', '--unlink=0', '--invalidate=0', '--fadvise_hint=0']:
                self.assertIn(required, cmd)
            self.assertFalse(any('drop_caches' in part or '--rw=write' in part for part in cmd))
        self.assertEqual([a['direct'] for a in arms[:2]], [1, 0])
        self.assertEqual([a['direct'] for a in arms[40:42]], [0, 1])
        self.assertEqual(arms[0]['seed'], arms[1]['seed'])
        self.assertNotEqual(arms[0]['seed'], arms[40]['seed'])

    def test_fio_bandwidth_KiB_to_bytes_and_explicit_decimal_binary_units(self):
        read = {'bw': 1024, 'iops': 4, 'io_bytes': 1048576, 'runtime': 1000,
                'lat_ns': {'mean': 1234}}
        value = nvme.fio_metrics({'jobs': [{'error': 0, 'read': read}]})
        self.assertEqual(value['bandwidth_bytes_s'], 1048576)
        self.assertEqual(value['GB_s'], 1048576/1e9)
        self.assertEqual(value['GiB_s'], 1/1024)
        read.pop('bw'); read['bw_bytes'] = 2000000000
        self.assertEqual(nvme.fio_metrics({'jobs': [{'error': 0, 'read': read}]})['GB_s'], 2)
        with self.assertRaises(ValueError):
            nvme.fio_metrics({'jobs': [{'error': 0, 'read': read, 'write': {'io_bytes': 1}}]})

    def test_advice_only_held_exact_owned_inode_and_never_global(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'owned'
            path.write_bytes(b'x'*4096)
            expected = nvme.owned_file(path, 1024)
            fd = os.open(path, os.O_RDONLY|os.O_NOFOLLOW)
            try:
                with patch.object(nvme.os, 'posix_fadvise') as advise:
                    self.assertIn('not_guaranteed_cold', nvme.advice(fd, path, expected))
                    advise.assert_called_once_with(fd, 0, 4096, os.POSIX_FADV_DONTNEED)
                path.rename(Path(directory)/'old'); path.write_bytes(b'y'*4096)
                with patch.object(nvme.os, 'posix_fadvise') as advise:
                    with self.assertRaises(ValueError): nvme.advice(fd, path, expected)
                    advise.assert_not_called()
            finally: os.close(fd)

    def test_preflight_refuses_symlink_nonowner_and_insufficient_working_set(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'owned'; path.write_bytes(b'x'*4096)
            alias = Path(directory)/'alias'; alias.symlink_to(path)
            with self.assertRaises(ValueError): nvme.owned_file(alias, 0)
            with self.assertRaises(ValueError): nvme.owned_file(path, 4096)
            with patch.object(nvme.os, 'getuid', return_value=os.getuid()+1):
                with self.assertRaises(ValueError): nvme.owned_file(path, 0)


if __name__ == '__main__': unittest.main()
