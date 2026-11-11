"""Model-free fixtures: no boot paths, firmware writes, or external tools."""
import argparse
import importlib.util
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('memlat', Path(__file__).parents[1] / 'tools/memlat.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
STOCK = b'\xd0\x0d\xfe\xedstock'


def image(duplicate=False):
    data = bytearray(2048)
    data[:2] = b'MZ'
    struct.pack_into('<I', data, 0x3c, 64)
    data[64:68] = b'PE\0\0'
    struct.pack_into('<H', data, 70, 3)
    struct.pack_into('<H', data, 84, 240)
    struct.pack_into('<H', data, 88, 0x20b)
    for i, (name, content) in enumerate(((b'.dtbauto', STOCK), (b'.dtbauto', STOCK if duplicate else b'other'), (b'.linux', b'kernel'))):
        h = 328 + i * 40
        data[h:h + len(name)] = name
        struct.pack_into('<4I', data, h + 8, len(content), 4096 * (i + 1), 128, 512 + i * 256)
        data[512 + i * 256:512 + i * 256 + len(content)] = content
    return bytes(data)


class MemlatTests(unittest.TestCase):
    def test_only_selected_payload_and_virtual_size_change(self):
        original = image()
        changed = m.patch_uki(original, STOCK, STOCK + b'memlat')
        allowed = set(range(336, 340)) | set(range(512, 640))
        self.assertEqual(len(original), len(changed))
        self.assertTrue(all(a == b or i in allowed for i, (a, b) in enumerate(zip(original, changed))))

    def test_duplicate_exact_payload_refused(self):
        with self.assertRaises(ValueError):
            m.patch_uki(image(True), STOCK, STOCK)

    def test_bounds_signed_and_oversize_refused(self):
        for data in (image()[:600], b'bad'):
            with self.assertRaises(ValueError):
                m.patch_uki(data, STOCK, STOCK)
        signed = bytearray(image())
        struct.pack_into('<2I', signed, 232, 1500, 20)
        with self.assertRaises(ValueError):
            m.sections(signed)
        with self.assertRaises(ValueError):
            m.patch_uki(image(), STOCK, STOCK + bytes(128))

    def setup_install(self, root):
        source, dest, config, manifest = [root / name for name in ('input.efi', 'owned.efi', 'limine.conf', 'owned.json')]
        source.write_bytes(image())
        config.write_bytes(b'timeout: 5\ndefault_entry: 2\n/+Omarchy\n//Linux\n protocol: efi\n path: boot():/stock.efi\n')
        return argparse.Namespace(image=str(source), destination=str(dest), config=str(config), manifest=str(manifest), efi_path='boot():/owned.efi')

    def test_install_and_remove_preserve_default_and_intervening_changes(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(m, 'secure_boot', return_value='disabled'):
            args = self.setup_install(Path(tmp))
            config = Path(args.config)
            before = config.read_bytes()
            m.install(args)
            self.assertTrue(config.read_bytes().startswith(before))
            extra = b'\n/Another later entry\n protocol: efi\n path: boot():/later.efi\n'
            config.write_bytes(config.read_bytes() + extra)
            m.uninstall(args)
            self.assertEqual(config.read_bytes(), before + extra)
            self.assertFalse(Path(args.destination).exists())

    def test_install_refuses_invalid_or_ambiguous_existing_default(self):
        for old, new in ((b'default_entry: 2', b'default_entry: 3'),
                         (b'default_entry: 2', b'default_entry: 1'),
                         (b'default_entry: 2', b'default_entry: Stock')):
            with tempfile.TemporaryDirectory() as tmp, patch.object(m, 'secure_boot', return_value='disabled'):
                args = self.setup_install(Path(tmp))
                config = Path(args.config)
                config.write_bytes(config.read_bytes().replace(old, new))
                before = config.read_bytes()
                with self.assertRaises(ValueError):
                    m.install(args)
                self.assertEqual(config.read_bytes(), before)
                self.assertFalse(Path(args.destination).exists())
                self.assertFalse(Path(args.manifest).exists())

    def test_remove_refuses_owned_or_later_default_without_writes(self):
        for target in (b'3', b'4', b'Local AI MEMLAT opt-in'):
            with tempfile.TemporaryDirectory() as tmp, patch.object(m, 'secure_boot', return_value='disabled'):
                args = self.setup_install(Path(tmp))
                m.install(args)
                config = Path(args.config)
                after = config.read_bytes().replace(b'default_entry: 2', b'default_entry: ' + target)
                after += b'\n/Later\n protocol: efi\n path: boot():/later.efi\n'
                config.write_bytes(after)
                with self.assertRaises(ValueError):
                    m.uninstall(args)
                self.assertEqual(config.read_bytes(), after)
                self.assertTrue(Path(args.destination).exists())
                self.assertTrue(Path(args.manifest).exists())

    def test_modified_owned_artifacts_refused(self):
        for kind in ('image', 'entry'):
            with tempfile.TemporaryDirectory() as tmp, patch.object(m, 'secure_boot', return_value='disabled'):
                args = self.setup_install(Path(tmp))
                m.install(args)
                target = Path(args.destination if kind == 'image' else args.config)
                target.write_bytes(target.read_bytes().replace(b'opt-in', b'edited') if kind == 'entry' else b'changed')
                before = Path(args.config).read_bytes()
                with self.assertRaises(ValueError):
                    m.uninstall(args)
                self.assertEqual(Path(args.config).read_bytes(), before)
                self.assertTrue(Path(args.destination).exists())

    def test_secure_boot_unknown_or_enabled_refused_without_writes(self):
        for state in ('unknown', 'enabled'):
            with tempfile.TemporaryDirectory() as tmp, patch.object(m, 'secure_boot', return_value=state):
                args = self.setup_install(Path(tmp))
                with self.assertRaises(ValueError):
                    m.install(args)
                self.assertFalse(Path(args.destination).exists())
                self.assertFalse(Path(args.manifest).exists())


if __name__ == '__main__':
    unittest.main()
