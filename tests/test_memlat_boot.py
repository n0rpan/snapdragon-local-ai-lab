"""Boot-default and update rollback fixtures; never access the real ESP."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
import json
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('memlat_boot', Path(__file__).parents[1] / 'tools/memlat_boot.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
MENU = '''default_entry: 2
/+Omarchy
  //linux-omarchy
  protocol: efi
  path: boot():/EFI/Linux/stock.efi
  //Snapshots
    ///Old
    protocol: efi
    path: boot():/old.efi
/Local AI memlat test (7.3.0-40)
protocol: efi
path: boot():/EFI/LocalAI/test.efi
'''
OWN = 'Local AI memlat test (7.3.0-40)'


class BootDefaultTests(unittest.TestCase):
    def fixture(self, directory):
        root = Path(directory)
        config, stock, modules = root / 'limine.conf', root / 'stock.efi', root / 'modules'
        config.write_text(MENU)
        stock.write_bytes(b'qualified stock')
        modules.mkdir()
        return dict(config=str(config), stock_entry='Omarchy/linux-omarchy', memlat_entry=OWN,
                    stock_image=str(stock), stock_sha256=m.sha256(stock), modules=str(modules),
                    memlat_image='/boot/EFI/LocalAI/test.efi', memlat_sha256='known',
                    pending=str(root / 'pending'), image_directory=directory, boot_root=directory)

    def test_numeric_and_named_selection_survive_snapshot_renumbering(self):
        self.assertEqual(m.selected(MENU, m.entries(MENU)), 'Omarchy/linux-omarchy')
        named = MENU.replace('default_entry: 2', 'default_entry: ' + OWN)
        inserted = named.replace('/Local AI memlat', '/Another entry\nprotocol: efi\npath: boot():/other\n/Local AI memlat')
        self.assertEqual(m.selected(inserted, m.entries(inserted)), OWN)

    def test_enable_fallback_idempotence_and_preservation(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(m, 'verify_entry'):
            policy = self.fixture(directory)
            self.assertTrue(m.apply(policy, 'enable')['changed'])
            self.assertFalse(m.apply(policy, 'enable')['changed'])
            expected = MENU.replace('default_entry: 2', 'default_entry: Omarchy/linux-omarchy')
            self.assertEqual(m.apply(policy, 'fallback')['default'], 'Omarchy/linux-omarchy')
            self.assertEqual(Path(policy['config']).read_text(), expected)
            self.assertFalse(m.apply(policy, 'fallback')['changed'])

    def test_missing_modules_and_changed_stock_reject_enable(self):
        for failure in ('modules', 'hash'):
            with tempfile.TemporaryDirectory() as directory:
                policy = self.fixture(directory)
                if failure == 'modules':
                    Path(policy['modules']).rmdir()
                def verify(menu, name, image, expected=None):
                    if failure == 'hash' and expected == policy['stock_sha256']:
                        raise ValueError('changed stock')
                with patch.object(m, 'verify_entry', side_effect=verify), self.assertRaises(ValueError):
                    m.apply(policy, 'enable')
                self.assertEqual(Path(policy['config']).read_text(), MENU)

    def test_check_falls_back_if_stock_changed_or_modules_removed(self):
        for failure in ('stock', 'modules', 'image'):
            with tempfile.TemporaryDirectory() as directory, patch.object(m, 'verify_entry') as verify:
                policy = self.fixture(directory)
                m.apply(policy, 'enable')
                if failure == 'stock':
                    Path(policy['stock_image']).write_bytes(b'new kernel')
                elif failure == 'modules':
                    Path(policy['modules']).rmdir()
                else:
                    verify.side_effect = lambda menu, name, image, expected=None: (_ for _ in ()).throw(ValueError('bad image')) if name == OWN else None
                self.assertEqual(m.apply(policy, 'check')['default'], 'Omarchy/linux-omarchy')

    def test_other_user_default_is_not_overridden_by_update_hooks(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(m, 'verify_entry') as verify:
            policy = self.fixture(directory)
            for action in ('check', 'fallback'):
                self.assertFalse(m.apply(policy, action)['changed'])
            verify.assert_not_called()

    def test_ambiguity_and_remember_last_rejected(self):
        for text in (MENU + '\ndefault_entry: 1\n', MENU + '\nremember_last_entry: yes\n', MENU.replace('/+Omarchy', '/+Omarchy\ninclude: another')):
            with self.assertRaises(ValueError):
                m.selected(text, m.entries(text))

    def test_unbootable_fallback_and_bad_hash_rejected(self):
        menu = m.entries(MENU)
        for name, image in (('Omarchy', Path('/boot/no')), ('missing', Path('/boot/no'))):
            with self.assertRaises(ValueError):
                m.verify_entry(menu, name, image)
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / 'fake'
            image.write_bytes(b'payload')
            # A fake filesystem fixture retains the exact boot():/ mapping check.
            with patch.object(Path, 'relative_to', return_value=Path('EFI/Linux/stock.efi')):
                menu['Omarchy/linux-omarchy']['path'] += '#' + '0' * 128
                with self.assertRaises(ValueError):
                    m.verify_entry(menu, 'Omarchy/linux-omarchy', image)

    def test_compare_and_replace_refuses_intervening_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config'
            path.write_text('user edit')
            with self.assertRaises(ValueError):
                m.replace(path, 'old', 'new')
            self.assertEqual(path.read_text(), 'user edit')

    def test_update_prepares_marker_before_fallback_and_repeated_calls_keep_it(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(m, 'verify_entry'):
            policy = self.fixture(directory)
            m.apply(policy, 'enable')
            self.assertEqual(m.apply(policy, 'prepare-update')['default'], 'Omarchy/linux-omarchy')
            self.assertTrue(Path(policy['pending']).exists())
            self.assertFalse(m.apply(policy, 'prepare-update')['changed'])
            self.assertTrue(Path(policy['pending']).exists())

    def build_stub(self, policy, output):
        output.write_bytes(b'patched kernel fixture')
        return dict(kernel='new-kernel', modules=policy['modules'], stock_sha256=policy['stock_sha256'],
                    memlat_sha256=m.sha256(output), stock_dtb_sha256='fixture')

    def test_dry_run_never_changes_menu_policy_or_pending(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(m, 'build_image', side_effect=self.build_stub):
            policy = self.fixture(directory)
            path = Path(directory) / 'policy'
            path.write_text(json.dumps(policy))
            before = path.read_bytes()
            result = m.rebuild(path, policy, dry_run=True)
            self.assertTrue(result['passed'])
            self.assertFalse(result['changed'])
            self.assertEqual(Path(policy['config']).read_text(), MENU)
            self.assertEqual(path.read_bytes(), before)
            self.assertFalse(Path(policy['pending']).exists())

    def test_update_rebuild_is_repeatable_and_preserves_original_test_entry(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(m, 'verify_entry'), patch.object(m, 'build_image', side_effect=self.build_stub):
            policy = self.fixture(directory)
            path = Path(directory) / 'policy'
            path.write_text(json.dumps(policy))
            m.apply(policy, 'enable')
            for _ in range(2):
                m.apply(policy, 'prepare-update')
                result = m.rebuild(path, policy)
                self.assertEqual(result['default'], 'Local AI memlat (new-kernel)')
                self.assertFalse(Path(policy['pending']).exists())
                policy = json.loads(path.read_text())
                menu = m.entries(Path(policy['config']).read_text())
                self.assertIn(OWN, menu)
                self.assertIn(policy['memlat_entry'], menu)
                self.assertEqual(m.selected(Path(policy['config']).read_text(), menu), policy['memlat_entry'])

    def test_build_failure_keeps_stock_and_notifies(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(m, 'verify_entry'), patch.object(m, 'build_image', side_effect=ValueError('DTB mismatch')), patch.object(m, 'notify_failure') as notification:
            policy = self.fixture(directory)
            path = Path(directory) / 'policy'
            path.write_text(json.dumps(policy))
            m.apply(policy, 'enable')
            m.apply(policy, 'prepare-update')
            with self.assertRaises(ValueError):
                m.rebuild(path, policy)
            text = Path(policy['config']).read_text()
            self.assertEqual(m.selected(text, m.entries(text)), policy['stock_entry'])
            self.assertTrue(Path(policy['pending']).exists())
            notification.assert_called_once()

    def test_no_pending_means_no_build_or_default_change(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(m, 'build_image') as build:
            policy = self.fixture(directory)
            result = m.rebuild(Path(directory) / 'no-policy-needed', policy)
            self.assertFalse(result['changed'])
            build.assert_not_called()

    def test_rebuild_preserves_stock_outer_command_line(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(m, 'verify_entry'), patch.object(m, 'build_image', side_effect=self.build_stub):
            policy = self.fixture(directory)
            config = Path(policy['config'])
            config.write_text(MENU.replace('path: boot():/EFI/Linux/stock.efi', 'path: boot():/EFI/Linux/stock.efi\n  cmdline: root=fixture quiet extra=outer'))
            path = Path(directory) / 'policy'
            path.write_text(json.dumps(policy))
            m.rebuild(path, policy, force=True)
            updated = json.loads(path.read_text())
            menu = m.entries(config.read_text())
            self.assertEqual(menu[updated['memlat_entry']]['cmdline'], 'root=fixture quiet extra=outer')

    def test_generator_formatting_is_accepted_but_semantic_edits_are_not(self):
        block = '\n/Automatic\ncomment: owned\nprotocol: efi\npath: boot():/owned\n'
        normalized = '\n/Automatic\n  path: boot():/owned\n  protocol: efi\n  comment: owned\n'
        policy = dict(memlat_entry='Automatic', automatic_entry_block=block)
        text = MENU + normalized + '/Unrelated\nprotocol: efi\npath: boot():/other\n'
        self.assertNotIn('/Automatic', m.remove_owned_block(text, policy))
        for changed in (text.replace('/owned', '/different'), text.replace('  comment: owned', '  cmdline: unexpected')):
            with self.assertRaises(ValueError):
                m.remove_owned_block(changed, policy)

    def test_empty_command_line_does_not_consume_the_next_field(self):
        text = MENU.replace('  protocol: efi', '  cmdline:\n  protocol: efi', 1)
        self.assertEqual(m.entries(text)['Omarchy/linux-omarchy']['cmdline'], '')


if __name__ == '__main__':
    unittest.main()
