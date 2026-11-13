"""v0.1.3 installer: new panel code over an installed panel needs a shell restart (WR-1), which the
installer reports and only runs when asked; and the plugin id carries no login name (F58).
Everything runs in a temporary Omarchy root with a recording runner; no shell is touched."""
import json
import os
import pwd
import subprocess
import tempfile
import unittest
from importlib import resources
from pathlib import Path

from snapdragon_ai_lab.plugin_install import PLUGIN_ID, RESTART_COMMAND, code_files, install, restart_needed

PACKAGED = resources.files('snapdragon_ai_lab').joinpath('omarchy_plugin')


class PluginRestart(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / 'omarchy'; self.root.mkdir()
        (self.root / 'shell.json').write_text('{"bar":{}}\n')
        self.cli = self.root / 'local-ai'; self.cli.write_text('#!/bin/sh\nexit 0\n'); self.cli.chmod(0o700)
        self.info = {'omarchy_command_present': True, 'snapdragon_x_detected': True}
        self.commands = []
        self.target = self.root / 'plugins' / PLUGIN_ID

    def runner(self, command, **kwargs):
        self.commands.append(list(command))
        return subprocess.CompletedProcess(command, 0, '', '')

    def call(self, **kwargs):
        kwargs.setdefault('runner', self.runner)
        return install(config_root=self.root, cli_path=self.cli, platform_info=self.info, **kwargs)

    def restarts(self):
        return [command for command in self.commands if command == list(RESTART_COMMAND)]

    def test_first_install_needs_no_restart(self):
        result = self.call()
        self.assertEqual((result['restart_required'], result['shell_restarted']), (False, False))
        self.assertNotIn('next_step', result)
        self.assertEqual(result['restart_command'], 'omarchy restart shell')
        self.assertEqual(self.restarts(), [])

    def test_same_code_again_needs_no_restart_even_with_new_texts(self):
        self.call()
        catalog = self.target / 'i18n/en.json'
        catalog.write_text(json.dumps({'title': 'Older text'}))
        (self.target / 'manifest.json').write_text('{"version": "0.0.1"}')
        result = self.call(restart_shell=True)
        self.assertEqual((result['restart_required'], result['shell_restarted']), (False, False))
        self.assertEqual(self.restarts(), [])
        self.assertEqual(json.loads(catalog.read_text())['title'], 'Local AI')

    def test_changed_code_reports_the_restart_and_does_not_run_it_by_default(self):
        self.call()
        (self.target / 'NowCard.qml').write_text('import QtQuick\nItem {}\n')
        preview = self.call(dry_run=True)
        self.assertTrue(preview['restart_required'])
        self.assertIn('omarchy restart shell', preview['next_step'])
        self.assertEqual((self.target / 'NowCard.qml').read_text(), 'import QtQuick\nItem {}\n')  # dry run wrote nothing
        result = self.call()
        self.assertEqual((result['restart_required'], result['shell_restarted']), (True, False))
        self.assertIn('omarchy restart shell', result['next_step'])
        self.assertEqual(self.restarts(), [])
        self.assertEqual((self.target / 'NowCard.qml').read_bytes(), PACKAGED.joinpath('NowCard.qml').read_bytes())

    def test_each_kind_of_code_change_counts(self):
        self.call()
        changes = (lambda: (self.target / 'Model.js').write_text('.pragma library\n'),
                   lambda: (self.target / 'Extra.qml').write_text('import QtQuick\nItem {}\n'),
                   lambda: (self.target / 'Label.qml').unlink(),
                   lambda: (self.target / 'qmldir').write_text('module old\n'))
        for change in changes:
            change()
            self.assertTrue(self.call(dry_run=True)['restart_required'])
            self.call()
            self.assertFalse(self.call(dry_run=True)['restart_required'])

    def test_restart_runs_only_when_asked_and_after_the_install(self):
        self.call()
        (self.target / 'Panel.qml').write_text('import QtQuick\nItem {}\n')
        seen = []

        def runner(command, **kwargs):
            if list(command) == list(RESTART_COMMAND):
                seen.append((self.target / 'Panel.qml').read_bytes() == PACKAGED.joinpath('Panel.qml').read_bytes())
            return self.runner(command, **kwargs)
        result = self.call(restart_shell=True, runner=runner)
        self.assertEqual((result['restart_required'], result['shell_restarted']), (True, True))
        self.assertNotIn('next_step', result)
        self.assertEqual(self.restarts(), [['omarchy', 'restart', 'shell']])
        self.assertEqual(seen, [True])  # the new code was in place before the shell restarted
        self.assertEqual(self.commands[-1], ['omarchy', 'restart', 'shell'])

    def test_dry_run_never_restarts(self):
        self.call()
        (self.target / 'Panel.qml').write_text('import QtQuick\nItem {}\n')
        result = self.call(dry_run=True, restart_shell=True)
        self.assertEqual((result['restart_required'], result['shell_restarted']), (True, False))
        self.assertEqual(self.restarts(), [])

    def test_failed_restart_keeps_the_install_and_says_what_to_run(self):
        self.call()
        (self.target / 'Panel.qml').write_text('import QtQuick\nItem {}\n')
        for failure in (subprocess.CalledProcessError(1, 'omarchy'), FileNotFoundError('omarchy'),
                        subprocess.TimeoutExpired('omarchy', 5)):
            with self.subTest(failure=type(failure).__name__):
                (self.target / 'Panel.qml').write_text('import QtQuick\nItem {}\n')

                def runner(command, **kwargs):
                    if list(command) == list(RESTART_COMMAND):
                        raise failure
                    return self.runner(command, **kwargs)
                result = self.call(restart_shell=True, runner=runner)
                self.assertEqual((result['restart_required'], result['shell_restarted']), (True, False))
                self.assertIn('omarchy restart shell', result['next_step'])
                self.assertEqual((self.target / 'Panel.qml').read_bytes(), PACKAGED.joinpath('Panel.qml').read_bytes())
                self.assertTrue(Path(result['backup_directory']).is_dir())

    def test_migration_from_another_id_needs_no_restart(self):
        layout = {'right': [{'id': 'prototype.local-ai'}]}
        (self.root / 'shell.json').write_text(json.dumps({'bar': {'layout': layout}}))
        old = self.root / 'plugins/prototype.local-ai'; old.mkdir(parents=True)
        (old / 'Panel.qml').write_text('import QtQuick\nItem {}\n')
        result = self.call(replace_plugin='prototype.local-ai', restart_shell=True)
        self.assertEqual((result['restart_required'], result['shell_restarted']), (False, False))
        self.assertEqual(self.restarts(), [])

    def test_code_files_cover_what_the_shell_compiles(self):
        files = code_files(PACKAGED)
        for name in ('Panel.qml', 'NowCard.qml', 'ModelsView.qml', 'ActivityView.qml', 'StorageView.qml', 'Model.js'):
            self.assertIn(name, files)
        self.assertFalse([name for name in files if name.endswith('.json')])
        self.assertFalse(restart_needed(self.target, PACKAGED))  # nothing installed yet


class PluginIdentity(unittest.TestCase):
    def test_plugin_id_carries_no_login_name(self):
        """F58: the id is the public one everywhere; no login name, so nothing to migrate from."""
        login = pwd.getpwuid(os.getuid()).pw_name
        manifest = json.loads(PACKAGED.joinpath('manifest.json').read_text())
        panel = PACKAGED.joinpath('Panel.qml').read_text()
        self.assertEqual(PLUGIN_ID, 'snapdragon.local-ai')
        self.assertEqual(manifest['id'], PLUGIN_ID)
        self.assertIn('moduleName: "%s"' % PLUGIN_ID, panel)
        self.assertIn('ipcTarget: "%s"' % PLUGIN_ID, panel)
        for path in PACKAGED.iterdir():
            if path.is_file():
                text = path.read_text()
                self.assertNotRegex(text, r'[a-z][a-z0-9-]*\.local-ai(?![a-z-])(?<!snapdragon\.local-ai)', path.name)
                self.assertNotIn('/home/', text, path.name)
                self.assertNotIn(login.lower() + '.local-ai', text.lower(), path.name)


if __name__ == '__main__':
    unittest.main()
