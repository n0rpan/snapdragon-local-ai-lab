"""Audit S7: expected conditions (no Omarchy, no notification daemon, a bad mode) give one line,
never a traceback."""
import importlib.machinery
import importlib.util
import io
from pathlib import Path
import unittest
from unittest.mock import patch

from snapdragon_ai_lab import __main__ as entry
from snapdragon_ai_lab.runtime import launch_local_plan as launcher

RUNTIME = Path(launcher.__file__).parent
loader = importlib.machinery.SourceFileLoader('friendly_errors_cli', str(RUNTIME / 'local-ai'))
spec = importlib.util.spec_from_loader(loader.name, loader)
cli = importlib.util.module_from_spec(spec)
loader.exec_module(cli)
MISSING = FileNotFoundError(2, 'No such file or directory')


class CliWithoutDesktop(unittest.TestCase):
    def test_notify_without_notify_send_writes_stderr(self):
        with patch.object(cli.subprocess, 'run', side_effect=MISSING), \
                patch('sys.stderr', new_callable=io.StringIO) as err:
            cli.notify('hello')
        self.assertIn('hello', err.getvalue())

    def test_panel_without_omarchy_shell_returns_1(self):
        with patch.object(cli.subprocess, 'run', side_effect=MISSING), \
                patch('sys.stderr', new_callable=io.StringIO) as err:
            self.assertEqual(cli.action('panel'), 1)
        self.assertIn('omarchy-shell', err.getvalue())

    def test_chat_and_images_without_omarchy_return_1(self):
        for name in ('chat', 'images'):
            with self.subTest(name=name), patch.object(cli, 'ensure_service', return_value=True), \
                    patch.object(cli, 'request', return_value=(200, {})), \
                    patch.object(cli.subprocess, 'Popen', side_effect=MISSING), \
                    patch.object(cli, 'notify') as notice:
                self.assertEqual(cli.action(name), 1)
            self.assertIn('omarchy', notice.call_args.args[0])


class InstallPluginOffOmarchy(unittest.TestCase):
    def test_refusal_is_one_line_and_exit_1(self):
        with patch.object(entry, 'install', side_effect=RuntimeError('Omarchy is required for this optional panel')), \
                patch('sys.stderr', new_callable=io.StringIO) as err, self.assertRaises(SystemExit) as stop:
            entry.main(['install-plugin', '--dry-run'])
        self.assertEqual(stop.exception.code, 1)
        self.assertEqual(err.getvalue(), 'install-plugin: Omarchy is required for this optional panel\n')


class PlanLaunchMode(unittest.TestCase):
    def test_plan_launch_refuses_an_unknown_mode(self):
        with patch.dict(launcher.os.environ, {launcher.OPENCODE_MODE_ENV: 'huge'}), \
                patch.object(launcher.os, 'execvpe', side_effect=AssertionError('must not launch')), \
                self.assertRaisesRegex(ValueError, 'unsupported_opencode_mode'):
            launcher.main(['/synthetic/project'])

    def test_plan_launch_accepts_the_known_modes(self):
        for mode in ('', 'fast', 'deep', 'max'):
            with self.subTest(mode=mode), patch.dict(launcher.os.environ, {launcher.OPENCODE_MODE_ENV: mode}), \
                    patch.object(launcher.os, 'execvpe') as execute:
                launcher.main(['/synthetic/project'])
            execute.assert_called_once()


if __name__ == '__main__':
    unittest.main()
