"""S14 + S5 (status wording): `local-ai status` says what the controller reports and exits 1
when it is offline; load/select notices carry the controller's own reason."""
import contextlib
import importlib.machinery
import importlib.util
import io
import json
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / 'src/snapdragon_ai_lab/runtime/local-ai'
loader = importlib.machinery.SourceFileLoader('status_cli_local_ai', str(SCRIPT))
spec = importlib.util.spec_from_loader(loader.name, loader)
cli = importlib.util.module_from_spec(spec)
loader.exec_module(cli)


def fake_request(chat):
    def request(url, post=False, timeout=2, payload=None):
        if url.endswith('/_local/status'):
            return (200, chat) if chat is not None else (0, {})
        return 0, {}
    return request


class StatusCliTests(unittest.TestCase):
    def run_status(self, chat, *extra):
        out = io.StringIO()
        with patch.object(cli, 'request', side_effect=fake_request(chat)), \
                patch.object(cli, 'memory_gib', return_value=(20.0, 32.0)), \
                patch.object(cli, 'flash_download_status', return_value={'flash_download_state': 'absent'}), \
                patch.object(cli, 'flash_trial_status', return_value={'flash_trial_active': False}), \
                patch.object(cli, 'compute_busy', return_value=False), \
                contextlib.redirect_stdout(out):
            code = cli.main(['status', *extra])
        return code, out.getvalue().strip()

    def test_offline_says_offline_and_exits_1(self):
        code, text = self.run_status(None)
        self.assertEqual(code, 1)
        self.assertEqual(text, 'Qwen: offline | Images: offline | Available: 20.0 GiB')

    def test_offline_json_still_prints_for_the_panel(self):
        code, text = self.run_status(None, '--json')
        self.assertEqual((code, json.loads(text)['chat_state']), (1, 'offline'))

    def test_wording_for_each_controller_state(self):
        for state in ('ready', 'unloaded', 'loading', 'stopping'):
            with self.subTest(state=state):
                code, text = self.run_status({'state': state})
                self.assertEqual(code, 0)
                self.assertTrue(text.startswith(f'Qwen: {state} | Images: offline'), text)
        code, text = self.run_status({'state': 'something-new'})
        self.assertTrue(text.startswith('Qwen: offline'), 'unknown states are not shown verbatim')


class NoticeReasonTests(unittest.TestCase):
    def test_load_failure_notice_carries_the_controller_reason(self):
        body = {'error': 'Local model cannot load under the current memory conditions.'}
        with patch.object(cli, 'ensure_service', return_value=True), \
                patch.object(cli, 'request', return_value=(503, body)), \
                patch.object(cli, 'notify') as notify:
            self.assertEqual(cli.action('load'), 1)
        notify.assert_called_once_with(body['error'])

    def test_openai_shaped_and_empty_bodies(self):
        self.assertEqual(cli.reason({'error': {'message': ' busy '}}, 'x'), 'busy')
        self.assertEqual(cli.reason({}, 'fallback'), 'fallback')
        self.assertEqual(cli.reason({'error': 7}, 'fallback'), 'fallback')


if __name__ == '__main__':
    unittest.main()


import sys  # noqa: E402,I001
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _stub_manager import StubManagerCase  # noqa: E402
from snapdragon_ai_lab.runtime import controller  # noqa: E402


class UnexpectedStartErrorTests(StubManagerCase):
    def test_unexpected_start_error_keeps_its_cause(self):
        with patch.object(controller.subprocess, 'Popen', side_effect=PermissionError(13, 'denied')), \
                contextlib.redirect_stderr(io.StringIO()) as err, self.assertRaises(PermissionError):
            self.manager.begin(allow_start=True, model_alias='stub-local')
        failure = self.manager.status()['last_failure']
        self.assertEqual(failure['reason'], 'Local model could not be started.')
        self.assertIn('PermissionError', failure['detail'])
        self.assertIn('Traceback', err.getvalue(), 'the traceback goes to the journal')
        self.assertEqual(controller.unexpected_cause(ValueError('a\nb')), 'ValueError: a')
