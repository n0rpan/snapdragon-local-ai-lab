"""Audit S5: a stop that fails inside a control action answers 503, never a dropped connection."""
from contextlib import ExitStack
import http.client
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

from snapdragon_ai_lab.runtime import controller


class StopErrorReplies(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.config = self.root / 'runtime.json'
        self.config.write_text(json.dumps({'model_alias': 'known', 'selected_model': 'known',
                                           'profiles': {'known': {'ready': False}, 'other': {'ready': False}}}))
        for name, value in (('CONFIG', self.config), ('HARNESS_STATE', self.root / 'primers'),
                            ('LEASE', self.root / 'inference.lock')):
            self.stack.enter_context(patch.object(controller, name, value))
        self.stack.enter_context(patch.dict(controller.os.environ, {'LOCAL_AI_IMAGE_NPU': '0'}))
        self.stack.enter_context(patch('sys.stderr'))
        self.manager = controller.Manager()
        self.manager.stop_event.set()
        self.addCleanup(self.manager.close)
        self.server = controller.Server(('127.0.0.1', 0), controller.Handler)
        self.server.manager = self.manager
        thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(thread.join, 10)
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)

    def call(self, path, body):
        connection = http.client.HTTPConnection('127.0.0.1', self.server.server_address[1], timeout=10)
        try:
            connection.request('POST', path, body=body)
            response = connection.getresponse()
            return response.status, json.loads(response.read())
        finally:
            connection.close()

    def failing_cleanup(self, pending):
        def cleanup():
            if pending:
                self.manager.stop_pending = 'engine pid 1 has not exited: still running'
            raise RuntimeError('synthetic teardown failure')
        return cleanup

    def test_unload_answers_503_when_the_stop_raises(self):
        for pending in (False, True):
            with self.subTest(pending=pending), patch.object(self.manager, 'cleanup', self.failing_cleanup(pending)):
                status, body = self.call('/_local/unload', b'{}')
                self.assertEqual(status, 503)
                self.assertIn('Retry shortly', body['error'])
            self.manager.stop_pending = None

    def test_unload_answers_503_while_the_exit_stays_unproven(self):
        def cleanup():
            self.manager.stop_pending = 'engine pid 1 has not exited: still running'
        with patch.object(self.manager, 'cleanup', cleanup):
            status, body = self.call('/_local/unload', b'{}')
        self.manager.stop_pending = None
        self.assertEqual(status, 503)
        self.assertIn('has not exited yet', body['error'])

    def test_select_answers_503_when_switching_must_stop_a_stuck_engine(self):
        self.manager.loaded_config = {'model_alias': 'known'}
        with patch.object(self.manager, 'cleanup', self.failing_cleanup(True)), \
                patch.object(self.manager, 'eligible', return_value=True):
            status, body = self.call('/_local/select', b'{"model": "other"}')
        self.manager.stop_pending = None
        self.manager.loaded_config = None
        self.assertEqual(status, 503)
        self.assertIn('has not exited yet', body['error'])
        self.assertEqual(json.loads(self.config.read_text())['selected_model'], 'known')

    def test_stop_for_control_passes_a_clean_stop(self):
        with patch.object(self.manager, 'cleanup', Mock()) as cleanup:
            self.manager.stop_for_control()
        cleanup.assert_called_once_with()


if __name__ == '__main__':
    unittest.main()
