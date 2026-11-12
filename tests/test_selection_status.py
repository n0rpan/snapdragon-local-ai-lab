"""Invalid selections must not break status or be persisted by a bad resolver."""
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from snapdragon_ai_lab.runtime import controller


class SelectionStatusTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.config = self.root / 'runtime.json'
        self.config.write_text(json.dumps({'model_alias': 'known', 'selected_model': 'invalid',
                                           'profiles': {'known': {'ready': False}}}))
        for name, value in (('CONFIG', self.config), ('HARNESS_STATE', self.root / 'primers'),
                            ('LEASE', self.root / 'inference.lock')):
            self.stack.enter_context(patch.object(controller, name, value))
        self.stack.enter_context(patch.dict(controller.os.environ, {'LOCAL_AI_IMAGE_NPU': '0'}))
        self.manager = controller.Manager()
        self.manager.stop_event.set()
        self.addCleanup(self.manager.close)

    def test_unloaded_invalid_selection_returns_json_and_does_not_repair_silently(self):
        before = self.config.read_bytes()
        row = self.manager.status()
        self.assertEqual(row['state'], 'unloaded')
        self.assertIsNone(row['selected_model'])
        self.assertEqual(row['selection_error'], 'invalid_or_unavailable_selection')
        self.assertFalse(row['validated'])
        json.dumps(row, allow_nan=False)
        self.assertEqual(self.config.read_bytes(), before)

    def test_loaded_metadata_survives_an_invalid_selected_profile(self):
        self.manager.loaded_config = {'model_alias': 'known', 'backend_kind': 'oxe', 'ready': False,
                                      'context_size': 32768, 'execution_policy_mode': 'memlat'}
        row = self.manager.status()
        self.assertEqual((row['model'], row['context_size'], row['backend_kind']), ('known', 32768, 'oxe'))
        self.assertIsNone(row['selected_model'])
        self.assertIsNotNone(row['selection_error'])

    def test_resolver_cannot_persist_an_unregistered_alias(self):
        before = self.config.read_bytes()
        with patch.object(self.manager, 'config', return_value={'model_alias': 'synthetic-invalid', 'ready': True}), \
                patch.object(self.manager, 'eligible', return_value=True):
            with self.assertRaisesRegex(controller.Unavailable, 'registered'):
                self.manager.select('synthetic-invalid')
        self.assertEqual(self.config.read_bytes(), before)

    def test_registered_selection_remains_available(self):
        with patch.object(self.manager, 'config', return_value={'model_alias': 'known', 'ready': True}):
            self.assertEqual(self.manager.select('known'), 'known')
        self.assertEqual(json.loads(self.config.read_text())['selected_model'], 'known')
        self.assertIsNone(self.manager.status()['selection_error'])


if __name__ == '__main__':
    unittest.main()
