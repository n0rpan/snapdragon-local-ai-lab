import copy
import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from snapdragon_ai_lab.runtime import oxe_runtime as oxe


def config():
    return {'backend_kind': 'oxe', 'ready': True, 'model_alias': 'ornith-oxe',
            'model_path': '/models/model.gguf', 'server_binary': '/engine/oxe-server',
            'server_sha256': 'a'*64, 'sha256': 'b'*64, 'expected_bytes': 100,
            'threads': 8, 'context_size': 16384, 'prefix_cache': {'enabled': False},
            'oxe_args': ['--model', '{model}', '--ctx', '{context}', '--threads', '{threads}']}


class OxeScaffoldTests(unittest.TestCase):
    def test_polling_disabled_never_opens_files(self):
        with patch.dict(os.environ, {}, clear=True), patch('os.open', side_effect=AssertionError('opened')):
            self.assertFalse(oxe.eligible(config()))
            with self.assertRaisesRegex(ValueError, 'not_enabled'):
                oxe.command(config(), 8093)

    def test_disabled_controller_cannot_select_or_start_the_engine(self):
        from snapdragon_ai_lab.runtime import controller
        manager = controller.Manager()
        manager.stop_event.set()
        try:
            with patch.dict(os.environ, {}, clear=True), patch.object(manager, 'config', return_value=config()), \
                    patch.object(controller.subprocess, 'Popen') as launch:
                with self.assertRaises(controller.Unavailable): manager.select('ornith-oxe')
                with self.assertRaises(controller.Unavailable): manager.begin(allow_start=True)
                launch.assert_not_called()
                self.assertIsNone(manager.lease)
        finally:
            manager.close()

    def test_enabled_metadata_and_explicit_cli(self):
        with patch.dict(os.environ, {oxe.ENABLE_ENV: '1'}):
            p = config(); before = copy.deepcopy(p)
            command = oxe.command(p, 8093)
            self.assertEqual(command[:5], ['/engine/oxe-server', '--host', '127.0.0.1', '--port', '8093'])
            self.assertIn('/models/model.gguf', command)
            self.assertIn('16384', command)
            self.assertEqual(p, before)
            p['ready'] = False
            self.assertFalse(oxe.eligible(p))

    def test_cli_cannot_override_controller_bind_or_use_unknown_placeholders(self):
        with patch.dict(os.environ, {oxe.ENABLE_ENV: '1'}):
            for args in [['--host=0.0.0.0'], ['{model.__class__}'], ['--port', '80']]:
                p = config(); p['oxe_args'] = args
                self.assertFalse(oxe.eligible(p))
            with self.assertRaisesRegex(ValueError, 'port'):
                oxe.command(config(), True)

    def test_external_llama_slot_restore_refused_until_supported(self):
        with patch.dict(os.environ, {oxe.ENABLE_ENV: '1'}):
            p = config(); p['prefix_cache']['enabled'] = True
            self.assertFalse(oxe.eligible(p))

    def test_engine_hash_checked_only_at_startup(self):
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp)/'server'; file.write_bytes(b'fixed synthetic binary')
            p = config(); p.update(server_binary=str(file), server_sha256=hashlib.sha256(file.read_bytes()).hexdigest())
            oxe.verify_engine(p)
            file.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'identity_changed'): oxe.verify_engine(p)

    def test_request_defaults_preserve_explicit_choices(self):
        p = {'temperature': .2, 'chat_template_kwargs': {'enable_thinking': True}}
        self.assertEqual(oxe.request_defaults(p, '/v1/chat/completions')['temperature'], .2)
        self.assertTrue(oxe.request_defaults(p, '/v1/chat/completions')['chat_template_kwargs']['enable_thinking'])
        self.assertFalse(oxe.request_defaults({}, '/v1/chat/completions')['chat_template_kwargs']['enable_thinking'])
        self.assertEqual(p, {'temperature': .2, 'chat_template_kwargs': {'enable_thinking': True}})

    def test_health_requires_ready_status(self):
        for raw in [b'bad', b'[]', b'{"status":"loading"}']:
            self.assertFalse(oxe.health_ready(raw))
        self.assertTrue(oxe.health_ready(b'{"status":"ok"}'))


if __name__ == '__main__': unittest.main()
