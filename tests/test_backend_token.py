"""B1 (v0.1.1), controller half: per-launch backend token for the Rust engine, sent by the proxy;
browser context headers never reach an engine. Model-free: no engine, no weights."""
import hashlib
import http.client
import json
import os
from pathlib import Path
import stat
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from snapdragon_ai_lab.runtime import classic_beta as beta, controller, oxe_runtime, profile_policy
from test_controller_security import SecurityBase, CHAT
from test_oxe_production import profile, stage_tool


class TokenFile(unittest.TestCase):
    def test_token_file_is_private_unique_and_removable(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = {oxe_runtime.RUNTIME_DIR_ENV: os.path.join(tmp, 'run')}
            token, path = oxe_runtime.new_api_key(8093, env)
            token2, path2 = oxe_runtime.new_api_key(8093, env)
            self.assertNotEqual(token, token2); self.assertNotEqual(path, path2)
            self.assertGreaterEqual(len(token), 40)
            self.assertEqual(path.read_text(), token)
            self.assertEqual(stat.S_IMODE(os.stat(path).st_mode), 0o600)
            self.assertEqual(stat.S_IMODE(os.stat(path.parent).st_mode), 0o700)
            self.assertEqual(oxe_runtime.backend_headers(token), {'Authorization': 'Bearer ' + token})
            self.assertEqual(oxe_runtime.backend_headers(None), {})
            oxe_runtime.remove_api_key(path); oxe_runtime.remove_api_key(path)
            self.assertFalse(path.exists())

    def test_xdg_runtime_dir_is_used_and_shared_dirs_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            token, path = oxe_runtime.new_api_key(1, {'XDG_RUNTIME_DIR': tmp})
            self.assertEqual(path.parent, Path(tmp) / 'local-ai')
            shared = Path(tmp) / 'shared'; shared.mkdir(); shared.chmod(0o755)
            with self.assertRaises(OSError):
                oxe_runtime.new_api_key(1, {oxe_runtime.RUNTIME_DIR_ENV: str(shared)})
            link = Path(tmp) / 'link'; link.symlink_to(Path(tmp) / 'local-ai')
            with self.assertRaises(OSError):
                oxe_runtime.new_api_key(1, {oxe_runtime.RUNTIME_DIR_ENV: str(link)})

    def test_without_runtime_dir_a_private_temporary_dir_is_used(self):
        token, path = oxe_runtime.new_api_key(1, {})
        try:
            self.assertEqual(stat.S_IMODE(os.stat(path.parent).st_mode), 0o700)
        finally:
            oxe_runtime.remove_api_key(path); os.rmdir(path.parent)


class LaunchPassesToken(unittest.TestCase):
    def test_oxe_launch_gets_a_fresh_key_file_that_the_profile_cannot_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            model = tmp / 'model.gguf'; model.write_bytes(b'm' * 100)
            engine = tmp / 'oxe-server'; engine.write_bytes(b'synthetic rust server')
            p = profile(); p['model_path'] = str(model)
            p['memlat_overrides']['oxe'] = stage_tool.oxe_override(
                str(engine), hashlib.sha256(engine.read_bytes()).hexdigest())
            p['memlat_overrides']['oxe'].setdefault('environment', {})[oxe_runtime.API_KEY_ENV] = '/profile/key'
            config = tmp / 'runtime.json'
            config.write_text(json.dumps({**profile(alias='qwen3.8-27b-local'), 'profiles': {beta.ALIAS: p}}))
            launched = {}

            def popen(args, **kwargs):
                key_file = Path(kwargs['env'][oxe_runtime.API_KEY_ENV])
                launched.update(args=args, key_file=key_file, existed=key_file.exists(),
                                mode=stat.S_IMODE(os.stat(key_file).st_mode), token=key_file.read_text())
                raise RuntimeError('synthetic stop before launch')
            manager = controller.Manager(); manager.stop_event.set()
            try:
                with patch.dict(os.environ, {'LOCAL_AI_ENABLE_OXE': '1',
                                             oxe_runtime.RUNTIME_DIR_ENV: str(tmp / 'run')}), \
                        patch.object(controller, 'CONFIG', config), patch.object(controller, 'LEASE', tmp / 'lease'), \
                        patch.object(controller, 'ENGINE_DIAGNOSTIC_LOG', tmp / 'engine-stderr.log'), \
                        patch.object(profile_policy, 'memlat_available', return_value=True), \
                        patch.object(controller, 'available_memory_bytes', return_value=None), \
                        patch.object(controller.memory_policy, 'MemoryGuard', return_value=Mock()), \
                        patch.object(controller.subprocess, 'Popen', side_effect=popen):
                    with self.assertRaisesRegex(RuntimeError, 'synthetic stop'):
                        manager.begin(allow_start=True, model_alias=beta.ALIAS)
            finally:
                manager.close()
            self.assertTrue(launched['existed'])
            self.assertEqual(launched['key_file'].parent, tmp / 'run')
            self.assertEqual(launched['mode'], 0o600)
            self.assertGreaterEqual(len(launched['token']), 40)
            self.assertNotIn('--api-key-file', launched['args'])  # env only: an older engine still starts
            self.assertFalse(launched['key_file'].exists(), 'a failed launch must not leave its token behind')


class ProxySendsToken(SecurityBase):
    def setUp(self):
        super().setUp()
        self.headers_seen = []
        seen = self.headers_seen
        handler = self.backend.RequestHandlerClass
        original = handler.do_POST

        def do_POST(inner):
            seen.append({k.lower(): v for k, v in inner.headers.items()})
            return original(inner)
        handler.do_POST = do_POST
        self.addCleanup(setattr, handler, 'do_POST', original)

    def chat(self, process, headers):
        self.manager.begin_queued = Mock(return_value=process)
        status, _ = self.send('POST', CHAT, host=f'127.0.0.1:{self.port}', body=self.inference_body(),
                              headers={'Content-Type': 'application/json', **headers})
        return status, self.headers_seen[-1]

    def test_token_replaces_client_credentials_and_browser_headers_are_dropped(self):
        process = SimpleNamespace(_local_ai_api_key='per-launch-secret', poll=lambda: None)
        status, sent = self.chat(process, {'Origin': f'http://127.0.0.1:{self.port}',
                                           'Authorization': 'Bearer client-value', 'x-api-key': 'client',
                                           'Sec-Fetch-Site': 'same-origin', 'Sec-Fetch-Mode': 'cors'})
        self.assertEqual(status, 200)
        self.assertEqual(sent['authorization'], 'Bearer per-launch-secret')
        self.assertNotIn('origin', sent); self.assertNotIn('x-api-key', sent)
        self.assertFalse([k for k in sent if k.startswith('sec-fetch-')])
        self.assertEqual(sent['host'], f'127.0.0.1:{self.backend.server_port}')

    def test_origin_is_dropped_for_a_backend_without_token(self):
        status, sent = self.chat(None, {'Origin': f'http://127.0.0.1:{self.port}'})
        self.assertEqual(status, 200)
        self.assertNotIn('origin', sent); self.assertNotIn('authorization', sent)


if __name__ == '__main__':
    unittest.main()
