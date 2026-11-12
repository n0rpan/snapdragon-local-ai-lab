"""Whole-pair staging/install/rollback in a fixture; no model, device or service."""
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import stage_studio_npu as tool
import install_staged_release as installer


class StageStudioNpu(unittest.TestCase):
    def test_effective_dropin_preserves_exec_reset_and_other_settings(self):
        raw = b'[Service]\nExecStart=\nExecStart=/usr/bin/python3 /old/studio\nEnvironment=OXI_BIN=/qualified/oxi\n'
        updated = tool.rewrite_unit(raw, '/new/release', 'studio').decode()
        self.assertIn('ExecStart=\nExecStart=/usr/bin/python3 "/new/release/studio"', updated)
        self.assertIn('Environment=OXI_BIN=/qualified/oxi', updated)
        with self.assertRaises(ValueError):
            tool.rewrite_unit(raw + b'ExecStart=/other\n', '/new/release', 'studio')

    def test_pair_preserves_settings_and_guarded_rollback_restores_both_services(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {k: root / k for k in ('controller_unit', 'studio_unit', 'image_settings')}
            units = b'[Service]\nWorkingDirectory=/old\nExecStart=/usr/bin/python3 /old/ENTRY\nMemoryMax=28G\nMemorySwapMax=0\n'
            paths['controller_unit'].write_bytes(units.replace(b'ENTRY', b'controller.py'))
            paths['studio_unit'].write_bytes(units.replace(b'ENTRY', b'studio'))
            settings = {'oxi': {'prompt_cache': True, 'vae_conv': 'int8'}, 'unrelated': {'keep': 4}}
            paths['image_settings'].write_text(json.dumps(settings))
            originals = {k: p.read_bytes() for k, p in paths.items()}
            profile = root / 'profile.json'; profile.write_bytes(b'{"fixture":true}'); profile.chmod(0o600)
            stage = root / 'stage'
            with patch.object(tool.image_npu_worker, 'load_profile') as verify:
                manifest = tool.stage(stage, paths=paths,
                    expected={k: tool.guard.sha(v) for k, v in originals.items()},
                    releases_root=root / 'releases', profile_path=profile,
                    profile_sha=tool.guard.sha(profile.read_bytes()))
                verify.assert_called_once_with(profile)
            self.assertEqual({k: p.read_bytes() for k, p in paths.items()}, originals)
            updated = json.loads((stage / 'runtime-settings.json').read_text())
            self.assertEqual(updated['unrelated'], settings['unrelated'])
            self.assertEqual(updated['oxi']['vae_conv'], 'int8')
            self.assertTrue(updated['oxi']['prompt_cache'])
            self.assertTrue(updated['oxi']['npu'])
            self.assertIn('LOCAL_AI_IMAGE_NPU=1', (stage / 'local-ai-chat.service').read_text())
            self.assertIn('/studio"', (stage / 'prompt-studio-app.service').read_text())
            self.assertFalse(manifest['runtime_json_changed'])
            self.assertTrue((stage / 'runtime-source/image_npu_worker.py').is_file())
            self.assertEqual(installer.run(stage, False, out=io.StringIO()), 0)
            self.assertEqual(installer.run(stage, True, out=io.StringIO()), 0)
            calls = root / 'calls'
            fake = root / 'systemctl'
            fake.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "' + str(calls) + '"\n')
            fake.chmod(0o700)
            names = manifest['staged_files']
            payload = {k: (stage / names[k]).read_bytes() for k in paths}
            rollback = stage / 'rollback/rollback.py'
            rollback.write_bytes(tool.guard.render_rollback(paths, payload, originals, staged_files=names,
                systemctl=str(fake), services=['local-ai-chat.service', 'prompt-studio-app.service']))
            paths['studio_unit'].write_bytes(b'newer-install')
            refused = subprocess.run([sys.executable, str(rollback), '--check'], capture_output=True)
            self.assertEqual(refused.returncode, 3)
            self.assertFalse(calls.exists())
            paths['studio_unit'].write_bytes(payload['studio_unit'])
            done = subprocess.run([sys.executable, str(rollback)], capture_output=True, text=True)
            self.assertEqual(done.returncode, 0, done.stderr)
            self.assertEqual({k: p.read_bytes() for k, p in paths.items()}, originals)
            self.assertIn('stop local-ai-chat.service prompt-studio-app.service', calls.read_text())


if __name__ == '__main__':
    unittest.main()
