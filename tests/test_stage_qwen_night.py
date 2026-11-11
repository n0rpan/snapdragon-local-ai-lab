import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('qwen_stage', Path(__file__).resolve().parents[1] / 'tools/stage_qwen_night.py')
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


def fixture():
    return {'selected_model': 'ornith-35b-a3b-local', 'model_alias': 'qwen-local',
            'profiles': {'ornith-35b-a3b-local': {'untouched': True},
                         module.ALIAS: {'model_path': '/models/qwen.gguf', 'sha256': module.MODEL_SHA,
                                        'context_size': 16384, 'threads': 4,
                                        'memlat_overrides': {'oxe': {
                                            'server_binary': '/old/oxe-server', 'server_sha256': 'b' * 64,
                                            'oxe_args': ['--model', '{model}']}}}}}


class QwenStageTests(unittest.TestCase):
    def test_only_qwen_changes_and_stock_launch_stays_unchanged(self):
        before = fixture(); untouched = copy.deepcopy(before)
        result = module.patch_runtime(before, '/new/oxe-server', next(iter(module.SERVERS)), '/private/sessions')
        self.assertEqual(before, untouched)
        self.assertEqual(result['selected_model'], before['selected_model'])
        self.assertEqual(result['profiles']['ornith-35b-a3b-local'], before['profiles']['ornith-35b-a3b-local'])
        old = before['profiles'][module.ALIAS]; new = result['profiles'][module.ALIAS]
        self.assertEqual({k: v for k, v in new.items() if k not in ('memlat_overrides', 'default_max_tokens')},
                         {k: v for k, v in old.items() if k != 'memlat_overrides'})
        effective = module.profile_policy.resolve({**new, 'model_alias': module.ALIAS}, available=True, oxe_enabled=True)
        args = effective['oxe_args']; i = args.index('--spec-type')
        self.assertEqual(args[i + 1], 'none'); self.assertNotIn('--mtp-head', args)
        self.assertEqual(effective['context_size'], 32768)
        self.assertEqual(effective['environment']['LOCAL_AI_ENGINE_LEDGER'], '1')
        self.assertFalse(effective['validation']['qualified'])

    def test_wrong_model_and_unknown_engine_refused(self):
        with self.assertRaisesRegex(ValueError, 'unknown_night_binary'):
            module.patch_runtime(fixture(), '/new/bin', 'c' * 64, '/sessions')
        config = fixture(); config['profiles'][module.ALIAS]['sha256'] = 'd' * 64
        with self.assertRaisesRegex(ValueError, 'model_identity'):
            module.patch_runtime(config, '/new/bin', next(iter(module.SERVERS)), '/sessions')

    def test_provider_only_qwen_limits_change(self):
        before = {'providers': {'local-qwen': {'apiKey': 'dummy', 'models': {
            module.ALIAS: {'limit': {'context': 16384, 'input': 12288, 'output': 16384}},
            'qwen-local': {'limit': {'context': 32768}}}}}}
        result = module.patch_opencode(before)
        result['providers']['local-qwen']['models'][module.ALIAS] = before['providers']['local-qwen']['models'][module.ALIAS]
        self.assertEqual(result, before)

    def test_private_sessions_cannot_be_staged_on_public_weight_partition(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, 'private_sessions_must_stay_on_encrypted_storage'):
                module.stage(Path(tmp) / 'stage', runtime='/unused', unit='/unused', opencode='/unused',
                             server='/unused', preimages={}, releases_root='/private/releases',
                             sessions='/var/lib/local-ai-models/qwen/sessions')
            self.assertFalse((Path(tmp) / 'stage').exists())

    def test_stage_preserves_live_preimages_and_writes_private_rollback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); runtime = root / 'runtime'; unit = root / 'unit'; oc = root / 'opencode'; engine = root / 'engine'
            runtime.write_text(json.dumps(fixture())); unit.write_text('WorkingDirectory=/old\nExecStart=/usr/bin/python3 /old/controller.py\nMemoryMax=28G\n')
            oc.write_text(json.dumps({'providers': {'local-qwen': {'models': {module.ALIAS: {'limit': {}}}}}}))
            engine.write_bytes(b'fixture executable'); digest = hashlib.sha256(engine.read_bytes()).hexdigest()
            originals = {k: p.read_bytes() for k, p in {'runtime': runtime, 'unit': unit, 'opencode': oc}.items()}
            preimages = {k: hashlib.sha256(v).hexdigest() for k, v in originals.items()}
            with patch.dict(module.SERVERS, {digest: 'test'}):
                result = module.stage(root / 'stage', runtime=runtime, unit=unit, opencode=oc, server=engine,
                                      preimages=preimages, releases_root=root / 'releases', sessions=root / 'sessions')
            self.assertFalse(result['qualified']); self.assertFalse(result['installed']); self.assertFalse(result['models_started'])
            for key, source in {'runtime': runtime, 'unit': unit, 'opencode': oc}.items():
                self.assertEqual(source.read_bytes(), originals[key])
                self.assertEqual((root / 'stage/rollback' / (key + '.original')).read_bytes(), originals[key])
            self.assertEqual((root / 'stage/runtime.json').stat().st_mode & 0o777, 0o600)
            self.assertEqual((root / 'stage').stat().st_mode & 0o777, 0o700)
            compile((root / 'stage/rollback/rollback.py').read_text(), 'rollback.py', 'exec')
            self.assertEqual((root / 'stage/engine/oxe-server').read_bytes(), engine.read_bytes())


if __name__ == '__main__':
    unittest.main()
