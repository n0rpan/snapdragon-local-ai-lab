"""M3: Flash status is derived from one source (the installed profile's validation + receipts),
and the storage row reaches 'ready' instead of 'weights ready, tests pending' forever."""
import hashlib
import importlib.machinery
import importlib.util
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from snapdragon_ai_lab.runtime import model_catalog

SCRIPT = Path(__file__).resolve().parents[1] / 'src/snapdragon_ai_lab/runtime/local-ai'
loader = importlib.machinery.SourceFileLoader('flash_qual_local_ai', str(SCRIPT))
spec = importlib.util.spec_from_loader(loader.name, loader)
cli = importlib.util.module_from_spec(spec)
loader.exec_module(cli)

ENGINE = 'f' * 64


class QualificationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir='/tmp')
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def receipt(self, name, body=b'{"pass": true}'):
        path = self.root / name
        path.write_bytes(body)
        return str(path), hashlib.sha256(body).hexdigest()

    def profile(self, **validation):
        report, digest = self.receipt('receipt.json')
        proof, proof_digest = self.receipt('guard.json', b'{"guard": 2}')
        base = {'qualified': True, 'server_sha256': ENGINE, 'report': report, 'receipt_sha256': digest,
                'proof_sources': {'guard2': {'path': proof, 'sha256': proof_digest}}}
        base.update(validation)
        return {'backend_kind': 'oxe', 'server_sha256': ENGINE, 'validation': base}

    def test_qualified_needs_matching_engine_hash_and_every_receipt(self):
        self.assertEqual(model_catalog.qualification(self.profile()), 'qualified')
        self.assertEqual(model_catalog.qualification(self.profile(server_sha256='e' * 64)), 'stale')
        self.assertEqual(model_catalog.qualification(self.profile(qualified=False)), 'unqualified')
        self.assertEqual(model_catalog.qualification({'backend_kind': 'oxe'}), 'unqualified')
        no_receipts = self.profile()
        for key in ('report', 'receipt_sha256', 'proof_sources'):
            no_receipts['validation'].pop(key)
        self.assertEqual(model_catalog.qualification(no_receipts), 'receipt-missing')

    def test_missing_or_changed_receipt_is_not_qualified(self):
        profile = self.profile()
        Path(profile['validation']['proof_sources']['guard2']['path']).unlink()
        self.assertEqual(model_catalog.qualification(profile), 'receipt-missing')
        profile = self.profile()
        Path(profile['validation']['report']).write_bytes(b'{"pass": false}')
        self.assertEqual(model_catalog.qualification(profile), 'receipt-missing')

    def test_phase_after_verified_weights(self):
        self.assertEqual(model_catalog.flash_phase(True, 'qualified'), 'ready')
        self.assertEqual(model_catalog.flash_phase(True, 'stale'), 'weights-ready-testing')
        self.assertIsNone(model_catalog.flash_phase(False, 'qualified'))


class CliPhaseTests(unittest.TestCase):
    def test_verified_weights_are_ready_only_when_the_controller_reports_qualified(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as tmp:
            root = Path(tmp)
            state, models = root / 'state', root / 'models'
            state.mkdir(); models.mkdir()
            body = b'x' * 1024
            shard = models / 'flash-00001.gguf'
            shard.write_bytes(body)
            sha = hashlib.sha256(body).hexdigest()
            (state / 'manifest.json').write_text(json.dumps({
                'repository': 'unsloth/Qwen3.8-Flash-Next-GGUF', 'revision': 'a' * 40, 'total_bytes': len(body),
                'models': [{'id': 'flash-next-1', 'sha256': sha, 'bytes': len(body), 'path': str(shard)}]}))
            (state / 'flash-next-1-verified.json').write_text(json.dumps({
                'id': 'flash-next-1', 'sha256': sha, 'bytes': len(body), 'verified_at': time.time()}))
            with patch.object(cli, 'FLASH_STATE', state), patch.object(cli, 'FLASH_MODELS', models), \
                    patch.object(cli, 'flash_worker_active', return_value=False):
                self.assertEqual(cli.flash_download_status('qualified')['flash_download_state'], 'ready')
                for other in (None, 'stale', 'unqualified', 'receipt-missing'):
                    self.assertEqual(cli.flash_download_status(other)['flash_download_state'], 'weights-ready-testing')


if __name__ == '__main__':
    unittest.main()
