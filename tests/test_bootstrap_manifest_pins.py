"""Catch executable-input drift before a bootstrap reaches a host or the network."""
import hashlib
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class BootstrapPins(unittest.TestCase):
    def test_reviewed_manifest_inputs_match_the_checkout(self):
        manifests = list((ROOT / 'bootstrap').glob('*manifest.json'))
        manifests.append(ROOT / 'bootstrap/install-checkpoint.json')
        self.assertTrue(manifests)
        for manifest in manifests:
            data = json.loads(manifest.read_text())
            for group in ('files', 'source_dependencies'):
                for name, expected in data.get(group, {}).items():
                    with self.subTest(manifest=manifest.name, file=name):
                        actual = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                        self.assertEqual(actual, expected, 're-review and re-pin the changed input')
