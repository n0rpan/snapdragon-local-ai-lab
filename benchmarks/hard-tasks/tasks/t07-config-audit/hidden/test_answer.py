"""Hidden checks: ANSWER.json lists the required keys; the svc package is untouched."""
import hashlib
import json
import os
import unittest
from pathlib import Path

PROJECT = Path(os.environ["PROJECT"])

REQUIRED = [
    "cache.ttl",
    "db.name",
    "db.primary.host",
    "db.replica.host",
    "db.user",
    "http.api_key",
    "http.base_url",
    "service.name",
    "service.region",
    "worker.queue",
]

ORIGINAL_SHA256 = {
        "cache.py": "320e1d15a40328e93a7be94954bd4dfa10787a915703f451d96c0ba7075141e0",
        "config.py": "912b7d88d7ac58f77a2d0393c4eade2caac30ed41cd380afbf21016519b56d93",
        "db.py": "8c1f7c1eeba1a0acbf16d3f2e9c08e1ad9b7146991525519570e7fe4158bf199",
        "http.py": "823547845dd60adac7619f0d5cfabc50e6c14eeb5ca5c9dd9496f9b63ef53f0b",
        "__init__.py": "187e95dcea510d28d777e93ece7eef5d5047deb5695d1857ad6674dcc1907120",
        "legacy_sync.py": "8706629b79f9f0f5dbaba843820b6485b82cd7ea1dca8e6ae9c27c1c56343d20",
        "main.py": "254d5c4668321f87f0af0c891ae00bc5686a783b4c7a14f933e5456f7088f3de",
        "util.py": "2bf79714d414e5b6e3ef813c0c070e424444eca22d91e667b70d85a634743d0b",
        "worker.py": "2380c1fee03f49e5a8ce3a5f0cee724cd7717a8c1fc317b999b47032086cc8ef",
}


class SvcUnmodifiedTest(unittest.TestCase):
    def test_svc_files_byte_identical(self):
        for name, digest in sorted(ORIGINAL_SHA256.items()):
            path = PROJECT / "svc" / name
            self.assertTrue(path.is_file(), f"missing svc/{name}")
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest, f"svc/{name} was modified")


class AnswerTest(unittest.TestCase):
    def load(self):
        path = PROJECT / "ANSWER.json"
        self.assertTrue(path.is_file(), "ANSWER.json missing in project root")
        return json.loads(path.read_text(encoding="utf-8"))

    def test_required_is_list_of_strings(self):
        data = self.load()
        self.assertIsInstance(data, dict)
        self.assertIn("required", data)
        self.assertIsInstance(data["required"], list)
        for item in data["required"]:
            self.assertIsInstance(item, str)

    def test_required_matches_ground_truth(self):
        data = self.load()
        self.assertEqual(set(data["required"]), set(REQUIRED))


if __name__ == "__main__":
    unittest.main()
