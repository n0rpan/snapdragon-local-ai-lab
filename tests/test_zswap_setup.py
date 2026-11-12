"""S2: the zswap setup script fails when zswap does not read back as enabled."""
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / 'packaging/zswap/local-ai-zswap-setup'


class ZswapSetupTests(unittest.TestCase):
    def run_script(self, params):
        return subprocess.run(['/bin/sh', str(SCRIPT)], env={**os.environ, 'LOCAL_AI_ZSWAP_PARAMS': str(params)},
                              capture_output=True, text=True, timeout=30)

    def params(self, root):
        params = Path(root) / 'parameters'
        params.mkdir()
        for name, value in (('compressor', 'lzo'), ('max_pool_percent', '20'), ('shrinker_enabled', 'N'), ('enabled', 'N')):
            (params / name).write_text(value)
        return params

    def test_success_when_every_value_reads_back(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as tmp:
            params = self.params(tmp)
            done = self.run_script(params)
            self.assertEqual(done.returncode, 0, done.stderr)
            self.assertEqual((params / 'enabled').read_text(), 'Y')
            self.assertEqual((params / 'compressor').read_text(), 'zstd')

    def test_failure_when_enabled_does_not_stick(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as tmp:
            params = self.params(tmp)
            (params / 'enabled').unlink()
            (params / 'enabled').symlink_to('/dev/null')  # accepts the write, reads back empty
            done = self.run_script(params)
            self.assertNotEqual(done.returncode, 0)
            self.assertIn("enabled reads back", done.stderr)

    def test_failure_without_zswap(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as tmp:
            done = self.run_script(Path(tmp) / 'missing')
            self.assertNotEqual(done.returncode, 0)


if __name__ == '__main__':
    unittest.main()
