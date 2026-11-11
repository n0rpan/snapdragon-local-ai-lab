import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import install_staged_release as isr  # noqa: E402


def h(b):
    return hashlib.sha256(b).hexdigest()


class InstallStagedReleaseTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        root = Path(self._tmp.name)
        self.stage = root / 'stage'
        self.live = root / 'live'
        self.stage.mkdir()
        self.live.mkdir()
        self.old = {'runtime': b'old-r', 'unit': b'old-u', 'opencode': b'old-o'}
        self.new = {'runtime': b'new-r', 'unit': b'new-u', 'opencode': b'new-o'}
        names = isr.DEFAULT_STAGED
        paths = {}
        for k in self.old:
            (self.live / k).write_bytes(self.old[k])
            (self.live / k).chmod(0o600)
            (self.stage / names[k]).write_bytes(self.new[k])
            paths[k] = str(self.live / k)
        self.manifest = {'paths': paths, 'preimage_sha256': {k: h(v) for k, v in self.old.items()}}
        self.write_manifest()

    def write_manifest(self):
        (self.stage / 'manifest.json').write_text(json.dumps(self.manifest))

    def run_cli(self, *args):
        with mock.patch('sys.stderr', io.StringIO()), mock.patch('sys.stdout', io.StringIO()):
            return isr.main([str(self.stage), *args])

    def live_bytes(self):
        return {k: (self.live / k).read_bytes() for k in self.old}

    def test_dry_run_default_writes_nothing(self):
        self.assertEqual(self.run_cli(), 0)
        self.assertEqual(self.live_bytes(), self.old)
        self.assertEqual(self.run_cli('--dry-run'), 0)
        self.assertEqual(self.live_bytes(), self.old)

    def test_apply_installs_and_keeps_mode_no_temp_left(self):
        self.assertEqual(self.run_cli('--apply'), 0)
        self.assertEqual(self.live_bytes(), self.new)
        self.assertEqual((self.live / 'runtime').stat().st_mode & 0o777, 0o600)
        self.assertEqual(sorted(p.name for p in self.live.iterdir()), ['opencode', 'runtime', 'unit'])

    def test_preimage_mismatch_refuses_without_touching_anything(self):
        (self.live / 'unit').write_bytes(b'drifted')
        self.assertEqual(self.run_cli('--apply'), 2)
        self.assertEqual((self.live / 'runtime').read_bytes(), b'old-r')
        self.assertEqual((self.live / 'unit').read_bytes(), b'drifted')

    def test_second_apply_refuses(self):
        self.assertEqual(self.run_cli('--apply'), 0)
        self.assertEqual(self.run_cli('--apply'), 2)

    def test_partial_recovery_preserves_a_later_unrelated_edit(self):
        target = self.live / 'runtime'
        target.write_bytes(b'newer-owner-data')
        item = {'key': 'runtime', 'live': target, 'preimage': h(self.old['runtime']),
                'new': h(self.new['runtime']), 'preimage_bytes': self.old['runtime']}
        restored, failed = isr.recover_partial([item])
        self.assertEqual(restored, [])
        self.assertEqual(len(failed), 1)
        self.assertEqual(target.read_bytes(), b'newer-owner-data')

    def test_missing_staged_file_or_hash_refuses(self):
        (self.stage / 'unit').unlink()
        self.assertEqual(self.run_cli('--apply'), 2)
        (self.stage / 'unit').write_bytes(b'new-u')
        del self.manifest['preimage_sha256']['opencode']
        self.write_manifest()
        self.assertEqual(self.run_cli('--apply'), 2)
        self.assertEqual(self.live_bytes(), self.old)

    def test_post_install_mismatch_fails(self):
        real = isr.sha256_file
        calls = {'n': 0}

        def flaky(path):
            if Path(path) == self.live / 'runtime' and calls.get('armed'):
                return 'bad'
            return real(path)

        orig = isr.install_atomic

        def install_then_arm(item):
            orig(item)
            calls['armed'] = True

        with mock.patch.object(isr, 'sha256_file', flaky), mock.patch.object(isr, 'install_atomic', install_then_arm):
            with self.assertRaises(isr.RefuseError):
                isr.run(self.stage, True, out=io.StringIO())

    def test_apply_and_dry_run_conflict(self):
        with mock.patch('sys.stderr', io.StringIO()):
            with self.assertRaises(SystemExit):
                isr.main([str(self.stage), '--apply', '--dry-run'])


if __name__ == '__main__':
    unittest.main()
