"""Model-free tests of the guarded stage rollback: temp dirs and a fake systemctl only."""
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import install_staged_release as isr  # noqa: E402
import stage_rollback_guard as guard  # noqa: E402

KEYS = ('runtime', 'unit', 'opencode')


def h(data):
    return hashlib.sha256(data).hexdigest()


class RollbackGuardTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.live = self.root / 'live'
        self.staged = self.root / 'staged'
        self.live.mkdir(); self.staged.mkdir()
        self.calls = self.root / 'systemctl.log'
        self.systemctl = self.root / 'fake-systemctl'
        self.systemctl.write_text('#!/bin/sh\necho "$@" >> %s\n' % self.calls)
        self.systemctl.chmod(0o700)
        self.v0 = {k: ('v0-' + k).encode() for k in KEYS}
        for k in KEYS:
            (self.live / k).write_bytes(self.v0[k]); (self.live / k).chmod(0o600)
        self.paths = {k: str(self.live / k) for k in KEYS}

    def make_stage(self, name, old, new, guarded=True):
        stage = self.staged / name
        (stage / 'rollback').mkdir(parents=True)
        for k in old:
            (stage / guard.DEFAULT_STAGED[k]).write_bytes(new[k])
            (stage / 'rollback' / (k + '.original')).write_bytes(old[k])
        paths = {k: self.paths[k] for k in old}
        if guarded:
            script = guard.render_rollback(paths, new, old, systemctl=str(self.systemctl))
            manifest = guard.manifest_binding(paths, new, old)
        else:
            script = b'#!/usr/bin/python3\nraise SystemExit("unguarded legacy rollback must never run in tests")\n'
            manifest = {'paths': paths, 'preimage_sha256': {k: h(v) for k, v in old.items()},
                        'staged_files': {'unit': 'local-ai-chat.service'}}
        (stage / 'rollback' / 'rollback.py').write_bytes(script)
        (stage / 'rollback' / 'rollback.py').chmod(0o700)
        (stage / 'manifest.json').write_text(json.dumps(manifest))
        return stage

    def install(self, stage, **kwargs):
        out = io.StringIO()
        self.assertEqual(isr.run(stage, True, out=out, **kwargs), 0)
        return out.getvalue()

    def rollback(self, stage, *args):
        return subprocess.run([sys.executable, str(stage / 'rollback' / 'rollback.py'), *args],
                              capture_output=True, text=True, timeout=30)

    def live_bytes(self):
        return {k: (self.live / k).read_bytes() for k in KEYS}

    def service_calls(self):
        return self.calls.read_text().splitlines() if self.calls.exists() else []

    def two_installs(self):
        self.v1 = {k: ('v1-' + k).encode() for k in KEYS}
        self.v2 = {**self.v1, 'runtime': b'v2-runtime', 'unit': b'v2-unit'}
        a = self.make_stage('stage-a', self.v0, self.v1)
        self.install(a)
        b = self.make_stage('stage-b', {k: self.v1[k] for k in ('runtime', 'unit')},
                            {k: self.v2[k] for k in ('runtime', 'unit')})
        return a, b

    def test_two_installs_stale_first_rollback_refused_then_newest_first_works(self):
        a, b = self.two_installs()
        text = self.install(b)
        self.assertIn('superseded stage: stage-a', text)
        self.assertEqual(self.live_bytes(), self.v2)

        stale = self.rollback(a)
        self.assertEqual(stale.returncode, guard.EXIT_STALE, stale.stderr)
        self.assertIn('stale rollback of stage-a', stale.stderr)
        self.assertIn('newer install stage-b', stale.stderr)
        self.assertEqual(self.service_calls(), [])  # refused BEFORE the service is stopped
        self.assertEqual(self.live_bytes(), self.v2)
        self.assertEqual(self.rollback(a, '--check').returncode, guard.EXIT_STALE)

        check = self.rollback(b, '--check')
        self.assertEqual(check.returncode, 0, check.stderr)
        self.assertEqual((self.service_calls(), self.live_bytes()), ([], self.v2))

        newest = self.rollback(b)
        self.assertEqual(newest.returncode, 0, newest.stderr)
        self.assertEqual(self.live_bytes(), self.v1)
        self.assertEqual(self.service_calls(), ['--user stop local-ai-chat.service', '--user daemon-reload',
                                                '--user start local-ai-chat.service'])
        self.assertEqual(len(list((b / 'rollback/receipts').glob('rollback-*.json'))), 1)
        self.assertEqual(os.stat(self.live / 'runtime').st_mode & 0o777, 0o600)

        older = self.rollback(a)  # stage-a is the current install again: one command still works
        self.assertEqual(older.returncode, 0, older.stderr)
        self.assertEqual(self.live_bytes(), self.v0)
        again = self.rollback(a)  # already at the preimage: no-op, service untouched
        self.assertEqual(again.returncode, 0)
        self.assertEqual(len(self.service_calls()), 6)

    def test_retire_superseded_keeps_receipts_and_preimages_and_refuses_rollback(self):
        a, b = self.two_installs()
        with self.assertRaisesRegex(guard.GuardError, 'CURRENT install'):
            guard.retire(a)
        before = sorted(p.name for p in (a / 'rollback').iterdir()) + sorted(p.name for p in (a / 'receipts').iterdir())
        self.assertIn('retired superseded stage stage-a', self.install(b, retire_superseded=True))
        after = sorted(p.name for p in (a / 'rollback').iterdir()) + sorted(p.name for p in (a / 'receipts').iterdir())
        self.assertEqual(before, after)
        for k in KEYS:
            self.assertEqual((a / 'rollback' / (k + '.original')).read_bytes(), self.v0[k])
        self.assertTrue(json.loads((a / 'manifest.json').read_text())['retired'])
        self.assertEqual(json.loads((a / 'retired.json').read_text())['superseded_by'], 'stage-b')

        refused = self.rollback(a)
        self.assertEqual(refused.returncode, guard.EXIT_RETIRED, refused.stderr)
        self.assertIn('superseded by stage-b', refused.stderr)
        self.assertEqual((self.service_calls(), self.live_bytes()), ([], self.v2))
        with self.assertRaisesRegex(isr.RefuseError, 'retired'):
            isr.load_plan(a)
        with self.assertRaisesRegex(guard.GuardError, 'not the current install'):
            guard.reinstate(a)

        self.assertEqual(self.rollback(b).returncode, 0)  # current stage: one command
        self.assertEqual(self.live_bytes(), self.v1)
        self.assertEqual(self.rollback(a).returncode, guard.EXIT_RETIRED)  # still retired until reinstated
        guard.reinstate(a)
        self.assertEqual(self.rollback(a).returncode, 0)
        self.assertEqual(self.live_bytes(), self.v0)

    def test_retire_deduplicates_symlink_and_canonical_stage_paths(self):
        a, b = self.two_installs()
        (self.staged / 'latest-a').symlink_to(a, target_is_directory=True)
        (self.staged / 'latest-current').symlink_to(b, target_is_directory=True)
        found = guard.find_superseded(b, {k: h(self.v1[k]) for k in ('runtime', 'unit')})
        self.assertEqual(found, [a.resolve()])
        text = self.install(b, retire_superseded=True)
        self.assertEqual(text.count('retired superseded stage stage-a'), 1)
        self.assertTrue(guard.is_retired(a))
        self.assertFalse(guard.is_retired(b))
        self.assertEqual(self.live_bytes(), self.v2)
        self.assertEqual(self.rollback(b).returncode, 0)
        self.assertEqual(self.live_bytes(), self.v1)

    def test_tampered_or_missing_preimage_is_refused_before_service_stop(self):
        a, _ = self.two_installs()
        original = a / 'rollback/unit.original'
        original.write_bytes(b'tampered')
        result = self.rollback(a)
        self.assertEqual(result.returncode, guard.EXIT_TAMPERED, result.stderr)
        self.assertIn('unit.original', result.stderr)
        self.assertEqual((self.service_calls(), self.live_bytes()), ([], self.v1))
        original.unlink()
        self.assertEqual(self.rollback(a).returncode, guard.EXIT_TAMPERED)
        self.assertEqual(self.live_bytes(), self.v1)

    def test_manifest_binding_mismatch_is_refused(self):
        a, _ = self.two_installs()
        manifest = json.loads((a / 'manifest.json').read_text())
        manifest['preimage_sha256']['runtime'] = h(b'other')
        (a / 'manifest.json').write_text(json.dumps(manifest))
        result = self.rollback(a)
        self.assertEqual(result.returncode, guard.EXIT_TAMPERED, result.stderr)
        self.assertIn('manifest preimage_sha256.runtime', result.stderr)
        self.assertEqual(self.service_calls(), [])

    def test_installer_refuses_stage_whose_preimage_or_payload_was_modified(self):
        v1 = {k: b'v1-' + k.encode() for k in KEYS}
        a = self.make_stage('stage-a', self.v0, v1)
        (a / 'rollback/opencode.original').write_bytes(b'tampered')
        with self.assertRaisesRegex(isr.RefuseError, 'opencode.original'):
            isr.run(a, True, out=io.StringIO())
        (a / 'rollback/opencode.original').write_bytes(self.v0['opencode'])
        (a / 'runtime.json').write_bytes(b'swapped payload')
        with self.assertRaisesRegex(isr.RefuseError, 'staged file changed'):
            isr.run(a, True, out=io.StringIO())
        self.assertEqual(self.live_bytes(), self.v0)

    def test_partial_install_failure_restores_preimages_automatically(self):
        v1 = {k: b'v1-' + k.encode() for k in KEYS}
        a = self.make_stage('stage-a', self.v0, v1)
        real = isr.install_atomic

        def flaky(item):
            if item['key'] == 'unit' and 'src_bytes' not in item:
                raise OSError('disk full (injected)')
            real(item)
        out = io.StringIO()
        with mock.patch.object(isr, 'install_atomic', flaky), self.assertRaisesRegex(isr.RefuseError, 'disk full'):
            isr.run(a, True, out=out)
        self.assertIn('every file is back at its preimage', out.getvalue())
        self.assertEqual(self.live_bytes(), self.v0)
        receipt = json.loads(next((a / 'receipts').glob('install-*.json')).read_text())
        self.assertEqual((receipt['action'], receipt['replaced_before_failure'], receipt['restore_failed']),
                         ('install-failed', ['runtime'], []))
        self.assertEqual(self.rollback(a).returncode, 0)  # nothing to do, service untouched
        self.assertEqual(self.service_calls(), [])
        self.install(a)  # the stage is still installable after the recovered failure
        self.assertEqual(self.live_bytes(), v1)

    def test_partial_install_left_behind_is_recovered_by_stage_rollback(self):
        v1 = {k: b'v1-' + k.encode() for k in KEYS}
        a = self.make_stage('stage-a', self.v0, v1)
        real = isr.install_atomic

        def dies(item):  # second file fails AND the automatic restore fails too (process killed)
            if item['key'] == 'unit' or 'src_bytes' in item:
                raise OSError('killed (injected)')
            real(item)
        out = io.StringIO()
        with mock.patch.object(isr, 'install_atomic', dies), self.assertRaises(isr.RefuseError):
            isr.run(a, True, out=out)
        self.assertIn('run the stage rollback', out.getvalue())
        self.assertEqual(self.live_bytes(), {**self.v0, 'runtime': v1['runtime']})  # mixed state
        check = self.rollback(a, '--check')
        self.assertEqual(check.returncode, 0, check.stderr)
        self.assertIn('would restore runtime', check.stdout)
        result = self.rollback(a)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.live_bytes(), self.v0)
        self.assertEqual(json.loads(next((a / 'rollback/receipts').glob('*.json')).read_text())['restored'], ['runtime'])

    def test_hand_edit_of_live_file_is_refused_as_stale_without_owner(self):
        a, _ = self.two_installs()
        (self.live / 'opencode').write_bytes(b'edited by hand')
        result = self.rollback(a)
        self.assertEqual(result.returncode, guard.EXIT_STALE)
        self.assertIn('no sibling stage matches', result.stderr)
        self.assertEqual(self.service_calls(), [])

    def test_bind_guards_a_legacy_stage_and_names_a_legacy_newer_install(self):
        v1 = {k: b'v1-' + k.encode() for k in KEYS}
        v2 = {'runtime': b'v2-runtime'}
        a = self.make_stage('legacy-a', self.v0, v1, guarded=False)
        self.install(a)
        b = self.make_stage('legacy-b', {'runtime': v1['runtime']}, v2, guarded=False)
        self.install(b)
        with self.assertRaisesRegex(guard.GuardError, 'not guarded'):
            guard.read_binding(a)
        binding = guard.bind(a, self.paths, systemctl=str(self.systemctl))
        self.assertEqual(binding['installed_sha256']['runtime'], h(v1['runtime']))
        self.assertIn('unguarded legacy', (a / 'rollback/rollback.py.unguarded').read_text())
        self.assertFalse(os.access(a / 'rollback/rollback.py.unguarded', os.X_OK))
        self.assertEqual(json.loads((a / 'manifest.json').read_text())['installed_sha256'], binding['installed_sha256'])
        result = self.rollback(a)
        self.assertEqual(result.returncode, guard.EXIT_STALE)
        self.assertIn('newer install legacy-b', result.stderr)  # found by payload hash, legacy-b is unguarded
        self.assertEqual(guard.status(a)['files']['runtime']['live'], 'other')
        with self.assertRaisesRegex(guard.GuardError, 'without a --path'):
            guard.bind(b, {})
        (b / 'rollback/runtime.original').write_bytes(b'tampered')
        with self.assertRaisesRegex(guard.GuardError, 'does not match manifest'):
            guard.bind(b, {'runtime': self.paths['runtime']})

    def test_cli_retire_and_status(self):
        a, b = self.two_installs()
        self.install(b)
        tool = str(Path(guard.__file__))
        done = subprocess.run([sys.executable, tool, 'retire', str(a), '--superseded-by', 'stage-b'],
                              capture_output=True, text=True, timeout=30)
        self.assertEqual(done.returncode, 0, done.stderr)
        twice = subprocess.run([sys.executable, tool, 'retire', str(a)], capture_output=True, text=True, timeout=30)
        self.assertEqual(twice.returncode, 2)
        state = json.loads(subprocess.run([sys.executable, tool, 'status', str(b)], capture_output=True,
                                          text=True, timeout=30).stdout)
        self.assertEqual((state['retired'], state['files']['runtime']['live']), (False, 'installed'))
        self.assertEqual(subprocess.run([sys.executable, tool, 'check', str(b)], capture_output=True,
                                        timeout=30).returncode, 0)


    def test_pair_stops_and_restarts_together_after_identity_checks(self):
        new = {k: b'paired-' + k.encode() for k in KEYS}
        stage = self.make_stage('paired', self.v0, new)
        (stage / 'rollback/rollback.py').write_bytes(guard.render_rollback(
            self.paths, new, self.v0, systemctl=str(self.systemctl),
            services=['local-ai-chat.service', 'prompt-studio-app.service']))
        self.install(stage)
        self.assertEqual(self.rollback(stage, '--check').returncode, 0)
        self.assertEqual(self.service_calls(), [])
        done = self.rollback(stage)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(self.live_bytes(), self.v0)
        self.assertEqual(self.service_calls(), [
            '--user stop local-ai-chat.service prompt-studio-app.service',
            '--user daemon-reload', '--user start local-ai-chat.service prompt-studio-app.service'])


if __name__ == '__main__':
    unittest.main()
