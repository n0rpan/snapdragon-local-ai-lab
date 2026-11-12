"""Model-free tests of the v0.1.2 FINAL stage generator: temp paths and a fake systemctl only."""
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import install_staged_release as isr  # noqa: E402
import stage_rollback_guard as guard  # noqa: E402
import stage_v012_final as gen  # noqa: E402

MOE = {'ornith-35b-a3b-local': ('a' * 64, 1073741824, 32768), 'qwen3.6-35b-a3b-local': ('b' * 64, 536870912, 16384)}


def h(data):
    return hashlib.sha256(data).hexdigest()


def tree(path):
    return {p.relative_to(path).as_posix(): (h(p.read_bytes()), p.stat().st_mode & 0o7777)
            for p in sorted(Path(path).rglob('*')) if p.is_file()}


def have_git_head():
    try:
        subprocess.run(['git', '-C', str(ROOT), 'rev-parse', '--verify', 'HEAD^{commit}'], check=True, capture_output=True)
        return True
    except (OSError, subprocess.CalledProcessError):
        return False


@unittest.skipUnless(have_git_head(), 'needs the lab git checkout (git archive of HEAD)')
class FinalStageTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.live = self.root / 'home'
        (self.live / 'bin').mkdir(parents=True)
        self.calls = self.root / 'systemctl.log'
        self.systemctl = self.root / 'fake-systemctl'
        self.systemctl.write_text('#!/bin/sh\necho "$@" >> %s\n' % self.calls)
        self.systemctl.chmod(0o700)
        self.paths = {'unit': self.live / 'local-ai-chat.service', 'runtime': self.live / 'runtime.json',
                      'opencode': self.live / 'opencode.json', 'cli': self.live / 'bin' / 'local-ai',
                      'launcher': self.live / 'bin' / 'launch_local_plan.py'}
        for key, path in self.paths.items():
            path.write_bytes(('live-preimage-' + key + '\n').encode())
            path.chmod(0o700 if key in ('cli', 'launcher') else 0o600)
        self.base = self.make_base(self.root / 'staged' / 'base')
        self.out = self.root / 'staged' / 'final'

    # -- fixtures -------------------------------------------------------------------------
    def runtime(self, engine_dir, server_sha, flash_pref='18'):
        profiles = {}
        for alias, (model_sha, ram, mtp) in MOE.items():
            profiles[alias] = {
                'display_name': 'Ornith 1.5 35B A3B' if alias.startswith('ornith') else 'Qwen 3.6 35B MoE',
                'sha256': model_sha, 'release_channel': 'beta',
                'memlat_overrides': {'oxe': {
                    'server_binary': engine_dir + '/oxe-server', 'server_sha256': server_sha, 'context_size': 131072,
                    'oxe_args': ['--ctx-size', '{context}', '--ctx-checkpoints', '8', '--session-ram-bytes', str(ram),
                                 '--park-disk-min-bytes', '536870912', '--mtp-max-ctx', str(mtp)],
                    'environment': {'OXE_VL': '15'},
                    'validation': {'kind': 'stale', 'engine_source': '4d1c28b', 'qualified': False}}}}
        profiles[gen.FLASH] = {'display_name': 'Flash', 'memlat_overrides': {'oxe_flash': {
            'cache_pref': flash_pref, 'prefix_store': {'enabled': True}, 'sidecar_path': '/x/experts-r4.bin',
            'oxe_args': ['--flash-sidecar', '/x/experts-r4.bin', '--flash-psi-some', '0', '--flash-psi-full', '0']}},
            'bigmoe': {'shards': [{'sha256': 'c' * 64}]}}
        return {'model_alias': 'ornith-35b-a3b-local', 'cache': dict(gen.CACHE), 'profiles': profiles}

    def opencode(self, compaction=False):
        moe = {'limit': {'context': 131072, 'input': 120000, 'output': 16384}}
        models = {'qwen-local': {'limit': {'context': 16384, 'input': 12288, 'output': 16384}},
                  gen.FLASH: {'limit': {'context': 16384, 'input': 12288, 'output': 2048}},
                  'ornith-35b-a3b-local': dict(moe), 'qwen3.6-35b-a3b-local': dict(moe)}
        cfg = {'$schema': 'https://opencode.ai/config.json',
               'providers': {'local-qwen': {'settings': {'headerTimeout': 1800000}, 'models': models}}}
        if compaction:
            cfg['compaction'] = {'auto': True, 'keep': {'tokens': 8000}}
        return cfg

    def make_base(self, base, flash_pref='18', compaction=False):
        (base / 'rollback').mkdir(parents=True)
        (base / 'engine').mkdir()
        (base / 'harness-registry').mkdir()
        server, oxe = b'fake-oxe-server-binary', b'fake-oxe-cli-binary'
        (base / 'engine' / 'oxe-server').write_bytes(server)
        (base / 'engine' / 'oxe').write_bytes(oxe)
        engine_dir = str(self.root / 'releases' / 'oxe-v012-chat-base')
        release_dir = str(self.root / 'runtime-releases' / 'v012-chat-base')
        (base / 'runtime.json').write_bytes(gen.dump_json(self.runtime(engine_dir, h(server), flash_pref)))
        gate = self.runtime(str(base / 'engine'), h(server), flash_pref)
        (base / 'runtime.gate.json').write_bytes(gen.dump_json(gate))
        (base / 'opencode.json').write_bytes(gen.dump_json(self.opencode(compaction)))
        unit = '\n'.join(['[Service]', gen.WATCHDOG, 'WorkingDirectory=' + release_dir,
                          'ExecStart=/usr/bin/python3 "%s/controller.py"' % release_dir,
                          'Environment="LOCAL_AI_LEASE=/x/inference.lock"', 'Environment="LOCAL_LLM_CONFIG=/x/runtime.json"',
                          'Restart=on-failure', 'MemoryMax=25G']) + '\n'
        (base / 'local-ai-chat.service').write_text(unit)
        (base / 'local-ai').write_bytes(b'old cli payload')
        (base / 'T1-PRIMERS.json').write_text('{}\n')
        (base / 'harness-registry' / 'index.json').write_text('{}\n')
        pre = {}
        for key in ('unit', 'runtime', 'opencode', 'cli'):
            pre[key] = self.paths[key].read_bytes()
            (base / 'rollback' / (key + '.original')).write_bytes(pre[key])
        manifest = {'installed': False, 'paths': {k: str(self.paths[k]) for k in pre},
                    'preimage_sha256': {k: h(v) for k, v in pre.items()},
                    'staged_files': {'unit': 'local-ai-chat.service', 'runtime': 'runtime.json',
                                     'opencode': 'opencode.json', 'cli': 'local-ai'},
                    'server_sha256': h(server), 'oxe_sha256': h(oxe), 'engine_dir': engine_dir,
                    'release_dir': release_dir, 'engine_source': '3' * 40,
                    'profile_requirements': {a: {'context_size': 131072, 'checkpoints': 8, 'session_ram_bytes': ram,
                                                 'mtp_max_ctx': mtp, 'model_sha256': s}
                                             for a, (s, ram, mtp) in MOE.items()}}
        (base / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        return base

    def generate(self, **kw):
        args = dict(base=self.base, out=self.out, lab_repo=ROOT, lab_rev='HEAD', engine_rev='d' * 40,
                    systemctl=str(self.systemctl), forbidden_roots=())
        args.update(kw)
        return gen.generate(**args)

    def live_state(self):
        return {k: (p.read_bytes(), p.stat().st_mode & 0o7777) for k, p in self.paths.items()}

    def rollback(self, *args):
        return subprocess.run([sys.executable, str(self.out / 'rollback' / 'rollback.py'), *args],
                              capture_output=True, text=True, timeout=60)

    # -- tests ----------------------------------------------------------------------------
    def test_generates_five_guarded_payloads_with_preimage_modes(self):
        base_before, live_before = tree(self.base), self.live_state()
        m = self.generate()
        self.assertEqual(tree(self.base), base_before, 'base stage must stay immutable')
        self.assertEqual(self.live_state(), live_before, 'live files are only read')
        self.assertEqual(set(m['paths']), set(gen.KEYS))
        self.assertEqual(m['preimage_mode'], {'unit': '0600', 'runtime': '0600', 'opencode': '0600',
                                              'cli': '0700', 'launcher': '0700'})
        self.assertEqual(m['preimage_sha256']['launcher'], h(b'live-preimage-launcher\n'))
        lab_launcher = subprocess.run(['git', '-C', str(ROOT), 'show', 'HEAD:' + gen.RUNTIME_SRC + '/launch_local_plan.py'],
                                      check=True, capture_output=True).stdout
        self.assertEqual((self.out / 'launch_local_plan.py').read_bytes(), lab_launcher)
        opencode = json.loads((self.out / 'opencode.json').read_text())
        self.assertNotIn('compaction', opencode)
        for name in gen.MOE_MODELS:
            self.assertEqual(opencode['providers']['local-qwen']['models'][name]['limit']['input'], 65536)
        deep = json.loads((self.out / 'opencode-mode-overlay.deep.json').read_text())
        self.assertEqual(deep['providers']['local-qwen']['models']['ornith-35b-a3b-local']['limit']['input'], 120000)
        runtime = json.loads((self.out / 'runtime.json').read_text())
        oxe = runtime['profiles']['ornith-35b-a3b-local']['memlat_overrides']['oxe']
        self.assertEqual(oxe['validation']['engine_source'], '3' * 40)  # carried server keeps its own source
        self.assertEqual(m['engine_artifacts']['oxe-server']['origin'], 'carried from base stage (not rebuilt)')
        unit = (self.out / 'local-ai-chat.service').read_text()
        self.assertIn('WorkingDirectory=' + m['release_dir'], unit)
        self.assertEqual(unit.count(gen.WATCHDOG), 1)
        gate = json.loads((self.out / 'runtime.gate.json').read_text())
        self.assertEqual(gate['profiles']['qwen3.6-35b-a3b-local']['memlat_overrides']['oxe']['server_binary'],
                         str(self.out / 'engine' / 'oxe-server'))
        for path in self.out.rglob('*'):
            if path.is_file():
                self.assertFalse(path.stat().st_mode & 0o222, path)
        self.assertEqual(gen.verify(self.out), len(json.loads((self.out / gen.SEAL).read_text())['files']))
        bundled = subprocess.run([sys.executable, '-I', str(self.out / 'tools' / 'stage_v012_final.py'),
                                  '--verify', str(self.out)], capture_output=True, text=True, timeout=60)
        self.assertEqual(bundled.returncode, 0, bundled.stderr)
        self.assertFalse(self.calls.exists(), 'the generator never calls systemctl')

    def test_install_then_rollback_restores_bytes_and_modes(self):
        before = self.live_state()
        self.generate()
        dry = io.StringIO()
        self.assertEqual(isr.run(self.out, False, out=dry), 0)
        self.assertEqual(self.live_state(), before)
        self.assertEqual(isr.run(self.out, True, out=io.StringIO()), 0)
        for key, path in self.paths.items():
            self.assertEqual(path.read_bytes(), (self.out / gen.STAGED[key]).read_bytes())
            self.assertEqual(path.stat().st_mode & 0o7777, before[key][1])
        self.assertFalse(self.calls.exists(), 'the installer never controls services')
        for key in ('cli', 'launcher'):  # a hand chmod would be undone by the exact-mode rollback
            self.paths[key].chmod(0o600)
        self.assertEqual(self.rollback('--check').returncode, 0)
        result = self.rollback()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.live_state(), before)
        self.assertEqual(self.calls.read_text().splitlines(),
                         ['--user stop local-ai-chat.service', '--user daemon-reload', '--user start local-ai-chat.service'])
        self.assertTrue(os.access(self.paths['cli'], os.X_OK) and os.access(self.paths['launcher'], os.X_OK))

    def test_install_refuses_preimage_or_mode_mismatch(self):
        self.generate()
        self.paths['launcher'].chmod(0o600)
        before = self.live_state()
        with self.assertRaisesRegex(isr.RefuseError, 'launcher: live mode 0600'):
            isr.run(self.out, True, out=io.StringIO())
        self.paths['launcher'].chmod(0o700)
        self.paths['opencode'].write_bytes(b'edited by hand')
        with self.assertRaisesRegex(isr.RefuseError, 'opencode: live sha256'):
            isr.run(self.out, True, out=io.StringIO())
        self.assertEqual(self.live_state()['unit'], before['unit'])
        self.assertFalse((self.out / 'receipts').exists())

    def test_rollback_refuses_manifest_mode_tamper(self):
        self.generate()
        self.assertEqual(isr.run(self.out, True, out=io.StringIO()), 0)
        manifest = self.out / 'manifest.json'
        manifest.chmod(0o600)
        data = json.loads(manifest.read_text())
        data['preimage_mode']['cli'] = '0600'
        manifest.write_text(json.dumps(data))
        result = self.rollback()
        self.assertEqual(result.returncode, 5)
        self.assertIn('preimage_mode.cli', result.stderr)
        self.assertFalse(self.calls.exists())
        with self.assertRaisesRegex(gen.StageError, 'seal mismatch'):
            gen.verify(self.out)

    def test_generator_refusals(self):
        self.paths['runtime'].write_bytes(b'drifted')
        with self.assertRaisesRegex(gen.StageError, 'not the base preimage'):
            self.generate()
        self.assertFalse(self.out.exists())
        self.paths['runtime'].write_bytes(b'live-preimage-runtime\n')
        self.out.mkdir()
        with self.assertRaisesRegex(gen.StageError, 'fresh'):
            self.generate()
        self.out.rmdir()
        with self.assertRaisesRegex(gen.StageError, 'live configuration'):
            self.generate(forbidden_roots=(self.root,))
        bad = self.make_base(self.root / 'staged' / 'bad-flash', flash_pref='14')
        with self.assertRaisesRegex(gen.StageError, 'cache_pref must be "18"'):
            self.generate(base=bad)
        self.assertFalse(self.out.exists())
        with self.assertRaisesRegex(gen.StageError, 'source'):
            self.generate(server=self.base / 'engine' / 'oxe-server')

    def test_global_compaction_is_dropped_and_new_server_rebinds(self):
        base = self.make_base(self.root / 'staged' / 'with-compaction', compaction=True)
        server = self.root / 'new-oxe-server'
        server.write_bytes(b'final-3 server')
        m = self.generate(base=base, server=server, server_source='e' * 40)
        self.assertNotIn('compaction', json.loads((self.out / 'opencode.json').read_text()))
        self.assertEqual(m['server_sha256'], h(b'final-3 server'))
        self.assertTrue(m['engine_dir'].endswith('oxe-v012-final-' + h(b'final-3 server')[:12]))
        runtime = json.loads((self.out / 'runtime.json').read_text())
        oxe = runtime['profiles']['qwen3.6-35b-a3b-local']['memlat_overrides']['oxe']
        self.assertEqual(oxe['server_binary'], m['engine_dir'] + '/oxe-server')
        self.assertEqual(oxe['validation']['engine_source'], 'e' * 40)
        self.assertEqual(m['engine_artifacts']['oxe']['source'], '3' * 40)

    def test_opencode_ornith_beta_label_is_removed(self):
        path = self.base / 'opencode.json'
        cfg = json.loads(path.read_text())
        cfg['providers']['local-qwen']['models']['ornith-35b-a3b-local']['name'] = 'Ornith 1.5 35B A3B (beta)'
        path.write_text(json.dumps(cfg, indent=2) + '\n')
        self.generate()
        proposed = json.loads((self.out / 'opencode.json').read_text())
        self.assertEqual(proposed['providers']['local-qwen']['models']['ornith-35b-a3b-local']['name'],
                         'Ornith 1.5 35B A3B')
        proposed['providers']['local-qwen']['models']['ornith-35b-a3b-local']['name'] += ' (beta)'
        self.assertTrue(any('beta' in e for e in gen.check_opencode(proposed, cfg)))

    def test_n9_chmod_step(self):
        d = self.root / 'n9'
        (d / 'datasets').mkdir(parents=True)
        (d / 'datasets').chmod(0o755)
        (d / 'datasets' / 'keep.txt').write_text('kept')
        (d / 'backup.json').write_text('{}')
        (d / 'backup.json').chmod(0o644)
        (d / 'x.lock').write_text('')
        (d / 'x.lock').chmod(0o664)

        def entry(path, target, kind):
            st = os.lstat(path)
            return {'path': str(path), 'dev': st.st_dev, 'inode': st.st_ino, 'uid': st.st_uid,
                    'mode': st.st_mode & 0o7777, 'target_mode': target, 'kind': kind}
        inventory = {'applied': False, 'chmod_only': True, 'files': [
            entry(d / 'datasets', 0o700, 'dataset_directory'), entry(d / 'backup.json', 0o600, 'backup'),
            entry(d / 'x.lock', 0o600, 'hf_lock')]}
        inv = self.root / 'inventory.json'
        inv.write_text(json.dumps(inventory))
        self.generate(n9_inventory=inv)
        script = self.out / 'n9' / 'n9_chmod.py'
        run = lambda *a: subprocess.run([sys.executable, str(script), *a], capture_output=True, text=True, timeout=30)
        modes = lambda: [os.lstat(e['path']).st_mode & 0o7777 for e in inventory['files']]
        check = run('--check')
        self.assertEqual(check.returncode, 0, check.stderr)
        self.assertEqual(modes(), [0o755, 0o644, 0o664])
        self.assertEqual(run('--apply').returncode, 0)
        self.assertEqual(modes(), [0o700, 0o600, 0o600])
        self.assertEqual((d / 'datasets' / 'keep.txt').read_text(), 'kept')
        self.assertEqual(run('--apply').returncode, 0)  # idempotent
        self.assertEqual(run('--restore').returncode, 0)
        self.assertEqual(modes(), [0o755, 0o644, 0o664])
        (d / 'x.lock').unlink()
        (d / 'x.lock').symlink_to(d / 'backup.json')
        refused = run('--apply')
        self.assertEqual(refused.returncode, 3)
        self.assertEqual([os.lstat(e['path']).st_mode & 0o777 for e in inventory['files'][:2]], [0o755, 0o644])
        self.assertTrue((d / 'datasets' / 'keep.txt').exists())
        self.assertFalse(self.calls.exists())


class ModeBindingTest(unittest.TestCase):
    def test_render_rejects_bad_modes(self):
        paths, data = {'cli': '/x/cli'}, {'cli': b'a'}
        for bad in (0o4755, 0o500, '9', 'rwx', {'other': 0o600}):
            with self.assertRaises(guard.GuardError):
                guard.render_rollback(paths, data, data, modes=bad if isinstance(bad, dict) else {'cli': bad})
        text = guard.render_rollback(paths, data, data, modes={'cli': '0700'}).decode()
        self.assertIn('"mode": 448', text)
        self.assertEqual(guard.manifest_binding(paths, data, data, modes={'cli': 0o700})['preimage_mode'], {'cli': '0700'})
        self.assertNotIn('preimage_mode', guard.manifest_binding(paths, data, data))


if __name__ == '__main__':
    unittest.main()
