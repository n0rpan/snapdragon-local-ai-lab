"""Staged Flash package: manifest, engine copy, candidate hash, ledger, install and rollback.

Model-free. Every path is a temp directory under tempfile.mkdtemp(); systemctl is a fake
script that only appends its argv to a log. Nothing touches ~/.config, ~/.local or a real service.
"""
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from snapdragon_ai_lab.runtime import engine_ledger, profile_policy
from test_flash_oxe_production import ALIAS, PEAK, UNIT, stage_tool, staged_runtime

GIB = 1024**3
EXPECTED_CALLS = ['--user stop local-ai-chat.service', '--user daemon-reload', '--user start local-ai-chat.service']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


class FlashStagePackage(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix='flash-stage-package-'))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        live = self.tmp / 'live'
        live.mkdir()
        self.runtime = live / 'runtime.json'
        self.unit = live / 'local-ai-chat.service'
        self.runtime.write_text(json.dumps(staged_runtime()))
        self.unit.write_text(UNIT)
        build = self.tmp / 'build'
        build.mkdir()
        self.engine = build / 'oxe-server'
        self.engine.write_bytes(b'synthetic rust flash server\n' * 64)
        self.sidecar = build / 'sidecar.bin'
        self.sidecar.write_bytes(b's' * 4096)
        self.stage_dir = self.tmp / 'stage'
        self.engines = self.tmp / 'engines'
        self.releases = self.tmp / 'releases'
        self.live_before = (self.runtime.read_bytes(), self.unit.read_bytes())
        # The root package is patched to a synthetic copy: these tests never depend on packaging/flash-qos.
        self.qos_packaging = self.tmp / 'flash-qos'
        self.qos_packaging.mkdir()
        for name, text in (('local-ai-flash-qos', '#!/bin/sh\n# synthetic helper\n'),
                           ('local-ai-flash-qos@.service', '[Unit]\nDescription=synthetic\n'),
                           ('49-local-ai-flash-qos.rules', '// synthetic rule\n'),
                           ('README.md', '# synthetic\n')):
            (self.qos_packaging / name).write_text(text)
        (self.qos_packaging / 'local-ai-flash-qos').chmod(0o755)
        patcher = patch.object(stage_tool, 'QOS_PACKAGING', self.qos_packaging)
        patcher.start()
        self.addCleanup(patcher.stop)

    def run_stage(self, **extra):
        kwargs = dict(runtime_path=self.runtime, runtime_sha=digest(self.runtime), unit_path=self.unit,
                      unit_sha=digest(self.unit), server_binary=self.engine, releases_root=self.releases,
                      engine_releases_root=self.engines, sidecar_path=self.sidecar, measured_peak_bytes=PEAK,
                      candidate_sha256=digest(self.engine))
        kwargs.update(extra)
        return stage_tool.stage(self.stage_dir, **kwargs)

    def staged_override(self):
        staged = json.loads((self.stage_dir / 'runtime.json').read_text())
        return staged['profiles'][ALIAS]['memlat_overrides']['oxe_flash']

    def fake_systemctl(self):
        bindir = self.tmp / 'bin'
        bindir.mkdir()
        self.log = self.tmp / 'systemctl.log'
        fake = bindir / 'systemctl'
        fake.write_text('#!/bin/sh\necho "$*" >> ' + shlex.quote(str(self.log)) + '\n')
        fake.chmod(0o700)
        return fake

    def under_test(self, script, fake):
        """Copy a generated script beside itself with SYSTEMCTL pointed at the fake binary."""
        text = script.read_text()
        # install.py names systemctl in a constant; the guarded rollback.py (shared guard) in its binding
        old, new = "SYSTEMCTL = 'systemctl'", 'SYSTEMCTL = %r' % str(fake)
        if old not in text:
            old, new = "SYSTEMCTL = BINDING['systemctl']", 'SYSTEMCTL = %r' % str(fake)
        self.assertEqual(text.count(old), 1)
        copy = script.with_name(script.stem + '_under_test.py')
        copy.write_text(text.replace(old, new))
        copy.chmod(0o700)
        return copy

    def run_script(self, script, *args):
        if script.name.startswith('install') and not args:
            args = ('--allow-unqualified-candidate',)
        return subprocess.run([sys.executable, str(script), *[a for a in args if a]], capture_output=True, text=True,
                              timeout=120)

    def calls(self):
        return self.log.read_text().splitlines() if self.log.exists() else []

    def live_bytes(self):
        return (self.runtime.read_bytes(), self.unit.read_bytes())

    def test_manifest_records_identity_and_nothing_installed(self):
        manifest = self.run_stage()
        self.assertEqual(json.loads((self.stage_dir / 'manifest.json').read_text()), manifest)
        server_sha = digest(self.engine)
        self.assertEqual(manifest['alias'], ALIAS)
        self.assertEqual(manifest['stage'], str(self.stage_dir))
        self.assertEqual(manifest['server_sha256'], server_sha)
        self.assertEqual(manifest['engine_install_path'],
                         str(self.engines / ('oxe-flash-' + server_sha[:12]) / 'oxe-server'))
        self.assertEqual(manifest['engine_source'], str(self.engine))
        self.assertEqual(manifest['cache_gib'], 20)
        self.assertIsNone(manifest['fallback_cache_gib'])
        self.assertIsNone(manifest['fallback_measured_peak_bytes'])
        self.assertEqual(manifest['measured_peak_bytes'], PEAK)
        self.assertEqual(manifest['session_reserve_bytes'], 512 * 1024**2)
        self.assertFalse(manifest['qualified'])
        self.assertEqual(manifest['runtime_sha256'],
                         {'preimage': sha_bytes(self.live_before[0]), 'staged': digest(self.stage_dir / 'runtime.json')})
        self.assertEqual(manifest['unit_sha256'],
                         {'preimage': sha_bytes(self.live_before[1]),
                          'staged': digest(self.stage_dir / 'local-ai-chat.service')})
        self.assertIs(manifest['installed'], False)
        self.assertIs(manifest['models_started'], False)
        self.assertTrue((self.stage_dir / 'install.py').is_file())
        self.assertEqual(stat.S_IMODE((self.stage_dir / 'install.py').stat().st_mode), 0o700)
        self.assertFalse(self.engines.exists())
        self.assertFalse(self.releases.exists())
        self.assertEqual(self.live_bytes(), self.live_before)

    def test_engine_is_copied_with_matching_hash_and_mode(self):
        self.run_stage()
        copied = self.stage_dir / 'engine' / 'oxe-server'
        self.assertEqual(copied.read_bytes(), self.engine.read_bytes())
        self.assertEqual(digest(copied), digest(self.engine))
        self.assertEqual(stat.S_IMODE(copied.stat().st_mode), 0o700)

    def test_runtime_names_the_install_path_and_candidate_is_unqualified(self):
        manifest = self.run_stage()
        override = self.staged_override()
        self.assertEqual(override['server_binary'], manifest['engine_install_path'])
        self.assertEqual(override['server_sha256'], digest(self.engine))
        self.assertEqual(override['validation'], {'kind': 'oxe-rust-flash-candidate', 'server_sha256': digest(self.engine),
                                                  'report': 'docs/FLASH-RUST-PRODUCTION.md', 'qualified': False})
        self.assertEqual(override['session_reserve_bytes'], stage_tool.SESSION_RESERVE_BYTES)
        self.assertFalse(self.engines.exists())

    def test_wrong_candidate_hash_is_refused_and_nothing_is_staged(self):
        with self.assertRaisesRegex(ValueError, 'unqualified_rust_flash_server'):
            self.run_stage(candidate_sha256='e' * 64)
        self.assertFalse(self.stage_dir.exists())

    def test_no_candidate_and_no_qualified_hash_is_refused(self):
        self.assertIsNone(stage_tool.QUALIFIED_FLASH_SERVER_SHA)
        with self.assertRaisesRegex(ValueError, 'unqualified_rust_flash_server'):
            self.run_stage(candidate_sha256=None)
        self.assertFalse(self.stage_dir.exists())

    def test_matching_qualified_hash_marks_the_package_qualified(self):
        with patch.object(stage_tool, 'QUALIFIED_FLASH_SERVER_SHA', digest(self.engine)):
            manifest = self.run_stage(candidate_sha256=None)
        self.assertTrue(manifest['qualified'])
        validation = self.staged_override()['validation']
        self.assertEqual(validation['kind'], 'oxe-rust-flash-qualification')
        self.assertIs(validation['qualified'], True)

    def test_flash_args_keep_four_checkpoints_no_session_ram_and_the_chosen_cache(self):
        args = stage_tool.FLASH_ARGS
        self.assertEqual(args[args.index('--ctx-checkpoints') + 1], '4')
        self.assertEqual(args[args.index('--session-ram-bytes') + 1], '0')
        self.assertNotIn('--turn-checkpoints', args)
        for cache in (16, 18, 20):
            self.stage_dir = self.tmp / ('stage-%d' % cache)
            manifest = self.run_stage(cache_gib=cache)
            override = self.staged_override()
            oxe = override['oxe_args']
            self.assertEqual(oxe[oxe.index('--flash-cache-gib') + 1], str(cache))
            self.assertEqual(oxe[oxe.index('--ctx-checkpoints') + 1], '4')
            self.assertEqual(oxe[oxe.index('--session-ram-bytes') + 1], '0')
            self.assertEqual(manifest['cache_gib'], cache)
            self.assertEqual(override['cache_gib'], cache)

    def test_fallback_cache_is_recorded_in_the_manifest(self):
        manifest = self.run_stage(fallback_peak_bytes=23 * GIB)
        self.assertEqual(manifest['fallback_cache_gib'], 18)
        self.assertEqual(manifest['fallback_measured_peak_bytes'], 23 * GIB)
        self.assertEqual(self.staged_override()['fallback']['cache_gib'], 18)

    def test_ledger_env_reaches_the_resolved_flash_profile(self):
        self.run_stage()
        staged = json.loads((self.stage_dir / 'runtime.json').read_text())
        profile = staged['profiles'][ALIAS]
        merged = {**{k: v for k, v in staged.items() if k != 'profiles'}, **profile}
        effective = profile_policy.resolve(merged, available=True, oxe_flash_enabled=True)
        self.assertEqual(effective['backend_kind'], 'oxe')
        self.assertEqual(effective['environment'][engine_ledger.ENABLE_ENV], '1')
        self.assertTrue(engine_ledger.enabled(effective, effective['oxe_args']))
        stock = profile_policy.resolve(merged, available=True, oxe_flash_enabled=False)
        self.assertFalse(engine_ledger.enabled(stock, []))

    def test_install_then_rollback_round_trip_with_fake_systemctl(self):
        manifest = self.run_stage()
        fake = self.fake_systemctl()
        done = self.run_script(self.under_test(self.stage_dir / 'install.py', fake))
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn(str(self.stage_dir / 'rollback' / 'rollback.py'), done.stdout)
        self.assertEqual(self.runtime.read_bytes(), (self.stage_dir / 'runtime.json').read_bytes())
        self.assertEqual(self.unit.read_bytes(), (self.stage_dir / 'local-ai-chat.service').read_bytes())
        engine = Path(manifest['engine_install_path'])
        self.assertEqual(digest(engine), digest(self.engine))
        self.assertEqual(stat.S_IMODE(engine.stat().st_mode), 0o700)
        self.assertEqual(stat.S_IMODE(engine.parent.stat().st_mode), 0o700)
        self.assertTrue((Path(manifest['release_dir']) / 'controller.py').is_file())
        self.assertEqual(self.calls(), EXPECTED_CALLS)

        undone = self.run_script(self.under_test(self.stage_dir / 'rollback' / 'rollback.py', fake))
        self.assertEqual(undone.returncode, 0, undone.stderr)
        self.assertIn('engine and release directories were left in place', undone.stdout)
        self.assertEqual(self.live_bytes(), self.live_before)
        self.assertEqual(self.calls(), EXPECTED_CALLS * 2)
        self.assertEqual(digest(engine), digest(self.engine))
        self.assertTrue(Path(manifest['release_dir']).is_dir())

    def test_install_exits_3_and_changes_nothing_when_live_runtime_moved_on(self):
        self.run_stage()
        fake = self.fake_systemctl()
        self.runtime.write_text(self.runtime.read_text() + '\n')
        before = self.live_bytes()
        done = self.run_script(self.under_test(self.stage_dir / 'install.py', fake))
        self.assertEqual(done.returncode, 3, done.stderr)
        self.assertEqual(self.live_bytes(), before)
        self.assertFalse(self.engines.exists())
        self.assertFalse(self.releases.exists())
        self.assertEqual(self.calls(), [])

    def test_engine_reads_the_staged_sidecar_with_the_io_config(self):
        self.run_stage(io_mode='spin')
        override = self.staged_override()
        args = override['oxe_args']
        self.assertEqual(args[args.index('--flash-sidecar') + 1], str(self.sidecar))
        self.assertEqual(override['sidecar_path'], str(self.sidecar))
        self.assertEqual(override['environment']['OXF_IO'], 'spins=1000000')
        self.assertNotIn('idle_qos', override)

    def test_qos_mode_is_the_default_io_layout_with_the_idle_qos_hook(self):
        manifest = self.run_stage()
        override = self.staged_override()
        self.assertNotIn('OXF_IO', override['environment'])
        self.assertEqual(override['idle_qos'], {'unit': 'local-ai-flash-qos@{pid}.service', 'cpu': 8, 'latency_us': 100})
        args = override['oxe_args']
        self.assertEqual(args[args.index('--flash-io-cpu') + 1], '8')
        self.assertEqual(override['environment']['GOMP_CPU_AFFINITY'], '0-7 9-11')
        self.assertEqual(manifest['io_mode'], 'qos')
        self.assertEqual(manifest['root_install'], 'system/install-root.sh')
        self.assertIn("ROOT_INSTALL = 'system/install-root.sh'\n", (self.stage_dir / 'install.py').read_text())

    def test_qos_mode_copies_the_root_package_with_modes(self):
        self.run_stage()
        system = self.stage_dir / 'system'
        names = {'local-ai-flash-qos', 'local-ai-flash-qos@.service', '49-local-ai-flash-qos.rules', 'README.md',
                 'install-root.sh', 'rollback-root.sh'}
        self.assertEqual({p.name for p in system.iterdir()}, names)
        for name, mode in (('local-ai-flash-qos', 0o700), ('install-root.sh', 0o700), ('rollback-root.sh', 0o700),
                           ('local-ai-flash-qos@.service', 0o600), ('49-local-ai-flash-qos.rules', 0o600),
                           ('README.md', 0o600)):
            with self.subTest(name=name):
                self.assertEqual(stat.S_IMODE((system / name).stat().st_mode), mode)
        self.assertEqual((system / 'local-ai-flash-qos').read_bytes(), (self.qos_packaging / 'local-ai-flash-qos').read_bytes())

    def test_qos_mode_refuses_a_missing_or_incomplete_root_package_before_writing(self):
        shutil.rmtree(self.qos_packaging)
        with self.assertRaisesRegex(ValueError, 'flash_qos_packaging_missing'):
            self.run_stage()
        self.assertFalse(self.stage_dir.exists())
        self.assertFalse(self.engines.exists())
        self.assertEqual(self.live_bytes(), self.live_before)
        self.qos_packaging.mkdir()
        (self.qos_packaging / 'local-ai-flash-qos').write_text('x')
        (self.qos_packaging / 'local-ai-flash-qos@.service').write_text('x')
        with self.assertRaisesRegex(ValueError, 'flash_qos_packaging_missing'):
            self.run_stage()
        self.assertFalse(self.stage_dir.exists())
        self.assertEqual(self.live_bytes(), self.live_before)

    def test_spin_mode_is_the_fallback_without_hook_or_root_part(self):
        manifest = self.run_stage(io_mode='spin')
        override = self.staged_override()
        self.assertEqual(override['environment']['OXF_IO'], 'spins=1000000')
        self.assertNotIn('idle_qos', override)
        args = override['oxe_args']
        self.assertEqual(args[args.index('--flash-io-cpu') + 1], '8')
        self.assertEqual(override['environment']['GOMP_CPU_AFFINITY'], '0-7 9-11')
        self.assertEqual(manifest['io_mode'], 'spin')
        self.assertIsNone(manifest['root_install'])
        self.assertFalse((self.stage_dir / 'system').exists())
        self.assertIn('ROOT_INSTALL = None\n', (self.stage_dir / 'install.py').read_text())
        self.stage_dir = self.tmp / 'stage-fast'
        with self.assertRaisesRegex(ValueError, 'flash_io_mode_invalid'):
            self.run_stage(io_mode='fast')
        self.assertFalse(self.stage_dir.exists())

    def test_spin_mode_does_not_need_the_root_package(self):
        shutil.rmtree(self.qos_packaging)
        manifest = self.run_stage(io_mode='spin')
        self.assertIsNone(manifest['root_install'])

    def test_root_install_script_installs_exactly_three_files_and_starts_nothing(self):
        self.run_stage()
        text = (self.stage_dir / 'system' / 'install-root.sh').read_text()
        self.assertTrue(text.startswith('#!/bin/sh\n'))
        self.assertIn('set -eu', text)
        self.assertIn('id -u', text)
        self.assertIn('readlink -f "$0"', text)
        installs = [line.split() for line in text.splitlines() if line.startswith('install ')]
        self.assertEqual([line[-1] for line in installs], [
            '/usr/local/libexec/local-ai-flash-qos',
            '/etc/systemd/system/local-ai-flash-qos@.service',
            '/etc/polkit-1/rules.d/49-local-ai-flash-qos.rules'])
        self.assertEqual([line[line.index('-m') + 1] for line in installs], ['0755', '0644', '0644'])
        for line in installs:
            i = line.index('-o')
            self.assertEqual(line[i:i + 4], ['-o', 'root', '-g', 'root'])
        self.assertIn('-D', installs[0])  # /usr/local/libexec may not exist
        self.assertIn('systemctl daemon-reload', text)
        self.assertNotIn('systemctl start', text)
        self.assertNotIn('enable', text)

    def root_identity_fixture(self):
        """Fake NSS/root/install commands; never write to actual system destinations."""
        (self.qos_packaging / 'local-ai-flash-qos').write_text('#!/usr/bin/python3\nALLOWED_UID = None\n')
        (self.qos_packaging / '49-local-ai-flash-qos.rules').write_text('var ALLOWED_USER = null;\n')
        self.run_stage()
        bindir = self.tmp / 'root-tools'; bindir.mkdir()
        installed = self.tmp / 'root-install-output'; installed.mkdir()
        log = self.tmp / 'root-calls.jsonl'
        fake = '''#!__PYTHON__
import json, os, shutil, sys
from pathlib import Path
name, args = Path(sys.argv[0]).name, sys.argv[1:]
user, uid = os.environ['QOS_TEST_USER'], os.environ['QOS_TEST_UID']
if name == 'id':
    if args == ['-u']: print('0'); raise SystemExit(0)
    account = args[-1]
    if account not in (user, 'root'): raise SystemExit(1)
    if args[0] == '-u': print('0' if account == 'root' else uid)
    elif args[0] == '-un': print('root' if account == 'root' else os.environ.get('QOS_TEST_CANONICAL', user))
    else: raise SystemExit(1)
elif name == 'getent':
    if args == ['passwd', 'root']: print('root:x:0:0:root:/root:/bin/sh')
    elif args == ['passwd', user]: print(user + ':x:' + os.environ.get('QOS_TEST_ACCOUNT_UID', uid) + ':123:test:/nonexistent:/bin/sh')
    else: raise SystemExit(2)
elif name == 'install':
    destination = Path(os.environ['QOS_TEST_OUTPUT']) / Path(args[-1]).name
    shutil.copyfile(args[-2], destination)
    destination.chmod(int(args[args.index('-m') + 1], 8))
elif name != 'systemctl': raise SystemExit(1)
if name in ('install', 'systemctl'):
    with open(os.environ['QOS_TEST_LOG'], 'a') as stream: stream.write(json.dumps([name, *args]) + '\\n')
'''.replace('__PYTHON__', sys.executable)
        for name in ('id', 'getent', 'install', 'systemctl'):
            path = bindir / name; path.write_text(fake); path.chmod(0o700)
        env = {**os.environ, 'PATH': str(bindir) + os.pathsep + os.environ['PATH'],
               'SUDO_USER': 'localai_test', 'QOS_TEST_USER': 'localai_test', 'QOS_TEST_UID': '24680',
               'QOS_TEST_OUTPUT': str(installed), 'QOS_TEST_LOG': str(log)}
        return self.stage_dir / 'system/install-root.sh', env, installed, log

    def test_root_install_renders_same_sudo_account_into_helper_and_rule(self):
        script, env, installed, log = self.root_identity_fixture()
        done = subprocess.run(['/bin/sh', str(script)], env=env, capture_output=True, text=True, timeout=30)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn('ALLOWED_UID = 24680', (installed / 'local-ai-flash-qos').read_text())
        self.assertEqual((installed / '49-local-ai-flash-qos.rules').read_text(), 'var ALLOWED_USER = "localai_test";\n')
        self.assertEqual((script.parent / 'local-ai-flash-qos').read_text(), '#!/usr/bin/python3\nALLOWED_UID = None\n')
        calls = [json.loads(line) for line in log.read_text().splitlines()]
        self.assertEqual(len([call for call in calls if call[0] == 'install']), 3)
        self.assertEqual(calls[-1], ['systemctl', 'daemon-reload'])
        for call in calls[:-1]:
            index = call.index('-o')
            self.assertEqual(call[index:index+4], ['-o', 'root', '-g', 'root'])

    def test_root_install_explicit_user_and_invalid_identity_fail_closed(self):
        script, env, installed, log = self.root_identity_fixture()
        for supplied, extra, args in (
            ('', {}, []), ('root', {}, []), ('missing', {}, []), ('bad;name', {}, []),
            ('localai_test', {'QOS_TEST_ACCOUNT_UID': '24681'}, []),
            ('localai_test', {'QOS_TEST_CANONICAL': 'somebody_else'}, []),
            ('localai_test', {}, ['--user', 'root']),
        ):
            with self.subTest(sudo_user=supplied, extra=extra, args=args):
                done = subprocess.run(['/bin/sh', str(script), *args], env={**env, **extra, 'SUDO_USER': supplied},
                                      capture_output=True, text=True, timeout=30)
                self.assertNotEqual(done.returncode, 0, done.stdout)
                self.assertFalse(log.exists(), 'identity refusal must precede every system write/reload')
                self.assertEqual(list(installed.iterdir()), [])
        done = subprocess.run(['/bin/sh', str(script), '--user', 'localai_test'],
                              env={**env, 'SUDO_USER': 'root'}, capture_output=True, text=True, timeout=30)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn('ALLOWED_UID = 24680', (installed / 'local-ai-flash-qos').read_text())

    def test_root_rollback_script_removes_exactly_the_three_files(self):
        self.run_stage()
        text = (self.stage_dir / 'system' / 'rollback-root.sh').read_text()
        self.assertTrue(text.startswith('#!/bin/sh\n'))
        self.assertIn('id -u', text)
        self.assertIn("systemctl stop 'local-ai-flash-qos@*.service' || true", text)
        self.assertIn('reset || true', text)
        self.assertIn('systemctl daemon-reload', text)
        self.assertNotIn('rm -rf', text)
        removed = [line.split()[2:] for line in text.splitlines() if line.startswith('rm -f ')]
        self.assertEqual(removed, [[
            '/usr/local/libexec/local-ai-flash-qos',
            '/etc/systemd/system/local-ai-flash-qos@.service',
            '/etc/polkit-1/rules.d/49-local-ai-flash-qos.rules']])

    @unittest.skipIf(hasattr(os, 'geteuid') and os.geteuid() == 0, 'refusal path needs a non-root user')
    def test_root_scripts_refuse_to_run_as_a_user(self):
        self.run_stage()
        for name in ('install-root.sh', 'rollback-root.sh'):
            with self.subTest(name=name):
                done = subprocess.run(['/bin/sh', str(self.stage_dir / 'system' / name)], capture_output=True,
                                      text=True, timeout=30)
                self.assertEqual(done.returncode, 1, done.stderr)
                self.assertIn('must run as root', done.stderr)

    def test_package_has_no_warmup_path(self):
        # Stream::warm double-read bug: no hot-set file, no stored-state load, no RAM parking
        manifest = self.run_stage()
        args = self.staged_override()['oxe_args']
        for flag in stage_tool.WARMUP_FLAGS:
            self.assertNotIn(flag, args)
        self.assertEqual(args[args.index('--session-ram-bytes') + 1], '0')
        self.assertIs(manifest['warmup'], False)
        for bad in (['--flash-hotset', 'default'], ['--prefix-cache', 'default'], ['--session-cache=/x']):
            with self.assertRaisesRegex(ValueError, 'flash_warmup_paths_must_stay_off'):
                stage_tool.check_no_warmup(args + bad)
        parked = [('1073741824' if i and args[i - 1] == '--session-ram-bytes' else a) for i, a in enumerate(args)]
        with self.assertRaisesRegex(ValueError, 'flash_warmup_paths_must_stay_off'):
            stage_tool.check_no_warmup(parked)

    def test_install_refuses_an_unqualified_candidate_without_the_explicit_flag(self):
        self.run_stage()
        result = self.run_script(self.under_test(self.stage_dir / 'install.py', self.fake_systemctl()), '')
        self.assertEqual(result.returncode, 6, result.stderr)
        self.assertEqual(self.live_bytes(), self.live_before)
        self.assertEqual(self.calls(), [])
        self.assertFalse(self.engines.exists())

    def test_install_exits_4_on_a_different_engine_at_the_install_path(self):
        manifest = self.run_stage()
        fake = self.fake_systemctl()
        target = Path(manifest['engine_install_path'])
        target.parent.mkdir(parents=True, mode=0o700)
        target.write_bytes(b'a different engine')
        done = self.run_script(self.under_test(self.stage_dir / 'install.py', fake))
        self.assertEqual(done.returncode, 4, done.stderr)
        self.assertEqual(self.live_bytes(), self.live_before)
        self.assertEqual(target.read_bytes(), b'a different engine')
        self.assertFalse(Path(manifest['release_dir']).exists())
        self.assertEqual(self.calls(), [])


if __name__ == '__main__':
    unittest.main()
