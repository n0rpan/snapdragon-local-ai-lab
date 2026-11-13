"""v0.1.3 lane H (F-IMG-SHIP6): Studio's stall watchdog on oxi, probe sd-cli and NPU worker runs.

Model-free: the engines are tiny Python stub scripts run through the real run_engine boundary.
HOME/XDG come from tests/_isolation; Studio's config, output, stack and lease paths are temp dirs.
"""
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import time
import unittest
import unittest.mock
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / 'src/snapdragon_ai_lab/runtime/studio'
PROMPT = 'a secret red bicycle'

STALL = '''import os, signal, sys, time
open(%(pidfile)r, 'w').write(str(os.getpid()))
if %(ignore_term)r:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
print('loading tensors', flush=True)
time.sleep(%(sleep)r)
'''

STEADY = '''import sys, time
for i in range(%(lines)d):
    sys.stdout.write('\\r  %%d/%(lines)d - 0.10s/it' %% (i + 1)); sys.stdout.flush()
    time.sleep(0.1)
print(flush=True)
'''

SD_CLI = '''import struct, sys
args = sys.argv[1:]
def opt(name):
    return args[args.index(name) + 1]
width, height = int(opt('-W')), int(opt('-H'))
ihdr = struct.pack('>II', width, height) + b'\\x08\\x02\\x00\\x00\\x00'
open(opt('-o'), 'wb').write(b'\\x89PNG\\r\\n\\x1a\\n' + struct.pack('>I', 13) + b'IHDR' + ihdr + b'\\x00' * 4)
print('sampling completed', flush=True)
'''


def write_script(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'#!{sys.executable}\n' + body)
    path.chmod(0o700)


def gone(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    # Still visible: only acceptable as an unreaped zombie of some other parent (never ours).
    try:
        return Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].startswith('Z')
    except (OSError, IndexError):
        return True


class StudioCase(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        environment = {'IMG_OUTDIR': str(self.root / 'output'), 'LOCAL_IMAGE_CONFIG_DIR': str(self.root / 'config'),
                       'LOCAL_AI_LEASE': str(self.root / 'inference.lock'), 'SD_HOME': str(self.root / 'stack')}
        with patch.dict(os.environ, environment):
            for key in ('OXI_BIN', 'SD_CLI_PROBE', 'OXI_COND_CACHE', 'OXI_WORK', 'SD_CLI', 'OXI_SIZE_GATE_DIR'):
                os.environ.pop(key, None)
            loader = importlib.machinery.SourceFileLoader('v013_watchdog_studio', str(SOURCE))
            spec = importlib.util.spec_from_loader(loader.name, loader)
            self.studio = importlib.util.module_from_spec(spec)
            loader.exec_module(self.studio)
        s = self.studio
        self.assertTrue(str(s.PROJECT).startswith(str(self.root)))
        for path in (s.KLEIN_FILES['q8'], s.KLEIN_LLM, s.KLEIN_VAE):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'stub')
        s.PROJECT.mkdir(parents=True, exist_ok=True)
        s.OXI_STALL_GRACE_SECONDS = 0.5
        self.pidfile = self.root / 'stub.pid'
        write_script(s.SD_PROBE_CLI, STALL % {'pidfile': str(self.pidfile), 'ignore_term': False, 'sleep': 60})
        write_script(s.SDCLI, SD_CLI)
        self.stalling_oxi()

    def stalling_oxi(self, ignore_term=False, sleep=60):
        write_script(self.studio.OXI_BIN, STALL % {'pidfile': str(self.pidfile), 'ignore_term': ignore_term,
                                                    'sleep': sleep})

    def settings(self, **oxi):
        (self.studio.PROJECT / 'runtime-settings.json').write_text(json.dumps({'oxi': oxi}))

    def stub_pid(self):
        return int(self.pidfile.read_text())


class Setting(StudioCase):
    def test_default_and_validation(self):
        s = self.studio
        self.assertEqual(s.oxi_stall_seconds(), 180.0)
        for value, expected in ((0, 0.0), (2.5, 2.5), (600, 600.0), (-1, 180.0), ('60', 180.0),
                                (True, 180.0), (None, 180.0), (10 ** 9, 180.0)):
            self.settings(stall_seconds=value)
            self.assertEqual(s.oxi_stall_seconds(), expected, value)
        (s.PROJECT / 'runtime-settings.json').write_text('{broken')
        self.assertEqual(s.oxi_stall_seconds(), 180.0)


class Watchdog(StudioCase):
    def test_silent_oxi_is_stopped_and_fails_as_oxi_failure(self):
        s = self.studio
        self.settings(stall_seconds=0.5)
        started = time.monotonic()
        with self.assertRaises(s.OxiFailure) as caught:
            s.run_engine([str(s.OXI_BIN), 'generate'], {}, None, PROMPT, prompt_flag='--cond-prompt-file')
        self.assertLess(time.monotonic() - started, 10)
        self.assertNotIsInstance(caught.exception, s.OxiNpuRecovery)
        self.assertIn('stalled', str(caught.exception))
        self.assertNotIn(PROMPT, str(caught.exception))
        self.assertTrue(gone(self.stub_pid()), 'the stalled engine process survived')

    def test_sigterm_ignoring_engine_is_killed_after_the_grace(self):
        s = self.studio
        self.settings(stall_seconds=0.5)
        self.stalling_oxi(ignore_term=True)
        started = time.monotonic()
        with self.assertRaises(s.OxiFailure):
            s.run_engine([str(s.OXI_BIN), 'generate'], {})
        self.assertLess(time.monotonic() - started, 10)
        self.assertTrue(gone(self.stub_pid()))

    def test_silent_probe_sd_cli_is_stopped(self):
        s = self.studio
        self.settings(stall_seconds=0.5)
        with self.assertRaises(s.OxiFailure) as caught:
            s.run_engine([str(s.SD_PROBE_CLI), '--x'], {}, None, PROMPT, stop=s.handover_stop())
        self.assertIn('sd-cli (probe)', str(caught.exception))
        self.assertTrue(gone(self.stub_pid()))

    def test_steady_progress_longer_than_the_limit_is_not_stopped(self):
        s = self.studio
        self.settings(stall_seconds=0.6)
        write_script(s.OXI_BIN, STEADY % {'lines': 20})       # ~2 s of CR progress, never 0.6 s silent
        job = {}
        self.assertEqual(s.run_engine([str(s.OXI_BIN), 'generate'], job), 0)
        self.assertEqual(job.get('state'), 'generating 20/20')

    def test_zero_disables_the_watchdog(self):
        s = self.studio
        self.settings(stall_seconds=0)
        self.stalling_oxi(sleep=1.5)
        self.assertEqual(s.run_engine([str(s.OXI_BIN), 'generate'], {}), 0)

    def test_plain_sd_cli_is_never_watched(self):
        s = self.studio
        self.settings(stall_seconds=0.3)
        write_script(s.SDCLI, STALL % {'pidfile': str(self.pidfile), 'ignore_term': False, 'sleep': 1.5})
        self.assertEqual(s.run_engine([str(s.SDCLI), '-o', str(self.root / 'x.png')], {}), 0)

    def test_stalled_npu_worker_is_recovery_never_a_cpu_retry(self):
        s = self.studio
        self.settings(stall_seconds=0.5)
        job = {'npu_requested': True, 'npu_profile': '/owned/profile.json'}
        permit = unittest.mock.Mock()
        permit.pass_fds.return_value = ()
        permit.fileno.return_value = 5
        permit.connection.fileno.return_value = 6
        s.IMAGE_LEASES[id(job)] = permit
        self.addCleanup(s.IMAGE_LEASES.pop, id(job), None)
        real_popen = subprocess.Popen
        seen = {}

        def popen(command, **options):
            seen['command'] = command
            options.pop('pass_fds', None)
            return real_popen([str(s.OXI_BIN)], **options)   # the worker stands in as a silent stub
        with patch.object(s.subprocess, 'Popen', side_effect=popen):
            with self.assertRaises(s.OxiNpuRecovery) as caught:
                s.run_engine([str(s.OXI_BIN), 'generate', '--width', '512'], job, None, PROMPT,
                             prompt_flag='--cond-prompt-file')
        self.assertTrue(seen['command'][2].endswith('image_npu_worker.py'))
        self.assertIn('NPU worker stalled', str(caught.exception))
        self.assertIn('refusing a CPU retry', str(caught.exception))
        self.assertTrue(gone(self.stub_pid()))


class AutoFallback(StudioCase):
    def run_plan(self, **body):
        s = self.studio
        plan = s.plan_generation({'prompt': PROMPT, 'model': 'klein-q8', 'width': 512, 'height': 512, **body})
        self.assertEqual(plan['engine']['engine'], 'oxi')
        s.JOBS['job'] = {'state': 'queued'}
        s.run_pipeline('job', plan['steps'], open(self.root / 'lease', 'a'), plan['environment'],
                       plan['prompt'], plan['results'])
        return s.JOBS['job']

    def test_auto_finishes_on_sd_cpp_after_a_stall(self):
        self.settings(stall_seconds=0.5)
        job = self.run_plan(seed=7)
        self.assertEqual(job['state'], 'done', job)
        self.assertEqual(job['engine'], 'sd.cpp')
        self.assertIn('stalled', job['engine_note'])
        self.assertNotIn(PROMPT, json.dumps({k: v for k, v in job.items() if k != 'images'}))
        self.assertEqual(job['images'][0]['backend'], 'sd.cpp')
        self.assertTrue(gone(self.stub_pid()))

    def test_oxi_setting_reports_the_stall_without_sd_cpp(self):
        self.studio.save_engine_setting('oxi')
        self.settings(stall_seconds=0.5)
        job = self.run_plan()
        self.assertEqual(job['state'], 'error')
        self.assertIn('stalled', job['error'])
        self.assertNotIn(PROMPT, job['error'])


if __name__ == '__main__':
    unittest.main()
