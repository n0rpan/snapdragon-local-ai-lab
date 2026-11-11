"""Model-free, root-free tests for packaging/flash-qos/local-ai-flash-qos.

The helper runs as a subprocess against a temp fake sysfs and procfs via the
test-only environment overrides. The unit and polkit rule are checked as text
only; they are not exercised here.
"""

import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FLASH_QOS_DIR = REPO / 'packaging' / 'flash-qos'
HELPER = FLASH_QOS_DIR / 'local-ai-flash-qos'
UNIT = FLASH_QOS_DIR / 'local-ai-flash-qos@.service'
RULE = FLASH_QOS_DIR / '49-local-ai-flash-qos.rules'
PID = 424242


def setUpModule():
    if hasattr(os, 'geteuid') and os.geteuid() == 0:
        # The helper ignores the test overrides as root and would touch the real sysfs.
        raise unittest.SkipTest('flash-qos helper tests must not run as root')


class _HelperCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix='flashqos-test-')
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.sysroot = os.path.join(self.tmp, 'sys')
        self.procroot = os.path.join(self.tmp, 'proc')
        os.makedirs(self.procroot)
        self.knob = os.path.join(self.sysroot, 'cpu8', 'power', 'pm_qos_resume_latency_us')
        self.env = dict(os.environ)
        self.env['LOCAL_AI_FLASH_QOS_SYSFS_ROOT'] = self.sysroot
        self.env['LOCAL_AI_FLASH_QOS_PROC_ROOT'] = self.procroot

    def write_knob(self, text='0\n'):
        os.makedirs(os.path.dirname(self.knob), exist_ok=True)
        with open(self.knob, 'w') as f:
            f.write(text)

    def read_knob(self):
        with open(self.knob) as f:
            return f.read()

    def make_proc(self, pid=PID, comm='oxe-server\n'):
        pdir = os.path.join(self.procroot, str(pid))
        os.mkdir(pdir)
        with open(os.path.join(pdir, 'comm'), 'w') as f:
            f.write(comm)
        return pdir

    def make_copy(self, uid):
        src = HELPER.read_text()
        needle = 'ALLOWED_UID = None'
        self.assertEqual(src.count(needle), 1)
        path = os.path.join(self.tmp, 'bin', 'local-ai-flash-qos')
        os.makedirs(os.path.dirname(path))
        with open(path, 'w') as f:
            f.write(src.replace(needle, 'ALLOWED_UID = %d' % uid))
        return path

    def run_helper(self, *args, helper=None):
        cmd = [sys.executable, '-I', str(helper or HELPER), *args]
        return subprocess.run(cmd, env=self.env, capture_output=True, text=True, timeout=20)

    def start_hold(self, helper, pid=PID):
        proc = subprocess.Popen(
            [sys.executable, '-I', helper, 'hold', str(pid)],
            env=self.env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        def reap():
            if proc.poll() is None:
                proc.kill()
                proc.communicate()
        self.addCleanup(reap)
        return proc

    def wait_for_knob(self, expected, timeout=5.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.read_knob().strip() == expected:
                return True
            time.sleep(0.05)
        return False


class TestBasicCommands(_HelperCase):
    def test_status_prints_value(self):
        self.write_knob('0\n')
        res = self.run_helper('status')
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertEqual(res.stdout.strip(), '0')

    def test_reset_clears_to_zero(self):
        self.write_knob('100\n')
        res = self.run_helper('reset')
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertEqual(self.read_knob().strip(), '0')

    def test_reset_when_already_zero(self):
        self.write_knob('0\n')
        res = self.run_helper('reset')
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertEqual(self.read_knob().strip(), '0')


class TestHold(_HelperCase):
    def test_hold_then_process_vanishes_releases(self):
        helper = self.make_copy(os.getuid())
        pdir = self.make_proc()
        self.write_knob('0\n')
        proc = self.start_hold(helper)
        self.assertTrue(self.wait_for_knob('100'), 'knob never reached 100')
        shutil.rmtree(pdir)
        try:
            out, err = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            self.fail('hold did not exit within 5 s after the engine pid vanished')
        self.assertEqual(proc.returncode, 0, err)
        self.assertEqual(self.read_knob().strip(), '0')
        self.assertIn('hold pid=%d cpu=8 latency_us=100' % PID, out)
        self.assertIn('released', out)

    def test_sigterm_while_holding_clears_and_exits_zero(self):
        helper = self.make_copy(os.getuid())
        self.make_proc()
        self.write_knob('0\n')
        proc = self.start_hold(helper)
        self.assertTrue(self.wait_for_knob('100'), 'knob never reached 100')
        proc.send_signal(signal.SIGTERM)
        try:
            out, err = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            self.fail('hold did not exit within 5 s after SIGTERM')
        self.assertEqual(proc.returncode, 0, err)
        self.assertEqual(self.read_knob().strip(), '0')

    def test_wrong_comm_exits_3_and_leaves_knob(self):
        helper = self.make_copy(os.getuid())
        self.make_proc(comm='bash\n')
        self.write_knob('0\n')
        res = self.run_helper('hold', str(PID), helper=helper)
        self.assertEqual(res.returncode, 3, res.stdout + res.stderr)
        self.assertEqual(res.stderr.count('\n'), 1)
        self.assertEqual(self.read_knob(), '0\n')

    def test_wrong_owner_exits_3_and_leaves_knob(self):
        helper = self.make_copy(os.getuid() + 1)
        self.make_proc()
        self.write_knob('0\n')
        res = self.run_helper('hold', str(PID), helper=helper)
        self.assertEqual(res.returncode, 3, res.stdout + res.stderr)
        self.assertEqual(self.read_knob(), '0\n')

    def test_unconfigured_source_refuses_hold_without_writing(self):
        self.make_proc()
        self.write_knob('0\n')
        result = self.run_helper('hold', str(PID))
        self.assertEqual(result.returncode, 3, result.stderr)
        self.assertIn('no installing user configured', result.stderr)
        self.assertEqual(self.read_knob(), '0\n')

    def test_missing_knob_exits_5_for_hold_and_reset(self):
        helper = self.make_copy(os.getuid())
        self.make_proc()
        res = self.run_helper('hold', str(PID), helper=helper)
        self.assertEqual(res.returncode, 5, res.stderr)
        self.assertFalse(os.path.exists(self.knob))
        res = self.run_helper('reset')
        self.assertEqual(res.returncode, 5, res.stderr)
        self.assertFalse(os.path.exists(self.knob))


class TestBadArguments(_HelperCase):
    def test_bad_pid_arguments_exit_2_and_leave_knob(self):
        helper = self.make_copy(os.getuid())
        self.make_proc()
        self.write_knob('0\n')
        for bad in ('0', 'abc', '12 ; rm', '-1', ''):
            with self.subTest(pid=bad):
                res = self.run_helper('hold', bad, helper=helper)
                self.assertEqual(res.returncode, 2, res.stderr)
                self.assertEqual(self.read_knob(), '0\n')

    def test_unknown_command_exits_2(self):
        self.write_knob('0\n')
        res = self.run_helper('frobnicate')
        self.assertEqual(res.returncode, 2)
        self.assertIn('usage', res.stderr)


class TestUnitAndRule(unittest.TestCase):
    def test_unit_has_exact_exec_lines(self):
        lines = UNIT.read_text().splitlines()
        self.assertIn('ExecStart=/usr/local/libexec/local-ai-flash-qos hold %i', lines)
        self.assertIn('ExecStopPost=/usr/local/libexec/local-ai-flash-qos reset', lines)

    def test_polkit_rule_scope(self):
        text = RULE.read_text()
        self.assertIn(r'/^local-ai-flash-qos@[1-9][0-9]{0,9}\.service$/', text)
        self.assertIn('var ALLOWED_USER = null;', text)
        self.assertIn('typeof ALLOWED_USER == "string"', text)
        self.assertIn('action.lookup("unit")', text)
        self.assertIn('action.lookup("verb")', text)
        verbs = re.findall(r'lookup\("verb"\)\s*==\s*"([a-z-]+)"', text)
        self.assertEqual(sorted(verbs), ['start', 'stop'])
        action_ids = set(re.findall(r'org\.[A-Za-z0-9_.\-]+', text))
        self.assertEqual(action_ids, {'org.freedesktop.systemd1.manage-units'})


class TestSourceStatic(unittest.TestCase):
    def test_helper_source_constraints(self):
        src = HELPER.read_text()
        self.assertIn('pm_qos_resume_latency_us', src)
        for banned in ('subprocess', 'os.system', 'shell=True'):
            self.assertNotIn(banned, src)
        self.assertIn('os.geteuid() != 0', src)
        guard = src.index('os.geteuid() != 0')
        self.assertEqual(src.count('os.environ'), 2)
        for name in ('LOCAL_AI_FLASH_QOS_SYSFS_ROOT', 'LOCAL_AI_FLASH_QOS_PROC_ROOT'):
            self.assertIn(name, src)
            self.assertGreater(src.index(name), guard, name + ' used before the root guard')


if __name__ == '__main__':
    unittest.main()
