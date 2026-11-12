"""S1: the reaper survives an exception, and cleanup() always releases the lease and state."""
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from snapdragon_ai_lab.runtime import controller


class ReaperCleanupTests(unittest.TestCase):
    def manager(self):
        m = controller.Manager()
        m.stop_event.set()
        m.thread.join(2)
        return m

    def test_reaper_keeps_running_after_a_failed_pass(self):
        m = self.manager()
        m.stop_event.clear()
        calls = []

        def reap_once():
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError('synthetic')
        m.reap_once = reap_once
        with patch('sys.stderr'):
            thread = threading.Thread(target=m.reaper, daemon=True)
            thread.start()
            deadline = time.monotonic() + 10
            while len(calls) < 2 and time.monotonic() < deadline:
                time.sleep(0.05)
            m.stop_event.set()
            thread.join(5)
        self.assertGreaterEqual(len(calls), 2, 'the reaper died after its first exception')

    def test_cleanup_releases_the_lease_when_a_guard_stop_raises(self):
        m = self.manager()
        with tempfile.TemporaryDirectory(dir='/tmp') as tmp:
            lease = (Path(tmp) / 'lease').open('a')
            guard = Mock()
            guard.stop.side_effect = RuntimeError('guard thread wedged')
            with m.mu:
                m.lease, m.state = lease, 'ready'
                m.loaded_config = {'model_alias': 'synthetic'}
                m.classic_memory_guard = guard
                with self.assertRaises(RuntimeError):
                    m.cleanup()
            self.assertTrue(lease.closed)
            self.assertIsNone(m.lease)
            self.assertEqual(m.state, 'unloaded')
            self.assertIsNone(m.loaded_config)

    def test_cleanup_releases_the_lease_when_the_first_step_raises(self):
        # Codex review: an exception BEFORE the old finally (here the very first read) must still
        # release the lease and state, and the later teardown steps must still run.
        m = self.manager()
        with tempfile.TemporaryDirectory(dir='/tmp') as tmp:
            lease = (Path(tmp) / 'lease').open('a')
            runtime = Mock()
            runtime.guard_failure.side_effect = RuntimeError('status read failed')
            classic = Mock()
            classic.failure.return_value = None
            with m.mu:
                m.lease, m.state = lease, 'ready'
                m.loaded_config = {'model_alias': 'synthetic'}
                m.backend_runtime = runtime
                m.flash_power_guard = classic
                with self.assertRaisesRegex(RuntimeError, 'status read failed'):
                    m.cleanup()
            self.assertTrue(lease.closed)
            self.assertIsNone(m.lease)
            self.assertEqual(m.state, 'unloaded')
            runtime.stop_guard.assert_called_once()
            classic.stop.assert_called_once()

    def test_cleanup_kills_the_engine_before_releasing_the_lease_when_stopping_raises(self):
        m = self.manager()
        child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], start_new_session=True)
        self.addCleanup(lambda: child.poll() is None and (child.kill(), child.wait()))
        real_killpg = controller.os.killpg
        calls = []

        def killpg(pid, sig):
            calls.append(sig)
            if len(calls) == 1:
                raise RuntimeError('signal delivery failed')
            return real_killpg(pid, sig)
        with tempfile.TemporaryDirectory(dir='/tmp') as tmp:
            lease = (Path(tmp) / 'lease').open('a')
            with m.mu, patch.object(controller.os, 'killpg', side_effect=killpg):
                m.lease, m.state, m.process = lease, 'ready', child
                m.loaded_config = {'model_alias': 'synthetic'}
                with self.assertRaisesRegex(RuntimeError, 'signal delivery failed'):
                    m.cleanup()
            self.assertIsNotNone(child.poll(), 'the engine must be dead before the lease is released')
            self.assertTrue(lease.closed)
            self.assertIsNone(m.process)
            self.assertEqual(m.state, 'unloaded')

    def unconfirmed_case(self, killpg):
        """Codex proof (audit2_stop_contract_proof.py): a fake engine that never exits."""
        m = self.manager()
        fake = Mock(pid=99_999_999, _local_ai_api_key_file=None, _local_ai_engine_death_logged=True)
        fake.poll.return_value = None
        fake.wait.side_effect = subprocess.TimeoutExpired('synthetic-engine', 5)
        tmp = tempfile.TemporaryDirectory(dir='/tmp')
        self.addCleanup(tmp.cleanup)
        lease = (Path(tmp.name) / 'lease').open('a+')
        self.addCleanup(lease.close)
        with m.mu:
            m.lease, m.process, m.state = lease, fake, 'ready'
            m.loaded_config = {'model_alias': 'synthetic'}
        with patch.object(controller.os, 'killpg', side_effect=killpg), patch('sys.stderr'):
            with m.mu, self.assertRaises(Exception):
                m.cleanup()
        return m, fake, lease

    def assert_ownership_kept(self, m, fake, lease):
        self.assertFalse(lease.closed, 'lease released while the engine may still run')
        self.assertIs(m.lease, lease)
        self.assertIs(m.process, fake, 'controller forgot a live engine')
        self.assertEqual(m.state, 'stopping')
        self.assertIsNotNone(m.stop_pending)
        with patch.object(controller.os, 'killpg', side_effect=PermissionError('still')), patch('sys.stderr'):
            with self.assertRaisesRegex(controller.Unavailable, 'has not exited yet'):
                m.begin(allow_start=True, model_alias='synthetic')
        self.assertFalse(lease.closed)

    def test_failed_signal_keeps_the_lease_and_the_process(self):
        self.assert_ownership_kept(*self.unconfirmed_case(PermissionError('synthetic cannot stop')))

    def test_sigkill_with_wait_timeout_keeps_the_lease_and_the_process(self):
        m, fake, lease = self.unconfirmed_case(lambda pid, sig: None)
        self.assert_ownership_kept(m, fake, lease)
        # Once the engine is really gone, the reaper's retry releases everything.
        fake.poll.return_value = -9
        fake.wait.side_effect = None
        with patch.object(controller.os, 'killpg'), patch('sys.stderr'):
            m.reap_once()
        self.assertTrue(lease.closed)
        self.assertIsNone(m.process)
        self.assertEqual(m.state, 'unloaded')
        self.assertIsNone(m.stop_pending)

    def test_stuck_engine_waits_are_bounded_and_retries_skip_the_grace(self):
        # Codex review 3: an engine in D-state ignores SIGKILL for a while; no wait may be unbounded.
        m = self.manager()
        waits, signals = [], []

        def wait(timeout=None):
            waits.append(timeout)
            if timeout is None:
                raise AssertionError('unbounded wait on a stuck engine')
            raise subprocess.TimeoutExpired('stuck-engine', timeout)
        fake = Mock(pid=99_999_998, _local_ai_api_key_file=None, _local_ai_engine_death_logged=True)
        fake.poll.return_value = None
        fake.wait.side_effect = wait
        tmp = tempfile.TemporaryDirectory(dir='/tmp')
        self.addCleanup(tmp.cleanup)
        lease = (Path(tmp.name) / 'lease').open('a+')
        self.addCleanup(lease.close)
        with m.mu:
            m.lease, m.process, m.state = lease, fake, 'ready'
            m.loaded_config = {'model_alias': 'synthetic'}
        with patch.object(controller.os, 'killpg', side_effect=lambda pid, sig: signals.append(sig)), \
                patch('sys.stderr'):
            with m.mu, self.assertRaises(Exception):
                m.cleanup()
            self.assertNotIn(None, waits)
            self.assertLessEqual(sum(waits), controller.TERM_WAIT_SECONDS + 2 * controller.KILL_WAIT_SECONDS)
            self.assertEqual(m.state, 'stopping')
            self.assertFalse(lease.closed)
            # The reaper's retry sends SIGKILL only (no new SIGTERM grace) and stays bounded.
            waits.clear(); signals.clear()
            with self.assertRaises(subprocess.TimeoutExpired):  # the reaper loop logs and retries
                m.reap_once()
        self.assertNotIn(None, waits)
        self.assertLessEqual(sum(waits), 2 * controller.KILL_WAIT_SECONDS)
        self.assertNotIn(controller.signal.SIGTERM, signals)
        self.assertIn(controller.signal.SIGKILL, signals)
        self.assertEqual(m.state, 'stopping')

    def test_status_answers_stopping_promptly_while_a_retry_holds_the_lock(self):
        m = self.manager()
        m.stop_pending = 'engine pid 1 has not exited'
        holding, release = threading.Event(), threading.Event()

        def hold():
            with m.mu:
                holding.set()
                release.wait(5)
        holder = threading.Thread(target=hold, daemon=True)
        holder.start()
        self.assertTrue(holding.wait(2))
        try:
            started = time.monotonic()
            status = m.status()
            self.assertLess(time.monotonic() - started, 0.5)
            self.assertEqual(status['state'], 'stopping')
            self.assertEqual(status['stop_pending'], 'engine pid 1 has not exited')
        finally:
            release.set(); holder.join(2)
            m.stop_pending = None


if __name__ == '__main__':
    unittest.main()
