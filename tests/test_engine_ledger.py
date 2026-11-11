"""No models: exercise the inherited pipe and bounded ledger storage."""
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from snapdragon_ai_lab.runtime import engine_ledger as ledger


def config():
    return {'backend_kind': 'oxe', 'model_alias': 'ornith-test',
            'server_sha256': 'a' * 64, 'environment': {ledger.ENABLE_ENV: '1'}}


class LedgerTests(unittest.TestCase):
    def test_controller_keeps_reader_handle_when_start_returns_none(self):
        from snapdragon_ai_lab.runtime import controller
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); model = root / 'model.gguf'; model.write_bytes(b'x' * 100)
            profile = {**config(), 'environment': {}, 'ready': True, 'model_path': str(model),
                       'server_binary': str(root / 'fake-server'), 'sha256': 'b' * 64,
                       'expected_bytes': 100, 'threads': 8, 'context_size': 32768,
                       'prefix_cache': {'enabled': False}, 'oxe_args': ['--model', '{model}']}
            process = Mock(pid=999999); process.poll.return_value = None
            guard = Mock(); guard.failure.return_value = None
            response = Mock(status=200); response.read.return_value = b'{"status":"ok"}'
            connection = Mock(); connection.getresponse.return_value = response
            reader = Mock(); reader.start.return_value = None
            manager = controller.Manager(); manager.stop_event.set()
            try:
                with patch.dict(os.environ, {'LOCAL_AI_ENABLE_OXE': '1'}), \
                        patch.object(manager, 'config', return_value=profile), \
                        patch.object(controller, 'LEASE', root / 'lease'), \
                        patch.object(controller, 'ENGINE_DIAGNOSTIC_LOG', root / 'private/stderr.log'), \
                        patch.object(controller, 'available_memory_bytes', return_value=None), \
                        patch.object(controller.oxe_runtime, 'verify_engine'), \
                        patch.object(controller.memory_policy, 'MemoryGuard', return_value=guard), \
                        patch.object(controller.subprocess, 'Popen', return_value=process), \
                        patch.object(controller.http.client, 'HTTPConnection', return_value=connection), \
                        patch.object(controller, 'EngineStderrReader', return_value=reader), \
                        patch.object(controller.os, 'killpg'):
                    manager.begin(allow_start=True)
                    self.assertIs(manager.engine_reader, reader)
                    manager.cleanup()
                    reader.stop.assert_called_once_with(timeout=5)
            finally:
                manager.close()

    def test_controller_passes_only_the_ledger_fd_and_cleans_up_failed_launch(self):
        from snapdragon_ai_lab.runtime import controller
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); model = root / 'model.gguf'; model.write_bytes(b'x' * 100)
            profile = {**config(), 'ready': True, 'model_path': str(model),
                       'server_binary': str(root / 'fake-server'), 'sha256': 'b' * 64,
                       'expected_bytes': 100, 'threads': 8, 'context_size': 32768,
                       'prefix_cache': {'enabled': False}, 'oxe_args': ['--model', '{model}']}
            seen = {}
            def stop_before_launch(argv, **kwargs):
                seen.update(argv=argv, **kwargs)
                fd, = kwargs['pass_fds']
                os.fstat(fd)
                self.assertEqual(argv[-2:], ['--ledger', f'/proc/self/fd/{fd}'])
                raise RuntimeError('synthetic launch failure')
            manager = controller.Manager(); manager.stop_event.set()
            try:
                with patch.dict(os.environ, {'LOCAL_AI_ENABLE_OXE': '1'}), \
                        patch.object(manager, 'config', return_value=profile), \
                        patch.object(controller, 'LEASE', root / 'lease'), \
                        patch.object(controller, 'ENGINE_DIAGNOSTIC_LOG', root / 'private/stderr.log'), \
                        patch.object(controller, 'ENGINE_LEDGER_LOG', root / 'private/ledger.jsonl'), \
                        patch.object(controller, 'available_memory_bytes', return_value=None), \
                        patch.object(controller.oxe_runtime, 'verify_engine'), \
                        patch.object(controller.memory_policy, 'MemoryGuard', return_value=Mock()), \
                        patch.object(controller.subprocess, 'Popen', side_effect=stop_before_launch):
                    with self.assertRaisesRegex(RuntimeError, 'synthetic launch failure'):
                        manager.begin(allow_start=True)
                self.assertIsNone(manager.engine_ledger)
                self.assertIsNone(manager.lease)
                for fd in seen['pass_fds']:
                    with self.assertRaises(OSError): os.fstat(fd)
            finally:
                manager.close()

    def test_opt_in_is_per_profile_and_polling_has_no_side_effects(self):
        with patch.dict(os.environ, {ledger.ENABLE_ENV: '1'}), \
                patch.object(ledger.os, 'pipe', side_effect=AssertionError('pipe created')):
            p = config(); p['environment'] = {}
            self.assertFalse(ledger.enabled(p, ['server']))
            p = config(); p['backend_kind'] = 'llama'
            self.assertFalse(ledger.enabled(p, ['server']))
            self.assertTrue(ledger.enabled(config(), ['server']))
            for args in [['server', '--ledger', '/explicit/log'], ['server', '--ledger=/explicit/log']]:
                self.assertFalse(ledger.enabled(config(), args))

    def test_child_opens_inherited_descriptor_and_reader_rotates_private_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / 'private/ledger.jsonl'
            bridge = ledger.NativeLedger(log, config())
            bridge.writer.max_bytes = 500
            child = None
            try:
                # Same open flags as Rust OpenOptions(create + append + mode0600).
                code = ('import os,sys,json; '
                        'f=os.open(sys.argv[1],os.O_WRONLY|os.O_CREAT|os.O_APPEND,0o600); '
                        '[os.write(f,(json.dumps({"n_prompt":64,"predicted_n":i,"prompt_ms":1.5,'
                        '"cache_src":"live","stop":"eos","prompt":"DO NOT LOG ME"})+"\\n").encode()) for i in range(20)]; '
                        'os.close(f)')
                child = subprocess.Popen([sys.executable, '-c', code, bridge.arguments[1]],
                                         pass_fds=bridge.pass_fds)
                bridge.start()
                self.assertEqual(child.wait(timeout=5), 0)
                self.assertTrue(bridge.stop(timeout=2))
                self.assertEqual(bridge.records, 20)
                self.assertEqual(bridge.dropped, 0)
                files = list(log.parent.glob('ledger.jsonl*'))
                self.assertLessEqual(len(files), 3)
                self.assertEqual(stat.S_IMODE(log.parent.stat().st_mode), 0o700)
                for path in files:
                    self.assertLessEqual(path.stat().st_size, 500)
                    self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
                    self.assertNotIn('DO NOT LOG ME', path.read_text())
                    for line in path.read_text().splitlines():
                        row = json.loads(line)
                        self.assertEqual(row['model_alias'], 'ornith-test')
                        self.assertEqual(row['engine_sha256'], 'a' * 64)
                        self.assertNotIn('prompt', row)
            finally:
                if child is not None and child.poll() is None:
                    child.kill(); child.wait()
                bridge.stop(timeout=2)

    def test_failed_launch_closes_both_pipe_ends_without_a_reader_thread(self):
        with tempfile.TemporaryDirectory() as tmp:
            bridge = ledger.NativeLedger(Path(tmp) / 'private/ledger.jsonl', config())
            read_fd = bridge.reader.stream.fileno(); write_fd = bridge.write_fd
            self.assertTrue(bridge.stop())
            self.assertTrue(bridge.stop())
            for fd in [read_fd, write_fd]:
                with self.assertRaises(OSError):
                    os.fstat(fd)

    def test_non_numeric_or_malformed_records_are_discarded(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / 'private/ledger.jsonl'
            bridge = ledger.NativeLedger(log, config())
            try:
                for line in ['not json', '[]', '{"prompt":"secret"}',
                             '{"n_prompt":1,"predicted_n":1,"prompt_ms":NaN}']:
                    bridge.write(line)
                self.assertEqual(bridge.dropped, 4)
                self.assertFalse(log.exists())
                bridge.write('{"n_prompt":1,"predicted_n":1,"prompt_ms":2,"stop":"secret user content"}')
                row = json.loads(log.read_text())
                self.assertNotIn('stop', row)
            finally:
                bridge.stop()

    def test_unsafe_directory_is_refused_before_pipe_creation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / 'real').mkdir(); (root / 'link').symlink_to(root / 'real')
            with patch.object(ledger.os, 'pipe', side_effect=AssertionError('pipe created')):
                with self.assertRaisesRegex(OSError, 'unavailable'):
                    ledger.NativeLedger(root / 'link/ledger.jsonl', config())


if __name__ == '__main__':
    unittest.main()
