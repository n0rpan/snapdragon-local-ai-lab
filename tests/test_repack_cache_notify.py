import os as _os, tempfile as _tempfile
_os.environ.setdefault('LOCAL_AI_LEDGER_DIR', _tempfile.mkdtemp(prefix='local-ai-test-ledger-'))  # never the real request ledger
import io, json, os, stat, tempfile, threading, unittest
from email.message import Message
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from snapdragon_ai_lab.runtime import controller

MODEL_SHA = 'a' * 64
ENGINE_SHA = 'b' * 64


class BlockingStream:
    """Fake child stderr: yields lines, then blocks until closed (EOF)."""
    def __init__(self, lines=()):
        self.lines = list(lines)
        self.eof = threading.Event()
        self.closed = False
    def readline(self, limit=-1):
        if self.lines: return self.lines.pop(0)
        self.eof.wait(10)
        return b''
    def close(self): self.closed = True


def read_lines(path):
    return Path(path).read_text().splitlines()


class ReaderPrivacy(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.log = Path(self.tmp.name) / 'state' / 'local-ai' / 'repack-cache-errors.log'
        self.notify = Mock()

    def tearDown(self): self.tmp.cleanup()

    def run_reader(self, data):
        reader = controller.RepackErrorReader(io.BytesIO(data), log_path=self.log, label='qwen-local', notify=self.notify)
        reader.run()
        return reader

    def test_only_error_lines_reach_log(self):
        data = (b'srv  log: prompt="SECRET PROMPT ggml-repack-cache: ERROR fake"\n'
                b'ggml-repack-cache: INFO hit key=abc path=/x msg="SECRET OUTPUT"\n'
                b'ggml-repack-cache: ERROR open-failed key=0123456789abcdef path=/c/f errno=13\n'
                b'  ggml-repack-cache: ERROR indented SECRET\n'
                b'user: SECRET PROMPT\n')
        self.run_reader(data)
        text = self.log.read_text()
        self.assertNotIn('SECRET', text)
        lines = read_lines(self.log)
        self.assertEqual(len(lines), 1)
        self.assertTrue(lines[0].endswith('qwen-local ggml-repack-cache: ERROR open-failed key=0123456789abcdef path=/c/f errno=13'))

    def test_optional_tee_gets_raw_bounded_chunks(self):
        writer = Mock()
        data = b'user output SECRET\nggml-repack-cache: ERROR bad\n'
        reader = controller.RepackErrorReader(io.BytesIO(data), log_path=self.log,
            label='qwen-local', notify=self.notify, tee_writer=writer)
        reader.run()
        self.assertEqual(b''.join(call.args[0] for call in writer.write.call_args_list), data)
        self.assertNotIn('SECRET', self.log.read_text())

    def test_continuation_of_long_line_is_not_treated_as_line_start(self):
        filler = b'prompt ' + b'x' * (8192 - len(b'prompt '))
        self.run_reader(filler + b'ggml-repack-cache: ERROR SECRET\n')
        self.assertFalse(self.log.exists())
        self.notify.assert_not_called()

    def test_no_errors_means_no_log_and_no_notification(self):
        self.run_reader(b'loading model\nggml-repack-cache: INFO miss key=a path=b\n')
        self.assertFalse(self.log.exists())
        self.notify.assert_not_called()

    def test_line_length_and_count_are_bounded(self):
        long_line = b'ggml-repack-cache: ERROR ' + b'z' * 5000 + b'\n'
        self.run_reader(long_line * 100)
        lines = read_lines(self.log)
        self.assertEqual(len(lines), controller.REPACK_ERROR_LINES_PER_LOAD)
        for line in lines:
            message = line.split(' ', 2)[2]
            self.assertLessEqual(len(message), controller.REPACK_ERROR_LINE_CHARS)

    def test_control_characters_are_neutralised(self):
        self.run_reader(b'ggml-repack-cache: ERROR bad\x1b[31m path=\x00x\n')
        self.assertNotIn('\x1b', self.log.read_text())
        self.assertNotIn('\x00', self.log.read_text())

    def test_notify_once_per_load(self):
        self.run_reader(b'ggml-repack-cache: ERROR a\nggml-repack-cache: ERROR b\nggml-repack-cache: ERROR c\n')
        self.notify.assert_called_once_with(self.log)
        self.assertEqual(len(read_lines(self.log)), 3)
        # A new load gets its own (single) notification.
        self.run_reader(b'ggml-repack-cache: ERROR d\n')
        self.assertEqual(self.notify.call_count, 2)

    def test_permissions_owner_only(self):
        self.log.parent.mkdir(parents=True, mode=0o755)
        os.chmod(self.log.parent, 0o755)
        self.run_reader(b'ggml-repack-cache: ERROR a\n')
        self.assertEqual(stat.S_IMODE(os.stat(self.log.parent).st_mode), 0o700)
        self.assertEqual(stat.S_IMODE(os.stat(self.log).st_mode), 0o600)

    def test_existing_log_is_tightened(self):
        self.log.parent.mkdir(parents=True)
        self.log.write_text('old\n'); os.chmod(self.log, 0o644)
        self.run_reader(b'ggml-repack-cache: ERROR a\n')
        self.assertEqual(stat.S_IMODE(os.stat(self.log).st_mode), 0o600)

    def test_rotation_keeps_bounded_old_files(self):
        with patch.object(controller, 'REPACK_ERROR_LOG_BYTES', 300):
            for index in range(40):
                controller.write_repack_error(self.log, f'line {index:03d} ' + 'e' * 40 + '\n')
        names = sorted(p.name for p in self.log.parent.iterdir())
        self.assertEqual(names, ['repack-cache-errors.log', 'repack-cache-errors.log.1', 'repack-cache-errors.log.2'])
        for path in self.log.parent.iterdir():
            self.assertLessEqual(path.stat().st_size, 300)
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
        self.assertIn('line 039', self.log.read_text())

    def test_missing_notify_send_never_fails(self):
        with patch.object(controller.shutil, 'which', return_value=None), patch.object(controller.subprocess, 'run') as run:
            controller.notify_repack_errors(self.log)
            run.assert_not_called()
        with patch.object(controller.shutil, 'which', return_value='/usr/bin/notify-send'), \
                patch.object(controller.subprocess, 'run', side_effect=OSError) as run:
            controller.notify_repack_errors(self.log)
            run.assert_called_once()
            self.assertIn(str(self.log), run.call_args.args[0][-1])


class ControllerIntegration(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.model = root / 'model.gguf'; self.model.write_bytes(b'gguf')
        self.config_path = root / 'runtime.json'
        self.lease = root / 'generation.lock'
        self.log = root / 'state' / 'repack-cache-errors.log'
        self.engine_log = root / 'state' / 'engine-stderr.log'
        self.patches = [patch.object(controller, 'CONFIG', self.config_path),
                        patch.object(controller, 'LEASE', self.lease),
                        patch.object(controller, 'REPACK_ERROR_LOG', self.log),
                        patch.object(controller, 'ENGINE_DIAGNOSTIC_LOG', self.engine_log),
                        patch.object(controller, 'START_SECONDS', 0),
                        patch.object(controller, 'available_memory_bytes', return_value=None),
                        patch.object(controller.memory_policy, 'MemoryGuard')]
        for item in self.patches: item.start()
        self.engine_reader_starts = []
        self.engine_reader_patch = patch.object(controller, 'EngineStderrReader', side_effect=self.make_engine_reader)
        self.engine_reader_patch.start()
        with patch.object(controller.threading.Thread, 'start'):  # no idle reaper thread
            self.manager = controller.Manager()

    def make_engine_reader(self, stream, writer):
        reader = Mock()
        reader.start.return_value = reader
        reader.stop.return_value = True
        self.engine_reader_starts.append((stream, writer, reader))
        return reader

    def tearDown(self):
        self.engine_reader_patch.stop()
        for item in reversed(self.patches): item.stop()
        self.tmp.cleanup()

    def write_config(self, environment, **extra):
        self.config_path.write_text(json.dumps({'model_alias': 'qwen-base', 'ready': True,
            'model_path': str(self.model), 'server_binary': '/nonexistent/llama-server',
            'environment': environment, **extra}))

    def launch(self, stream=None):
        """Start a load with a fake native process; the load then times out (START_SECONDS=0)."""
        process = Mock(pid=424242, stderr=stream)
        process.poll.return_value = None
        with patch.object(controller.subprocess, 'Popen', return_value=process) as popen, \
                patch.object(controller.os, 'killpg', side_effect=lambda *a: stream and stream.eof.set()):
            with self.assertRaises(controller.Unavailable):
                self.manager.begin(allow_start=True)
        return popen

    def test_cache_off_uses_diagnostic_reader_and_no_identity(self):
        self.write_config({'LLAMA_CPU_SMALL_BATCH_LIMIT': '4'}, sha256=MODEL_SHA, server_sha256=ENGINE_SHA)
        popen = self.launch()
        kwargs = popen.call_args.kwargs
        self.assertIs(kwargs['stderr'], controller.subprocess.PIPE)
        self.assertIs(kwargs['stdout'], controller.subprocess.DEVNULL)
        self.assertNotIn('GGML_CPU_REPACK_CACHE_MODEL_SHA256', kwargs['env'])
        self.assertNotIn('GGML_CPU_REPACK_CACHE_ENGINE_ID', kwargs['env'])
        self.assertIsNone(self.manager.repack_reader)
        self.assertEqual(len(self.engine_reader_starts), 1)
        self.assertIsNone(self.manager.engine_reader)

    def test_status_poll_does_not_create_engine_log(self):
        self.assertIsNone(self.manager.engine_log_writer)
        with patch.object(self.manager, 'config', return_value={}):
            self.manager.status()
        self.assertFalse(self.engine_log.parent.exists())

    def test_cache_on_injects_identity_and_reader_is_joined_on_cleanup(self):
        self.write_config({'GGML_CPU_REPACK_CACHE': '/private/repack', 'GGML_CPU_REPACK_CACHE_ENGINE_ID': 'stale'},
                          sha256=MODEL_SHA, server_sha256=ENGINE_SHA)
        stream = BlockingStream([b'user prompt SECRET\n', b'ggml-repack-cache: ERROR write-failed key=k path=p\n'])
        with patch.object(controller, 'notify_repack_errors') as notify:
            popen = self.launch(stream)
        kwargs = popen.call_args.kwargs
        self.assertIs(kwargs['stderr'], controller.subprocess.PIPE)
        self.assertIs(kwargs['stdout'], controller.subprocess.DEVNULL)
        self.assertEqual(kwargs['env']['GGML_CPU_REPACK_CACHE_MODEL_SHA256'], MODEL_SHA)
        self.assertEqual(kwargs['env']['GGML_CPU_REPACK_CACHE_ENGINE_ID'], ENGINE_SHA)
        # begin() failed and cleaned up: the reader saw EOF and was joined.
        self.assertIsNone(self.manager.repack_reader)
        self.assertTrue(stream.closed)
        self.assertFalse(any(t.name == 'repack-cache-stderr' and t.is_alive() for t in threading.enumerate()))
        self.assertNotIn('SECRET', self.log.read_text())
        self.assertIn('SECRET', self.engine_log.read_text())
        self.assertEqual(len(read_lines(self.log)), 1)
        notify.assert_called_once_with(self.log)

    def test_cache_on_without_exact_identity_loads_without_cache(self):
        for extra in ({'sha256': MODEL_SHA}, {'sha256': 'A' * 64, 'server_sha256': ENGINE_SHA}):
            with self.subTest(extra=extra):
                self.log.unlink(missing_ok=True)
                self.write_config({'GGML_CPU_REPACK_CACHE': '/private/repack', 'GGML_CPU_REPACK_CACHE_VERIFY': '1',
                                   'GGML_CPU_REPACK_CACHE_ENGINE_ID': 'stale'}, **extra)
                with patch.object(controller, 'notify_repack_errors') as notify, \
                        patch.dict(controller.os.environ, {'GGML_CPU_REPACK_CACHE_MODEL_SHA256': 'c' * 64}):
                    popen = self.launch()
                popen.assert_called_once()
                kwargs = popen.call_args.kwargs
                self.assertIs(kwargs['stderr'], controller.subprocess.PIPE)
                self.assertFalse([key for key in kwargs['env'] if key.startswith('GGML_CPU_REPACK_CACHE')])
                self.assertIsNone(self.manager.repack_reader)
                lines = read_lines(self.log)
                self.assertEqual(len(lines), 1)
                self.assertTrue(lines[0].endswith('local-ai: ERROR repack-cache-disabled missing-identity alias=qwen-base'))
                self.assertEqual(stat.S_IMODE(os.stat(self.log).st_mode), 0o600)
                notify.assert_called_once_with(self.log)

    def test_native_death_event_is_structured_and_idempotent(self):
        self.manager.engine_log_writer = controller.EngineLogWriter(self.engine_log)
        process = SimpleNamespace(pid=7654, poll=lambda: -9)
        config = {'model_alias': 'qwen-base', 'server_sha256': ENGINE_SHA}
        self.assertTrue(self.manager.record_engine_death(process, config, 'memory pressure'))
        self.assertFalse(self.manager.record_engine_death(process, config))
        records = [json.loads(line) for line in self.engine_log.read_text().splitlines()]
        self.assertEqual(records, [{'event': 'engine_died', 'pid': 7654, 'exit_code': -9,
            'model_alias': 'qwen-base', 'engine_sha256': ENGINE_SHA, 'guard_failure': 'memory pressure'}])

    def test_cleanup_records_child_death_before_clearing_snapshot(self):
        self.manager.engine_log_writer = controller.EngineLogWriter(self.engine_log)
        process = SimpleNamespace(pid=7655, poll=lambda: -9, wait=lambda: -9)
        config = {'model_alias': 'qwen-base', 'server_sha256': ENGINE_SHA}
        self.manager.process = process
        self.manager.loaded_config = config
        self.manager.cleanup()
        event = json.loads(self.engine_log.read_text())
        self.assertEqual((event['event'], event['pid'], event['model_alias']),
                         ('engine_died', 7655, 'qwen-base'))
        self.assertIsNone(self.manager.process)

    def test_native_death_before_headers_returns_502(self):
        self.manager.engine_log_writer = controller.EngineLogWriter(self.engine_log)
        process = SimpleNamespace(pid=8765, poll=lambda: -11)
        config = {'model_alias': 'qwen-base', 'server_sha256': ENGINE_SHA,
                  'backend_kind': 'llama'}
        process._local_ai_engine_config = config
        self.manager.process = process
        self.manager.loaded_config = config
        self.manager.config = lambda alias=None: config
        self.manager.begin_queued = lambda **kwargs: process
        self.manager.prepare_prefix = lambda *args: None
        self.manager.end = lambda **kwargs: None
        handler = object.__new__(controller.Handler)
        handler.server = SimpleNamespace(manager=self.manager, server_port=8092)
        handler.connection = object()
        handler.headers = Message()
        handler.headers['Content-Length'] = '2'
        handler.headers['Host'] = '127.0.0.1:8092'  # F80 Host allowlist
        handler.command = 'POST'
        handler.path = '/completion'
        handler.rfile = io.BytesIO(b'{}')
        handler.close_connection = False
        replies = []
        handler.json_reply = lambda code, value: replies.append((code, value))

        class BrokenBackend:
            sock = object()
            def __init__(self, *args, **kwargs): pass
            def request(self, *args, **kwargs): pass
            def getresponse(self): raise ConnectionResetError('synthetic backend reset')
            def close(self): pass

        class WatcherStub:
            def __init__(self, client, upstream):
                self.disconnected = threading.Event()
            def start(self): return self
            def stop(self): pass

        with patch.object(controller.http.client, 'HTTPConnection', BrokenBackend), \
                patch.object(controller, 'ClientDisconnect', WatcherStub):
            handler.forward()
        self.assertEqual(replies[0][0], 502)
        events = [json.loads(line) for line in self.engine_log.read_text().splitlines()]
        self.assertEqual([event['event'] for event in events], ['engine_died', 'request_end'])
        self.assertEqual(events[-1]['error'], 'backend_died')

if __name__ == '__main__':
    unittest.main()
