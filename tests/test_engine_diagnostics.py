import os
import stat
import tempfile
import time
import unittest
from unittest.mock import patch
from pathlib import Path

from snapdragon_ai_lab.runtime.engine_diagnostics import EngineLogWriter, EngineStderrReader


class EngineLogWriterTests(unittest.TestCase):
    def test_private_log_and_bounded_rotation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log_dir = root / "diagnostics"
            log = log_dir / "engine.log"
            writer = EngineLogWriter(log, max_bytes=10, backups=2)
            writer.write("one\n")
            writer.write("two\n")
            writer.write("three\n")
            writer.write("four\n")

            self.assertIsNone(writer.error)
            self.assertEqual(stat.S_IMODE(log_dir.stat().st_mode), 0o700)
            self.assertEqual(stat.S_IMODE(log.stat().st_mode), 0o600)
            self.assertEqual(log.read_bytes(), b"four\n")
            self.assertEqual((log.with_name("engine.log.1")).read_bytes(), b"three\n")
            self.assertEqual((log.with_name("engine.log.2")).read_bytes(), b"one\ntwo\n")

    def test_sanitizes_controls_and_caps_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "engine.log"
            writer = EngineLogWriter(log)
            writer.write(b"bad\x1b\x00\xff\n" + b"x" * 100_000)
            data = log.read_bytes()
            self.assertIn(b"bad\\x1b\\x00\xef\xbf\xbd\n", data)
            self.assertLessEqual(len(data), 64 * 1024 + 32)

    def test_large_records_are_capped_before_rotation(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "engine.log"
            writer = EngineLogWriter(log, max_bytes=16 * 1024, backups=1)
            large_record = "diagnostic " + ("x" * (1024 * 1024)) + "\n"
            writer.write(large_record)
            first_size = log.stat().st_size
            writer.write(large_record)

            self.assertIsNone(writer.error)
            self.assertLessEqual(first_size, 16 * 1024)
            self.assertLessEqual(log.stat().st_size, 16 * 1024)
            self.assertIn(b"...[input truncated]", log.with_name("engine.log.1").read_bytes())

    def test_rejects_symlink_parent_and_keeps_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            real = root / "real"
            real.mkdir()
            link = root / "link"
            link.symlink_to(real, target_is_directory=True)
            log = link / "engine.log"
            writer = EngineLogWriter(log)
            writer.write("diagnostic\n")
            self.assertIsNotNone(writer.error)
            self.assertFalse((real / "engine.log").exists())

    def test_unsafe_disk_error_does_not_escape_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "engine.log"
            log.symlink_to(Path(tmp) / "elsewhere")
            writer = EngineLogWriter(log)
            writer.write("kept safe\n")
            self.assertIsNotNone(writer.error)

    def test_rejects_hard_link_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "target"
            target.write_bytes(b"untouched")
            log = Path(tmp) / "engine.log"
            os.link(target, log)
            writer = EngineLogWriter(log)
            writer.write("not appended\n")
            self.assertIsNotNone(writer.error)
            self.assertEqual(target.read_bytes(), b"untouched")


class EngineStderrReaderTests(unittest.TestCase):
    def test_stop_after_thread_start_failure_closes_stream(self):
        with tempfile.TemporaryDirectory() as tmp:
            writer = EngineLogWriter(Path(tmp) / 'engine.log')
            read_fd, write_fd = os.pipe()
            stream = os.fdopen(read_fd, 'rb', buffering=0)
            reader = EngineStderrReader(stream, writer)
            try:
                with patch('threading.Thread.start', side_effect=RuntimeError('synthetic thread failure')):
                    with self.assertRaisesRegex(RuntimeError, 'synthetic thread failure'):
                        reader.start()
                self.assertTrue(reader.stop(timeout=0.1))
                self.assertTrue(stream.closed)
            finally:
                os.close(write_fd)

    def test_pipe_is_drained_sanitized_and_closed_at_eof(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "engine.log"
            writer = EngineLogWriter(log)
            read_fd, write_fd = os.pipe()
            stream = os.fdopen(read_fd, "rb", buffering=0)
            reader = EngineStderrReader(stream, writer)
            reader.start()
            os.write(write_fd, b"hello\x1b\n" + b"z" * 30_000 + b"\npartial")
            os.close(write_fd)

            self.assertTrue(reader.stop(timeout=2.0))
            logged = log.read_bytes()
            self.assertIn(b"hello\\x1b\n", logged)
            self.assertIn(b"...[line truncated]\n", logged)
            self.assertTrue(logged.endswith(b"partial\n"))
            self.assertTrue(stream.closed)

    def test_stop_without_pipe_data_is_bounded(self):
        with tempfile.TemporaryDirectory() as tmp:
            writer = EngineLogWriter(Path(tmp) / "engine.log")
            read_fd, write_fd = os.pipe()
            stream = os.fdopen(read_fd, "rb", buffering=0)
            reader = EngineStderrReader(stream, writer)
            reader.start()
            started = time.monotonic()
            self.assertTrue(reader.stop(timeout=0.8))
            self.assertLess(time.monotonic() - started, 1.0)
            self.assertTrue(stream.closed)
            os.close(write_fd)


if __name__ == "__main__":
    unittest.main()
