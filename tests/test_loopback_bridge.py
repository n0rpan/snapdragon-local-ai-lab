import sys
"""Loopback-only bridge tests; all endpoints are local temporary fixtures."""
import importlib.util
from pathlib import Path
import socket
import tempfile
import threading
import time
import unittest

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
from _isolation import socket_tmpdir  # noqa: E402  (N7: short AF_UNIX paths)

SOURCE = Path(__file__).resolve().parents[1] / "benchmarks/classic/loopback_bridge.py"
SPEC = importlib.util.spec_from_file_location("loopback_bridge", SOURCE)
bridge = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bridge)


def tcp_echo(listener):
    conn, _ = listener.accept()
    with conn:
        while True:
            data = conn.recv(65536)
            if not data:
                conn.sendall(b"EOF-ACK")
                return
            conn.sendall(data)


class LoopbackBridgeTests(unittest.TestCase):
    def setUp(self):
        bridge.STOP.clear()

    def tearDown(self):
        bridge.STOP.set()

    def test_host_bridge_large_duplex_stream_and_eof_half_close(self):
        with tempfile.TemporaryDirectory(dir=socket_tmpdir()) as tmp:
            directory = Path(tmp)
            directory.chmod(0o700)
            path = directory / "controller.sock"
            captures = directory / "requests"
            captures.mkdir(mode=0o700)
            target = socket.socket()
            target.bind(("127.0.0.1", 0))
            target.listen(1)
            port = target.getsockname()[1]
            server_thread = threading.Thread(target=tcp_echo, args=(target,), daemon=True)
            server_thread.start()
            bridge_thread = threading.Thread(target=bridge.host_mode,
                    args=(path, "127.0.0.1", port, captures), daemon=True)
            bridge_thread.start()
            # The socket file appears at bind(), before the bridge's chmod and listen(): wait
            # until it accepts a connection (a loaded CI host can lose that race otherwise).
            client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            client.settimeout(5)
            deadline = time.monotonic() + 5
            while True:
                try:
                    client.connect(str(path))
                    break
                except (FileNotFoundError, ConnectionRefusedError):
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(0.01)
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            payload = bytes(range(256)) * 8192
            received = bytearray()

            def send_then_half_close():
                client.sendall(payload)
                client.shutdown(socket.SHUT_WR)

            sender = threading.Thread(target=send_then_half_close, daemon=True)
            sender.start()
            while len(received) < len(payload) + len(b"EOF-ACK"):
                chunk = client.recv(65536)
                if not chunk:
                    break
                received.extend(chunk)
            sender.join(5)
            client.close()
            self.assertFalse(sender.is_alive())
            self.assertEqual(bytes(received), payload + b"EOF-ACK")
            bridge.STOP.set()
            bridge_thread.join(2)
            target.close()
            server_thread.join(2)
            self.assertFalse(bridge_thread.is_alive())
            self.assertFalse(path.exists())
            files = list(captures.glob('request-*.http'))
            self.assertEqual(len(files), 1)
            self.assertEqual(files[0].stat().st_mode & 0o777, 0o600)
            self.assertEqual(files[0].read_bytes(), payload)

    def test_capture_requires_a_private_regular_directory(self):
        with tempfile.TemporaryDirectory(dir=socket_tmpdir()) as tmp:
            directory = Path(tmp)
            captures = directory / 'requests'
            captures.mkdir(mode=0o755)
            with self.assertRaisesRegex(ValueError, 'owner-only'):
                bridge.host_mode(directory/'controller.sock', '127.0.0.1', 8000, captures)
            captures.chmod(0o700)
            link = directory/'linked'
            link.symlink_to(captures, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, 'owner-only'):
                bridge.host_mode(directory/'controller.sock', '127.0.0.1', 8000, link)

    def test_rejects_non_loopback_target(self):
        with self.assertRaisesRegex(ValueError, "127.0.0.1"):
            bridge.destination("192.0.2.1", 8000)

    def test_never_replaces_existing_socket_path(self):
        with tempfile.TemporaryDirectory(dir=socket_tmpdir()) as tmp:
            directory = Path(tmp)
            directory.chmod(0o700)
            path = directory / "occupied"
            path.write_text("keep me")
            with self.assertRaises(FileExistsError):
                bridge.host_listener(path)
            self.assertEqual(path.read_text(), "keep me")

    def test_inside_listener_binds_localhost_and_forwards_to_private_unix_socket(self):
        with tempfile.TemporaryDirectory(dir=socket_tmpdir()) as tmp:
            directory = Path(tmp)
            directory.chmod(0o700)
            path = directory / "controller.sock"
            unix_listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            unix_listener.bind(str(path))
            path.chmod(0o600)
            unix_listener.listen(1)
            server_thread = threading.Thread(target=tcp_echo, args=(unix_listener,), daemon=True)
            # tcp_echo is transport agnostic and consumes one accepted stream.
            server_thread.start()
            reservation = socket.socket()
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
            reservation.close()
            inside_thread = threading.Thread(target=bridge.inside_mode, args=(path, port), daemon=True)
            inside_thread.start()
            # Starting the thread does not mean listen() has run yet. Retry the
            # actual client connection; a separate readiness probe would consume
            # the fixture's single accepted stream.
            deadline = time.monotonic() + 2
            while True:
                try:
                    client = socket.create_connection(("127.0.0.1", port), timeout=2)
                    break
                except ConnectionRefusedError:
                    if not inside_thread.is_alive() or time.monotonic() >= deadline:
                        raise
                    time.sleep(0.01)
            client.settimeout(2)
            client.sendall(b"request")
            client.shutdown(socket.SHUT_WR)
            received = bytearray()
            while True:
                chunk = client.recv(1024)
                if not chunk:
                    break
                received.extend(chunk)
            client.close()
            self.assertEqual(bytes(received), b"requestEOF-ACK")
            bridge.STOP.set()
            inside_thread.join(2)
            unix_listener.close()
            server_thread.join(2)
            self.assertFalse(inside_thread.is_alive())


if __name__ == "__main__":
    unittest.main()
