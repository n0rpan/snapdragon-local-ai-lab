import os as _os, tempfile as _tempfile
_os.environ.setdefault('LOCAL_AI_LEDGER_DIR', _tempfile.mkdtemp(prefix='local-ai-test-ledger-'))  # never the real request ledger
import argparse
from contextlib import contextmanager, redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

MODULE = Path(__file__).resolve().parents[1] / 'tools/run_synthetic_screen.py'
spec = importlib.util.spec_from_file_location('screen', MODULE)
screen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(screen)

MERGE = '''def merge_intervals(intervals):
    result = []
    for start, end in sorted(intervals):
        if result and start <= result[-1][1]:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([start, end])
    return result
'''
PORT = '''def parse_port(value):
    if not isinstance(value, str):
        raise ValueError()
    value = value.strip()
    if not value or any(c < '0' or c > '9' for c in value):
        raise ValueError()
    port = int(value)
    if not 1 <= port <= 65535:
        raise ValueError()
    return port
'''
ANNOTATED_PORT = PORT.replace('def parse_port(value):', 'def parse_port(value: str) -> int:')
ITER_MERGE = '''def merge_intervals(intervals: list[list[int]]) -> list[list[int]]:
    stream = iter(sorted(intervals))
    first = next(stream, None)
    if first is None:
        return []
    result = [list(first)]
    for start, end in stream:
        if start <= result[-1][1]:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([start, end])
    return result
'''


@contextmanager
def server(answers, status=200, extra=None, drip=False):
    class Handler(BaseHTTPRequestHandler):
        calls = []

        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            self.calls.append((self.path, body, self.headers.get('Authorization')))
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            if status == 302:
                self.send_header('Location', 'http://127.0.0.1:1/private')
            obj = {'choices': [{'message': {'content': answers[min(len(self.calls)-1, len(answers)-1)]}}],
                   'usage': {'prompt_tokens': 10, 'completion_tokens': 20,
                             'secret_text': 'must-not-be-exported', 'total_tokens': float('nan')}}
            if extra:
                obj.update(extra)
            data = json.dumps(obj).encode()
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            try:
                if drip:
                    for byte in data:
                        self.wfile.write(bytes([byte]))
                        self.wfile.flush()
                        time.sleep(.02)
                else:
                    self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass
    httpd = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{httpd.server_port}/v1', Handler
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=2)


class ScreenTests(unittest.TestCase):
    def test_no_run_does_not_touch_network_or_subprocess(self):
        with patch.object(screen, 'run_screen', side_effect=AssertionError('must not run')):
            with redirect_stdout(io.StringIO()):
                self.assertEqual(screen.main([]), 0)

    def test_only_exact_loopback_endpoint(self):
        for bad in ('https://127.0.0.1:80/v1', 'http://localhost:80/v1',
                    'http://127.0.0.2:80/v1', 'http://example.org/v1',
                    'http://user:key@127.0.0.1:80/v1', 'http://127.0.0.1:80/v1?x=1',
                    'http://127.0.0.1:80/v1#x', 'http://127.0.0.1:80/api'):
            with self.subTest(endpoint=bad), self.assertRaises(ValueError):
                screen.endpoint_parts(bad)

    def test_redirect_never_followed(self):
        with server([MERGE], status=302) as (endpoint, handler):
            with self.assertRaises(ValueError):
                screen.request(endpoint, {}, 2)
            self.assertEqual(len(handler.calls), 1)

    def test_numeric_metadata_only_and_key_explicit(self):
        with server([MERGE]) as (endpoint, handler):
            code, meta = screen.request(endpoint, {}, 2, 'synthetic-key')
            self.assertEqual(code, MERGE)
            self.assertEqual(meta['usage'], {'prompt_tokens': 10, 'completion_tokens': 20})
            self.assertEqual(handler.calls[0][2], 'Bearer synthetic-key')
            self.assertNotIn('synthetic-key', json.dumps(meta))
            self.assertNotIn('must-not-be-exported', json.dumps(meta))

    def test_response_and_source_restrictions(self):
        with server(['x' * (screen.MAX_RESPONSE + 1)]) as (endpoint, _):
            with self.assertRaises(ValueError):
                screen.request(endpoint, {}, 2)
        for code in ('import os\n', 'print(1)\n',
                     'def parse_port(x=print(1)): return 1',
                     'def parse_port(x): return x.__class__',
                     'def parse_port(x): return open("/etc/passwd").read()',
                     'def parse_port(x):\n global a\n return 1',
                     'def parse_port(x): raise SystemExit(0)'):
            with self.subTest(code=code), self.assertRaises(ValueError):
                screen.validate_code(code, 'parse_port')

    def test_trickling_response_has_absolute_deadline(self):
        with server([PORT], drip=True) as (endpoint, _):
            started = time.monotonic()
            with self.assertRaises((TimeoutError, ValueError, OSError)):
                screen.request(endpoint, {}, .15)
            self.assertLess(time.monotonic() - started, 1)

    def test_restricted_code_never_starts_a_checker(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(screen.subprocess, 'Popen') as start:
            with self.assertRaises(ValueError):
                screen.check_code('import os', {'function': 'parse_port'}, Path(tmp))
            start.assert_not_called()

    def test_safe_annotations_and_unsafe_annotation_calls(self):
        for annotation in ('str', 'list[int]', 'tuple[int, ...]', 'int | None',
                           'None', '"ForwardReference"'):
            screen.validate_code(f'def parse_port(value: {annotation}) -> int:\n return 1', 'parse_port')
        for annotation in ('str()', 'list[int()]', 'str.upper', '__import__("os")'):
            with self.subTest(annotation=annotation), self.assertRaises(screen.UnsupportedSource):
                screen.validate_code(f'def parse_port(value: {annotation}):\n return 1', 'parse_port')

    @unittest.skipUnless(shutil.which('bwrap'), 'Linux bwrap required')
    def test_actual_offline_checker_correct_and_incorrect(self):
        cases = json.loads(screen.FIXTURE.read_text())['cases']
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for number, code in enumerate((MERGE, PORT, 'def parse_port(value): return 80')):
                directory = root / str(number)
                directory.mkdir()
                case = cases[0 if number == 0 else 1]
                result = screen.check_code(code, case, directory)
                self.assertEqual(result['passed'], number < 2)
                self.assertEqual(result['checker_status'], 'passed' if number < 2 else 'assertion_failed')

    @unittest.skipUnless(shutil.which('bwrap'), 'Linux bwrap required')
    def test_annotated_and_iterator_solutions_pass_actual_assertions(self):
        cases = json.loads(screen.FIXTURE.read_text())['cases']
        with tempfile.TemporaryDirectory() as tmp:
            for number, code in enumerate((ITER_MERGE, ANNOTATED_PORT)):
                directory = Path(tmp) / str(number)
                directory.mkdir()
                self.assertTrue(screen.check_code(code, cases[number], directory)['passed'])

    @unittest.skipUnless(shutil.which('bwrap'), 'Linux bwrap required')
    def test_single_python_fence_is_normalized_before_unchanged_oracle(self):
        case=json.loads(screen.FIXTURE.read_text())['cases'][1]
        with tempfile.TemporaryDirectory() as tmp:
            for index,code in enumerate((PORT,'def parse_port(value): return 80')):
                directory=Path(tmp)/str(index);directory.mkdir()
                result=screen.check_code('```python\n'+code+'\n```',case,directory)
                self.assertEqual(result['passed'],index==0)
                self.assertEqual(result['source_format'],'python_fence')
        with self.assertRaises(SyntaxError):
            screen.validate_code(screen.normalize_code('Here is the answer:\n```python\n'+PORT+'\n```')[0],'parse_port')

    @unittest.skipUnless(shutil.which('bwrap'), 'Linux bwrap required')
    def test_single_underscore_helpers_and_locals_pass_actual_assertions(self):
        code = '''def _ascii_digits(value: str) -> bool:
    return bool(value) and all('0' <= _char <= '9' for _char in value)

def parse_port(value: str) -> int:
    if not isinstance(value, str):
        raise ValueError()
    value = value.strip()
    if not _ascii_digits(value):
        raise ValueError()
    _port = int(value)
    if not 1 <= _port <= 65535:
        raise ValueError()
    return _port
'''
        case = json.loads(screen.FIXTURE.read_text())['cases'][1]
        with tempfile.TemporaryDirectory() as tmp:
            self.assertTrue(screen.check_code(code, case, Path(tmp))['passed'])

    @unittest.skipUnless(shutil.which('bwrap'), 'Linux bwrap required')
    def test_checker_restriction_is_separate_from_wrong_assertion(self):
        with server(['def merge_intervals(x): return getattr(x, "value")',
                     'def parse_port(value): return 80']) as (endpoint, _), tempfile.TemporaryDirectory() as tmp:
            args = argparse.Namespace(endpoint=endpoint, model='synthetic-local', key_env=None,
                output_parent=Path(tmp), max_tokens=256, seed=42, timeout_seconds=2)
            _, summary = screen.run_screen(args)
            self.assertEqual([r['checker_status'] for r in summary['rows']],
                             ['unsupported_source', 'assertion_failed'])
            self.assertFalse(summary['passed'])

    @unittest.skipUnless(shutil.which('bwrap'), 'Linux bwrap required')
    def test_sandbox_mounts_hide_home_network_and_protect_sources(self):
        # Trusted probe only, never model output: exercise the actual sandbox contract.
        probe = '''import os, socket
assert 'SYNTHETIC_HOST_SECRET' not in os.environ
assert not os.path.exists('/home')
for name in ('candidate.py', 'assertions.py', 'checker.py'):
 try: open('/work/' + name, 'w')
 except OSError: pass
 else: raise AssertionError('writable protected source')
s = socket.socket()
s.settimeout(.2)
assert s.connect_ex(('127.0.0.1', PORT)) != 0
print('ISOLATED')
'''
        with server([PORT]) as (endpoint, _), tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            port, _ = screen.endpoint_parts(endpoint)
            for name, text in [('candidate.py', PORT), ('assertions.py', ''),
                               ('checker.py', probe.replace('PORT', str(port)))]:
                screen.private_write(root / name, text.encode())
            with patch.dict(os.environ, {'SYNTHETIC_HOST_SECRET': 'private'}):
                result = screen.subprocess.run(screen.sandbox_argv(root/'candidate.py', root/'assertions.py', root/'checker.py'),
                    env={}, stdin=screen.subprocess.DEVNULL, capture_output=True, timeout=5)
            self.assertEqual(result.returncode, 0, result.stderr.decode())
            self.assertEqual(result.stdout, b'ISOLATED\n')

    @unittest.skipUnless(shutil.which('bwrap'), 'Linux bwrap required')
    def test_two_serial_requests_private_evidence_and_safe_summary(self):
        with server([MERGE, PORT]) as (endpoint, handler), tempfile.TemporaryDirectory() as tmp:
            args = argparse.Namespace(endpoint=endpoint, model='synthetic-local', key_env=None,
                output_parent=Path(tmp), max_tokens=256, seed=42, timeout_seconds=2)
            output, summary = screen.run_screen(args)
            self.assertTrue(summary['passed'])
            self.assertEqual(len(handler.calls), 2)
            self.assertFalse(summary['agent_qualified'])
            self.assertFalse(summary['ram_recovery_verified'])
            self.assertEqual(output.stat().st_mode & 0o777, 0o700)
            for file in output.rglob('*'):
                if file.is_file():
                    self.assertEqual(file.stat().st_mode & 0o777, 0o600)
            text = (output/'summary.json').read_text()
            self.assertNotIn(MERGE, text)
            self.assertNotIn('must-not-be-exported', text)


if __name__ == '__main__':
    unittest.main()
