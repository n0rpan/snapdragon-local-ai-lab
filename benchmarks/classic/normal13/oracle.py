#!/usr/bin/env python3
"""Explicit, local-only two-request screen; generated code runs only in bwrap."""
import argparse
import ast
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import re
import selectors
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from urllib.parse import urlsplit

FIXTURE = Path(__file__).resolve().parents[1] / 'benchmarks/fixtures/code-screen-v1.json'
MAX_RESPONSE = 262144
MAX_CODE = 65536
MAX_STDIO = 16384
USAGE_KEYS = ('prompt_tokens', 'completion_tokens', 'total_tokens')
TIMING_KEYS = ('prompt_n', 'prompt_ms', 'prompt_per_second', 'predicted_n',
               'predicted_ms', 'predicted_per_second')
SAFE_BUILTINS = ('abs', 'all', 'any', 'bool', 'dict', 'enumerate', 'float', 'int',
                 'isinstance', 'len', 'list', 'max', 'min', 'range', 'reversed',
                 'set', 'sorted', 'str', 'sum', 'tuple', 'zip', 'iter', 'next',
                 'filter', 'map', 'slice', 'divmod', 'pow', 'round', 'ord', 'chr',
                 'bytes', 'frozenset', 'StopIteration', 'ValueError', 'TypeError')
ANNOTATION_NAMES = {'int', 'str', 'float', 'bool', 'list', 'tuple', 'dict', 'set',
                    'frozenset', 'bytes', 'object'}


class UnsupportedSource(ValueError):
    """A checker policy restriction, distinct from an incorrect assertion."""


def safe_annotation(node):
    """Only builtins, literal forward references, unions and builtin containers."""
    if isinstance(node, ast.Name):
        return node.id in ANNOTATION_NAMES
    if isinstance(node, ast.Constant):
        # Forward-reference strings are inert; never evaluate their contents.
        return node.value is None or type(node.value) is str or node.value is Ellipsis
    if isinstance(node, ast.Subscript):
        return isinstance(node.value, ast.Name) and node.value.id in {
            'list', 'tuple', 'dict', 'set', 'frozenset'} and safe_annotation(node.slice)
    if isinstance(node, ast.Tuple):
        return all(safe_annotation(x) for x in node.elts)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return safe_annotation(node.left) and safe_annotation(node.right)
    return False


def endpoint_parts(endpoint):
    u = urlsplit(endpoint)
    if (u.scheme != 'http' or u.hostname != '127.0.0.1' or u.username is not None
            or u.password is not None or u.query or u.fragment or u.path not in ('/v1', '/v1/')
            or not u.port or not 1 <= u.port <= 65535):
        raise ValueError('endpoint must be http://127.0.0.1:PORT/v1')
    return u.port, '/v1/chat/completions'


def numeric_fields(value, keys):
    if not isinstance(value, dict):
        return {}
    return {k: value[k] for k in keys if type(value.get(k)) in (int, float)
            and math.isfinite(value[k]) and 0 <= value[k] <= 10**15}


def private_write(path, data):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as f:
        f.write(data)


def request(endpoint, payload, timeout, key=None):
    port, path = endpoint_parts(endpoint)
    headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
    if key is not None:
        if not key or len(key) > 4096 or '\r' in key or '\n' in key:
            raise ValueError('invalid explicit key')
        headers['Authorization'] = 'Bearer ' + key
    conn = http.client.HTTPConnection('127.0.0.1', port, timeout=timeout)
    started = time.monotonic()
    sockets = []

    def expire():
        # Shutdown interrupts a slow/trickling read even when per-read timeouts reset.
        for owned in sockets:
            try:
                owned.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    timer = threading.Timer(timeout, expire)
    timer.daemon = True
    timer.start()
    try:
        conn.connect()
        sockets.append(conn.sock)
        remaining = timeout - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError('request deadline')
        conn.sock.settimeout(remaining)
        conn.request('POST', path, json.dumps(payload).encode(), headers)
        response = conn.getresponse()
        # http.client does not follow redirects; never report arbitrary response text.
        if response.status != 200:
            raise ValueError('HTTP response was not 200')
        if response.getheader('Content-Type', '').split(';')[0].strip().lower() != 'application/json':
            raise ValueError('response was not JSON')
        length = response.getheader('Content-Length')
        if length is not None and (not length.isdecimal() or int(length) > MAX_RESPONSE):
            raise ValueError('response size exceeds bound')
        chunks, size = [], 0
        while True:
            if response.fp is None:
                break
            remaining = timeout - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError('request deadline')
            # read1 returns available bytes rather than waiting to fill a large read.
            response.fp.raw._sock.settimeout(remaining)
            chunk = response.read1(min(8192, MAX_RESPONSE + 1 - size))
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
            if size > MAX_RESPONSE:
                raise ValueError('response size exceeds bound')
        obj = json.loads(b''.join(chunks))
        if time.monotonic() - started >= timeout:
            raise TimeoutError('request deadline')
        choices = obj.get('choices')
        if not isinstance(choices, list) or len(choices) != 1:
            raise ValueError('expected exactly one choice')
        message = choices[0].get('message', {})
        code = message.get('content')
        if not isinstance(code, str) or not code.strip() or len(code.encode()) > MAX_CODE or message.get('tool_calls'):
            raise ValueError('expected bounded code text')
        return code, {'request_seconds': time.monotonic() - started,
                      'usage': numeric_fields(obj.get('usage'), USAGE_KEYS),
                      'server_timings': numeric_fields(obj.get('timings'), TIMING_KEYS)}
    finally:
        conn.close()
        timer.cancel()
        timer.join(timeout=1)


def normalize_code(code):
    text=code.strip()
    fenced=re.fullmatch(r'```(?:python|py)?[ \t]*\r?\n(.*?)\r?\n```',text,re.DOTALL|re.IGNORECASE)
    return (fenced.group(1),'python_fence') if fenced else (text,'plain_python')


def validate_code(code, function):
    tree = ast.parse(code)
    if not tree.body or not all(isinstance(x, ast.FunctionDef) for x in tree.body):
        raise UnsupportedSource('only plain function definitions are accepted')
    if function not in [x.name for x in tree.body]:
        raise ValueError('required function missing')
    allowed_names = set(SAFE_BUILTINS)
    banned = {'eval', 'exec', 'compile', 'open', 'input', 'globals', 'locals', 'vars',
              'getattr', 'setattr', 'delattr', 'type', 'dir', 'help', 'breakpoint',
              'exit', 'quit', 'SystemExit', 'BaseException'}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal, ast.ClassDef,
                             ast.AsyncFunctionDef, ast.Await, ast.Yield, ast.YieldFrom)):
            raise UnsupportedSource('unsupported code construct')
        if isinstance(node, ast.Name) and (node.id.startswith('__') or node.id in banned):
            raise UnsupportedSource('restricted name')
        if isinstance(node, ast.Attribute) and node.attr.startswith('_'):
            raise UnsupportedSource('restricted attribute')
        if isinstance(node, ast.FunctionDef):
            if node.name.startswith('__') or node.name in allowed_names or node.decorator_list or getattr(node, 'type_params', []):
                raise UnsupportedSource('restricted function definition')
            if node.returns and not safe_annotation(node.returns):
                raise UnsupportedSource('unsafe return annotation')
            args = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
            args += [x for x in (node.args.vararg, node.args.kwarg) if x]
            if any(a.annotation and not safe_annotation(a.annotation) for a in args):
                raise UnsupportedSource('unsafe argument annotation')
            for default in [*node.args.defaults, *[x for x in node.args.kw_defaults if x is not None]]:
                if not isinstance(default, ast.Constant):
                    raise UnsupportedSource('only literal defaults accepted')
    return tree


def checker_text():
    # This trusted program is mounted read-only and starts before generated code.
    return '''import ast, builtins, resource
resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
resource.setrlimit(resource.RLIMIT_AS, (268435456, 268435456))
resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
resource.setrlimit(resource.RLIMIT_NPROC, (16, 16))
source = open('/work/candidate.py').read()
assertions = open('/work/assertions.py').read()
scope = {'__builtins__': {n: getattr(builtins, n) for n in %r}}
# Validate outside, then strip approved annotations without evaluating strings,
# names or subscripts. This also avoids builtin-generic version differences.
tree = ast.parse(source)
for node in ast.walk(tree):
 if isinstance(node, ast.FunctionDef):
  node.returns = None
 if isinstance(node, ast.arg):
  node.annotation = None
try:
 exec(compile(tree, '/work/candidate.py', 'exec'), scope)
 exec(compile(assertions, '/work/assertions.py', 'exec'), scope)
except AssertionError:
 print('ASSERTION_FAILED')
 raise SystemExit(1)
except BaseException:
 print('RUNTIME_ERROR')
 raise SystemExit(2)
print('PASS')
''' % ((SAFE_BUILTINS + ('AssertionError',)),)


def sandbox_argv(code_path, assertions_path, checker_path):
    bwrap = shutil.which('bwrap')
    python = Path(sys.executable).resolve()
    if sys.platform != 'linux' or not bwrap or not python.is_relative_to('/usr'):
        raise RuntimeError('requires Linux, bwrap, and a Python interpreter under /usr')
    argv = [bwrap, '--unshare-all', '--die-with-parent', '--new-session', '--clearenv',
            '--cap-drop', 'ALL', '--ro-bind', '/usr', '/usr']
    for name in ('bin', 'lib', 'lib64'):
        p = Path('/') / name
        if p.is_symlink():
            argv += ['--symlink', os.readlink(p), str(p)]
        elif p.is_dir():
            argv += ['--ro-bind', str(p), str(p)]
    return argv + ['--proc', '/proc', '--dev', '/dev', '--tmpfs', '/tmp', '--dir', '/work',
                   '--ro-bind', str(code_path), '/work/candidate.py',
                   '--ro-bind', str(assertions_path), '/work/assertions.py',
                   '--ro-bind', str(checker_path), '/work/checker.py',
                   '--chdir', '/work', '--setenv', 'HOME', '/tmp', '--setenv', 'LANG', 'C.UTF-8',
                   str(python), '-I', '-B', '/work/checker.py']


def check_code(code, case, directory):
    code,source_format=normalize_code(code)
    validate_code(code, case['function'])
    code_path, assertions_path, checker_path = [directory / n for n in ('candidate.py', 'assertions.py', 'checker.py')]
    for path, text in ((code_path, code), (assertions_path, case['independent_test_body']), (checker_path, checker_text())):
        private_write(path, text.encode())
    started = time.monotonic()
    child = subprocess.Popen(sandbox_argv(code_path, assertions_path, checker_path),
                             stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             env={}, start_new_session=True)
    output = bytearray()
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(child.stdout, selectors.EVENT_READ)
            while selector.get_map():
                remaining = 5 - (time.monotonic() - started)
                if remaining <= 0:
                    raise TimeoutError('checker deadline')
                for key, _ in selector.select(min(remaining, .1)):
                    data = os.read(key.fd, 4096)
                    if not data:
                        selector.unregister(key.fileobj)
                    else:
                        output.extend(data)
                        if len(output) > MAX_STDIO:
                            raise ValueError('checker stdio limit')
        exit_code = child.wait(timeout=max(.01, 5 - (time.monotonic() - started)))
        passed = exit_code == 0 and output == b'PASS\n'
        status = ('passed' if passed else 'assertion_failed' if exit_code == 1 and output == b'ASSERTION_FAILED\n'
                  else 'runtime_error' if exit_code == 2 and output == b'RUNTIME_ERROR\n'
                  else 'runtime_or_sandbox_error')
        return {'passed': passed, 'checker_status': status,'source_format':source_format,
                'checker_exit': exit_code, 'checker_seconds': time.monotonic() - started}
    finally:
        if child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        child.wait(timeout=5)
        child.stdout.close()


def run_screen(args):
    endpoint_parts(args.endpoint)
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}', args.model):
        raise ValueError('invalid model alias')
    key = None
    if args.key_env:
        if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]{0,127}', args.key_env):
            raise ValueError('invalid key environment variable name')
        key = os.environ[args.key_env]
    fixture_bytes = FIXTURE.read_bytes()
    fixture = json.loads(fixture_bytes)
    if [c['function'] for c in fixture['cases']] != ['merge_intervals', 'parse_port']:
        raise ValueError('unexpected fixed fixture')
    # Check sandbox availability before sending any request.
    sandbox_argv(Path('/candidate'), Path('/assertions'), Path('/checker'))
    output = Path(tempfile.mkdtemp(prefix='synthetic-screen-', dir=args.output_parent))
    summary = {'schema_version': 1, 'workload': fixture['workload'], 'scope': 'two-request-screen-only',
               'agent_qualified': False, 'cold_load_measured_separately': False,
               'ram_recovery_verified': False, 'model': args.model,
               'fixture_sha256': hashlib.sha256(fixture_bytes).hexdigest(),
               'parameters': {'max_tokens': args.max_tokens, 'temperature': 0, 'seed': args.seed,
                              'request_timeout_seconds': args.timeout_seconds}, 'rows': []}
    for number, case in enumerate(fixture['cases']):
        directory = output / str(number + 1)
        directory.mkdir(mode=0o700)
        private_write(directory / 'request.json', json.dumps({'prompt': case['prompt']}).encode())
        row = {'id': case['id'], 'passed': False}
        try:
            code, metadata = request(args.endpoint, {'model': args.model,
                'messages': [{'role': 'user', 'content': case['prompt']}], 'stream': False,
                'max_tokens': args.max_tokens, 'temperature': 0, 'seed': args.seed}, args.timeout_seconds, key)
            private_write(directory / 'answer.txt', code.encode())
            row.update(metadata)
            row.update(check_code(code, case, directory))
        except Exception as error:
            row['error_kind'] = type(error).__name__
            if isinstance(error, UnsupportedSource):
                row['checker_status'] = 'unsupported_source'
            elif isinstance(error, SyntaxError):
                row['checker_status'] = 'invalid_source'
            else:
                row.setdefault('checker_status', 'request_or_checker_error')
        summary['rows'].append(row)
    summary['passed'] = all(r['passed'] for r in summary['rows'])
    private_write(output / 'summary.json', (json.dumps(summary, indent=2, allow_nan=False) + '\n').encode())
    return output, summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', action='store_true', help='explicitly send two requests and run offline checks')
    p.add_argument('--endpoint', default='http://127.0.0.1:8092/v1')
    p.add_argument('--model', default='qwen-local')
    p.add_argument('--key-env', help='explicit environment variable containing an optional API key')
    p.add_argument('--output-parent', type=Path, help='existing parent for a fresh private evidence directory')
    p.add_argument('--timeout-seconds', type=int, default=900, choices=range(1, 1801), metavar='1..1800')
    p.add_argument('--max-tokens', type=int, default=1024, choices=range(1, 2049), metavar='1..2048')
    p.add_argument('--seed', type=int, default=42, choices=range(0, 2**31), metavar='0..2147483647')
    args = p.parse_args(argv)
    if not args.run:
        p.print_help()
        return 0
    try:
        output, summary = run_screen(args)
    except Exception as error:
        print('screen setup failed: ' + type(error).__name__, file=sys.stderr)
        return 2
    print('Private evidence: ' + str(output))
    print('Two-request screen: ' + ('PASS' if summary['passed'] else 'FAIL'))
    return 0 if summary['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
