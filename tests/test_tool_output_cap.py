"""v0.1.3 TOOL-OUTPUT CAP (flag LOCAL_AI_TOOL_OUTPUT_CAP, default off): model-free.

Covers the marker's determinism, the head/tail/error-line keep, exact read-more ranges (and all),
the store's permissions/TTL/size cap, the never-truncate rules, prefix stability across turns, the
read-more MCP server over real stdio, and a fake OpenCode tool loop through the real controller.
"""
import os as _os, tempfile as _tempfile
_os.environ.setdefault('LOCAL_AI_LEDGER_DIR', _tempfile.mkdtemp(prefix='local-ai-test-ledger-'))  # never the real request ledger
import hashlib
import http.client
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from unittest.mock import patch

from snapdragon_ai_lab.runtime import controller, tool_output_cap as cap
from snapdragon_ai_lab.runtime import launch_local_plan as launcher
from snapdragon_ai_lab.runtime.engine_diagnostics import EngineLogWriter

READER = 'local_ai_read_tool_output'
TOOLS = [{'type': 'function', 'function': {'name': n, 'parameters': {'type': 'object'}}}
         for n in ('shell', 'read', 'grep', READER)]


def big_log(n=3000, errors=(1234, 2345)):
    return '\n'.join(('step %d FAILED: assertion error in test_%d' % (i, i)) if i in errors else
                     'step %d ok compiling module_%d' % (i, i) for i in range(1, n + 1))


def call(call_id, name, arguments):
    return {'role': 'assistant', 'content': None, 'tool_calls': [{'id': call_id, 'type': 'function',
            'function': {'name': name, 'arguments': json.dumps(arguments)}}]}


def conversation(output, name='shell', arguments=None, tools=TOOLS, call_id='c1'):
    return {'model': 'm', 'tools': tools, 'messages': [
        {'role': 'system', 'content': 'sys'}, {'role': 'user', 'content': 'build it'},
        call(call_id, name, arguments if arguments is not None else {'command': 'make'}),
        {'role': 'tool', 'tool_call_id': call_id, 'content': output}]}


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.env = {cap.ENV: 'on', cap.ENV + '_STORE': str(Path(self.tmp.name) / 'store')}
        self.cfg = cap.settings(self.env)
        self.store = cap.Store(self.cfg['store'], self.cfg['ttl_secs'], self.cfg['budget_bytes'])

    def capped(self, payload, headers=None, cfg=None):
        return cap.cap_request(payload, headers or {'x-opencode-session-id': 'ses_a'}, cfg or self.cfg, self.store)

    @staticmethod
    def tool_text(payload, index=-1):
        return [m for m in payload['messages'] if m['role'] == 'tool'][index]['content']


class Render(Base):
    def test_defaults_are_off(self):
        self.assertFalse(cap.settings({})['enabled'])
        self.assertFalse(cap.settings({cap.ENV: 'maybe'})['enabled'])
        self.assertEqual(cap.settings({})['tokens'], 2000)
        self.assertEqual(cap.settings({cap.ENV + '_TOKENS': '5'})['tokens'], 2000, 'out-of-range -> default')

    def test_deterministic_bytes(self):
        text = big_log()
        digest = hashlib.sha256(text.encode()).hexdigest()
        self.assertEqual(cap.render(text, digest, READER, self.cfg), cap.render(text, digest, READER, self.cfg))
        a, _ = self.capped(conversation(text))
        other = cap.Store(Path(self.tmp.name) / 'other', 3600, 1 << 30)
        b, _ = cap.cap_request(conversation(text), {'x-opencode-session-id': 'ses_b'}, self.cfg, other)
        self.assertEqual(json.dumps(a), json.dumps(b), 'no session, clock or store path in the render')

    def test_keeps_head_tail_and_error_lines(self):
        text = big_log()
        out, stats = self.capped(conversation(text))
        rendered = self.tool_text(out)
        lines = rendered.split('\n')
        self.assertEqual(lines[:40], text.split('\n')[:40])
        self.assertEqual(lines[-60:], text.split('\n')[-60:])
        self.assertIn('L1234: step 1234 FAILED: assertion error in test_1234', rendered)
        self.assertIn('L2345: step 2345 FAILED', rendered)
        marker = next(x for x in lines if x.startswith('<<truncated:'))
        ident = hashlib.sha256(text.encode()).hexdigest()[:12]
        self.assertIn('lines 41-2940 of 3000 hidden', marker)
        self.assertIn('%s tool: id=%s offset=41 limit=200, or all=true' % (READER, ident), marker)
        self.assertEqual(stats['capped'], 1)
        self.assertLessEqual(cap.estimate_tokens(rendered), self.cfg['tokens'] * 1.3)
        self.assertGreater(stats['hidden_tokens'], 10000)

    def test_long_lines_are_clipped_and_pointed_to(self):
        text = 'x' * 50000
        out, _ = self.capped(conversation(text))
        rendered = self.tool_text(out)
        self.assertIn('<<line 1: +49600 chars; read it with offset=1 limit=1>>', rendered)
        self.assertIn('offset=1 limit=1, or all=true', rendered)
        self.assertLess(len(rendered), 1000)

    def test_error_lines_are_capped(self):
        text = '\n'.join('error %d' % i for i in range(5000))
        rendered = self.tool_text(self.capped(conversation(text))[0])
        self.assertEqual(sum(1 for x in rendered.split('\n') if x.startswith('L')), 20)


class ReadMore(Base):
    def setUp(self):
        super().setUp()
        self.text = big_log()
        self.out, _ = self.capped(conversation(self.text))
        self.ident = hashlib.sha256(self.text.encode()).hexdigest()[:12]

    def test_exact_ranges(self):
        lines = self.text.split('\n')
        for offset, limit in ((41, 200), (1, 1), (2999, 5), (1234, 1)):
            got, footer = cap.read_more(self.store, self.ident, offset, limit)
            self.assertEqual(got, '\n'.join(lines[offset - 1:offset - 1 + limit]))
        self.assertEqual(footer, '<<lines 1234-1234 of 3000; next: offset=1235>>')
        self.assertTrue(cap.read_more(self.store, self.ident, 2999, 5)[1].endswith('end of output>>'))

    def test_all_is_the_exact_output(self):
        got, footer = cap.read_more(self.store, self.ident, all_lines=True)
        self.assertEqual(got, self.text)
        self.assertIsNone(footer)
        self.assertEqual(cap.read_more(self.store, hashlib.sha256(self.text.encode()).hexdigest(), all_lines=True)[0], self.text)

    def test_bad_requests(self):
        with self.assertRaises(KeyError): cap.read_more(self.store, 'deadbeef0000')
        with self.assertRaises(KeyError): cap.read_more(self.store, '../../etc/passwd')
        with self.assertRaises(ValueError): cap.read_more(self.store, self.ident, 3001, 1)
        with self.assertRaises(ValueError): cap.read_more(self.store, self.ident, 0, 1)

    def test_mcp_handle(self):
        reply = cap.handle({'jsonrpc': '2.0', 'id': 1, 'method': 'tools/call', 'params': {
            'name': cap.TOOL, 'arguments': {'id': self.ident, 'offset': 41, 'limit': 3}}}, self.store)
        content = reply['result']['content']
        self.assertEqual(content[0]['text'], '\n'.join(self.text.split('\n')[40:43]))
        self.assertFalse(reply['result']['isError'])
        reply = cap.handle({'jsonrpc': '2.0', 'id': 2, 'method': 'tools/call', 'params': {
            'name': cap.TOOL, 'arguments': {'id': 'ffffffffffff'}}}, self.store)
        self.assertTrue(reply['result']['isError'])
        self.assertIsNone(cap.handle({'jsonrpc': '2.0', 'method': 'notifications/initialized'}, self.store))

    def test_mcp_server_over_stdio(self):
        script = Path(cap.__file__)
        requests = [{'jsonrpc': '2.0', 'id': 1, 'method': 'initialize', 'params': {'protocolVersion': '2025-06-18'}},
                    {'jsonrpc': '2.0', 'method': 'notifications/initialized'},
                    {'jsonrpc': '2.0', 'id': 2, 'method': 'tools/list'},
                    {'jsonrpc': '2.0', 'id': 3, 'method': 'tools/call',
                     'params': {'name': cap.TOOL, 'arguments': {'id': self.ident, 'all': True}}}]
        env = {**os.environ, **self.env}
        run = subprocess.run([sys.executable, '-I', str(script), 'mcp'], input=''.join(json.dumps(r) + '\n' for r in requests),
                             capture_output=True, text=True, env=env, timeout=30)
        replies = [json.loads(x) for x in run.stdout.splitlines()]
        self.assertEqual([r['id'] for r in replies], [1, 2, 3])
        self.assertEqual(replies[0]['result']['protocolVersion'], '2025-06-18')
        self.assertEqual(replies[1]['result']['tools'][0]['name'], cap.TOOL)
        self.assertEqual(replies[2]['result']['content'][0]['text'], self.text)


class StoreRules(Base):
    def test_permissions(self):
        self.capped(conversation(big_log()))
        root = self.cfg['store']
        self.assertEqual(stat.S_IMODE(root.stat().st_mode), 0o700)
        for path in root.rglob('*'):
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o700 if path.is_dir() else 0o600, path)

    def test_ttl_and_size_cap(self):
        store = cap.Store(self.cfg['store'], ttl_secs=60, budget_bytes=200_000)
        for n in range(4):
            cap.cap_request(conversation(big_log(3000 + n)), {'x-opencode-session-id': 's'}, self.cfg, store)
        raws = sorted(self.cfg['store'].rglob('*.raw'), key=lambda p: p.stat().st_mtime)
        self.assertLess(sum(p.stat().st_size for p in self.cfg['store'].rglob('*.*')), 200_000)
        self.assertGreaterEqual(len(raws), 1)
        self.assertEqual(store.gc(now=time.time() + 120), len(raws), 'all expired after the TTL')
        self.assertEqual(list(self.cfg['store'].rglob('*.raw')), [])
        self.assertEqual(list(self.cfg['store'].rglob('*.cap')), [])

    def test_lookup_keeps_entries_in_use(self):
        text = big_log()
        self.capped(conversation(text))
        raw = next(self.cfg['store'].rglob('*.raw'))
        os.utime(raw, (1, 1))
        self.capped(conversation(text))  # reused -> touched
        self.assertGreater(raw.stat().st_mtime, time.time() - 60)

    def test_store_failure_fails_open(self):
        blocked = Path(self.tmp.name) / 'file'
        blocked.write_text('x')
        cfg = cap.settings({**self.env, cap.ENV + '_STORE': str(blocked / 'store')})
        payload = conversation(big_log())
        out, stats = cap.cap_request(payload, {}, cfg)
        self.assertIs(out, payload)
        self.assertEqual(stats['capped'], 0)


class NeverTruncate(Base):
    def assertUntouched(self, payload, cfg=None):
        out, stats = self.capped(payload, cfg=cfg)
        self.assertIs(out, payload)
        self.assertEqual(stats['capped'], 0)

    def test_flag_off(self):
        self.assertUntouched(conversation(big_log()), cfg=cap.settings({cap.ENV + '_STORE': self.env[cap.ENV + '_STORE']}))

    def test_below_threshold(self):
        self.assertUntouched(conversation(big_log(50)))

    def test_no_read_more_tool_offered(self):
        self.assertUntouched(conversation(big_log(), tools=TOOLS[:3]))
        self.assertUntouched(conversation(big_log(), tools=None))

    def test_read_edit_and_unknown_tools(self):
        for name in ('read', 'edit', 'write', 'task', 'some_mcp_tool', READER):
            self.assertUntouched(conversation(big_log(), name=name, arguments={'path': 'x'}))

    def test_explicit_range_and_no_truncate_hint(self):
        policy = cap.settings({**self.env, cap.ENV + '_TOOLS': 'read=on'})
        self.assertUntouched(conversation(big_log(), name='read', arguments={'path': 'x', 'offset': 1, 'limit': 5000}), cfg=policy)
        out, _ = self.capped(conversation(big_log(), name='read', arguments={'path': 'x'}), cfg=policy)
        self.assertIn('<<truncated:', self.tool_text(out), 'read=on caps a plain read')
        # the same output from a later call that asks for no truncation is not given the earlier render
        self.assertUntouched(conversation(big_log(), arguments={'command': 'make', 'no_truncate': True}, call_id='c9'))

    def test_policy_override_off(self):
        self.assertUntouched(conversation(big_log()), cfg=cap.settings({**self.env, cap.ENV + '_TOOLS': 'shell=off'}))

    def test_history_first_seen_uncapped_stays_uncapped(self):
        # the result already sits before a later assistant turn (e.g. the flag was just turned on)
        payload = conversation(big_log())
        payload['messages'] += [{'role': 'assistant', 'content': 'ok'}, {'role': 'user', 'content': 'next'}]
        self.assertUntouched(payload)


class PrefixStability(Base):
    def test_next_turn_reuses_the_same_bytes(self):
        text = big_log()
        first, _ = self.capped(conversation(text))
        later = conversation(text)
        ident = hashlib.sha256(text.encode()).hexdigest()[:12]
        read = cap.read_more(self.store, ident, 41, 200)[0] * 3  # a big read-more result is never capped
        later['messages'] += [call('c2', READER, {'id': ident, 'offset': 41, 'limit': 600}),
                              {'role': 'tool', 'tool_call_id': 'c2', 'content': read}]
        second, stats = self.capped(later)
        n = len(first['messages'])
        self.assertEqual(json.dumps(second['messages'][:n]), json.dumps(first['messages']))
        self.assertEqual(second['messages'][-1]['content'], read)
        self.assertEqual(stats, {'capped': 0, 'reused': 1, 'hidden_tokens': 0, 'reads': 1})
        # a different threshold later does not re-render history already sent capped
        third, _ = self.capped(later, cfg=cap.settings({**self.env, cap.ENV + '_TOKENS': '900'}))
        self.assertEqual(json.dumps(third['messages'][:n]), json.dumps(first['messages']))


class LaunchOverlay(unittest.TestCase):
    def test_flag_off_leaves_the_overlay(self):
        env = {'OPENCODE_CONFIG_CONTENT': '{"model":"x"}'}
        self.assertIs(launcher.with_tool_output_cap(env), env)

    def test_flag_on_registers_a_direct_mcp_tool(self):
        env = {'OPENCODE_CONFIG_CONTENT': '{"model":"x"}', 'LOCAL_AI_TOOL_OUTPUT_CAP': 'on',
               'LOCAL_AI_TOOL_OUTPUT_MCP': str(Path(cap.__file__).resolve())}
        overlay = json.loads(launcher.with_tool_output_cap(env)['OPENCODE_CONFIG_CONTENT'])
        entry = overlay['mcp']['servers'][cap.SERVER_NAME]
        self.assertEqual(entry['type'], 'local')
        self.assertFalse(entry['codemode'])
        self.assertEqual(entry['command'][1:], [str(Path(cap.__file__).resolve()), 'mcp'])
        self.assertEqual(overlay['model'], 'x')
        self.assertEqual(launcher.TOOL_CAP_SERVER + '_' + cap.TOOL, READER)

    def test_missing_script_registers_nothing(self):
        env = {'LOCAL_AI_TOOL_OUTPUT_CAP': 'on', 'LOCAL_AI_TOOL_OUTPUT_MCP': '/nonexistent/tool_output_cap.py'}
        self.assertNotIn('mcp', json.loads(launcher.with_tool_output_cap(env)['OPENCODE_CONFIG_CONTENT']))


class FakeOpenCodeLoop(unittest.TestCase):
    """A scripted OpenCode-like client -> real controller proxy -> fake engine that records bodies."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.bodies = []
        test_case = self

        class Engine(BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'

            def log_message(self, *_args):
                pass

            def do_POST(self):
                test_case.bodies.append(self.rfile.read(int(self.headers['Content-Length'])))
                data = json.dumps({'id': 'c', 'object': 'chat.completion', 'created': 1, 'model': 'm', 'choices': [
                    {'index': 0, 'message': {'role': 'assistant', 'content': 'ok'}, 'finish_reason': 'stop'}]}).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        backend = ThreadingHTTPServer(('127.0.0.1', 0), Engine)
        backend.daemon_threads = True
        threading.Thread(target=backend.serve_forever, daemon=True).start()
        self.addCleanup(backend.server_close); self.addCleanup(backend.shutdown)
        process = SimpleNamespace(pid=9999, poll=lambda: None)
        config = {'model_alias': 'synthetic-loopback', 'backend_kind': 'llama', 'server_sha256': 'a' * 64}
        process._local_ai_engine_config = config
        manager = SimpleNamespace(backend_runtime=None, current_prefix_key=None, cache={}, loaded_config=config,
                                  process=process,
                                  engine_log_writer=EngineLogWriter(Path(self.tmp.name) / 'engine-stderr.log'),
                                  config=lambda _alias=None: config, begin_queued=lambda **_kwargs: process,
                                  prepare_prefix=lambda *_args: None, end=lambda **_kwargs: None,
                                  record_engine_death=lambda *_args: None)
        self.proxy = controller.Server(('127.0.0.1', 0), controller.Handler)
        self.proxy.manager = manager
        threading.Thread(target=self.proxy.serve_forever, daemon=True).start()
        self.addCleanup(self.proxy.server_close); self.addCleanup(self.proxy.shutdown)
        self.env = {cap.ENV: 'on', cap.ENV + '_STORE': str(Path(self.tmp.name) / 'store')}
        for item in (patch.object(controller, 'BACKEND_PORT', backend.server_port), patch.dict(os.environ, self.env)):
            item.start(); self.addCleanup(item.stop)

    def post(self, payload):
        client = http.client.HTTPConnection('127.0.0.1', self.proxy.server_port, timeout=10)
        self.addCleanup(client.close)
        client.request('POST', '/v1/chat/completions', body=json.dumps(payload).encode(),
                       headers={'Content-Type': 'application/json', 'x-opencode-session-id': 'ses_loop'})
        response = client.getresponse()
        response.read()
        return response.status, json.loads(self.bodies[-1])

    def test_cap_read_more_and_prefix_through_the_controller(self):
        text = big_log()
        payload = conversation(text)
        status, seen = self.post(payload)
        self.assertEqual(status, 200)
        rendered = [m for m in seen['messages'] if m['role'] == 'tool'][0]['content']
        self.assertIn('<<truncated: lines 41-2940 of 3000 hidden', rendered)
        ident = rendered.split('id=')[1][:12]
        # the agent calls the read-more tool (the MCP server) and gets the exact hidden lines
        reply = cap.handle({'jsonrpc': '2.0', 'id': 7, 'method': 'tools/call', 'params': {
            'name': cap.TOOL, 'arguments': {'id': ident, 'offset': 1230, 'limit': 10}}},
            cap.Store(Path(self.env[cap.ENV + '_STORE']), 3600, 1 << 30))
        exact = reply['result']['content'][0]['text']
        self.assertEqual(exact, '\n'.join(text.split('\n')[1229:1239]))
        payload['messages'] += [call('c2', READER, {'id': ident, 'offset': 1230, 'limit': 10}),
                                {'role': 'tool', 'tool_call_id': 'c2', 'content': exact}]
        _, seen2 = self.post(payload)
        n = len(seen['messages'])
        self.assertEqual(json.dumps(seen2['messages'][:n]), json.dumps(seen['messages']), 'prefix bytes stable')
        self.assertEqual(seen2['messages'][-1]['content'], exact)
        # flag off: the controller forwards the request untouched (the A/B baseline)
        with patch.dict(os.environ, {cap.ENV: 'off'}):
            _, seen3 = self.post(payload)
        self.assertEqual(seen3['messages'], payload['messages'])


if __name__ == '__main__':
    unittest.main()
