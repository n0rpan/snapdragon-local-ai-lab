"""Tests for tools/session_replay.py. Only local mock servers on 127.0.0.1 ephemeral ports."""
import contextlib
import copy
import hashlib
import http.client
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from session_replay_mock import MockServer  # noqa: E402

_spec = importlib.util.spec_from_file_location('session_replay', ROOT / 'tools/session_replay.py')
sr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sr)

SYS_A = 'You are the synthetic coding agent A. ' * 4
SYS_B = 'Subagent B has different synthetic instructions. ' * 3
TOOLS = [{'type': 'function', 'function': {'name': 'read_file', 'description': 'read a file',
                                             'parameters': {'type': 'object', 'properties': {}}}}]
CHAT = '/v1/chat/completions'
# Expected per-turn results of session_turns() against the deterministic mock.
EXPECTED_TOKENS = [7, 7, 7, 12, 5, 7]
EXPECTED_SOURCE = ['usage', 'usage', 'usage', 'timings', 'usage', 'usage']
EXPECTED_FINISH = ['stop', 'stop', 'stop', 'length', 'length', 'stop']


def session_turns():
    """Six turns: two agents, stream and non-stream, varied sampling fields."""
    return [
        (CHAT, {'model': 'm', 'stream': True, 'stream_options': {'include_usage': True}, 'tools': TOOLS,
                'messages': [{'role': 'system', 'content': SYS_A},
                             {'role': 'user', 'content': 'hello one'}]}),
        (CHAT, {'model': 'm', 'stream': False, 'temperature': 0, 'tools': TOOLS,
                'messages': [{'role': 'system', 'content': SYS_B},
                             {'role': 'user', 'content': 'agent b task'}]}),
        (CHAT, {'model': 'm', 'stream': True, 'stream_options': {'include_usage': True},
                'temperature': 0.55, 'top_p': 0.95, 'max_tokens': 32000, 'stop': ['\n\n'],
                'messages': [{'role': 'system', 'content': SYS_A},
                             {'role': 'user', 'content': 'two'},
                             {'role': 'assistant', 'content': 'reply'},
                             {'role': 'user', 'content': 'two again'}]}),
        (CHAT, {'model': 'm', 'stream': True, 'n_predict': 12, 'ignore_eos': True,
                'messages': [{'role': 'system', 'content': SYS_A},
                             {'role': 'user', 'content': 'three'}]}),
        (CHAT, {'model': 'm', 'stream': True, 'stream_options': {'include_usage': True},
                'seed': 42, 'max_tokens': 5,
                'messages': [{'role': 'system', 'content': SYS_B},
                             {'role': 'user', 'content': 'tool please'},
                             {'role': 'tool', 'content': 'result text'}]}),
        (CHAT, {'model': 'm', 'stream': False,
                'messages': [{'role': 'system', 'content': SYS_A},
                             {'role': 'user', 'content': 'four'}]}),
    ]


def post(port, path, body):
    """POST through the proxy or mock; returns (status, bytes, error-or-None)."""
    conn = http.client.HTTPConnection('127.0.0.1', port, timeout=20)
    try:
        conn.request('POST', path, body=json.dumps(body),
                     headers={'Content-Type': 'application/json'})
        resp = conn.getresponse()
        return resp.status, resp.read(), None
    except (OSError, http.client.HTTPException) as exc:
        return None, b'', f'{type(exc).__name__}: {exc}'
    finally:
        conn.close()


def get(port, path):
    conn = http.client.HTTPConnection('127.0.0.1', port, timeout=20)
    try:
        conn.request('GET', path)
        resp = conn.getresponse()
        return resp.status, resp.read()
    finally:
        conn.close()


def run_session(port, turns):
    return [post(port, path, body) for path, body in turns]


def run_main(argv):
    out = io.StringIO()
    err = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = sr.main(argv)
    return code, out.getvalue(), err.getvalue()


def wait_turns(recording, n, timeout=10):
    """The recorder appends a turn AFTER the client has its last byte (tools/session_replay.py
    _proxy: final chunk, flush, then record_turn), so a reader right after the session can miss the
    last turn or see half a line: wait until `n` complete turn lines are on disk."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            text = Path(recording).read_text()
        except OSError:
            text = ''
        complete = text[:text.rfind('\n') + 1]
        if len([l for l in complete.splitlines()[1:] if l.strip()]) >= n:
            return
        time.sleep(0.01)


class RecorderFixture(unittest.TestCase):
    """Starts a mock upstream and a recorder in front of it; both on ephemeral 127.0.0.1 ports."""

    def start_mock(self, **kwargs):
        mock = MockServer(**kwargs).start()
        self.addCleanup(mock.stop)
        return mock

    def start_recorder(self, upstream_url, out, allow_git=False):
        server = sr.RecorderServer(('127.0.0.1', 0), upstream_url, out, allow_git=allow_git)
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
        thread.start()

        def _stop():
            server.shutdown()
            server.server_close()
            thread.join(5)

        self.addCleanup(_stop)
        return server

    def make_recording(self, turns, out, fail_at=()):
        mock = self.start_mock(fail_at=fail_at)
        recorder = self.start_recorder(mock.url, out)
        results = run_session(recorder.server_address[1], turns)
        wait_turns(recorder.recording, len(turns))
        return recorder.recording, mock, results


class RecordTests(RecorderFixture):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix='sr-record-'))
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_proxy_is_transparent_and_records_turns(self):
        direct = self.start_mock()
        direct_results = run_session(int(direct.url.rsplit(':', 1)[1]), session_turns())
        direct_models = get(int(direct.url.rsplit(':', 1)[1]), '/v1/models')

        upstream = self.start_mock()
        out = self.tmp / 'rec'
        recorder = self.start_recorder(upstream.url, out)
        port = recorder.server_address[1]
        proxied = run_session(port, session_turns())
        proxied_models = get(port, '/v1/models')
        wait_turns(out / 'recording.jsonl', 6)  # the flake (v0.1.3 integration): read before turn 6 was appended

        self.assertEqual(len(proxied), 6)
        for (status_d, data_d, err_d), (status_p, data_p, err_p) in zip(direct_results, proxied):
            self.assertIsNone(err_p)
            self.assertEqual(status_p, status_d)
            self.assertEqual(data_p, data_d, 'proxied response bytes differ from direct response')
        self.assertEqual(proxied_models, direct_models)

        # The proxy forwards request bodies verbatim.
        sent = [body for _, body in session_turns()]
        self.assertEqual([r['body'] for r in upstream.received], sent)
        self.assertTrue(all(r['origin'] == 'lab' for r in upstream.received))

        # Recording: header then six turns.
        # The recorder appends a turn after the client already has its reply: wait (bounded) for the
        # last row instead of racing it (a slow CI runner saw 3 of 6).
        deadline = time.monotonic() + 10
        while True:
            text = out.joinpath('recording.jsonl').read_text()
            if text.count('\n') >= 7 or time.monotonic() > deadline:
                break
            time.sleep(0.05)
        lines = [json.loads(x) for x in text.splitlines()]
        self.assertEqual(lines[0]['kind'], 'header')
        self.assertEqual(lines[0]['version'], 1)
        self.assertEqual(lines[0]['props'], upstream.props)
        turns = lines[1:]
        self.assertEqual(len(turns), 6)
        self.assertEqual([t['seq'] for t in turns], [1, 2, 3, 4, 5, 6])
        self.assertEqual([t['completion_tokens'] for t in turns], EXPECTED_TOKENS)
        self.assertEqual([t['completion_tokens_source'] for t in turns], EXPECTED_SOURCE)
        self.assertEqual([t['finish_reason'] for t in turns], EXPECTED_FINISH)
        self.assertTrue(all(t['status'] == 200 and not t['aborted'] and t['error'] is None
                            for t in turns))
        self.assertEqual([t['body'] for t in turns], sent)
        self.assertEqual(turns[0]['usage']['completion_tokens'], 7)
        self.assertIsNone(turns[3]['usage'])
        self.assertEqual(turns[3]['timings']['predicted_n'], 12)

        self.assertEqual(stat.S_IMODE(os.stat(out).st_mode), 0o700)
        self.assertEqual(stat.S_IMODE(os.stat(out / 'recording.jsonl').st_mode), 0o600)

    def test_aborted_stream_is_recorded_and_replay_skips_it(self):
        turns = [session_turns()[0], session_turns()[2], session_turns()[0]]
        recording, _, results = self.make_recording(turns, self.tmp / 'rec', fail_at={1})
        self.assertIsNone(results[0][2])
        _, recorded, _ = sr.load_recording(recording)
        self.assertEqual([t['aborted'] for t in recorded], [False, True, False])
        self.assertTrue(recorded[1]['error'])

        replay_mock = self.start_mock()
        out = self.tmp / 'result.json'
        code, _, _ = run_main(['replay', str(recording), '--server', replay_mock.url,
                               '--out', str(out), '--repeat', '1'])
        self.assertEqual(code, 0)
        result = json.loads(out.read_text())
        skipped = result['repeats'][0]['skipped']
        self.assertEqual(len(skipped), 1)
        self.assertEqual(skipped[0]['seq'], 2)
        self.assertIn('aborted', skipped[0]['reason'])
        self.assertEqual(result['repeats'][0]['totals']['n_turns'], 2)

    def test_git_guard(self):
        repo = self.tmp / 'repo'
        repo.mkdir()
        try:
            subprocess.run(['git', 'init', '-q', str(repo)], check=True, capture_output=True)
        except (OSError, subprocess.CalledProcessError) as exc:
            self.skipTest(f'git unavailable: {exc}')
        upstream = self.start_mock()
        out = repo / 'rec'
        code, _, err = run_main(['record', '--listen', '127.0.0.1:0', '--upstream', upstream.url,
                                 '--out', str(out)])
        self.assertEqual(code, 2)
        self.assertIn('git work tree', err)
        self.assertFalse(out.exists())

        allowed = sr.RecorderServer(('127.0.0.1', 0), upstream.url, out, allow_git=True)
        self.addCleanup(allowed.server_close)
        self.assertTrue(allowed.recording.exists())
        self.assertEqual(stat.S_IMODE(os.stat(out).st_mode), 0o700)

    def test_arguments_parse_listen_and_allow_git(self):
        args = sr.build_parser().parse_args(['record', '--upstream', 'http://127.0.0.1:1',
                                             '--out', '/x', '--allow-git'])
        self.assertTrue(args.allow_git)
        self.assertEqual(sr.parse_listen(args.listen), ('127.0.0.1', 18200))
        with self.assertRaises(sr.UsageError):
            sr.parse_listen('no-port-here')


class ReplayTests(RecorderFixture):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix='sr-replay-'))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.recording, _, _ = self.make_recording(session_turns(), self.tmp / 'rec')

    def replay(self, mock, name='result.json', extra=()):
        out = self.tmp / name
        code, stdout, stderr = run_main(['replay', str(self.recording), '--server', mock.url,
                                         '--out', str(out), *extra])
        return code, json.loads(out.read_text()), stdout, stderr

    def test_replay_rewrites_bodies_and_measures_equal_work(self):
        mock = self.start_mock()
        code, result, stdout, _ = self.replay(mock, extra=['--repeat', '1', '--label', 'unit'])
        self.assertEqual(code, 0)
        self.assertIn('repeat', stdout)

        scrub_body = mock.received[0]['body']
        self.assertEqual(scrub_body['messages'][0]['content'][:6], 'scrub ')
        self.assertEqual(len(scrub_body['messages'][0]['content']), 6 + 32)
        self.assertEqual(scrub_body['n_predict'], 1)
        self.assertEqual(scrub_body['temperature'], 0)

        replayed = [r['body'] for r in mock.received[1:]]
        self.assertEqual(len(replayed), 6)
        for body, target in zip(replayed, EXPECTED_TOKENS):
            self.assertEqual(body['temperature'], 0)
            self.assertEqual(body['n_predict'], target)
            self.assertEqual(body['max_tokens'], target)
            self.assertIs(body['ignore_eos'], True)
            self.assertIs(body['cache_prompt'], True)
            self.assertIs(body['stream'], True)
            self.assertEqual(body['stream_options'], {'include_usage': True})
            for key in ('stop', 'seed', 'top_p', 'top_k', 'min_p', 'max_completion_tokens'):
                self.assertNotIn(key, body)

        self.assertEqual(result['label'], 'unit')
        self.assertEqual(result['recording_sha256'],
                         hashlib.sha256(self.recording.read_bytes()).hexdigest())
        self.assertEqual(result['props'], mock.props)
        self.assertIn('data', result['models'])

        rep = result['repeats'][0]
        self.assertIsNotNone(rep['scrub'])
        self.assertEqual(rep['scrub']['status'], 200)
        self.assertEqual(len(rep['turns']), 6)
        self.assertEqual(rep['totals']['n_turns'], 6)
        self.assertTrue(rep['totals']['equal_work_all'])
        self.assertTrue(all(row['equal_work'] for row in rep['turns']))
        self.assertEqual([row['predicted_n'] for row in rep['turns']], EXPECTED_TOKENS)
        self.assertEqual(rep['totals']['predicted_tokens'], sum(EXPECTED_TOKENS))
        self.assertEqual(rep['totals']['prompt_tokens_new'],
                         sum(row['prompt_n'] for row in rep['turns']))
        self.assertAlmostEqual(sum(row['wall_ms'] for row in rep['turns']) / 1000.0,
                               rep['totals']['wall_s'], places=9)
        agents = [row['agent'] for row in rep['turns']]
        self.assertEqual(agents[0], agents[2])
        self.assertNotEqual(agents[0], agents[1])
        self.assertTrue(all(row['wall_ms'] > 0 for row in rep['turns']))
        self.assertTrue(rep['env'])
        self.assertIn('spread_pct', result['summary'])

    def test_repeat_two_and_scrub_between_repeats(self):
        mock = self.start_mock()
        code, result, _, _ = self.replay(mock, extra=['--repeat', '2'])
        self.assertEqual(code, 0)
        self.assertEqual(len(result['repeats']), 2)
        self.assertEqual(len(mock.received), 2 * (1 + 6))
        self.assertEqual(mock.received[7]['body']['messages'][0]['content'][:6], 'scrub ')
        for rep in result['repeats']:
            self.assertEqual(rep['totals']['predicted_tokens'], sum(EXPECTED_TOKENS))
        self.assertEqual(len(result['summary']['wall_s_all']), 2)
        self.assertLess(result['summary']['spread_pct'], 200.0)

    def test_no_scrub_flag(self):
        mock = self.start_mock()
        code, result, _, _ = self.replay(mock, extra=['--no-scrub'])
        self.assertEqual(code, 0)
        self.assertEqual(len(mock.received), 6)
        self.assertIsNone(result['repeats'][0]['scrub'])

    def test_ignored_n_predict_fails_equal_work(self):
        mock = self.start_mock(ignore_n_predict=True)
        code, result, _, _ = self.replay(mock, extra=['--repeat', '1'])
        self.assertEqual(code, 3)
        self.assertTrue(result['repeats'][0]['turns'])
        self.assertFalse(result['repeats'][0]['totals']['equal_work_all'])
        self.assertTrue(all(row['predicted_n'] == 3 and not row['equal_work']
                            for row in result['repeats'][0]['turns']))


class CompareTests(RecorderFixture):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix='sr-compare-'))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.recording, _, _ = self.make_recording(session_turns(), self.tmp / 'rec')
        mock = self.start_mock()
        self.a_path = self.tmp / 'a.json'
        code, _, _ = run_main(['replay', str(self.recording), '--server', mock.url,
                               '--out', str(self.a_path), '--repeat', '2'])
        self.assertEqual(code, 0)

    def test_same_file_is_within_gate(self):
        code, out, _ = run_main(['compare', str(self.a_path), str(self.a_path)])
        self.assertEqual(code, 0)
        self.assertIn('WITHIN', out)

    def test_two_real_runs_agree_within_wide_gate(self):
        mock = self.start_mock()
        b_path = self.tmp / 'b.json'
        code, _, _ = run_main(['replay', str(self.recording), '--server', mock.url,
                               '--out', str(b_path), '--repeat', '2'])
        self.assertEqual(code, 0)
        code, out, _ = run_main(['compare', str(self.a_path), str(b_path), '--gate-pct', '100'])
        self.assertEqual(code, 0)
        self.assertIn('totals wall', out)

    def test_ten_percent_slower_fails_default_gate(self):
        result = json.loads(self.a_path.read_text())
        for rep in result['repeats']:
            rep['totals']['wall_s'] *= 1.10
            for row in rep['turns']:
                row['wall_ms'] *= 1.10
        b_path = self.tmp / 'slow.json'
        b_path.write_text(json.dumps(result))
        code, out, _ = run_main(['compare', str(self.a_path), str(b_path)])
        self.assertEqual(code, 1)
        self.assertIn('OUTSIDE', out)

    def test_different_recordings_are_refused(self):
        result = json.loads(self.a_path.read_text())
        result['recording_sha256'] = 'f' * 64
        other = self.tmp / 'other.json'
        other.write_text(json.dumps(result))
        code, _, err = run_main(['compare', str(self.a_path), str(other)])
        self.assertEqual(code, 2)
        self.assertIn('different recordings', err)


class InfoTests(RecorderFixture):
    def test_info_reports_sampling_variants_and_agents(self):
        tmp = Path(tempfile.mkdtemp(prefix='sr-info-'))
        self.addCleanup(shutil.rmtree, tmp, True)
        recording, _, _ = self.make_recording(session_turns(), tmp / 'rec')
        code, out, _ = run_main(['info', str(recording)])
        self.assertEqual(code, 0)
        self.assertIn('temperature: absent=4', out)
        self.assertIn('0.55=1', out)
        self.assertIn('top_p: absent=5, 0.95=1', out)
        self.assertIn('completion tokens (non-aborted): 45', out)
        self.assertIn('agents: 2', out)
        self.assertIn('turns: 6', out)


class TokensTests(RecorderFixture):
    def test_tokens_per_turn(self):
        tmp = Path(tempfile.mkdtemp(prefix='sr-tokens-'))
        self.addCleanup(shutil.rmtree, tmp, True)
        recording, _, _ = self.make_recording(session_turns(), tmp / 'rec')
        turns = [json.loads(l) for l in recording.read_text().splitlines()[1:]]
        self.assertTrue(all('response' in t for t in turns))
        self.assertTrue(any(t['response']['content'] for t in turns))
        mock = self.start_mock()
        out = tmp / 'tokens.jsonl'
        code, stdout, _ = run_main(['tokens', str(recording), '--server', mock.url, '--out', str(out)])
        rows = [json.loads(l) for l in out.read_text().splitlines()]
        self.assertEqual(len(rows), 6)
        for r in rows:
            self.assertTrue(r['prompt_ids'] and isinstance(r['prompt_ids'][0], int))
            self.assertIn('completion_ids', r)
            self.assertIn('prompt_ms', r)
        self.assertEqual(code, 0, stdout)
        self.assertEqual(oct(out.stat().st_mode & 0o777), '0o600')


class EnvSamplerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix='sr-env-'))
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_reads_fake_sysfs_tree(self):
        sysfs = self.tmp / 'sys'
        zone0 = sysfs / 'class/thermal/thermal_zone0'
        zone1 = sysfs / 'class/thermal/thermal_zone1'
        policy = sysfs / 'devices/system/cpu/cpufreq/policy0'
        for d in (zone0, zone1, policy):
            d.mkdir(parents=True)
        (zone0 / 'type').write_text('x86_pkg_temp\n')
        (zone0 / 'temp').write_text('45500\n')
        (zone1 / 'type').write_text('acpitz\n')
        (zone1 / 'temp').write_text('40000\n')
        (policy / 'scaling_cur_freq').write_text('2400000\n')
        proc = self.tmp / 'proc'
        proc.mkdir()
        (proc / 'meminfo').write_text('MemTotal: 1 kB\nMemAvailable:    2048000 kB\n')
        sample = sr.sample_env(sys_root=sysfs, proc_root=proc)
        self.assertEqual(sample['thermal'], {'x86_pkg_temp': 45.5, 'acpitz': 40.0})
        self.assertEqual(sample['cpufreq_mhz'], {'policy0': 2400.0})
        self.assertEqual(sample['mem_available_mib'], 2000.0)

    def test_missing_files_are_tolerated(self):
        sample = sr.sample_env(sys_root=self.tmp / 'nope', proc_root=self.tmp / 'nope2')
        self.assertEqual(sample['thermal'], {})
        self.assertEqual(sample['cpufreq_mhz'], {})
        self.assertIsNone(sample['mem_available_mib'])
        self.assertIn('t', sample)


if __name__ == '__main__':
    unittest.main()


class RewriteDefaultsTest(unittest.TestCase):
    def test_oxe_defaults(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location('sr', Path(__file__).resolve().parents[1] / 'tools/session_replay.py')
        sr = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sr)
        body = {'messages': [], 'temperature': 0.55, 'stop': ['x']}
        out = sr.rewrite_body(body, 7, '/v1/chat/completions')
        self.assertEqual(out['chat_template_kwargs'], {'enable_thinking': False})
        self.assertEqual((out['temperature'], out['n_predict'], out['ignore_eos']), (0, 7, True))
        kept = sr.rewrite_body({'messages': [], 'chat_template_kwargs': {'enable_thinking': True}}, 3, '/v1/chat/completions')
        self.assertTrue(kept['chat_template_kwargs']['enable_thinking'])
        bare = sr.rewrite_body({'messages': []}, 3, '/v1/chat/completions', oxe_defaults=False)
        self.assertNotIn('chat_template_kwargs', bare)
        self.assertNotIn('temperature', body.get('chat_template_kwargs', {}))


class TemplateKwargArmTest(unittest.TestCase):
    def test_forced_template_kwargs(self):
        kw = sr.parse_template_kwargs(['preserve_thinking=true', 'n=3'])
        self.assertEqual(kw, {'preserve_thinking': True, 'n': 3})
        out = sr.rewrite_body({'messages': [], 'chat_template_kwargs': {'preserve_thinking': False}}, 3,
                              '/v1/chat/completions', template_kwargs=kw)
        self.assertEqual(out['chat_template_kwargs'], {'preserve_thinking': True, 'enable_thinking': False, 'n': 3})
        plain = sr.rewrite_body({'prompt': 'x'}, 3, '/completion', template_kwargs=kw)
        self.assertNotIn('chat_template_kwargs', plain)
        self.assertEqual(sr.parse_template_kwargs(None), {})
        for bad in ('novalue', '=1', 'k=not json'):
            with self.assertRaises(sr.UsageError):
                sr.parse_template_kwargs([bad])

    def test_token_counterfactual_parser_and_loopback_owner_only_auth(self):
        args = sr.build_parser().parse_args(['tokens', 'synthetic.jsonl', '--server', 'http://127.0.0.1:8098',
                                           '--out', 'tokens.jsonl', '--template-kwarg', 'preserve_thinking=true'])
        self.assertEqual(sr.parse_template_kwargs(args.template_kwarg), {'preserve_thinking': True})
        with tempfile.TemporaryDirectory() as d:
            key = Path(d) / 'key'
            key.write_text('a' * 32 + '\n')
            key.chmod(0o600)
            args.api_key_file = str(key)
            self.assertEqual(sr.local_auth_header(args, sr.parse_url(args.server)),
                             {'Authorization': 'Bearer ' + 'a' * 32})
            with self.assertRaises(sr.UsageError):
                sr.local_auth_header(args, sr.parse_url('https://example.invalid'))
            key.chmod(0o644)
            with self.assertRaises(sr.UsageError):
                sr.local_auth_header(args, sr.parse_url(args.server))


class RenderReplyTest(unittest.TestCase):
    def test_qwen_xml_tool_call(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location('sr2', Path(__file__).resolve().parents[1] / 'tools/session_replay.py')
        sr = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sr)
        got = sr.render_reply({'content': '', 'reasoning': '', 'tool_calls': [{'name': 'read', 'arguments': '{"filePath": "/p/calc.py"}'}]})
        # the engine's READ_CALL test fixture (crates/server/src/tests.rs)
        self.assertEqual(got, "<tool_call>\n<function=read>\n<parameter=filePath>\n/p/calc.py\n</parameter>\n</function>\n</tool_call>")
        self.assertEqual(sr.render_reply({'content': 'Done.', 'tool_calls': []}), 'Done.')


class ConcurrencyTest(unittest.TestCase):
    def test_overlap_share(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location('sr3', Path(__file__).resolve().parents[1] / 'tools/session_replay.py')
        sr = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sr)
        turns = [{'t_start': 0, 't_first_byte': 1, 't_end': 11},   # decode 10 s
                 {'t_start': 6, 't_first_byte': 7, 't_end': 9},    # decode 2 s, inside the first
                 {'t_start': 20, 't_first_byte': 20, 't_end': 28}] # serial, decode 8 s
        ov = sr.concurrency(turns)
        self.assertEqual((ov['requests'], ov['overlapping_requests']), (3, 2))
        self.assertAlmostEqual(ov['decode_s'], 20.0)
        # first: other in flight 6..9 -> 3 s; second: whole 7..9 -> 2 s
        self.assertAlmostEqual(ov['overlap_decode_s'], 5.0)
        self.assertAlmostEqual(ov['overlap_share_pct'], 25.0)
