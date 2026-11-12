"""Model-free tests for tools/opencode_compaction_live.py (no OpenCode, no controller, no hardware)."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location('opencode_compaction_live', ROOT / 'tools/opencode_compaction_live.py')
tool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tool)
launcher = tool.load_launcher()
MOE = ('qwen3.6-35b-a3b-local', 'ornith-35b-a3b-local')


def step(tokens, mid):
    return {'type': 'step-finish', 'messageID': mid, 'tokens': {'input': tokens, 'output': 0, 'reasoning': 0,
                                                                 'cache': {'read': 0, 'write': 0}}}


def text(chars, mid):
    return {'type': 'text', 'messageID': mid, 'text': 'x' * chars}


def export(parts):
    return {'info': {'id': 'ses'}, 'messages': [{'info': {'id': p.get('messageID')}, 'parts': [p]} for p in parts]}


class Plan(unittest.TestCase):
    def test_triggers_match_opencode_rule(self):
        self.assertEqual(tool.opencode_trigger(16384, 131072), 13927)
        self.assertEqual(tool.opencode_trigger(32768, 131072), 16768)
        self.assertEqual(tool.opencode_trigger(120000, 131072), 102000)
        self.assertEqual(tool.opencode_trigger(65536, 131072), 49536)
        self.assertEqual(tool.opencode_trigger(12288, 16384), 10445)
        with self.assertRaises(ValueError):
            tool.opencode_trigger(0, 0)

    def test_overlay_cuts_only_input_and_keeps_preset_and_isolation(self):
        for model in ('moe', 'ornith'):
            for window, size in tool.WINDOWS.items():
                with self.subTest(model=model, window=window):
                    overlay = tool.build_overlay(launcher, model, window)
                    models = overlay['providers']['local-qwen']['models']
                    for mid in MOE:
                        self.assertEqual(models[mid]['limit'], {'context': 131072, 'input': size, 'output': 16384})
                    for small in launcher.SMALL_CONTEXT_MODEL_IDS:
                        self.assertTrue(models[small]['disabled'])
                    self.assertEqual(overlay['compaction'], {'auto': True, 'keep': {'tokens': 8000}})
                    self.assertEqual(overlay['model'], launcher.CLASSIC_MODELS[model])
        for bad in (('flash', '16k'), ('dense', '16k'), ('moe', '64k')):
            with self.assertRaises(ValueError):
                tool.build_overlay(launcher, *bad)

    def test_commands_use_standalone_json_and_session(self):
        first = tool.commands('opencode', 'local-qwen/x', prompt='p')
        self.assertEqual(first['run'], ['opencode', 'run', '--standalone', '--format', 'json', '-m', 'local-qwen/x',
                                        '--archived', '--title', 'compaction-live', 'p'])
        later = tool.commands('opencode', 'local-qwen/x', session='ses_1', prompt='p')
        self.assertIn('--session', later['run'])
        self.assertNotIn('--archived', later['run'])
        self.assertEqual(later['export'], ['opencode', 'session', 'export', '--standalone', 'ses_1'])
        self.assertEqual(first['models'], ['opencode', 'models', '--standalone'])

    def test_filler_is_deterministic_and_sized(self):
        a, b = tool.filler_prompt(3, 1500), tool.filler_prompt(3, 1500)
        self.assertEqual(a, b)
        self.assertNotEqual(a, tool.filler_prompt(4, 1500))
        self.assertTrue(1500 * 4 <= len(a) <= 1500 * 4 + 400)
        self.assertIn('ACK 3', a)

    def test_dry_run_prints_plan_and_runs_nothing(self):
        def forbidden(*_a, **_k):
            raise AssertionError('dry run must not execute')
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(tool.main(['--model', 'ornith', '--window', '32k', '--dry-run'], runner=forbidden), 0)
        value = json.loads(out.getvalue())
        self.assertEqual((value['trigger'], value['keep'], value['model_ref']),
                         (16768, 8000, 'local-qwen/ornith-35b-a3b-local'))


class Parsing(unittest.TestCase):
    def test_v227_typed_export_counts_finished_steps_and_auto_compaction(self):
        doc = {'messages': [
            {'type': 'user', 'id': 'u1', 'text': 'x' * 6000},
            {'type': 'assistant', 'id': 'a1', 'finish': 'stop',
             'tokens': {'input': 3000, 'output': 4, 'cache': {'read': 10000}},
             'content': [{'type': 'text', 'text': 'ACK'}]},
            {'type': 'compaction', 'id': 'c1', 'status': 'completed', 'reason': 'auto',
             'metadata': {'keepID': 'u1'}, 'summary': 'A synthetic summary.'},
        ]}
        events, order = tool.analyse_export(doc)
        report = tool.compaction_report(events, order, 13927, 8000, 1500, 1024)
        self.assertEqual(report['failures'], [])
        self.assertEqual(report['last_measured_before'], 13004)
        self.assertEqual(report['tail_tokens_estimate'], 1501)
        self.assertTrue(report['tail_start_known'])
        doc['messages'][-1]['metadata']['keepID'] = None
        events, order = tool.analyse_export(doc)
        self.assertTrue(tool.compaction_report(events, order, 13927, 8000, 1500, 1024)['tail_explicitly_empty'])
        doc['messages'][-1]['reason'] = 'manual'
        events, order = tool.analyse_export(doc)
        self.assertIn('not_auto_threshold_compaction', tool.compaction_report(events, order, 13927, 8000, 1500, 1024)['failures'])
        doc['messages'][-1]['metadata'] = {}
        events, order = tool.analyse_export(doc)
        self.assertIn('compaction_tail_boundary_unknown', tool.compaction_report(events, order, 13927, 8000, 1500, 1024)['failures'])

    def test_v227_incomplete_or_embedded_compaction_is_not_a_completed_event(self):
        doc = {'messages': [
            {'type': 'user', 'id': 'u', 'text': '{"type":"compaction","auto":true}'},
            {'type': 'compaction', 'id': 'c', 'reason': 'auto', 'status': 'running'},
        ]}
        events, order = tool.analyse_export(doc)
        self.assertEqual(tool.compaction_report(events, order, 13927, 8000, 1500, 1024)['failures'], ['no_compaction'])

    def test_session_id_from_json_events(self):
        out = 'noise\n' + json.dumps({'type': 'step_start', 'sessionID': 'ses_9'}) + '\n'
        self.assertEqual(tool.session_from_events(out), 'ses_9')
        with self.assertRaises(RuntimeError):
            tool.session_from_events('')
        with self.assertRaisesRegex(RuntimeError, 'Model unavailable'):
            tool.session_from_events(json.dumps({'type': 'error', 'sessionID': '', 'error': {'message': 'Model unavailable'}}))

    def test_isolation_report(self):
        good = '\n'.join(f'local-qwen/{m}' for m in MOE) + '\nopencode/big-pickle\n'
        self.assertEqual(tool.isolation_report(good, launcher)['failures'], [])
        bad = good + 'local-qwen/qwen3.8-flash-next-local\n'
        self.assertEqual(tool.isolation_report(bad, launcher)['failures'], ['small_context_model_selectable'])
        self.assertEqual(tool.isolation_report('local-qwen/' + MOE[0], launcher)['failures'], ['moe_model_not_listed'])

    def _report(self, parts, trigger=13927, prompt=1500):
        events, order = tool.analyse_export(export(parts))
        return tool.compaction_report(events, order, trigger, 8000, prompt, 1024)

    def test_compaction_at_threshold_with_tail_within_keep(self):
        parts = [text(6000, 'm1'), step(7000, 'm2'), text(6000, 'm3'), step(10500, 'm4'),
                 text(6000, 'm5'), step(13000, 'm6'),
                 {'type': 'compaction', 'auto': True, 'tail_start_id': 'm5', 'messageID': 'm7'}]
        report = self._report(parts)
        self.assertTrue(report['compacted'])
        self.assertEqual(report['failures'], [])
        self.assertEqual(report['tail_tokens_estimate'], 1500)
        self.assertEqual(report['last_measured_before'], 13000)

    def test_detects_missing_early_late_overflow_and_fat_tail(self):
        self.assertEqual(self._report([step(9000, 'm1')])['failures'], ['no_compaction'])
        early = [step(4000, 'm1'), {'type': 'compaction', 'auto': True, 'messageID': 'm2'}]
        self.assertIn('compacted_early', self._report(early)['failures'])
        late = [step(15500, 'm1'), step(16000, 'm2'), {'type': 'compaction', 'auto': True, 'messageID': 'm3'}]
        self.assertIn('compacted_late', self._report(late)['failures'])
        overflow = [step(14000, 'm1'), {'type': 'compaction', 'auto': True, 'overflow': True, 'messageID': 'm2'}]
        self.assertIn('not_auto_threshold_compaction', self._report(overflow)['failures'])
        fat = [text(40000, 'm1'), step(14000, 'm2'), {'type': 'compaction', 'auto': True, 'tail_start_id': 'm1', 'messageID': 'm3'}]
        self.assertIn('tail_exceeds_keep', self._report(fat)['failures'])


class Ledger(unittest.TestCase):
    ROWS = [{'turn_kind': 'chat', 'miss_class': 'append-ok', 'cache_n': 9000},
            {'turn_kind': 'compaction', 'cache_n': 12000},
            {'turn_kind': 'post-compaction', 'compact_layout': 'ok', 'cache_n': 5600, 'miss_class': 'compaction'},
            {'turn_kind': 'post-compaction', 'compact_layout': 'ok', 'cache_n': 9000, 'miss_class': 'append-ok'}]

    def test_good_sequence_passes(self):
        report = tool.ledger_report(self.ROWS, 4000)
        self.assertEqual(report['failures'], [])
        self.assertEqual((report['post_compaction_cache_n'], report['next_miss_class']), (5600, 'append-ok'))

    def test_each_failure_is_named(self):
        def mutate(index, **change):
            rows = [dict(r) for r in self.ROWS]
            rows[index].update(change)
            return tool.ledger_report(rows, 4000)['failures']
        self.assertEqual(tool.ledger_report(self.ROWS[:1], 4000)['failures'], ['ledger_no_compaction_turn'])
        self.assertIn('post_compaction_layout_not_ok', mutate(2, compact_layout='summary-moved'))
        self.assertIn('post_compaction_prefix_miss', mutate(2, cache_n=100))
        self.assertIn('post_compaction_unknown_miss', mutate(2, miss_class='unknown'))
        self.assertIn('c3_not_restored', mutate(3, miss_class='history-edit'))
        self.assertIn('unknown_miss_in_window', mutate(0, miss_class='unknown'))
        self.assertIn('ledger_no_second_post_compaction_turn', tool.ledger_report(self.ROWS[:3], 4000)['failures'])

    def test_rows_filtered_by_time_model_and_path(self):
        with tempfile.TemporaryDirectory() as d:
            rows = [{'t': 5.0, 'model': MOE[1], 'path': '/v1/chat/completions', 'turn_kind': 'chat'},
                    {'t': 1.0, 'model': MOE[1], 'path': '/v1/chat/completions'},
                    {'t': 6.0, 'model': MOE[0], 'path': '/v1/chat/completions'},
                    {'t': 7.0, 'model': MOE[1], 'path': '/v1/models'},
                    {'t': 4.0, 'model': MOE[1], 'path': '/v1/chat/completions', 'turn_kind': 'compaction'}]
            Path(d, 'requests-20261112.jsonl').write_text('\n'.join(map(json.dumps, rows)) + '\nnot json\n')
            got = tool.ledger_rows(d, 2.0, MOE[1])
        self.assertEqual([r['t'] for r in got], [4.0, 5.0])


class LiveWithFakeRunner(unittest.TestCase):
    def test_live_loop_stops_two_turns_after_compaction_and_passes(self):
        calls, state = [], {'turn': 0}

        def runner(command, env, cwd, timeout):
            calls.append(command)
            overlay = json.loads(env['OPENCODE_CONFIG_CONTENT'])
            self.assertEqual(overlay['compaction']['keep']['tokens'], 8000)
            if command[1:] == ['--version']:
                return subprocess.CompletedProcess(command, 0, '2.2.7\n', '')
            if command[1] == 'models':
                return subprocess.CompletedProcess(command, 0, '\n'.join(f'local-qwen/{m}' for m in MOE), '')
            if command[1] == 'run':
                state['turn'] += 1
                return subprocess.CompletedProcess(command, 0, json.dumps({'type': 'text', 'sessionID': 'ses_1'}), '')
            if command[1] == 'session':
                n = state['turn']
                parts = [step(2000 + 2200 * i, f'm{i}') for i in range(1, min(n, 6) + 1)]
                if n >= 6:
                    parts.append({'type': 'compaction', 'auto': True, 'messageID': 'mc'})
                return subprocess.CompletedProcess(command, 0, json.dumps(export(parts)), '')
            raise AssertionError(command)

        with tempfile.TemporaryDirectory() as d:
            rows = [dict(r, t=10**12, model=MOE[1], path='/v1/chat/completions') for r in Ledger.ROWS]
            Path(d, 'requests-29991231.jsonl').write_text('\n'.join(map(json.dumps, rows)) + '\n')
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = tool.main(['--model', 'ornith', '--window', '16k', '--ledger-dir', d], runner=runner,
                                 status=lambda url, key: {'selected_model': 'ornith'}, environment={})
        receipt = json.loads(out.getvalue())
        self.assertEqual(receipt['failures'], [])
        self.assertEqual((receipt['mode'], receipt['origin'], receipt['controller_url']),
                         ('fast', 'lab', 'http://127.0.0.1:8092/v1'))
        self.assertEqual(code, 0)
        self.assertEqual(receipt['turns'], 8)
        runs = [c for c in calls if c[1] == 'run']
        self.assertIn('--archived', runs[0])
        self.assertTrue(all('--session' in c for c in runs[1:]))

    def test_wrong_selected_model_refuses_before_any_prompt(self):
        def runner(command, env, cwd, timeout):
            if command[1:] == ['--version']:
                return subprocess.CompletedProcess(command, 0, '2.2.7', '')
            raise AssertionError('must not prompt')
        with self.assertRaisesRegex(RuntimeError, 'select_the_model'):
            tool.main(['--model', 'moe', '--window', '16k'], runner=runner,
                      status=lambda url, key: {'selected_model': 'flash'}, environment={})


class Bindings(unittest.TestCase):
    """Codex 17:11/17:20: REAL-1 origin, loopback test endpoint, fresh token, inherited base, mode."""
    def test_overlay_marks_lab_origin_and_points_at_the_controller(self):
        overlay = tool.build_overlay(launcher, 'ornith', '16k', url='http://127.0.0.1:18092/v1')
        settings = overlay['providers']['local-qwen']['settings']
        self.assertEqual(settings['headers'], {'X-Local-AI-Origin': 'lab'})
        self.assertEqual(settings['baseURL'], 'http://127.0.0.1:18092/v1')
        self.assertNotIn('apiKey', settings)

    def test_controller_url_is_loopback_only(self):
        ok = {'http://127.0.0.1:8092/v1': 'http://127.0.0.1:8092/v1', 'http://127.0.0.1:18092/v1/': 'http://127.0.0.1:18092/v1',
              'http://localhost:18096': 'http://localhost:18096/v1', 'http://[::1]:18092/v1': 'http://[::1]:18092/v1',
              'http://127.0.0.2:9000/v1': 'http://127.0.0.2:9000/v1'}
        for url, want in ok.items():
            self.assertEqual(tool.controller_url(url), want)
        for bad in ('http://192.168.1.5:8092/v1', 'http://example.com:8092/v1', 'http://127.0.0.1/v1',
                    'ftp://127.0.0.1:8092/v1', 'http://user:pw@127.0.0.1:8092/v1', 'http://127.0.0.1:8092/v2',
                    'http://127.0.0.1:8092/v1?x=1', 'http://0.0.0.0:8092/v1', 'http://127.0.0.1:8092/v1\n', None):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                tool.controller_url(bad)
        with self.assertRaises(ValueError):
            tool.main(['--model', 'moe', '--window', '16k', '--dry-run', '--controller-url', 'http://10.0.0.1:8092/v1'],
                      environment={})

    def test_mode_defaults_to_fast_and_deep_is_explicit_window_unchanged(self):
        for mode in ('fast', 'deep'):
            overlay = tool.build_overlay(launcher, 'moe', '32k', mode)
            for mid in MOE:
                self.assertEqual(overlay['providers']['local-qwen']['models'][mid]['limit'],
                                 {'context': 131072, 'input': 32768, 'output': 16384})
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            tool.main(['--model', 'moe', '--window', '16k', '--dry-run'], environment={})
        self.assertEqual(json.loads(out.getvalue())['mode'], 'fast')
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            tool.main(['--model', 'moe', '--window', '16k', '--dry-run', '--mode', 'deep'], environment={})
        self.assertEqual(json.loads(out.getvalue())['mode'], 'deep')
        with self.assertRaises(ValueError):
            tool.build_overlay(launcher, 'moe', '16k', 'medium')

    def test_inherited_base_is_kept(self):
        base = json.dumps({'unrelated': {'keep': True}, 'providers': {'local-qwen': {'settings': {'timeout': 5}}}})
        overlay = tool.build_overlay(launcher, 'ornith', '16k', base=base)
        self.assertEqual(overlay['unrelated'], {'keep': True})
        self.assertEqual(overlay['providers']['local-qwen']['settings']['timeout'], 5)
        self.assertEqual(overlay['providers']['local-qwen']['settings']['headers']['X-Local-AI-Origin'], 'lab')

    def test_live_merges_inherited_base_and_uses_one_target(self):
        seen = {}

        def runner(command, env, cwd, timeout):
            seen['overlay'] = json.loads(env['OPENCODE_CONFIG_CONTENT'])
            if command[1:] == ['--version']:
                return subprocess.CompletedProcess(command, 0, '2.2.7', '')
            raise AssertionError('stop after the preflight')

        def status(url, key):
            seen['status'] = (url, key)
            return {'selected_model': 'flash'}
        environment = {'OPENCODE_CONFIG_CONTENT': json.dumps({'unrelated': 1}), 'GATE_TOKEN': 'sekrit-token'}
        with self.assertRaisesRegex(RuntimeError, 'select_the_model'):
            tool.main(['--model', 'moe', '--window', '16k', '--controller-url', 'http://127.0.0.1:18092/v1',
                       '--api-key-env', 'GATE_TOKEN'], runner=runner, status=status, environment=environment)
        self.assertEqual(seen['overlay']['unrelated'], 1)
        self.assertEqual(seen['overlay']['providers']['local-qwen']['settings']['baseURL'], 'http://127.0.0.1:18092/v1')
        self.assertEqual(seen['overlay']['providers']['local-qwen']['settings']['apiKey'], 'sekrit-token')
        self.assertEqual(seen['status'], ('http://127.0.0.1:18092/v1', 'sekrit-token'))

    def test_key_never_printed_in_dry_run_or_receipt(self):
        environment = {'GATE_TOKEN': 'sekrit-token'}
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            tool.main(['--model', 'ornith', '--window', '16k', '--dry-run', '--api-key-env', 'GATE_TOKEN'],
                      environment=environment)
        self.assertNotIn('sekrit-token', out.getvalue())
        self.assertEqual(json.loads(out.getvalue())['overlay']['providers']['local-qwen']['settings']['apiKey'], '<redacted>')
        with self.assertRaises(ValueError):
            tool.main(['--model', 'ornith', '--window', '16k', '--dry-run', '--api-key-env', 'MISSING'], environment={})
        state = {'turn': 0}

        def runner(command, env, cwd, timeout):
            if command[1:] == ['--version']:
                return subprocess.CompletedProcess(command, 0, '2.2.7', '')
            if command[1] == 'models':
                return subprocess.CompletedProcess(command, 0, '\n'.join(f'local-qwen/{m}' for m in MOE), '')
            if command[1] == 'run':
                state['turn'] += 1
                return subprocess.CompletedProcess(command, 0, json.dumps({'sessionID': 'ses_1'}), '')
            return subprocess.CompletedProcess(command, 0, json.dumps(export([step(100, 'm1')])), '')
        with tempfile.TemporaryDirectory() as d:
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                tool.main(['--model', 'ornith', '--window', '16k', '--ledger-dir', d, '--max-turns', '2',
                           '--api-key-env', 'GATE_TOKEN', '--out', str(Path(d) / 'r.json')], runner=runner,
                          status=lambda url, key: {'selected_model': 'ornith-35b-a3b-local'}, environment=environment)
            self.assertNotIn('sekrit-token', out.getvalue())
            self.assertNotIn('sekrit-token', Path(d, 'r.json').read_text())

    def test_status_request_is_lab_and_targets_the_controller(self):
        captured = {}

        class Response(io.BytesIO):
            def __enter__(self): return self
            def __exit__(self, *a): return False

        def urlopen(request, timeout):
            captured['url'], captured['headers'] = request.full_url, dict(request.header_items())
            return Response(b'{"selected_model": "moe"}')
        from unittest.mock import patch
        with patch.object(tool.urllib.request, 'urlopen', urlopen):
            value = tool.controller_status('http://127.0.0.1:18092/v1', 'tok')
        self.assertEqual(value, {'selected_model': 'moe'})
        self.assertEqual(captured['url'], 'http://127.0.0.1:18092/_local/status')
        self.assertEqual(captured['headers'].get('X-local-ai-origin'), 'lab')
        self.assertEqual(captured['headers'].get('Authorization'), 'Bearer tok')


if __name__ == '__main__':
    unittest.main()
