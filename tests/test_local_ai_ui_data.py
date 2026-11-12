"""UI data commands of local-ai: status keys, harness and model rows, ledger, storage, harness actions.

No network and no live controller: request() and the paths are replaced or point at temporary files.
The cache commands run against a fake controller on 127.0.0.1 with an ephemeral port.
"""
import contextlib
import http.server
import importlib.machinery
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / 'src/snapdragon_ai_lab/runtime/local-ai'
# SourceFileLoader, not runpy: runpy returns a copy of the globals, so patches would not reach the functions.
loader = importlib.machinery.SourceFileLoader('ui_data_local_ai', str(SCRIPT))
spec = importlib.util.spec_from_loader(loader.name, loader)
cli = importlib.util.module_from_spec(spec)
loader.exec_module(cli)

ORNITH, MOE, DENSE, FLASH = 'ornith-35b-a3b-local', 'qwen3.6-35b-a3b-local', 'qwen3.8-27b-local', 'qwen3.8-flash-next-local'
IDLE_TRIAL = dict(flash_trial_active=False, flash_trial_kind='', flash_trial_elapsed_seconds=0)
OLD_KEYS = {
    'chat_state', 'active_requests', 'idle_seconds', 'compute_busy', 'model', 'selected_model', 'prefix_cache_enabled',
    'backend', 'npu_status', 'image_state', 'image_model', 'available_gib', 'flash_available', 'ornith_available',
    'execution_policy_mode', 'flash_download_state', 'flash_download_bytes', 'flash_download_total_bytes',
    'flash_download_shards', 'flash_verified_shards', 'flash_download_active', 'flash_waiting_verification',
    'flash_trial_active', 'flash_trial_kind', 'flash_trial_elapsed_seconds',
}
STATUS = cli.CHAT + '/_local/status'
MODELS = cli.CHAT + '/_local/models'


class Controller(unittest.TestCase):
    """Fake controller answers, fixed local files, and a request() mock that never leaves the process."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.meminfo = root / 'meminfo'
        self.meminfo.write_text('MemTotal:       33554432 kB\nMemAvailable:   16777216 kB\n')
        (root / 'flash-state').mkdir()
        self.answers = {
            cli.CHAT + '/v1/models': (200, {'object': 'list', 'data': [
                {'id': FLASH, 'object': 'model'}, {'id': ORNITH, 'object': 'model'}]}),
            cli.IMAGES + '/activity': (200, {'state': 'idle'}),
        }
        self.start(patch.object(cli, 'MEMINFO', self.meminfo))
        self.start(patch.object(cli, 'FLASH_STATE', root / 'flash-state'))
        self.start(patch.object(cli, 'LEASE', root / 'lease'))
        self.start(patch.object(cli, 'NPU_STATUS', root / 'npu.json'))
        self.start(patch.object(cli, 'flash_trial_status', return_value=dict(IDLE_TRIAL)))
        self.request = self.start(patch.object(cli, 'request', side_effect=self.fake_request))

    def start(self, patcher):
        started = patcher.start()
        self.addCleanup(patcher.stop)
        return started

    def fake_request(self, url, post=False, timeout=2, payload=None):
        return self.answers.get(url, (0, {}))

    def serve(self, body, code=200, models=None):
        if models is not None:
            body = {**body, 'capabilities': ['models']}
        self.answers[STATUS] = (code, body)
        self.answers[MODELS] = (200, {'data': models}) if models is not None else (404, {})


class Status(Controller):
    def test_production_controller_adds_only_total_and_has_error(self):
        self.serve({'state': 'ready', 'model': MOE, 'selected_model': MOE, 'active_requests': 0,
                    'idle_unload_seconds': 600, 'backend': 'cpu', 'prefix_cache': {'enabled': True},
                    'execution_policy_mode': 'stock'})
        result = cli.status()
        self.assertEqual(set(result), OLD_KEYS | {'total_gib', 'has_error'})
        self.assertEqual((result['total_gib'], result['has_error'], result['model']), (32.0, False, 'moe'))
        self.assertEqual(result['available_gib'], 16.0)

    def test_valid_controller_values_are_added(self):
        self.serve({'state': 'ready', 'model': MOE, 'queued_requests': 3, 'context_size': 16384,
                    'loaded_model': MOE, 'resident_bytes': 5 * 1024**3, 'idle_unload_remaining_seconds': 42.5,
                    'last_error': {'code': 'x'}}, models=[])
        result = cli.status()
        self.assertEqual(result['queued_requests'], 3)
        self.assertEqual(result['context_size'], 16384)
        self.assertEqual(result['loaded_model'], 'moe')
        self.assertEqual(result['resident_gib'], 5.0)
        self.assertEqual(result['idle_remaining_seconds'], 42.5)
        self.assertTrue(result['has_error'])
        self.assertEqual(result['models'], [])

    def test_loaded_model_uses_the_alias_map(self):
        self.serve({'loaded_model': FLASH})
        self.assertEqual(cli.status()['loaded_model'], 'flash')
        self.serve({'loaded_model': ORNITH})
        self.assertEqual(cli.status()['loaded_model'], 'ornith')
        self.serve({'loaded_model': ['unhashable']})
        self.assertIsNone(cli.status()['loaded_model'])

    def test_invalid_values_are_omitted_and_null_or_unknown_is_none(self):
        self.serve({'queued_requests': 129, 'context_size': True, 'loaded_model': 'unknown-x',
                    'resident_bytes': 0, 'idle_unload_remaining_seconds': 90000, 'last_error': ''})
        result = cli.status()
        for key in ('queued_requests', 'context_size', 'resident_gib', 'idle_remaining_seconds'):
            with self.subTest(key=key):
                self.assertNotIn(key, result)
        self.assertIsNone(result['loaded_model'])
        self.assertFalse(result['has_error'])
        self.serve({'resident_bytes': None, 'idle_unload_remaining_seconds': None, 'loaded_model': None})
        result = cli.status()
        self.assertEqual((result['resident_gib'], result['idle_remaining_seconds'], result['loaded_model']),
                         (None, None, None))

    def test_loaded_model_absent_when_controller_does_not_send_it(self):
        self.serve({'state': 'ready'})
        self.assertNotIn('loaded_model', cli.status())

    def test_controller_down_has_no_error_key_and_no_models(self):
        self.serve({}, code=500)
        result = cli.status()
        self.assertEqual(result['chat_state'], 'offline')
        self.assertNotIn('has_error', result)
        self.assertNotIn('models', result)
        self.assertEqual(result['total_gib'], 32.0)

    def test_status_calls_use_one_second_timeout(self):
        self.serve({'state': 'ready'}, models=[])
        cli.status()
        self.request.assert_any_call(MODELS, timeout=1)
        self.request.assert_any_call(STATUS, timeout=1)

    def test_hostile_strings_never_reach_the_output(self):
        self.serve({
            'harnesses': {'<img src=x>': {'last_used': 5, 'prefixes': []}, 'ok-name': {'last_used': 1, 'prefixes': []}},
            'last_error': '<script>alert(1)</script>',
        }, models=[{'key': '<b>', 'id': 'nope', 'engine': '<i>'}, {'id': MOE, 'engine': 'rust', 'name': '<u>x</u>'}])
        result = cli.status()
        text = json.dumps(result)
        for bad in ('<img', '<script', 'alert', '<b>', '<i>', '<u>', 'nope'):
            with self.subTest(text=bad):
                self.assertNotIn(bad, text)
        self.assertEqual([row['name'] for row in result['harnesses']], ['ok-name'])
        self.assertEqual([row['key'] for row in result['models']], ['moe'])
        self.assertTrue(result['has_error'])

    def test_harness_rows_flattened_and_ordered(self):
        self.serve({'harnesses': {
            'pi': {'warm': 1, 'last_used': 100.0, 'prefixes': [
                {'key': 'a', 'warm': False, 'last_used': 90.0, 'tools': 1, 'last_cache_n': 5, 'last_prompt_n': 6},
                {'key': 'b', 'warm': True, 'last_used': 99.0, 'tools': 4, 'last_cache_n': 1554, 'last_prompt_n': 20}]},
            'dsh': {'warm': 0, 'last_used': 200.0, 'prefixes': [{'key': 'c', 'warm': False, 'last_used': 200.0, 'tools': 2}]},
            'other': {'last_used': 300.0, 'prefixes': []},
            'bad name': {'last_used': 400.0, 'prefixes': []},
            'a' * 33: {'last_used': 401.0, 'prefixes': []},
            'idle': {'last_used': 50.0, 'prefixes': []},
        }})
        self.assertEqual(cli.status()['harnesses'], [
            dict(name='dsh', warm=False, last_used=200.0, tools=2, cache_n=0, prompt_n=0, prefixes=1),
            dict(name='pi', warm=True, last_used=100.0, tools=4, cache_n=1554, prompt_n=20, prefixes=2),
            dict(name='idle', warm=False, last_used=50.0, tools=0, cache_n=0, prompt_n=0, prefixes=0),
        ])

    def test_harness_rows_capped_at_twelve_newest(self):
        self.serve({'harnesses': {f'h{i}': {'last_used': float(i), 'prefixes': []} for i in range(14)}})
        names = [row['name'] for row in cli.status()['harnesses']]
        self.assertEqual(len(names), 12)
        self.assertEqual(names[0], 'h13')

    def test_harnesses_key_absent_when_not_a_dict(self):
        self.serve({'harnesses': ['pi']})
        self.assertNotIn('harnesses', cli.status())

    def test_model_rows_reduced_to_allowlisted_values(self):
        self.serve({'state': 'ready'}, models=[
            {'id': ORNITH, 'engine': 'rust', 'engine_status': 'production', 'memory_gib': 19.2,
             'battery_start_percent': 30, 'prefix': 'warm', 'selected': True, 'default': True, 'state': 'ready',
             'name': 'secret'},
            {'id': MOE, 'engine': 'evil', 'engine_status': 'x', 'memory_gib': 5000, 'battery_start_percent': 150,
             'prefix': 'hot', 'selected': 'yes', 'default': False, 'state': 'bogus'},
            {'id': 'unknown-alias', 'engine': 'rust'},
            {'key': 'flash', 'engine': 'stream', 'memory_gib': 25, 'battery_start_percent': 0.0},
        ])
        rows = {row['key']: row for row in cli.status()['models']}
        self.assertEqual(rows['ornith'], dict(key='ornith', engine='rust', engine_status='production', memory_gib=19.2,
            battery_start_percent=30, prefix='warm', selected=True, default=True, state='ready'))
        self.assertEqual(rows['moe'], dict(key='moe', engine=None, engine_status=None, memory_gib=None,
            battery_start_percent=None, prefix=None, selected=False, default=False, state=None))
        self.assertEqual(rows['flash'], dict(key='flash', engine='stream', engine_status=None, memory_gib=25,
            battery_start_percent=None, prefix=None, selected=False, default=False, state=None))
        self.assertEqual(sorted(rows), ['flash', 'moe', 'ornith'])

    def test_models_key_absent_unless_controller_answers_with_data_list(self):
        self.serve({'state': 'ready'})
        self.assertNotIn('models', cli.status())
        self.answers[MODELS] = (200, {'data': 'nope'})
        self.assertNotIn('models', cli.status())

    def test_models_route_is_not_probed_without_the_capability(self):
        # A controller without the route would forward the GET to the loaded engine.
        self.serve({'state': 'ready'})
        self.answers[MODELS] = (200, {'data': [{'id': ORNITH, 'engine': 'rust'}]})
        self.assertNotIn('models', cli.status())
        self.assertNotIn(MODELS, [call.args[0] for call in self.request.call_args_list])
        self.serve({'state': 'ready', 'capabilities': 'models'})
        self.assertNotIn('models', cli.status())


class Ledger(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.directory = self.root / 'ledger'
        patcher = patch.dict(os.environ, {'LOCAL_AI_LEDGER_DIR': str(self.directory)})
        patcher.start()
        self.addCleanup(patcher.stop)

    def line(self, t, **fields):
        row = {'t': t, 'path': '/v1/chat/completions', 'status': 200, 'stream': True, 'model': ORNITH,
               'client': 'opencode/2.2.6', 'queue_wait_ms': 1.0, 'admit_ms': 0.0, 'ttfb_ms': 90.0,
               'wall_ms': 500.0, 'cache_n': 10, 'prompt_n': 20, 'prompt_ms': 30.0, 'predicted_n': 40,
               'predicted_ms': 800.0, 'finish_reason': 'stop'}
        row.update(fields)
        return json.dumps(row)

    def write(self, lines, days_ago=0):
        self.directory.mkdir(exist_ok=True)
        stamp = time.strftime('%Y%m%d', time.localtime(time.time() - 86400 * days_ago))
        (self.directory / f'requests-{stamp}.jsonl').write_text('\n'.join(lines) + '\n')

    def times(self, limit=50):
        return [row['t'] for row in cli.ledger_rows(limit)['rows']]

    def test_missing_directory(self):
        self.assertEqual(cli.ledger_rows(50), {'ledger_available': False, 'rows': []})

    def test_directory_without_files_is_available_and_empty(self):
        self.directory.mkdir()
        self.assertEqual(cli.ledger_rows(50), {'ledger_available': True, 'rows': []})

    def test_newest_first_and_limit(self):
        self.write([self.line(t) for t in (1, 2, 3, 4, 5)])
        self.assertEqual(self.times(), [5, 4, 3, 2, 1])
        self.assertEqual(self.times(2), [5, 4])

    def test_yesterday_is_read_only_when_today_is_short(self):
        self.write([self.line(1), self.line(2)], days_ago=1)
        self.write([self.line(3)])
        self.assertEqual(self.times(5), [3, 2, 1])
        self.assertEqual(self.times(1), [3])

    def test_bad_lines_are_skipped(self):
        self.write([self.line(1), 'not json', '[1, 2]', '"text"', '', self.line(2),
                    '{"t": 1, "path": 5}', '{"', '[' * 100000])
        self.assertEqual(self.times(), [2, 1])

    def test_only_completion_paths_count(self):
        self.write([self.line(1, path='/v1/models'), self.line(2, path='/completion'),
                    self.line(3, path='/v1/completions'), self.line(4, path='/v1/embeddings'),
                    self.line(5, path='/completions/extra')])
        self.assertEqual(self.times(), [3, 2])

    def test_row_mapping_is_exact(self):
        self.write([self.line(7, harness='pi', client='opencode/2.2.6', finish_reason='length', stream=False)])
        self.assertEqual(cli.ledger_rows(5)['rows'], [dict(t=7, model='ornith', harness='pi', status=200,
            stream=False, queue_wait_ms=1.0, admit_ms=0.0, ttfb_ms=90.0, wall_ms=500.0, cache_n=10, prompt_n=20,
            prompt_ms=30.0, predicted_n=40, predicted_ms=800.0, finish='length')])

    def test_model_alias_mapping(self):
        aliases = [DENSE, MOE, ORNITH, FLASH, 'other-model', None]
        self.write([self.line(i + 1, model=alias) for i, alias in enumerate(aliases)])
        self.assertEqual([row['model'] for row in cli.ledger_rows(50)['rows']],
            [None, None, 'flash', 'ornith', 'moe', 'dense'])

    def test_harness_falls_back_to_client_product_name(self):
        self.write([
            self.line(1, harness='pi'),
            self.line(2),
            self.line(3, harness='<img>', client='pi/1.0'),
            self.line(4, harness='bad name', client='<x>/1'),
            self.line(5, client=None),
            self.line(6, harness='a' * 33, client='a' * 33),
        ])
        self.assertEqual({row['t']: row['harness'] for row in cli.ledger_rows(50)['rows']},
            {1: 'pi', 2: 'opencode', 3: 'pi', 4: None, 5: None, 6: None})

    def test_finish_and_number_checks(self):
        self.write([
            self.line(1, finish_reason='tool_calls'), self.line(2, finish_reason='weird'),
            self.line(3, finish_reason=None), self.line(4, stream='yes', wall_ms=-5, ttfb_ms=float('nan')),
        ])
        rows = {row['t']: row for row in cli.ledger_rows(50)['rows']}
        self.assertEqual(rows[1]['finish'], 'tool_calls')
        self.assertIsNone(rows[2]['finish'])
        self.assertIsNone(rows[3]['finish'])
        self.assertEqual((rows[4]['stream'], rows[4]['wall_ms'], rows[4]['ttfb_ms']), (None, None, None))

    def test_symlinked_file_is_refused(self):
        outside = self.root / 'outside.jsonl'
        outside.write_text(self.line(9) + '\n')
        self.directory.mkdir()
        stamp = time.strftime('%Y%m%d')
        os.symlink(outside, self.directory / f'requests-{stamp}.jsonl')
        self.assertEqual(cli.ledger_rows(50), {'ledger_available': True, 'rows': []})

    def test_only_the_last_512_kib_of_a_large_file_is_read(self):
        self.write([self.line(t, pad='x' * 4000) for t in range(150)])
        self.assertGreater((self.directory / f'requests-{time.strftime("%Y%m%d")}.jsonl').stat().st_size, 512 * 1024)
        times = self.times(200)
        self.assertLess(len(times), 150)
        self.assertEqual(times[0], 149)
        self.assertGreater(min(times), 0)
        self.assertEqual(times, list(range(149, 149 - len(times), -1)))


class LedgerCli(unittest.TestCase):
    def run_cli(self, *args, ledger=None):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        home = Path(temporary.name)
        env = {'HOME': str(home), 'PATH': os.environ.get('PATH', ''),
               'LOCAL_AI_LEDGER_DIR': str(home / 'ledger')}
        if ledger is not None:
            (home / 'ledger').mkdir()
            stamp = time.strftime('%Y%m%d')
            (home / 'ledger' / f'requests-{stamp}.jsonl').write_text(
                '\n'.join(json.dumps({'t': t, 'path': '/v1/chat/completions', 'status': 200}) for t in ledger) + '\n')
        return subprocess.run([sys.executable, '-I', str(SCRIPT), *args], env=env,
            capture_output=True, text=True, timeout=60)

    def test_json_output_and_limit_clamp(self):
        result = self.run_cli('ledger', '--json', '--limit', '0', ledger=[1, 2, 3])
        self.assertEqual(result.returncode, 0)
        value = json.loads(result.stdout)
        self.assertTrue(value['ledger_available'])
        self.assertEqual([row['t'] for row in value['rows']], [3])

    def test_text_output(self):
        result = self.run_cli('ledger', '--limit', '2', ledger=[1, 2, 3])
        self.assertEqual((result.returncode, result.stdout.strip()), (0, '2 requests recorded'))

    def test_no_ledger_is_exit_zero(self):
        result = self.run_cli('ledger')
        self.assertEqual((result.returncode, result.stdout.strip()), (0, 'No request ledger'))
        result = self.run_cli('ledger', '--json')
        self.assertEqual(json.loads(result.stdout), {'ledger_available': False, 'rows': []})

    def test_harness_prepare_rejects_bad_name_before_any_work(self):
        result = self.run_cli('harness-prepare', '--harness', 'a b')
        self.assertEqual(result.returncode, 2)


class HarnessActions(Controller):
    def test_prepare_rejects_bad_names_without_side_effects(self):
        with patch.object(cli, 'ensure_service') as ensure:
            for bad in (None, '', 'a b', '<img src=x>', 'a' * 33):
                with self.subTest(harness=bad):
                    self.assertEqual(cli.action('harness-prepare', harness=bad), 2)
            ensure.assert_not_called()
        self.request.assert_not_called()

    def test_prepare_posts_and_maps_status_to_exit_code(self):
        url = cli.CHAT + '/_local/harness/prepare'
        self.answers[url] = (200, {})
        with patch.object(cli, 'ensure_service', return_value=True), patch.object(cli, 'notify') as notify:
            self.assertEqual(cli.action('harness-prepare', harness='pi'), 0)
            self.request.assert_called_once_with(url, post=True, timeout=300, payload={'harness': 'pi'})
            self.answers[url] = (409, {})
            self.assertEqual(cli.action('harness-prepare', harness='pi'), 1)
            notify.assert_not_called()

    def test_prepare_needs_the_chat_service(self):
        with patch.object(cli, 'ensure_service', return_value=False), patch.object(cli, 'notify') as notify:
            self.assertEqual(cli.action('harness-prepare', harness='pi'), 1)
        self.request.assert_not_called()
        notify.assert_not_called()

    def test_forget_with_and_without_a_name(self):
        url = cli.CHAT + '/_local/harness/forget'
        self.answers[url] = (200, {})
        self.assertEqual(cli.action('harness-forget', harness='pi'), 0)
        self.request.assert_called_with(url, post=True, timeout=15, payload={'harness': 'pi'})
        self.assertEqual(cli.action('harness-forget'), 0)
        self.request.assert_called_with(url, post=True, timeout=15, payload={})
        self.answers[url] = (503, {})
        self.assertEqual(cli.action('harness-forget'), 1)

    def test_forget_rejects_bad_name(self):
        self.assertEqual(cli.action('harness-forget', harness='<x>'), 2)
        self.request.assert_not_called()


class Storage(unittest.TestCase):
    def test_storage_rows_are_sanitized_and_delete_validates_id(self):
        rows = [{'id': 'rust-weights-0123456789abcdef', 'kind': 'rust-weights', 'state': 'unreferenced', 'deletable': True,
                 'bytes': 5, 'name': 'a.bin', 'model': '', 'paths': ['/x']},
                {'id': 'rust-prefix-0123456789abcdef', 'kind': 'rust-prefix', 'state': 'referenced', 'deletable': True, 'bytes': 1},
                {'id': '../bad', 'kind': 'rust-weights', 'state': 'unreferenced'}]
        with patch.object(cli, 'request', return_value=(200, {'data': rows, 'capabilities': ['storage']})):
            value = cli.storage_status()
        self.assertTrue(value['storage_available'])
        self.assertEqual(len(value['entries']), 2)
        self.assertNotIn('paths', value['entries'][0])
        self.assertFalse(value['entries'][1]['deletable'])
        with patch.object(cli, 'ensure_service', return_value=True), patch.object(cli, 'request') as request:
            self.assertEqual(cli.action('storage-delete', '../bad'), 2)
            self.assertEqual(cli.action('storage-delete'), 2)
            request.assert_not_called()

    def test_storage_unavailable_when_controller_does_not_answer(self):
        with patch.object(cli, 'request', return_value=(0, {})):
            self.assertEqual(cli.storage_status(), {'storage_available': False, 'entries': []})

    def test_storage_route_is_not_probed_without_the_capability(self):
        with patch.object(cli, 'request', return_value=(200, {'data': []})) as request:
            self.assertEqual(cli.storage_status(), {'storage_available': False, 'entries': []})
        request.assert_called_once_with(cli.CHAT + '/_local/status', timeout=1)

    def test_delete_posts_id_and_reports_refusal(self):
        entry = 'rust-weights-0123456789abcdef'
        with patch.object(cli, 'ensure_service', return_value=True), \
                patch.object(cli, 'request', return_value=(200, {})) as request, patch.object(cli, 'notify') as notify:
            self.assertEqual(cli.action('storage-delete', entry), 0)
            request.assert_called_once_with(cli.CHAT + '/_local/storage/delete', post=True, timeout=120, payload={'id': entry})
            request.return_value = (409, {})
            self.assertEqual(cli.action('storage-delete', entry), 1)
            notify.assert_called_once()


if __name__ == '__main__':
    unittest.main()


class FlashCacheMetadata(unittest.TestCase):
    def test_cache_facts_are_allowlisted(self):
        self.assertEqual(cli.flash_cache_row({'tier_gib': 20, 'pref': '20',
            'reason': 'preferred-swap-tolerant', 'margin_gib': 0.39, 'private_path': '/private'}),
            {'tier_gib': 20, 'pref': '20', 'reason': 'preferred-swap-tolerant', 'margin_gib': 0.39})
        self.assertIsNone(cli.flash_cache_row({'tier_gib': 999, 'pref': 'bad', 'margin_gib': float('nan')}))

    def test_stop_pending_reaches_the_panel_only_while_stopping(self):
        reason = 'engine pid 7 has not exited: still running'
        self.assertEqual(cli.optional_status({'state': 'stopping', 'stop_pending': reason + '\n'}, {})['stop_pending'], reason)
        self.assertEqual(len(cli.optional_status({'state': 'stopping', 'stop_pending': 'y' * 999}, {})['stop_pending']), 200)
        for chat in ({'state': 'ready', 'stop_pending': reason}, {'state': 'stopping', 'stop_pending': 5},
                     {'state': 'stopping', 'stop_pending': '  '}, {'state': 'stopping'}):
            self.assertNotIn('stop_pending', cli.optional_status(chat, {}))

    def test_cached_status_propagates_without_extra_requests(self):
        cache = {'tier_gib': 18, 'pref': '20', 'reason': 'preferred-swap-tolerant'}
        result = cli.optional_status({'flash_cache': cache}, {})
        self.assertEqual(result['flash_cache'], cache)


class FlashCacheAction(unittest.TestCase):
    def test_setting_calls_only_nonwaking_control_route(self):
        with patch.object(cli, 'request', return_value=(200, {})) as request, \
             patch.object(cli, 'ensure_service', side_effect=AssertionError('must not start service')):
            self.assertEqual(cli.action('flash-cache', pref='20'), 0)
            request.assert_called_once_with(cli.CHAT + '/_local/flash/cache', post=True, timeout=15, payload={'pref': '20'})

    def test_bad_or_busy_setting_fails_without_other_actions(self):
        with patch.object(cli, 'request', return_value=(409, {})) as request, patch.object(cli, 'notify'):
            for pref in (None, '21', '20GiB', 20):
                self.assertEqual(cli.action('flash-cache', pref=pref), 2)
            request.assert_not_called()
            self.assertEqual(cli.action('flash-cache', pref='auto'), 1)


class FakeController(http.server.BaseHTTPRequestHandler):
    """Answers the cache routes from class-level tables and records every request."""
    calls = []
    answers = {}

    def log_message(self, *args):
        pass

    def handle_request(self):
        length = int(self.headers.get('Content-Length', '0') or 0)
        body = json.loads(self.rfile.read(length)) if length else None
        FakeController.calls.append((self.command, self.path, body))
        status, payload = FakeController.answers.get((self.command, self.path), (404, {}))
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    do_GET = handle_request
    do_POST = handle_request


class CacheCommands(unittest.TestCase):
    """`local-ai cache` over a real HTTP round trip to a fake controller (never port 8092)."""
    CACHE, BUDGET, CLEAR = '/_local/cache', '/_local/cache/budget', '/_local/cache/clear'

    def setUp(self):
        FakeController.calls = []
        FakeController.answers = {}
        self.server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), FakeController)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.chat = patch.object(cli, 'CHAT', f'http://127.0.0.1:{self.server.server_address[1]}')
        self.chat.start()

    def tearDown(self):
        self.chat.stop()
        self.server.shutdown()
        self.server.server_close()

    def run_cache(self, *words, as_json=False, session=None, clear_all=False):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = cli.cache_main(list(words), as_json, session, clear_all)
        return code, out.getvalue(), err.getvalue()

    def test_status_is_allowlisted_and_has_no_model_or_path_keys(self):
        FakeController.answers[('GET', self.CACHE)] = (200, {
            'budget_bytes': 64000000000, 'used_bytes': 18400000000, 'ttl_secs': 604800, 'invalid_settings': False,
            'models': {'0123456789abcdef': {'bytes': 1}}, 'engine': {'budget_bytes': 1},
            'sessions': {'count': 3, 'oldest_atime': time.time() - 7200, 'newest_atime': time.time()}})
        code, out, _ = self.run_cache('status', as_json=True)
        value = json.loads(out)
        self.assertEqual(code, 0)
        self.assertEqual(set(value), {'cache_available', 'budget_bytes', 'used_bytes', 'ttl_secs', 'default_budget_bytes',
                                      'invalid_settings', 'sessions'})
        self.assertEqual(set(value['sessions']), {'count', 'oldest_age_secs'})
        self.assertEqual(value['sessions']['count'], 3)
        self.assertTrue(7195 <= value['sessions']['oldest_age_secs'] <= 7300)
        self.assertEqual(value['default_budget_bytes'], 64000000000)
        self.assertNotIn('0123456789abcdef', out)

    def test_status_reads_the_invalid_settings_list(self):
        """Codex review item 5: the controller sends state_cache.settings' list of invalid keys."""
        from snapdragon_ai_lab.runtime import state_cache
        for stored, want in (({'disk_budget_bytes': 5}, True), ({'ttl_secs': 1}, True), ({}, False)):
            invalid = state_cache.settings({'cache': stored})['invalid']
            FakeController.answers[('GET', self.CACHE)] = (200, {'budget_bytes': 64000000000, 'used_bytes': 0,
                'ttl_secs': 604800, 'invalid_settings': invalid, 'sessions': {'count': 0}})
            value = json.loads(self.run_cache('status', as_json=True)[1])
            self.assertIs(value['invalid_settings'], want, stored)
        FakeController.answers[('GET', self.CACHE)] = (200, {'invalid_settings': True, 'sessions': {}})
        self.assertIs(json.loads(self.run_cache('status', as_json=True)[1])['invalid_settings'], True)  # legacy

    def test_status_when_controller_does_not_answer_exits_one(self):
        code, out, _ = self.run_cache('status', as_json=True)
        self.assertEqual((code, json.loads(out)), (1, {'cache_available': False}))

    def test_text_status_line(self):
        FakeController.answers[('GET', self.CACHE)] = (200, {'budget_bytes': 64000000000, 'used_bytes': 18400000000,
            'ttl_secs': 604800, 'invalid_settings': False, 'sessions': {'count': 1, 'oldest_atime': time.time() - 90000}})
        code, out, _ = self.run_cache('status')
        self.assertEqual(code, 0)
        self.assertIn('18.40 GB used of 64 GB, 1 conversation, oldest 25 h old', out)

    def test_bare_cache_prints_usage_without_any_request(self):
        code, out, _ = self.run_cache()
        self.assertEqual(code, 0)
        self.assertIn('local-ai cache budget GB|off', out)
        self.assertEqual(FakeController.calls, [])

    def test_budget_sends_decimal_bytes_and_maps_answers(self):
        FakeController.answers[('POST', self.BUDGET)] = (200, {'budget_bytes': 16000000000, 'freed_bytes': 0,
                                                             'applied_by': 'controller'})
        code, out, _ = self.run_cache('budget', '16', as_json=True)
        self.assertEqual((code, json.loads(out)['budget_bytes']), (0, 16000000000))
        self.assertEqual(FakeController.calls[-1], ('POST', self.BUDGET, {'bytes': 16000000000}))
        code, _, _ = self.run_cache('budget', 'off')
        self.assertEqual((code, FakeController.calls[-1][2]), (0, {'bytes': 0}))
        FakeController.answers[('POST', self.BUDGET)] = (409, {'error': 'busy'})
        code, out, err = self.run_cache('budget', '32', as_json=True)
        self.assertEqual(code, 1)
        self.assertIn('error', json.loads(out))

    def test_bad_budgets_are_refused_before_any_request(self):
        for text in ('3', '3.9', '4097', 'nan', 'inf', 'GB', '', '-16', '64abc'):
            with self.subTest(text=text):
                code, _, err = self.run_cache('budget', text)
                self.assertEqual(code, 2)
                self.assertIn('Usage:', err)
        self.assertEqual(FakeController.calls, [])
        self.assertEqual(cli.cache_budget_bytes('4'), 4000000000)
        self.assertEqual(cli.cache_budget_bytes('128'), 128000000000)

    def test_clear_defaults_to_all_and_session_scope_is_validated(self):
        FakeController.answers[('POST', self.CLEAR)] = (200, {'cleared': True, 'state_cache': {'freed_bytes': 2500000000}})
        code, out, _ = self.run_cache('clear', as_json=True)
        self.assertEqual(code, 0)
        self.assertEqual(FakeController.calls[-1], ('POST', self.CLEAR, {'scope': 'all'}))
        self.assertEqual(json.loads(out)['state_cache']['freed_bytes'], 2500000000)
        code, _, _ = self.run_cache('clear', clear_all=True)
        self.assertEqual(FakeController.calls[-1][2], {'scope': 'all'})
        code, _, _ = self.run_cache('clear', session='conv-01_A')
        self.assertEqual((code, FakeController.calls[-1][2]), (0, {'scope': 'session', 'session': 'conv-01_A'}))
        sent = len(FakeController.calls)
        for bad in ('a b', '../x', 'x' * 129, ''):
            with self.subTest(session=bad):
                self.assertEqual(self.run_cache('clear', session=bad)[0], 2)
        self.assertEqual(self.run_cache('clear', session='abc', clear_all=True)[0], 2)
        self.assertEqual(len(FakeController.calls), sent)

    def test_busy_clear_exits_one_with_a_plain_reason(self):
        FakeController.answers[('POST', self.CLEAR)] = (409, {'error': 'A request is running.'})
        code, _, err = self.run_cache('clear')
        self.assertEqual(code, 1)
        self.assertIn('Try again', err)

    def test_unknown_words_are_usage_errors(self):
        for words in (('bogus',), ('status', 'extra'), ('budget',), ('clear', 'now')):
            with self.subTest(words=words):
                self.assertEqual(self.run_cache(*words)[0], 2)
        self.assertEqual(FakeController.calls, [])

    def test_cache_facts_are_optional_and_allowlisted_in_ledger_rows(self):
        base = {'t': 7, 'path': '/v1/chat/completions', 'model': 'qwen3.6-35b-a3b-local', 'status': 200}
        plain = cli.ledger_row(base)
        self.assertNotIn('cache_src', plain)
        self.assertNotIn('miss_class', plain)
        row = cli.ledger_row({**base, 'cache_src': 'store', 'resume_src': 'disk', 'miss_class': 'unknown',
                              'echo_tokens': 12, 'miss_tokens': 40, 'session': 'secret-hash', 'prompt': 'PRIVATE'})
        self.assertEqual({k: row[k] for k in ('cache_src', 'resume_src', 'miss_class', 'echo_tokens', 'miss_tokens')},
                         {'cache_src': 'store', 'resume_src': 'disk', 'miss_class': 'unknown', 'echo_tokens': 12, 'miss_tokens': 40})
        self.assertNotIn('session', row)
        self.assertNotIn('PRIVATE', json.dumps(row))
        bad = cli.ledger_row({**base, 'cache_src': 'nonsense', 'resume_src': '/private/path', 'miss_class': 'x', 'echo_tokens': -1})
        self.assertFalse({'cache_src', 'resume_src', 'miss_class', 'echo_tokens'} & set(bad))
        row = cli.ledger_row({**base, 'cache_src': 'state', 'resume_src': 'disk', 'resume_ms': 38.2,
                              'preload_src': 'disk', 'preload_ms': 812.0})
        self.assertEqual({k: row[k] for k in ('cache_src', 'resume_src', 'resume_ms', 'preload_src', 'preload_ms')},
                         {'cache_src': 'state', 'resume_src': 'disk', 'resume_ms': 38.2, 'preload_src': 'disk', 'preload_ms': 812.0})
        bad = cli.ledger_row({**base, 'preload_src': 'ram', 'resume_ms': -1, 'preload_ms': 'x'})
        self.assertFalse({'preload_src', 'resume_ms', 'preload_ms'} & set(bad))
