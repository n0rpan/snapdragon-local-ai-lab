"""Panel view model (Model.js) runs under node and matches the UI contract sections 3 to 5."""
from pathlib import Path
import json
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
MODEL_JS = ROOT / 'src/snapdragon_ai_lab/omarchy_plugin/Model.js'
FIXTURES = ROOT / 'tests/fixtures/panel'

DRIVER = r"""
const fs = require('fs');
const req = JSON.parse(fs.readFileSync(0, 'utf8'));
const api = new Function(req.src + '\nreturn {parse: parse, parseLedger: parseLedger, parseStorage: parseStorage, countdown: countdown, gib: gib, harnessEstimate: harnessEstimate};')();
const out = req.calls.map(function (c) { return api[c[0]].apply(null, c[1]); });
process.stdout.write(JSON.stringify(out));
"""

EXISTING_STATUS_KEYS = {
    'chat_state', 'active_requests', 'idle_seconds', 'compute_busy', 'model', 'selected_model',
    'prefix_cache_enabled', 'backend', 'npu_status', 'image_state', 'image_model', 'available_gib',
    'flash_available', 'ornith_available', 'execution_policy_mode', 'flash_download_active',
    'flash_download_bytes', 'flash_download_total_bytes', 'flash_download_shards',
    'flash_download_state', 'flash_verified_shards', 'flash_waiting_verification',
    'flash_trial_active', 'flash_trial_kind', 'flash_trial_elapsed_seconds',
}


def strip_pragma(source):
    return ''.join(line for line in source.splitlines(True) if not line.startswith('.pragma'))


def fixture(name):
    return (FIXTURES / name).read_text(encoding='utf-8')


@unittest.skipUnless(shutil.which('node'), 'node is not installed')
class PanelModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.src = strip_pragma(MODEL_JS.read_text(encoding='utf-8'))

    def run_calls(self, *calls):
        payload = json.dumps({'src': self.src, 'calls': [[name, list(args)] for name, args in calls]})
        done = subprocess.run(['node', '-e', DRIVER], input=payload, capture_output=True, text=True, check=True)
        return json.loads(done.stdout)

    def call(self, name, *args):
        return self.run_calls((name, args))[0]

    def parse(self, value, *extra):
        text = value if isinstance(value, str) else json.dumps(value)
        return self.call('parse', text, *extra)

    def models(self, vm):
        return {row['key']: row for row in vm['models']}

    def test_stripped_file_passes_node_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'Model.js'
            path.write_text(self.src, encoding='utf-8')
            subprocess.run(['node', '--check', str(path)], check=True)

    def test_fallback_for_garbage_inputs(self):
        fallback = self.parse({})
        self.assertEqual(fallback['service'], 'offline')
        self.assertEqual(fallback['unloadBlocked'], 'nothing')
        self.assertFalse(fallback['canUnload'])
        self.assertEqual(fallback['loaded'], '')
        self.assertEqual(fallback['ram'], {'totalGib': -1, 'availableGib': -1, 'modelGib': -1, 'otherGib': -1})
        self.assertEqual(fallback['idle'], {'totalSeconds': 600, 'remainingSeconds': -1})
        self.assertFalse(fallback['hasHarnessData'])
        self.assertEqual(fallback['harnesses'], [])
        self.assertEqual([row['key'] for row in fallback['models']], ['ornith', 'moe', 'flash'])
        for bad in ['not json', '{', '[]', '42', 'null', '"text"', '{"chat_state": "ready"', 'x' * 16385]:
            with self.subTest(text=bad[:20]):
                self.assertEqual(self.parse(bad), fallback)
        for bad in [5, None, [], ['ready']]:
            with self.subTest(value=bad):
                self.assertEqual(self.call('parse', bad), fallback)

    def test_status_length_limit_is_16384_characters(self):
        base = len(json.dumps({'chat_state': 'ready', 'pad': ''}))
        at_limit = json.dumps({'chat_state': 'ready', 'pad': 'x' * (16384 - base)})
        self.assertEqual(len(at_limit), 16384)
        self.assertEqual(self.parse(at_limit)['service'], 'ready')
        over = json.dumps({'chat_state': 'ready', 'pad': 'x' * (16385 - base)})
        self.assertEqual(self.parse(over)['service'], 'offline')

    def test_second_argument_is_ignored(self):
        text = fixture('status_production.json')
        self.assertEqual(self.parse(text, 12345), self.parse(text))
        self.assertEqual(self.parse(text, None), self.parse(text))

    def test_production_status_gives_usable_vm_with_unknowns(self):
        text = fixture('status_production.json')
        self.assertLessEqual(set(json.loads(text)), EXISTING_STATUS_KEYS)
        vm = self.parse(text)
        self.assertEqual(vm['service'], 'ready')
        self.assertEqual(vm['phase'], 'ready')
        self.assertEqual(vm['selected'], 'moe')
        self.assertEqual(vm['loaded'], 'moe')
        self.assertEqual(vm['backend'], 'npu')
        self.assertEqual(vm['ram'], {'totalGib': -1, 'availableGib': 30.5, 'modelGib': -1, 'otherGib': -1})
        self.assertEqual(vm['idle'], {'totalSeconds': 600, 'remainingSeconds': -1})
        self.assertFalse(vm['hasHarnessData'])
        self.assertEqual(vm['harnesses'], [])
        self.assertTrue(vm['canUnload'])
        self.assertEqual(vm['unloadBlocked'], '')
        self.assertTrue(vm['prefixCache'])
        self.assertEqual(vm['flash']['state'], 'absent')
        models = self.models(vm)
        self.assertNotIn('dense', models)
        self.assertEqual(models['moe']['state'], 'loaded')
        self.assertTrue(models['moe']['loaded'])
        self.assertEqual(models['ornith']['state'], 'cold')
        self.assertEqual(models['ornith']['group'], 'named')
        self.assertTrue(models['ornith']['isDefault'])
        self.assertEqual(models['flash']['group'], 'heavy')
        self.assertTrue(models['ornith']['available'])
        self.assertTrue(models['flash']['available'])
        for row in vm['models']:
            with self.subTest(key=row['key']):
                self.assertEqual((row['engine'], row['prefix']), ('', ''))
                self.assertEqual((row['needGib'], row['batteryStart'], row['fits']), (-1, -1, 'unknown'))

    def test_full_status_derives_ram_idle_models_and_harnesses(self):
        vm = self.parse(fixture('status_full.json'))
        self.assertEqual(vm['service'], 'ready')
        self.assertEqual(vm['loaded'], 'moe')
        self.assertEqual(vm['ram'], {'totalGib': 62.5, 'availableGib': 30.5, 'modelGib': 18.25, 'otherGib': 13.75})
        self.assertEqual(vm['idle'], {'totalSeconds': 900, 'remainingSeconds': 412.5})
        models = self.models(vm)
        self.assertEqual(list(models), ['ornith', 'moe', 'flash'])
        self.assertEqual(models['ornith'], {
            'key': 'ornith', 'group': 'named', 'available': True, 'selected': False, 'loaded': False,
            'isDefault': True, 'state': 'cold', 'engine': 'rust', 'prefix': 'warm', 'needGib': 22,
            'fits': 'yes', 'batteryStart': 40,
        })
        self.assertEqual(models['moe']['state'], 'loaded')
        self.assertTrue(models['moe']['selected'])
        self.assertEqual(models['moe']['fits'], 'yes')
        self.assertEqual(models['flash']['engine'], 'stream')
        self.assertEqual(models['flash']['prefix'], 'cold')
        self.assertEqual(models['flash']['batteryStart'], 60)
        self.assertEqual(len(vm['harnesses']), 2)
        opencode, claude = vm['harnesses']
        self.assertEqual(opencode, {
            'name': 'opencode', 'state': 'ready', 'estimateSeconds': 50, 'tools': 14,
            'lastUsed': 1789000000, 'cachedTokens': 4800, 'evaluatedTokens': 788, 'prefixes': 2,
        })
        self.assertEqual(claude['state'], 'warmup')
        self.assertEqual(claude['estimateSeconds'], 50)
        self.assertTrue(vm['hasHarnessData'])

    def test_harness_list_is_capped_and_bad_names_are_dropped(self):
        rows = [{'name': 'h%d' % i, 'warm': True, 'cache_n': 1, 'prompt_n': 0} for i in range(14)]
        rows.insert(3, {'name': 'bad name', 'warm': True})
        rows.insert(4, {'name': 'x' * 33, 'warm': True})
        rows.insert(5, {'name': 7, 'warm': True})
        vm = self.parse({'harnesses': rows})
        self.assertEqual(len(vm['harnesses']), 12)
        self.assertEqual([h['name'] for h in vm['harnesses'][:3]], ['h0', 'h1', 'h2'])
        self.assertTrue(vm['hasHarnessData'])
        self.assertEqual(self.parse({'harnesses': []})['hasHarnessData'], True)

    def test_markup_never_reaches_the_output(self):
        status = json.loads(fixture('status_full.json'))
        status.update({
            'loaded_model': '<b>', 'model': '<script>', 'selected_model': '<i>', 'image_model': '<q>',
            'image_state': '<img>', 'chat_state': '<svg>', 'backend': '<em>', 'flash_download_state': '<svg>',
            'flash_trial_kind': '<b>', 'flash_trial_active': True,
            'models': [{'key': '<b>', 'engine': '<img>', 'prefix': '<x>', 'state': '<i>'}],
            'harnesses': [{'name': '<img src=x>', 'warm': True, 'cache_n': 1, 'prompt_n': 1}],
        })
        vm = self.parse(status)
        self.assertNotIn('<', json.dumps(vm))
        self.assertEqual([h['name'] for h in vm['harnesses']], [])
        self.assertEqual(vm['loaded'], '')
        self.assertEqual(vm['selected'], '')
        ledger = json.loads(fixture('ledger.json'))
        ledger['rows'][0].update({'model': '<b>', 'harness': '<img src=x>', 'finish': '<i>'})
        self.assertNotIn('<', json.dumps(self.call('parseLedger', json.dumps(ledger))))
        storage = json.loads(fixture('storage.json'))
        storage['entries'][0].update({'id': '<img src=x>', 'name': '<b>x</b>', 'kind': '<i>'})
        self.assertNotIn('<', json.dumps(self.call('parseStorage', json.dumps(storage))))

    def test_dense_only_when_selected_or_loaded(self):
        base = {'chat_state': 'unloaded', 'selected_model': 'moe', 'models': [{'key': 'dense', 'memory_gib': 20}]}
        self.assertNotIn('dense', self.models(self.parse(base)))
        selected = self.models(self.parse(dict(base, selected_model='dense')))
        self.assertEqual(selected['dense']['state'], 'cold')
        self.assertTrue(selected['dense']['selected'])
        self.assertEqual(selected['dense']['group'], 'named')
        loaded = self.models(self.parse(dict(base, chat_state='ready', loaded_model='dense')))
        self.assertEqual(loaded['dense']['state'], 'loaded')
        self.assertTrue(loaded['dense']['loaded'])
        self.assertEqual(list(loaded), ['ornith', 'moe', 'flash', 'dense'])

    def test_model_states(self):
        loading = self.parse({'chat_state': 'loading', 'model': 'moe', 'selected_model': 'flash', 'flash_available': True})
        self.assertEqual(loading['phase'], 'loading')
        self.assertEqual(loading['loaded'], 'moe')
        self.assertEqual(self.models(loading)['moe']['state'], 'loading')
        self.assertEqual(self.models(loading)['flash']['state'], 'cold')
        selecting = self.parse({'chat_state': 'loading', 'selected_model': 'flash', 'flash_available': True})
        self.assertEqual(self.models(selecting)['flash']['state'], 'loading')
        self.assertEqual(selecting['loaded'], '')
        ready = self.parse({'chat_state': 'ready', 'loaded_model': 'ornith', 'model': 'moe'})
        self.assertEqual(ready['loaded'], 'ornith')
        self.assertEqual(self.models(ready)['ornith']['state'], 'loaded')
        self.assertEqual(self.models(ready)['moe']['state'], 'cold')
        unloaded = self.parse({'chat_state': 'unloaded', 'model': 'moe', 'loaded_model': 'moe'})
        self.assertEqual(unloaded['phase'], 'sleeping')
        self.assertEqual(unloaded['loaded'], 'moe')
        self.assertEqual(self.models(unloaded)['moe']['state'], 'cold')

    def test_model_availability_and_disabled_rows(self):
        vm = self.parse({'chat_state': 'ready', 'ornith_available': False, 'flash_available': 'yes'})
        models = self.models(vm)
        self.assertFalse(models['ornith']['available'])
        self.assertFalse(models['flash']['available'])
        self.assertTrue(models['moe']['available'])

    def test_fits_yes_no_unknown(self):
        status = {
            'chat_state': 'ready', 'model': 'moe', 'loaded_model': 'moe', 'available_gib': 10,
            'resident_gib': 12, 'total_gib': 40,
            'models': [
                {'key': 'ornith', 'memory_gib': 20},
                {'key': 'flash', 'memory_gib': 40},
                {'key': 'moe', 'memory_gib': 60},
            ],
        }
        models = self.models(self.parse(status))
        self.assertEqual(models['ornith']['fits'], 'yes')
        self.assertEqual(models['flash']['fits'], 'no')
        self.assertEqual(models['moe']['fits'], 'yes', 'loaded model is released first')
        released = dict(status, resident_gib=None)
        self.assertEqual(self.models(self.parse(released))['ornith']['fits'], 'no')
        self.assertEqual(self.models(self.parse(dict(status, available_gib=None)))['ornith']['fits'], 'unknown')
        self.assertEqual(self.models(self.parse(dict(status, models=[{'key': 'ornith'}])))['ornith']['fits'], 'unknown')
        self.assertEqual(self.models(self.parse(dict(status, models=[{'key': 'ornith', 'memory_gib': 0}])))['ornith']['needGib'], -1)

    def test_ram_bounds_and_other_floor(self):
        vm = self.parse({'total_gib': 4097, 'available_gib': 2000, 'resident_gib': 5000})
        self.assertEqual(vm['ram'], {'totalGib': -1, 'availableGib': -1, 'modelGib': -1, 'otherGib': -1})
        vm = self.parse({'total_gib': 10, 'available_gib': 9, 'resident_gib': 4})
        self.assertEqual(vm['ram']['otherGib'], 0)
        vm = self.parse({'total_gib': 10, 'available_gib': 9})
        self.assertEqual(vm['ram']['otherGib'], 1)
        vm = self.parse({'total_gib': 0, 'available_gib': 5, 'resident_gib': 2})
        self.assertEqual(vm['ram']['totalGib'], -1)
        self.assertEqual(vm['ram']['otherGib'], -1)

    def test_idle_bounds(self):
        self.assertEqual(self.parse({'idle_seconds': 0})['idle']['totalSeconds'], 600)
        self.assertEqual(self.parse({'idle_seconds': 86401})['idle']['totalSeconds'], 600)
        self.assertEqual(self.parse({'idle_seconds': 86400})['idle']['totalSeconds'], 86400)
        self.assertEqual(self.parse({'idle_remaining_seconds': -1})['idle']['remainingSeconds'], -1)
        self.assertEqual(self.parse({'idle_remaining_seconds': 86401})['idle']['remainingSeconds'], -1)
        self.assertEqual(self.parse({'idle_remaining_seconds': 0})['idle']['remainingSeconds'], 0)

    def test_unload_blocked_for_each_state(self):
        cases = [
            ({'chat_state': 'unloaded', 'active_requests': 3}, 'nothing', False),
            ({'chat_state': 'offline', 'image_state': 'generating'}, 'nothing', False),
            ({'chat_state': 'loading', 'active_requests': 2}, 'loading', False),
            ({'chat_state': 'ready', 'active_requests': 2}, 'answering', False),
            ({'chat_state': 'ready', 'image_state': 'generating'}, 'images', False),
            ({'chat_state': 'ready', 'active_requests': 0, 'image_state': 'idle'}, '', True),
        ]
        for status, blocked, can in cases:
            with self.subTest(status=status):
                vm = self.parse(status)
                self.assertEqual(vm['unloadBlocked'], blocked)
                self.assertEqual(vm['canUnload'], can)

    def test_phase_values(self):
        self.assertEqual(self.parse({'chat_state': 'offline'})['phase'], 'offline')
        self.assertEqual(self.parse({'chat_state': 'unloaded'})['phase'], 'sleeping')
        self.assertEqual(self.parse({'chat_state': 'loading'})['phase'], 'loading')
        self.assertEqual(self.parse({'chat_state': 'ready', 'active_requests': 1})['phase'], 'answering')
        self.assertEqual(self.parse({'chat_state': 'ready', 'queued_requests': 4})['queued'], 4)

    def test_images_and_flags(self):
        vm = self.parse({'image_state': 'generating', 'image_model': 'q8', 'compute_busy': True, 'has_error': True})
        self.assertEqual((vm['images'], vm['imageModel'], vm['computeBusy'], vm['hasError']), ('generating', 'Q8', True, True))
        vm = self.parse({'image_state': 'x', 'image_model': 'q5', 'backend': 'tpu'})
        self.assertEqual((vm['images'], vm['imageModel'], vm['backend']), ('offline', '', 'cpu'))

    def test_flash_fields_are_ported_from_status_js(self):
        vm = self.parse({
            'flash_download_state': 'downloading', 'flash_download_active': False,
            'flash_download_total_bytes': 1000, 'flash_download_bytes': 333,
            'flash_download_shards': 4, 'flash_verified_shards': 2,
        })
        self.assertEqual(vm['flash'], {
            'state': 'paused', 'percent': 33.3, 'doneBytes': 333, 'totalBytes': 1000, 'verified': 2,
            'shards': 4, 'waiting': False, 'trialActive': False, 'trialKind': '', 'trialElapsed': 0,
        })
        vm = self.parse({
            'flash_download_state': 'verifying', 'flash_download_active': True,
            'flash_download_total_bytes': 1000, 'flash_download_bytes': 1000,
            'flash_download_shards': 4, 'flash_verified_shards': 1, 'flash_waiting_verification': True,
        })
        self.assertEqual((vm['flash']['state'], vm['flash']['waiting']), ('verifying', True))
        vm = self.parse({
            'flash_download_state': 'weights-ready-testing', 'flash_download_total_bytes': 1000,
            'flash_download_bytes': 1000, 'flash_download_shards': 4, 'flash_verified_shards': 4,
            'flash_trial_active': True, 'flash_trial_kind': 'coding', 'flash_trial_elapsed_seconds': 42,
        })
        self.assertEqual((vm['flash']['state'], vm['flash']['percent']), ('weights-ready-testing', 100))
        self.assertEqual((vm['flash']['trialActive'], vm['flash']['trialKind'], vm['flash']['trialElapsed']), (True, 'coding', 42))
        vm = self.parse({'flash_trial_active': True, 'flash_trial_kind': '<b>', 'flash_trial_elapsed_seconds': 3})
        self.assertEqual((vm['flash']['trialActive'], vm['flash']['trialKind'], vm['flash']['trialElapsed']), (False, '', 0))
        vm = self.parse({'flash_download_state': 'downloading', 'flash_download_active': True,
                         'flash_download_total_bytes': 0, 'flash_download_bytes': 5, 'flash_download_shards': 1})
        self.assertEqual((vm['flash']['state'], vm['flash']['percent']), ('absent', 0))

    def test_ledger_derivations(self):
        ledger = self.call('parseLedger', fixture('ledger.json'))
        self.assertTrue(ledger['available'])
        self.assertEqual(len(ledger['rows']), 3)
        a, b, c = ledger['rows']
        self.assertEqual((a['t'], a['model'], a['harness'], a['ok'], a['status']), (1000, 'moe', 'opencode', True, 200))
        self.assertEqual((a['queueMs'], a['loadMs'], a['ttftMs'], a['wallMs']), (5, 1200, 1505, 3000))
        self.assertEqual((a['tokS'], a['cacheN'], a['promptN'], a['cacheShare'], a['predictedN'], a['finish']),
                         (50, 800, 200, 0.8, 100, 'stop'))
        self.assertEqual((b['harness'], b['model'], b['ok'], b['finish']), ('', 'ornith', False, ''))
        self.assertEqual((b['loadMs'], b['ttftMs'], b['tokS'], b['cacheShare']), (0, -1, -1, -1))
        self.assertEqual((c['model'], c['loadMs'], c['queueMs'], c['ttftMs'], c['tokS']), ('dense', 0, -1, 400, 200))
        self.assertAlmostEqual(c['cacheShare'], 0.1)
        self.assertEqual(ledger['summary'], {
            'count': 3, 'failed': 1, 'medianTtftMs': 952.5, 'medianTokS': 125,
            'cacheShare': 0.45, 'lastT': 2000,
        })

    def test_ledger_median_odd_count_and_row_cap(self):
        rows = [{'t': i, 'status': 200, 'ttfb_ms': 10 * (i + 1), 'predicted_n': 10, 'predicted_ms': 1000}
                for i in range(3)]
        summary = self.call('parseLedger', json.dumps({'ledger_available': True, 'rows': rows}))['summary']
        self.assertEqual((summary['medianTtftMs'], summary['medianTokS']), (20, 10))
        many = [{'t': i, 'status': 200} for i in range(205)]
        ledger = self.call('parseLedger', json.dumps({'ledger_available': True, 'rows': many}))
        self.assertEqual(len(ledger['rows']), 200)
        self.assertEqual(ledger['summary']['count'], 200)
        self.assertEqual(ledger['summary']['lastT'], 199)

    def test_ledger_unavailable_and_bad_input(self):
        empty = {'available': False, 'rows': [], 'summary': {'count': 0, 'failed': 0, 'medianTtftMs': -1,
                                                             'medianTokS': -1, 'cacheShare': -1, 'lastT': 0}}
        self.assertEqual(self.call('parseLedger', '{"ledger_available": false, "rows": []}'), empty)
        self.assertEqual(self.call('parseLedger', 'not json'), empty)
        self.assertEqual(self.call('parseLedger', json.dumps({'ledger_available': True, 'rows': 'x'})), empty)
        self.assertEqual(self.call('parseLedger', 'x' * 262145), empty)
        self.assertEqual(self.call('parseLedger', 7), empty)

    def test_storage_parsing(self):
        storage = self.call('parseStorage', fixture('storage.json'))
        self.assertTrue(storage['available'])
        ids = [entry['id'] for entry in storage['entries']]
        self.assertEqual(ids, ['rust-weights-0123456789abcdef', 'rust-prefix-fedcba9876543210',
                               'llama-repack-aaaaaaaaaaaaaaaa', 'llama-prefix-cccccccccccccccc'])
        weights, prefix, repack, old = storage['entries']
        self.assertEqual((weights['deletable'], weights['name'], weights['model']), (False, 'moe.gguf', 'moe'))
        self.assertEqual((prefix['deletable'], prefix['model']), (True, 'other'))
        self.assertFalse(repack['deletable'])
        self.assertEqual((old['name'], old['model'], old['deletable']), ('', '', True))
        self.assertEqual(storage['totalBytes'], 19327352832 + 1073741824 + 536870912 + 100)
        self.assertEqual(storage['freeableBytes'], 1073741824 + 100)

    def test_storage_fallback(self):
        empty = {'available': False, 'entries': [], 'totalBytes': 0, 'freeableBytes': 0}
        self.assertEqual(self.call('parseStorage', json.dumps({'storage_available': False, 'entries': []})), empty)
        self.assertEqual(self.call('parseStorage', json.dumps({'storage_available': 'yes', 'entries': []})), empty)
        self.assertEqual(self.call('parseStorage', '[1]'), empty)
        self.assertEqual(self.call('parseStorage', 'x' * 65537), empty)
        self.assertEqual(self.call('parseStorage', None), empty)

    def test_countdown(self):
        results = self.run_calls(('countdown', [582]), ('countdown', [59]), ('countdown', [3600]),
                                 ('countdown', [0]), ('countdown', [-1]), ('countdown', ['12']), ('countdown', [None]))
        self.assertEqual(results, ['9:42', '0:59', '60:00', '0:00', '', '', ''])

    def test_gib(self):
        results = self.run_calls(('gib', [1073741824]), ('gib', [1610612736]), ('gib', [0]),
                                 ('gib', [-5]), ('gib', ['x']), ('gib', [5368709120]))
        self.assertEqual(results, [1, 1.5, 0, 0, 0, 5])

    def test_harness_estimate(self):
        results = self.run_calls(('harnessEstimate', [5588]), ('harnessEstimate', [0]), ('harnessEstimate', [-3]),
                                 ('harnessEstimate', ['x']), ('harnessEstimate', [620]), ('harnessEstimate', [621]))
        self.assertEqual(results, [50, -1, -1, -1, 5, 10])


if __name__ == '__main__':
    unittest.main()
