"""v0.1.3 panel view model: context modes, session preload, own-action failures, the status
command's exit states, alias rows and the retired dense model. Model.js runs under node; no
controller, no shell and no real state are involved (the status texts are built here)."""
from pathlib import Path
import json
import re
import shutil
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]
PLUGIN = ROOT / 'src/snapdragon_ai_lab/omarchy_plugin'
NAMES = ('parse', 'parseLedger', 'parseStorage', 'notice', 'ownFailure', 'preloadPhase', 'contextOf',
         'preloadOf', 'failureOf', 'parseShare')
DRIVER = r"""
const fs = require('fs');
const req = JSON.parse(fs.readFileSync(0, 'utf8'));
const api = new Function(req.src + '\nreturn {' + req.names.map(function (n) { return n + ': ' + n; }).join(', ') + '};')();
process.stdout.write(JSON.stringify(req.calls.map(function (c) { return api[c[0]].apply(null, c[1]); })));
"""

# The contract of V013-PLAN section 3, exactly as the controller status carries it.
CONTEXT = {'cap': 131072, 'mode': 'deep', 'modes': [
    {'id': 'fast', 'compact_at': 48000, 'cap': 131072, 'available': True, 'reason': None},
    {'id': 'deep', 'compact_at': 100000, 'cap': 131072, 'available': True, 'reason': None},
    {'id': 'max', 'compact_at': 200000, 'cap': 262144, 'available': False, 'reason': 'needs RSS proof'}]}
READY = {'chat_state': 'ready', 'active_requests': 0, 'queued_requests': 0, 'selected_model': 'ornith',
         'loaded_model': 'ornith', 'image_state': 'idle', 'context': CONTEXT}


@unittest.skipUnless(shutil.which('node'), 'node is not installed')
class PanelV013Model(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = (PLUGIN / 'Model.js').read_text(encoding='utf-8')
        cls.src = ''.join(line for line in source.splitlines(True) if not line.startswith('.pragma'))

    def run_calls(self, *calls):
        payload = json.dumps({'src': self.src, 'names': NAMES, 'calls': [[name, list(args)] for name, args in calls]})
        done = subprocess.run(['node', '-e', DRIVER], input=payload, capture_output=True, text=True, check=True)
        return json.loads(done.stdout)

    def call(self, name, *args):
        return self.run_calls((name, args))[0]

    def parse(self, value, *extra):
        return self.call('parse', value if isinstance(value, str) else json.dumps(value), 0, *extra)

    # ---- context modes -------------------------------------------------------------------

    def test_context_modes_follow_the_status_contract(self):
        vm = self.parse(READY)
        context = vm['context']
        self.assertEqual((context['cap'], context['mode'], context['canChange'], vm['contextKey']),
                         (131072, 'deep', True, 'ornith'))
        self.assertEqual(context['modes'], [
            {'id': 'fast', 'compactAt': 48000, 'cap': 131072, 'available': True, 'reason': ''},
            {'id': 'deep', 'compactAt': 100000, 'cap': 131072, 'available': True, 'reason': ''},
            {'id': 'max', 'compactAt': 200000, 'cap': 262144, 'available': False, 'reason': 'needs RSS proof'}])

    def test_each_mode_can_be_the_current_one_and_unknown_modes_select_nothing(self):
        for mode in ('fast', 'deep', 'max'):
            with self.subTest(mode=mode):
                self.assertEqual(self.parse({**READY, 'context': {**CONTEXT, 'mode': mode}})['context']['mode'], mode)
        for mode in ('turbo', '', None, 3):
            with self.subTest(mode=mode):
                self.assertEqual(self.parse({**READY, 'context': {**CONTEXT, 'mode': mode}})['context']['mode'], '')

    def test_modes_belong_to_the_moe_model_in_play_only(self):
        self.assertEqual(self.parse({**READY, 'loaded_model': 'moe'})['contextKey'], 'moe')
        sleeping = self.parse({**READY, 'chat_state': 'unloaded', 'loaded_model': None, 'selected_model': 'moe'})
        self.assertEqual((sleeping['contextKey'], sleeping['context']['mode']), ('moe', 'deep'))
        flash = self.parse({**READY, 'loaded_model': 'flash', 'flash_available': True})
        self.assertEqual((flash['context'], flash['contextKey']), (None, ''))
        nothing = self.parse({**READY, 'chat_state': 'unloaded', 'loaded_model': None, 'selected_model': None})
        self.assertEqual((nothing['context'], nothing['contextKey']), (None, ''))

    def test_modes_cannot_change_while_the_model_is_busy(self):
        for changes in ({'active_requests': 1}, {'queued_requests': 1}, {'chat_state': 'loading'},
                        {'chat_state': 'stopping'}, {'chat_state': 'offline'}):
            with self.subTest(changes=changes):
                self.assertFalse(self.parse({**READY, **changes})['context']['canChange'])

    def test_bad_context_blocks_are_dropped_or_bounded(self):
        for bad in (None, 'deep', [], {}, {'mode': 'deep'}, {'mode': 'deep', 'modes': 'fast'},
                    {'mode': 'deep', 'modes': [{'id': 'turbo'}, 7, None]}):
            with self.subTest(bad=bad):
                self.assertIsNone(self.parse({**READY, 'context': bad})['context'])
        odd = self.parse({**READY, 'context': {'cap': -1, 'mode': 'max', 'modes': [
            {'id': 'max', 'compact_at': 'many', 'cap': 1e12, 'available': 'yes', 'reason': 'a\nb\x1b' + 'x' * 300},
            {'id': 'fast', 'available': True}, {'id': 'fast', 'available': False}]}})['context']
        self.assertEqual([m['id'] for m in odd['modes']], ['fast', 'max'])  # panel order, no duplicates
        self.assertEqual((odd['cap'], odd['mode']), (-1, 'max'))
        worst = odd['modes'][1]
        self.assertEqual((worst['compactAt'], worst['cap'], worst['available']), (-1, -1, False))
        self.assertEqual(len(worst['reason']), 160)
        self.assertNotRegex(worst['reason'], r'[\x00-\x1f]')
        self.assertTrue(odd['modes'][0]['available'])

    # ---- session preload -----------------------------------------------------------------

    def test_preload_states(self):
        def phase(preload, **changes):
            return self.run_calls(('parse', [json.dumps({**READY, **changes, 'preload': preload}), 0]))[0]
        running = phase({'state': 'running', 'src': 'disk', 'ms': None, 'age_s': 0})
        self.assertEqual(running['preload'], {'state': 'running', 'src': 'disk', 'ms': -1, 'ageS': 0})
        self.assertEqual(self.call('preloadPhase', running), 'running')
        done = phase({'state': 'done', 'src': 'disk', 'ms': 3200, 'age_s': 12})
        self.assertEqual(done['preload'], {'state': 'done', 'src': 'disk', 'ms': 3200, 'ageS': 12})
        self.assertEqual(self.call('preloadPhase', done), 'done')
        self.assertEqual(self.call('preloadPhase', phase({'state': 'done', 'src': 'disk', 'ms': 3200, 'age_s': 601})), '')
        for state in ('idle', 'cancelled', 'unavailable', 'bogus', None):
            with self.subTest(state=state):
                self.assertEqual(self.call('preloadPhase', phase({'state': state})), '')
        # only a loaded, idle model shows it: not while answering, loading or with nothing loaded
        for changes in ({'active_requests': 1}, {'chat_state': 'loading'}, {'chat_state': 'unloaded'}):
            with self.subTest(changes=changes):
                self.assertEqual(self.call('preloadPhase', phase({'state': 'running'}, **changes)), '')
        for missing in (None, 'running', [], 4):
            with self.subTest(missing=missing):
                self.assertEqual(phase(missing)['preload'], {'state': 'unavailable', 'src': '', 'ms': -1, 'ageS': -1})
        self.assertEqual(self.call('preloadPhase', None), '')

    # ---- failures: only the panel's own action ------------------------------------------------

    def test_last_failure_is_parsed_and_bounded(self):
        vm = self.parse({**READY, 'last_failure': {'reason': 'Not enough free RAM\nfor Ornith\x07', 'alias': 'ornith',
                                                   't': 1794000000, 'origin': 'organic', 'kind': 'start'}})
        self.assertEqual(vm['failure'], {'reason': 'Not enough free RAM for Ornith', 't': 1794000000,
                                         'origin': 'organic', 'model': 'ornith'})
        for nothing in (None, 'failed', [], {}, {'t': 0, 'reason': ''}):
            with self.subTest(nothing=nothing):
                self.assertIsNone(self.parse({**READY, 'last_failure': nothing})['failure'])
        odd = self.parse({**READY, 'last_failure': {'reason': 'x' * 500, 't': 5, 'origin': 'somebody',
                                                    'alias': 'qwen-unknown-local'}})['failure']
        self.assertEqual((len(odd['reason']), odd['origin'], odd['model']), (200, '', ''))

    def test_flat_error_keys_give_the_same_failure_without_a_reason(self):
        """`local-ai status --json` carries error_at / error_origin instead of the last_failure object."""
        vm = self.parse({**READY, 'has_error': True, 'error_at': 1794000000, 'error_origin': 'organic'})
        self.assertEqual(vm['failure'], {'reason': '', 't': 1794000000, 'origin': 'organic', 'model': ''})
        lab = self.parse({**READY, 'has_error': False, 'error_at': 1794000000, 'error_origin': 'lab'})
        self.assertEqual(lab['failure']['origin'], 'lab')
        for at, origin in ((None, None), (0, 'organic'), ('now', 'organic'), (-5, 'lab')):
            with self.subTest(at=at):
                self.assertIsNone(self.parse({**READY, 'error_at': at, 'error_origin': origin})['failure'])
        self.assertEqual(self.parse({**READY, 'error_at': 9, 'error_origin': 'someone'})['failure']['origin'], '')
        # the object wins when both are present
        both = self.parse({**READY, 'error_at': 9, 'error_origin': 'lab',
                           'last_failure': {'reason': 'No RAM', 't': 12, 'origin': 'organic'}})
        self.assertEqual((both['failure']['t'], both['failure']['origin']), (12, 'organic'))

    def test_failure_shows_only_for_the_panels_own_action_and_newer_than_it(self):
        now = 1794000000
        calm = {'hasError': False, 'failure': None}

        def vm(t, origin='organic'):
            return {'hasError': True, 'failure': {'reason': 'No RAM', 't': t, 'origin': origin, 'model': 'ornith'}}

        def notice(*args):
            return self.call('notice', *args)
        empty = {'rows': []}
        # own: the action started at now-20, the failure was stamped inside it
        self.assertEqual(notice(vm(now - 15), empty, now, now - 20.4, 0, now - 10), 'failure')
        self.assertEqual(notice(vm(now - 20), empty, now, now - 20.4, 0, 0), 'failure')       # still running
        self.assertEqual(notice(vm(now - 15, 'panel'), empty, now, now - 20, 0, now - 10), 'failure')
        self.assertEqual(notice(vm(now - 9), empty, now, now - 20, 0, now - 10), 'failure')  # stamp rounding
        # foreign: lab traffic, an unnamed origin, older than the action, after it ended, no action at all
        self.assertEqual(notice(vm(now - 15, 'lab'), empty, now, now - 20, 0, now - 10), '')
        self.assertEqual(notice(vm(now - 15, ''), empty, now, now - 20, 0, now - 10), '')
        self.assertEqual(notice(vm(now - 60), empty, now, now - 20, 0, now - 10), '')
        self.assertEqual(notice(vm(now - 5), empty, now, now - 20, 0, now - 10), '')
        self.assertEqual(notice(vm(now - 15), empty, now, 0, 0, 0), '')
        # a foreign failure never rides on has_error either (the pre-v0.1.3 "start" rule)
        self.assertEqual(notice(vm(now - 600), empty, now, now - 20, 0, now - 10), '')
        # it expires like every notice, and a clock that runs backwards shows nothing
        self.assertEqual(notice(vm(now - 700), empty, now, now - 705, 0, 0), '')
        self.assertEqual(notice(vm(now + 60), empty, now, now - 20, 0, 0), '')
        # a failed panel action without an own controller failure keeps the generic line
        self.assertEqual(notice(vm(now - 60, 'lab'), empty, now, now - 20, now - 10, now - 10), 'action')
        self.assertEqual(notice(calm, empty, now, now - 20, now - 10, now - 10), 'action')
        # the own failure's reason outranks the generic line
        self.assertEqual(notice(vm(now - 15), empty, now, now - 20, now - 10, now - 10), 'failure')
        # a controller without last_failure keeps the old rule
        self.assertEqual(notice({'hasError': True}, empty, now, now - 30, 0, 0), 'start')
        self.assertEqual(notice({'hasError': True, 'failure': None}, empty, now, now - 30, 0, 0), 'start')
        self.assertTrue(self.call('ownFailure', vm(now - 15)['failure'], now, now - 20, now - 10))
        for bad in (None, {}, {'t': now - 15}, {'t': 'now', 'origin': 'organic'}):
            with self.subTest(bad=bad):
                self.assertFalse(self.call('ownFailure', bad, now, now - 20, now - 10))

    # ---- the status command's exit: offline, stopping and error are different states ----------

    def test_exit_states_are_three_calm_states_with_reasons(self):
        offline = self.parse({'chat_state': 'offline', 'selected_model': 'ornith'}, 1)
        self.assertEqual((offline['phase'], offline['fault'], offline['selected']), ('offline', '', 'ornith'))
        stopping = self.parse({'chat_state': 'stopping', 'stop_pending': 'engine has not exited'}, 0)
        self.assertEqual((stopping['phase'], stopping['fault'], stopping['stopReason']),
                         ('stopping', '', 'engine has not exited'))
        for text, code, kind in (('', 2, 'command'), ('Traceback (most recent call last):', 1, 'unreadable'),
                                 ('', 0, 'unreadable'), ('[]', 0, 'unreadable'), ('', -1, 'missing'),
                                 ('', 127, 'missing'), ('', 126, 'missing'), ('{', 137, 'command'),
                                 ('x' * 16385, 0, 'unreadable')):
            with self.subTest(text=text[:12], code=code):
                vm = self.parse(text, code)
                self.assertEqual((vm['phase'], vm['fault'], vm['faultCode']), ('error', kind, code))
                self.assertEqual((vm['service'], vm['canLoad'], vm['canUnload'], vm['loaded']),
                                 ('offline', False, False, ''))
                self.assertEqual([row['key'] for row in vm['models']], ['ornith', 'moe', 'flash'])
        # a readable answer wins over an odd exit code; without a code the old fallback stays
        self.assertEqual(self.parse(READY, 3)['phase'], 'ready')
        self.assertEqual((self.parse('')['phase'], self.parse('')['fault']), ('offline', ''))
        self.assertEqual(self.call('parse', 'not json', 0, None)['phase'], 'offline')

    # ---- chooser rows --------------------------------------------------------------------------

    def test_alias_rows_are_not_chooser_rows(self):
        vm = self.parse({'chat_state': 'unloaded', 'selected_model': 'ornith', 'ornith_available': True,
                         'available_gib': 25, 'models': [
                             {'key': 'ornith', 'alias_of': 'ornith-1.5-35b-a3b-local', 'memory_gib': 99},
                             {'key': 'moe', 'alias_of': 'qwen3.6-35b-a3b-local', 'legacy': True, 'memory_gib': 98},
                             {'key': 'ornith', 'memory_gib': 19.2, 'engine': 'rust'},
                             {'key': 'moe', 'memory_gib': 20.1, 'engine': 'rust'}]})
        rows = {row['key']: row for row in vm['models']}
        self.assertEqual(list(rows), ['ornith', 'moe', 'flash'])
        self.assertEqual((rows['ornith']['needGib'], rows['moe']['needGib']), (19.2, 20.1))
        only_alias = self.parse({'chat_state': 'unloaded', 'models': [{'key': 'moe', 'alias_of': 'x', 'memory_gib': 98}]})
        self.assertEqual({row['key']: row['needGib'] for row in only_alias['models']}['moe'], -1)

    def test_flash_cache_tier_basis_reads_as_the_ram_the_model_needs(self):
        vm = self.parse({'chat_state': 'unloaded', 'flash_available': True, 'available_gib': 12, 'models': [
            {'key': 'flash', 'memory_gib': 21.5, 'memory_basis': 'cache_pref_tier'},
            {'key': 'moe', 'memory_gib': 20.1, 'memory_basis': 'qualified_working_set'}]})
        rows = {row['key']: row for row in vm['models']}
        self.assertEqual((rows['flash']['needGib'], rows['flash']['fits'], 'workingSetGib' in rows['flash']), (21.5, 'no', False))
        self.assertEqual((rows['moe']['needGib'], rows['moe']['workingSetGib']), (-1, 20.1))

    def test_share_state_is_parsed_and_bounded(self):
        none = self.call('parseShare', '')
        self.assertEqual((none['available'], none['enabled'], none['mode']), (False, False, 'off'))
        for bad in ('not json', '[]', '{}', json.dumps({'enabled': 'yes', 'mode': 'lan'}),
                    json.dumps({'enabled': True, 'mode': 'internet'}), 'x' * 8193, None):
            with self.subTest(bad=str(bad)[:16]):
                self.assertEqual(self.call('parseShare', bad), none)
        off = self.call('parseShare', json.dumps({'enabled': False, 'mode': 'off', 'active': False, 'urls': []}))
        self.assertEqual((off['available'], off['enabled'], off['active']), (True, False, False))
        on = self.call('parseShare', json.dumps({
            'enabled': True, 'mode': 'tailscale', 'active': True, 'encryption': 'wireguard', 'key_fingerprint': 'ab12:cd34',
            'remote_requests': 4, 'remote_active': 1, 'remote_refused': -2,
            'urls': ['http://laptop.example.ts.net:8443/v1', 'javascript:alert(1)', 7, 'http://host/v1?x=<b>',
                     'https://[2001:db8::1]:8443/v1'], 'error': 'a\nb' + 'x' * 300}))
        self.assertEqual((on['enabled'], on['active'], on['mode'], on['encryption'], on['fingerprint']),
                         (True, True, 'tailscale', 'wireguard', 'ab12:cd34'))
        self.assertEqual(on['urls'], ['http://laptop.example.ts.net:8443/v1', 'https://[2001:db8::1]:8443/v1'])
        self.assertEqual(len(on['error']), 160)
        self.assertEqual((on['remoteRequests'], on['remoteActive'], on['remoteRefused']), (4, 1, 0))
        odd = self.call('parseShare', json.dumps({'enabled': True, 'mode': 'off', 'active': True, 'key_fingerprint': '<b>'}))
        self.assertEqual((odd['enabled'], odd['active'], odd['fingerprint']), (False, False, ''))  # the mode decides

    def test_no_dense_27b_labels_are_left(self):
        catalog = json.loads((PLUGIN / 'i18n/en.json').read_text(encoding='utf-8'))
        self.assertEqual([key for key in catalog if 'dense' in key], [])
        for key, value in catalog.items():
            for text in ([value] if isinstance(value, str) else list(value.values())):
                self.assertNotRegex(text, r'(?i)27B|dense', key)
        for path in sorted(PLUGIN.glob('*.qml')) + [PLUGIN / 'Model.js']:
            with self.subTest(path.name):
                self.assertNotRegex(path.read_text(encoding='utf-8'), r'"dense"|27[bB]')
        ledger = self.call('parseLedger', json.dumps({'ledger_available': True, 'rows': [
            {'t': 5, 'model': 'dense', 'status': 200}]}))
        self.assertEqual(ledger['rows'][0]['model'], '')
        storage = self.call('parseStorage', json.dumps({'storage_available': True, 'entries': [
            {'id': 'rust-weights-0123456789abcdef', 'kind': 'rust-weights', 'state': 'unreferenced', 'bytes': 5,
             'model': 'dense', 'deletable': True}]}))
        self.assertEqual(storage['entries'][0]['model'], '')  # the Storage tab files it under "Other"

    # ---- activity rows -------------------------------------------------------------------------

    def test_ledger_rows_carry_mode_and_resume_time_when_present(self):
        ledger = self.call('parseLedger', json.dumps({'ledger_available': True, 'rows': [
            {'t': 9, 'model': 'ornith', 'status': 200, 'mode': 'deep', 'resume_src': 'disk', 'resume_ms': 420},
            {'t': 8, 'model': 'ornith', 'status': 200, 'mode': 'turbo', 'resume_ms': -4},
            {'t': 7, 'model': 'ornith', 'status': 200}]}))
        self.assertEqual([(row['mode'], row['resumeMs'], row['cacheSrc']) for row in ledger['rows']],
                         [('deep', 420, 'disk'), ('', -1, ''), ('', -1, '')])

    # ---- texts ---------------------------------------------------------------------------------

    def test_every_text_key_the_panel_asks_for_exists(self):
        """No raw key or blank label: every literal key (and every key built from a known set) is in en.json."""
        catalog = json.loads((PLUGIN / 'i18n/en.json').read_text(encoding='utf-8'))
        literal = re.compile(r'''\b(?:t|tn|word)\(\s*"([A-Za-z0-9_.-]+)"\s*[),]''')
        asked = set()
        for path in sorted(PLUGIN.glob('*.qml')):
            asked.update(key for key in literal.findall(path.read_text(encoding='utf-8')) if not key.endswith('.'))
        self.assertGreater(len(asked), 80)
        for key in ('now.resuming', 'now.sessionTip', 'now.error', 'mode.title', 'mode.keep', 'cache.clearAll'):
            self.assertIn(key, asked)
        built = ['bar.' + p for p in ('offline', 'sleeping', 'loading', 'ready', 'answering', 'error')]
        built += ['mode.' + m + s for m in ('fast', 'deep', 'max') for s in ('', '.cost')]
        built += ['now.error.' + k for k in ('missing', 'command', 'unreadable')]
        built += ['now.sessionTip.' + s for s in ('ram', 'disk')]
        built += ['activity.tip.resume.' + s for s in ('ram', 'disk')]
        built += ['model.' + k + s for k in ('ornith', 'moe', 'flash') for s in ('', '.short', '.hint')]
        built += ['now.blocked.' + k for k in ('answering', 'images', 'loading', 'stopping')]
        built += ['mode.reason.' + k for k in ('rss', 'stale', 'memory', 'ornith', 'engine', 'none')]
        built += ['share.' + k for k in ('title', 'off', 'tailscale', 'lan', 'off.tip', 'tailscale.tip', 'lan.tip',
                                        'active', 'activePlain', 'inactive', 'key', 'lanWarning', 'remote', 'remoteNow', 'refused',
                                        'confirm.tailscale', 'confirm.lan', 'confirm.yes')]
        built += ['now.countdown', 'now.countdownIdle', 'now.sessionCountdown', 'now.sessionCountdownIdle',
                  'mode.tip', 'mode.tipBusy', 'bar.stopping', 'bar.stoppingReason', 'bar.unlisted']
        for key in sorted(asked | set(built)):
            with self.subTest(key=key):
                self.assertIn(key, catalog)
                value = catalog[key]
                texts = [value] if isinstance(value, str) else list(value.values())
                self.assertTrue(all(isinstance(text, str) and text.strip() for text in texts))

    def test_mode_texts_say_what_each_mode_costs_in_plain_words(self):
        catalog = json.loads((PLUGIN / 'i18n/en.json').read_text(encoding='utf-8'))
        for mode in ('fast', 'deep', 'max'):
            self.assertRegex(catalog['mode.' + mode + '.cost'], r'(?i)summarises')
        self.assertIn('%1', catalog['mode.keep'])
        self.assertIn('%1', catalog['mode.unavailable'])
        for key in ('mode.tip', 'mode.tipBusy', 'mode.fast.cost', 'mode.deep.cost', 'mode.max.cost', 'mode.reason.rss',
                    'mode.reason.stale', 'mode.reason.memory', 'mode.reason.ornith', 'mode.reason.engine'):
            self.assertNotRegex(catalog[key], r'(?i)RSS|compact|n_ctx|KV\b', key)  # no lab jargon
        self.assertRegex(catalog['cache.clearConfirm'], r'(?i)stored-prompt store')
        self.assertRegex(catalog['cache.clearConfirm'], r'(?i)saved conversation')


if __name__ == '__main__':
    unittest.main()
