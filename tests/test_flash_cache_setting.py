"""Idle setting changes against a temporary config and loopback HTTP server; no engine."""
import copy
import http.client
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch
from snapdragon_ai_lab.runtime import controller, flash_cache_pref as fcp
from test_flash_admission_rule import flash_profile, make_config, GIB

ALIAS = 'qwen3.8-flash-next-local'
ROUTE = '/_local/flash/cache'


class FlashCacheSetting(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'runtime.json'
        profile = flash_profile()
        profile['memlat_overrides']['oxe_flash'].update(cache_pref='20', desktop_floor_gib=2.0)
        self.value = {'model_alias': 'other', 'selected_model': 'other', 'profiles': {ALIAS: profile},
                      'unrelated': {'keep': 'all values'}}
        self.path.write_text(json.dumps(self.value)); self.path.chmod(0o600)
        self.config_patch = patch.object(controller, 'CONFIG', self.path); self.config_patch.start()
        self.addCleanup(self.config_patch.stop)
        self.env_patch = patch.dict(os.environ, {}, clear=True); self.env_patch.start()
        self.addCleanup(self.env_patch.stop)
        # S2: 20/18 are offered only with swap; pin the host's swap instead of reading /proc/meminfo.
        self.swap_patch = patch.object(controller, 'swap_bytes', return_value=(16 * GIB, 16 * GIB))
        self.swap_patch.start(); self.addCleanup(self.swap_patch.stop)
        self.manager = controller.Manager.__new__(controller.Manager)
        self.manager.mu = threading.RLock(); self.manager.active = 0
        self.manager.loaded_config = None
        self.manager.stop_pending = None  # read by stop_for_control() (audit S5)
        self.manager.queue_length = Mock(return_value=0)
        self.manager.config = Mock(return_value=make_config(cache_gib=14, peak=22 * GIB))
        self.manager.cleanup = Mock()
        self.manager.begin = Mock(side_effect=AssertionError('settings must never wake a model'))
        self.server = controller.Server(('127.0.0.1', 0), controller.Handler)
        self.server.manager = self.manager
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={'poll_interval': .01}, daemon=True)
        self.thread.start()
        self.addCleanup(self.stop_server)

    def stop_server(self):
        self.server.shutdown(); self.server.server_close(); self.thread.join(timeout=2)

    def request(self, method='GET', payload=None):
        connection = http.client.HTTPConnection('127.0.0.1', self.server.server_port, timeout=3)
        try:
            connection.request(method, ROUTE, body=json.dumps(payload) if payload is not None else None)
            response = connection.getresponse()
            return response.status, json.loads(response.read())
        finally:
            connection.close()

    def test_get_is_nonwaking_and_reports_enabled_profile(self):
        before = self.path.read_bytes()
        status, body = self.request()
        self.assertEqual(status, 200)
        self.assertEqual(body, {'enabled': True, 'pref': '20', 'choices': ['auto', '20', '18', '16', '14'],
                                'desktop_floor_gib': 2.0, 'editable': True})
        self.assertEqual(self.path.read_bytes(), before)
        self.manager.begin.assert_not_called(); self.manager.cleanup.assert_not_called()

    def test_change_only_preference_while_other_model_remains_loaded(self):
        self.manager.loaded_config = {'model_alias': 'other'}
        status, body = self.request('POST', {'pref': '14'})
        expected = copy.deepcopy(self.value)
        expected['profiles'][ALIAS]['memlat_overrides']['oxe_flash']['cache_pref'] = '14'
        self.assertEqual(status, 200); self.assertEqual(body['pref'], '14')
        self.assertEqual(json.loads(self.path.read_text()), expected)
        self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)
        self.manager.cleanup.assert_not_called(); self.manager.begin.assert_not_called()
        self.assertEqual(list(self.path.parent.glob('.flash-cache-*')), [])

    def test_idle_flash_unloads_before_atomic_replacement_and_noop_keeps_it(self):
        self.manager.loaded_config = {'model_alias': ALIAS}
        self.assertEqual(self.request('POST', {'pref': '20'})[0], 200)
        self.manager.cleanup.assert_not_called()
        self.manager.cleanup.side_effect = lambda: self.assertEqual(json.loads(self.path.read_text()), self.value)
        self.assertEqual(self.request('POST', {'pref': '18'})[0], 200)
        self.manager.cleanup.assert_called_once()
        self.assertEqual(self.request()[1]['pref'], '18')

    def test_busy_or_queued_rejects_without_writes(self):
        before = self.path.read_bytes()
        self.manager.active = 1
        self.assertEqual(self.request('POST', {'pref': '14'})[0], 409)
        self.manager.active = 0; self.manager.queue_length.return_value = 1
        self.assertEqual(self.request('POST', {'pref': '14'})[0], 409)
        self.assertEqual(self.path.read_bytes(), before); self.manager.cleanup.assert_not_called()

    def test_fixed_profile_stays_disabled_and_environment_override_cannot_be_overwritten(self):
        os.environ[fcp.ENV] = '18'
        self.assertFalse(self.request()[1]['editable'])
        self.assertEqual(self.request('POST', {'pref': '14'})[0], 409)
        del os.environ[fcp.ENV]
        del self.value['profiles'][ALIAS]['memlat_overrides']['oxe_flash']['cache_pref']
        self.path.write_text(json.dumps(self.value)); before = self.path.read_bytes()
        self.assertEqual(self.request(), (200, {'enabled': False}))
        self.assertEqual(self.request('POST', {'pref': '20'})[0], 409)
        self.assertEqual(self.path.read_bytes(), before)

    def test_no_swap_host_offers_and_accepts_no_20_or_18(self):
        # S2 (v0.1.1): without swap the MemTotal rule would page the desktop to nowhere.
        before = self.path.read_bytes()
        for swap in ((0, 0), (None, None)):
            with self.subTest(swap=swap), patch.object(controller, 'swap_bytes', return_value=swap):
                status, body = self.request()
                self.assertEqual(status, 200)
                self.assertEqual(body['choices'], ['auto', '16', '14'])
                for pref in ('20', '18'):
                    status, body = self.request('POST', {'pref': pref})
                    self.assertEqual(status, 400); self.assertIn('need swap', body['error'])
                self.assertEqual(self.path.read_bytes(), before)
        with patch.object(controller, 'swap_bytes', return_value=(0, 0)):
            self.assertEqual(self.request('POST', {'pref': '16'})[0], 200)

    def test_malformed_preferences_and_extra_fields_are_400(self):
        before = self.path.read_bytes()
        for payload in ({}, {'pref': 20}, {'pref': True}, {'pref': None}, {'pref': '21'},
                        {'pref': '20GiB'}, {'pref': '14', 'desktop_floor_gib': 0}, []):
            with self.subTest(payload=payload):
                self.assertEqual(self.request('POST', payload)[0], 400)
        self.assertEqual(self.path.read_bytes(), before); self.manager.cleanup.assert_not_called()

    def test_failed_atomic_replace_keeps_original_and_cleans_temporary(self):
        before = self.path.read_bytes()
        with patch.object(controller.os, 'replace', side_effect=OSError('synthetic replace failure')):
            self.assertEqual(self.request('POST', {'pref': 'auto'})[0], 503)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(list(self.path.parent.glob('.flash-cache-*')), [])
        self.assertEqual(self.request('POST', {'pref': 'auto'})[0], 200)
