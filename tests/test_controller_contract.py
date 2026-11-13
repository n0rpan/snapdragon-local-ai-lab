"""v0.1.4 contract for the controller module split: what `controller` exposes must not change.

Tests and tools reach the controller through one namespace (`controller.CONFIG`,
`controller.Manager`, `controller.subprocess`, ...). These lists were taken from the single-file
controller at v0.1.3 (6a42475); a split may move code, never drop or rename any of it. The
status key order and a few control replies are compared exactly.
"""
from contextlib import ExitStack
import http.client
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from snapdragon_ai_lab.runtime import controller, lan_share

NAMES = '''
    BACKEND_PORT BaseHTTPRequestHandler CAPABILITIES CONFIG CPUFREQ_ROOT ClientDisconnect
    ENGINE_DIAGNOSTIC_LOG ENGINE_LEDGER_LOG EngineLogWriter EngineStderrReader FLASH_BACKEND_PORT
    FLASH_CHAT_HTML HARNESS_STATE HOP_HEADERS Handler HarnessRegistry IDLE_SECONDS INFERENCE_PATHS
    JOURNAL_HINT KILL_WAIT_SECONDS LAB_ORIGIN LEASE MAX_CPU_POLICIES MAX_THERMAL_ZONES MODEL_HELPER_PATHS
    Manager ModelBusy ORIGIN_HEADER PRELOAD_FOLLOW_SECONDS PRELOAD_POLL_SECONDS PRELOAD_STATES
    PRELOAD_TIMEOUT PROXY_PORT Path PrefixStore QUEUE_WAIT_SECONDS REAL_PRODUCTION_PATHS REPACK_CACHE_ENV
    REPACK_ERROR_LINES_PER_LOAD REPACK_ERROR_LINE_CHARS REPACK_ERROR_LOG REPACK_ERROR_LOG_BYTES
    REPACK_ERROR_LOG_KEEP REPACK_ERROR_PREFIX ROOT RepackErrorReader SHA256_RE START_SECONDS Server
    TERM_WAIT_SECONDS THERMAL_ROOT ThreadingHTTPServer Unavailable _BadFlashChat _REAL_HOME _log _sysfs_int
    available_memory_bytes bigmoe_runtime cache_turn canonical_target cgroup_memory_max classic_beta
    collections contextmanager disable_repack_cache engine_ledger engine_summary errno explicit_route fcntl
    guard_production_path gzip harness_compat harness_registry hashlib html http idle_seconds image_npu
    image_slot io json lan_share learning_settings listening_inodes listening_process log_effective_engines
    logging main make_plan memory_policy model_catalog notify_repack_errors open_lease_append openai_api os
    oxe_runtime parse_qs port_listener_in_group preload_view profile_policy pwd re repack_cache_enabled
    repack_cache_identity request_defaults request_ledger request_origin require_backend_port resident_bytes
    resource shutil signal socket start_lan_share stat state_cache storage_view subprocess swap_bytes
    swap_guard_mode sys tempfile thermal_sample threading time timings_from_tail total_memory_bytes
    traceback under_test_runner unexpected_cause urlsplit write_repack_error
'''.split()

MANAGER_METHODS = '''
    __init__ reserve_image _engine_writer record_engine_death config eligible models catalog
    flash_qualification flash_cache_setting set_flash_cache_pref held_lease delete_storage select
    context_block set_mode _flash_idle_qos cleanup _swap_guard_hook record_failure clear_failure
    last_failure reaper reap_once queue_length begin_queued begin _begin backend_key backend_json
    prepare_prefix active_block harness_learning harness_status set_harness_learning cache_settings
    runtime_value state_root state_engine_live state_engine_json preload_state note_preload
    note_preload_engine preload_status cache_usage set_cache_budget clear_state_cache clear_legacy_prefix
    prepare_harness end unload_info status close
'''.split()

MANAGER_FIELDS = '''
    mu process lease active inferences last_used state cache flash_cache_status flash_swap_guard_reported
    loaded_config state_engine state_preload launches session_hints preload_info remote_preempt prefix_store
    queue_cv queue_waiters queue_reserved backend_runtime classic_memory_guard flash_power_guard
    repack_reader engine_reader engine_log_writer engine_ledger flash_idle_qos_pid flash_idle_qos_unit
    flash_idle_qos_started flash_idle_qos_error engine_death_lock last_memory_guard last_failure_info
    activity_origin stop_pending _start_attempt current_prefix_key prefix_stats harness_registry
    compat_stats answering_since image_active image_broker image_broker_error harness_mu stop_event thread
'''.split()

HANDLER_METHODS = '''
    protocol_version log_message manager reply json_reply api_settings cors_reply_headers do_HEAD do_GET
    do_POST do_OPTIONS do_DELETE browser_origin_allowed host_allowed forward _forward
'''.split()

STATUS_KEYS = '''
    state validated download_state downloaded_bytes expected_bytes model selected_model selection_error
    backend context_size threads execution_policy_mode memory_policy validation flash_memory gpu_layers
    active_requests queued_requests idle_unload_seconds image_active image_broker_error loaded_model
    resident_bytes idle_unload_remaining_seconds unload prefix_cache backend_kind engine engine_note
    flash_qualification engine_ledger harnesses request_compat harness_learning flash_idle_qos capabilities
    active last_error last_failure error_at error_origin context preload stop_pending image_slot_stale
    lan_share chat_url api_base
'''.split()
BUSY_KEYS = ['state', 'active_requests', 'capabilities', 'active']
ACTIVE_KEYS = ['phase', 'model', 'seconds', 'queued_requests', 'prompt_tokens_done', 'prompt_tokens_total',
               'tokens_out', 'tokens_per_second']
CAPABILITIES = ('models', 'storage', 'harness', 'harness_learning', 'ledger', 'active', 'openai_aliases',
                'state_cache', 'context_modes', 'preload')


class Namespace(unittest.TestCase):
    def test_every_old_top_level_name_is_still_there(self):
        self.assertEqual([name for name in NAMES if not hasattr(controller, name)], [])

    def test_manager_and_handler_keep_their_members(self):
        self.assertEqual([name for name in MANAGER_METHODS if not hasattr(controller.Manager, name)], [])
        self.assertEqual([name for name in HANDLER_METHODS if not hasattr(controller.Handler, name)], [])
        self.assertIsInstance(controller.Manager.last_failure, property)
        self.assertIsInstance(controller.Handler.manager, property)
        self.assertTrue(issubclass(controller.ModelBusy, controller.Unavailable))
        self.assertTrue(issubclass(controller.Handler, controller.BaseHTTPRequestHandler))
        self.assertTrue(issubclass(controller.Server, controller.ThreadingHTTPServer))

    def test_capabilities(self):
        self.assertEqual(controller.CAPABILITIES, CAPABILITIES)


class Stack(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.config = self.root / 'runtime.json'
        self.config.write_text(json.dumps({'model_alias': 'known', 'selected_model': 'invalid',
                                           'profiles': {'known': {'ready': False}}}))
        for name, value in (('CONFIG', self.config), ('HARNESS_STATE', self.root / 'primers'),
                            ('LEASE', self.root / 'inference.lock')):
            self.stack.enter_context(patch.object(controller, name, value))
        self.stack.enter_context(patch.dict(controller.os.environ, {'LOCAL_AI_IMAGE_NPU': '0'}))
        self.manager = controller.Manager()
        self.manager.stop_event.set()
        self.addCleanup(self.manager.close)


class ManagerState(Stack):
    def test_state_fields(self):
        self.assertEqual(sorted(vars(self.manager)), sorted(MANAGER_FIELDS))

    def test_status_key_order(self):
        row = self.manager.status()
        self.assertEqual(list(row), STATUS_KEYS)
        self.assertEqual(list(row['active']), ACTIVE_KEYS)
        self.assertEqual(row['lan_share'], lan_share.off_status(controller.PROXY_PORT))
        self.assertEqual((row['chat_url'], row['api_base']),
                         ('http://127.0.0.1:%d' % controller.PROXY_PORT, 'http://127.0.0.1:%d/v1' % controller.PROXY_PORT))

    def test_status_while_the_lock_is_busy(self):
        held = threading.Event()
        release = threading.Event()

        def hold():
            with self.manager.mu:
                held.set()
                release.wait(10)
        thread = threading.Thread(target=hold)
        thread.start()
        try:
            self.assertTrue(held.wait(10))
            row = self.manager.status()
        finally:
            release.set()
            thread.join(10)
        self.assertEqual(list(row), BUSY_KEYS)
        self.assertEqual((row['state'], row['active_requests'], row['active']['phase']), ('loading', 0, 'loading'))


class ControlReplies(Stack):
    def setUp(self):
        super().setUp()
        self.server = controller.Server(('127.0.0.1', 0), controller.Handler)
        self.server.manager = self.manager
        self.port = self.server.server_address[1]
        thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(thread.join, 10)
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)

    def call(self, method, path, body=None, headers=None):
        connection = http.client.HTTPConnection('127.0.0.1', self.port, timeout=10)
        try:
            connection.request(method, path, body=body, headers=headers or {})
            response = connection.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            connection.close()

    def test_exact_control_replies(self):
        status, headers, body = self.call('GET', '/_local/nope')
        self.assertEqual((status, body), (404, b'{"error": "Unknown local control route."}'))
        self.assertEqual((headers['Content-Type'], headers['Cache-Control'], headers['Content-Length']),
                         ('application/json', 'no-store', str(len(body))))
        self.assertEqual(self.call('GET', '/_local/status', headers={'Host': 'example.test'})[::2], (
            403, b'{"error": "Host not allowed: use 127.0.0.1, localhost or [::1] with this port."}'))
        self.assertEqual(self.call('POST', '/_local/mode', body=b'{}')[::2], (
            400, b'{"error": "Send {\\"model\\": alias, \\"mode\\": \\"fast\\"|\\"deep\\"|\\"max\\"}."}'))
        self.assertEqual(self.call('POST', '/_local/unload', body=b'{"model": "known"}')[::2],
                         (409, b'{"error": "That model is not loaded."}'))
        self.assertEqual(self.call('POST', '/_local/harness/forget', body=b'{}')[::2],
                         (200, b'{"forgotten": "all", "ignored": false}'))
        self.assertEqual(self.call('GET', '/_local/share')[::2],
                         (200, json.dumps(lan_share.off_status(controller.PROXY_PORT)).encode()))

    def test_status_over_http_keeps_the_key_order(self):
        status, _, body = self.call('GET', '/_local/status')
        self.assertEqual(status, 200)
        self.assertEqual(list(json.loads(body)), STATUS_KEYS)


if __name__ == '__main__':
    unittest.main()
