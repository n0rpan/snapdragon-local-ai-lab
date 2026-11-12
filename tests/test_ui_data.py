"""Panel/OpenCode-facing data: catalog rows, prefix warmth, storage listing and safe delete.

Synthetic files only; no live config, controller port, model or lease is touched.
"""
import copy
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from snapdragon_ai_lab.runtime import classic_beta, controller, model_catalog, storage_view

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / 'src/snapdragon_ai_lab/runtime'
PLUGIN = ROOT / 'src/snapdragon_ai_lab/omarchy_plugin'
loader = importlib.machinery.SourceFileLoader('ui_data_local_ai', str(RUNTIME / 'local-ai'))
spec = importlib.util.spec_from_loader(loader.name, loader)
cli = importlib.util.module_from_spec(spec); loader.exec_module(cli)
plan_spec = importlib.util.spec_from_file_location('ui_data_plan', RUNTIME / 'launch_local_plan.py')
launcher = importlib.util.module_from_spec(plan_spec); plan_spec.loader.exec_module(launcher)

ORNITH, MOE, DENSE, FLASH = 'ornith-35b-a3b-local', 'qwen3.6-35b-a3b-local', 'qwen3.8-27b-local', 'qwen3.8-flash-next-local'
SHA = {ORNITH: 'a' * 64, MOE: 'b' * 64, DENSE: 'd' * 64}
CLASSIC, RUST = '1' * 64, '6' * 64


def rust_entry(directory, model, server, name='e' * 64):
    identity = f'model=sha256:{model} server=sha256:{server} state=oxe-state-v1 ubatch=256'.encode()
    path = Path(directory) / (name + '.bin')
    path.write_bytes(b'OXEPFX01' + len(identity).to_bytes(4, 'little') + identity + b'\0' * 64)
    return path


def llama_entry(directory, model, server, name='f' * 64):
    Path(directory, name + '.json').write_text(json.dumps({'model_sha256': model, 'server_sha256': server}))
    Path(directory, name + '.bin').write_bytes(b'\0' * 32)


def document(home):
    def profile(alias, size, rust=True):
        value = {'ready': True, 'model_alias': alias, 'sha256': SHA[alias], 'expected_bytes': size,
                 'model_path': f'/synthetic/{alias}.gguf', 'server_binary': '/synthetic/llama-server',
                 'server_sha256': CLASSIC, 'threads': 4, 'context_size': 16384,
                 'prefix_cache': {'enabled': True, 'directory': str(Path(home) / '.cache/local-ai/prefixes')}}
        if rust:
            value['memlat_overrides'] = {'oxe': {'server_binary': '/synthetic/oxe-server', 'server_sha256': RUST,
                                                 'oxe_args': ['--model', '{model}', '--prefix-cache', 'default']}}
        return value
    base = profile(DENSE, 16 * 1024**3, rust=False)
    flash = {'ready': True, 'model_alias': FLASH, 'backend_kind': 'bigmoe-cpu-stream', 'display_name': 'Flash-Next planning',
             'bigmoe': {'admission': {'admitted_budget_bytes': 21 * 1024**3}}, 'prefix_cache': {'enabled': False}}
    return {**base, 'selected_model': ORNITH, 'profiles': {
        DENSE: copy.deepcopy(base), MOE: profile(MOE, 21 * 1024**3), FLASH: flash,
        ORNITH: {**profile(ORNITH, 16 * 1024**3), 'display_name': 'Ornith 1.5 35B A3B (beta)', 'release_channel': 'beta'}}}


class Catalog(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(); self.home = Path(self.temporary.name)
        (self.home / '.cache/local-ai/prefixes').mkdir(parents=True)
        (self.home / '.cache/local-ai/snapdragon-engine/prefix').mkdir(parents=True)
        self.config = self.home / 'runtime.json'; self.config.write_text(json.dumps(document(self.home)))

    def tearDown(self): self.temporary.cleanup()

    def manager(self):
        manager = object.__new__(controller.Manager)
        manager.loaded_config, manager.state = None, 'unloaded'
        return manager

    def rows(self, oxe_enabled):
        # M4: Rust is the default; only an explicit 0 selects the classic fallback.
        env = {'LOCAL_AI_ENABLE_OXE': '1' if oxe_enabled else '0'}
        with patch.object(controller, 'CONFIG', self.config), patch.dict(os.environ, env, clear=False), \
                patch.object(controller.profile_policy, 'memlat_available', return_value=True), \
                patch.object(Path, 'home', return_value=self.home), \
                patch.object(controller.Manager, 'eligible', return_value=True):
            manager = self.manager()
            return manager.catalog(), manager.models()

    def test_rows_show_engine_status_default_and_clean_names(self):
        catalog, models = self.rows(oxe_enabled=True)
        rows = {row['id']: row for row in catalog['data']}
        self.assertEqual([row['id'] for row in catalog['data']], [ORNITH, MOE, FLASH, DENSE])
        self.assertEqual(catalog['default_model'], ORNITH)
        self.assertEqual((rows[ORNITH]['engine'], rows[ORNITH]['engine_status']), ('rust', 'production'))
        self.assertEqual((rows[DENSE]['engine'], rows[DENSE]['engine_status']), ('llama', 'classic'))
        self.assertEqual((rows[FLASH]['engine'], rows[FLASH]['battery_start_percent']), ('stream', 30))
        self.assertEqual(rows[FLASH]['memory_gib'], 25.0)
        self.assertEqual(rows[ORNITH]['memory_gib'], 19.0)
        self.assertTrue(rows[ORNITH]['default'] and rows[ORNITH]['selected'])
        self.assertEqual(rows[FLASH]['prefix'], 'off')
        names = {row['id']: row['name'] for row in models}
        self.assertEqual(names[ORNITH], 'Ornith 1.5 35B A3B')
        self.assertEqual(names[FLASH], 'Qwen3.8 Flash-Next')
        self.assertFalse(any('beta' in n.lower() or 'planning' in n.lower() for n in names.values()))
        self.assertEqual(names['qwen-local'], model_catalog.SELECTED_ALIAS_NAME)

    def test_rust_flag_off_reports_classic_fallback(self):
        catalog, _ = self.rows(oxe_enabled=False)
        rows = {row['id']: row for row in catalog['data']}
        self.assertEqual((rows[ORNITH]['engine'], rows[ORNITH]['engine_status']), ('llama', 'fallback'))

    def test_prefix_warmth_needs_exact_model_and_engine(self):
        rust_dir = self.home / '.cache/local-ai/snapdragon-engine/prefix'
        rust_entry(rust_dir, SHA[MOE], RUST)
        rust_entry(rust_dir, SHA[ORNITH], '9' * 64, name='c' * 64)  # older engine: cold
        llama_entry(self.home / '.cache/local-ai/prefixes', SHA[DENSE], CLASSIC)
        rows = {row['id']: row for row in self.rows(oxe_enabled=True)[0]['data']}
        self.assertEqual((rows[MOE]['prefix'], rows[ORNITH]['prefix'], rows[DENSE]['prefix']), ('warm', 'cold', 'warm'))

    def test_loaded_row_state(self):
        with patch.object(controller, 'CONFIG', self.config), patch.object(controller.Manager, 'eligible', return_value=True), \
                patch.object(Path, 'home', return_value=self.home):
            manager = self.manager(); manager.loaded_config, manager.state = {'model_alias': MOE}, 'ready'
            rows = {row['id']: row['state'] for row in manager.catalog()['data']}
        self.assertEqual((rows[MOE], rows[ORNITH]), ('ready', 'unloaded'))

    def test_header_reader_is_bounded_and_ignores_foreign_files(self):
        directory = self.home / '.cache/local-ai/snapdragon-engine/prefix'
        (directory / ('1' * 64 + '.bin')).write_bytes(b'OXEPFX01' + (10**6).to_bytes(4, 'little'))
        (directory / 'notes.bin').write_bytes(b'x')
        self.assertEqual(model_catalog.prefix_identities(directory, 'rust'), {})


class Storage(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(); self.home = Path(self.temporary.name)
        cache = self.home / '.cache/local-ai'; engine = cache / 'snapdragon-engine'
        for d in (engine / 'prefix', engine / 'flash', cache / 'prefixes', cache / 'repack/x-ornith-35b-a3b-local.gguf-0123456789ab',
                  cache / 'repack/retired.gguf-0123456789ab'):
            d.mkdir(parents=True)
        (engine / 'flash/experts-r4.bin').write_bytes(b'\0' * 100)
        self.live = engine / f"{SHA[ORNITH]}-w1-ee6b9a1-2719b331.bin"; self.live.write_bytes(b'\0' * 10)
        self.old = engine / f"{'0' * 64}-w1-ee6b9a1-2719b331-mtp.bin"; self.old.write_bytes(b'\0' * 20)
        rust_entry(engine / 'prefix', SHA[MOE], RUST)
        self.stale_prefix = rust_entry(engine / 'prefix', SHA[MOE], '9' * 64, name='c' * 64)
        llama_entry(cache / 'prefixes', SHA[DENSE], CLASSIC)
        llama_entry(cache / 'prefixes', '7' * 64, CLASSIC, name='7' * 64)
        (cache / 'repack/retired.gguf-0123456789ab/blob').write_bytes(b'\0' * 30)
        self.document = document(self.home)
        self.document['profiles'][ORNITH]['model_path'] = '/synthetic/x-ornith-35b-a3b-local.gguf'

    def tearDown(self): self.temporary.cleanup()

    def rows(self, busy=()):
        return {(r['kind'], r['name']): r for r in storage_view.inventory(self.document, home=self.home, busy=set(busy))}

    def test_listing_marks_references_and_sizes(self):
        rows = self.rows()
        state = {key: (r['state'], r['deletable']) for key, r in rows.items()}
        self.assertEqual(state[('rust-weights', self.live.name)], ('referenced', False))
        self.assertEqual(state[('rust-weights', self.old.name)], ('unreferenced', True))
        self.assertEqual(state[('rust-prefix', SHA[MOE][:12] + '-' + RUST[:12])], ('referenced', False))
        self.assertEqual(sorted(r['state'] for k, r in rows.items() if k[0] == 'rust-prefix'), ['referenced', 'unreferenced'])
        self.assertEqual(state[('flash-sidecar', 'flash')], ('protected', False))
        self.assertEqual(state[('llama-prefix', SHA[DENSE][:12] + '-' + CLASSIC[:12])], ('referenced', False))
        self.assertEqual(state[('llama-prefix', '7' * 12 + '-' + CLASSIC[:12])], ('unreferenced', True))
        self.assertEqual(state[('llama-repack', 'x-ornith-35b-a3b-local.gguf-0123456789ab')], ('referenced', False))
        self.assertEqual(state[('llama-repack', 'retired.gguf-0123456789ab')], ('unreferenced', True))
        self.assertEqual(rows[('flash-sidecar', 'flash')]['bytes'], 100)
        self.assertEqual(rows[('llama-prefix', '7' * 12 + '-' + CLASSIC[:12])]['bytes'], 32 + len(json.dumps({'model_sha256': '7' * 64, 'server_sha256': CLASSIC})))
        self.assertTrue(all('paths' not in r for r in storage_view.public(list(rows.values()))))

    def test_delete_only_unreferenced_unused(self):
        rows = self.rows()
        for key in [('rust-weights', self.live.name), ('flash-sidecar', 'flash'), ('rust-prefix', SHA[MOE][:12] + '-' + RUST[:12])]:
            with self.assertRaises(PermissionError):
                storage_view.delete(self.document, rows[key]['id'], home=self.home, busy=set())
        self.assertTrue(self.live.exists() and (self.home / '.cache/local-ai/snapdragon-engine/flash').exists())
        old = rows[('rust-weights', self.old.name)]['id']
        with self.assertRaises(PermissionError):  # mapped by a process
            storage_view.delete(self.document, old, home=self.home, busy={str(self.old)})
        self.assertEqual(storage_view.delete(self.document, old, home=self.home, busy=set())['freed_bytes'], 20)
        self.assertFalse(self.old.exists())
        repack = rows[('llama-repack', 'retired.gguf-0123456789ab')]['id']
        storage_view.delete(self.document, repack, home=self.home, busy=set())
        self.assertFalse((self.home / '.cache/local-ai/repack/retired.gguf-0123456789ab').exists())
        stale = [r for k, r in rows.items() if k[0] == 'rust-prefix' and r['state'] == 'unreferenced'][0]['id']
        storage_view.delete(self.document, stale, home=self.home, busy=set())
        self.assertFalse(self.stale_prefix.exists())
        with self.assertRaises(LookupError):
            storage_view.delete(self.document, old, home=self.home, busy=set())
        for bad in (None, '../x', 'rust-weights-XYZ'):
            with self.assertRaises(ValueError):
                storage_view.delete(self.document, bad, home=self.home, busy=set())

    def test_symlinked_entries_are_not_listed(self):
        target = self.home / 'elsewhere'; target.mkdir()
        (self.home / '.cache/local-ai/repack/link.gguf-0123456789ab').symlink_to(target)
        self.assertNotIn(('llama-repack', 'link.gguf-0123456789ab'), self.rows())

    def test_controller_delete_requires_free_lease_and_no_request(self):
        config = self.home / 'runtime.json'; config.write_text(json.dumps(self.document))
        lease = self.home / 'lease'
        manager = object.__new__(controller.Manager)
        import threading; manager.mu = threading.RLock(); manager.active = 0; manager.lease = None
        old = self.rows()[('rust-weights', self.old.name)]['id']
        with patch.object(controller, 'CONFIG', config), patch.object(controller, 'LEASE', lease), \
                patch.object(Path, 'home', return_value=self.home), patch.object(storage_view, 'active_paths', return_value=set()):
            manager.active = 1
            with self.assertRaises(controller.Unavailable): manager.delete_storage(old)
            manager.active = 0
            holder = lease.open('a'); controller.fcntl.flock(holder, controller.fcntl.LOCK_EX)
            try:
                with self.assertRaises(controller.Unavailable): manager.delete_storage(old)
            finally: holder.close()
            self.assertTrue(self.old.exists())
            self.assertEqual(manager.delete_storage(old)['deleted'], old)
        self.assertFalse(self.old.exists())


class Labels(unittest.TestCase):
    def test_ornith_names_drop_beta_but_channel_gate_remains(self):
        fragment = classic_beta.opencode_fragment()['providers']['local-qwen']['models'][classic_beta.ALIAS]
        self.assertEqual(fragment['name'], 'Ornith 1.5 35B A3B')
        self.assertEqual(classic_beta.RELEASE_CHANNELS, {'beta', 'production'})
        with patch.object(classic_beta, '_check_candidate_report'):
            profile = {'ready': True, 'release_channel': 'production', 'validation': {'report_sha256': 'a' * 64,
                       'root_reviewed_report_sha256': 'a' * 64}}
            profile['validation']['report_content_sha256'] = classic_beta.digest({})
            self.assertTrue(classic_beta._candidate_eligible(profile))
            self.assertFalse(classic_beta._candidate_eligible({**profile, 'release_channel': 'experimental'}))

    def test_ornith_qualified_request_defaults_are_kept(self):
        value = classic_beta.request_defaults({'messages': []}, '/v1/chat/completions')
        self.assertEqual((value['temperature'], value['max_tokens']), (0, 2048))

    def test_panel_strings_have_no_beta_or_planning_only_framing(self):
        strings = json.loads((PLUGIN / 'i18n/en.json').read_text())
        text = json.dumps(strings).lower()
        self.assertNotIn('beta', text)
        self.assertNotIn('planning', text)
        for key in set(__import__('re').findall(r'localAiStrings\.t\("([A-Za-z.]+)"\)', (PLUGIN / 'Panel.qml').read_text())):
            self.assertIn(key, strings)


class Launch(unittest.TestCase):
    def test_flash_opens_build_agent_with_pinned_request_policy(self):
        value = json.loads(launcher.environment_for_classic_model({}, 'flash')['OPENCODE_CONFIG_CONTENT'])
        self.assertEqual(value['model'], 'local-qwen/' + FLASH)
        self.assertNotIn('default_agent', value)
        self.assertNotIn('policies', value)
        body = value['providers']['local-qwen']['body']
        self.assertEqual((body['temperature'], body['chat_template_kwargs']), (0, {'enable_thinking': False}))

    def test_code_action_with_flash_selected_launches_build_not_planner(self):
        current = {'selected_model': 'flash', 'flash_available': True}
        with patch.object(cli, 'ensure_service', return_value=True), patch.object(cli, 'qualified_flash_offered', return_value=True), \
                patch.object(cli, 'status', return_value=current), patch.object(cli.subprocess, 'Popen') as popen:
            self.assertEqual(cli.action('code'), 0)
            command = popen.call_args.args[0]
            self.assertEqual(command[-3:-1], ['--model', 'flash'])
            self.assertEqual(cli.action('plan'), 0)
            self.assertNotIn('--model', popen.call_args.args[0])


# The Status.js parsing checks moved with the panel redesign: Model.js is covered by test_panel_model.py.


class PluginStrings(unittest.TestCase):
    """Every literal key the plugin asks for exists in en.json, and no component id shadows the
    translations object (the blank-panel bug: ids must be localAiStrings, never strings or tr)."""
    def test_every_literal_key_in_plugin_qml_exists(self):
        import re
        strings = json.loads((PLUGIN / 'i18n/en.json').read_text())
        checked = 0
        for path in sorted(PLUGIN.glob('*.qml')):
            text = path.read_text(encoding='utf-8')
            for key in set(re.findall(r'\b(?:t|tn|word)\(\s*"([A-Za-z][A-Za-z0-9.\-]*)"\s*[,)]', text)):
                checked += 1
                with self.subTest(file=path.name, key=key):
                    self.assertIn(key, strings)
        self.assertGreater(checked, 40)

    def test_no_component_id_or_binding_shadows_the_translations(self):
        import re
        for path in sorted(PLUGIN.glob('*.qml')):
            text = path.read_text(encoding='utf-8')
            with self.subTest(file=path.name):
                self.assertIsNone(re.search(r'\bid:\s*(strings|tr)\b', text))
                self.assertIsNone(re.search(r'\bstrings:\s*strings\b', text))
                self.assertIsNone(re.search(r'\btr:\s*tr\b', text))


if __name__ == '__main__':
    unittest.main()
