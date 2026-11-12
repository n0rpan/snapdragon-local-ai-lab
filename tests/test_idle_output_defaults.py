"""Approved idle/output defaults (2026-11-10): synthetic only, never starts an engine."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from snapdragon_ai_lab.runtime import classic_beta as beta, controller, oxe_runtime as oxe, profile_policy

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('stage_idle_output_defaults', ROOT / 'tools/stage_idle_output_defaults.py')
stage_tool = importlib.util.module_from_spec(spec); spec.loader.exec_module(stage_tool)
from test_oxe_production import RUST_SHA, UNIT, profile, stage_tool as oxe_stage  # noqa: E402

CHAT = '/v1/chat/completions'
ORNITH = {'backend_kind': 'oxe', 'model_alias': beta.ALIAS, 'context_size': 32768, 'default_max_tokens': 16384}
QWEN = {'backend_kind': 'oxe', 'model_alias': 'qwen3.6-35b-a3b-local', 'context_size': 16384, 'default_max_tokens': 16384}


class OxeOutputDefault(unittest.TestCase):
    def test_default_only_when_client_omits_every_length_field(self):
        self.assertEqual(oxe.request_defaults({'messages': []}, CHAT, ORNITH)['max_tokens'], 16384)
        self.assertEqual(oxe.request_defaults({'prompt': 'x'}, '/completion', QWEN)['n_predict'], 16384)
        self.assertNotIn('max_tokens', oxe.request_defaults({'prompt': 'x'}, '/completion', QWEN))
        self.assertEqual(oxe.request_defaults({'prompt': 'x'}, '/v1/completions', QWEN)['max_tokens'], 16384)
        for field in oxe.LENGTH_FIELDS:
            for value in (1, 64, 100000, None, -1):
                sent = {'messages': [], field: value}
                got = oxe.request_defaults(sent, CHAT, ORNITH)
                self.assertEqual({k: got[k] for k in oxe.LENGTH_FIELDS if k in got}, {field: value})

    def test_bounded_by_context_and_opt_in_per_profile(self):
        self.assertEqual(oxe.request_defaults({}, CHAT, {**QWEN, 'context_size': 8192})['max_tokens'], 8192)
        for config in (None, {}, {'context_size': 16384}, {**QWEN, 'default_max_tokens': 0},
                       {**QWEN, 'default_max_tokens': '16384'}, {**QWEN, 'default_max_tokens': True}):
            self.assertNotIn('max_tokens', oxe.request_defaults({}, CHAT, config))
        self.assertEqual(oxe.request_defaults({'input': 'x'}, '/tokenize', QWEN), {'input': 'x', 'temperature': 0, 'cache_prompt': True})

    def test_sampling_and_thinking_defaults_unchanged(self):
        got = oxe.request_defaults({}, CHAT, ORNITH)
        self.assertEqual((got['temperature'], got['cache_prompt'], got['chat_template_kwargs']), (0, True, {'enable_thinking': False, 'preserve_thinking': True}))
        explicit = {'temperature': .7, 'chat_template_kwargs': {'enable_thinking': True}, 'max_tokens': 10}
        # CACHE-PERFECT P1: preserve_thinking on every chat request (thinking on too)
        self.assertEqual(oxe.request_defaults(explicit, CHAT, ORNITH),
                         {**explicit, 'cache_prompt': True,
                          'chat_template_kwargs': {'enable_thinking': True, 'preserve_thinking': True}})


class ControllerDefaults(unittest.TestCase):
    def test_idle_unload_default_is_600_seconds(self):
        if 'LOCAL_LLM_IDLE_SECONDS' not in os.environ:
            self.assertEqual(controller.IDLE_SECONDS, 600)

    def test_oxe_ornith_skips_classic_2048_cap(self):
        self.assertEqual(controller.request_defaults(ORNITH, {}, CHAT)['max_tokens'], 16384)
        self.assertEqual(controller.request_defaults(ORNITH, {'max_tokens': 300}, CHAT)['max_tokens'], 300)
        classic = {'model_alias': beta.ALIAS}
        self.assertEqual(controller.request_defaults(classic, {}, CHAT)['max_tokens'], 2048)
        llama = {'model_alias': 'qwen3.8-27b-local'}
        payload = {}
        self.assertIs(controller.request_defaults(llama, payload, CHAT), payload)

    def test_qwen_local_and_unnamed_requests_resolve_to_selected_defaults(self):
        runtime = stage_tool.add_output_defaults(staged_oxe_runtime())
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / 'runtime.json'; config.write_text(json.dumps(runtime))
            manager = controller.Manager(); manager.stop_event.set()
            try:
                with patch.dict(os.environ, {'LOCAL_AI_ENABLE_OXE': '1'}), patch.object(controller, 'CONFIG', config), \
                        patch.object(profile_policy, 'memlat_available', return_value=True):
                    for alias in (None, '', 'qwen-local', beta.ALIAS):
                        target = manager.config(alias)
                        self.assertEqual((target['model_alias'], target['backend_kind']), (beta.ALIAS, 'oxe'))
                        self.assertEqual(controller.request_defaults(target, {}, CHAT)['max_tokens'], 16384)
                    qwen = manager.config('qwen3.6-35b-a3b-local')
                    self.assertEqual(controller.request_defaults(qwen, {}, CHAT)['max_tokens'], 16384)
                    dense = manager.config('qwen3.8-27b-local')
                    self.assertNotIn('max_tokens', controller.request_defaults(dense, {}, CHAT))
            finally:
                manager.close()


def staged_oxe_runtime():
    base = profile(alias='qwen3.8-27b-local', model_sha='9' * 64); del base['memlat_overrides']
    profiles = {'qwen3.8-27b-local': copy.deepcopy(base)}
    for alias, model_sha in oxe_stage.QUALIFIED_MODELS.items():
        p = profile(alias=alias, model_sha=model_sha)
        p['memlat_overrides']['oxe'] = oxe_stage.oxe_override('/synthetic/oxe-server', RUST_SHA)
        profiles[alias] = p
    return {**base, 'selected_model': beta.ALIAS, 'profiles': profiles}


def opencode():
    limit = lambda ctx, out: {'context': ctx, 'input': ctx - 4096, 'output': out}
    models = {'qwen-local': limit(16384, 2048), 'qwen3.8-27b-local': limit(16384, 4096),
              'qwen3.6-35b-a3b-local': limit(16384, 4096), 'qwen3.8-flash-next-local': limit(16384, 2048),
              beta.ALIAS: limit(32768, 2048)}
    return {'model': 'local-qwen/' + beta.ALIAS,
            'providers': {'local-qwen': {'settings': {'apiKey': 'synthetic-secret'},
                                         'models': {k: {'name': k, 'limit': v} for k, v in models.items()}}}}


class StageTool(unittest.TestCase):
    def test_runtime_adds_default_only_to_oxe_ornith_and_qwen(self):
        before = staged_oxe_runtime()
        after = stage_tool.add_output_defaults(before)
        self.assertEqual(before, staged_oxe_runtime())
        for alias in stage_tool.OUTPUT_ALIASES:
            changed = copy.deepcopy(after['profiles'][alias])
            self.assertEqual(changed.pop('default_max_tokens'), 16384)
            self.assertEqual(changed, before['profiles'][alias])
        self.assertEqual(after['profiles']['qwen3.8-27b-local'], before['profiles']['qwen3.8-27b-local'])
        self.assertEqual({k: v for k, v in after.items() if k != 'profiles'}, {k: v for k, v in before.items() if k != 'profiles'})
        with self.assertRaisesRegex(ValueError, 'already'):
            stage_tool.add_output_defaults(after)

    def test_opencode_raises_only_the_three_output_caps(self):
        before = opencode(); after = stage_tool.raise_opencode_output(before)
        models, old = after['providers']['local-qwen']['models'], before['providers']['local-qwen']['models']
        for alias in models:
            expected = copy.deepcopy(old[alias])
            if alias in stage_tool.OPENCODE_ALIASES:
                expected['limit']['output'] = 16384
            self.assertEqual(models[alias], expected)
        self.assertEqual(after['providers']['local-qwen']['settings'], before['providers']['local-qwen']['settings'])
        with self.assertRaisesRegex(ValueError, 'already'):
            stage_tool.raise_opencode_output(after)

    def test_unit_sets_idle_once_and_points_at_release(self):
        unit = stage_tool.rewrite_unit(UNIT, '/synthetic/runtime-releases/new')
        self.assertEqual(unit.count('Environment="LOCAL_LLM_IDLE_SECONDS=600"'), 1)
        self.assertIn('ExecStart=/usr/bin/python3 "/synthetic/runtime-releases/new/controller.py"\n', unit)
        self.assertNotIn('/old', unit)
        with self.assertRaises(ValueError):
            stage_tool.rewrite_unit(unit, '/synthetic/runtime-releases/new')

    def test_stage_writes_payload_and_exact_rollback_preimages(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            live = {'runtime': tmp / 'runtime.json', 'unit': tmp / 'unit.service', 'opencode': tmp / 'opencode.json'}
            live['runtime'].write_text(json.dumps(staged_oxe_runtime()))
            live['unit'].write_text(UNIT)
            live['opencode'].write_text(json.dumps(opencode(), indent=4))
            digest = {k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in live.items()}
            kwargs = dict(runtime_path=live['runtime'], runtime_sha=digest['runtime'], unit_path=live['unit'],
                          unit_sha=digest['unit'], opencode_path=live['opencode'], opencode_sha=digest['opencode'],
                          releases_root=tmp / 'releases')
            with patch.object(profile_policy, 'memlat_available', return_value=True):
                with self.assertRaisesRegex(ValueError, 'identity_changed'):
                    stage_tool.stage(tmp / 'bad', **{**kwargs, 'opencode_sha': '0' * 64})
                self.assertFalse((tmp / 'bad').exists())
                with self.assertRaisesRegex(ValueError, 'outside'):
                    stage_tool.stage(ROOT / 'stage-in-git', **kwargs)
                result = stage_tool.stage(tmp / 'stage', **kwargs)
            stage = tmp / 'stage'
            for name, path in live.items():
                self.assertEqual((stage / 'rollback' / (name + '.original')).read_bytes(), path.read_bytes())
            self.assertEqual(live['unit'].read_text(), UNIT)
            self.assertIn(result['release_dir'], (stage / 'local-ai-chat.service').read_text())
            staged = json.loads((stage / 'runtime.json').read_text())
            self.assertEqual(staged['profiles'][beta.ALIAS]['default_max_tokens'], 16384)
            self.assertEqual(json.loads((stage / 'opencode.json').read_text())['providers']['local-qwen']['models']['qwen-local']['limit']['output'], 16384)
            self.assertNotIn('synthetic-secret', json.dumps(result))
            self.assertEqual((stage / 'opencode.json').stat().st_mode & 0o777, 0o600)
            self.assertEqual(stage.stat().st_mode & 0o777, 0o700)
            self.assertTrue((stage / 'runtime-source/controller.py').is_file())
            compile((stage / 'rollback/rollback.py').read_text(), 'rollback.py', 'exec')
            self.assertIn('BINDING = json.loads(', (stage / 'rollback/rollback.py').read_text())  # guarded
            self.assertEqual(result['installed_sha256']['opencode'], hashlib.sha256((stage / 'opencode.json').read_bytes()).hexdigest())
            self.assertEqual(result['preimage_sha256']['unit'], hashlib.sha256((stage / 'rollback/unit.original').read_bytes()).hexdigest())


if __name__ == '__main__':
    unittest.main()
