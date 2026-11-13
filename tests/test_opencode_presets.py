"""MoE-only OpenCode deep/fast presets: limits, keep, small-model isolation, Flash untouched."""
import json
import unittest
from unittest.mock import patch

from snapdragon_ai_lab.runtime import launch_local_plan as launcher

MOE_IDS = ('qwen3.6-35b-a3b-local', 'ornith-35b-a3b-local')
SMALL_IDS = ('qwen-local', 'qwen3.8-27b-local', 'qwen3.8-flash-next-local')


def overlay(model, environment=None, mode=None):
    result = launcher.environment_for_classic_model(dict(environment or {}), model, mode)
    return json.loads(result['OPENCODE_CONFIG_CONTENT'])


def opencode_trigger(limit):
    # OpenCode 2.2.7 SessionCompaction with no compaction.buffer:
    # o=min(input,context); trigger=o-max(floor(o*0.15), 16000 if o>=32000 else 0).
    o = min(limit['input'], limit['context'])
    return o - max(int(o * 0.15), 16000 if o >= 32000 else 0)


class MoePresets(unittest.TestCase):
    def test_explicit_deep_sets_limits_keep_and_trigger(self):
        for model in launcher.MOE_PRESET_MODELS:
            for environment in ({launcher.OPENCODE_MODE_ENV: 'deep'},):
                with self.subTest(model=model, environment=environment):
                    value = overlay(model, environment)
                    models = value['providers']['local-qwen']['models']
                    for model_id in MOE_IDS:
                        limit = models[model_id]['limit']
                        self.assertEqual(limit, {'context': 131072, 'input': 120000, 'output': 16384})
                        self.assertEqual(opencode_trigger(limit), 102000)
                        self.assertNotIn('disabled', models[model_id])
                    self.assertEqual(value['compaction'], {'auto': True, 'keep': {'tokens': 8000}})

    def test_fast_is_default_and_sets_smaller_input_and_trigger(self):
        # O-105: unset or empty LOCAL_AI_OPENCODE_MODE means fast; deep is explicit.
        self.assertEqual(launcher.DEFAULT_OPENCODE_MODE, 'fast')
        for model in launcher.MOE_PRESET_MODELS:
            for environment in ({}, {launcher.OPENCODE_MODE_ENV: ''}, {launcher.OPENCODE_MODE_ENV: 'fast'}):
                with self.subTest(model=model, environment=environment):
                    value = overlay(model, environment)
                    for model_id in MOE_IDS:
                        limit = value['providers']['local-qwen']['models'][model_id]['limit']
                        self.assertEqual(limit, {'context': 131072, 'input': 65536, 'output': 16384})
                        self.assertEqual(opencode_trigger(limit), 49536)
                    self.assertEqual(value['compaction']['keep'], {'tokens': 8000})

    def test_explicit_mode_argument_overrides_environment(self):
        value = overlay('moe', {launcher.OPENCODE_MODE_ENV: 'deep'}, mode='fast')
        self.assertEqual(value['providers']['local-qwen']['models'][MOE_IDS[0]]['limit']['input'], 65536)

    def test_moe_launch_disables_every_small_context_model(self):
        # Global keep=8000 must never reach a 16K model in the same instance.
        self.assertEqual(launcher.SMALL_CONTEXT_MODEL_IDS, SMALL_IDS)
        for model in launcher.MOE_PRESET_MODELS:
            for mode in ('deep', 'fast'):
                with self.subTest(model=model, mode=mode):
                    models = overlay(model, mode=mode)['providers']['local-qwen']['models']
                    self.assertEqual(set(models), set(MOE_IDS) | set(SMALL_IDS))
                    for model_id in SMALL_IDS:
                        self.assertEqual(models[model_id], {'disabled': True})

    def test_preset_uses_only_opencode_227_native_keys(self):
        # Legacy provider.whitelist/blacklist are dropped by 2.2.7; compaction is global.
        value = overlay('ornith')
        self.assertNotIn('provider', value)
        provider = value['providers']['local-qwen']
        self.assertTrue(set(provider) <= {'body', 'models', 'settings'})
        self.assertEqual(set(provider['settings']), {'headerTimeout', 'chunkTimeout'})
        self.assertFalse({'whitelist', 'blacklist'} & set(provider))
        self.assertEqual(set(value['compaction']), {'auto', 'keep'})
        for entry in provider['models'].values():
            self.assertTrue(set(entry) <= {'limit', 'disabled'})

    def test_preset_overrides_conflicting_incoming_overlay(self):
        incoming = {'compaction': {'keep': {'tokens': 30000}, 'buffer': '50%', 'auto': False},
                    'providers': {'local-qwen': {'models': {
                        'qwen-local': {'disabled': False, 'name': 'keep-name'},
                        MOE_IDS[0]: {'disabled': True, 'name': 'MoE'}}}},
                    'unrelated': 1}
        value = overlay('moe', {'OPENCODE_CONFIG_CONTENT': json.dumps(incoming)})
        self.assertEqual(value['compaction'], {'auto': True, 'keep': {'tokens': 8000}})
        models = value['providers']['local-qwen']['models']
        self.assertEqual(models['qwen-local'], {'disabled': True, 'name': 'keep-name'})
        self.assertNotIn('disabled', models[MOE_IDS[0]])
        self.assertEqual(models[MOE_IDS[0]]['name'], 'MoE')
        self.assertEqual(value['unrelated'], 1)

    def test_flash_launches_are_unchanged_in_every_mode(self):
        for mode in (None, 'deep', 'fast'):
            environment = {} if mode is None else {launcher.OPENCODE_MODE_ENV: mode}
            with self.subTest(mode=mode):
                flash = overlay('flash', environment)
                self.assertEqual(flash, {'model': launcher.CLASSIC_MODELS['flash'],
                                         'providers': {'local-qwen': {'body': {
                                             'temperature': 0, 'cache_prompt': True,
                                             'chat_template_kwargs': {'enable_thinking': False}}}}})
        planning = json.loads(launcher.environment_for_named_agent({})['OPENCODE_CONFIG_CONTENT'])
        self.assertNotIn('compaction', planning)
        self.assertNotIn('providers', planning)

    def test_invalid_mode_fails_loudly_and_never_launches(self):
        for bad in ('Deep', 'medium', ' fast', 'fast\n', '0'):
            with self.subTest(bad=bad):
                with self.assertRaisesRegex(ValueError, 'unsupported_opencode_mode'):
                    launcher.environment_for_classic_model({launcher.OPENCODE_MODE_ENV: bad}, 'moe')
                with self.assertRaisesRegex(ValueError, 'unsupported_opencode_mode'):
                    launcher.environment_for_classic_model({}, 'ornith', bad)
                with self.assertRaisesRegex(ValueError, 'unsupported_opencode_mode'):
                    launcher.apply_moe_preset({}, bad)
                with patch.dict(launcher.os.environ, {launcher.OPENCODE_MODE_ENV: bad}, clear=True), \
                        patch.object(launcher.os, 'execvpe', side_effect=AssertionError('must not launch')):
                    with self.assertRaises(ValueError):
                        launcher.main(['--model', 'moe', '/synthetic/project'])

    def test_main_passes_fast_preset_to_opencode(self):
        with patch.dict(launcher.os.environ, {launcher.OPENCODE_MODE_ENV: 'fast'}, clear=True), \
                patch.object(launcher.os, 'execvpe') as execute:
            launcher.main(['--model', 'ornith', '/synthetic/project'])
        command, arguments, environment = execute.call_args.args
        self.assertEqual((command, arguments), ('opencode', ['opencode', '--standalone', '/synthetic/project']))
        value = json.loads(environment['OPENCODE_CONFIG_CONTENT'])
        self.assertEqual(value['providers']['local-qwen']['models'][MOE_IDS[1]]['limit']['input'], 65536)
        self.assertEqual(value['model'], launcher.CLASSIC_MODELS['ornith'])

    def test_malformed_incoming_structures_are_rejected(self):
        for incoming in ({'compaction': []}, {'providers': {'local-qwen': {'models': []}}},
                         {'providers': {'local-qwen': {'models': {'qwen-local': 'x'}}}},
                         {'providers': {'local-qwen': {'models': {MOE_IDS[0]: 1}}}}):
            with self.subTest(incoming=incoming), self.assertRaises(ValueError):
                launcher.environment_for_classic_model({'OPENCODE_CONFIG_CONTENT': json.dumps(incoming)}, 'moe')


class RetiredDense(unittest.TestCase):
    """Audit S1: the dense 27B is retired; its launch key explains instead of launching."""

    def test_dense_launch_key_is_refused_with_the_retired_reason(self):
        self.assertNotIn('dense', launcher.CLASSIC_MODELS)
        with self.assertRaises(ValueError):
            overlay('dense')
        with patch.object(launcher.os, 'execvpe', side_effect=AssertionError('must not launch')), \
                self.assertRaisesRegex(ValueError, 'retired'):
            launcher.main(['--model', 'dense', '/synthetic/project'])

    def test_retired_id_stays_disabled_under_a_moe_launch(self):
        self.assertIn('qwen3.8-27b-local', launcher.SMALL_CONTEXT_MODEL_IDS)


if __name__ == '__main__':
    unittest.main()
