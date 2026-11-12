"""M4: the Rust engines are the default on a clean install; the classic stack is an explicit opt-in,
and a missing Rust binary fails loudly instead of silently running llama.cpp/BigMoe."""
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from snapdragon_ai_lab.runtime import controller, oxe_runtime, profile_policy

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / 'examples/runtime/llm-runtime.json'


def example_profile(home='/srv/example'):
    value = json.loads(EXAMPLE.read_text().replace('@HOME@', home))
    base = {k: v for k, v in value.items() if k != 'profiles'}
    return {**base, **value['profiles']['qwen3.6-35b-a3b-local']}


class SwitchTests(unittest.TestCase):
    def test_unset_means_on_and_only_explicit_off_values_switch_off(self):
        for value, expected in ((None, True), ('1', True), ('', True), ('yes', True),
                                ('0', False), ('off', False), ('FALSE', False), (' no ', False)):
            env = {} if value is None else {'X': value}
            with self.subTest(value=value):
                self.assertEqual(profile_policy.switch_on('X', env), expected)
                self.assertEqual(oxe_runtime.switch_on('X', env), expected)


class CleanInstallTests(unittest.TestCase):
    def test_example_moe_profile_resolves_to_rust_without_flags_or_memlat(self):
        with patch.dict(os.environ, {}, clear=True):
            result = profile_policy.resolve(example_profile(), available=False)
        self.assertEqual(result['backend_kind'], 'oxe')
        self.assertTrue(result['server_binary'].endswith('/oxe-server'))
        self.assertEqual(result['extra_args'], [])
        self.assertEqual(controller.engine_summary(result)['engine'], 'rust')

    def test_explicit_off_selects_the_classic_fallback_and_says_so(self):
        with patch.dict(os.environ, {profile_policy.OXE_ENABLE_ENV: '0'}, clear=True):
            result = profile_policy.resolve(example_profile(), available=False)
            summary = controller.engine_summary(result)
        self.assertNotEqual(result.get('backend_kind'), 'oxe')
        self.assertEqual(summary['engine'], 'llama.cpp')
        self.assertIn('LOCAL_AI_ENABLE_OXE=0', summary['engine_note'])

    def test_legacy_memlat_location_off_memlat_is_reported_not_silent(self):
        profile = example_profile()
        profile['memlat_overrides'] = {'oxe': profile.pop('rust_engine')['oxe']}
        with patch.dict(os.environ, {}, clear=True):
            result = profile_policy.resolve(profile, available=False)
            summary = controller.engine_summary(result)
        self.assertEqual(summary['engine'], 'llama.cpp')
        self.assertIn('MEMLAT', summary['engine_note'])
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(profile_policy.resolve(profile, available=True)['backend_kind'], 'oxe')

    def test_same_engine_in_both_locations_is_refused(self):
        profile = example_profile()
        profile['memlat_overrides'] = {'oxe': profile['rust_engine']['oxe']}
        with self.assertRaisesRegex(ValueError, 'rust_engine_also_in_memlat_overrides'):
            profile_policy.resolve(profile, available=True)

    def test_missing_binary_message_names_the_path_and_the_fallback_switch(self):
        with patch.dict(os.environ, {}, clear=True):
            result = profile_policy.resolve(example_profile('/nonexistent-home'), available=False)
            result['ready'] = True
            reason = oxe_runtime.ineligible_reason(result)
        self.assertIn('/nonexistent-home/.local/share/ai-stack/releases/oxe/oxe-server', reason)
        self.assertIn('LOCAL_AI_ENABLE_OXE=0', reason)
        with patch.dict(os.environ, {oxe_runtime.ENABLE_ENV: '0'}):
            self.assertIn('switched off', oxe_runtime.ineligible_reason(result))

    def test_start_with_a_missing_binary_fails_loudly(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as tmp, patch.dict(os.environ, {}, clear=True):
            root = Path(tmp)
            model = root / 'model.gguf'
            model.write_bytes(b'GGUF')
            config = profile_policy.resolve(example_profile(tmp), available=False)
            config.update(ready=True, model_path=str(model), expected_bytes=4)
            manager = controller.Manager()
            manager.stop_event.set()
            manager.config = Mock(return_value=config)
            with patch.object(controller, 'LEASE', root / 'lease'), \
                    patch.object(controller, 'require_backend_port'), \
                    patch.object(controller, 'available_memory_bytes', return_value=64 * 1024**3), \
                    patch.object(controller.subprocess, 'Popen') as launch:
                with self.assertRaisesRegex(controller.Unavailable, 'Rust engine binary is missing.*LOCAL_AI_ENABLE_OXE=0'):
                    manager.begin(allow_start=True, model_alias='qwen3.6-35b-a3b-local')
                launch.assert_not_called()
            self.assertIsNone(manager.lease)
            self.assertIn('binary is missing', manager.status()['last_error'])

    def test_startup_log_names_each_profile_engine(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as tmp, patch.dict(os.environ, {}, clear=True):
            path = Path(tmp) / 'runtime.json'
            path.write_text(EXAMPLE.read_text().replace('@HOME@', tmp))
            out = io.StringIO()
            manager = controller.Manager()
            manager.stop_event.set()
            with patch.object(controller, 'CONFIG', path):
                controller.log_effective_engines(manager, stream=out)
        lines = out.getvalue().splitlines()
        self.assertTrue(any(line.startswith('local-ai: qwen3.6-35b-a3b-local: engine rust') for line in lines), lines)
        self.assertTrue(any(line.startswith('local-ai: ornith-35b-a3b-local: engine rust') for line in lines), lines)
        self.assertFalse(any('qwen3.8-27b-local' in line for line in lines), 'the retired 27B profile is gone')


    def test_controller_paths_default_to_xdg_outside_a_test_runner(self):
        # A child without unittest loaded and a throwaway HOME: prints the defaults only.
        with tempfile.TemporaryDirectory() as home:
            env = {k: v for k, v in os.environ.items()
                   if k not in ('LOCAL_LLM_CONFIG', 'LOCAL_AI_LEASE', 'XDG_CONFIG_HOME', 'XDG_STATE_HOME')}
            env.update(HOME=home, PYTHONPATH=str(ROOT / 'src'))
            code = 'from snapdragon_ai_lab.runtime import controller as c; print(c.CONFIG); print(c.LEASE)'
            out = subprocess.run([sys.executable, '-c', code], env=env, capture_output=True, text=True,
                                 check=True, timeout=60).stdout.split()
        self.assertEqual(out, [f'{home}/.config/local-ai/llm/runtime.json', f'{home}/.local/state/local-ai/inference.lock'])


if __name__ == '__main__':
    unittest.main()
