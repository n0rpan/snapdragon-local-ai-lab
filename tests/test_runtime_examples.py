import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from snapdragon_ai_lab.runtime import controller,prefix_plan

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('render_runtime_examples',ROOT/'tools/render_runtime_examples.py')
renderer=importlib.util.module_from_spec(spec);spec.loader.exec_module(renderer)

# The home directory of whoever runs the tests must never leak into generated files.
DEVELOPER_HOME=__import__('pwd').getpwuid(__import__('os').getuid()).pw_dir

class Examples(unittest.TestCase):
    def test_staging_and_controller_alias_prefix_contract_without_manager(self):
        with tempfile.TemporaryDirectory() as temporary:
            output=Path(temporary)/'staged';home=Path('/synthetic/home')
            result=renderer.render(output,home);self.assertFalse(result['installed'])
            config=json.loads((output/'llm-runtime.json').read_text())
            manager=object.__new__(controller.Manager)  # no constructor, thread, service or model
            with patch.object(controller,'CONFIG',output/'llm-runtime.json'):
                selected=manager.config('qwen-local')
                self.assertEqual(selected['model_alias'],'ornith-35b-a3b-local')
                for alias in config['profiles']:
                    value=manager.config(alias);self.assertFalse(value['ready'])
                    plan=prefix_plan.make_plan({'messages':[{'role':'system','content':'fixed synthetic instruction'},{'role':'user','content':'PRIVATE_SENTINEL'}],'tools':[]},value,min_characters=0)
                    self.assertIsNotNone(plan);self.assertNotIn('PRIVATE_SENTINEL',str(plan.primer))
            with self.assertRaises(ValueError):renderer.render(output,home)

    def test_manual_recipe_installs_every_imported_chat_module_and_launcher(self):
        recipe=(ROOT/'docs/lab-notes/RUNTIME-SETUP.md').read_text()
        for name in ('controller.py','prefix_plan.py','prefix_store.py','bigmoe_adapter.py','bigmoe_runtime.py','exec_no_thp.py','slim_request.py','launch_local_plan.py'):
            self.assertIn(name,recipe)

    def test_json_opencode_v2_and_disabled_image_examples(self):
        root=ROOT/'examples/runtime'
        for path in root.rglob('*.json'):
            value=json.loads(path.read_text());self.assertNotIn(DEVELOPER_HOME,path.read_text())
        config=json.loads((root/'opencode-v2.json').read_text())
        provider=config['providers']['local-qwen'];self.assertIn('settings',provider)
        self.assertEqual(config['model'],'local-qwen/qwen3.6-35b-a3b-local')
        self.assertEqual(provider['models']['qwen-local']['limit']['output'],2048)
        self.assertTrue(all(provider['models'][name]['limit']['output']==4096 for name in ('qwen3.6-35b-a3b-local','qwen3.8-27b-local')))
        fragment=json.loads((ROOT/'examples/flash-next-opencode.fragment.json').read_text())
        self.assertNotIn('model',fragment)
        self.assertEqual(fragment['agents']['local-plan']['model'],'local-qwen/qwen3.8-flash-next-local')
        self.assertEqual(fragment['providers']['local-qwen']['models']['qwen3.8-flash-next-local']['limit']['output'],2048)
        self.assertEqual(provider['settings']['baseURL'],'http://127.0.0.1:8092/v1')
        self.assertEqual(config['agents']['title'],{'disabled':True})
        self.assertFalse(json.loads((root/'images/cpu-profile.json').read_text())['validated'])
        self.assertFalse(json.loads((root/'images/gpu-profile.json').read_text())['enabled'])
        presets=json.loads((root/'images/presets.json').read_text())
        self.assertEqual([presets[key]['model'] for key in ('draft','final')],['q8','q8'])

if __name__=='__main__':unittest.main()
