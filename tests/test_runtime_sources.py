"""Source package checks only; never import/run image UI or load a model."""
import importlib
from pathlib import Path
import unittest

# The home directory of whoever runs the tests must never leak into generated files.
DEVELOPER_HOME=__import__('pwd').getpwuid(__import__('os').getuid()).pw_dir

class RuntimeSources(unittest.TestCase):
    def test_controller_import_has_no_manager_or_service_side_effect(self):
        module=importlib.import_module('snapdragon_ai_lab.runtime.controller')
        self.assertTrue(callable(module.main))
        self.assertFalse(any(isinstance(value,module.Manager) for value in vars(module).values()))
    def test_text_resources_are_portable_and_syntax_valid(self):
        runtime=Path(__file__).resolve().parents[1]/'src/snapdragon_ai_lab/runtime'
        for name in ('controller.py','classic_beta.py','prefix_plan.py','prefix_store.py','bigmoe_runtime.py','bigmoe_adapter.py','exec_no_thp.py','slim_request.py','launch_local_plan.py','local-ai','studio'):
            text=(runtime/name).read_text()
            self.assertNotIn(DEVELOPER_HOME,text)
            compile(text,name,'exec')

if __name__=='__main__':unittest.main()
