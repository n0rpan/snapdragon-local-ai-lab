import importlib.machinery,importlib.util,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch,Mock
from snapdragon_ai_lab.runtime import controller,classic_beta
RUNTIME=Path(controller.__file__).parent
loader=importlib.machinery.SourceFileLoader('retired_model_test_cli',str(RUNTIME/'local-ai'));spec=importlib.util.spec_from_loader(loader.name,loader);cli=importlib.util.module_from_spec(spec);loader.exec_module(cli)
class BetaSelector(unittest.TestCase):
 def test_loaded_beta_explicit_request_respects_revoked_fresh_proof(self):
  with patch.object(controller.threading.Thread,'start'):
   manager=controller.Manager()
   loaded={'model_alias':classic_beta.ALIAS,'ready':True}
   manager.loaded_config=loaded
   manager.process=Mock()
   manager.process.poll.return_value=None
   manager.config=Mock(return_value={**loaded,'ready':False})
   with patch.object(manager,'eligible',return_value=False),patch.object(controller.subprocess,'Popen') as load:
    with self.assertRaises(controller.Unavailable):
     manager.begin(allow_start=True,inference=True,model_alias=classic_beta.ALIAS)
    load.assert_not_called()
    self.assertIs(manager.loaded_config,loaded)
    self.assertEqual(manager.active,0)
 def test_unqualified_beta_cannot_change_selection_or_load(self):
  with tempfile.TemporaryDirectory() as directory:
   config=Path(directory)/'runtime.json';original={'model_alias':'qwen-base','selected_model':'qwen-base','ready':True,'profiles':{classic_beta.ALIAS:{'ready':False}}};config.write_text(json.dumps(original))
   manager=controller.Manager()
   try:
    with patch.object(controller,'CONFIG',config),patch.object(classic_beta,'eligible',return_value=False),patch.object(controller.subprocess,'Popen') as load:
     with self.assertRaises(controller.Unavailable):manager.select(classic_beta.ALIAS)
     self.assertEqual(json.loads(config.read_text()),original);load.assert_not_called()
   finally:manager.stop_event.set();manager.thread.join(2)
 def test_retired_alias_cannot_fall_back_to_base_on_legacy_runtime(self):
  with tempfile.TemporaryDirectory() as directory:
   config=Path(directory)/'runtime.json';config.write_text(json.dumps({'model_alias':'qwen-base','ready':True}))
   manager=object.__new__(controller.Manager)
   with patch.object(controller,'CONFIG',config):
    with self.assertRaises(controller.Unavailable):manager.config('arete-27b-local')
 def test_retired_ready_profile_is_hidden_and_cannot_select_or_load(self):
  alias='arete-27b-local'
  with tempfile.TemporaryDirectory() as directory,patch.object(controller.threading.Thread,'start'):
   config=Path(directory)/'runtime.json'
   original={'model_alias':'qwen-base','selected_model':'qwen-base','ready':True,'profiles':{alias:{'ready':True}}}
   config.write_text(json.dumps(original));manager=controller.Manager()
   with patch.object(controller,'CONFIG',config),patch.object(controller.subprocess,'Popen') as load:
    self.assertNotIn(alias,{x['id'] for x in manager.models()})
    with self.assertRaises(controller.Unavailable):manager.select(alias)
    with self.assertRaises(controller.Unavailable):manager.begin(allow_start=True,inference=True,model_alias=alias)
    self.assertEqual(json.loads(config.read_text()),original);load.assert_not_called()
if __name__=='__main__':unittest.main()
