"""Selected classic model overlays preserve the ordinary agent and permissions."""
import importlib.machinery
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from snapdragon_ai_lab.runtime import launch_local_plan as launcher

RUNTIME=Path(launcher.__file__).parent
loader=importlib.machinery.SourceFileLoader('reviewed_local_ai',str(RUNTIME/'local-ai'))
spec=importlib.util.spec_from_loader(loader.name,loader)
cli=importlib.util.module_from_spec(spec);loader.exec_module(cli)


class SelectedModelLaunch(unittest.TestCase):
    def test_classic_overlay_changes_only_model_and_preserves_environment(self):
        original={'model':'local-qwen/qwen3.6-35b-a3b-local','default_agent':'build',
                  'agents':{'build':{'system':'PRESERVE'}},'policies':[{'action':'tool.use','resource':'read:*','effect':'allow'}],
                  'unrelated':{'preserve':True}}
        for selected,expected in launcher.CLASSIC_MODELS.items():
            environment={'KEEP_ENV':'preserved','OPENCODE_CONFIG_CONTENT':json.dumps(original)}
            with self.subTest(selected=selected),patch.dict(launcher.os.environ,environment,clear=True),patch.object(launcher.os,'execvpe') as execute:
                launcher.main(['--model',selected,'/synthetic/project'])
            command,arguments,updated=execute.call_args.args
            self.assertEqual(command,'opencode')
            self.assertEqual(arguments,['opencode','--standalone','/synthetic/project'])
            self.assertEqual(updated['KEEP_ENV'],'preserved')
            expected_overlay={**original,'model':expected}
            if selected in ('ornith','flash'):expected_overlay['providers']={'local-qwen':{'body':{'temperature':0,'cache_prompt':True,'chat_template_kwargs':{'enable_thinking':False}}}}
            if selected in launcher.MOE_PRESET_MODELS:launcher.apply_moe_preset(expected_overlay,launcher.DEFAULT_OPENCODE_MODE)
            self.assertEqual(json.loads(updated['OPENCODE_CONFIG_CONTENT']),expected_overlay)
            self.assertEqual(json.loads(environment['OPENCODE_CONFIG_CONTENT']),original)

    def test_panel_code_dispatches_classic_enum_to_existing_helper(self):
        for selected in ('moe','ornith'):  # O95-DENSE: the 27B CLI key is gone
            with self.subTest(selected=selected),patch.object(cli,'ensure_service',return_value=True),patch.object(cli,'status',return_value={'selected_model':selected}),patch.object(cli.subprocess,'Popen') as launch:
                self.assertEqual(cli.action('code'),0)
            self.assertEqual(launch.call_args.args[0],['omarchy','launch','terminal',cli.sys.executable,str(RUNTIME/'launch_local_plan.py'),'--model',selected,str(cli.CODE_DIRECTORY)])

    def test_unrecognized_model_or_overlay_cannot_change_launch(self):
        # Flash is a general-purpose build model now (no longer planning-only).
        for arguments in (['--model','arete','/synthetic/project'],['--model','planner','/synthetic/project'],['--model','moe','relative']):
            with self.assertRaises(ValueError),patch.object(launcher.os,'execvpe',side_effect=AssertionError('must not launch')):launcher.main(arguments)
        with self.assertRaises(ValueError):launcher.environment_for_classic_model({'OPENCODE_CONFIG_CONTENT':'[]'},'dense')

    def test_missing_or_invalid_selection_never_falls_back_to_client_default(self):
        for current in ({}, {'selected_model': 'ornith-oxe'}, {'selected_model': 'flash', 'flash_available': False}):
            with self.subTest(current=current), patch.object(cli, 'ensure_service', return_value=True), \
                    patch.object(cli, 'status', return_value=current), patch.object(cli, 'notify') as notice, \
                    patch.object(cli.subprocess, 'Popen') as spawn:
                self.assertEqual(cli.action('code'), 1)
                spawn.assert_not_called()
                notice.assert_called_once()

    def test_sleeping_controller_starts_before_qualified_choice_query(self):
        for name,gate in (('ornith','qualified_ornith_offered'),('flash','qualified_flash_offered'),('plan','qualified_flash_offered')):
            order=[]
            with patch.object(cli,'ensure_service',side_effect=lambda *_:order.append('start') or True),patch.object(cli,gate,side_effect=lambda:order.append('catalog') or False),patch.object(cli,'notify'),patch.object(cli.subprocess,'Popen') as spawn:
                self.assertEqual(cli.action(name),1)
            self.assertEqual(order,['start','catalog']);spawn.assert_not_called()

    def test_beta_launch_preserves_explicit_request_choices_and_permissions(self):
        original={'agents':{'build':{'system':'PRESERVE'}},'policies':[{'action':'tool.use','resource':'*','effect':'allow'}],
                  'providers':{'local-qwen':{'settings':{'baseURL':'http://127.0.0.1:8092/v1'},'body':{'temperature':.7,'cache_prompt':False,'chat_template_kwargs':{'enable_thinking':True}}}}}
        result=launcher.environment_for_classic_model({'OPENCODE_CONFIG_CONTENT':json.dumps(original)},'ornith')
        expected=launcher.apply_moe_preset(json.loads(json.dumps({**original,'model':launcher.CLASSIC_MODELS['ornith']})),launcher.DEFAULT_OPENCODE_MODE)
        self.assertEqual(json.loads(result['OPENCODE_CONFIG_CONTENT']),expected)
        self.assertEqual(expected['providers']['local-qwen']['body'],original['providers']['local-qwen']['body'])


if __name__=='__main__':unittest.main()
