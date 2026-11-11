"""Mocked transport/oracle tests only; no native or model-generated execution."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock,patch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('classic_screen_driver',HERE/'screen.py')
screen=importlib.util.module_from_spec(spec)
spec.loader.exec_module(screen)


class ClassicScreen(unittest.TestCase):
    def answer(self,content='synthetic code'):
        return {'choices':[{'finish_reason':'stop','message':{'content':content}}]}

    def test_two_original_code_cases_have_at_most_one_bounded_repair(self):
        selected=[c for c in screen.checks.CODE_CASES if c[0] in ('merge_intervals','parse_port')]
        self.assertEqual([c[0] for c in selected],['merge_intervals','parse_port'])
        with tempfile.TemporaryDirectory() as tmp:
            api=Mock(return_value=self.answer())
            with patch.object(screen.checks,'code_check',side_effect=[{'passed':False,'reason':'Independent checks failed'},{'passed':True}]) as oracle:
                row=screen.code_case(*selected[0],Path(tmp),api)
            self.assertTrue(row['passed'])
            self.assertTrue(row['repair_attempted'])
            self.assertEqual(api.call_count,2)
            self.assertEqual(oracle.call_count,2)
            self.assertEqual(api.call_args_list[0].args[1]['messages'][0]['content'],selected[0][1])
            self.assertEqual(oracle.call_args.args[2],selected[0][2])
            with patch.object(screen.checks,'code_check',return_value={'passed':False}) as oracle:
                failed=screen.code_case(*selected[1],Path(tmp),api)
                self.assertFalse(failed['passed'])
                self.assertEqual(oracle.call_count,2)

    def test_original_checker_rejects_top_level_execution_without_starting_process(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(screen.checks.subprocess,'Popen') as child:
            result=screen.checks.code_check('print("unsafe")','parse_port','',Path(tmp))
            self.assertFalse(result['passed'])
            child.assert_not_called()

    def test_five_original_functional_cases_and_final_json_schema(self):
        replies=[
            self.answer('READY'),self.answer('{"Kestrel":12,"Mora":7}'),
            self.answer('{"Kestrel":9,"Mora":10}'),
            {'choices':[{'finish_reason':'tool_calls','message':{'tool_calls':[{'id':'own-call',
             'function':{'name':'lookup_stock','arguments':'{"item_id":"synthetic_adapter"}'}}]}}]},
            self.answer('{"item_id":"synthetic_adapter","stock":17}')]
        api=Mock(side_effect=replies)
        report={}
        with tempfile.TemporaryDirectory() as tmp:
            screen.functional(report,api,Path(tmp),'screen')
        self.assertEqual([r['id'] for r in report['functional']],
            ['ready','chat-record','chat-context-json','structured-tool-call','tool-result-final'])
        self.assertTrue(all(r['passed'] for r in report['functional']))
        final=api.call_args.args[1]
        self.assertEqual(final['response_format']['json_schema']['schema'],
            {'type':'object','properties':{'item_id':{'type':'string'},'stock':{'type':'integer'}},
             'required':['item_id','stock'],'additionalProperties':False})
        self.assertEqual(final['messages'][-1]['content'],'{"item_id": "synthetic_adapter", "stock": 17}')
        self.assertFalse(final['cache_prompt'])

    def test_profile_threads_context_and_mask_args_reach_actual_server_command(self):
        profile=dict(server_binary='/synthetic/server',model_path='/synthetic/model',threads=8,
            context_size=16384,extra_args=['--cpu-mask','0xff','--cpu-strict','1'],prefix_cache={'enabled':False})
        command=screen.server_args(profile,Path('/synthetic/output'),12345)
        self.assertEqual(command[command.index('--threads')+1],'8')
        self.assertEqual(command[command.index('--ctx-size')+1],'16384')
        self.assertEqual(command[-4:],profile['extra_args'])
        self.assertEqual(command[command.index('--host')+1],'127.0.0.1')

    def test_cancellation_drains_only_dedicated_driver_children(self):
        with patch.object(Path,'read_text',return_value='1234 5678'), \
             patch.object(screen.os,'kill') as kill, \
             patch.object(screen.os,'waitpid',side_effect=[(1234,0),(5678,0),ChildProcessError()]):
            self.assertTrue(screen.reap_remaining())
            self.assertEqual([call.args[0] for call in kill.call_args_list],[1234,5678])

    def test_screen_source_freeze_detects_drift(self):
        self.assertEqual(len(screen.source_gate()),64)
        with patch.object(screen.common,'sha',return_value='f'*64):
            with self.assertRaisesRegex(ValueError,'screen_source_changed'):screen.source_gate()


if __name__=='__main__':unittest.main()
