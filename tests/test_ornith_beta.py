"""Qualification, identity drift and opt-in staging; no model or live config access."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from snapdragon_ai_lab.runtime import classic_beta as beta,controller

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('stage_ornith_beta',ROOT/'tools/stage_ornith_beta.py')
stager=importlib.util.module_from_spec(spec);spec.loader.exec_module(stager)


class OrnithBeta(unittest.TestCase):
    def evidence(self):
        profile={**copy.deepcopy(beta.POLICY),'ready':False,'model_alias':'ornith-compact-candidate','display_name':'Synthetic candidate','sha256':beta.MODEL_SHA,'expected_bytes':beta.MODEL_BYTES,'server_sha256':beta.SERVER_SHA,'model_path':'/synthetic/model.gguf','server_binary':'/synthetic/llama-server'}
        identity=dict(bytes=beta.MODEL_BYTES,device=55,inode=123,mtime_ns=456)
        task=dict(passed=True,expected_test_cases=13,independent_test_cases_run=13,complete=True,protected_unchanged=True,changed_modules=['parse_port.py'],unexpected_files=[],opencode_exit=0,test_exit=0,readers_joined=True,completed_tools=6,test_commands_completed=1)
        agent=dict(passed=True,stage='normal-agent',model_role='ornith',profile=profile,source_manifest_sha256=beta.AGENT_SOURCE_SHA,candidate_screen_quality_passed=True,matched_screen_sha256={'ornith':'b'*64},inventory_receipt_sha256='c'*64,guard_failures=[],identity=dict(model_bytes=identity['bytes'],model_device=55,model_inode=123,model_mtime_ns=456),measurement=dict(passed=True,server_reaped=True,server_reader_joined=True,agent=task),**{k:True for k in ('owned_native_reaped','all_children_reaped','reader_joined','ram_recovered','lease_free_after','post_source_pins_passed')})
        screen=dict(passed=True,stage='screen',model_role='ornith',profile=copy.deepcopy(profile),source_manifest_sha256=beta.SCREEN_SOURCE_SHA,workload_version='ornith-matched-classic-v3-tool-json-schema',inventory_receipt_sha256='c'*64,cases=[dict(case=k,passed=True) for k in ('merge_intervals','parse_port')],functional=[dict(id=k,passed=True) for k in ('ready','chat-record','chat-context-json','structured-tool-call','tool-result-final')],**{k:True for k in ('owned_native_reaped','guard_joined','ram_recovered','lease_free_after')})
        screen.update(guard_failures=[],native_returncode=0,identity=copy.deepcopy(agent['identity']))
        inventory=dict(passed=True,inventory=dict(tensor_inventory_complete=True,tensor_payload_read=False,header_bytes_read=1024,file_identity=identity))
        return agent,screen,inventory

    def qualify(self,rows=None,review='a'*64):
        return beta.qualify(*(self.evidence() if rows is None else rows),normal_sha='a'*64,screen_sha='b'*64,inventory_sha='c'*64,reviewed_normal_sha=review)

    def test_actual_contract_requires_all_quality_and_root_review(self):
        result=self.qualify();self.assertEqual(result['model_alias'],beta.ALIAS);self.assertTrue(result['ready']);self.assertEqual(result['prefix_cache'],{'enabled':False})
        for mutate in (lambda r:r[0]['measurement']['agent'].update(independent_test_cases_run=12),lambda r:r[0].update(ram_recovered=False),lambda r:r[0].update(all_children_reaped=False),lambda r:r[1]['functional'][-1].update(passed=False),lambda r:r[0]['profile'].update(threads=6),lambda r:r[2]['inventory']['file_identity'].update(device=57)):
            rows=self.evidence();mutate(rows)
            with self.assertRaises(ValueError):self.qualify(rows)
        with self.assertRaises(ValueError):self.qualify(review='d'*64)

    def test_identity_drift_or_untested_prefix_flags_hides_beta(self):
        profile=self.qualify();expected=profile['validation']['model_identity']
        with patch.object(beta,'model_identity',return_value=expected):self.assertTrue(beta.eligible(profile))
        for key in expected:
            changed={**expected,key:expected[key]+1}
            with patch.object(beta,'model_identity',return_value=changed):self.assertFalse(beta.eligible(profile),key)
        for mutate in (lambda p:p.update(ready=False),lambda p:p.update(prefix_cache={'enabled':True}),lambda p:p['extra_args'].extend(['--slots']),lambda p:p['validation'].update(root_reviewed_normal13_sha256='d'*64),lambda p:p.update(backend_kind='bigmoe-cpu-stream')):
            row=copy.deepcopy(profile);mutate(row)
            with patch.object(beta,'model_identity',return_value=expected):self.assertFalse(beta.eligible(row))

    def test_legacy_ornith_proof_cannot_authorize_the_arete_alias(self):
        profile=self.qualify()
        expected=profile['validation']['model_identity']
        profile['model_alias']='arete-27b-local'
        with patch.object(beta,'model_identity',return_value=expected):
            self.assertFalse(beta.eligible(profile))

    def test_profile_addition_preserves_every_existing_default_and_qwen_field(self):
        original={'ready':True,'selected_model':'qwen3.6-35b-a3b-local','profiles':{'qwen3.6-35b-a3b-local':{'ready':True,'prefix_cache':{'enabled':True}},'other':{'unchanged':True}},'prefix_cache':{'enabled':True},'keep':'unaltered'}
        result=beta.add_profile(original,self.qualify());self.assertEqual({k:v for k,v in result.items() if k!='profiles'},{k:v for k,v in original.items() if k!='profiles'})
        self.assertEqual({k:v for k,v in result['profiles'].items() if k!=beta.ALIAS},original['profiles']);self.assertNotIn(beta.ALIAS,original['profiles'])
        fragment=beta.opencode_fragment();self.assertNotIn('model',fragment);self.assertNotIn('settings',fragment['providers']['local-qwen']);self.assertEqual(fragment['providers']['local-qwen']['models'][beta.ALIAS]['limit']['output'],2048)

    def test_engine_identity_is_startup_only_and_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'owned-engine-byte-fixture';path.write_bytes(b'OWN_SYNTHETIC_ENGINE_BYTES');expected=hashlib.sha256(path.read_bytes()).hexdigest()
            with patch.object(beta,'SERVER_SHA',expected):
                beta.verify_engine({'server_binary':str(path),'server_sha256':expected})
                path.write_bytes(b'CHANGED_SYNTHETIC_BYTES')
                with self.assertRaises(ValueError):beta.verify_engine({'server_binary':str(path),'server_sha256':expected})
        profile=self.qualify()
        with patch.object(beta,'model_identity',return_value=profile['validation']['model_identity']),patch.object(beta,'verify_engine',side_effect=AssertionError('metadata poll must not hash binaries')):
            self.assertTrue(beta.eligible(profile))

    def test_beta_request_defaults_preserve_explicit_client_experiments(self):
        original={'messages':[{'role':'user','content':'OWN_SYNTHETIC'}]}
        value=beta.request_defaults(original,'/v1/chat/completions')
        self.assertEqual(value,{**original,'temperature':0,'cache_prompt':True,'max_tokens':2048,'chat_template_kwargs':{'enable_thinking':False}})
        self.assertEqual(original,{'messages':[{'role':'user','content':'OWN_SYNTHETIC'}]})
        explicit={**original,'temperature':.7,'cache_prompt':False,'max_completion_tokens':64,'chat_template_kwargs':{'enable_thinking':True},'response_format':{'type':'json_object'}}
        self.assertEqual(beta.request_defaults(explicit,'/v1/chat/completions'),explicit)
        self.assertEqual(beta.request_defaults(original,'/tokenize'),original)
        self.assertEqual(beta.request_defaults({'prompt':'own'},'/completion')['n_predict'],2048)

    def test_catalog_and_select_use_existing_controller_without_loading(self):
        with tempfile.TemporaryDirectory() as temporary:
            config=Path(temporary)/'runtime.json';config.write_text(json.dumps(beta.add_profile({'model_alias':'qwen','selected_model':'qwen','ready':True,'profiles':{'qwen':{'ready':True}}},self.qualify())))
            manager=object.__new__(controller.Manager)
            with patch.object(controller,'CONFIG',config),patch.object(beta,'model_identity',return_value=self.qualify()['validation']['model_identity']):
                self.assertIn(beta.ALIAS,{x['id'] for x in manager.models()})
            with patch.object(controller,'CONFIG',config),patch.object(beta,'model_identity',side_effect=FileNotFoundError):
                self.assertNotIn(beta.ALIAS,{x['id'] for x in manager.models()})

    def test_staging_is_create_only_with_exact_rollback_and_no_default_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);rows=self.evidence();runtime={'selected_model':'qwen','profiles':{'qwen':{'ready':True}},'prefix_cache':{'enabled':True}}
            inputs=[]
            for name,value in zip(('runtime','agent','screen','inventory'),(runtime,*rows)):
                p=root/(name+'.json');p.write_text(json.dumps(value));inputs.append((p,stager.sha(p.read_bytes())))
            (runtime_path,runtime_sha),(agent_path,agent_sha),(screen_path,screen_sha),(inventory_path,inventory_sha)=inputs
            # Wire synthetic hashes exactly as the real producer does.
            rows[0]['matched_screen_sha256']['ornith']=screen_sha;rows[0]['inventory_receipt_sha256']=inventory_sha;rows[1]['inventory_receipt_sha256']=inventory_sha
            screen_path.write_text(json.dumps(rows[1]));screen_sha=stager.sha(screen_path.read_bytes());rows[0]['matched_screen_sha256']['ornith']=screen_sha
            agent_path.write_text(json.dumps(rows[0]));agent_sha=stager.sha(agent_path.read_bytes())
            args=dict(runtime_path=runtime_path,runtime_sha=runtime_sha,agent_path=agent_path,agent_sha=agent_sha,screen_path=screen_path,screen_sha=screen_sha,inventory_path=inventory_path,inventory_sha=inventory_sha,reviewed_agent_sha=agent_sha)
            with patch.object(beta,'model_identity',return_value=rows[2]['inventory']['file_identity']):
                result=stager.stage(root/'stage',**args)
                with self.assertRaises(ValueError):stager.stage(root/'stage',**args)
            self.assertFalse(result['installed']);self.assertFalse(result['models_started']);self.assertEqual((root/'stage/rollback-runtime.json').read_bytes(),runtime_path.read_bytes());self.assertEqual(json.loads((root/'stage/runtime.json').read_text())['selected_model'],'qwen')
            self.assertTrue((root/'stage/runtime-source/classic_beta.py').is_file());self.assertEqual(json.loads(runtime_path.read_text()),runtime)


if __name__=='__main__':unittest.main()
