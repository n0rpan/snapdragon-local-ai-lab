import copy,json,unittest
from snapdragon_ai_lab.results import export_controller_gate
class ControllerExport(unittest.TestCase):
    def fixture(self):
        return {'passed':True,'agent_result':{'passed':True,'verified_completion_seconds':136,
          'profile':{'server_sha256':'a'*64,'target_model_sha256':'b'*64},
          'secret_model_path':'/private/model','stage_text':'secret user text',
          'warm_session':{'passed':True,'seconds':88,'test_file_unchanged':True,
            'full_backend_reload':True,'prefix_reused':True,'first_turn_new_tokens':10,'first_turn_cached_tokens':100,'controller_cache':{'restores':1}}}}
    def test_export_keeps_evidence_without_raw_content(self):
        result=export_controller_gate(self.fixture());self.assertTrue(result['passed'])
        self.assertNotIn('private',json.dumps(result));self.assertNotIn('secret',json.dumps(result))
    def test_missing_reload_cannot_claim_pass(self):
        source=self.fixture();source['agent_result']['warm_session']['full_backend_reload']=False
        self.assertFalse(export_controller_gate(source)['passed'])
    def test_nonfinite_time_cannot_claim_pass(self):
        source=self.fixture();source['agent_result']['warm_session']['seconds']=float('nan')
        self.assertFalse(export_controller_gate(source)['passed'])
if __name__=='__main__':unittest.main()
