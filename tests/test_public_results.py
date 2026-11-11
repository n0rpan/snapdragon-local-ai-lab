import json
import unittest
from snapdragon_ai_lab.results import export_screen

class PublicExport(unittest.TestCase):
    def test_only_metrics_survive_export(self):
        raw={'api_key':'example-secret','rows':[{'configuration':{'id':'trial-1','threads':4,'prompt':'secret'},
             'passed':True,'request_seconds':12.5,'evidence':'/home/example/private.json',
             'text':'private answer','server_args':['secret'],
             'case_timings':{'code-example':{'prompt_n':100,'prompt_ms':50,'messages':['private']}}}]}
        output=json.dumps(export_screen(raw))
        for forbidden in ['secret','private','/home/','messages','server_args','api_key']:
            self.assertNotIn(forbidden,output)
        self.assertIn('12.5',output)

    def test_paths_are_rejected_as_identifiers(self):
        with self.assertRaises(ValueError):
            export_screen({'rows':[{'configuration':{'id':'/home/example'}}]})

    def test_nonfinite_and_boolean_timings_are_not_numbers(self):
        result=export_screen({'rows':[{'request_seconds':float('nan'),'load_seconds':True}]})
        self.assertIsNone(result['rows'][0]['request_seconds_total'])
        self.assertIsNone(result['rows'][0]['load_seconds'])

    def test_provenance_accepts_only_digests(self):
        result=export_screen({'target_sha256':'a'*64})
        self.assertEqual(result['identities']['target_sha256'],'a'*64)
        with self.assertRaises(ValueError):
            export_screen({'binary_sha256':'/home/example/private-engine'})

if __name__=='__main__':unittest.main()
