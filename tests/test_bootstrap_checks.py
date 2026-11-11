"""Fake executable bytes and injected runner; no native engine or compiler executed."""
import hashlib
import tempfile
from pathlib import Path
import unittest
from snapdragon_ai_lab import bootstrap_checks as c
from snapdragon_ai_lab.bootstrap_guard import Stopped
class Checks(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name);folder=self.root/'build/bin';folder.mkdir(parents=True)
        self.expected={}
        for name in [*c.TESTS,'bmoe-cli','bmoe-server']:
            p=folder/name;p.write_bytes(('synthetic '+name).encode());p.chmod(0o700)
            if name in c.TESTS:self.expected[name]=hashlib.sha256(p.read_bytes()).hexdigest()
        self.result={'passed':True,'built':True,'models_run':False,'binary_sha256':{'bin/bmoe-server':c.sha(folder/'bmoe-server'),'bin/bmoe-cli':c.sha(folder/'bmoe-cli')}}
    def test_exact_four_checks_rehash_and_never_qualify(self):
        calls=[]
        class Stub:
            def call(_,args):calls.append(Path(args[0]).name);return 'all checks passed\n'
        result=c.run_checks(self.root,self.result,self.expected,Stub())
        self.assertEqual(calls,list(c.TESTS));self.assertTrue(result['model_free_checks_passed']);self.assertFalse(result['qualified']);self.assertFalse(result['models_run']);self.assertTrue(all(x['exit']==0 and not x['skipped'] for x in result['rows']))
    def test_changed_binary_and_zero_exit_skip_rejected(self):
        class Stub:
            def call(_,args):return 'SKIPPED unsupported case\n'
        with self.assertRaisesRegex(Stopped,'skip_rejected'):c.run_checks(self.root,self.result,self.expected,Stub())
        (self.root/'build/bin/bmoe_config_test').write_bytes(b'changed')
        with self.assertRaisesRegex(Stopped,'identity_mismatch'):c.run_checks(self.root,self.result,self.expected,Stub())
    def test_incomplete_registry_and_failed_native_process_rejected(self):
        class Fail:
            def call(_,args):raise Stopped('command_failed_fixture_1')
        with self.assertRaisesRegex(Stopped,'exact_four'):c.run_checks(self.root,self.result,{},Fail())
        with self.assertRaisesRegex(Stopped,'command_failed_fixture_1'):c.run_checks(self.root,self.result,self.expected,Fail())
if __name__=='__main__':unittest.main()
