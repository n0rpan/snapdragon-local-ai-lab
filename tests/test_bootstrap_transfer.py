"""Fake-response resume tests; no sockets, canonical lease, weights or compiler."""
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from snapdragon_ai_lab import bootstrap_transfer as d
from snapdragon_ai_lab.bootstrap_guard import Stopped
from snapdragon_ai_lab.bootstrap_download import space_needed

class Okay:
    def require(self):pass
    def network_call(self,fn):return fn()
    def report(self):return {'synthetic':True}
class Response(io.BytesIO):
    def __init__(self,payload,status,headers):super().__init__(payload);self.status=status;self.headers=headers
class Opener:
    def __init__(self,response):self.response=response;self.headers=[]
    def open(self,request,timeout):self.headers.append(dict(request.header_items()));return self.response
class Transfer(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.payload=b'abcdef';self.asset={'id':'fixture','url':'https://example.invalid/pinned','bytes':6,'sha256':hashlib.sha256(self.payload).hexdigest(),'stack_relative_path':'asset.bin'}
    def partial(self,data=b'abc',offset=3):
        p=self.root/'.asset.bin.partial';p.write_bytes(data);p.chmod(0o600)
        d.atomic_json(self.root/'.asset.bin.partial.json',{'url':self.asset['url'],'expected_bytes':6,'expected_sha256':self.asset['sha256'],'completed_bytes':offset,'prefix_sha256':hashlib.sha256(data[:offset]).hexdigest()})
    def test_resume_range_exact_cached_hash_and_create_only_end(self):
        self.partial(b'abcUNSYNCED',3)
        # Unsynced tail may not exceed expected total; use a valid short tail.
        (self.root/'.asset.bin.partial').write_bytes(b'abcX')
        opener=Opener(Response(b'def',206,{'Content-Range':'bytes 3-5/6','Content-Length':'3'}))
        result=d.download(self.asset,self.root,Okay(),opener)
        self.assertEqual(result['status'],'verified_complete');self.assertEqual((self.root/'asset.bin').read_bytes(),self.payload)
        self.assertEqual(opener.headers[0]['Range'],'bytes=3-');self.assertFalse((self.root/'.asset.bin.partial').exists())
        self.assertEqual(d.download(self.asset,self.root,Okay(),None)['status'],'already_verified_complete')
    def test_ignored_range_never_appends_or_reads_body(self):
        self.partial();response=Response(b'abcdef',200,{'Content-Length':'6'})
        with self.assertRaisesRegex(Stopped,'range_not_honored'):d.download(self.asset,self.root,Okay(),Opener(response))
        self.assertEqual((self.root/'.asset.bin.partial').read_bytes(),b'abc');self.assertFalse((self.root/'asset.bin').exists())
    def test_incomplete_eof_is_durable_and_resumable(self):
        with self.assertRaisesRegex(Stopped,'incomplete_resume'):d.download(self.asset,self.root,Okay(),Opener(Response(b'abc',200,{'Content-Length':'6'})))
        value=json.loads((self.root/'.asset.bin.partial.json').read_text());self.assertEqual(value['completed_bytes'],3)
        self.assertEqual(value['prefix_sha256'],hashlib.sha256(b'abc').hexdigest())
        d.download(self.asset,self.root,Okay(),Opener(Response(b'def',206,{'Content-Range':'bytes 3-5/6'})))
        self.assertEqual((self.root/'asset.bin').read_bytes(),self.payload)
    def test_corrupt_hash_and_existing_final_preserved(self):
        with self.assertRaisesRegex(Stopped,'sha256_mismatch'):d.download(self.asset,self.root,Okay(),Opener(Response(b'BADBAD',200,{'Content-Length':'6'})))
        self.assertFalse((self.root/'asset.bin').exists())
        target=self.root/'asset.bin';target.write_bytes(b'retain');target.chmod(0o600)
        with self.assertRaisesRegex(Stopped,'existing_final_identity'):d.download(self.asset,self.root,Okay(),None)
        self.assertEqual(target.read_bytes(),b'retain')
    def test_complete_partial_can_finish_without_network(self):
        self.partial(self.payload,6);d.download(self.asset,self.root,Okay(),None)
        self.assertEqual((self.root/'asset.bin').read_bytes(),self.payload)
    def test_stop_saves_prefix_and_same_resume_identity(self):
        class StopResponse(Response):
            def read1(self,n):
                if self.tell():raise Stopped('synthetic_power_stop')
                return self.read(3)
        with self.assertRaisesRegex(Stopped,'synthetic_power_stop'):d.download(self.asset,self.root,Okay(),Opener(StopResponse(b'abcdef',200,{})))
        value=json.loads((self.root/'.asset.bin.partial.json').read_text());self.assertEqual(value['completed_bytes'],3);self.assertEqual(value['stop_reason'],'synthetic_power_stop')
        value['url']='https://example.invalid/other';d.atomic_json(self.root/'.asset.bin.partial.json',value)
        with self.assertRaisesRegex(Stopped,'identity_changed'):d.download(self.asset,self.root,Okay(),None)
    def test_space_preflight_uses_checkpoint_and_rejects_symlink_metadata(self):
        self.partial();self.assertEqual(space_needed([self.asset],self.root),1024**3+3)
        nested={**self.asset,'stack_relative_path':'escape/asset.bin'};(self.root/'escape').symlink_to(self.root,target_is_directory=True)
        with self.assertRaisesRegex(Stopped,'subpath_symlink'):space_needed([nested],self.root)

if __name__=='__main__':unittest.main()
