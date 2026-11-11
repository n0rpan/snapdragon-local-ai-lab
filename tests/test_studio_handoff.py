"""Owned HTTP/lock tests only; no image/chat model or live endpoint is used."""
import os as _os, tempfile as _tempfile
_os.environ.setdefault('LOCAL_AI_LEDGER_DIR', _tempfile.mkdtemp(prefix='local-ai-test-ledger-'))  # never the real request ledger
import fcntl
import importlib.machinery
import importlib.util
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from urllib.request import Request

SOURCE=Path(__file__).resolve().parents[1]/'src/snapdragon_ai_lab/runtime/studio'

class Handoff(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.root=Path(self.temporary.name)
        environment={'IMG_OUTDIR':str(self.root/'output'),'LOCAL_IMAGE_CONFIG_DIR':str(self.root/'config'),
                     'LOCAL_AI_LEASE':str(self.root/'inference.lock'),'SD_HOME':str(self.root/'stack')}
        with patch.dict(os.environ,environment):
            loader=importlib.machinery.SourceFileLoader('handoff_studio',str(SOURCE))
            spec=importlib.util.spec_from_loader(loader.name,loader);self.studio=importlib.util.module_from_spec(spec);loader.exec_module(self.studio)
        self.owner=self.studio.GENERATION_LOCK.open('a');self.generation=self.studio.GENERATION_LOCK.open('a')
        self.calls=0;self.mode='idle';self.delay=0;self.release=True;self.observed_timeout=None
        fixture=self
        class Control(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_POST(self):
                fixture.calls+=1
                assert self.path=='/_local/unload'
                assert self.rfile.read(int(self.headers['Content-Length']))==b'{}'
                if fixture.mode=='active':code,value=409,{'error':'A request is running.'}
                else:
                    time.sleep(fixture.delay)
                    if fixture.release:fcntl.flock(fixture.owner,fcntl.LOCK_UN)
                    code,value=200,{'state':'unloaded'}
                body=json.dumps(value).encode();self.send_response(code)
                self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
        self.server=ThreadingHTTPServer(('127.0.0.1',0),Control)
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
        original=self.studio.urlopen
        def redirect(request,timeout):
            self.assertEqual(request.full_url,'http://127.0.0.1:8092/_local/unload')
            self.observed_timeout=timeout
            owned=Request(f'http://127.0.0.1:{self.server.server_port}/_local/unload',data=request.data,headers=dict(request.header_items()))
            return original(owned,timeout=timeout)
        self.redirect=patch.object(self.studio,'urlopen',side_effect=redirect);self.redirect.start()
    def tearDown(self):
        self.server.shutdown();self.server.server_close();self.thread.join(2)
        self.owner.close();self.generation.close();self.redirect.stop();self.temporary.cleanup()
    def hold(self):fcntl.flock(self.owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def assert_other_holder(self):
        with self.studio.GENERATION_LOCK.open('a') as third:
            with self.assertRaises(BlockingIOError):fcntl.flock(third,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def test_slow_idle_unload_confirms_release_before_image_claim(self):
        self.hold();self.delay=1.2;started=time.monotonic()
        self.assertTrue(self.studio.claim_generation_slot(self.generation))
        self.assertGreaterEqual(time.monotonic()-started,1.2)
        self.assertEqual(self.observed_timeout,15);self.assertEqual(self.calls,1)
        self.assert_other_holder()
    def test_active_chat_is_refused_without_releasing_its_lock(self):
        self.hold();self.mode='active';started=time.monotonic()
        self.assertFalse(self.studio.claim_generation_slot(self.generation))
        self.assertLess(time.monotonic()-started,1)
        self.assertEqual(self.calls,1);self.assert_other_holder()
    def test_unloaded_controller_cannot_override_another_lease_holder(self):
        self.hold();self.release=False
        self.assertFalse(self.studio.claim_generation_slot(self.generation))
        self.assertEqual(self.calls,1);self.assert_other_holder()
    def test_timeout_is_not_treated_as_unloaded_success(self):
        self.hold()
        with patch.object(self.studio,'urlopen',side_effect=TimeoutError):
            self.assertFalse(self.studio.claim_generation_slot(self.generation))
        self.assert_other_holder()
    def test_free_slot_is_claimed_without_contacting_chat(self):
        self.assertTrue(self.studio.claim_generation_slot(self.generation))
        self.assertEqual(self.calls,0);self.assert_other_holder()

if __name__=='__main__':unittest.main()
