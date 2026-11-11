"""Catalog polling stays usable while backend lifecycle cleanup owns its lock."""
import os as _os, tempfile as _tempfile
_os.environ.setdefault('LOCAL_AI_LEDGER_DIR', _tempfile.mkdtemp(prefix='local-ai-test-ledger-'))  # never the real request ledger
import http.client
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from snapdragon_ai_lab.runtime import controller


class MetadataCatalog(unittest.TestCase):
    def test_busy_lifecycle_catalog_is_nonwaking_and_filters_unqualified_profiles(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);config=root/'runtime.json'
            value={'ready':True,'model_alias':'classic-a','selected_model':'classic-a',
                   'profiles':{'classic-b':{'ready':True},
                               'flash-unqualified':{'ready':False,'backend_kind':'bigmoe-cpu-stream'}}}
            config.write_text(json.dumps(value))
            with patch.object(controller,'CONFIG',config):
                manager=controller.Manager()
                server=controller.Server(('127.0.0.1',0),controller.Handler);server.manager=manager
                worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
                def request(path,method='GET'):
                    connection=http.client.HTTPConnection('127.0.0.1',server.server_port,timeout=2)
                    try:
                        connection.request(method,path);response=connection.getresponse()
                        return response.status,response.read()
                    finally:connection.close()
                try:
                    with manager.mu,patch.object(manager,'begin',side_effect=AssertionError('catalog must not load')):
                        for path in ('/v1/models','/models'):
                            status,body=request(path)
                            self.assertEqual(status,200)
                            self.assertEqual({row['id'] for row in json.loads(body)['data']},{'local-default','qwen-local','classic-a','classic-b'})
                            self.assertEqual(request(path,'HEAD'),(200,b''))
                        self.assertEqual(request('/health')[0],503)
                        value['selected_model']='flash-unqualified';config.write_text(json.dumps(value))
                        status,body=request('/v1/models')
                        self.assertEqual(status,200)
                        self.assertEqual({row['id'] for row in json.loads(body)['data']},{'classic-a','classic-b'})
                    self.assertIsNone(manager.process);self.assertIsNone(manager.lease)
                finally:
                    server.shutdown();server.server_close();manager.close();worker.join(timeout=2)


if __name__=='__main__':unittest.main()
