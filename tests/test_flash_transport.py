"""Proxy transport and browser origins with a fake runtime and owned HTTP servers."""
from email.message import Message
import http.client
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
from types import SimpleNamespace
import threading
import unittest
from unittest.mock import patch
from snapdragon_ai_lab.runtime import controller

class FlashTransport(unittest.TestCase):
    def setUp(self):
        self.seen=[];seen=self.seen
        class Native(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_POST(self):
                body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                seen.append({'host':self.headers['Host'],'origin':self.headers.get('Origin'),
                             'authorization':self.headers.get('Authorization'),'model':body['model']})
                data=b'{"choices":[{"message":{"content":"READY"}}]}'
                self.send_response(200);self.send_header('Content-Type','application/json')
                self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
        self.native=ThreadingHTTPServer(('127.0.0.1',0),Native)
        self.native_thread=threading.Thread(target=self.native.serve_forever,daemon=True);self.native_thread.start()
        self.runtime=SimpleNamespace(spec=SimpleNamespace(port=self.native.server_port,backend_model_id='synthetic-native'),
                                     headers=lambda:{'Authorization':'Bearer SYNTHETIC_PRIVATE_KEY'})
        self.profile={'model_alias':'synthetic-flash','backend_kind':controller.bigmoe_runtime.KIND}
        self.begins=0
        def begin(**kwargs):self.begins+=1
        self.manager=SimpleNamespace(config=lambda *args:self.profile,loaded_config=None,
            backend_runtime=self.runtime,begin=begin,begin_queued=begin,end=lambda **kwargs:None,eligible=lambda config:True,
            current_prefix_key=None,cache={})
        self.proxy=controller.Server(('127.0.0.1',0),controller.Handler);self.proxy.manager=self.manager
        self.proxy_thread=threading.Thread(target=self.proxy.serve_forever,daemon=True);self.proxy_thread.start()
        def normalize(config,body,port):
            self.assertEqual(port,self.native.server_port)
            return {**body,'model':'synthetic-native'}
        self.normalizer=patch.object(controller.bigmoe_runtime,'normalize_chat',side_effect=normalize)
        self.normalizer.start();self.port=patch.object(controller,'FLASH_BACKEND_PORT',self.native.server_port);self.port.start()
        self.ledger=patch.dict('os.environ',{'LOCAL_AI_LEDGER':'0'});self.ledger.start()
    def tearDown(self):
        self.ledger.stop()
        self.proxy.shutdown();self.proxy.server_close();self.native.shutdown();self.native.server_close()
        self.proxy_thread.join(2);self.native_thread.join(2);self.normalizer.stop();self.port.stop()
    def request(self,headers):
        connection=http.client.HTTPConnection('127.0.0.1',self.proxy.server_port,timeout=2)
        try:
            connection.request('POST','/v1/chat/completions',body=json.dumps({'model':'synthetic-flash','messages':[]}),headers=headers)
            response=connection.getresponse();return response.status,response.read()
        finally:connection.close()
    def test_local_browser_origin_is_rewritten_and_raw_client_stays_originless(self):
        for host in ('127.0.0.1','localhost'):
            self.assertEqual(self.request({'Origin':f'http://{host}:{self.proxy.server_port}','Sec-Fetch-Site':'same-origin'})[0],200)
        self.assertEqual(self.request({})[0],200)
        self.assertEqual(self.begins,3)
        self.assertEqual([row['origin'] for row in self.seen],[f'http://127.0.0.1:{self.native.server_port}']*2+[None])
        self.assertTrue(all(row['host']==f'127.0.0.1:{self.native.server_port}' for row in self.seen))
        self.assertTrue(all(row['authorization']=='Bearer SYNTHETIC_PRIVATE_KEY' for row in self.seen))
    def test_foreign_browser_does_not_wake_or_reach_native(self):
        for headers in ({'Origin':'https://example.invalid'},{'Sec-Fetch-Site':'cross-site'},
                        {'Origin':f'http://127.0.0.1:{self.proxy.server_port}','Sec-Fetch-Site':'same-site'}):
            self.assertEqual(self.request(headers)[0],403)
        self.assertEqual(self.begins,0);self.assertEqual(self.seen,[])
    def test_deliberate_chat_navigation_is_lazy_but_background_navigation_is_refused(self):
        navigation={'Sec-Fetch-Site':'same-site','Sec-Fetch-Mode':'navigate',
                    'Sec-Fetch-Dest':'document','Sec-Fetch-User':'?1'}
        for changed,expected in (({},200),({'Sec-Fetch-Dest':'image'},403),({'Sec-Fetch-User':'?0'},403)):
            connection=http.client.HTTPConnection('127.0.0.1',self.proxy.server_port,timeout=2)
            try:
                connection.request('GET','/',headers={**navigation,**changed})
                response=connection.getresponse();body=response.read()
                self.assertEqual(response.status,expected)
                if expected==200:self.assertIn(b'Local Flash chat',body)
            finally:connection.close()
        self.assertEqual(self.begins,0);self.assertEqual(self.seen,[])

    def test_duplicate_browser_headers_are_rejected(self):
        handler=object.__new__(controller.Handler);handler.server=self.proxy;handler.headers=Message()
        handler.headers.add_header('Origin',f'http://127.0.0.1:{self.proxy.server_port}')
        handler.headers.add_header('Origin','https://example.invalid')
        self.assertFalse(handler.browser_origin_allowed())

if __name__=='__main__':unittest.main()
