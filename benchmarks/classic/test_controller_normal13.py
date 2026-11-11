"""Controller-client setup/gates only; no network, native engine or OpenCode run."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock,patch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('controller_normal13_driver',HERE/'controller_normal13.py')
driver=importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)
inner_spec=importlib.util.spec_from_file_location('controller_client_inner',HERE/'controller-normal13/inside_client.py')
inner=importlib.util.module_from_spec(inner_spec)
inner_spec.loader.exec_module(inner)


class ControllerNormal13(unittest.TestCase):
    def profile(self):
        return dict(server_binary='/synthetic/engine/server',model_path='/synthetic/model.gguf',
                    threads=8,context_size=16384,extra_args=[],environment={},model_alias='ornith-35b-a3b-local')

    def test_loopback_only_controller_target(self):
        self.assertEqual(driver.controller_port('http://127.0.0.1:18092'),18092)
        for target in ('https://127.0.0.1:8092','http://localhost:8092','http://192.0.2.1:8092',
                       'http://user@127.0.0.1:8092','http://127.0.0.1','http://127.0.0.1:8092/path',
                       'http://127.0.0.1:8092/?token=x'):
            with self.subTest(target=target),self.assertRaises(ValueError):driver.controller_port(target)

    def test_original_fixture_and_request_alias_with_no_engine_model_bind(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)
            profile=self.profile()
            fixture=driver.prepare(output,profile)
            self.assertEqual((output/'project/test_port.py').read_bytes(),
                             (driver.common.ASSETS/'fixtures/test_port.py').read_bytes())
            self.assertEqual((output/'inside.py').read_bytes(),(driver.common.ASSETS/'inside.py').read_bytes())
            config=json.loads((output/'opencode.json').read_text())
            self.assertEqual(config['model'],'local-qwen/'+profile['model_alias'])
            self.assertEqual(list(config['providers']['local-qwen']['models']),[profile['model_alias']])
            command=driver.sandbox_command(output,profile,fixture,Path('/synthetic/opencode'),Path('/tmp/short/socket'))
            self.assertIn('--unshare-all',command)
            self.assertNotIn('--share-net',command)
            self.assertNotIn('/model.gguf',command)
            self.assertNotIn('/engine',command)
            self.assertNotIn(profile['model_path'],command)
            self.assertNotIn(str(Path(profile['server_binary']).parent),command)
            self.assertEqual(command[-1],'/work/inside_client.py')
            binding=command.index('/work/controller.sock')
            self.assertEqual(command[binding-2],'--ro-bind')
            self.assertGreater(binding,command.index('/work'))
            self.assertLess(binding,command.index('--chdir'))

    def test_client_gate_requires_independent_13_complete_tools_and_protected_tests(self):
        agent=dict(passed=True,protected_unchanged=True,complete=True,readers_joined=True,
                   expected_test_cases=13,independent_test_cases_run=13,changed_modules=['parse_port.py'],
                   unexpected_files=[],opencode_exit=0,test_exit=0,completed_tools=1,test_commands_completed=1)
        self.assertTrue(inner.agent_passed(agent))
        for key,value in (('independent_test_cases_run',12),('complete',False),('readers_joined',False),
                          ('protected_unchanged',False),('completed_tools',0),('test_commands_completed',0),
                          ('opencode_exit',1),('unexpected_files',['extra'])):
            with self.subTest(key=key):self.assertFalse(inner.agent_passed({**agent,key:value}))

    def test_inner_starts_only_bridge_and_reuses_original_agent(self):
        profile={'model_alias':'synthetic-alias'}
        fixture={'prompt':'own fixture'}
        agent=dict(passed=True,protected_unchanged=True,complete=True,readers_joined=True,
                   expected_test_cases=13,independent_test_cases_run=13,changed_modules=['parse_port.py'],
                   unexpected_files=[],opencode_exit=0,test_exit=0,completed_tools=1,test_commands_completed=1)
        bridge=Mock()
        bridge.poll.return_value=None
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)
            (output/'profile.json').write_text(json.dumps(profile))
            (output/'fixture.json').write_text(json.dumps(fixture))
            def terminate():bridge.poll.return_value=0
            bridge.terminate.side_effect=terminate
            with patch.object(inner,'ROOT',output),patch.object(inner.subprocess,'Popen',return_value=bridge) as launch, \
                 patch.object(inner.threading,'Thread') as thread, \
                 patch.object(inner.original,'api',return_value=({'data':[{'id':'synthetic-alias'}]},.01)), \
                 patch.object(inner.original,'run_agent',return_value=agent) as run_agent:
                thread.return_value.is_alive.return_value=False
                self.assertEqual(inner.main(),0)
                self.assertEqual(launch.call_count,1)
                self.assertIn('/work/loopback_bridge.py',launch.call_args.args[0])
                self.assertNotIn('/work/inside.py',launch.call_args.args[0])
                run_agent.assert_called_once_with(fixture)
            result=json.loads((output/'controller-result.json').read_text())
            self.assertFalse(result['native_server_started'])
            self.assertFalse(result['canonical_lease_acquired'])

    def test_frozen_controller_sources_detect_drift(self):
        self.assertEqual(len(driver.source_gate()),64)
        with patch.object(driver.common,'sha',return_value='f'*64):
            with self.assertRaisesRegex(ValueError,'controller_client_source_changed'):driver.source_gate()

    def test_private_capture_verifies_system_date_and_complete_bodies(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            body = json.dumps({'messages':[{'role':'system','content':"Today's date: Wed Nov 11 2026"}],
                               'tools':[{'type':'function','function':{'name':'edit'}}]}).encode()
            raw = (b'GET /v1/models HTTP/1.1\r\nHost: localhost\r\n\r\n'
                   b'POST /v1/chat/completions HTTP/1.1\r\nContent-Length: '+str(len(body)).encode()
                   +b'\r\n\r\n'+body)
            path=root/'request-000.http'
            path.write_bytes(raw)
            rows=driver.captured_requests(root)
            self.assertEqual(len(rows),1)
            self.assertEqual(rows[0]['tools'],['edit'])
            self.assertNotIn('messages',rows[0])
            self.assertEqual(rows[0]['system_dates'], ["Today's date: Wed Nov 11 2026"])
            path.write_bytes(raw[:-1])
            with self.assertRaisesRegex(ValueError,'incomplete_captured_request_body'):
                driver.captured_requests(root)


if __name__=='__main__':unittest.main()
