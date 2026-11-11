"""Preparation and independent result-gate tests; no engine/OpenCode executions."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('normal13_driver',HERE/'normal13.py')
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


class Normal13Preparation(unittest.TestCase):
    def test_prepare_preserves_fixture_and_protected_readonly_namespace(self):
        profile = dict(server_binary='/synthetic/engine/llama-server',model_path='/synthetic/model.gguf',
                       threads=8,context_size=16384,extra_args=['--spec-type','none'],environment={},
                       prefix_cache={'enabled':False})
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            fixture = driver.prepare(output,profile)
            for name in ('test_port.py','parse_port.py'):
                self.assertEqual((output/'project'/name).read_bytes(),(driver.ASSETS/'fixtures'/name).read_bytes())
            self.assertEqual(fixture['prompt'],(driver.ASSETS/'fixtures/TASK.txt').read_text())
            configured = json.loads((output/'profile.json').read_text())
            self.assertEqual((configured['threads'],configured['context_size']),(8,16384))
            command = driver.sandbox_command(output,profile,fixture,Path('/synthetic/opencode'))
            self.assertIn('--unshare-all',command)
            self.assertNotIn('--share-net',command)
            self.assertIn('--clearenv',command)
            self.assertIn('--die-with-parent',command)
            protected = command.index(str(output/'project/test_port.py'))
            self.assertEqual(command[protected-1],'--ro-bind')
            self.assertNotIn(str(Path.home()),command)

    def test_gate_requires_real_independent_13_checks_and_complete_lifecycle(self):
        report = dict(passed=True,server_reaped=True,server_reader_joined=True,
            agent=dict(passed=True,protected_unchanged=True,complete=True,readers_joined=True,
                expected_test_cases=13,independent_test_cases_run=13,changed_modules=['parse_port.py'],
                unexpected_files=[],opencode_exit=0,test_exit=0,completed_tools=1,test_commands_completed=1))
        self.assertTrue(driver.agent_gate(report))
        for key,value in (('independent_test_cases_run',12),('protected_unchanged',False),
                          ('complete',False),('completed_tools',0),('test_commands_completed',0),
                          ('unexpected_files',['extra']),('opencode_exit',1)):
            with self.subTest(key=key):
                changed = {**report,'agent':{**report['agent'],key:value}}
                self.assertFalse(driver.agent_gate(changed))
        self.assertFalse(driver.agent_gate({**report,'server_reaped':False}))

    def test_portable_opencode_pin_resolves_against_home(self):
        pinned = {'path': '$HOME/.local/bin/opencode'}
        self.assertEqual(driver.opencode_path(None, pinned), Path.home() / '.local/bin/opencode')
        self.assertEqual(driver.opencode_path('/explicit/opencode', pinned), Path('/explicit/opencode'))

    def test_checked_source_manifest_detects_drift(self):
        self.assertEqual(len(driver.source_gate()),64)
        with patch.object(driver,'sha',return_value='f'*64):
            with self.assertRaisesRegex(ValueError,'normal13_source_changed'):
                driver.source_gate()


if __name__ == '__main__':unittest.main()
