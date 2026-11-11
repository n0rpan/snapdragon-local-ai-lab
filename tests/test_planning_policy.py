import importlib.util
import json
from pathlib import Path
import tempfile
from unittest.mock import patch, MagicMock
from snapdragon_ai_lab.runtime import controller
import unittest

from snapdragon_ai_lab.runtime import bigmoe_runtime as runtime

HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('plan_launch', HERE/'src/snapdragon_ai_lab/runtime/launch_local_plan.py')
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class PlanningPolicy(unittest.TestCase):
    def test_battery_start_and_stop_are_distinct_from_benchmark_ac_requirement(self):
        for external in (False, True):
            state = {'external_online':external,'battery_status':'Discharging','battery_percent':30}
            self.assertIsNone(runtime.power_failure(state,starting=True))
            self.assertIsNone(runtime.power_failure({**state,'battery_percent':21}))
            self.assertEqual(runtime.power_failure({**state,'battery_percent':29},starting=True),'flash_battery_start_reserve')
            self.assertEqual(runtime.power_failure({**state,'battery_percent':20}),'flash_battery_stop_reserve')
        for value in [None,{}, {'battery_status':'Full','battery_percent':True},
                      {'battery_status':'Full','battery_percent':float('nan')},
                      {'battery_status':'Full','battery_percent':101}]:
            self.assertEqual(runtime.power_failure(value),'flash_power_telemetry_unavailable')

    def test_power_snapshot_charge_fallback_and_unknown_values(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td);battery=root/'battery';battery.mkdir()
            for key,value in {'type':'Battery','status':'Discharging','charge_now':'35','charge_full':'100'}.items():
                (battery/key).write_text(value)
            self.assertEqual(runtime.power_snapshot(root)['battery_percent'],35)
            self.assertFalse(runtime.power_snapshot(root)['external_online'])
            (battery/'capacity').write_text('nan')
            self.assertIsNone(runtime.power_snapshot(root)['battery_percent'])

    def test_launch_overlay_is_read_only_and_preserves_build_and_global_model(self):
        original={'model':'local-qwen/qwen3.6-35b-a3b-local','agents':{'build':{'system':'KEEP'}}}
        environment={'OPENCODE_CONFIG_CONTENT':json.dumps(original),'SYNTHETIC_SECRET':'never printed'}
        updated=launcher.environment_for_named_agent(environment)
        value=json.loads(updated['OPENCODE_CONFIG_CONTENT'])
        self.assertEqual(value['model'],original['model'])
        self.assertEqual(value['agents']['build'],original['agents']['build'])
        self.assertEqual(value['default_agent'],'local-plan')
        self.assertEqual(value['agents']['local-plan']['model'],launcher.MODEL)
        self.assertEqual(value['policies'][-2:],launcher.READ_ONLY_POLICIES)
        self.assertEqual(json.loads(environment['OPENCODE_CONFIG_CONTENT']),original)
        self.assertNotIn('SYNTHETIC_SECRET',value)

    def test_classic_prefix_rpc_keeps_900_second_timeout(self):
        manager=controller.Manager()
        connection=MagicMock();connection.getresponse.return_value.status=200
        connection.getresponse.return_value.read.return_value=b'{"n_restored":1}'
        with patch.object(controller.http.client,'HTTPConnection',return_value=connection) as factory:
            self.assertEqual(manager.backend_json('/slots/0?action=restore',{'filename':'synthetic'}),{'n_restored':1})
        factory.assert_called_once_with('127.0.0.1',controller.BACKEND_PORT,timeout=900)

    def test_profile_stays_disabled_without_actual_final_agent_receipt(self):
        profile=json.loads((HERE/'examples/flash-next-planning.pending.json').read_text())
        self.assertFalse(profile['ready'])
        self.assertFalse(runtime.eligible(profile))
        self.assertEqual(profile['bigmoe']['admission']['measured_peak_bytes'],None)


if __name__=='__main__':unittest.main()
