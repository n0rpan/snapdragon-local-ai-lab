import tempfile
import unittest
from pathlib import Path
from snapdragon_ai_lab import power as power_guard

class PowerChecks(unittest.TestCase):
    def state(self,percent=50,status='Charging',online=True):
        return {'external_online':online,'battery_percent':percent,'battery_status':status}

    def test_temporary_discharge_is_allowed_above_reserve(self):
        self.assertTrue(power_guard.start_allowed(self.state(status='Discharging'))[0])
        self.assertIsNone(power_guard.stop_reason(self.state(status='Discharging')))
        self.assertFalse(power_guard.start_allowed(self.state(percent=20,status='Discharging'))[0])

    def test_low_reserve_waits_before_start_but_has_hysteresis(self):
        self.assertFalse(power_guard.start_allowed(self.state(percent=27))[0])
        self.assertIsNone(power_guard.stop_reason(self.state(percent=27)))
        self.assertTrue(power_guard.start_allowed(self.state(percent=30))[0])
        self.assertIsNotNone(power_guard.stop_reason(self.state(percent=20)))

    def test_power_loss_and_missing_telemetry_stop(self):
        self.assertIsNotNone(power_guard.stop_reason(self.state(online=False)))
        self.assertFalse(power_guard.start_allowed(self.state(percent=None))[0])

    def test_long_jobs_can_require_a_larger_starting_reserve(self):
        self.assertFalse(power_guard.start_allowed(self.state(percent=35),min_percent=40)[0])

    def test_full_and_charge_limit_states_are_supported(self):
        for status in ('Full','Not charging'):
            self.assertTrue(power_guard.start_allowed(self.state(percent=80,status=status))[0])

    def test_energy_only_battery_is_measured_without_capacity(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            values={'battery':{'type':'Battery','status':'Charging','energy_now':'19070000','energy_full':'71460000'},
                    'usb':{'type':'USB','online':'1'}}
            for name,fields in values.items():
                device=root/name;device.mkdir()
                for key,value in fields.items():(device/key).write_text(value)
            state=power_guard.snapshot(root)
            self.assertAlmostEqual(state['battery_percent'],26.686258,places=5)
            self.assertFalse(power_guard.start_allowed(state)[0])

    def test_reported_charging_cannot_hide_sustained_energy_loss(self):
        samples=[(t,{**self.state(percent=35-t/100),'battery_power_now_uw':-15_000_000}) for t in (0,30,60)]
        self.assertIsNotNone(power_guard.discharge_trend(samples))
        self.assertIsNone(power_guard.discharge_trend(samples[:2]))

    def test_transient_percentage_drop_with_positive_charge_power_is_not_loss(self):
        samples=[(t,{**self.state(percent=35-t/100),'battery_power_now_uw':8_000_000}) for t in (0,30,60)]
        self.assertIsNone(power_guard.discharge_trend(samples))

if __name__=='__main__':
    unittest.main()
