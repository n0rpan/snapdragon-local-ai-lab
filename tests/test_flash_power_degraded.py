"""USB-C battmgr outages must not conceal a valid low-battery measurement."""
import tempfile
import unittest
from pathlib import Path
from snapdragon_ai_lab.runtime import bigmoe_runtime as power

class DegradedPower(unittest.TestCase):
    def state(self, percent=None, online=True, status='Not charging'):
        return dict(external_online=online, battery_percent=percent, battery_status=status)

    def test_online_usb_permits_only_missing_reserve(self):
        self.assertIsNone(power.power_failure(self.state(),starting=True))
        self.assertIsNone(power.power_failure(self.state()))
        self.assertEqual(power.power_failure(self.state(online=False)),'flash_power_telemetry_unavailable')
        for invalid in (True,float('nan'),101,-1):
            self.assertEqual(power.power_failure(self.state(invalid)),'flash_power_telemetry_unavailable')

    def test_known_reserve_always_wins_over_external_power(self):
        for online in (True,False):
            self.assertEqual(power.power_failure(self.state(20,online)),'flash_battery_stop_reserve')
            self.assertEqual(power.power_failure(self.state(29,online),starting=True),'flash_battery_start_reserve')
            self.assertEqual(power.power_failure(self.state(5,online,status=None)),'flash_battery_stop_reserve')
            self.assertIsNone(power.power_failure(self.state(30,online),starting=True))

    def test_empty_capacity_uses_real_energy_before_mains_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);bat=root/'battery';usb=root/'ucsi-source-psy-test'
            bat.mkdir();usb.mkdir()
            for name,value in {'type':'Battery','status':'Discharging','capacity':'','energy_now':'15','energy_full':'100'}.items():
                (bat/name).write_text(value)
            (usb/'type').write_text('USB_PD_DRP');(usb/'online').write_text('1')
            value=power.power_snapshot(root)
            self.assertTrue(value['external_online']);self.assertEqual(value['battery_percent'],15)
            self.assertEqual(power.power_failure(value),'flash_battery_stop_reserve')
            (bat/'energy_now').write_text('');(bat/'charge_now').write_text('35');(bat/'charge_full').write_text('100')
            self.assertEqual(power.power_snapshot(root)['battery_percent'],35)
            (bat/'charge_now').unlink()
            self.assertIsNone(power.power_snapshot(root)['battery_percent'])
            self.assertIsNone(power.power_failure(power.power_snapshot(root)))
            (usb/'online').write_text('0')
            self.assertEqual(power.power_failure(power.power_snapshot(root)),'flash_power_telemetry_unavailable')

    def test_degraded_transition_logged_once_and_warned_after_ten_minutes(self):
        state=[self.state()];clock=[0.];logs=[]
        monitor=power.PowerTelemetry(lambda:state[0],clock=lambda:clock[0],log=logs.append)
        monitor.sample();clock[0]=599;monitor.sample();self.assertEqual(len(logs),1)
        clock[0]=600;monitor.sample();clock[0]=601;monitor.sample();self.assertEqual(len(logs),2)
        self.assertIn('10 minutes',logs[-1]);self.assertEqual(monitor.metadata()['longest_degraded_seconds'],601)
        state[0]=self.state(80);monitor.sample();self.assertIn('recovered',logs[-1])
        self.assertFalse(monitor.metadata()['degraded_now'])

    def test_usb_loss_stops_during_degraded_period(self):
        state=[self.state()];monitor=power.PowerTelemetry(lambda:state[0],log=lambda _:None)
        self.assertIsNone(power.power_failure(monitor.sample()))
        state[0]=self.state(online=False)
        self.assertEqual(power.power_failure(monitor.sample()),'flash_power_telemetry_unavailable')

if __name__=='__main__':unittest.main()
