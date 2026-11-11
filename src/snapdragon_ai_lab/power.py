"""Power checks for our laptop benchmarks; no system power settings are changed."""
from pathlib import Path

def snapshot(root=Path('/sys/class/power_supply')):
    supplies=[]
    for device in root.glob('*'):
        row={'device':device.name}
        for name in ('type','scope','online','status','capacity','energy_now','energy_full','charge_now','charge_full','power_now'):
            try:row[name]=(device/name).read_text().strip()
            except OSError:pass
        supplies.append(row)
    batteries=[s for s in supplies if s.get('type')=='Battery' and s.get('scope')!='Device']
    main=next((s for s in batteries if int(s.get('energy_full','0'))>0 or int(s.get('charge_full','0'))>0),batteries[0] if batteries else {})
    percent=None
    try:
        if 'capacity' in main:percent=float(main['capacity'])
        else:
            for kind in ('energy','charge'):
                full=float(main.get(kind+'_full',0))
                if full>0:
                    percent=100*float(main[kind+'_now'])/full
                    break
    except (ValueError,KeyError):pass
    try:power_now=int(main['power_now'])
    except (ValueError,KeyError):power_now=None
    return {'external_online':any(s.get('type') in ('Mains','USB','USB_C','USB_PD','Wireless') and s.get('online')=='1' for s in supplies),
            'battery_status':main.get('status'),'battery_percent':percent,
            'battery_device':main.get('device'),'battery_power_now_uw':power_now}

def discharge_trend(samples):
    """Require an energy decline over a minute, not a single power spike."""
    if len(samples)<2:return None
    now,state=samples[-1]
    earlier=[(t,s) for t,s in samples[:-1] if now-t>=60 and s.get('battery_percent') is not None]
    if not earlier or state.get('battery_percent') is None:return None
    then,old=earlier[-1]
    drop=old['battery_percent']-state['battery_percent']
    powers=[s['battery_power_now_uw'] for t,s in samples if t>=then and s.get('battery_power_now_uw') is not None]
    average=sum(powers)/len(powers) if powers else None
    if (average is not None and average < -2_000_000 and drop>=0.15) or (average is None and drop>=0.3):
        return f'Sustained battery energy loss despite reported charging: {drop:.2f} percentage points in {now-then:.0f}s'
    return None

def start_allowed(state=None,min_percent=30):
    state=snapshot() if state is None else state
    reason=stop_reason(state)
    if reason:return False,reason
    if state['battery_percent']<min_percent:return False,f'Waiting for {min_percent} percent battery reserve'
    return True,'External power and battery reserve verified'

def stop_reason(state=None):
    state=snapshot() if state is None else state
    if not state.get('external_online'):return 'External power is unavailable'
    if state.get('battery_status') not in ('Charging','Full','Not charging','Discharging'):
        return 'Battery state cannot be verified'
    percent=state.get('battery_percent')
    if percent is None:return 'Battery reserve telemetry is unavailable'
    if percent<=20:return 'Battery reserve reached the 20 percent stop threshold'
    return None
