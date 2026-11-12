#!/usr/bin/python3
"""Stop the bound SCMI memlat client before warm reboot; never touch other drivers."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

DRIVER = 'scmi-qcom-devfreq-memlat'
CLIENT = 'qcom-generic-ext'
KERNEL_SHUTDOWN_UNITS = frozenset(
    [f'{kind}.target' for kind in ('reboot', 'poweroff', 'halt', 'kexec')]
    + [f'systemd-{kind}.service' for kind in ('reboot', 'poweroff', 'halt', 'kexec')])
SOFT_REBOOT_UNITS = frozenset(('soft-reboot.target', 'systemd-soft-reboot.service'))


def discover(root=Path('/sys')):
    bus = root/'bus/scmi_protocol'
    devices = bus/'devices'
    matches = []
    if devices.exists():
        for device in devices.iterdir():
            try:
                if ((device/'protocol_id').read_text().strip() == '0x80'
                        and (device/'name').read_text().strip() == CLIENT):
                    matches.append(device)
            except FileNotFoundError:
                continue
    if len(matches) > 1:
        raise RuntimeError('multiple vendor-protocol clients; refusing ambiguous teardown')
    driver = bus/'drivers'/DRIVER
    device = matches[0] if matches else None
    if device is not None and (device/'driver').exists():
        if (device/'driver').resolve() != driver.resolve():
            raise RuntimeError('protocol client is bound to a different driver')
    return driver, device


def bound(driver, device):
    return device is not None and (device/'driver').exists() and (device/'driver').resolve() == driver.resolve()


def system_state():
    # Nonzero is normal for "stopping"; an unavailable PID1 is NOT authorization.
    result = subprocess.run(['/usr/bin/systemctl', 'is-system-running'], capture_output=True,
                            text=True, timeout=2, check=False)
    return result.stdout.strip()


def system_jobs():
    # Use PID1's typed ListJobs API rather than parsing a human table.
    result = subprocess.run(['/usr/bin/busctl', '--system', '--json=short', '--timeout=2',
        'call', 'org.freedesktop.systemd1', '/org/freedesktop/systemd1',
        'org.freedesktop.systemd1.Manager', 'ListJobs'], capture_output=True,
        text=True, timeout=3, check=True)
    response = json.loads(result.stdout)
    if (not isinstance(response, dict) or response.get('type') != 'a(usssoo)' or not isinstance(response.get('data'), list)
            or len(response['data']) != 1 or not isinstance(response['data'][0], list)):
        raise ValueError('unexpected PID1 job schema')
    jobs = []
    for row in response['data'][0]:
        if (not isinstance(row, list) or len(row) != 6 or type(row[0]) is not int
                or not all(isinstance(value, str) for value in row[1:])):
            raise ValueError('unexpected PID1 job row')
        jobs.append((row[1], row[2]))
    return jobs


def control(action, root=Path('/sys'), *, write=None, clock=time.monotonic, sleep=time.sleep,
            state=system_state, jobs=system_jobs):
    if action == 'shutdown':
        try:
            if state() != 'stopping':
                return {'action': 'shutdown', 'state': 'not-shutting-down', 'changed': False}
            queued = jobs()
            starts = {unit for unit, kind in queued if kind == 'start'}
        except (OSError, subprocess.SubprocessError, ValueError, TypeError) as error:
            return {'action': 'shutdown', 'state': 'query-failed',
                    'error_type': type(error).__name__, 'changed': False}
        if starts & SOFT_REBOOT_UNITS:
            return {'action': 'shutdown', 'state': 'soft-reboot', 'changed': False}
        if not starts & KERNEL_SHUTDOWN_UNITS:
            return {'action': 'shutdown', 'state': 'no-kernel-shutdown-job', 'changed': False}
        action = 'unbind'
    driver, device = discover(root)
    if device is None:
        if action == 'bind':
            raise RuntimeError('no memlat protocol client exists on this boot')
        return {'action': action, 'state': 'absent', 'changed': False}
    attached = bound(driver, device)
    result = {'action': action, 'device': device.name, 'protocol': '0x80',
              'driver': DRIVER, 'state': 'bound' if attached else 'unbound', 'changed': False}
    if action == 'status' or attached == (action == 'bind'):
        return result
    if write is None:
        if os.geteuid() != 0:
            raise PermissionError('root is required for the exact memlat bind/unbind')
        write = lambda path, value: path.write_text(value)
    write(driver/action, device.name)
    deadline = clock()+5
    while bound(driver, device) != (action == 'bind'):
        if clock() >= deadline:
            raise RuntimeError('memlat binding did not reach the requested state')
        sleep(.05)
    result.update(state='bound' if action == 'bind' else 'unbound', changed=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    # Deliberately expose NO runtime unbind: live rebind returns EBUSY on the tested firmware.
    parser.add_argument('action', choices=('status', 'shutdown'))
    args = parser.parse_args()
    print(json.dumps(control(args.action), sort_keys=True))
