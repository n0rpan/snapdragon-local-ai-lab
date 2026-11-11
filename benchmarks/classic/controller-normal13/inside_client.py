"""Existing protected OpenCode caller against a controller through a Unix bridge."""
import json
from pathlib import Path
import signal
import subprocess
import threading
import time
import inside as original

ROOT=Path('/work')


def agent_passed(agent):
    exact=dict(expected_test_cases=13,independent_test_cases_run=13,changed_modules=['parse_port.py'],
               unexpected_files=[],opencode_exit=0,test_exit=0)
    return (all(agent.get(k) is True for k in ('passed','protected_unchanged','complete','readers_joined')) and
            all(agent.get(k)==v and type(agent.get(k)) is type(v) for k,v in exact.items()) and
            type(agent.get('completed_tools')) is int and agent['completed_tools']>0 and
            type(agent.get('test_commands_completed')) is int and agent['test_commands_completed']>0)


def main():
    profile=json.loads((ROOT/'profile.json').read_text())
    report=dict(passed=False,native_server_started=False,canonical_lease_acquired=False)
    bridge=reader=None
    began=time.monotonic()
    try:
        bridge=subprocess.Popen(['/usr/bin/python3','-B','/work/loopback_bridge.py','inside',
                                 '--socket','/work/controller.sock','--port','18096'],
                                stdout=subprocess.PIPE,stderr=subprocess.STDOUT,start_new_session=True)
        reader=threading.Thread(target=original.capture,args=(bridge.stdout,ROOT/'inside-bridge.log'),daemon=True)
        reader.start()
        until=time.monotonic()+15
        while True:
            if bridge.poll() is not None:raise RuntimeError('namespace_bridge_exited')
            try:
                # Catalog polling does not load the controller's native backend.
                catalog,_=original.api('/v1/models',timeout=2)
                aliases={row.get('id') for row in catalog.get('data',[]) if isinstance(row,dict)}
                if profile['model_alias'] not in aliases:raise RuntimeError('requested_controller_model_not_listed')
                break
            except (OSError,ValueError):
                if time.monotonic()>until:raise TimeoutError('namespace_bridge_deadline')
                time.sleep(.1)
        report['bridge_ready_seconds']=time.monotonic()-began
        report['agent']=original.run_agent(json.loads((ROOT/'fixture.json').read_text()))
        report['passed']=agent_passed(report['agent'])
    except Exception as error:
        report['error']=type(error).__name__+':'+str(error)
    finally:
        if bridge is not None and bridge.poll() is None:
            bridge.terminate()
            try:bridge.wait(timeout=3)
            except subprocess.TimeoutExpired:bridge.kill();bridge.wait(timeout=3)
        if reader:reader.join(5)
        report.update(inside_bridge_reaped=bridge is not None and bridge.poll() is not None,
                      inside_bridge_reader_joined=reader is not None and not reader.is_alive(),
                      wall_seconds=time.monotonic()-began)
        report['passed']=report['passed'] and report['inside_bridge_reaped'] and report['inside_bridge_reader_joined']
        (ROOT/'controller-result.json').write_text(json.dumps(report,indent=2)+'\n')
    return 0 if report['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
