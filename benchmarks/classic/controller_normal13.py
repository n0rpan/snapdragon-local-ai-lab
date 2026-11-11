#!/usr/bin/env python3
"""Protected OpenCode normal13 against an already-running staged controller.

Starts only bridges and a private bwrap client namespace. The controller owns
native startup, memory guarding, canonical lease, unload and RAM recovery.
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
from urllib.parse import urlsplit
import normal13 as common
import loopback_bridge

HERE=Path(__file__).resolve().parent
ASSETS=HERE/'controller-normal13'


def source_gate():
    manifest=json.loads((ASSETS/'source-manifest.json').read_text())
    for name,expected in manifest['files'].items():
        if common.sha(HERE/name)!=expected:raise ValueError('controller_client_source_changed:'+name)
    return common.sha(ASSETS/'source-manifest.json')


def controller_port(url):
    parsed=urlsplit(url)
    if (parsed.scheme!='http' or parsed.hostname!='127.0.0.1' or parsed.username is not None or
            parsed.password is not None or parsed.path not in ('','/') or parsed.query or parsed.fragment or
            parsed.port is None or not 1<=parsed.port<=65535):
        raise ValueError('explicit_http_ipv4_loopback_controller_required')
    return parsed.port


def socket_identity(path):
    loopback_bridge.owned_directory(path)
    info=Path(path).lstat()
    if not stat.S_ISSOCK(info.st_mode) or info.st_uid!=os.getuid() or info.st_mode&0o077:
        raise ValueError('owned_private_controller_socket_required')
    return (info.st_dev,info.st_ino,info.st_uid,info.st_mode)


def captured_requests(directory):
    """Read only the synthetic client's captured request direction, outside the model process."""
    rows = []
    for path in sorted(directory.glob('request-*.http')):
        raw = path.read_bytes()
        if len(raw) > 4 * 1024**2:
            raise ValueError('request_capture_size_invalid')
        offset = 0
        while offset < len(raw):
            boundary = raw.find(b'\r\n\r\n', offset)
            if boundary < 0:
                raise ValueError('incomplete_captured_request_header')
            lines = raw[offset:boundary].split(b'\r\n')
            method, target, _ = lines[0].decode('ascii').split()
            headers = dict(line.decode('latin1').split(':', 1) for line in lines[1:])
            headers = {key.lower(): value.strip() for key, value in headers.items()}
            if headers.get('transfer-encoding'):
                raise ValueError('chunked_request_capture_requires_explicit_decoder')
            size = int(headers.get('content-length', '0'))
            if size < 0:
                raise ValueError('negative_captured_request_length')
            start = boundary + 4
            end = start + size
            if end > len(raw):
                raise ValueError('incomplete_captured_request_body')
            body = raw[start:end]
            offset = end
            if method != 'POST' or target != '/v1/chat/completions':
                continue
            request = json.loads(body)
            system = '\n'.join(row.get('content', '') for row in request.get('messages', [])
                               if row.get('role') == 'system' and isinstance(row.get('content'), str))
            dates = [line.strip() for line in system.splitlines() if "Today's date:" in line]
            rows.append(dict(file=path.name, body_sha256=hashlib.sha256(body).hexdigest(),
                             bytes=size, system_dates=dates,
                             tools=[tool.get('function', {}).get('name') for tool in request.get('tools', [])]))
    if not rows or not all(row['system_dates'] for row in rows):
        raise ValueError('captured_system_date_missing')
    return rows


def prepare(output,profile):
    fixture=common.prepare(output,profile)
    alias=profile.get('model_alias')
    if not isinstance(alias,str) or not alias or any(c in alias for c in '\r\n\0/'):
        raise ValueError('explicit_controller_model_alias_required')
    configured=json.loads((output/'profile.json').read_text())
    configured['model_alias']=alias
    common.write_json(output/'profile.json',configured)
    config=json.loads((output/'opencode.json').read_text())
    provider=config['providers']['local-qwen']
    metadata=provider['models']['comparison']
    provider['models']={alias:metadata}
    config['model']='local-qwen/'+alias
    common.write_json(output/'opencode.json',config)
    (output/'inside_client.py').write_bytes((ASSETS/'inside_client.py').read_bytes())
    (output/'loopback_bridge.py').write_bytes((HERE/'loopback_bridge.py').read_bytes())
    return fixture


def sandbox_command(output,profile,fixture,opencode,socket_path):
    # Retain all original isolation/protected-test bindings. No engine/model is
    # required for this client, so strip precisely their original bind tuples.
    original=common.sandbox_command(output,profile,fixture,opencode)
    command=[]
    i=0
    while i<len(original):
        if original[i:i+2]==['--dir','/engine']:
            i+=2
        elif original[i]=='--ro-bind' and i+2<len(original) and original[i+2] in ('/engine','/model.gguf'):
            i+=3
        else:
            command.append(original[i]);i+=1
    if command[-1]!='/work/inside.py':raise ValueError('unsupported_original_sandbox_command')
    command[-1]='/work/inside_client.py'
    position=command.index('--chdir')
    return command[:position]+['--ro-bind',str(socket_path),'/work/controller.sock']+command[position:]


def stop(child):
    if child is not None and child.poll() is None:
        try:os.killpg(child.pid,signal.SIGTERM)
        except ProcessLookupError:pass
        try:child.wait(timeout=3)
        except subprocess.TimeoutExpired:
            try:os.killpg(child.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            child.wait(timeout=3)


def drain():
    # Dedicated driver is a subreaper; namespace death kills its workers.
    until=time.monotonic()+3
    while True:
        try:pid,_=os.waitpid(-1,os.WNOHANG)
        except ChildProcessError:return True
        if not pid:
            if time.monotonic()>=until:return False
            time.sleep(.02)


def run(profile_path,output,*,controller=None,socket_path=None,opencode=None,capture_requests=False):
    source_sha=source_gate()
    raw=profile_path.read_bytes()
    profile=json.loads(raw)
    if (controller is None)==(socket_path is None):raise ValueError('choose_controller_or_owned_socket')
    port=controller_port(controller) if controller is not None else None
    if not output.is_absolute() or '..' in output.parts or any(p.is_symlink() for p in (output,*output.parents)):
        raise ValueError('fresh_absolute_output_required')
    pinned=json.loads((common.ASSETS/'opencode-binary.expected.json').read_text())
    opencode = common.opencode_path(opencode, pinned)
    opencode_identity=common.identity(opencode)
    if common.sha(opencode)!=pinned['sha256']:raise ValueError('qualified_opencode_changed')
    output.mkdir(mode=0o700,exist_ok=False)
    fixture=prepare(output,profile)
    if capture_requests:
        if port is None:
            raise ValueError('request_capture_requires_owned_bridge')
        (output/'requests').mkdir(mode=0o700)
    prepared={name:common.sha(output/name) for name in ('inside.py','oracle.py','bounded_log.py','fixture.json',
        'profile.json','opencode.json','inside_client.py','loopback_bridge.py')}
    report=dict(kind='staged-controller-opencode-normal13',passed=False,model_alias=profile['model_alias'],
        native_server_started=False,canonical_lease_acquired=False,
        controller_native_cleanup_owned=False,source_manifest_sha256=source_sha,
        profile_sha256=hashlib.sha256(raw).hexdigest(),opencode_sha256=pinned['sha256'],
        declared_engine_sha256=profile.get('server_sha256'),declared_model_sha256=profile.get('sha256'),
        model_payload_read_by_driver=False)
    child=reader=host=host_reader=None
    socket_directory=None
    began=time.monotonic()
    def interrupted(number,_):raise RuntimeError('owned_client_signal_'+str(number))
    previous={n:signal.signal(n,interrupted) for n in (signal.SIGTERM,signal.SIGINT)}
    try:
        if ctypes.CDLL(None).prctl(36,1,0,0,0):raise ValueError('subreaper_required')
        if port is not None:
            # AF_UNIX paths have a small kernel limit; output paths may be long.
            socket_directory=tempfile.TemporaryDirectory(prefix='local-ai-controller-')
            socket_path=Path(socket_directory.name)/'controller.sock'
            host_args=[sys.executable,'-B',str(HERE/'loopback_bridge.py'),'host','--socket',str(socket_path),
                '--target-host','127.0.0.1','--target-port',str(port)]
            if capture_requests:
                host_args += ['--capture-dir', str(output/'requests')]
            host=subprocess.Popen(host_args,stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,start_new_session=True)
            host_reader=threading.Thread(target=common.capture,args=(host.stdout,output/'host-bridge.log'),daemon=True)
            host_reader.start()
            until=time.monotonic()+5
            while not socket_path.exists():
                if host.poll() is not None:raise RuntimeError('host_bridge_exited')
                if time.monotonic()>until:raise TimeoutError('host_bridge_deadline')
                time.sleep(.05)
        before_socket=socket_identity(socket_path)
        child=subprocess.Popen(sandbox_command(output,profile,fixture,opencode,socket_path),stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,start_new_session=True)
        reader=threading.Thread(target=common.capture,args=(child.stdout,output/'sandbox.log'),daemon=True)
        reader.start()
        child.wait(timeout=930)
        result=output/'controller-result.json'
        if result.stat().st_size>8*1024**2:raise ValueError('bounded_controller_result_required')
        inner=json.loads(result.read_text())
        agent=inner.get('agent',{})
        report.update(normal13_passed=inner.get('passed') is True,
            normal13_cases=agent.get('independent_test_cases_run'),task_seconds=agent.get('seconds'),
            inside_bridge_reaped=inner.get('inside_bridge_reaped'),inside_bridge_reader_joined=inner.get('inside_bridge_reader_joined'),
            checks={k:agent.get(k) for k in ('passed','complete','expected_test_cases','independent_test_cases_run',
                'opencode_exit','test_exit','completed_tools','test_commands_completed','protected_unchanged',
                'changed_modules','unexpected_files','readers_joined')})
        report['socket_identity_unchanged']=socket_identity(socket_path)==before_socket
        report['passed']=child.returncode==0 and report['normal13_passed']
    except Exception as error:report['error']=type(error).__name__+':'+str(error)
    finally:
        stop(child)
        stop(host)
        for worker in (reader,host_reader):
            if worker:worker.join(5)
        if capture_requests:
            try:
                report['captured_requests'] = captured_requests(output/'requests')
            except (OSError,ValueError) as error:
                report['error'] = type(error).__name__+':'+str(error)
        report['client_namespace_reaped']=child is not None and child.poll() is not None
        report['owned_host_bridge_reaped']=host is None or host.poll() is not None
        report['client_reader_joined']=reader is not None and not reader.is_alive()
        report['host_bridge_reader_joined']=host_reader is None or not host_reader.is_alive()
        report['all_client_children_reaped']=drain()
        if socket_directory:socket_directory.cleanup()
        for number,handler in previous.items():signal.signal(number,handler)
        try:
            report['post_source_pins_passed']=source_gate()==source_sha
            report['prepared_inputs_unchanged']=all(common.sha(output/name)==expected for name,expected in prepared.items())
            report['profile_unchanged']=profile_path.read_bytes()==raw
            report['opencode_identity_unchanged']=common.identity(opencode)==opencode_identity
            report['protected_test_sha256']=common.sha(output/'project/test_port.py')
            report['protected_unchanged']=report['protected_test_sha256']==fixture['protected']['test_port.py']
        except (OSError,ValueError):report['post_source_pins_passed']=False
        report['passed']=report['passed'] and 'error' not in report and all(report.get(k) is True for k in
            ('client_namespace_reaped','owned_host_bridge_reaped','client_reader_joined','host_bridge_reader_joined',
             'all_client_children_reaped','post_source_pins_passed','prepared_inputs_unchanged','profile_unchanged',
             'opencode_identity_unchanged','protected_unchanged','socket_identity_unchanged'))
        report['wall_seconds']=time.monotonic()-began
        common.write_json(output/'summary.json',report)
    print(json.dumps(report),flush=True)
    return 0 if report['passed'] else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    target=parser.add_mutually_exclusive_group(required=True)
    target.add_argument('--controller')
    target.add_argument('--socket',type=Path)
    parser.add_argument('--opencode',type=Path)
    parser.add_argument('--capture-requests',action='store_true')
    args=parser.parse_args()
    return run(args.profile,args.output,controller=args.controller,socket_path=args.socket,opencode=args.opencode,
               capture_requests=args.capture_requests)


if __name__=='__main__':raise SystemExit(main())
