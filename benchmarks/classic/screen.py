#!/usr/bin/env python3
"""Two original code tasks + five chat/tool cases on an explicit real CPU server.

Run under roofline/run.py: that parent owns lease, memory/power and temperature
policy. Model-generated code executes exclusively in the original bwrap checker.
"""
import argparse
import ctypes
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import threading
import time

HERE = Path(__file__).resolve().parent
ASSETS = HERE/'screen'
sys.path.insert(0,str(ASSETS))
import checks
from functional import functional
# Read-only utilities from the existing adapter; its source/assets stay untouched.
import normal13 as common


def source_gate():
    manifest = json.loads((ASSETS/'source-manifest.json').read_text())
    for name, expected in manifest['files'].items():
        if common.sha(HERE/name) != expected:
            raise ValueError('screen_source_changed:' + name)
    return common.sha(ASSETS/'source-manifest.json')


def request(port, path, body=None, timeout=120):
    connection = http.client.HTTPConnection('127.0.0.1',port,timeout=min(timeout,5))
    started = time.monotonic()
    timer = None
    try:
        connection.connect()
        sock = connection.sock
        sock.settimeout(timeout)
        def expire():
            try:sock.shutdown(2)
            except OSError:pass
        timer = threading.Timer(timeout,expire)
        timer.daemon = True
        timer.start()
        data = None if body is None else json.dumps(body).encode()
        connection.request('GET' if data is None else 'POST',path,body=data,headers={'Content-Type':'application/json'})
        response = connection.getresponse()
        raw = response.read(2*1024**2+1)
        if response.status != 200 or len(raw) > 2*1024**2:
            raise ValueError('bounded_successful_response_required')
        if time.monotonic()-started >= timeout:
            raise TimeoutError('request_deadline')
        return json.loads(raw)
    finally:
        if timer:timer.cancel()
        connection.close()


def repair_feedback(oracle):
    # Original safe V6 feedback: no source lines or hidden exception values.
    reason=str(oracle.get('reason') or 'Independent checks failed')[:512]
    trace=[]
    for line in str(oracle.get('failure') or '').splitlines():
        if line=='Traceback (most recent call last):':trace.append(line)
        elif re.match(r'^\s*File \".*\", line [0-9]+, in ',line):
            match=re.search(r'line ([0-9]+), in ([A-Za-z0-9_<>]+)',line)
            if match:trace.append('  File "<independent-check>", line '+match[1]+', in '+match[2])
        else:
            match=re.match(r'^([A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception))(?::.*)?$',line)
            if match:trace.append(match[1])
    diagnostic='\n'.join(trace)[:1536] or 'No safe traceback detail is available.'
    return ('Your implementation failed the independent checks below. Re-read the original requirements and return corrected Python code only, without markdown.\n'
            'Independent check result: '+reason+'\nFailure traceback (source lines and exception values omitted):\n'+diagnostic)


def code_case(name, prompt, tests, directory, api):
    row = dict(case=name,passed=False,first_pass_passed=False,repair_attempted=False,attempts=[])
    messages = [{'role':'user','content':prompt}]
    for attempt in (1,2):
        began = time.monotonic()
        response = api('/v1/chat/completions',dict(model='classic-quota-screen',messages=messages,
            temperature=0,seed=42,max_tokens=512,stream=False,cache_prompt=False,
            chat_template_kwargs={'enable_thinking':False}),timeout=120)
        raw = json.dumps(response,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()
        if len(raw)>2*1024**2:raise ValueError('bounded_code_response_required')
        common.write_json(directory/f'{name}-attempt{attempt}-response.json',response)
        choices = response.get('choices',[])
        item = dict(attempt=attempt,passed=False,response_sha256=hashlib.sha256(raw).hexdigest(),
                    timings=response.get('timings'),usage=response.get('usage'))
        row['attempts'].append(item)
        if len(choices)!=1:raise ValueError('single_complete_code_choice_required')
        choice = choices[0]
        message = choice.get('message',{})
        code = message.get('content')
        if choice.get('finish_reason')!='stop' or not isinstance(code,str) or not code.strip() or message.get('tool_calls'):
            raise ValueError('complete_code_response_required')
        # This exact copied checker AST-filters and runs ONLY in its own bwrap.
        oracle = checks.code_check(code,name,tests,directory)
        item.update(answer_sha256=hashlib.sha256(code.encode()).hexdigest(),oracle=oracle,
                    passed=oracle.get('passed') is True,request_and_oracle_seconds=time.monotonic()-began)
        if item['passed']:
            row.update(passed=True,first_pass_passed=attempt==1)
            break
        if attempt==2 or oracle.get('passed') is not False:
            break
        row['repair_attempted'] = True
        messages = [{'role':'user','content':prompt},{'role':'assistant','content':code},
                    {'role':'user','content':repair_feedback(oracle)}]
    row['repaired_passed'] = row['passed'] and row['repair_attempted']
    return row


def server_args(profile, output, port):
    args = [profile['server_binary'],'--model',profile['model_path'],'--alias','classic-quota-screen',
            '--host','127.0.0.1','--port',str(port),'--ctx-size',str(profile['context_size']),
            '--threads',str(profile['threads']),'--n-gpu-layers','0',*profile['extra_args']]
    if profile.get('prefix_cache',{}).get('enabled'):
        (output/'prefixes').mkdir(mode=0o700)
        args += ['--slots','--slot-save-path',str(output/'prefixes')]
    return args



def reap_remaining():
    """Drain only this dedicated driver's children after cancellation/errors."""
    children = Path(f'/proc/self/task/{os.getpid()}/children').read_text().split()
    for value in children:
        try:os.kill(int(value),signal.SIGKILL)
        except ProcessLookupError:pass
    until = time.monotonic()+3
    while True:
        try:pid,_ = os.waitpid(-1,os.WNOHANG)
        except ChildProcessError:return True
        if not pid:
            if time.monotonic() >= until:return False
            time.sleep(.02)


def run(profile_path, output):
    sources = source_gate()
    raw = profile_path.read_bytes()
    profile = json.loads(raw)
    if profile.get('gpu_layers',0)!=0 or any(type(profile.get(k)) is not int or profile[k]<=0 for k in ('threads','context_size')):
        raise ValueError('explicit_cpu_profile_required')
    identities = {key:common.identity(profile[key]) for key in ('model_path','server_binary')}
    if identities['model_path']['bytes']!=profile['expected_bytes'] or common.sha(profile['server_binary'])!=profile['server_sha256']:
        raise ValueError('profile_identity_mismatch')
    if not output.is_absolute() or '..' in output.parts or any(p.is_symlink() for p in (output,*output.parents)):
        raise ValueError('fresh_absolute_output_required')
    output.mkdir(mode=0o700,exist_ok=False)
    for name in ('home','config','data','cache','state'):(output/name).mkdir(mode=0o700)
    common.write_json(output/'effective-profile.json',profile)
    report = dict(kind='real-classic-screen',passed=False,screen_passed=False,cases=[],functional=[],
        source_manifest_sha256=sources,profile_sha256=hashlib.sha256(raw).hexdigest(),
        engine_sha256=profile['server_sha256'],model_identity=identities['model_path'],
        model_payload_read_by_driver=False,production_changed=False,
        prefix_slots_only=bool(profile.get('prefix_cache',{}).get('enabled')))
    child = reader = None
    began = time.monotonic()
    def interrupted(number,_):raise RuntimeError('owned_trial_signal_'+str(number))
    previous = {n:signal.signal(n,interrupted) for n in (signal.SIGTERM,signal.SIGINT)}
    try:
        if ctypes.CDLL(None).prctl(36,1,0,0,0):raise ValueError('subreaper_required')
        with socket.socket() as probe:
            probe.bind(('127.0.0.1',0))
            port = probe.getsockname()[1]
        environment = dict(os.environ,LD_LIBRARY_PATH=str(Path(profile['server_binary']).parent),HOME=str(output/'home'),XDG_CONFIG_HOME=str(output/'config'),
            XDG_DATA_HOME=str(output/'data'),XDG_CACHE_HOME=str(output/'cache'),XDG_STATE_HOME=str(output/'state'))
        environment.update(profile['environment'])
        child = subprocess.Popen(server_args(profile,output,port),stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,stderr=subprocess.STDOUT,env=environment,start_new_session=True,umask=0o077)
        reader = threading.Thread(target=common.capture,args=(child.stdout,output/'server.log'),daemon=True)
        reader.start()
        api = lambda path,body=None,timeout=120:request(port,path,body,timeout)
        while True:
            if child.poll() is not None:raise RuntimeError('server_exited_during_load')
            try:
                if api('/health',timeout=2).get('status')=='ok':break
            except (OSError,http.client.HTTPException,ValueError):pass
            if time.monotonic()-began>180:raise TimeoutError('load_deadline')
            time.sleep(.25)
        (output/'native-ready').write_text('LOCAL_AI_MODEL_READY\n')
        report['load_seconds'] = time.monotonic()-began
        for name,prompt,tests in checks.CODE_CASES:
            if name in ('merge_intervals','parse_port'):
                report['cases'].append(code_case(name,prompt,tests,output,api))
        functional(report,api,output,'screen')
        report['screen_passed'] = (len(report['cases'])==2 and all(r.get('passed') is True for r in report['cases']) and
                                   len(report['functional'])==5 and all(r.get('passed') is True for r in report['functional']))
    except Exception as error:
        report['error'] = type(error).__name__+':'+str(error)
    finally:
        if child is not None and child.poll() is None:
            try:os.killpg(child.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:os.killpg(child.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                child.wait(timeout=3)
        if reader:reader.join(5)
        report['all_children_reaped'] = reap_remaining()
        report.update(native_reaped=child is not None and child.poll() is not None,
            reader_joined=reader is not None and not reader.is_alive(),native_returncode=None if child is None else child.returncode)
        for number,handler in previous.items():signal.signal(number,handler)
        try:
            report['post_source_pins_passed'] = source_gate()==sources
            report['profile_unchanged'] = profile_path.read_bytes()==raw
            report['model_engine_identity_unchanged'] = all(common.identity(profile[k])==v for k,v in identities.items())
        except (OSError,ValueError):report['post_source_pins_passed']=False
        report['passed'] = report['screen_passed'] and 'error' not in report and report['native_returncode'] in (0,-15) and all(report.get(k) is True for k in
            ('native_reaped','reader_joined','all_children_reaped','post_source_pins_passed','profile_unchanged','model_engine_identity_unchanged'))
        report['screen_passed'] = report['passed']
        report['wall_seconds'] = time.monotonic()-began
        common.write_json(output/'summary.json',report)
    print(json.dumps(report),flush=True)
    return 0 if report['passed'] else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    return run(args.profile,args.output)


if __name__=='__main__':raise SystemExit(main())
