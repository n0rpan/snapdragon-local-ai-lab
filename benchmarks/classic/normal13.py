#!/usr/bin/env python3
"""Thin offline real-server/OpenCode normal13 adapter; run under roofline/run.py.

Owns only a fresh private output directory and its bwrap process tree. The outer
runner owns the inference lease, memory/power policy and thermal observations.
"""
import argparse
import copy
import ctypes
import hashlib
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import threading
import time

HERE = Path(__file__).resolve().parent
ASSETS = HERE / 'normal13'
sys.path.insert(0, str(ASSETS))
from bounded_log import BuildLog


def sha(path):
    check = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while data := stream.read(1024**2):
            check.update(data)
    return check.hexdigest()


def opencode_path(explicit, pinned):
    if explicit is not None:
        return Path(explicit)
    path = pinned['path']
    if path.startswith('$HOME/'):
        return Path.home() / path.removeprefix('$HOME/')
    return Path(path)


def identity(path):
    info = Path(path).lstat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError('regular_file_required')
    return dict(bytes=info.st_size,device=info.st_dev,inode=info.st_ino,
                mtime_ns=info.st_mtime_ns,ctime_ns=info.st_ctime_ns)


def source_gate():
    manifest = json.loads((ASSETS/'source-manifest.json').read_text())
    for name, expected in manifest['files'].items():
        if sha(HERE/name) != expected:
            raise ValueError('normal13_source_changed:' + name)
    return sha(ASSETS/'source-manifest.json')


def write_json(path, value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def prepare(output, profile):
    """Existing protected task and oracle bytes stay unchanged."""
    for name in ('home','config','data','cache','state','project'):
        (output/name).mkdir(mode=0o700)
    for name in ('inside.py','oracle.py','bounded_log.py'):
        (output/name).write_bytes((ASSETS/name).read_bytes())
    for name in ('parse_port.py','test_port.py'):
        (output/'project'/name).write_bytes((ASSETS/'fixtures'/name).read_bytes())
    fixture = dict(prompt=(ASSETS/'fixtures/TASK.txt').read_text(),test='test_port.py',
                   initial={'parse_port.py':sha(output/'project/parse_port.py')},
                   protected={'test_port.py':sha(output/'project/test_port.py')})
    write_json(output/'fixture.json',fixture)
    write_json(output/'profile.json',dict(stage='normal-agent',binary_name=Path(profile['server_binary']).name,
        threads=profile['threads'],context_size=profile['context_size'],extra_args=profile['extra_args'],
        environment=profile['environment'],prefix_cache=profile.get('prefix_cache',{'enabled':False})))
    config = json.loads((ASSETS/'opencode-base.synthetic.json').read_text())
    provider = config['providers']['local-qwen']
    provider['settings'].update(baseURL='http://127.0.0.1:18096/v1',apiKey='local-only',headerTimeout=900000,chunkTimeout=900000)
    provider['settings']['headers'] = {'X-Local-AI-Origin': 'lab'}
    metadata = copy.deepcopy(provider['models']['qwen3.6-35b-a3b-local'])
    context = profile['context_size']
    metadata.update(name='Private classic normal13 task',limit={'context':context,'input':min(12288,context-2048),'output':2048})
    provider['models'] = {'comparison':metadata}
    provider.setdefault('body',{}).update(temperature=0,cache_prompt=True,chat_template_kwargs={'enable_thinking':False})
    config.update(model='local-qwen/comparison',providers={'local-qwen':provider},update='disable',warming=False)
    config.setdefault('agents',{})['title'] = {'disabled':True}
    config['policies'] = [{'action':'provider.use','resource':'*','effect':'deny'},
                          {'action':'provider.use','resource':'local-qwen','effect':'allow'}]
    write_json(output/'opencode.json',config)
    return fixture


def sandbox_command(output, profile, fixture, opencode):
    command = ['/usr/bin/bwrap','--unshare-all','--die-with-parent','--new-session','--cap-drop','ALL',
        '--clearenv','--ro-bind','/usr','/usr','--symlink','usr/lib','/lib','--symlink','usr/bin','/bin',
        '--proc','/proc','--dev','/dev','--tmpfs','/tmp','--dir','/engine',
        '--ro-bind',str(Path(profile['server_binary']).parent),'/engine',
        '--ro-bind',profile['model_path'],'/model.gguf','--bind',str(output),'/work',
        '--ro-bind',str(opencode),'/opencode']
    for name in fixture['protected']:
        command += ['--ro-bind',str(output/'project'/name),'/work/project/'+name]
    environment = dict(HOME='/work/home',PATH='/usr/bin',SHELL='/usr/bin/bash',LANG='C.UTF-8',LC_ALL='C.UTF-8',
        XDG_CONFIG_HOME='/work/config',XDG_DATA_HOME='/work/data',XDG_CACHE_HOME='/work/cache',
        XDG_STATE_HOME='/work/state',OPENCODE_CONFIG='/work/opencode.json')
    for key,value in environment.items():
        command += ['--setenv',key,value]
    return command + ['--chdir','/work','/usr/bin/python3','-B','/work/inside.py']


def agent_gate(report):
    agent = report.get('agent',{})
    exact = dict(expected_test_cases=13,independent_test_cases_run=13,changed_modules=['parse_port.py'],
                 unexpected_files=[],opencode_exit=0,test_exit=0)
    return (report.get('passed') is True and report.get('server_reaped') is True and
        report.get('server_reader_joined') is True and
        all(agent.get(k) is True for k in ('passed','protected_unchanged','complete','readers_joined')) and
        all(agent.get(k) == v and type(agent.get(k)) is type(v) for k,v in exact.items()) and
        type(agent.get('completed_tools')) is int and agent['completed_tools'] > 0 and
        type(agent.get('test_commands_completed')) is int and agent['test_commands_completed'] > 0)


def capture(stream, destination):
    with destination.open('wb') as handle:
        log = BuildLog(handle)
        while data := stream.read(8192):
            log.append(data)
    stream.close()


def run(profile_path, output, opencode=None, timeout=1140):
    source_sha = source_gate()
    raw = profile_path.read_bytes()
    profile = json.loads(raw)
    if any(type(profile.get(k)) is not int or profile[k] <= 0 for k in ('threads','context_size')) or profile['context_size'] <= 2048:
        raise ValueError('explicit_valid_threads_and_context_required')
    if profile.get('gpu_layers',0) != 0:
        raise ValueError('cpu_only_adapter')
    identities = {key:identity(profile[key]) for key in ('model_path','server_binary')}
    if identities['model_path']['bytes'] != profile['expected_bytes'] or sha(profile['server_binary']) != profile['server_sha256']:
        raise ValueError('profile_identity_mismatch')
    pinned = json.loads((ASSETS/'opencode-binary.expected.json').read_text())
    opencode = opencode_path(opencode, pinned)
    opencode_identity = identity(opencode)
    if sha(opencode) != pinned['sha256']:
        raise ValueError('qualified_opencode_identity_changed')
    if not output.is_absolute() or '..' in output.parts or any(p.is_symlink() for p in (output,*output.parents)):
        raise ValueError('fresh_absolute_output_required')
    output.mkdir(mode=0o700,exist_ok=False)
    fixture = prepare(output,profile)
    prepared = {name:sha(output/name) for name in ('inside.py','oracle.py','bounded_log.py',
                'fixture.json','profile.json','opencode.json')}
    write_json(output/'effective-profile.json',profile)
    result = dict(kind='real-opencode-normal13',passed=False,profile_sha256=hashlib.sha256(raw).hexdigest(),
                  source_manifest_sha256=source_sha,engine_sha256=profile['server_sha256'],
                  opencode_sha256=pinned['sha256'],model_identity=identities['model_path'],
                  model_payload_read_by_driver=False,production_changed=False,
                  prefix_slots_only=bool(profile.get('prefix_cache',{}).get('enabled')))
    child = reader = None
    started = time.monotonic()
    def interrupted(number, _):
        raise RuntimeError('owned_trial_signal_' + str(number))
    previous = {n:signal.signal(n,interrupted) for n in (signal.SIGTERM,signal.SIGINT)}
    try:
        if ctypes.CDLL(None).prctl(36,1,0,0,0):
            raise ValueError('subreaper_required')
        child = subprocess.Popen(sandbox_command(output,profile,fixture,opencode),stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT,start_new_session=True)
        reader = threading.Thread(target=capture,args=(child.stdout,output/'sandbox.log'),daemon=True)
        reader.start()
        child.wait(timeout=timeout)
        path = output/'result.json'
        if path.stat().st_size > 8*1024**2:
            raise ValueError('bounded_inner_report_required')
        measurement = json.loads(path.read_text())
        agent = measurement.get('agent',{})
        result.update(normal13_passed=agent_gate(measurement),normal13_cases=agent.get('independent_test_cases_run'),
                      protected_unchanged=agent.get('protected_unchanged'),
                      native_reaped=measurement.get('server_reaped'),
                      readers_joined=measurement.get('server_reader_joined') is True and agent.get('readers_joined') is True,
                      checks={k:agent.get(k) for k in ('complete','expected_test_cases','independent_test_cases_run',
                        'opencode_exit','test_exit','completed_tools','test_commands_completed','protected_unchanged',
                        'changed_modules','unexpected_files','readers_joined')})
        result['passed'] = child.returncode == 0 and result['normal13_passed']
    except Exception as error:
        result['error'] = type(error).__name__ + ':' + str(error)
    finally:
        if child is not None and child.poll() is None:
            try:os.killpg(child.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:child.wait(timeout=3)
            except subprocess.TimeoutExpired:
                try:os.killpg(child.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                child.wait(timeout=3)
        if reader is not None:
            reader.join(5)
        result['owned_native_reaped'] = child is not None and child.poll() is not None
        result['reader_joined'] = reader is not None and not reader.is_alive()
        until = time.monotonic()+3
        while True:
            try:
                pid,_ = os.waitpid(-1,os.WNOHANG)
            except ChildProcessError:
                result['all_children_reaped'] = True
                break
            if not pid:
                if time.monotonic() >= until:
                    result['all_children_reaped'] = False
                    break
                time.sleep(.02)
        for number,handler in previous.items():
            signal.signal(number,handler)
        try:
            result['post_source_pins_passed'] = source_gate() == source_sha
            result['prepared_inputs_unchanged'] = all(sha(output/name) == expected for name,expected in prepared.items())
            result['prepared_input_sha256'] = prepared
            result['profile_unchanged'] = profile_path.read_bytes() == raw
            result['model_engine_identity_unchanged'] = all(identity(profile[k]) == v for k,v in identities.items())
            result['opencode_identity_unchanged'] = identity(opencode) == opencode_identity
            result['protected_test_sha256'] = sha(output/'project/test_port.py')
            result['protected_unchanged'] = result.get('protected_unchanged') is True and result['protected_test_sha256'] == fixture['protected']['test_port.py']
        except (OSError,ValueError):
            result['post_source_pins_passed'] = False
        result['passed'] = result['passed'] and all(result.get(k) is True for k in
            ('owned_native_reaped','reader_joined','all_children_reaped','post_source_pins_passed',
             'profile_unchanged','model_engine_identity_unchanged','opencode_identity_unchanged','protected_unchanged',
             'prepared_inputs_unchanged'))
        result['wall_seconds'] = time.monotonic()-started
        write_json(output/'summary.json',result)
    print(json.dumps(result),flush=True)
    return 0 if result['passed'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--opencode',type=Path)
    args = parser.parse_args()
    return run(args.profile,args.output,args.opencode)


if __name__ == '__main__':
    raise SystemExit(main())
