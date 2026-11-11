"""MIT. Only synthetic work/model/engine enter this offline bwrap namespace."""
import collections,hashlib,http.client,json,os,re,selectors,signal,subprocess,threading,time
from pathlib import Path
import oracle
from bounded_log import BuildLog

ROOT=Path('/work')
PORT=18096

def terminal_message(obj,tools=False):
    choices=obj.get('choices')
    if not isinstance(choices,list) or len(choices)!=1:raise ValueError('single_choice_required')
    choice=choices[0];message=choice.get('message',{})
    expected='tool_calls' if tools else 'stop'
    if choice.get('finish_reason')!=expected:raise ValueError('terminal_completion_required')
    if tools:
        calls=message.get('tool_calls')
        if not isinstance(calls,list) or len(calls)!=1:raise ValueError('one_tool_call_required')
    elif not isinstance(message.get('content'),str) or not message['content'].strip() or message.get('tool_calls'):
        raise ValueError('final_text_required')
    return message

def api(path,body=None,timeout=180):
    connection=http.client.HTTPConnection('127.0.0.1',PORT,timeout=min(timeout,5))
    started=time.monotonic();timer=None
    try:
        connection.connect();sock=connection.sock;sock.settimeout(timeout)
        def expire():
            try:sock.shutdown(2)
            except OSError:pass
        timer=threading.Timer(timeout,expire);timer.daemon=True;timer.start()
        raw=None if body is None else json.dumps(body).encode()
        connection.request('GET' if raw is None else 'POST',path,body=raw,headers={'Content-Type':'application/json'})
        response=connection.getresponse();data=response.read(1024*1024+1)
        if response.status!=200 or len(data)>1024*1024:raise ValueError('bounded_successful_api_response_required')
        if time.monotonic()-started>=timeout:raise TimeoutError('request_deadline')
        return json.loads(data),(time.monotonic()-started)
    finally:
        if timer:timer.cancel()
        connection.close()

def chat(messages,tokens=128,tools=None,expect_tools=None):
    payload=dict(model='comparison',messages=messages,stream=False,max_tokens=tokens,temperature=0,seed=42,
                 chat_template_kwargs={'enable_thinking':False})
    if expect_tools is None:expect_tools=bool(tools)
    if tools:payload.update(tools=tools,tool_choice='auto' if expect_tools else 'none')
    obj,seconds=api('/v1/chat/completions',payload,240)
    return terminal_message(obj,tools=expect_tools),dict(seconds=seconds,usage=obj.get('usage'),timings=obj.get('timings'),finish_reason=obj['choices'][0]['finish_reason'])

def capture(stream,path,tracker=None,metadata=None):
    with path.open('wb') as output:
        log=BuildLog(output)
        while piece:=stream.read(8192):
            log.append(piece)
            if tracker:tracker.feed(piece)
        if metadata is not None:metadata.update(total_bytes=log.total,retained_bytes=path.stat().st_size,truncated=log.truncated)
    stream.close()

class Completion:
    def __init__(self):self.buffer=b'';self.steps=0;self.reason=None;self.text='';self.errors=0;self.malformed=0;self.tools=0;self.tests=0
    def feed(self,piece):
        self.buffer+=piece
        if len(self.buffer)>1024*1024:self.malformed+=1;self.buffer=b'';return
        while b'\n' in self.buffer:
            line,self.buffer=self.buffer.split(b'\n',1)
            if not line.strip():continue
            try:self.event(json.loads(line))
            except (ValueError,UnicodeError,AttributeError,TypeError):self.malformed+=1
    def event(self,value):
        kind=value.get('type');part=value.get('part',{})
        if kind=='step_start':self.steps+=1;self.reason=None;self.text=''
        elif kind=='text':self.text+=part.get('text','')
        elif kind=='step_finish':self.reason=part.get('reason')
        elif kind=='error':self.errors+=1
        elif kind=='tool_use' and part.get('state',{}).get('status')=='completed':
            self.tools+=1;state=part['state'];command=state.get('input',{}).get('command','');output=state.get('output','')
            if part.get('tool')=='shell' and ('test_port.py' in command or 'test_reservations.py' in command) and ('ALL_TESTS_PASSED' in output or ('Ran 12 tests' in output and '\nOK' in output)):self.tests+=1
    def result(self):return dict(complete=self.steps>0 and self.reason=='stop' and bool(self.text.strip()) and not self.errors and not self.malformed and not self.buffer.strip(),terminal_reason=self.reason,final_text=self.text,completed_tools=self.tools,test_commands_completed=self.tests,error_events=self.errors,malformed_events=self.malformed)

def run_agent(fixture):
    tracker=Completion();started=time.monotonic();code=None;error=None;logs={}
    child=subprocess.Popen(['/opencode','--standalone','run','--auto','--format','json','--archived',fixture['prompt']],cwd=ROOT/'project',stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
    readers=[]
    for key,stream in [('stdout',child.stdout),('stderr',child.stderr)]:
        logs[key]={}
        thread=threading.Thread(target=capture,args=(stream,ROOT/f'agent-{key}.log',tracker if key=='stdout' else None,logs[key]),daemon=True);thread.start();readers.append(thread)
    try:code=child.wait(timeout=900)
    except subprocess.TimeoutExpired:error='agent_deadline'
    finally:
        if child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=3)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=3)
        for thread in readers:thread.join(5)
    try:
        check=subprocess.run(['/usr/bin/python3',fixture['test']],cwd=ROOT/'project',capture_output=True,timeout=10)
        test_exit=check.returncode;test_output=(check.stdout+check.stderr).decode(errors='replace')
    except subprocess.TimeoutExpired as caught:
        test_exit=None;test_output=((caught.stdout or b'')+(caught.stderr or b'')).decode(errors='replace');error=error or 'independent_checker_deadline'
    project=ROOT/'project'
    protected=all((project/n).is_file() and not (project/n).is_symlink() and hashlib.sha256((project/n).read_bytes()).hexdigest()==sha for n,sha in fixture['protected'].items())
    changed=[n for n,sha in fixture['initial'].items() if (project/n).is_file() and not (project/n).is_symlink() and hashlib.sha256((project/n).read_bytes()).hexdigest()!=sha]
    expected=set(fixture['initial'])|set(fixture['protected'])
    unexpected=[str(p.relative_to(project)) for p in project.rglob('*') if (p.is_file() or p.is_symlink()) and '__pycache__' not in p.parts and str(p.relative_to(project)) not in expected]
    match=re.search(r'Ran (\d+) tests? in ',test_output)
    result=tracker.result();result.update(opencode_exit=child.returncode,test_exit=test_exit,test_output=test_output,protected_unchanged=protected,changed_modules=changed,unexpected_files=unexpected,seconds=time.monotonic()-started,readers_joined=not any(t.is_alive() for t in readers),logs=logs,error=error,
        expected_test_cases=12 if fixture['test']=='test_reservations.py' else 13,independent_test_cases_run=int(match.group(1)) if match else (13 if test_exit==0 and 'ALL_TESTS_PASSED' in test_output else None))
    result['passed']=not error and code==0 and result['complete'] and result['completed_tools']>0 and result['test_commands_completed']>0 and test_exit==0 and protected and set(changed)==set(fixture['initial']) and not unexpected and result['readers_joined']
    return result

def suite(report):
    # Existing code oracle/AST/bwrap checks; preserve unsupported-source failures.
    for case in json.loads((ROOT/'code-screen-v1.json').read_text())['cases']:
        row=dict(id=case['id'],passed=False);directory=ROOT/('code-'+case['id']);directory.mkdir()
        try:
            message,metadata=chat([{'role':'user','content':case['prompt']}],512)
            row.update(metadata);(directory/'answer.txt').write_text(message['content'])
            row.update(oracle.check_code(message['content'],case,directory))
        except Exception as error:row['error']=type(error).__name__
        report['cases'].append(row)
    messages=[{'role':'user','content':'A synthetic record says Kestrel has 12 clips and Mora has 7. Output only JSON with keys Kestrel and Mora and their integer counts.'}]
    message,metadata=chat(messages,64)
    try:passed=json.loads(message['content'])=={'Kestrel':12,'Mora':7}
    except ValueError:passed=False
    report['cases'].append(dict(id='chat-record',passed=passed,output=message['content'],metadata=metadata))
    messages.extend([message,{'role':'user','content':'Move three clips from Kestrel to Mora. Output only JSON with keys Kestrel and Mora and their new integer counts.'}])
    message,metadata=chat(messages,64)
    try:passed=json.loads(message['content'])=={'Kestrel':9,'Mora':10}
    except ValueError:passed=False
    report['cases'].append(dict(id='chat-context-json',passed=passed,output=message['content'],metadata=metadata))
    tools=[{'type':'function','function':{'name':'lookup_stock','description':'Look up the count for a synthetic item.','parameters':{'type':'object','properties':{'item_id':{'type':'string'}},'required':['item_id'],'additionalProperties':False}}}]
    messages=[{'role':'system','content':'For inventory questions call lookup_stock. Never guess the count. After its result, output only JSON with item_id and stock.'},{'role':'user','content':'How many units of synthetic_adapter are in stock?'}]
    call,metadata=chat(messages,128,tools)
    item=call['tool_calls'][0];function=item.get('function',{})
    passed=function.get('name')=='lookup_stock' and json.loads(function.get('arguments','{}'))=={'item_id':'synthetic_adapter'} and isinstance(item.get('id'),str) and bool(item['id'])
    report['cases'].append(dict(id='structured-tool-call',passed=passed,metadata=metadata,tool_message=call))
    if passed:
        messages.extend([call,{'role':'tool','tool_call_id':item['id'],'content':json.dumps({'item_id':'synthetic_adapter','stock':17})}])
        message,metadata=chat(messages,96,tools,False)
        try:passed=json.loads(message['content'])=={'item_id':'synthetic_adapter','stock':17}
        except ValueError:passed=False
        report['cases'].append(dict(id='tool-result-final',passed=passed,output=message['content'],metadata=metadata))
    for fixture in json.loads((ROOT/'factual-chat.json').read_text())['cases']:
        message,metadata=chat([{'role':'user','content':fixture['prompt']}],192)
        report['cases'].append(dict(id=fixture['id'],output=message['content'],metadata=metadata,manual_review=True,
            refusal_marker_present=any(s in message['content'].lower() for s in ('cannot discuss','can\'t discuss','cannot provide','can\'t provide','unable to discuss','not permitted to'))))

def main():
    profile=json.loads((ROOT/'profile.json').read_text());report=dict(stage=profile['stage'],cases=[],passed=False)
    args=['/engine/'+profile['binary_name'],'--model','/model.gguf','--alias','comparison','--host','127.0.0.1','--port',str(PORT),'--ctx-size',str(profile['context_size']),'--threads',str(profile['threads']),'--n-gpu-layers','0',*profile['extra_args']]
    if profile.get('prefix_cache',{}).get('enabled'):
        (ROOT/'prefixes').mkdir(mode=0o700)
        args += ['--slots','--slot-save-path','/work/prefixes']
    environment=dict(os.environ,LD_LIBRARY_PATH='/engine',**profile['environment']);born=time.monotonic()
    server=subprocess.Popen(args,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,env=environment)
    report['server_log']={}
    reader=threading.Thread(target=capture,args=(server.stdout,ROOT/'server.log',None,report['server_log']),daemon=True);reader.start()
    try:
        until=time.monotonic()+180
        while True:
            if server.poll() is not None:raise RuntimeError('server_exited_during_load')
            try:
                health,_=api('/health',timeout=2)
                if health.get('status')=='ok':
                    (ROOT/'native-ready').write_text('LOCAL_AI_MODEL_READY\n')
                    break
            except (OSError,http.client.HTTPException,ValueError):pass
            if time.monotonic()>until:raise TimeoutError('load_deadline')
            time.sleep(.25)
        report['load_seconds']=time.monotonic()-born
        props,_=api('/props',timeout=5);(ROOT/'props.json').write_text(json.dumps(props,indent=2))
        message,metadata=chat([{'role':'user','content':'What is 17 plus 25? Output only the integer.'}],32)
        report['smoke']=dict(output=message['content'],metadata=metadata,passed=message['content'].strip()=='42')
        if not report['smoke']['passed']:raise ValueError('smoke_answer_failed')
        if profile['stage']=='suite':suite(report)
        elif profile['stage'] in ('normal-agent','stress-agent'):report['agent']=run_agent(json.loads((ROOT/'fixture.json').read_text()))
        report['passed']=report['smoke']['passed'] and all(row.get('passed',True) for row in report['cases']) and report.get('agent',{}).get('passed',True)
        report['manual_review_required']=any(row.get('manual_review') for row in report['cases'])
    except BaseException as error:report['error']=type(error).__name__
    finally:
        if server.poll() is None:
            server.terminate()
            try:server.wait(timeout=5)
            except subprocess.TimeoutExpired:server.kill();server.wait(timeout=3)
        reader.join(5);report.update(server_reaped=server.poll() is not None,server_exit=server.returncode,server_reader_joined=not reader.is_alive(),total_seconds=time.monotonic()-born)
        report['passed']=report['passed'] and report['server_reaped'] and report['server_reader_joined']
        (ROOT/'result.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    return 0 if report['passed'] else 1

if __name__=='__main__':raise SystemExit(main())
