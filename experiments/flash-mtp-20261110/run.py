"""One Flash MTP acceptance arm. Caller supplies the existing lease/scope/MemoryGuard."""
from pathlib import Path
import argparse,ctypes,hashlib,importlib.util,json,math,os,re,selectors,signal,subprocess,sys,time
ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--depth',type=int,choices=range(4),required=True);ap.add_argument('--workload',choices=['fixed353','real-code'],required=True);ap.add_argument('--batch-noise-proof',type=Path);ap.add_argument('--build-proof',type=Path);a=ap.parse_args()
h=Path.home();opt=Path(os.environ.get('LAB_RESULTS_DIR','results'));out=a.output;out.mkdir(mode=0o700,parents=True,exist_ok=False)
sp=importlib.util.spec_from_file_location('flash_fixed',opt/'flash-next-research/code-scheduling/screen.py');m=importlib.util.module_from_spec(sp);sys.modules[sp.name]=m;sp.loader.exec_module(m)
proof=json.loads((a.build_proof or opt/'flash-mtp-acceptance-20261110/build.json').read_text());binary=Path(proof.get('binary_path',str(opt/'flash-mtp-acceptance-20261110/build/bin/bmoe-cli')));sha=hashlib.sha256(binary.read_bytes()).hexdigest();assert sha==proof['binary_sha256']
model=next((h/'.local/share/ai-stack/models/llm/Qwen3.8-Flash-Next/UD-IQ3_XXS').glob('*00001*'));head=h/'.local/share/ai-stack/models/llm/Qwen3.8-Flash-Next/MTP/mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf'
argv=m.argv(binary,model,'overlap',True,4);argv[argv.index('--io-threads')+1]='8'
if a.depth:argv+=['--mtp','--mtp-head',str(head),'--mtp-head-merge','--spec-exact','--draft',str(a.depth),'--mtp-p-min','0']
env={k:v for k,v in os.environ.items() if not k.startswith(('BMOE_','GGML_','LLAMA_','OMP_'))};env.update(OMP_NUM_THREADS='6',OMP_WAIT_POLICY='PASSIVE',OMP_DYNAMIC='FALSE')
for key in ['HOME','XDG_CONFIG_HOME','XDG_STATE_HOME','XDG_CACHE_HOME','XDG_DATA_HOME']:
 d=out/key.lower();d.mkdir(mode=0o700);env[key]=str(d)
config=json.loads(subprocess.check_output(argv+['--show-config'],env=env,text=True,timeout=30));c=config['config'];assert c['spec']==('mtp' if a.depth else 'none')
if a.depth:assert c['draft']==a.depth and c['mtp-head']==str(head) and c['mtp-head-merge'] is True and c['spec-exact'] is True and c['mtp-p-min']==0
# Validate the baseline controls, while explicitly allowing only the measured IO/spec changes.
control=json.loads(json.dumps(config));control['config']['io-threads']=4
for key in ('spec','draft','mtp-head','mtp-head-merge','spec-exact','mtp-p-min'):
 if key in m.g.BASELINE:control['config'][key]=m.g.BASELINE[key]
m.resolved(control,'overlap',True,4);(out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
prompt=m.PROMPT;cap=512
if a.workload=='real-code':
 sp_real=importlib.util.spec_from_file_location('real_code',Path(__file__).with_name('real_code.py'));real=importlib.util.module_from_spec(sp_real);sp_real.loader.exec_module(real);prompt=real.PROMPT;cap=1536;m.CAP=cap
assert ctypes.CDLL(None,use_errno=True).prctl(41,1,0,0,0)==0
cg=Path('/sys/fs/cgroup')/next(v[3:] for v in Path('/proc/self/cgroup').read_text().splitlines() if v.startswith('0::')).lstrip('/');assert (cg/'memory.swap.max').read_text().strip()=='0'
child=None;start=time.monotonic();request=first=done=None;protocol=m.Protocol();row=dict(kind='flash-mtp-acceptance',workload=a.workload,workload_version='bounded-lru-v2' if a.workload=='real-code' else 'fixed353-v1',prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),depth=a.depth,binary_sha256=sha,argv=argv,passed=False,production_changed=False)
signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(SystemExit('cancelled')))
try:
 with (out/'native.log').open('w') as log:
  child=subprocess.Popen(argv+['--session'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,env=env,bufsize=0);buf=b''
  with selectors.DefaultSelector() as sel:
   sel.register(child.stdout,selectors.EVENT_READ)
   while child.poll() is None:
    if time.monotonic()-start>850:raise TimeoutError('bounded_MTP_arm850s')
    for key,_ in sel.select(.2):
     chunk=os.read(key.fd,65536)
     if not chunk:break
     buf+=chunk;assert len(buf)<2*1024**2
     while b'\n' in buf:
      line,buf=buf.split(b'\n',1);event=m.g.protocol(line)
      if event is None:continue
      kind,value=event;protocol.accept(kind,value);now=time.monotonic()
      if kind=='BMOE_READY':
       row['load_seconds']=now-start;row['ready']=value;(out/'native-ready').write_text('LOCAL_AI_MODEL_READY\n');child.stdin.write((json.dumps(dict(cmd='generate',id=1,prompt=prompt,n_predict=cap,think=False,clear_kv=True))+'\n').encode());child.stdin.flush();request=time.monotonic()
      elif kind=='BMOE_PROGRESS' and first is None:first=now
      elif kind=='BMOE_DONE':
       done=value;row['request_seconds']=now-request;row['ttft_seconds']=None if first is None else first-request;answer=value.pop('text');(out/'answer.txt').write_text(answer);row['answer_sha256']=hashlib.sha256(answer.encode()).hexdigest();row['metrics']={k:v for k,v in value.items() if isinstance(v,(int,float,bool,list))};child.stdin.write(b'{"cmd":"close"}\n');child.stdin.flush();child.stdin.close();child.stdin=None
    if done is not None:child.wait(timeout=15);break
  assert child.returncode==0 and done is not None and protocol.done
  row['budget_truncated']=done['tokens']==cap
  row['effective_decode_tokens_s']=done['tokens']/max(1e-9,row['request_seconds']-done['prefill_s'])
  proposed=done['mtp_proposed_by_pos'];accepted=done['mtp_accepted_by_pos'];assert sum(proposed)==done['mtp_drafted'] and sum(accepted)==done['mtp_accepted'];assert all(0<=x<=y for x,y in zip(accepted,proposed));row['acceptance_by_position']=[dict(position=i+1,proposed=n,accepted=accepted[i],rate=None if not n else accepted[i]/n) for i,n in enumerate(proposed) if n]
  if a.workload=='fixed353':
   row['oracle']=m.oracle(answer,done['tokens'],out);assert done['n_prompt']==156 and row['oracle']['passed']
   row['reference_identity_match']=done['tokens']==353 and row['answer_sha256']=='460169e27739a86fbb94340a3d7ffcbace214719d46fcda2cc8bb3d0fc0a4093'
   if not row['reference_identity_match']:
    assert a.depth and a.batch_noise_proof,'unclassified numeric divergence'
    assert a.batch_noise_proof.stat().st_size<=128*1024
    proof_bytes=a.batch_noise_proof.read_bytes();noise=json.loads(proof_bytes)
    assert all(math.isfinite(noise[k]) for k in ['verify_vs_single_mean_kld','production_batch_vs_single_mean_kld'])
    original=json.loads((opt/'flash-selective-quant-feasibility/real-selective-q8-v3/execution/original-verification.json').read_text())
    assert noise['target_shards_sha256']==[x['sha256'] for x in original] and noise['positions'][0]==100 and 100<noise['positions'][1]<=350 and noise['rows']==noise['positions'][1]-99
    assert 0<=noise['verify_vs_single_mean_kld']<=noise['production_batch_vs_single_mean_kld'],'numeric change exceeds measured production noise'
    row['batch_noise_proof']=dict(path=str(a.batch_noise_proof),sha256=hashlib.sha256(proof_bytes).hexdigest(),**noise)
   row['quality_qualified']=True
  else:
   control=out.parent/'real-code-d0/summary.json'
   if a.depth:
    ref=json.loads(control.read_text());row['reference_identity_match']=row['answer_sha256']==ref['answer_sha256'] and done['tokens']==ref['metrics']['tokens']
   row['oracle']=real.check(answer,out,m.checks.code_check);assert not row['budget_truncated'] and row['oracle']['passed']
   row['quality_qualified']=True
   row['tests_sha256']=hashlib.sha256(real.TESTS.encode()).hexdigest()
  row['passed']=True
finally:
 if child and child.poll() is None:
  child.terminate()
  try:child.wait(timeout=5)
  except subprocess.TimeoutExpired:child.kill();child.wait()
 row.update(wall_seconds=time.monotonic()-start,cgroup_peak_bytes=int((cg/'memory.peak').read_text()),cgroup_swap_bytes=int((cg/'memory.swap.current').read_text()),cgroup_events=(cg/'memory.events').read_text(),child_reaped=child is not None and child.poll() is not None)
 (out/'summary.json').write_text(json.dumps(row,indent=2)+'\n');print(json.dumps(row),flush=True)
