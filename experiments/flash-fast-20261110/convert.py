"""Convert and independently verify one dense-Q4 artifact; run under the existing leased guard."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,os,signal,subprocess,time
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--head',choices=['q6','q4'],required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();h=Path.home();opt=Path(os.environ.get('LAB_RESULTS_DIR','results'))
os.umask(0o077)
a.output.mkdir(parents=True,mode=0o700,exist_ok=False)
old=opt/'flash-selective-quant-feasibility/real-selective-q8-v3/convert_check.py';sp=importlib.util.spec_from_file_location('artifact_reader',old);v=importlib.util.module_from_spec(sp);sp.loader.exec_module(v);v.BLOCKS[12]=(256,144)
plan=json.loads((opt/'flash-fast-q4-20261110-plan-v2/plan.json').read_text());targets={r['name']:r['target_type'] for r in plan['tensors_plan']}
if a.head=='q6':targets['output.weight']='Q6_K'
typeids={name.upper():tid for tid,name in v.TYPES.items()};typeids['Q4_K']=12
original=json.loads((opt/'flash-selective-quant-feasibility/real-selective-q8-v3/execution/original-verification.json').read_text())
# Reuse the full original tensor hashes only while all identity fields still match.
for row in original:assert v.identity(row['path'])==row['identity'],'original changed since its complete payload verification'
assert set(targets)=={n for row in original for n in row['tensors']}
assert os.statvfs(a.output).f_bavail*os.statvfs(a.output).f_frsize>sum(row['bytes'] for row in original)+10*2**30
binary=opt/'flash-mtp-acceptance-20261110/build/bin/llama-quantize';typefile=opt/'flash-fast-q4-20261110-plan-v2'/('tensor-types-head-q6.txt' if a.head=='q6' else 'tensor-types.txt');prefix=a.output/('Flash-Next-dense-Q4-head'+a.head.upper())
argv=[str(binary),'--allow-requantize','--pure','--keep-split','--max-buffer-size','16','--override-kv','general.file_type=int:23','--override-kv','general.quantization_version=int:2','--tensor-type-file',str(typefile),original[0]['path'],str(prefix),'Q4_K','2']
row=dict(kind='flash-dense-q4-artifact',head=a.head,argv=argv,converter_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),type_map_sha256=hashlib.sha256(typefile.read_bytes()).hexdigest(),original_verification_reused=True,passed=False,quality_qualified=False,production_changed=False);start=time.monotonic()
def stop(*_):raise SystemExit('conversion guard stop')
signal.signal(signal.SIGTERM,stop)
def native(args,log):
 child=subprocess.Popen(args,stdout=log,stderr=log)
 try:
  code=child.wait(timeout=900)
  if code:raise subprocess.CalledProcessError(code,args)
 except BaseException:
  if child.poll() is None:
   child.terminate()
   try:child.wait(timeout=3)
   except subprocess.TimeoutExpired:child.kill();child.wait()
  raise
try:
 with (a.output/'dry-run.log').open('w') as log:native(argv[:1]+['--dry-run']+argv[1:],log)
 with (a.output/'conversion.log').open('w') as log:native(argv,log)
 class Guard:
  def require(self):pass # The outer existing MemoryGuard owns process-group cleanup.
 guard=Guard();scans=[];changed=[];unchanged=[]
 for i,ref in enumerate(original,1):
  path=Path(str(prefix)+f'-{i:05d}-of-00004.gguf');ident=v.identity(path)
  with path.open('rb') as f:
   os.fsync(f.fileno());r=v.HashReader(f,guard);metadata,ts,start_data,end_header=v.parse_header(r)
   assert {k:{x:y[x] for x in ['type','sha256']} for k,y in metadata.items()}==ref['metadata'],'typed metadata changed'
   assert [t['name'] for t in ts]==list(ref['tensors']),'tensor set/order changed'
   tensors={};cursor=0
   for t in ts:
    n=t['name'];prior=ref['tensors'][n];tid=typeids[targets[n]];assert t['type']==tid and t['shape']==prior['shape'] and t['offset']==cursor,(n,'type/shape/offset')
    absolute=start_data+t['offset'];assert absolute>=r.at and absolute+t['bytes']<=ident['st_size']
    r.consume(absolute-r.at);sha=hashlib.sha256();r.consume(t['bytes'],sha);digest=sha.hexdigest();tensors[n]=dict(type=t['type'],bytes=t['bytes'],sha256=digest)
    if tid==prior['type']:
     assert digest==prior['sha256'] and t['bytes']==prior['bytes'],(n,'unchanged payload differs');unchanged.append(n)
    else:
     assert '_exps.' not in n and prior['type'] in [14,8] and tid==12,(n,'unauthorized tensor change');changed.append(n)
    cursor=t['offset']+(t['bytes']+31)//32*32
   assert ident['st_size']==start_data+cursor,'candidate canonical size differs'
   r.consume(ident['st_size']-r.at);assert not f.read(1);file_sha=r.sha.hexdigest()
  assert v.identity(path)==ident
  scans.append(dict(path=str(path),bytes=ident['st_size'],sha256=file_sha,tensors=tensors))
 assert len(changed)==(352 if a.head=='q6' else 353);assert sum('_exps.' in n for n in unchanged)==144 and 'per_layer_token_embd.weight' in unchanged
 for ref in original:assert v.identity(ref['path'])==ref['identity'],'original changed during conversion'
 fd=os.open(a.output,os.O_RDONLY|os.O_DIRECTORY)
 try:os.fsync(fd)
 finally:os.close(fd)
 row.update(passed=True,changed_count=len(changed),unchanged_count=len(unchanged),expert_payloads_identical=144,ngram_payload_identical=True,files=scans)
finally:
 row['seconds']=time.monotonic()-start;(a.output/'artifact.json').write_text(json.dumps(row,indent=2)+'\n');print(json.dumps({k:y for k,y in row.items() if k not in ['files','argv']}),flush=True)
