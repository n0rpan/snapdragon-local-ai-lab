from pathlib import Path
import datetime,hashlib,json,os,signal,subprocess,sys,time
h=Path.home(); repo=Path(__file__).resolve().parents[2]
root=Path(os.environ['FLASH_M0_ENGINE'])  # engine checkout at bf97cfd
out=Path(os.environ['FLASH_M0_OUTPUT']); out.mkdir(mode=0o700,exist_ok=False)
sys.path.insert(0,str(repo/'src/snapdragon_ai_lab/runtime')); import memory_policy as mem
sys.path.insert(0,str(repo/'benchmarks/roofline')); from nvme_read import sensors
build=json.loads((root.parent/'local-ai-optimization/flash-m0-isolated-20261110/build.json').read_text())
(out/'build.json').write_text(json.dumps(build,indent=2)+'\n')
class Native:
 def __init__(self,pid): self.pid=pid; self.stamp=Path(f'/proc/{pid}/stat').read_text().split()[21]
 def poll(self):
  try:
   s=Path(f'/proc/{self.pid}/stat').read_text().split(); return 0 if s[21]!=self.stamp or s[2]=='Z' else None
  except FileNotFoundError: return 0
 def terminate(self):
  if self.poll() is None: os.kill(self.pid,signal.SIGTERM)
 def kill(self):
  if self.poll() is None: os.kill(self.pid,signal.SIGKILL)
def native_descendant(pid):
 try:
  if Path(f'/proc/{pid}/comm').read_text().strip()=='flash-m0':
   assert Path(f'/proc/{pid}').stat().st_uid==os.getuid(); return Native(pid)
  children=Path(f'/proc/{pid}/task/{pid}/children').read_text().split()
 except FileNotFoundError: return None
 for child in children:
  found=native_descendant(int(child))
  if found: return found
 return None
progress=[]
for spec in json.loads(os.environ['FLASH_M0_ARMS']):
 gib=spec['cache_gib']
 assert hashlib.sha256((root/'target/release/flash-m0').read_bytes()).hexdigest()==build['binary_sha256']
 assert subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()==build['source_commit']
 label=spec['label']; flags=['--io','demand','--cache-gib',str(gib),'--max-gib','28',*spec.get('extra',[])]; started=time.monotonic(); guard=None; native=None; cg=None; last_sample=None; failure=None; peak_rss={}; before=sensors(); peaks=dict(before)
 while True:
  logpath=out/(label+'.log')
  with logpath.open('a') as log:
   child=subprocess.Popen([sys.executable,str(h/'.local/state/local-ai/run_flash_m0_isolated.py'),*flags],cwd=root,stdout=subprocess.PIPE,stderr=log,text=True,start_new_session=True,env=dict(os.environ,**({'FLASH_M0_KCRYPTD_MASK':spec['crypto_mask']} if spec.get('crypto_mask') else {})))
   try:
    while child.poll() is None:
     if native is None:
      native=native_descendant(child.pid)
      if native:
       cg=Path('/sys/fs/cgroup')/next(x[3:] for x in Path(f'/proc/{native.pid}/cgroup').read_text().splitlines() if x.startswith('0::')).lstrip('/')
       assert (cg/'memory.max').read_text().strip()==str(28*2**30) and (cg/'memory.swap.max').read_text().strip()=='0'
       last_sample=mem.read_sample(cg)
       def read_sample():
        global last_sample
        try: last_sample=mem.read_sample(cg)
        except FileNotFoundError:
         # Scope files can disappear just before /proc exposes the final exit.
         time.sleep(.025)
         if native.poll() is None: raise
        return last_sample
       guard=mem.MemoryGuard(reader=read_sample,loading_timeout_seconds=120); guard.start(native)
     if guard:
      if 'cache warmed from the prefill routes:' in logpath.read_text(errors='replace'): guard.mark_ready()
      if guard.failure(): raise RuntimeError(guard.failure())
     power=sensors()
     for k,v in power.items(): peaks[k]=max(peaks.get(k,v),v)
     capacities=[v for k,v in power.items() if k.endswith('/capacity')]
     if capacities and min(capacities)<20: raise RuntimeError('battery_below20')
     if time.monotonic()-started>360: raise TimeoutError('bounded_cache_screen')
     if native and native.poll() is None:
      try:
       for line in Path(f'/proc/{native.pid}/status').read_text().splitlines():
        f=line.split()
        if f[0] in ('VmHWM:','VmRSS:','VmSwap:'): peak_rss[f[0][:-1]]=max(peak_rss.get(f[0][:-1],0),int(f[1])*1024)
      except FileNotFoundError: pass
     time.sleep(.25)
    stdout=child.communicate(timeout=3)[0]
   except BaseException as e:
    failure=type(e).__name__+':'+str(e)
    if native: native.terminate()
    try: os.killpg(child.pid,signal.SIGTERM)
    except ProcessLookupError: pass
    try: child.wait(timeout=3)
    except subprocess.TimeoutExpired:
     if native: native.kill()
     os.killpg(child.pid,signal.SIGKILL); child.wait()
   finally:
    if guard: guard.stop()
  if failure or child.returncode!=75: break
  if time.monotonic()-started>300: raise TimeoutError('lease wait beyond five minutes')
  time.sleep(5)
 row={'label':label,'cache_gib':gib,'flags':flags,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'wall_seconds':time.monotonic()-started,'returncode':child.returncode,'failure':failure,'guard':guard.metadata() if guard else None,'process_memory_peaks':peak_rss,'sensor_peaks':peaks,'binary_sha256':build['binary_sha256']}
 if not failure and child.returncode==0:
  path=(root/stdout.strip()).resolve(); assert path.is_relative_to(root/'results'); data=json.loads(path.read_text())
  if guard is None or guard.failure() is not None: row['failure']='final_guard:'+str(guard.failure() if guard else 'not_attached');failure=row['failure']
  row.update(raw_path=str(path),native=data); (out/(label+'.native.json')).write_text(json.dumps(data,indent=2)+'\n')
 (out/(label+'.json')).write_text(json.dumps(row,indent=2)+'\n'); progress.append({k:v for k,v in row.items() if k not in ('native','sensor_peaks')}); (out/'progress.json').write_text(json.dumps(progress,indent=2)+'\n')
 print(json.dumps({k:v for k,v in row.items() if k not in ('native','sensor_peaks','guard')}),flush=True)
 if failure or child.returncode: raise RuntimeError(f'{label} failed; no retry')
