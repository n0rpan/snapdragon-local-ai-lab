"""One leased, bounded-scope replay; optional temporary crypto-workqueue affinity."""
from pathlib import Path
import datetime,fcntl,json,os,signal,subprocess,sys,time
h=Path.home();root=Path(os.environ['FLASH_M0_ENGINE'])  # engine checkout at bf97cfd
lock=(h/'.local/state/local-ai/inference.lock').open('a')
try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
except BlockingIOError:sys.exit(75)
mask=os.environ.get('FLASH_M0_KCRYPTD_MASK'); path=Path('/sys/devices/virtual/workqueue/kcryptd-252:0-1/cpumask')
if mask not in (None,'e00','f00','c80','c88'):raise ValueError('unexpected test mask')
(root/'results').mkdir(exist_ok=True)
stem='flash-m0-isolated-'+datetime.datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+str(os.getpid())
out=root/'results'/(stem+'.json');proof=root/'results'/(stem+'.affinity.json');record={'requested':mask,'changed':False,'restored':False};child=None
before=path.read_text().strip();record['before']=before

def write(value):
 subprocess.run(['sudo','-n','tee',str(path)],input=value+'\n',text=True,stdout=subprocess.DEVNULL,check=True)

def stop(*_):raise SystemExit(143)
signal.signal(signal.SIGTERM,stop)
try:
 if mask:
  write(mask);record['changed']=True;record['during']=path.read_text().strip();assert int(record['during'].replace(',',''),16)==int(mask,16)
 cmd=['systemd-run','--user','--scope','--quiet','-p','MemoryMax=28G','-p','MemorySwapMax=0','--',str(root/'target/release/flash-m0'),'run','--cpu-list','0,1,2,4,5,6,8,9','--io-cpu','3','--cache-gib','16','--quota','prefill','--io','lookahead','--recall','0.9','--depth','2','--false-pos','off','--tokens','353','--warmup','8',*sys.argv[1:],'--out',str(out)]
 record['argv']=cmd;child=subprocess.Popen(cmd,cwd=root,stdout=subprocess.DEVNULL);rc=child.wait();record['returncode']=rc
finally:
 if child and child.poll() is None:
  child.terminate()
  try:child.wait(timeout=5)
  except subprocess.TimeoutExpired:child.kill();child.wait()
 if record['changed']:write(before)
 record['after']=path.read_text().strip();record['restored']=record['after']==before;proof.write_text(json.dumps(record,indent=2)+'\n')
 if not record['restored']:raise RuntimeError('crypto affinity restore failed')
if rc==0:print(str(out.relative_to(root)),flush=True)
sys.exit(rc)
