#!/usr/bin/env python3
"""Read-only fixed Flash storage screen. Run under roofline/run.py and its lease.

Without --run only prints metadata; no model payload or inference is read.
"""
from pathlib import Path
import argparse,json,os,resource,subprocess,time
import nvme_read as n

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--run',action='store_true');ap.add_argument('--file',type=Path,default=n.DEFAULT);ap.add_argument('--output',type=Path);args=ap.parse_args()
    if not args.run:
        print(json.dumps(dict(metadata_only=True,arms=30,timed_IO_seconds=60,readonly=True,blocks=[524288,2097152],depths=[1,4,8,16,32])));return
    if args.output is None or not args.output.is_absolute():raise ValueError('fresh_absolute_output_required')
    p=args.file;ram=int(next(v.split()[1] for v in Path('/proc/meminfo').read_text().splitlines() if v.startswith('MemTotal:')))*1024;ident=n.owned_file(p,ram);hz=os.sysconf('SC_CLK_TCK');out=args.output;out.mkdir(mode=0o700,exist_ok=False)
    def cpu():
     v=list(map(int,Path('/proc/stat').read_text().splitlines()[0].split()[1:]));return dict(kernel=(v[2]+v[5]+v[6])/hz,active=(v[0]+v[1]+v[2]+v[5]+v[6])/hz)
    a=cpu();start=time.monotonic();time.sleep(1);b=cpu();idle={k:(b[k]-a[k])/(time.monotonic()-start) for k in a};begin=time.monotonic();version=subprocess.check_output(['/usr/bin/fio','--version'],text=True).strip()
    for rep in range(3):
     for block in [512*1024,2*1024**2]:
      for qd in ([1,4,8,16,32] if rep%2==0 else [32,16,8,4,1]):
       assert n.owned_file(p,ram)==ident
       argv=n.command('/usr/bin/fio',p,ident['st_size'],'io_uring',2,dict(direct=1,block_bytes=block,qd=qd,seed=20261110+rep));start=time.monotonic();before=cpu();disks=n.diskstats();usage=resource.getrusage(resource.RUSAGE_CHILDREN)
       result=subprocess.run(argv,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=12,check=True);elapsed=time.monotonic()-start;after=cpu();usage2=resource.getrusage(resource.RUSAGE_CHILDREN);data=json.loads(result.stdout);m=n.fio_metrics(data);job=data['jobs'][0];gb=m['read_bytes']/1e9
       row=dict(kind='flash-P3-readonly-encrypted-file',rep=rep+1,block_bytes=block,qd=qd,fio_version=version,metrics=m,wall_seconds=elapsed,read_bytes= m['read_bytes'],host_kernel_core_seconds_per_GB=(after['kernel']-before['kernel'])/gb,host_active_core_seconds_per_GB=(after['active']-before['active'])/gb,idle_adjusted_kernel_core_seconds_per_GB=(after['kernel']-before['kernel']-idle['kernel']*elapsed)/gb,child_user_core_seconds_per_GB=(usage2.ru_utime-usage.ru_utime)/gb,child_system_core_seconds_per_GB=(usage2.ru_stime-usage.ru_stime)/gb,host_idle_control_core_seconds_per_second=idle,disk_read_bytes_systemwide={k:v-disks[k] for k,v in n.diskstats().items() if k in disks},fio_usr_cpu_percent=job.get('usr_cpu'),fio_sys_cpu_percent=job.get('sys_cpu'),passed=True,scope='O_DIRECT random reads of existing GGUF on btrfs over dm-crypt; includes the whole filesystem/crypto path, not isolated AES or raw media bandwidth')
       (out/f'r{rep+1}-b{block}-q{qd}.json').write_text(json.dumps(row,indent=2)+'\n');print(json.dumps(row),flush=True)
       if time.monotonic()-begin>210:raise TimeoutError('P3 bounded matrix210seconds')

if __name__=='__main__':main()
