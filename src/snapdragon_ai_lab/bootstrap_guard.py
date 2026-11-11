"""Finite user-unit guard shared by bootstrap fetch/build; no work on import."""
import fcntl
import json
import math
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import threading
import time
from . import power

GIB=1024**3
class Stopped(Exception):pass

def secure_directory(path):
    value=Path(path).lstat()
    if not stat.S_ISDIR(value.st_mode) or value.st_uid!=os.getuid() or value.st_mode&0o022:raise Stopped('unsafe_owned_directory')

def acquire_lease(home):
    directory=Path(home)/'.local/state/local-ai'
    current=Path(home)
    secure_directory(current)
    for part in ['.local','state','local-ai']:
        current=current/part
        if not current.exists():current.mkdir(mode=0o700)
        secure_directory(current)
    path=directory/'inference.lock'
    fd=os.open(path,os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW|os.O_CLOEXEC,0o600)
    try:
        value=os.fstat(fd);canonical=path.lstat()
        if not stat.S_ISREG(value.st_mode) or value.st_uid!=os.getuid() or value.st_mode&0o777!=0o600 or (value.st_dev,value.st_ino)!=(canonical.st_dev,canonical.st_ino):raise Stopped('unsafe_canonical_lease')
        for relative in ['Work/prompt-studio/generation.lock','Work/local-ai/prompt-studio/generation.lock']:
            legacy=Path(home)/relative
            if legacy.exists() and (legacy.stat().st_dev,legacy.stat().st_ino)!=(value.st_dev,value.st_ino):raise Stopped('legacy_lease_inode_mismatch')
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);return fd
    except BaseException:os.close(fd);raise

def seconds(value):
    parts=list(re.finditer(r'(\d+(?:\.\d+)?)(us|ms|min|s|h)',value));units={'us':1e-6,'ms':.001,'s':1,'min':60,'h':3600}
    if not parts or ''.join(p.group() for p in parts)!=''.join(value.split()):raise Stopped('finite_unit_deadline_required')
    return sum(float(p[1])*units[p[2]] for p in parts)

def unit_controls(kind):
    unit='local-ai-bootstrap-'+kind+'.service'
    relative=next((x[3:] for x in Path('/proc/self/cgroup').read_text().splitlines() if x.startswith('0::')),'')
    if Path(relative).name!=unit or '..' in Path(relative).parts:raise Stopped('dedicated_bootstrap_unit_required')
    group=Path('/sys/fs/cgroup')/relative.lstrip('/')
    values={k:(group/k).read_text().strip() for k in ['cpu.max','memory.max','memory.swap.max','pids.max']}
    quota,period=map(int,values['cpu.max'].split());limit=4*GIB if kind=='build' else GIB
    if not(0<quota<=(period if kind=='sources' else 2*period) and period>0 and 0<int(values['memory.max'])<=limit and int(values['memory.swap.max'])==0 and 0<int(values['pids.max'])<=(64 if kind=='sources' else 160)):raise Stopped('bootstrap_resource_caps_required')
    deadline=2700 if kind=='build' else 900 if kind=='sources' else 3700
    result=subprocess.run(['systemctl','--user','show',unit,'--property=RuntimeMaxUSec,TimeoutStopUSec,KillMode,SendSIGKILL'],check=True,capture_output=True,text=True,timeout=5)
    props=dict(line.split('=',1) for line in result.stdout.splitlines() if '=' in line)
    if not(0<seconds(props['RuntimeMaxUSec'])<=deadline and 0<seconds(props['TimeoutStopUSec'])<=15 and props['KillMode']=='control-group' and props['SendSIGKILL']=='yes'):raise Stopped('finite_unit_cleanup_required')
    return group,dict(values,**props),deadline

class Guard:
    def __init__(self,group,seconds_limit):
        self.group=Path(group);self.deadline=time.monotonic()+seconds_limit
        self.failure=None;self.history=[];self.swap_initial=None;self.swap_latest=None
        self.minimum_available=None;self.network_active=False;self.network_deadline=None
        self.stop=threading.Event();self.thread=threading.Thread(target=self.watch,daemon=True)
    def signal_stop(self,number,_frame):
        if number!=signal.SIGUSR1:self.failure='requested_stop'
        if self.network_active:
            self.network_active=False;self.network_deadline=None
            raise Stopped(self.failure or 'network_stop')
    def network_call(self,function,seconds_limit=5):
        self.require();self.network_deadline=time.monotonic()+seconds_limit
        try:
            self.network_active=True;self.require();value=function();self.require();return value
        finally:self.network_active=False;self.network_deadline=None
    def poll(self,starting=False,kind='fetch'):
        available=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
        self.minimum_available=available if self.minimum_available is None else min(available,self.minimum_available)
        if available<(8*GIB if starting and kind=='build' else 2*GIB):raise Stopped('actual_ram_reserve')
        for name in ['memory.swap.current','memory.swap.max']:
            if int((self.group/name).read_text())!=0:raise Stopped('own_swap_prohibited')
        vm=dict((k,int(v)) for k,v in (x.split() for x in Path('/proc/vmstat').read_text().splitlines()) if k in ['pswpin','pswpout'])
        pair=(vm['pswpin'],vm['pswpout'])
        if self.swap_initial is None:self.swap_initial=pair
        self.swap_latest=pair
        if pair[1]>self.swap_initial[1]:raise Stopped('global_swapout_increased')
        value=power.snapshot();percent=value.get('battery_percent');now=time.monotonic()
        if not value.get('external_online') or type(percent) not in [int,float] or not math.isfinite(percent) or not 0<=percent<=100 or value.get('battery_status') not in ['Charging','Full','Not charging','Discharging']:raise Stopped('power_unverified')
        if percent<=20 or (starting and percent<30):raise Stopped('battery_reserve')
        self.history.append((now,value));self.history=[x for x in self.history if now-x[0]<=125]
        if power.discharge_trend(self.history):raise Stopped('sustained_battery_discharge')
        if now>=self.deadline:raise Stopped('parent_deadline')
        if self.network_active and self.network_deadline is not None and now>=self.network_deadline:raise Stopped('network_wall_budget')
    def watch(self):
        while not self.stop.wait(.25):
            try:self.poll()
            except Exception as error:
                self.failure=str(error) if isinstance(error,Stopped) else 'resource_telemetry_failed'
                if self.network_active and not self.stop.is_set():os.kill(os.getpid(),signal.SIGUSR1)
                return
    def require(self):
        if self.failure:raise Stopped(self.failure)
    def close(self):
        self.stop.set()
        if self.thread.ident is not None:self.thread.join(timeout=2)
    def report(self):return {'minimum_available_bytes':self.minimum_available,'global_swap_initial_pages':self.swap_initial,'global_swap_latest_pages':self.swap_latest,'stop_reason':self.failure}

def own_children(group):return {int(x) for x in (group/'cgroup.procs').read_text().split()}-{os.getpid()}

def reap_owned(group):
    for sig in [signal.SIGTERM,signal.SIGKILL]:
        for pid in own_children(group):
            try:os.kill(pid,sig)
            except ProcessLookupError:pass
        end=time.monotonic()+5
        while own_children(group) and time.monotonic()<end:time.sleep(.1)
        if not own_children(group):return
    while own_children(group):time.sleep(.2) # Unit RuntimeMax is the hard fallback; retain lease meanwhile.
