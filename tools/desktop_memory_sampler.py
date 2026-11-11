#!/usr/bin/env python3
"""Numeric host/service memory samples; never open prompts, sessions or images."""
import argparse,datetime,json,os,re,stat,subprocess,time
from pathlib import Path

def meminfo():
 keys={'MemTotal','MemAvailable','SwapTotal','SwapFree','AnonPages','Cached','Slab','SReclaimable'}
 return {line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.split(':')[0] in keys}

def psi():
 return {line.split()[0]:{k:float(v) if k.startswith('avg') else int(v) for k,v in (field.split('=') for field in line.split()[1:])} for line in Path('/proc/pressure/memory').read_text().splitlines()}

def leased(path):
 info=path.lstat()
 if not stat.S_ISREG(info.st_mode) or info.st_uid!=os.getuid():return True
 identity=(os.major(info.st_dev),os.minor(info.st_dev),info.st_ino)
 for line in Path('/proc/locks').read_text().splitlines():
  for field in line.split():
   if re.fullmatch('[0-9a-fA-F]+:[0-9a-fA-F]+:[0-9]+',field):
    major,minor,inode=field.split(':')
    if (int(major,16),int(minor,16),int(inode))==identity:return True
 return False

def applications():
 counts={}
 for node in Path('/proc').iterdir():
  if not node.name.isdecimal():continue
  try:name=(node/'comm').read_text().strip()
  except (OSError,UnicodeError):continue
  if name in {'Hyprland','codex','claude','foot','kitty','ghostty'} or name.startswith(('chrom','firefox','brave','msedge')):counts[name]=counts.get(name,0)+1
 return counts

def service_memory():
 r=subprocess.run(['systemctl','--user','show','local-ai-chat.service','-p','ControlGroup','--value'],capture_output=True,text=True,timeout=2)
 group=r.stdout.strip()
 if r.returncode or not group.startswith('/') or '..' in Path(group).parts:return {}
 path=Path('/sys/fs/cgroup')/group.lstrip('/')
 try:
  values={k:int(v) for k,v in (line.split() for line in (path/'memory.stat').read_text().splitlines())}
  return dict(current_bytes=int((path/'memory.current').read_text()),swap_bytes=int((path/'memory.swap.current').read_text()),**{k:values[k] for k in ['anon','file','kernel','file_mapped'] if k in values})
 except OSError:return {}

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--seconds',type=int,default=43200);p.add_argument('--interval',type=int,default=30);a=p.parse_args()
 if not a.output.is_absolute() or a.output.is_symlink() or not 10<=a.interval<=300 or not 60<=a.seconds<=86400:raise ValueError('bounded absolute sample output required')
 a.output.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
 fd=os.open(a.output,os.O_WRONLY|os.O_APPEND|os.O_CREAT|os.O_NOFOLLOW,0o600)
 lease=Path.home()/'.local/state/local-ai/inference.lock';maintenance=Path.home()/'.local/state/local-ai/host-maintenance.flag';end=time.monotonic()+a.seconds
 with os.fdopen(fd,'a') as out:
  while time.monotonic()<end:
   try:
    memory=meminfo();busy=leased(lease);maint=maintenance.exists();apps=applications()
    row=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),host=memory,psi=psi(),inference_lease_busy=busy,maintenance=maint,desktop_floor_eligible=not busy and not maint,applications=apps,service=service_memory(),browser_present=any(n.startswith(('chrom','firefox','brave','msedge')) for n in apps))
   except (OSError,ValueError,subprocess.SubprocessError) as error:row=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),sample_error=type(error).__name__,desktop_floor_eligible=False)
   out.write(json.dumps(row)+'\n');out.flush();time.sleep(min(a.interval,max(0,end-time.monotonic())))
if __name__=='__main__':main()
