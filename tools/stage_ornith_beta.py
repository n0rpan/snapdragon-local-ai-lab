#!/usr/bin/env python3
"""Prepare a fresh reviewed beta payload; never install, start, fetch or hash models."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from snapdragon_ai_lab.runtime import classic_beta


def sha(data):return hashlib.sha256(data).hexdigest()


def read_bound(path,wanted):
    path=Path(path)
    if not path.is_absolute() or not classic_beta.valid_sha(wanted):raise ValueError('Absolute metadata path and exact SHA required')
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_CLOEXEC)
    with os.fdopen(fd,'rb') as stream:
        before=os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_uid!=os.getuid() or before.st_size>8*1024**2:raise ValueError('Bounded owned regular metadata required')
        data=stream.read(8*1024**2+1);after=os.fstat(stream.fileno())
    if len(data)>8*1024**2 or (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)!=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns) or sha(data)!=wanted:raise ValueError('Metadata identity changed')
    return json.loads(data),data


def stage(destination,*,runtime_path,runtime_sha,agent_path,agent_sha,screen_path,screen_sha,inventory_path,inventory_sha,reviewed_agent_sha):
    runtime,runtime_raw=read_bound(runtime_path,runtime_sha)
    agent,_=read_bound(agent_path,agent_sha);screen,_=read_bound(screen_path,screen_sha);inventory,_=read_bound(inventory_path,inventory_sha)
    profile=classic_beta.qualify(agent,screen,inventory,normal_sha=agent_sha,screen_sha=screen_sha,inventory_sha=inventory_sha,reviewed_normal_sha=reviewed_agent_sha)
    if not classic_beta.eligible(profile):raise ValueError('Current model stat differs from reviewed inventory; reproof required')
    updated=classic_beta.add_profile(runtime,profile)
    destination=Path(destination)
    if not destination.is_absolute() or '..' in destination.parts or destination.exists() or destination.is_symlink() or any(p.is_symlink() for p in destination.parents):raise ValueError('Fresh absolute staging directory required')
    # A stage is source/evidence, never an installation destination.
    for live in (Path.home()/'.config',Path.home()/'.local'):
        if destination==live or destination.is_relative_to(live):raise ValueError('Live configuration installation is outside this staging tool')
    destination.mkdir(mode=0o700)
    def put(name,data):
        target=destination/name;target.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
        fd=os.open(target,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
        with os.fdopen(fd,'wb') as stream:stream.write(data)
    def document(name,value):put(name,(json.dumps(value,indent=2,allow_nan=False)+'\n').encode())
    put('rollback-runtime.json',runtime_raw)
    document('runtime.json',updated)
    document('opencode.fragment.json',classic_beta.opencode_fragment())
    for path in sorted((ROOT/'src/snapdragon_ai_lab/runtime').iterdir()):
        if path.is_file() and (path.suffix=='.py' or path.name in ('local-ai','studio')):put('runtime-source/'+path.name,path.read_bytes())
    evidence=dict(installed=False,models_started=False,models_read_or_rehashed=False,profile=classic_beta.ALIAS,default_model_before=runtime.get('selected_model'),default_model_after=updated.get('selected_model'),runtime_preimage_sha256=runtime_sha,normal13_sha256=agent_sha,screen_sha256=screen_sha,inventory_sha256=inventory_sha,root_reviewed_normal13_sha256=reviewed_agent_sha,model_stat_rechecked=True,external_prefix_store=False,rollback='Restore exact rollback-runtime.json plus backed-up source/CLI/plugin/OpenCode files under idle lease; preserve this stage and receipts')
    document('stage-proof.json',evidence)
    files={p.relative_to(destination).as_posix():{'bytes':p.stat().st_size,'sha256':sha(p.read_bytes())} for p in destination.rglob('*') if p.is_file()}
    document('stage-manifest.json',{'schema':1,'installed':False,'qualified_beta_evidence_bound':True,'files':dict(sorted(files.items()))})
    return dict(stage=str(destination),profile=classic_beta.ALIAS,installed=False,models_started=False,stage_manifest_sha256=sha((destination/'stage-manifest.json').read_bytes()))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('destination',type=Path)
    for name in ('runtime','agent','screen','inventory'):
        parser.add_argument('--'+name,type=Path,required=True);parser.add_argument('--'+name+'-sha256',required=True)
    parser.add_argument('--root-reviewed-agent-sha256',required=True);args=parser.parse_args()
    print(json.dumps(stage(args.destination,runtime_path=args.runtime,runtime_sha=args.runtime_sha256,agent_path=args.agent,agent_sha=args.agent_sha256,screen_path=args.screen,screen_sha=args.screen_sha256,inventory_path=args.inventory,inventory_sha=args.inventory_sha256,reviewed_agent_sha=args.root_reviewed_agent_sha256)))


if __name__=='__main__':main()
