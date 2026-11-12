"""Guarded resumable selected-weight transfer. No build, installation or readiness change."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import re
from urllib.parse import quote
import time
from .bootstrap import DEFAULT_IMAGE,IMAGE_SETS,asset_plan,project_root,relative,sha
from .bootstrap_guard import Guard,Stopped,acquire_lease,secure_directory,unit_controls,own_children,reap_owned
from .bootstrap_transfer import download,atomic_json,owned_regular


def selected_assets(project,include_dense=False,include_q4=False,include_flash=False,image=DEFAULT_IMAGE):
    result=asset_plan(project,include_dense,image)
    document=json.loads((Path(project)/'dependencies/model-artifacts.json').read_text())
    ids=set()
    if include_q4:ids.add('z-image-turbo-q4')
    if include_flash:ids.add('qwen38-flash-next-ud-iq3-xxs')
    seen=set()
    for item in document['models']+document['image_assets']:
        if item['id'] not in ids:continue
        seen.add(item['id']);files=item.get('files') or [{'filename':item['file'],'bytes':item['bytes'],'sha256':item['sha256']}]
        for index,file in enumerate(files,1):
            relative(file['filename'])
            if not re.fullmatch('[0-9a-f]{40}',item['revision']) or not re.fullmatch('[0-9a-f]{64}',file['sha256']) or type(file['bytes'])is not int or file['bytes']<=0:raise Stopped('immutable_optional_asset_required')
            prefix='https://huggingface.co/'+item['repository']
            target_prefix={'z-image-turbo-q4':'models/images/','qwen38-flash-next-ud-iq3-xxs':'models/llm/Qwen3.8-Flash-Next/'}
            target=target_prefix[item['id']]+file['filename']
            result.append({'id':item['id']+('-shard-'+str(index) if len(files)>1 else ''),'selection_id':item['id'],'url':prefix+'/resolve/'+item['revision']+'/'+quote(file['filename'],safe='/'),'bytes':file['bytes'],'sha256':file['sha256'],'stack_relative_path':target,'license_card_url':prefix+'/blob/'+item['revision']+'/README.md','qualified':False})
    if seen!=ids:raise Stopped('optional_asset_missing_from_manifest')
    return result


def space_needed(assets,root):
    total=0;root=Path(root)
    for asset in assets:
        target=root/relative(asset['stack_relative_path'])
        if any(parent.is_symlink() for parent in target.parents if parent==root or root in parent.parents):raise Stopped('download_subpath_symlink')
        if target.is_file() and not target.is_symlink() and target.stat().st_size==asset['bytes']:continue
        offset=0;partial=target.with_name('.'+target.name+'.partial');metadata=target.with_name('.'+target.name+'.partial.json')
        if partial.is_file() and not partial.is_symlink() and metadata.is_file() and not metadata.is_symlink():
            fd=os.open(metadata,os.O_RDONLY|os.O_NOFOLLOW|os.O_CLOEXEC)
            with os.fdopen(fd,'r') as stream:
                owned_regular(stream.fileno());raw=stream.read(16385)
            if len(raw)>16384:raise Stopped('metadata_too_large')
            value=json.loads(raw)
            if (value.get('url'),value.get('expected_bytes'),value.get('expected_sha256'))==(asset['url'],asset['bytes'],asset['sha256']):
                candidate=value.get('completed_bytes')
                if type(candidate)is int and 0<=candidate<=partial.stat().st_size<=asset['bytes']:offset=candidate
        total+=asset['bytes']-offset
    return total+1024**3


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',action='store_true');parser.add_argument('--project',type=Path)
    parser.add_argument('--manifest-sha256');parser.add_argument('--work',type=Path);parser.add_argument('--include-dense',action='store_true')
    parser.add_argument('--include-q4',action='store_true');parser.add_argument('--include-flash',action='store_true')
    parser.add_argument('--image-model',choices=sorted(IMAGE_SETS),default=DEFAULT_IMAGE,help='default flux2-klein; z-image is an explicit opt-in')
    parser.add_argument('--only',action='append',help='Explicit selected asset ID; otherwise the four default MoE and image assets')
    args=parser.parse_args(argv)
    if not args.run:print('Prepared selected fetcher; explicit --run in the bounded fetch unit is required.');return 0
    project=project_root(args.project)
    if args.work is None or not args.manifest_sha256:parser.error('--work and reviewed --manifest-sha256 required')
    manifest_path=project/'bootstrap/download-manifest.json'
    if sha(manifest_path)!=args.manifest_sha256:raise Stopped('reviewed_download_manifest_required')
    manifest=json.loads(manifest_path.read_text())
    for name,wanted in manifest['files'].items():
        if sha(project/relative(name))!=wanted:raise Stopped('selected_fetch_input_drift')
    available=selected_assets(project,args.include_dense,args.include_q4,args.include_flash,args.image_model)
    if args.only:
        wanted=set(args.only)
        if not wanted<=({x['id'] for x in available}|{x.get('selection_id',x['id']) for x in available}):raise Stopped('asset_not_in_selected_default_or_dense_plan')
        available=[x for x in available if x['id'] in wanted or x.get('selection_id',x['id']) in wanted]
    group,controls,deadline=unit_controls('fetch');lease=acquire_lease(Path.home());guard=Guard(group,min(deadline-60,3600));work=None;start=time.monotonic()
    report={'passed':False,'qualified':False,'installed':False,'manifest_sha256':args.manifest_sha256,'controls':controls,'assets':[]}
    handlers={n:signal.signal(n,guard.signal_stop) for n in [signal.SIGTERM,signal.SIGINT,signal.SIGUSR1]}
    try:
        guard.poll(starting=True);guard.thread.start();candidate=args.work.absolute()
        if '..' in candidate.parts or candidate.is_symlink() or any(p.is_symlink() for p in candidate.parents):raise Stopped('safe_owned_download_root_required')
        if not candidate.exists():candidate.mkdir(parents=True,mode=0o700)
        secure_directory(candidate);work=candidate
        if shutil.disk_usage(work).free<space_needed(available,work):raise Stopped('insufficient_selected_download_space_plus1gib')
        for asset in available:report['assets'].append(download(asset,work,guard))
        guard.require();report['passed']=True
    except BaseException as error:report['passed']=False;report['error']=str(error) if isinstance(error,Stopped) else type(error).__name__
    finally:
        if own_children(group):reap_owned(group);report['passed']=False;report['error']='unexpected_download_children'
        guard.close();memory=dict((k,int(v)) for k,v in (line.split() for line in (group/'memory.events').read_text().splitlines()));pids=dict((k,int(v)) for k,v in (line.split() for line in (group/'pids.events').read_text().splitlines()))
        report.update(total_seconds=time.monotonic()-start,resources=guard.report(),actual_cgroup_peak_bytes=int((group/'memory.peak').read_text()),memory_events=memory,pids_events=pids,all_children_reaped=not own_children(group))
        # Download file-cache reclaim at memory.max is recorded, not mislabeled OOM.
        report['passed']=report['passed'] and guard.failure is None and all(memory.get(k,-1)==0 for k in ['oom','oom_kill']) and pids.get('max',-1)==0 and report['actual_cgroup_peak_bytes']<=1024**3 and int((group/'memory.swap.current').read_text())==0
        if work is not None:atomic_json(work/('fetch-attempt-'+str(time.time_ns())+'.json'),report)
        os.close(lease)
        for number,old in handlers.items():signal.signal(number,old)
    print(json.dumps({'passed':report['passed'],'qualified':False,'installed':False,'verified_assets':len(report['assets']),'error':report.get('error'),'total_seconds':report['total_seconds']}));return 0 if report['passed'] else 1

if __name__=='__main__':raise SystemExit(main())
