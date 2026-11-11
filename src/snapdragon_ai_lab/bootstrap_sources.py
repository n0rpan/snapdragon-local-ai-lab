"""Explicit finite source-only bootstrap: public fetch, patch, content verify. No build."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import time
import sys
from .bootstrap import project_root,relative,sha
from .bootstrap_guard import Guard,Stopped,acquire_lease,secure_directory,unit_controls,reap_owned,own_children
from .bootstrap_jobs import Runner


def git(runner,root,*args):
    return runner.call(['git','-c','credential.helper=','-c','protocol.file.allow=never','-C',str(root),*args])


def checkout(runner,destination,repository,commit):
    destination=Path(destination)
    if not re.fullmatch(r'https://github.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\.git',repository) or not re.fullmatch('[0-9a-f]{40}',commit):raise Stopped('pinned_public_git_source_required')
    if destination.exists() and (destination.is_symlink() or not destination.is_dir() or any(destination.iterdir())):raise Stopped('fresh_source_checkout_required')
    destination.mkdir(parents=True,exist_ok=True,mode=0o700)
    git(runner,destination,'init','--quiet')
    git(runner,destination,'fetch','--depth','1','--no-tags',repository,commit)
    git(runner,destination,'checkout','--detach','--quiet','FETCH_HEAD')
    if git(runner,destination,'rev-parse','HEAD').strip()!=commit:raise Stopped('public_source_commit_mismatch')
    if git(runner,destination,'status','--porcelain','--untracked-files=all').strip():raise Stopped('new_source_checkout_not_clean')
    audit_git(destination)


def audit_git(root):
    """Fresh self-contained checkouts only; audit before patch/verification commands."""
    import configparser
    root=Path(root)
    if not (root/'.git').is_dir() or (root/'.git').is_symlink():raise Stopped('self_contained_git_metadata_required')
    for path in root.rglob('*'):
        if path.is_symlink():
            if '.git' in path.relative_to(root).parts or not path.resolve().is_relative_to(root.resolve()):raise Stopped('source_symlink_escape')
        elif path.is_file() and path.stat().st_nlink!=1:raise Stopped('source_hardlinks_refused')
    cfg=configparser.ConfigParser(interpolation=None);cfg.read(root/'.git/config')
    if cfg.has_option('core','worktree') or any(x.lower().startswith('include') for x in cfg.sections()):raise Stopped('external_git_worktree_config_refused')
    for name in ['commondir','objects/info/alternates']:
        if (root/'.git'/name).exists():raise Stopped('shared_git_metadata_refused')


def apply_patch(runner,root,project,item):
    path=project/relative(item['path'])
    if sha(path)!=item['sha256']:raise Stopped('source_patch_identity_mismatch')
    audit_git(root)
    git(runner,root,'apply','--check',str(path));git(runner,root,'apply','--index','--whitespace=error-all',str(path))


def acquire_sources(recipe,source,project,runner):
    source=Path(source);project=Path(project)
    checkout(runner,source,recipe['repository'],recipe['commit'])
    for sub in recipe.get('submodules',[]):
        target=source/relative(sub['path'])
        gitlink=git(runner,source,'ls-tree','HEAD','--',sub['path']).split()
        if len(gitlink)<3 or gitlink[0]!='160000' or gitlink[2]!=sub['commit']:raise Stopped('declared_submodule_pin_mismatch')
        checkout(runner,target,sub['repository'],sub['commit'])
        for patch in sub.get('patches',[]):apply_patch(runner,target,project,patch)
    for patch in recipe.get('patches',[]):apply_patch(runner,source,project,patch)
    for name,wanted in recipe.get('required_postimages',{}).items():
        if sha(source/relative(name))!=wanted:raise Stopped('source_postimage_mismatch')
    if recipe.get('verifier'):runner.call([sys.executable,'-B',str(project/relative(recipe['verifier'])),str(source)])
    return {'source_commit':recipe['commit'],'source_verified':True,'compiled':False,'qualified':False,
        'submodule_commits':[x['commit'] for x in recipe.get('submodules',[])],
        'patch_sha256':[x['sha256'] for x in recipe.get('patches',[])]+[p['sha256'] for x in recipe.get('submodules',[]) for p in x.get('patches',[])],
        'new_local_git_commit_created':False,'private_commit_fetch_required':False}


def load_inputs(project,manifest_sha):
    path=project/'bootstrap/source-fetch-manifest.json'
    if not re.fullmatch('[0-9a-f]{64}',manifest_sha) or sha(path)!=manifest_sha:raise Stopped('reviewed_source_manifest_required')
    manifest=json.loads(path.read_text())
    for name,wanted in manifest['files'].items():
        if sha(project/relative(name))!=wanted:raise Stopped('source_executor_input_drift')
    return json.loads((project/'bootstrap/source-recipes.json').read_text())


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',action='store_true');parser.add_argument('--project',type=Path)
    parser.add_argument('--manifest-sha256');parser.add_argument('--engine',choices=['classic','images','flash'],default='flash');parser.add_argument('--work',type=Path)
    args=parser.parse_args(argv)
    if not args.run:print('Prepared source-only executor; explicit --run in the dedicated bounded unit is required.');return 0
    project=project_root(args.project)
    if not args.manifest_sha256 or args.work is None:parser.error('--manifest-sha256 and --work required')
    recipes=load_inputs(project,args.manifest_sha256);group,controls,deadline=unit_controls('sources')
    lease=acquire_lease(Path.home());guard=Guard(group,min(deadline-30,840));runner=None;report={'passed':False,'compiled':False,'qualified':False,'manifest_sha256':args.manifest_sha256,'controls':controls};started=time.monotonic();directory=None
    handlers={n:signal.signal(n,guard.signal_stop) for n in [signal.SIGTERM,signal.SIGINT,signal.SIGUSR1]}
    try:
        guard.poll(starting=True,kind='sources');guard.thread.start()
        work=args.work.absolute()
        if '..' in work.parts or work.exists() or work.is_symlink() or any(p.is_symlink() for p in work.parents):raise Stopped('fresh_source_work_directory_required')
        work.mkdir(parents=True,mode=0o700);directory=work;runner=Runner(guard,group,work)
        report.update(acquire_sources(recipes['engines'][args.engine],work/'source',project,runner));guard.require();report['passed']=True
    except BaseException as error:
        report['passed']=False;report['error']=str(error) if isinstance(error,Stopped) else type(error).__name__
    finally:
        if runner is not None:runner.close()
        if own_children(group):reap_owned(group);report['passed']=False;report['error']='orphaned_source_children'
        guard.close()
        memory=dict((k,int(v)) for k,v in (x.split() for x in (group/'memory.events').read_text().splitlines()))
        pids=dict((k,int(v)) for k,v in (x.split() for x in (group/'pids.events').read_text().splitlines()))
        report.update(total_seconds=time.monotonic()-started,resources=guard.report(),all_children_reaped=not own_children(group),actual_cgroup_peak_bytes=int((group/'memory.peak').read_text()),memory_events=memory,pids_events=pids)
        report['passed']=report['passed'] and guard.failure is None and all(memory.get(k,-1)==0 for k in ['max','oom','oom_kill']) and pids.get('max',-1)==0 and report['actual_cgroup_peak_bytes']<=1024**3 and int((group/'memory.swap.current').read_text())==0
        if directory is not None:
            path=directory/'source-result.json'
            descriptor=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
            with os.fdopen(descriptor,'w') as stream:stream.write(json.dumps(report,indent=2)+'\n');stream.flush();os.fsync(stream.fileno())
        os.close(lease)
        for number,old in handlers.items():signal.signal(number,old)
    print(json.dumps({'passed':report['passed'],'compiled':False,'qualified':False,'error':report.get('error'),'total_seconds':report['total_seconds']}));return 0 if report['passed'] else 1

if __name__=='__main__':raise SystemExit(main())
