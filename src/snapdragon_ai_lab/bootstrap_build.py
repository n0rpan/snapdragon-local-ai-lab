"""One guarded serial engine build from an explicitly passed source receipt.

No model execution, profile enabling or host installation. Default invocation is inert.
"""
import argparse
import json
import os
from pathlib import Path,PurePosixPath
import shutil
import signal
import sys
import tarfile
import time
from .bootstrap import project_root,relative,sha
from .bootstrap_guard import Guard,Stopped,acquire_lease,unit_controls,reap_owned,own_children
from .bootstrap_jobs import Runner
from .bootstrap_sources import audit_git,git,load_inputs
from .bootstrap_transfer import download


def extract_source(archive,destination,max_bytes=256*1024**2):
    destination=Path(destination)
    if destination.exists():raise Stopped('fresh_archive_destination_required')
    with tarfile.open(archive,'r:gz') as stream:
        members=stream.getmembers();total=0
        if len(members)>30000:raise Stopped('source_archive_member_limit')
        for item in members:
            name=PurePosixPath(item.name)
            if name.is_absolute() or '..' in name.parts or '\\' in item.name or not(item.isfile() or item.isdir()):raise Stopped('unsafe_source_archive_member')
            total+=item.size
            if total>max_bytes:raise Stopped('source_archive_size_limit')
        destination.mkdir(mode=0o700)
        for item in members:
            target=destination.joinpath(*PurePosixPath(item.name).parts)
            if item.isdir():target.mkdir(parents=True,exist_ok=True,mode=0o700);continue
            target.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
            with stream.extractfile(item) as source,target.open('xb') as out:shutil.copyfileobj(source,out,1024*1024)
            target.chmod(0o700 if item.mode&0o111 else 0o600)
    return destination


def source_matches(recipe,source,project,runner):
    audit_git(source)
    if git(runner,source,'rev-parse','HEAD').strip()!=recipe['commit']:raise Stopped('source_commit_drift')
    git(runner,source,'diff','--quiet','--ignore-submodules=all')
    changed=set(git(runner,source,'diff','--cached','--name-only','--ignore-submodules=all').splitlines())
    if changed!=set(recipe.get('required_postimages',{})):raise Stopped('source_patch_file_set_drift')
    for name,wanted in recipe.get('required_postimages',{}).items():
        if sha(source/relative(name))!=wanted:raise Stopped('source_postimage_drift')
    if recipe.get('verifier'):runner.call([sys.executable,'-B',str(project/recipe['verifier']),str(source)])
    for item in recipe.get('submodules',[]):
        child=source/relative(item['path']);audit_git(child)
        if git(runner,child,'rev-parse','HEAD').strip()!=item['commit']:raise Stopped('submodule_source_drift')
        git(runner,child,'diff','--quiet')
        if not item.get('patches') and git(runner,child,'diff','--cached','--name-only').strip():raise Stopped('unexpected_submodule_patch')


def archive_asset(kind,record):return {'id':kind,'url':record['url'],'bytes':record['bytes'],'sha256':record['sha256'],'stack_relative_path':kind+'.tar.gz'}


def prepare_archive(kind,record,work,guard,fetcher):
    cache=work/'dependency-cache';cache.mkdir(exist_ok=True,mode=0o700)
    asset=archive_asset(kind,record);fetcher(asset,cache,guard)
    archive=cache/asset['stack_relative_path']
    if archive.stat().st_size!=record['bytes'] or sha(archive)!=record['sha256']:raise Stopped('verified_dependency_archive_drift')
    return extract_source(archive,work/(kind+'-source'))


def validate_options(requested,actual,path):
    effective={}
    for key,value in requested.items():
        if key in ['CMAKE_C_COMPILER','CMAKE_CXX_COMPILER']:
            expected=shutil.which(value,path=path)
            observed=actual.get(key)
            try:matched=expected is not None and observed is not None and os.path.samefile(expected,observed)
            except OSError:matched=False
            if not matched:raise Stopped('configured_compiler_drift_'+key)
            effective[key]=str(Path(observed).resolve())
        elif actual.get(key)!=value:raise Stopped('configured_build_option_drift_'+key)
    return effective


def build_engine(recipe,source,project,work,guard,runner,catalog,fetcher=download):
    source=Path(source);project=Path(project);work=Path(work)
    copied=work/'source'
    shutil.copytree(source,copied,copy_function=shutil.copy2,symlinks=True)
    source_matches(recipe,copied,project,runner);guard.require()
    options=dict(recipe['options']);options.update(FETCHCONTENT_FULLY_DISCONNECTED='ON',FETCHCONTENT_UPDATES_DISCONNECTED='ON')
    dependencies={}
    if recipe.get('requires_kleidiai'):
        extracted=prepare_archive('kleidiai',catalog['kleidiai'],work,guard,fetcher)
        candidates=[extracted] if (extracted/'CMakeLists.txt').is_file() else [p for p in extracted.iterdir() if p.is_dir() and (p/'CMakeLists.txt').is_file()]
        if len(candidates)!=1:raise Stopped('kleidiai_source_root_unrecognized')
        options['FETCHCONTENT_SOURCE_DIR_KLEIDIAI']=str(candidates[0]);dependencies['kleidiai_sha256']=catalog['kleidiai']['sha256']
        ui=prepare_archive('llama-ui',catalog['llama_ui'],work,guard,fetcher)
        roots=[p.parent for p in ui.rglob('index.html')]
        if len(roots)!=1:raise Stopped('ui_dist_root_unrecognized')
        target=copied/'tools/ui/dist'
        if target.exists():raise Stopped('unexpected_existing_ui_dist')
        shutil.copytree(roots[0],target,copy_function=shutil.copy2)
        dependencies['llama_ui_sha256']=catalog['llama_ui']['sha256']
    build=work/'build'
    if build.exists():raise Stopped('fresh_build_directory_required')
    command=['cmake','-S',str(copied),'-B',str(build),'-G','Ninja']+['-D'+key+'='+value for key,value in sorted(options.items())]
    # Isolate CMake/build from all network, including file(DOWNLOAD), not only FetchContent.
    runner.call(['unshare','--user','--map-root-user','--net','--',*command])
    actual={}
    for line in (build/'CMakeCache.txt').read_text().splitlines():
        if '=' in line and not line.startswith(('#','//')):
            key,value=line.split('=',1);actual[key.split(':',1)[0]]=value
    effective_compilers=validate_options(options,actual,getattr(runner,'env',{}).get('PATH',os.environ.get('PATH','/usr/bin:/bin')))
    runner.call(['unshare','--user','--map-root-user','--net','--','cmake','--build',str(build),'--parallel','1','--target',*recipe['targets']])
    guard.require()
    binaries={name:sha(build/relative(name)) for name in recipe['binaries']}
    return {'built':True,'qualified':False,'profiles_enabled':False,'models_run':False,'build_options':options,'effective_compiler_paths':effective_compilers,'binary_sha256':binaries,'dependencies':dependencies}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',action='store_true');parser.add_argument('--project',type=Path)
    parser.add_argument('--manifest-sha256');parser.add_argument('--source-manifest-sha256');parser.add_argument('--source-receipt-sha256');parser.add_argument('--source-work',type=Path);parser.add_argument('--work',type=Path)
    parser.add_argument('--engine',choices=['classic','images','flash'],default='classic');args=parser.parse_args(argv)
    if not args.run:print('Prepared build executor; explicit --run in a dedicated bounded build unit is required.');return 0
    if any(x is None for x in [args.manifest_sha256,args.source_manifest_sha256,args.source_receipt_sha256,args.source_work,args.work]):parser.error('both manifest hashes, --source-work and fresh --work required')
    project=project_root(args.project);manifest_path=project/'bootstrap/build-manifest.json'
    if sha(manifest_path)!=args.manifest_sha256:raise Stopped('reviewed_build_manifest_required')
    manifest=json.loads(manifest_path.read_text())
    for name,wanted in manifest['files'].items():
        if sha(project/relative(name))!=wanted:raise Stopped('build_executor_input_drift')
    catalog=load_inputs(project,args.source_manifest_sha256)
    if sha(args.source_work/'source-result.json')!=args.source_receipt_sha256:raise Stopped('source_receipt_identity_mismatch')
    source_result=json.loads((args.source_work/'source-result.json').read_text())
    if source_result.get('passed')is not True or source_result.get('source_verified')is not True or source_result.get('manifest_sha256')!=args.source_manifest_sha256 or source_result.get('source_commit')!=catalog['engines'][args.engine]['commit']:raise Stopped('passed_matching_source_receipt_required')
    group,controls,deadline=unit_controls('build');lease=acquire_lease(Path.home());guard=Guard(group,min(deadline-60,2600));runner=None;directory=None;started=time.monotonic()
    report={'passed':False,'built':False,'qualified':False,'models_run':False,'profiles_enabled':False,'manifest_sha256':args.manifest_sha256,'source_receipt_sha256':sha(args.source_work/'source-result.json'),'controls':controls}
    handlers={n:signal.signal(n,guard.signal_stop) for n in [signal.SIGTERM,signal.SIGINT,signal.SIGUSR1]}
    try:
        guard.poll(starting=True,kind='build');guard.thread.start();work=args.work.absolute()
        if '..' in work.parts or work.exists() or work.is_symlink() or any(p.is_symlink() for p in work.parents):raise Stopped('fresh_build_work_directory_required')
        work.mkdir(parents=True,mode=0o700);directory=work;runner=Runner(guard,group,work)
        # Version commands are inside the explicit native build window, never preparation.
        report['tool_versions']={name:runner.call([name,'--version']).splitlines()[0] for name in ['git','cmake','ninja','cc','c++']}
        report.update(build_engine(catalog['engines'][args.engine],args.source_work/'source',project,work,guard,runner,catalog));guard.require();report['passed']=True
    except BaseException as error:report['passed']=False;report['error']=str(error) if isinstance(error,Stopped) else type(error).__name__
    finally:
        if runner is not None:runner.close()
        if own_children(group):reap_owned(group);report['passed']=False;report['error']='orphaned_build_children'
        guard.close();memory=dict((k,int(v)) for k,v in (line.split() for line in (group/'memory.events').read_text().splitlines()));pids=dict((k,int(v)) for k,v in (line.split() for line in (group/'pids.events').read_text().splitlines()))
        report.update(total_seconds=time.monotonic()-started,resources=guard.report(),all_children_reaped=not own_children(group),actual_cgroup_peak_bytes=int((group/'memory.peak').read_text()),memory_events=memory,pids_events=pids)
        report['passed']=report['passed'] and guard.failure is None and all(memory.get(k,-1)==0 for k in ['max','oom','oom_kill']) and pids.get('max',-1)==0 and report['actual_cgroup_peak_bytes']<=4*1024**3 and int((group/'memory.swap.current').read_text())==0
        if directory is not None:
            fd=os.open(directory/'build-result.json',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
            with os.fdopen(fd,'w') as stream:stream.write(json.dumps(report,indent=2)+'\n');stream.flush();os.fsync(stream.fileno())
        os.close(lease)
        for number,old in handlers.items():signal.signal(number,old)
    print(json.dumps({'passed':report['passed'],'built':report['built'],'qualified':False,'error':report.get('error'),'total_seconds':report['total_seconds']}));return 0 if report['passed'] else 1

if __name__=='__main__':raise SystemExit(main())
