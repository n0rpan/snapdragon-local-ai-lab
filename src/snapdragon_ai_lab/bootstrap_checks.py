"""Four fixed model-free native checks after a passed fresh build. Inert by default."""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import stat
import time
from .bootstrap import project_root,relative,sha
from .bootstrap_guard import Guard,Stopped,acquire_lease,unit_controls,own_children,reap_owned
from .bootstrap_jobs import Runner

TESTS=('bmoe_config_test','bmoe_params_test','bmoe_mapping_release_test','bmoe_file_reader_test')

def regular_executable(path,wanted):
    value=path.lstat()
    if path.parent.is_symlink() or not stat.S_ISREG(value.st_mode) or value.st_uid!=os.getuid() or value.st_nlink!=1 or not value.st_mode&0o100 or not re.fullmatch('[0-9a-f]{64}',wanted) or sha(path)!=wanted:raise Stopped('native_check_binary_identity_mismatch_'+path.name)


def run_checks(build_work,build_result,expected,runner):
    build=Path(build_work)/'build'
    if set(expected)!=set(TESTS):raise Stopped('exact_four_model_free_test_hashes_required')
    if build_result.get('passed')is not True or build_result.get('built')is not True or build_result.get('models_run')is not False:raise Stopped('passed_fresh_build_required')
    if set(build_result.get('binary_sha256',{}))!={'bin/bmoe-cli','bin/bmoe-server'}:raise Stopped('fresh_flash_binary_pair_required')
    for name,wanted in build_result['binary_sha256'].items():
        regular_executable(build/relative(name),wanted)
    rows=[]
    for name in TESTS:
        binary=build/'bin'/name;regular_executable(binary,expected[name]);start=time.monotonic()
        before=getattr(runner,'logged',None);output=runner.call([str(binary)])
        if before is not None:
            with (runner.directory/'commands.log').open('rb') as stream:
                stream.seek(before);output=stream.read(runner.logged-before).decode(errors='replace')
        if re.search(r'^\s*(?:SKIP(?:PED)?|NOT RUN)\b',output,re.I|re.M):raise Stopped('native_check_skip_rejected_'+name)
        regular_executable(binary,expected[name]);rows.append({'name':name,'exit':0,'skipped':False,'seconds':time.monotonic()-start,'binary_sha256':expected[name],'stdout_stderr_bytes_captured':len(output.encode())})
    return {'model_free_checks_passed':True,'rows':rows,'models_run':False,'qualified':False,'profiles_enabled':False}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',action='store_true');parser.add_argument('--project',type=Path)
    parser.add_argument('--manifest-sha256');parser.add_argument('--build-work',type=Path);parser.add_argument('--build-receipt-sha256')
    parser.add_argument('--test-hashes',type=Path);parser.add_argument('--test-hashes-sha256');parser.add_argument('--work',type=Path)
    args=parser.parse_args(argv)
    if not args.run:print('Prepared fixed model-free checks; explicit scheduled --run required.');return 0
    if any(x is None for x in [args.manifest_sha256,args.build_work,args.build_receipt_sha256,args.test_hashes,args.test_hashes_sha256,args.work]):parser.error('reviewed manifest, build receipt, test hashes and fresh work required')
    project=project_root(args.project);manifest_path=project/'bootstrap/checks-manifest.json'
    if sha(manifest_path)!=args.manifest_sha256:raise Stopped('reviewed_checks_manifest_required')
    manifest=json.loads(manifest_path.read_text())
    for name,wanted in manifest['files'].items():
        if sha(project/relative(name))!=wanted:raise Stopped('checks_executor_input_drift')
    receipt=args.build_work/'build-result.json'
    if sha(receipt)!=args.build_receipt_sha256 or sha(args.test_hashes)!=args.test_hashes_sha256:raise Stopped('native_check_input_receipt_drift')
    result=json.loads(receipt.read_text());expected=json.loads(args.test_hashes.read_text())
    if result.get('manifest_sha256')!=manifest['build_manifest_sha256']:raise Stopped('unexpected_fresh_build_recipe')
    # Existing source-kind unit is deliberately reused; these checks initialize no GGUF.
    group,controls,deadline=unit_controls('sources');lease=acquire_lease(Path.home());guard=Guard(group,min(deadline-30,840));runner=None;directory=None;start=time.monotonic()
    report={'passed':False,'model_free_checks_passed':False,'models_run':False,'qualified':False,'profiles_enabled':False,'controls':controls,'manifest_sha256':args.manifest_sha256,'build_receipt_sha256':args.build_receipt_sha256,'test_hashes_sha256':args.test_hashes_sha256}
    handlers={n:signal.signal(n,guard.signal_stop) for n in [signal.SIGTERM,signal.SIGINT,signal.SIGUSR1]}
    try:
        guard.poll(starting=True);guard.thread.start();work=args.work.absolute()
        if '..' in work.parts or work.exists() or work.is_symlink() or any(p.is_symlink() for p in work.parents):raise Stopped('fresh_native_checks_work_required')
        work.mkdir(parents=True,mode=0o700);directory=work;runner=Runner(guard,group,work)
        report.update(run_checks(args.build_work,result,expected,runner));guard.require();report['passed']=True
    except BaseException as error:report['passed']=False;report['error']=str(error) if isinstance(error,Stopped) else type(error).__name__
    finally:
        if runner is not None:runner.close()
        if own_children(group):reap_owned(group);report['passed']=False;report['error']='orphaned_native_check_children'
        guard.close();memory=dict((k,int(v)) for k,v in (line.split() for line in (group/'memory.events').read_text().splitlines()));pids=dict((k,int(v)) for k,v in (line.split() for line in (group/'pids.events').read_text().splitlines()))
        report.update(total_seconds=time.monotonic()-start,resources=guard.report(),all_children_reaped=not own_children(group),actual_cgroup_peak_bytes=int((group/'memory.peak').read_text()),memory_events=memory,pids_events=pids)
        report['passed']=report['passed'] and guard.failure is None and all(memory.get(k,-1)==0 for k in ['max','oom','oom_kill']) and pids.get('max',-1)==0 and report['actual_cgroup_peak_bytes']<=1024**3 and int((group/'memory.swap.current').read_text())==0
        if directory is not None:
            fd=os.open(directory/'native-checks-result.json',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
            with os.fdopen(fd,'w') as stream:stream.write(json.dumps(report,indent=2)+'\n');stream.flush();os.fsync(stream.fileno())
        os.close(lease)
        for number,old in handlers.items():signal.signal(number,old)
    print(json.dumps({'passed':report['passed'],'model_free_checks_passed':report['model_free_checks_passed'],'qualified':False,'models_run':False,'error':report.get('error')}));return 0 if report['passed'] else 1

if __name__=='__main__':raise SystemExit(main())
