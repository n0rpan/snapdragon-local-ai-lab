"""Portable fresh-prefix preparation. No host install, service, build or network CLI.

Run with ``python -m snapdragon_ai_lab.bootstrap`` from a source checkout. A staged
bundle is NOT an end-to-end qualification or an installed runtime.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import tempfile
from urllib.parse import quote, urlsplit

MOE='ornith-35b-a3b-local'
# The dense Qwen3.8 27B is retired (v0.1.2): no bootstrap path selects or stages it.
RETIRED_DENSE_FLAG='--include-dense was removed: the dense Qwen3.8 27B model was retired in v0.1.2'
# Image sets. The default is FLUX.2 klein 4B (Apache-2.0 for the 4B weights, text encoder and VAE),
# the same files Studio's klein path uses. Z-Image is an explicit opt-in (--image-model z-image).
# Neither set needs SDK, Hexagon or NPU artefacts; the default plan is CPU-only.
IMAGE_SETS={'flux2-klein':('flux-2-klein-4b-q8','flux-2-klein-text-encoder','flux-2-klein-vae'),
            'z-image':('z-image-turbo-q8','z-image-text-encoder','z-image-vae-compatible-candidate')}
DEFAULT_IMAGE='flux2-klein'
DEFAULT_IDS=('ornith-1.5-35b-a3b-apex-mtp-compact',)+IMAGE_SETS[DEFAULT_IMAGE]
RUNTIME=('controller.py','ctl_seams.py','ctl_env.py','ctl_host.py','ctl_repack.py','ctl_status.py','ctl_models.py','ctl_cache.py','ctl_lease.py','ctl_engines.py','ctl_lifecycle.py','ctl_manager.py','ctl_routes.py','ctl_proxy.py','ctl_proxy_prep.py','ctl_proxy_stream.py','ctl_proxy_ledger.py','ctl_http.py','classic_beta.py','memory_policy.py','profile_policy.py','oxe_runtime.py','flash_cache_pref.py','prefix_plan.py','prefix_store.py','bigmoe_adapter.py','bigmoe_runtime.py','exec_no_thp.py','exec_in_scope.py','slim_request.py','engine_diagnostics.py','client_disconnect.py','engine_ledger.py','explicit_route.py','harness_compat.py','harness_registry.py','openai_api.py','request_ledger.py','cache_miss.py','state_cache.py','cache_turn.py','model_catalog.py','storage_view.py','lan_share.py','tool_output_cap.py','image_npu.py','image_npu_worker.py','image_slot.py','image_size_gate.py','studio_page.py','studio_pixels.py','studio_context.py','studio_worker.py','studio_plan_generate.py','studio_plan_edit.py','studio_oxi.py','studio')


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk:=stream.read(1024*1024):digest.update(chunk)
    return digest.hexdigest()


def relative(raw):
    path=PurePosixPath(raw)
    if not raw or path.is_absolute() or any(p in ('..','.') for p in raw.split('/')) or '\\' in raw:
        raise ValueError('safe relative path required')
    return Path(*path.parts)


def fresh(path):
    path=Path(path).absolute()
    if path.exists() or path.is_symlink():raise ValueError('fresh destination required')
    if any(parent.is_symlink() for parent in path.parents):raise ValueError('symlinked destination parent refused')
    return path


def project_root(raw=None):
    root=Path(raw) if raw is not None else Path(__file__).resolve().parents[2]
    if not (root/'dependencies/model-artifacts.json').is_file():
        raise ValueError('public source checkout required; pass --project (wheel asset packaging pending)')
    return root.resolve()


def asset_plan(project,image=DEFAULT_IMAGE):
    if image not in IMAGE_SETS:raise ValueError('unknown image model: '+str(image))
    document=json.loads((project/'dependencies/model-artifacts.json').read_text())
    selected={DEFAULT_IDS[0],*IMAGE_SETS[image]}
    result=[]
    targets={'ornith-1.5-35b-a3b-apex-mtp-compact':'models/llm/Ornith-1.5-35B-A3B-APEX-MTP-Compact.gguf',
             'flux-2-klein-4b-q8':'models/images/flux2-klein/flux-2-klein-4b-Q8_0.gguf',
             'flux-2-klein-text-encoder':'models/images/flux2-klein/Qwen3-4B-Q8_0.gguf',
             'flux-2-klein-vae':'models/images/flux2-klein/flux2-vae.safetensors',
             'z-image-turbo-q8':'models/images/z_image_turbo-Q8_0.gguf',
             'z-image-text-encoder':'models/images/Qwen3-4B-Instruct-2507-Q4_K_M.gguf',
             'z-image-vae-compatible-candidate':'models/images/vae/vae.safetensors'}
    for item in document['models']+document['image_assets']:
        if item['id'] not in selected:continue
        files=item.get('files') or [{'filename':item['file'],'bytes':item['bytes'],'sha256':item['sha256']}]
        if len(files)!=1:raise ValueError('selected default must be a single artifact')
        file=files[0];relative(file['filename']);relative(targets[item['id']])
        if not re.fullmatch('[0-9a-f]{40}',item['revision']) or not re.fullmatch('[0-9a-f]{64}',file['sha256']) or type(file['bytes']) is not int or file['bytes']<=0:
            raise ValueError('immutable artifact identity required')
        prefix='https://huggingface.co/'+item['repository']
        result.append({'id':item['id'],'url':prefix+'/resolve/'+item['revision']+'/'+quote(file['filename'],safe='/'),
            'bytes':file['bytes'],'sha256':file['sha256'],'stack_relative_path':targets[item['id']],
            'license_card_url':prefix+'/blob/'+item['revision']+'/README.md','license_separate_from_wrapper':True})
    if {x['id'] for x in result}!=selected:raise ValueError('selected artifact missing from manifest')
    return result


def plan(project,home,image=DEFAULT_IMAGE):
    home=Path(home)
    if not home.is_absolute() or home==Path('/') or '..' in home.parts or any(c in str(home) for c in '\n\r\x00'):
        raise ValueError('absolute non-root single-line target home required')
    patches=json.loads((project/'patches/manifest.json').read_text())['patches']
    engines=[]
    for name in ['llama-small-batch-pool.patch','sd-cpu-repack.patch']:
        entry=next(x for x in patches if x['file']==name)
        if sha(project/'patches'/name)!=entry['sha256']:raise ValueError('published patch identity mismatch')
        engines.append({'repository':entry['upstream'],'commit':entry['upstream_commit'],'patch':name,'patch_sha256':entry['sha256']})
    assets=asset_plan(project,image)
    return {'schema':1,'status':'prepared_not_installed_not_qualified','target_home':str(home),'assets':assets,'selected_download_bytes':sum(x['bytes'] for x in assets),
        'model_manifest_sha256':sha(project/'dependencies/model-artifacts.json'),'engines':engines,
        'default_model':MOE,'image_model':image,'images':'q8_cpu_lazy_private',
        'npu_artifacts_required':False,'cpu_fallback':'sd-cpu-repack CPU path; NPU image path stays disabled until separately qualified',
        'network_or_native_execution':False,'end_to_end_parity_passed':False,
        'remaining_gates':['guarded selected fetch/source build','fresh binary/model identity qualification','transactional host install and OpenCode merge','real panel installation','isolated-user full task/image/lifecycle verification','rollback/uninstall verification']}


def stage(destination,home,project=None,image=DEFAULT_IMAGE):
    project=project_root(project);destination=fresh(destination);record=plan(project,home,image)
    payload={};home=Path(home)
    def put(name,data,mode=0o600):
        relative(name)
        if name in payload:raise ValueError('duplicate payload path')
        payload[name]=(data if isinstance(data,bytes) else data.encode(),mode)
    def encode(value):return json.dumps(value,indent=2)+'\n'
    template=json.loads((project/'examples/runtime/llm-runtime.json').read_text().replace('@HOME@',str(home)))
    allowed={MOE}
    if not allowed<=set(template['profiles']):raise ValueError('example runtime has no '+MOE+' profile')
    profiles={name:copy.deepcopy(value) for name,value in template['profiles'].items() if name in allowed}
    for value in profiles.values():value.update(ready=False,server_sha256='')
    config={**template,**profiles[MOE],'selected_model':MOE,'profiles':profiles,'ready':False,'server_sha256':''}
    put('.config/local-ai/llm/runtime.json',encode(config))
    for source in sorted((project/'examples/runtime/images').glob('*.json')):
        value=json.loads(source.read_text().replace('@HOME@',str(home)))
        if source.name=='cpu-profile.json':value.update(enabled=False,validated=False,sha256='',models=['q8'])
        if source.name=='gpu-profile.json':value.update(enabled=False,validated=False)
        put('.config/local-ai/images/'+source.name,encode(value))
    opencode=json.loads((project/'examples/runtime/opencode-v2.json').read_text())
    models=opencode['providers']['local-qwen']['models']
    opencode['providers']['local-qwen']['models']={k:v for k,v in models.items() if k in allowed or k=='qwen-local'}
    # An inactive fragment: do not replace unrelated providers/settings.
    put('.config/opencode/local-ai-bootstrap.fragment.json',encode(opencode))
    for name in RUNTIME:put('.local/share/ai-stack/runtime/'+name,(project/'src/snapdragon_ai_lab/runtime'/name).read_bytes())
    for name in ['local-ai','launch_local_plan.py']:
        put('.local/bin/'+name,(project/'src/snapdragon_ai_lab/runtime'/name).read_bytes(),0o700)
    for source in sorted((project/'examples/runtime/units').glob('*.service')):
        put('.config/systemd/user/'+source.name,source.read_bytes())
    for source in sorted((project/'src/snapdragon_ai_lab/omarchy_plugin').rglob('*')):
        if source.is_file():put('.config/omarchy/plugins/snapdragon.local-ai/'+source.relative_to(project/'src/snapdragon_ai_lab/omarchy_plugin').as_posix(),source.read_bytes())
    put('.config/local-ai/npu-status.json',encode({'ready':False,'state':'disabled','bootstrap_qualified':False}))
    put('.local/state/local-ai/inference.lock',b'')
    for name in ['LICENSE','NOTICE']:put('.local/share/ai-stack/notices/wrapper/'+name,(project/name).read_bytes())
    # All content is validated before creating the fresh bundle. No runtime module is imported.
    destination.mkdir(parents=True,mode=0o700)
    try:
        inventory={}
        for name,(data,mode) in payload.items():
            target=destination/'payload'/relative(name);target.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
            target.write_bytes(data);target.chmod(mode)
            inventory[name]={'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),'mode':mode}
        for name in ['Work','Pictures/sd/_uploads','.local/state/local-ai/prefix-cache','.local/share/ai-stack/releases/llama-cpu','.local/share/ai-stack/releases/sd-cpu-repack','.local/share/ai-stack/models/llm','.local/share/ai-stack/models/images/vae']:
            (destination/'payload'/relative(name)).mkdir(parents=True,exist_ok=True,mode=0o700)
        directories=[]
        for directory in (destination/'payload').rglob('*'):
            if directory.is_dir():directory.chmod(0o700);directories.append(directory.relative_to(destination/'payload').as_posix())
        (destination/'bootstrap-plan.json').write_text(encode(record));(destination/'bootstrap-plan.json').chmod(0o600)
        manifest={'schema':1,'files':inventory,'directories':sorted(directories),'plan_sha256':sha(destination/'bootstrap-plan.json'),'installed':False,'end_to_end_parity_passed':False}
        (destination/'stage-manifest.json').write_text(encode(manifest));(destination/'stage-manifest.json').chmod(0o600)
    except BaseException:
        # Leave any incomplete NEW stage visible; never delete or alter a user's existing tree.
        raise
    return verify_stage(destination)


def verify_stage(destination):
    destination=Path(destination)
    manifest=json.loads((destination/'stage-manifest.json').read_text())
    if sha(destination/'bootstrap-plan.json')!=manifest['plan_sha256']:raise ValueError('stage plan changed')
    actual=set();directories=set()
    for path in (destination/'payload').rglob('*'):
        if path.is_symlink():raise ValueError('stage symlink refused')
        if path.is_dir():
            directories.add(path.relative_to(destination/'payload').as_posix())
            if path.stat().st_mode&0o777!=0o700:raise ValueError('staged directory is not private')
        if path.is_file():actual.add(path.relative_to(destination/'payload').as_posix())
    if directories!=set(manifest['directories']):raise ValueError('unexpected or missing staged directory')
    if actual!=set(manifest['files']):raise ValueError('unexpected or missing staged file')
    for name,entry in manifest['files'].items():
        path=destination/'payload'/relative(name)
        if path.stat().st_size!=entry['bytes'] or sha(path)!=entry['sha256'] or path.stat().st_mode&0o777!=entry['mode']:
            raise ValueError('staged payload identity changed')
    return {'model_free_stage_verified':True,'files':len(actual),'installed':False,'end_to_end_parity_passed':False}


def verified_fetch(url,destination,expected_bytes,expected_sha256,*,opener,check):
    """Injected fetch primitive; NOT a CLI downloader. Caller owns lease/power/deadline.

    Production network/resume guard integration is deliberately a remaining gate.
    ``check`` must abort before further writes; tiny tests inject an in-memory source.
    """
    target=fresh(destination)
    if urlsplit(url).scheme!='https' or urlsplit(url).username or urlsplit(url).password:raise ValueError('HTTPS without embedded credentials required')
    if type(expected_bytes)is not int or expected_bytes<=0 or not re.fullmatch('[0-9a-f]{64}',expected_sha256):raise ValueError('size/SHA required')
    descriptor,name=tempfile.mkstemp(prefix='.bootstrap-partial-',dir=target.parent)
    try:
        digest=hashlib.sha256();count=0
        with os.fdopen(descriptor,'wb') as stream,opener(url) as response:
            final=urlsplit(response.geturl())
            if final.scheme!='https' or not final.hostname or final.username or final.password:raise ValueError('HTTPS final URL without credentials required')
            while True:
                check();block=response.read(1024*1024);check()
                if not block:break
                count+=len(block)
                if count>expected_bytes:raise ValueError('artifact exceeds expected size')
                stream.write(block);digest.update(block)
            stream.flush();os.fsync(stream.fileno())
        if count!=expected_bytes or digest.hexdigest()!=expected_sha256:raise ValueError('artifact size/SHA mismatch')
        if sha(name)!=expected_sha256:raise ValueError('cached artifact SHA mismatch')
        check();os.link(name,target,follow_symlinks=False)
        directory=os.open(target.parent,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(directory)
        finally:os.close(directory)
        return {'bytes':count,'sha256':digest.hexdigest()}
    finally:Path(name).unlink(missing_ok=True)


def apply_pinned_patch(checkout,patch,commit,patch_sha256,runner=subprocess.run):
    """Existing isolated checkout only; does not clone, build, or touch vendor trees."""
    checkout=Path(checkout);patch=Path(patch)
    if sha(patch)!=patch_sha256 or not re.fullmatch('[0-9a-f]{40}',commit):raise ValueError('source patch identity mismatch')
    # Reject copied absolute Git metadata before even asking Git about this tree.
    metadata=checkout/'.git'
    if metadata.is_symlink() or not metadata.is_dir():raise ValueError('self-contained isolated git directory required')
    for path in checkout.rglob('*'):
        if path.is_symlink():
            if '.git' in path.relative_to(checkout).parts or not path.resolve().is_relative_to(checkout.resolve()):raise ValueError('escaping source/Git symlink refused')
        elif path.is_file() and path.stat().st_nlink!=1:raise ValueError('shared source hardlink refused')
    import configparser
    cfg=configparser.ConfigParser(interpolation=None);cfg.read(metadata/'config')
    if cfg.has_option('core','worktree') or any(s.lower().startswith('include') for s in cfg.sections()):raise ValueError('external Git worktree/config refused')
    for name in ['objects/info/alternates','commondir']:
        if (metadata/name).exists():raise ValueError('shared Git objects refused')
    def run(*args):return runner(['git','-C',str(checkout),*args],check=True,capture_output=True,text=True,timeout=15)
    if run('rev-parse','HEAD').stdout.strip()!=commit:raise ValueError('source commit mismatch')
    if run('status','--porcelain','--untracked-files=all').stdout:raise ValueError('clean source checkout required')
    run('apply','--check',str(patch.resolve()));run('apply','--whitespace=error-all',str(patch.resolve()))
    return {'commit':commit,'patch_sha256':patch_sha256,'built':False}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--project',type=Path)
    actions=parser.add_subparsers(dest='action',required=True)
    for action in ['plan','stage']:
        command=actions.add_parser(action);command.add_argument('--target-home',type=Path,required=True);command.add_argument('--include-dense',action='store_true',help=argparse.SUPPRESS)
        command.add_argument('--image-model',choices=sorted(IMAGE_SETS),default=DEFAULT_IMAGE,help='default: flux2-klein; z-image is an explicit opt-in')
        if action=='stage':command.add_argument('destination',type=Path)
    actions.add_parser('verify-stage').add_argument('destination',type=Path)
    args=parser.parse_args(argv)
    if getattr(args,'include_dense',False):parser.exit(2,'bootstrap: '+RETIRED_DENSE_FLAG+'\n')
    try:
        if args.action=='verify-stage':result=verify_stage(args.destination)
        elif args.action=='plan':result=plan(project_root(args.project),args.target_home,args.image_model)
        else:result=stage(args.destination,args.target_home,args.project,args.image_model)
    except (OSError,ValueError,KeyError,StopIteration) as error:parser.exit(1,'bootstrap: '+str(error)+'\n')
    print(json.dumps(result,indent=2));return 0

if __name__=='__main__':raise SystemExit(main())
