"""Create-only source-payload transaction for an explicit sandbox target HOME.

This step never installs into the current user's HOME, starts services, changes a
live panel layout, downloads models or enables qualification/readiness flags.
"""
import argparse
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import secrets
import re
import stat
from .bootstrap import relative,sha,verify_stage
from .bootstrap_transfer import atomic_json

LEASE='.local/state/local-ai/inference.lock'
MAX_SOURCE_BYTES=8*1024**2

class InstallError(Exception):pass


def sandbox_path(path):
    path=Path(path).absolute();home=Path.home().absolute()
    if '..' in path.parts or path==home or path==Path('/') or any(path==home/name or path.is_relative_to(home/name) for name in ['.config','.local','Pictures']):raise InstallError('explicit_sandbox_home_or_journal_required')
    if path.is_relative_to(home) and path!=home and path.relative_to(home).parts[0].startswith('.'):raise InstallError('live_hidden_home_tree_refused')
    if path.is_symlink() or any(parent.is_symlink() for parent in path.parents):raise InstallError('symlinked_target_path_refused')
    return path


def owned_directory(path):
    value=Path(path).lstat()
    if not stat.S_ISDIR(value.st_mode) or value.st_uid!=os.getuid() or value.st_mode&0o022:raise InstallError('unsafe_existing_directory')


def read_json(path,limit=4*1024**2):
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_CLOEXEC)
    with os.fdopen(fd,'rb') as stream:
        info=os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_uid!=os.getuid() or info.st_size>limit:raise InstallError('unsafe_or_oversized_json')
        value=stream.read(limit+1)
    if len(value)>limit:raise InstallError('oversized_json')
    return json.loads(value)


def allowed_file(name):
    exact={LEASE,'.local/bin/local-ai','.local/bin/launch_local_plan.py','.config/opencode/local-ai-bootstrap.fragment.json',
           '.config/systemd/user/local-ai-chat.service','.config/systemd/user/prompt-studio-app.service'}
    prefixes=('.config/local-ai/','.config/omarchy/plugins/snapdragon.local-ai/','.local/share/ai-stack/runtime/','.local/share/ai-stack/notices/')
    return name in exact or name.startswith(prefixes)


def allowed_directories(files):
    leaves=['Work','Pictures/sd/_uploads','.local/state/local-ai/prefix-cache',
            '.local/share/ai-stack/releases/llama-cpu','.local/share/ai-stack/releases/sd-cpu-repack',
            '.local/share/ai-stack/models/llm','.local/share/ai-stack/models/images/vae']
    result=set()
    for name in [*files,*leaves]:
        path=relative(name)
        for parent in [path if name in leaves else path.parent,*path.parents]:
            if parent!=Path('.'):result.add(parent.as_posix())
    return result


def open_dir(root_fd,name=''):
    descriptor=os.dup(root_fd)
    try:
        for part in relative(name).parts if name else []:
            child=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=descriptor)
            os.close(descriptor);descriptor=child
            value=os.fstat(descriptor)
            if value.st_uid!=os.getuid() or value.st_mode&0o022:raise InstallError('unsafe_target_directory')
        return descriptor
    except BaseException:os.close(descriptor);raise


def identity(info):return {'dev':info.st_dev,'ino':info.st_ino,'mode':stat.S_IMODE(info.st_mode),'mtime_ns':info.st_mtime_ns}


def matches(parent,name,row):
    try:
        fd=os.open(name,os.O_RDONLY|os.O_NOFOLLOW|os.O_CLOEXEC,dir_fd=parent)
        with os.fdopen(fd,'rb') as stream:
            info=os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_uid!=os.getuid() or info.st_size!=row['bytes'] or any(identity(info)[key]!=row[key] for key in ['dev','ino','mode','mtime_ns']):return False
            data=stream.read(row['bytes']+1)
            after=os.fstat(stream.fileno())
            return len(data)==row['bytes'] and identity(after)==identity(info) and hashlib.sha256(data).hexdigest()==row['sha256']
    except (OSError,KeyError):return False


def rollback_state(state):
    removed=[];preserved=[];target=Path(state['target_home'])
    if state.get('root_identity') is None:return {'removed':removed,'preserved':preserved}
    try:info=target.lstat()
    except FileNotFoundError:return {'removed':[],'preserved':[],'target_already_absent':True}
    owned_directory(target)
    planned=state.get('planned_files',[])
    if not all(allowed_file(name) for name in planned) or not set(state.get('planned_directories',[]))<=allowed_directories(planned):raise InstallError('unsafe_journal_plan')
    if not stat.S_ISDIR(info.st_mode) or (info.st_dev,info.st_ino)!=(state['root_identity']['dev'],state['root_identity']['ino']):raise InstallError('target_root_identity_changed')
    root=os.open(target,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        for row in reversed(state['files']):
            if row['path'] not in planned or row['path']==LEASE:raise InstallError('unsafe_journal_file')
            if row.get('temporary') and not re.fullmatch(r'\.bootstrap-[0-9a-f]{24}',row['temporary']):raise InstallError('unsafe_journal_temporary')
            path=relative(row['path'])
            try:parent=open_dir(root,path.parent.as_posix() if path.parent!=Path('.') else '')
            except FileNotFoundError:continue
            except (OSError,InstallError):preserved.append(row['path']);continue
            try:
                for name,label in [(path.name,row['path']),(row.get('temporary'),row['path']+' (temporary)')]:
                    if not name:continue
                    try:os.stat(name,dir_fd=parent,follow_symlinks=False)
                    except FileNotFoundError:continue
                    if matches(parent,name,row):os.unlink(name,dir_fd=parent);os.fsync(parent);removed.append(label)
                    else:preserved.append(label)
            finally:os.close(parent)
        for row in reversed(state['directories']):
            if row['path'] not in state.get('planned_directories',[]):raise InstallError('unsafe_journal_directory')
            path=relative(row['path'])
            try:parent=open_dir(root,path.parent.as_posix() if path.parent!=Path('.') else '')
            except FileNotFoundError:continue
            except (OSError,InstallError):preserved.append(row['path']);continue
            try:
                try:info=os.stat(path.name,dir_fd=parent,follow_symlinks=False)
                except FileNotFoundError:continue
                if not stat.S_ISDIR(info.st_mode) or (info.st_dev,info.st_ino)!=(row['dev'],row['ino']) or stat.S_IMODE(info.st_mode)!=row['mode']:preserved.append(row['path']);continue
                try:os.rmdir(path.name,dir_fd=parent);os.fsync(parent);removed.append(row['path']+'/')
                except OSError as error:
                    if error.errno not in (errno.ENOTEMPTY,errno.EEXIST):raise
                    # Canonical anchor or new user files intentionally keep these directories.
            finally:os.close(parent)
    finally:os.close(root)
    if state.get('root_created') and state.get('canonical') is None:
        try:target.rmdir();removed.append('(new target home)')
        except OSError as error:
            if error.errno not in (errno.ENOTEMPTY,errno.EEXIST,errno.ENOENT):raise
    return {'removed':removed,'preserved':preserved,'canonical_lease_preserved':state.get('canonical') is not None}


def check_disabled(stage,manifest):
    lease=manifest['files'].get(LEASE)
    if not lease or lease.get('bytes')!=0 or lease.get('mode')!=0o600 or lease.get('sha256')!=hashlib.sha256(b'').hexdigest():raise InstallError('empty_private_staged_lease_required')
    for name in manifest['files']:
        relative(name)
        if not allowed_file(name):raise InstallError('unsupported_payload_path_'+name)
    if not set(manifest['directories'])<=allowed_directories(manifest['files']):raise InstallError('unsupported_payload_directory')
    config=read_json(stage/'payload/.config/local-ai/llm/runtime.json')
    if config.get('ready')is not False or any(value.get('ready')is not False for value in config.get('profiles',{}).values()):raise InstallError('source_install_cannot_enable_profiles')
    for name in ['cpu-profile.json','gpu-profile.json']:
        path=stage/'payload/.config/local-ai/images'/name
        if path.exists():
            value=read_json(path)
            if value.get('enabled')is not False or value.get('validated')is not False:raise InstallError('source_install_cannot_validate_images')


def install(stage,target_home,journal,*,stage_manifest_sha256,inject=None):
    stage=Path(stage);target=sandbox_path(target_home);journal=sandbox_path(journal)
    if journal.exists() or journal.is_symlink():raise InstallError('fresh_journal_required')
    owned_directory(journal.parent);owned_directory(target.parent)
    if journal.is_relative_to(target):raise InstallError('journal_must_live_outside_target_home')
    if sha(stage/'stage-manifest.json')!=stage_manifest_sha256:raise InstallError('stage_manifest_identity_mismatch')
    manifest=read_json(stage/'stage-manifest.json');plan=read_json(stage/'bootstrap-plan.json')
    if sum(row['bytes'] for row in manifest['files'].values())>MAX_SOURCE_BYTES:raise InstallError('source_payload_only_size_limit')
    verify_stage(stage)
    if Path(plan['target_home']).absolute()!=target:raise InstallError('stage_target_home_mismatch')
    if plan.get('end_to_end_parity_passed')is not False or manifest.get('installed')is not False:raise InstallError('unqualified_fresh_stage_required')
    check_disabled(stage,manifest)
    if target.exists():owned_directory(target)
    # Preflight every collision before the first target mutation. Never merge user config/layout.
    for name in manifest['directories']:
        path=target/relative(name)
        if path.exists() or path.is_symlink():owned_directory(path)
    for name in manifest['files']:
        path=target/relative(name)
        if name==LEASE:
            if path.exists() or path.is_symlink():
                info=path.lstat()
                if not stat.S_ISREG(info.st_mode) or info.st_uid!=os.getuid() or stat.S_IMODE(info.st_mode)!=0o600:raise InstallError('unsafe_existing_canonical_lease')
        elif path.exists() or path.is_symlink():raise InstallError('existing_managed_file_preserved_'+name)
    state={'schema':1,'kind':'sandbox-source-install','status':'installing','target_home':str(target),'stage_manifest_sha256':stage_manifest_sha256,
           'installed':False,'qualified':False,'services_started':False,'panel_layout_changed':False,'files':[],'directories':[],'canonical':None,'root_identity':None,'root_created':False,'planned_files':list(manifest['files']),'planned_directories':list(manifest['directories'])}
    fd=os.open(journal,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'w') as stream:json.dump(state,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    root=None;source=None;lease=None
    def save():atomic_json(journal,state)
    def mutation(label,name):
        save()
        if inject:inject(label,name)
    try:
        if not target.exists():target.mkdir(mode=0o700);state['root_created']=True
        state['root_identity']=identity(target.lstat());save()
        if state['root_created'] and inject:inject('target_home_created','')
        root=os.open(target,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
        source=os.open(stage/'payload',os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
        for name in sorted(manifest['directories'],key=lambda x:(len(relative(x).parts),x)):
            path=relative(name);parent=open_dir(root,path.parent.as_posix() if path.parent!=Path('.') else '')
            try:
                try:os.mkdir(path.name,0o700,dir_fd=parent)
                except FileExistsError:continue
                info=os.stat(path.name,dir_fd=parent,follow_symlinks=False);state['directories'].append({'path':name,**identity(info)})
                os.fsync(parent);mutation('directory_created',name)
            finally:os.close(parent)
        path=relative(LEASE);parent=open_dir(root,path.parent.as_posix())
        try:
            created=True
            try:lease=os.open(path.name,os.O_RDWR|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=parent)
            except FileExistsError:created=False;lease=os.open(path.name,os.O_RDWR|os.O_NOFOLLOW,dir_fd=parent)
            info=os.fstat(lease)
            if not stat.S_ISREG(info.st_mode) or info.st_uid!=os.getuid() or stat.S_IMODE(info.st_mode)!=0o600:raise InstallError('unsafe_canonical_lease')
            state['canonical']={'created':created,**identity(info)}
            fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
            os.fsync(lease);os.fsync(parent);mutation('canonical_preserved',LEASE)
        finally:os.close(parent)
        for name,row in manifest['files'].items():
            if name==LEASE:continue
            path=relative(name);parent=open_dir(root,path.parent.as_posix() if path.parent!=Path('.') else '')
            source_parent=open_dir(source,path.parent.as_posix() if path.parent!=Path('.') else '')
            temporary='.bootstrap-'+secrets.token_hex(12);entry=None;temporary_created=False
            try:
                incoming=os.open(path.name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=source_parent)
                with os.fdopen(incoming,'rb') as stream:data=stream.read(MAX_SOURCE_BYTES+1)
                if len(data)!=row['bytes'] or hashlib.sha256(data).hexdigest()!=row['sha256']:raise InstallError('source_payload_changed')
                out=os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,row['mode'],dir_fd=parent)
                temporary_created=True
                with os.fdopen(out,'wb') as stream:
                    stream.write(data);stream.flush();os.fchmod(stream.fileno(),row['mode']);os.fsync(stream.fileno());info=os.fstat(stream.fileno())
                entry={'path':name,'temporary':temporary,'bytes':row['bytes'],'sha256':row['sha256'],**identity(info)}
                state['files'].append(entry);mutation('temporary_written',name)
                os.link(temporary,path.name,src_dir_fd=parent,dst_dir_fd=parent,follow_symlinks=False);os.fsync(parent);mutation('file_published',name)
                os.unlink(temporary,dir_fd=parent);os.fsync(parent);entry['temporary']=None;mutation('temporary_removed',name)
            finally:
                if entry is None and temporary_created:
                    try:os.unlink(temporary,dir_fd=parent)
                    except FileNotFoundError:pass
                os.close(source_parent);os.close(parent)
        state.update(status='installed_unqualified',installed=True);save()
        return {'installed_source_payload':True,'qualified':False,'services_started':False,'panel_layout_changed':False,'journal_sha256':sha(journal),'canonical_inode':state['canonical']['ino']}
    except BaseException as error:
        state['error']=type(error).__name__;state['rollback']=rollback_state(state);state.update(status='rolled_back_after_failure',installed=False);save();raise
    finally:
        for descriptor in [source,root,lease]:
            if descriptor is not None:os.close(descriptor)


def rollback(journal,expected_sha256):
    journal=sandbox_path(journal)
    if sha(journal)!=expected_sha256:raise InstallError('journal_identity_mismatch')
    state=read_json(journal)
    if state.get('kind')!='sandbox-source-install' or state.get('schema')!=1:raise InstallError('unsupported_journal')
    sandbox_path(state['target_home'])
    lease=None;root=None;parent=None
    try:
        target=Path(state['target_home'])
        if target.exists() and state.get('canonical') is not None:
            root=os.open(target,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW);path=relative(LEASE);parent=open_dir(root,path.parent.as_posix())
            lease=os.open(path.name,os.O_RDWR|os.O_NOFOLLOW,dir_fd=parent);info=os.fstat(lease)
            if (info.st_dev,info.st_ino)!=(state['canonical']['dev'],state['canonical']['ino']):raise InstallError('canonical_inode_changed')
            fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        result=rollback_state(state);state.update(status='rolled_back',installed=False,rollback=result);atomic_json(journal,state)
    finally:
        for fd in [lease,parent,root]:
            if fd is not None:os.close(fd)
    return {**result,'qualified':False,'services_started':False,'journal_sha256':sha(journal)}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);actions=parser.add_subparsers(dest='action',required=True)
    apply=actions.add_parser('apply');apply.add_argument('--stage',type=Path,required=True);apply.add_argument('--target-home',type=Path,required=True);apply.add_argument('--journal',type=Path,required=True);apply.add_argument('--stage-manifest-sha256',required=True)
    undo=actions.add_parser('rollback');undo.add_argument('--journal',type=Path,required=True);undo.add_argument('--journal-sha256',required=True)
    args=parser.parse_args(argv)
    try:result=install(args.stage,args.target_home,args.journal,stage_manifest_sha256=args.stage_manifest_sha256) if args.action=='apply' else rollback(args.journal,args.journal_sha256)
    except (OSError,ValueError,KeyError,InstallError) as error:parser.exit(1,'bootstrap-install: '+str(error)+'\n')
    print(json.dumps(result,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
