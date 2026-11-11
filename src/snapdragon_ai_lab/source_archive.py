"""Export and verify a complete immutable public source tree, without Git metadata."""
from pathlib import Path, PurePosixPath
import argparse, gzip, hashlib, io, json, os, re, selectors, subprocess, tarfile, tempfile, time

TOP = {'.github','benchmarks','bootstrap','dependencies','docs','examples','experiments','patches','src','tests','third_party','tools'}
ROOT_FILES = {'.gitignore','README.md','CONTRIBUTING.md','LICENSE','NOTICE','pyproject.toml'}
REQUIRED = {'README.md','LICENSE','NOTICE','pyproject.toml','dependencies/model-artifacts.json','patches/manifest.json','bootstrap/source-recipes.json','examples/runtime/llm-runtime.json','examples/runtime/opencode-v2.json','src/snapdragon_ai_lab/bootstrap.py','src/snapdragon_ai_lab/runtime/controller.py','src/snapdragon_ai_lab/runtime/prefix_plan.py','src/snapdragon_ai_lab/runtime/prefix_store.py','src/snapdragon_ai_lab/runtime/studio','src/snapdragon_ai_lab/runtime/local-ai','src/snapdragon_ai_lab/omarchy_plugin/Panel.qml'}
MAX_FILE=16*1024**2; MAX_TOTAL=256*1024**2; MAX_FILES=10000
MANIFEST='SOURCE-ARCHIVE.json'
MAX_METADATA=64*1024; MAX_METADATA_TOTAL=16*1024**2; MAX_TAR=MAX_TOTAL+32*1024**2


def safe_path(name):
    p=PurePosixPath(name)
    if len(name.encode('utf-8'))>4096 or not name or p.is_absolute() or str(p)!=name or any(x in ('.','..') for x in p.parts) or '\\' in name or any(ord(c)<32 or ord(c)==127 for c in name):raise ValueError('unsafe_source_path')
    if p.parts[0] not in TOP and name not in ROOT_FILES:raise ValueError('nonpublic_top_level_path:'+name)
    if any(x in {'.git','private','models','__pycache__','.venv','node_modules','raw-results','SDK'}for x in p.parts)or p.name=='.env' or p.suffix.lower()in {'.gguf','.safetensors','.onnx','.bin','.so','.dll','.dylib','.pyc','.pyo','.pem','.key','.log','.png','.jpg','.jpeg','.webp','.part'}:raise ValueError('non_source_payload:'+name)
    return p


def nonsymlink(path):
    path=Path(path).absolute()
    if '..'in path.parts or any(p.is_symlink()for p in [path,*path.parents]):raise ValueError('symlinked_or_unsafe_path')
    return path


def git(repository,args,cap,deadline):
    remaining=min(20,deadline-time.monotonic())
    if remaining<=0:raise ValueError('archive_deadline')
    env={'PATH':os.defpath,'HOME':'/nonexistent','GIT_CONFIG_NOSYSTEM':'1','GIT_CONFIG_GLOBAL':'/dev/null','GIT_NO_REPLACE_OBJECTS':'1','GIT_OPTIONAL_LOCKS':'0','GIT_NO_LAZY_FETCH':'1','GIT_TERMINAL_PROMPT':'0','LC_ALL':'C'}
    child=subprocess.Popen(['git','-C',str(repository),*args],stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=env)
    output={child.stdout:bytearray(),child.stderr:bytearray()};end=time.monotonic()+remaining
    try:
        with selectors.DefaultSelector()as sel:
            for pipe in output:sel.register(pipe,selectors.EVENT_READ)
            while sel.get_map():
                if time.monotonic()>end:raise ValueError('git_read_deadline')
                for key,_ in sel.select(.1):
                    data=os.read(key.fileobj.fileno(),65536)
                    if not data:sel.unregister(key.fileobj);continue
                    output[key.fileobj].extend(data)
                    if len(output[key.fileobj])>(cap if key.fileobj is child.stdout else 65536):raise ValueError('git_output_cap')
        if child.wait(timeout=max(.1,end-time.monotonic()))!=0:raise ValueError('git_read_failed')
        return bytes(output[child.stdout])
    finally:
        if child.poll()is None:child.kill();child.wait(timeout=5)
        child.stdout.close();child.stderr.close()


def inventory(repository,commit,deadline):
    if not re.fullmatch('[0-9a-f]{40}',commit):raise ValueError('full_immutable_commit_required')
    repository=nonsymlink(repository)
    if not (repository/'.git').is_dir()or(repository/'.git').is_symlink():raise ValueError('self_contained_local_repository_required')
    if any((repository/'.git'/x).exists()for x in ['objects/info/alternates','commondir']):raise ValueError('external_git_objects_refused')
    if git(repository,['rev-parse','--verify',commit+'^{commit}'],256,deadline).decode().strip()!=commit:raise ValueError('exact_commit_required')
    tree=git(repository,['rev-parse',commit+'^{tree}'],256,deadline).decode().strip()
    commit_bytes=git(repository,['cat-file','commit',commit],65536,deadline)
    if hashlib.sha1(b'commit '+str(len(commit_bytes)).encode()+b'\0'+commit_bytes).hexdigest()!=commit or commit_bytes.split(b'\n',1)[0]!=b'tree '+tree.encode():raise ValueError('immutable_commit_object_identity_required')
    records=[]
    for raw in git(repository,['ls-tree','-r','-z','--full-tree',commit],4*1024**2,deadline).split(b'\0'):
        if not raw:continue
        header,name=raw.split(b'\t',1);mode,kind,oid=header.decode('ascii').split();name=name.decode('utf-8');safe_path(name)
        if mode not in ('100644','100755')or kind!='blob'or not re.fullmatch('[0-9a-f]{40}',oid):raise ValueError('only_regular_tracked_source_files')
        records.append((name,int(mode[-3:],8),oid))
        if len(records)>MAX_FILES:raise ValueError('source_file_count_cap')
    if not REQUIRED.issubset({x[0]for x in records}):raise ValueError('incomplete_public_source_commit')
    return repository,tree,sorted(records)


def put(tar,name,data,mode):
    info=tarfile.TarInfo(name);info.size=len(data);info.mode=mode;info.mtime=0;info.uid=info.gid=0;info.uname=info.gname='';tar.addfile(info,io.BytesIO(data))


def tree_digest(files):
    root={}
    for path,row in files.items():
        node=root;parts=path.split('/')
        for part in parts[:-1]:
            node=node.setdefault(part,{})
            if not isinstance(node,dict):raise ValueError('tree_file_directory_collision')
        if parts[-1] in node:raise ValueError('tree_file_directory_collision')
        node[parts[-1]]=(row['mode'],row['git_blob'])
    def digest(node):
        entries=[]
        for name,value in node.items():
            if isinstance(value,dict):mode=b'40000';oid=digest(value);key=name.encode()+b'/'
            else:mode=('100%03o'%value[0]).encode();oid=value[1];key=name.encode()
            entries.append((key,mode+b' '+name.encode()+b'\0'+bytes.fromhex(oid)))
        data=b''.join(row for _,row in sorted(entries));return hashlib.sha1(b'tree '+str(len(data)).encode()+b'\0'+data).hexdigest()
    return digest(root)


class BoundedTarReader:
    """Bound gzip output and physical metadata headers before tarfile sees them."""
    def __init__(self,stream):
        self.stream=stream;self.total=0;self.buffer=bytearray();self.payload_blocks=0;self.metadata=0;self.headers=0
    def read(self,size):
        if size<0:raise ValueError('unbounded_tar_read_refused')
        data=self.stream.read(min(size,65536,MAX_TAR-self.total+1));self.total+=len(data)
        if self.total>MAX_TAR:raise ValueError('decompressed_tar_cap')
        self.buffer.extend(data)
        while len(self.buffer)>=512:
            block=bytes(self.buffer[:512]);del self.buffer[:512]
            if self.payload_blocks:self.payload_blocks-=1;continue
            if block==b'\0'*512:continue
            self.headers+=1
            if self.headers>2*MAX_FILES+64:raise ValueError('tar_header_count_cap')
            length=tarfile.nti(block[124:136]);kind=block[156:157]
            if length<0:raise ValueError('negative_tar_physical_size')
            if kind==b'x':
                self.metadata+=length
                if length>MAX_METADATA or self.metadata>MAX_METADATA_TOTAL:raise ValueError('tar_extended_metadata_cap')
            elif kind not in (b'0',b'\0'):raise ValueError('unsupported_tar_header_before_parse')
            elif length>MAX_FILE:raise ValueError('tar_physical_file_cap')
            self.payload_blocks=(length+511)//512
        return data


def verify(archive,expected_sha256=None):
    archive=nonsymlink(archive)
    if not archive.is_file()or archive.stat().st_size>MAX_TOTAL:raise ValueError('bounded_regular_archive_required')
    if expected_sha256 is not None:
        if not re.fullmatch('[0-9a-f]{64}',expected_sha256)or hashlib.sha256(archive.read_bytes()).hexdigest()!=expected_sha256:raise ValueError('archive_checksum_mismatch')
    actual={};manifest=None;root=None;total=0
    with gzip.open(archive,'rb')as decoded, tarfile.open(fileobj=BoundedTarReader(decoded),mode='r|')as tar:
        for member in tar:
            if not member.isfile()or member.size>MAX_FILE or member.size<0 or member.mode not in (0o644,0o755):raise ValueError('unsafe_archive_member')
            parts=PurePosixPath(member.name).parts
            if len(parts)<2 or parts[0]in ('.','..')or '\\'in member.name:raise ValueError('unsafe_archive_root')
            if root is None:root=parts[0]
            if parts[0]!=root:raise ValueError('one_archive_root_required')
            relative='/'.join(parts[1:])
            if relative!=MANIFEST:safe_path(relative)
            if relative in actual or (relative==MANIFEST and manifest is not None):raise ValueError('duplicate_archive_member')
            data=tar.extractfile(member).read(MAX_FILE+1);total+=len(data)
            if len(data)!=member.size or total>MAX_TOTAL or len(actual)>=MAX_FILES:raise ValueError('archive_size_or_count_cap')
            if relative==MANIFEST:manifest=json.loads(data)
            else:actual[relative]={'bytes':len(data),'mode':member.mode,'sha256':hashlib.sha256(data).hexdigest(),'git_blob':hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()}
    if not isinstance(manifest,dict)or not re.fullmatch('[0-9a-f]{40}',manifest.get('commit',''))or root!='snapdragon-local-ai-lab-'+manifest['commit'][:12]or manifest.get('tree')!=tree_digest(actual)or manifest.get('files')!=actual or not REQUIRED.issubset(actual):raise ValueError('complete_exact_manifest_required')
    return {'passed':True,'commit':manifest['commit'],'root':root,'files':len(actual),'source_bytes':sum(x['bytes']for x in actual.values()),'qualified':False,'end_to_end_parity_passed':False}


def export(repository,commit,destination):
    deadline=time.monotonic()+120;destination=nonsymlink(destination);sidecar=Path(str(destination)+'.json')
    if destination.exists()or sidecar.exists()or sidecar.is_symlink()or not destination.parent.is_dir():raise ValueError('fresh_destination_and_sidecar_required')
    repository,tree,records=inventory(repository,commit,deadline);root='snapdragon-local-ai-lab-'+commit[:12]
    fd,temporary=tempfile.mkstemp(prefix='.'+destination.name+'.',dir=destination.parent);temporary=Path(temporary);published=False;report_temp=None
    try:
        manifest={'schema':1,'commit':commit,'tree':tree,'root':root,'files':{},'scope':'all regular tracked public files from exact commit; worktree/untracked/Git metadata excluded','qualified':False,'end_to_end_parity_passed':False};total=0
        with os.fdopen(fd,'wb')as output:
            with gzip.GzipFile(fileobj=output,mode='wb',mtime=0,filename='')as compressed:
                with tarfile.open(fileobj=compressed,mode='w|',format=tarfile.PAX_FORMAT)as tar:
                    for name,mode,oid in records:
                        data=git(repository,['cat-file','blob',oid],MAX_FILE,deadline);total+=len(data)
                        if total>MAX_TOTAL:raise ValueError('source_total_cap')
                        if hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()!=oid:raise ValueError('git_blob_identity_drift')
                        manifest['files'][name]={'bytes':len(data),'mode':mode,'sha256':hashlib.sha256(data).hexdigest(),'git_blob':oid};put(tar,root+'/'+name,data,mode)
                    payload=(json.dumps(manifest,indent=2,sort_keys=True)+'\n').encode()
                    if len(payload)>MAX_FILE:raise ValueError('manifest_cap')
                    put(tar,root+'/'+MANIFEST,payload,0o644)
            output.flush();os.fsync(output.fileno())
        report=verify(temporary);report.update(archive_sha256=hashlib.sha256(temporary.read_bytes()).hexdigest(),archive_bytes=temporary.stat().st_size,tree=tree,manifest_sha256=hashlib.sha256(payload).hexdigest(),installed=False,models_downloaded=False,compiled=False)
        report_fd,report_name=tempfile.mkstemp(prefix='.'+sidecar.name+'.',dir=destination.parent);report_temp=Path(report_name)
        with os.fdopen(report_fd,'w')as f:json.dump(report,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
        os.link(temporary,destination,follow_symlinks=False);published=True
        try:os.link(report_temp,sidecar,follow_symlinks=False)
        except BaseException:
            if destination.stat().st_ino==temporary.stat().st_ino:destination.unlink()
            published=False;raise
        directory=os.open(destination.parent,os.O_RDONLY|os.O_DIRECTORY);os.fsync(directory);os.close(directory)
        return report
    finally:
        temporary.unlink(missing_ok=True)
        if report_temp:report_temp.unlink(missing_ok=True)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True);e=sub.add_parser('export');e.add_argument('--repository',type=Path,required=True);e.add_argument('--commit',required=True);e.add_argument('--destination',type=Path,required=True);v=sub.add_parser('verify');v.add_argument('archive',type=Path);v.add_argument('--sha256',required=True);a=p.parse_args(argv)
    try:result=export(a.repository,a.commit,a.destination)if a.action=='export'else verify(a.archive,a.sha256)
    except (OSError,ValueError,tarfile.TarError,UnicodeError,json.JSONDecodeError)as error:p.exit(1,'source archive: '+str(error)+'\n')
    print(json.dumps(result,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
