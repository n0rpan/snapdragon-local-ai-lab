"""Parameterized pinned resume protocol derived from the reviewed Arete v4 recipe."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import ssl
import stat
import tempfile
import time
import urllib.parse
import urllib.request
from .bootstrap import relative
from .bootstrap_guard import Stopped,secure_directory
CHUNK=1024*1024

def atomic_json(path,value):
    fd,name=tempfile.mkstemp(prefix='.'+path.name+'.',dir=path.parent)
    try:
        with os.fdopen(fd,'w') as stream:
            json.dump(value,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
        os.replace(name,path);directory=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(directory)
        finally:os.close(directory)
    finally:Path(name).unlink(missing_ok=True)

def owned_regular(fd):
    value=os.fstat(fd)
    if not stat.S_ISREG(value.st_mode) or value.st_uid!=os.getuid() or value.st_mode&0o777!=0o600:raise Stopped('unsafe_owned_artifact')
    return value

def prefix_hash(stream,length,check):
    stream.seek(0);digest=hashlib.sha256();remaining=length
    while remaining:
        check();block=stream.read(min(CHUNK,remaining))
        if not block:raise Stopped('partial_shorter_than_checkpoint')
        digest.update(block);remaining-=len(block)
    return digest

def validate_response(status,headers,offset,total):
    if headers.get('Content-Encoding','identity').lower()!='identity':raise Stopped('encoded_response_rejected')
    if offset and status!=206:raise Stopped('resume_range_not_honored_no_append')
    if status==206:
        match=re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',headers.get('Content-Range',''))
        if not match:raise Stopped('invalid_content_range')
        start,end,claimed=map(int,match.groups())
        if start!=offset or claimed!=total or not start<=end<total:raise Stopped('content_range_identity_changed')
        expected=end-start+1
    elif status==200 and offset==0:expected=total
    else:raise Stopped('unexpected_http_status')
    size=headers.get('Content-Length')
    if size is not None and (not size.isdigit() or int(size)!=expected):raise Stopped('response_size_mismatch')
    return expected

class HttpsRedirects(urllib.request.HTTPRedirectHandler):
    max_redirections=5;max_repeats=2
    def redirect_request(self,request,fp,code,message,headers,url):
        value=urllib.parse.urlsplit(url)
        if value.scheme!='https' or not value.hostname or value.username or value.password:raise Stopped('https_redirect_required')
        return super().redirect_request(request,fp,code,message,headers,url)

def download(asset,root,guard,opener=None):
    root=Path(root);target=root/relative(asset['stack_relative_path']);url=asset['url'];size=asset['bytes'];wanted=asset['sha256']
    parsed=urllib.parse.urlsplit(url)
    if parsed.scheme!='https' or not parsed.hostname or parsed.username or parsed.password or type(size)is not int or size<=0 or not re.fullmatch('[0-9a-f]{64}',wanted):raise Stopped('immutable_artifact_required')
    secure_directory(root);current=root
    for component in target.parent.relative_to(root).parts:
        current=current/component
        if not current.exists():current.mkdir(mode=0o700)
        secure_directory(current)
    partial=target.with_name('.'+target.name+'.partial');metadata=target.with_name('.'+target.name+'.partial.json')
    if target.exists() or target.is_symlink():
        fd=os.open(target,os.O_RDONLY|os.O_NOFOLLOW|os.O_CLOEXEC)
        with os.fdopen(fd,'rb') as stream:
            value=owned_regular(stream.fileno())
            if value.st_size!=size or prefix_hash(stream,size,guard.require).hexdigest()!=wanted:raise Stopped('existing_final_identity_mismatch_preserved')
        return {'id':asset['id'],'status':'already_verified_complete','bytes':size,'sha256':wanted}
    if partial.exists() or partial.is_symlink():
        fd=os.open(metadata,os.O_RDONLY|os.O_NOFOLLOW|os.O_CLOEXEC)
        with os.fdopen(fd,'r') as stream:
            owned_regular(stream.fileno());raw=stream.read(16385)
        if len(raw)>16384:raise Stopped('metadata_too_large')
        meta=json.loads(raw)
        if (meta.get('url'),meta.get('expected_bytes'),meta.get('expected_sha256'))!=(url,size,wanted):raise Stopped('partial_artifact_identity_changed')
        fd=os.open(partial,os.O_RDWR|os.O_NOFOLLOW|os.O_CLOEXEC);value=owned_regular(fd);offset=meta.get('completed_bytes')
        if type(offset)is not int or not 0<=offset<=value.st_size<=size:os.close(fd);raise Stopped('invalid_checkpoint_offset')
    else:
        if metadata.exists() or metadata.is_symlink():raise Stopped('orphan_metadata_preserved')
        fd=os.open(partial,os.O_RDWR|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW|os.O_CLOEXEC,0o600);offset=0
        meta={'url':url,'expected_bytes':size,'expected_sha256':wanted,'completed_bytes':0,'prefix_sha256':hashlib.sha256(b'').hexdigest(),'status':'partial','qualified':False}
        atomic_json(metadata,meta)
    if shutil.disk_usage(root).free<size-offset+16*CHUNK:os.close(fd);raise Stopped('insufficient_free_disk')
    with os.fdopen(fd,'r+b') as stream:
        digest=prefix_hash(stream,offset,guard.require)
        if digest.hexdigest()!=meta['prefix_sha256']:raise Stopped('checkpoint_prefix_sha256_mismatch')
        stream.truncate(offset);stream.seek(offset);last=offset;when=time.monotonic()
        def checkpoint(status):
            stream.flush();os.fsync(stream.fileno())
            meta.update(status=status,completed_bytes=offset,prefix_sha256=digest.hexdigest(),resources=guard.report())
            atomic_json(metadata,meta)
        try:
            if offset<size:
                headers={'User-Agent':'snapdragon-local-ai-bootstrap/1','Accept-Encoding':'identity'}
                if offset:headers['Range']=f'bytes={offset}-'
                etag=meta.get('etag')
                if offset and isinstance(etag,str) and etag.startswith('"') and etag.endswith('"') and len(etag)<=256:headers['If-Range']=etag
                if opener is None:opener=urllib.request.build_opener(urllib.request.ProxyHandler({}),HttpsRedirects(),urllib.request.HTTPSHandler(context=ssl.create_default_context()))
                response=guard.network_call(lambda:opener.open(urllib.request.Request(url,headers=headers),timeout=5))
                with response:
                    validate_response(response.status,response.headers,offset,size)
                    remote=response.headers.get('ETag')
                    if remote and len(remote)<=256:meta['etag']=remote
                    while True:
                        guard.require();block=guard.network_call(lambda:response.read1(CHUNK));guard.require()
                        if not block:break
                        if offset+len(block)>size:raise Stopped('response_exceeds_expected_bytes')
                        stream.write(block);digest.update(block);offset+=len(block);now=time.monotonic()
                        if offset-last>=32*CHUNK or now-when>=5:checkpoint('partial');last=offset;when=now
            checkpoint('downloaded_unverified')
            if offset!=size:raise Stopped('download_incomplete_resume_available')
            if digest.hexdigest()!=wanted or prefix_hash(stream,size,guard.require).hexdigest()!=wanted:raise Stopped('final_cached_size_or_sha256_mismatch')
            guard.require();value=os.fstat(stream.fileno());current=partial.lstat()
            if (value.st_dev,value.st_ino)!=(current.st_dev,current.st_ino) or current.st_size!=size:raise Stopped('verified_partial_inode_changed')
            os.link(partial,target,follow_symlinks=False);directory=os.open(target.parent,os.O_RDONLY|os.O_DIRECTORY)
            try:os.fsync(directory)
            finally:os.close(directory)
            partial.unlink();meta.pop('stop_reason',None);meta.update(status='verified_complete',sha256=wanted,resources=guard.report());atomic_json(metadata,meta)
            return {'id':asset['id'],'status':'verified_complete','bytes':size,'sha256':wanted}
        except BaseException as error:
            if partial.exists():checkpoint('stopped_partial')
            meta.update(stop_reason=str(error) if isinstance(error,Stopped) else type(error).__name__,resources=guard.report());atomic_json(metadata,meta)
            raise
