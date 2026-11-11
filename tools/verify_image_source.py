#!/usr/bin/env python3
"""Read-only source audit and isolated temporary patch replay; no build.
Pass original clean Git checkout, repack snapshot, and exact patch paths.
Prints JSON; caller chooses where to save it. Leaves all supplied trees intact.
"""
import argparse,hashlib,json,pathlib,subprocess,tempfile

def git(repo,*args):return subprocess.check_output(['git','-C',str(repo),*args])
def digest(data):return hashlib.sha256(data).hexdigest()
def audit(repo,dest):
    rows=git(repo,'ls-tree','-r','-z','HEAD').split(b'\0');different=[];links=[];checked=0
    for row in rows:
        if not row:continue
        fields,name=row.split(b'\t',1);mode,kind,oid=fields.decode().split();name=name.decode()
        if kind=='commit':links.append({'path':name,'commit':oid});continue
        path=dest/name
        if not path.exists() and not path.is_symlink():different.append({'path':name,'status':'missing','expected_git_blob':oid});continue
        data=str(path.readlink()).encode() if path.is_symlink() else path.read_bytes()
        actual=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest();checked+=1
        if actual!=oid:
            original=git(repo,'show','HEAD:'+name)
            normalized=data.replace(b'\r\n',b'\n')==original.replace(b'\r\n',b'\n')
            different.append({'path':name,'status':'line_endings_only' if normalized else 'modified','expected_git_blob':oid,'actual_git_blob':actual,'sha256':digest(data)})
    return {'commit':git(repo,'rev-parse','HEAD').decode().strip(),'checked_tracked_blobs':checked,'differences':different,'gitlinks':links}
def main():
    parser=argparse.ArgumentParser();parser.add_argument('base',type=pathlib.Path);parser.add_argument('snapshot',type=pathlib.Path);parser.add_argument('patch',type=pathlib.Path);a=parser.parse_args()
    base=a.base.resolve();snapshot=a.snapshot.resolve();patch=a.patch.resolve()
    paths=[line[6:] for line in patch.read_text().splitlines() if line.startswith('+++ b/')]
    if len(paths)!=len(set(paths)) or any(pathlib.PurePosixPath(p).is_absolute() or '..' in pathlib.PurePosixPath(p).parts for p in paths):raise SystemExit('unexpected patch paths')
    files=[]
    with tempfile.TemporaryDirectory(prefix='sd-source-replay-') as temp:
        temp=pathlib.Path(temp)
        for p in paths:
            before=git(base,'show','HEAD:'+p);destination=temp/p;destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(before)
            files.append({'path':p,'preimage_git_blob':git(base,'rev-parse','HEAD:'+p).decode().strip(),'preimage_sha256':digest(before),'snapshot_sha256':digest((snapshot/p).read_bytes())})
        subprocess.run(['git','-C',str(temp),'apply','--check',str(patch)],check=True)
        subprocess.run(['git','-C',str(temp),'apply',str(patch)],check=True)
        for row in files:
            data=(temp/row['path']).read_bytes();row['replayed_sha256']=digest(data);row['postimage_matches_snapshot']=data==(snapshot/row['path']).read_bytes()
    result={'stable_diffusion':audit(base,snapshot),'ggml':audit(base/'ggml',snapshot/'ggml'),'patch_sha256':digest(patch.read_bytes()),'patch_files':files,'patch_replay_matches_all':all(r['postimage_matches_snapshot'] for r in files)}
    result['cpu_media_submodules']={name:audit(base/name,snapshot/name) for name in ['thirdparty/libwebp','thirdparty/libwebm']}
    allowed=set(paths)
    root_ok=all(row['path'] in allowed or (row['path']=='start-server.bat' and row['status']=='line_endings_only') for row in result['stable_diffusion']['differences'])
    media_ok=all(not item['differences'] or (name=='thirdparty/libwebp' and all(row['path']=='gradlew.bat' and row['status']=='line_endings_only' for row in item['differences'])) for name,item in result['cpu_media_submodules'].items())
    result['passed']=(result['patch_replay_matches_all'] and root_ok and media_ok and not result['ggml']['differences']
        and result['stable_diffusion']['commit']=='5a1ba5d895bec4e07deab80d6d791721cd59d9c2'
        and result['ggml']['commit']=='135ce1e0e3f2b0693d3de93d4f6b5fa0a1947820'
        and result['patch_sha256']=='64952ee28cee57cee8ad44fd4364845e8ebb6d7453ded162ae252770bcaa1ac0')
    print(json.dumps(result,indent=2))
    if not result['passed']:raise SystemExit(1)
if __name__=='__main__':main()
