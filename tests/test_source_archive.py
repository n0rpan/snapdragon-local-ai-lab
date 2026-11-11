"""Tiny owned Git objects/archives; no network, install, models or native builds."""
from pathlib import Path
import gzip, io, json, os, subprocess, tarfile, tempfile, unittest
from unittest import mock
from snapdragon_ai_lab import source_archive as a
class SourceArchive(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name);self.repo=self.root/'repo';self.repo.mkdir();self.git('init','-q')
        for name in a.REQUIRED:
            p=self.repo/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((name+'\n').encode())
        p=self.repo/'tools/owned.py';p.parent.mkdir(exist_ok=True);p.write_bytes(b'#!/usr/bin/env python3\n');p.chmod(0o755);self.git('add','.');self.commit_all();self.commit=self.git('rev-parse','HEAD').strip();self.destination=self.root/'source.tar.gz'
    def git(self,*args):return subprocess.check_output(['git','-C',str(self.repo),*args],text=True,env=dict(os.environ,GIT_CONFIG_NOSYSTEM='1',GIT_CONFIG_GLOBAL='/dev/null'))
    def commit_all(self):self.git('-c','user.name=Fixture','-c','user.email=fixture@example.invalid','commit','-qm','owned synthetic source')
    def test_exact_commit_bytes_permissions_complete_manifest(self):
        (self.repo/'README.md').write_text('uncommitted edit');(self.repo/'private').mkdir();(self.repo/'private/secret').write_text('never exported')
        result=a.export(self.repo,self.commit,self.destination);self.assertTrue(a.verify(self.destination,result['archive_sha256'])['passed']);self.assertFalse(result['end_to_end_parity_passed'])
        with tarfile.open(self.destination)as tar:
            names=tar.getnames();self.assertFalse(any('/.git/'in x or '/private/'in x for x in names));prefix=result['root']+'/'
            self.assertEqual(tar.extractfile(prefix+'README.md').read(),b'README.md\n');self.assertEqual(tar.getmember(prefix+'tools/owned.py').mode,0o755)
            manifest=json.load(tar.extractfile(prefix+a.MANIFEST));self.assertEqual(set(manifest['files']),a.REQUIRED|{'tools/owned.py'});self.assertEqual(manifest['tree'],self.git('rev-parse',self.commit+'^{tree}').strip())
    def test_deterministic_archive(self):
        x=a.export(self.repo,self.commit,self.destination);y=a.export(self.repo,self.commit,self.root/'other.tar.gz');self.assertEqual(x['archive_sha256'],y['archive_sha256'])
    def test_output_and_sidecar_clobber_refused(self):
        self.destination.write_bytes(b'keep')
        with self.assertRaises(ValueError):a.export(self.repo,self.commit,self.destination)
        self.assertEqual(self.destination.read_bytes(),b'keep');self.destination.unlink();Path(str(self.destination)+'.json').write_text('keep sidecar')
        with self.assertRaises(ValueError):a.export(self.repo,self.commit,self.destination)
        self.assertFalse(self.destination.exists())
    def test_branch_and_symlink_refused(self):
        with self.assertRaises(ValueError):a.export(self.repo,'HEAD',self.destination)
        alias=self.root/'alias';alias.symlink_to(self.repo,target_is_directory=True)
        with self.assertRaises(ValueError):a.export(alias,self.commit,self.destination)
        (self.repo/'tools/link').symlink_to('/etc/passwd');self.git('add','tools/link');self.commit_all()
        with self.assertRaises(ValueError):a.export(self.repo,self.git('rev-parse','HEAD').strip(),self.destination)
    def test_tracked_payloads_and_unsafe_paths_refused(self):
        for name in ['../escape','src/../escape','.git/config','src/cached.gguf','src/__pycache__/x.pyc','private/secret']:
            with self.assertRaises(ValueError):a.safe_path(name)
        (self.repo/'src/weights.gguf').write_bytes(b'not real weights');self.git('add','src/weights.gguf');self.commit_all()
        with self.assertRaises(ValueError):a.export(self.repo,self.git('rev-parse','HEAD').strip(),self.destination)
    def test_checksum_and_archive_traversal_refused(self):
        a.export(self.repo,self.commit,self.destination)
        with self.assertRaises(ValueError):a.verify(self.destination,'0'*64)
        bad=self.root/'bad.tar.gz'
        with tarfile.open(bad,'w:gz')as tar:
            info=tarfile.TarInfo('root/../escape');info.size=1;tar.addfile(info,io.BytesIO(b'x'))
        with self.assertRaises(ValueError):a.verify(bad)
    def test_git_reads_disable_lazy_fetch_and_prompts(self):
        with mock.patch.object(a.subprocess,'Popen',wraps=subprocess.Popen)as spawn:
            a.inventory(self.repo,self.commit,a.time.monotonic()+20)
        self.assertTrue(spawn.call_args_list)
        for call in spawn.call_args_list:
            self.assertEqual(call.kwargs['env']['GIT_NO_LAZY_FETCH'],'1');self.assertEqual(call.kwargs['env']['GIT_TERMINAL_PROMPT'],'0')
    def test_PAX_and_GNU_bomb_headers_refused_before_payload(self):
        for kind,size in [(tarfile.XHDTYPE,a.MAX_METADATA+1),(tarfile.GNUTYPE_LONGNAME,2**35)]:
            bad=self.root/('bad-'+kind.decode()+'.tar.gz');info=tarfile.TarInfo('metadata');info.type=kind;info.size=size
            raw=info.tobuf(format=tarfile.GNU_FORMAT)+b'\0'*1024;self.assertLess(len(raw),2048);bad.write_bytes(gzip.compress(raw,mtime=0))
            with self.assertRaisesRegex(ValueError,'tar_extended_metadata_cap|unsupported_tar_header_before_parse'):a.verify(bad)
    def test_aggregate_gzip_output_cap(self):
        bad=self.root/'compressed-bomb.tar.gz';bad.write_bytes(gzip.compress(b'\0'*8192,mtime=0))
        with mock.patch.object(a,'MAX_TAR',1024):
            with self.assertRaisesRegex(ValueError,'decompressed_tar_cap'):a.verify(bad)
    def test_legitimate_long_unicode_PAX_path(self):
        name='docs/'+('long-source-name-'*10)+'ångström.md';p=self.repo/name;p.parent.mkdir(exist_ok=True);p.write_bytes(b'owned source');self.git('add',name);self.commit_all()
        result=a.export(self.repo,self.git('rev-parse','HEAD').strip(),self.destination);self.assertTrue(a.verify(self.destination,result['archive_sha256'])['passed'])
if __name__=='__main__':unittest.main()
