"""Stub-only build and tiny archive tests; no compiler, network, or model."""
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from snapdragon_ai_lab import bootstrap_build as b
from snapdragon_ai_lab.bootstrap_guard import Stopped

class Okay:
    def require(self):pass

class Build(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
    def archive(self,name,member,body=b'tiny'):
        path=self.root/name
        with tarfile.open(path,'w:gz') as stream:
            entry=tarfile.TarInfo(member);entry.size=len(body);stream.addfile(entry,io.BytesIO(body))
        return path
    def test_safe_archive_and_escape_rejection_before_write(self):
        good=self.archive('good.tgz','source/CMakeLists.txt');b.extract_source(good,self.root/'good')
        self.assertEqual((self.root/'good/source/CMakeLists.txt').read_bytes(),b'tiny')
        bad=self.archive('bad.tgz','../escape')
        with self.assertRaisesRegex(Stopped,'unsafe_source_archive'):b.extract_source(bad,self.root/'bad')
        self.assertFalse((self.root/'escape').exists());self.assertFalse((self.root/'bad').exists())
    def test_archive_size_bound(self):
        archive=self.archive('big.tgz','file',b'12345')
        with self.assertRaisesRegex(Stopped,'size_limit'):b.extract_source(archive,self.root/'out',max_bytes=4)
    def fixture(self):
        source=self.root/'fetched';(source/'.git').mkdir(parents=True);(source/'.git/config').write_text('[core]\nbare=false\n')
        (source/'fixture.cpp').write_bytes(b'public fixture')
        recipe={'commit':'a'*40,'options':{'CMAKE_BUILD_TYPE':'Release','GGML_CPU_REPACK':'OFF'},'required_postimages':{'fixture.cpp':hashlib.sha256(b'public fixture').hexdigest()},'targets':['fixture-server'],'binaries':['bin/fixture-server']}
        return source,recipe
    def test_stub_build_is_serial_network_denied_and_never_qualified(self):
        source,recipe=self.fixture();work=self.root/'job';work.mkdir();commands=[]
        class Stub:
            def call(_,command):
                commands.append(command)
                if command[0]=='git':
                    if 'rev-parse' in command:return recipe['commit']+'\n'
                    if '--cached' in command:return 'fixture.cpp\n'
                    return ''
                self.assertEqual(command[:5],['unshare','--user','--map-root-user','--net','--'])
                inner=command[5:]
                if '-S' in inner:
                    build=Path(inner[inner.index('-B')+1]);build.mkdir()
                    opts=[x[2:] for x in inner if x.startswith('-D')]
                    (build/'CMakeCache.txt').write_text(''.join(key+':STRING='+value+'\n' for key,value in (x.split('=',1) for x in opts)))
                else:
                    self.assertEqual(inner[inner.index('--parallel')+1],'1')
                    output=Path(inner[inner.index('--build')+1])/'bin/fixture-server';output.parent.mkdir();output.write_bytes(b'stub artifact')
                return ''
        result=b.build_engine(recipe,source,self.root,work,Okay(),Stub(),{})
        self.assertTrue(result['built']);self.assertFalse(result['qualified']);self.assertFalse(result['profiles_enabled']);self.assertFalse(result['models_run'])
        self.assertEqual(result['build_options']['FETCHCONTENT_FULLY_DISCONNECTED'],'ON')
        self.assertEqual((source/'fixture.cpp').read_bytes(),b'public fixture')
        self.assertNotEqual((source/'fixture.cpp').stat().st_ino,(work/'source/fixture.cpp').stat().st_ino)
    def test_source_postimage_drift_prevents_any_cmake_call(self):
        source,recipe=self.fixture();(source/'fixture.cpp').write_bytes(b'changed');work=self.root/'job';work.mkdir()
        class Stub:
            def call(_,command):
                self.assertEqual(command[0],'git')
                if 'rev-parse' in command:return recipe['commit']+'\n'
                if '--cached' in command:return 'fixture.cpp\n'
                return ''
        with self.assertRaisesRegex(Stopped,'postimage_drift'):b.build_engine(recipe,source,self.root,work,Okay(),Stub(),{})

    def test_compiler_selector_normalizes_only_same_executable(self):
        compiler=self.root/'compiler';compiler.write_text('synthetic executable');compiler.chmod(0o700)
        selector=self.root/'cc';selector.symlink_to(compiler)
        result=b.validate_options({'CMAKE_C_COMPILER':'cc','GGML_CPU_REPACK':'OFF'},{'CMAKE_C_COMPILER':str(compiler),'GGML_CPU_REPACK':'OFF'},str(self.root))
        self.assertEqual(result['CMAKE_C_COMPILER'],str(compiler.resolve()))
        with self.assertRaisesRegex(Stopped,'option_drift_GGML_CPU_REPACK'):b.validate_options({'GGML_CPU_REPACK':'OFF'},{'GGML_CPU_REPACK':'ON'},str(self.root))
    def test_different_compiler_is_rejected(self):
        for name in ['cc','other']:
            p=self.root/name;p.write_text('different '+name);p.chmod(0o700)
        with self.assertRaisesRegex(Stopped,'compiler_drift_CMAKE_C_COMPILER'):b.validate_options({'CMAKE_C_COMPILER':'cc'},{'CMAKE_C_COMPILER':str(self.root/'other')},str(self.root))

if __name__=='__main__':unittest.main()
