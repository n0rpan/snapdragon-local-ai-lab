"""Tiny owned Work fixtures only. No real HOME/config/services/layout/model changes."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest
from unittest.mock import patch
from snapdragon_ai_lab import bootstrap as stage_api
from snapdragon_ai_lab import bootstrap_install as tx

PROJECT=Path(__file__).resolve().parents[1]

class Install(unittest.TestCase):
    def setUp(self):
        # Test roots are siblings of the checkout under Work, never the user's real HOME.
        self.temp=tempfile.TemporaryDirectory(prefix='.bootstrap-install-fixture-',dir=PROJECT.parent)
        self.addCleanup(self.temp.cleanup);self.base=Path(self.temp.name)
    def stage(self,name='case',real=False):
        folder=self.base/name;folder.mkdir();home=folder/'target-home';stage=folder/'stage';journal=folder/'journal.json'
        if real:stage_api.stage(stage,home,PROJECT)
        else:
            payload=stage/'payload';payload.mkdir(parents=True)
            files={'.config/local-ai/llm/runtime.json':(json.dumps({'ready':False,'profiles':{'moe':{'ready':False}}}).encode(),0o600),
                   '.local/bin/local-ai':(b'synthetic control source\n',0o700),tx.LEASE:(b'',0o600)}
            inventory={}
            for name,(body,mode) in files.items():
                path=payload/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(body);path.chmod(mode)
                inventory[name]={'bytes':len(body),'mode':mode,'sha256':hashlib.sha256(body).hexdigest()}
            directories=[]
            for path in payload.rglob('*'):
                if path.is_dir():path.chmod(0o700);directories.append(path.relative_to(payload).as_posix())
            plan=stage/'bootstrap-plan.json';plan.write_text(json.dumps({'target_home':str(home),'end_to_end_parity_passed':False}));plan.chmod(0o600)
            manifest=stage/'stage-manifest.json';manifest.write_text(json.dumps({'files':inventory,'directories':directories,'plan_sha256':tx.sha(plan),'installed':False,'end_to_end_parity_passed':False}));manifest.chmod(0o600)
        return stage,home,journal,tx.sha(stage/'stage-manifest.json')
    def test_real_source_payload_install_and_hash_bound_rollback(self):
        stage,home,journal,digest=self.stage(real=True)
        result=tx.install(stage,home,journal,stage_manifest_sha256=digest)
        self.assertTrue(result['installed_source_payload']);self.assertFalse(result['qualified']);self.assertFalse(result['services_started']);self.assertFalse(result['panel_layout_changed'])
        self.assertTrue((home/'.config/omarchy/plugins/snapdragon.local-ai/Panel.qml').is_file())
        self.assertEqual(json.loads((home/'.config/local-ai/images/presets.json').read_text())['draft']['model'],'q8')
        self.assertFalse(json.loads((home/'.config/local-ai/llm/runtime.json').read_text())['ready'])
        runtime=home/'.local/share/ai-stack/runtime'
        self.assertEqual((runtime/'flash_cache_pref.py').read_bytes(),
                         (PROJECT/'src/snapdragon_ai_lab/runtime/flash_cache_pref.py').read_bytes())
        # Isolated import from the installed payload, without checkout/PYTHONPATH assistance.
        subprocess.run([sys.executable,'-I','-B','-c',
                        'import sys;sys.path.insert(0,sys.argv[1]);import controller,profile_policy,oxe_runtime,flash_cache_pref',
                        str(runtime)],check=True,capture_output=True,text=True)
        inode=(home/tx.LEASE).stat().st_ino
        undone=tx.rollback(journal,result['journal_sha256'])
        self.assertEqual((home/tx.LEASE).stat().st_ino,inode);self.assertTrue(undone['canonical_lease_preserved'])
        self.assertFalse((home/'.local/bin/local-ai').exists());self.assertFalse((home/'.config/omarchy/plugins/snapdragon.local-ai/Panel.qml').exists())
    def test_existing_lease_providers_layout_sentinels_unchanged(self):
        stage,home,journal,digest=self.stage();home.mkdir()
        sentinels={'.config/opencode/config.json':b'private provider sentinel','.config/omarchy/shell.json':b'layout sentinel','unrelated.txt':b'keep'}
        for name,body in sentinels.items():
            path=home/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(body)
        lease=home/tx.LEASE;lease.parent.mkdir(parents=True);lease.write_bytes(b'existing lease bytes');lease.chmod(0o600)
        alias=home/'lease-alias';os.link(lease,alias);inode=lease.stat().st_ino
        result=tx.install(stage,home,journal,stage_manifest_sha256=digest);tx.rollback(journal,result['journal_sha256'])
        for name,body in sentinels.items():self.assertEqual((home/name).read_bytes(),body)
        self.assertEqual(lease.stat().st_ino,inode);self.assertEqual(alias.stat().st_ino,inode);self.assertEqual(lease.read_bytes(),b'existing lease bytes')
    def test_managed_collision_refused_before_any_journal_or_other_writes(self):
        stage,home,journal,digest=self.stage();path=home/'.local/bin/local-ai';path.parent.mkdir(parents=True);path.write_bytes(b'user control')
        with self.assertRaisesRegex(tx.InstallError,'existing_managed_file_preserved'):tx.install(stage,home,journal,stage_manifest_sha256=digest)
        self.assertEqual(path.read_bytes(),b'user control');self.assertFalse(journal.exists());self.assertFalse((home/'.config').exists())
    def test_failure_injection_after_every_target_mutation_is_reversible(self):
        stage,home,journal,digest=self.stage('baseline');events=[]
        with patch.object(os,'fsync',return_value=None):
            result=tx.install(stage,home,journal,stage_manifest_sha256=digest,inject=lambda label,name:events.append((label,name)))
            tx.rollback(journal,result['journal_sha256'])
            for failure in range(1,len(events)+1):
                stage,home,journal,digest=self.stage('fail-'+str(failure));seen=[]
                def inject(label,name):
                    seen.append((label,name))
                    if len(seen)==failure:raise RuntimeError('synthetic mutation failure')
                with self.assertRaisesRegex(RuntimeError,'synthetic mutation failure'):tx.install(stage,home,journal,stage_manifest_sha256=digest,inject=inject)
                state=json.loads(journal.read_text());self.assertEqual(state['status'],'rolled_back_after_failure')
                self.assertFalse((home/'.local/bin/local-ai').exists());self.assertFalse((home/'.config/local-ai/llm/runtime.json').exists())
                if home.exists():self.assertEqual(list(home.rglob('.bootstrap-*')),[])
        self.assertGreater(len(events),10)
    def test_modified_owned_file_and_unknown_user_file_are_preserved(self):
        stage,home,journal,digest=self.stage();result=tx.install(stage,home,journal,stage_manifest_sha256=digest)
        path=home/'.local/bin/local-ai';path.write_bytes(b'user modified source')
        unknown=home/'.config/local-ai/user-notes';unknown.write_bytes(b'user notes')
        undone=tx.rollback(journal,result['journal_sha256'])
        self.assertIn('.local/bin/local-ai',undone['preserved']);self.assertEqual(path.read_bytes(),b'user modified source');self.assertEqual(unknown.read_bytes(),b'user notes')
        self.assertFalse((home/'.config/local-ai/llm/runtime.json').exists())
    def test_rollback_hash_and_busy_canonical_lease_fail_closed(self):
        stage,home,journal,digest=self.stage();result=tx.install(stage,home,journal,stage_manifest_sha256=digest)
        with self.assertRaisesRegex(tx.InstallError,'journal_identity_mismatch'):tx.rollback(journal,'0'*64)
        fd=os.open(home/tx.LEASE,os.O_RDWR);fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            with self.assertRaises(BlockingIOError):tx.rollback(journal,result['journal_sha256'])
            self.assertTrue((home/'.local/bin/local-ai').exists());self.assertEqual(tx.sha(journal),result['journal_sha256'])
        finally:os.close(fd)
        tx.rollback(journal,result['journal_sha256'])
    def test_real_home_and_symlink_target_refused(self):
        with self.assertRaisesRegex(tx.InstallError,'sandbox'):tx.sandbox_path(Path.home())
        outside=self.base/'outside';outside.mkdir();link=self.base/'link';link.symlink_to(outside,target_is_directory=True)
        with self.assertRaisesRegex(tx.InstallError,'symlinked'):tx.sandbox_path(link/'child')

    def test_enabled_profile_is_rejected_even_with_recomputed_stage_hash(self):
        stage,home,journal,digest=self.stage();path=stage/'payload/.config/local-ai/llm/runtime.json'
        value=json.loads(path.read_text());value['ready']=True;path.write_text(json.dumps(value))
        manifest_path=stage/'stage-manifest.json';manifest=json.loads(manifest_path.read_text())
        manifest['files']['.config/local-ai/llm/runtime.json'].update(bytes=path.stat().st_size,sha256=tx.sha(path));manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(tx.InstallError,'cannot_enable_profiles'):tx.install(stage,home,journal,stage_manifest_sha256=tx.sha(manifest_path))
        self.assertFalse(home.exists());self.assertFalse(journal.exists())
    def test_create_only_race_preserves_new_foreign_file_and_rolls_back_ours(self):
        stage,home,journal,digest=self.stage()
        def inject(label,name):
            if label=='temporary_written' and name=='.local/bin/local-ai':(home/name).write_bytes(b'concurrent user source')
        with self.assertRaises(FileExistsError):tx.install(stage,home,journal,stage_manifest_sha256=digest,inject=inject)
        self.assertEqual((home/'.local/bin/local-ai').read_bytes(),b'concurrent user source')
        self.assertFalse((home/'.config/local-ai/llm/runtime.json').exists());self.assertEqual(list(home.rglob('.bootstrap-*')),[])
    def test_changed_parent_symlink_cannot_redirect_rollback(self):
        stage,home,journal,digest=self.stage();result=tx.install(stage,home,journal,stage_manifest_sha256=digest)
        outside=self.base/'outside';outside.mkdir();(outside/'local-ai').write_bytes(b'outside sentinel')
        folder=home/'.local/bin';folder.rename(home/'.local/bin.user');folder.symlink_to(outside,target_is_directory=True)
        undone=tx.rollback(journal,result['journal_sha256'])
        self.assertEqual((outside/'local-ai').read_bytes(),b'outside sentinel');self.assertTrue(folder.is_symlink());self.assertIn('.local/bin/local-ai',undone['preserved'])
        self.assertFalse((home/'.config/local-ai/llm/runtime.json').exists())

    def test_temporary_name_collision_never_deletes_preexisting_file(self):
        stage,home,journal,digest=self.stage();name='.bootstrap-'+'a'*24
        def inject(label,_name):
            if label=='canonical_preserved':(home/'.config/local-ai/llm'/name).write_bytes(b'foreign temporary sentinel')
        with patch.object(tx.secrets,'token_hex',return_value='a'*24):
            with self.assertRaises(FileExistsError):tx.install(stage,home,journal,stage_manifest_sha256=digest,inject=inject)
        self.assertEqual((home/'.config/local-ai/llm'/name).read_bytes(),b'foreign temporary sentinel')
        self.assertFalse((home/'.config/local-ai/llm/runtime.json').exists())

if __name__=='__main__':unittest.main()
