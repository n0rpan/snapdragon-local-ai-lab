"""Small fresh-prefix/mock-download/generated-Git tests; no host install/network/model."""
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from snapdragon_ai_lab import bootstrap as b

PROJECT=Path(__file__).resolve().parents[1]

# The home directory of whoever runs the tests must never leak into generated files.
DEVELOPER_HOME=__import__('pwd').getpwuid(__import__('os').getuid()).pw_dir

class FakeResponse(io.BytesIO):
    def geturl(self):return 'https://example.invalid/pinned'

class Bootstrap(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
    def test_default_only_moe_and_three_q8_pipeline_assets(self):
        plan=b.plan(PROJECT,self.root/'fresh-user')
        self.assertEqual({x['id'] for x in plan['assets']},set(b.DEFAULT_IDS))
        self.assertFalse(plan['end_to_end_parity_passed'])
        self.assertTrue(all('/resolve/' in x['url'] and len(x['sha256'])==64 for x in plan['assets']))
        self.assertEqual(len(b.asset_plan(PROJECT,True)),5)
    def test_fresh_stage_does_not_touch_target_home_or_import_studio(self):
        home=self.root/'target-user';destination=self.root/'stage'
        result=b.stage(destination,home,PROJECT)
        self.assertFalse(home.exists());self.assertFalse(result['installed']);self.assertFalse(result['end_to_end_parity_passed'])
        config=json.loads((destination/'payload/.config/local-ai/llm/runtime.json').read_text())
        self.assertEqual(set(config['profiles']),{b.MOE});self.assertEqual(config['selected_model'],b.MOE)
        self.assertFalse(config['ready']);self.assertEqual(config['server_sha256'],'')
        self.assertIn(str(home),config['model_path']);self.assertNotIn(DEVELOPER_HOME,json.dumps(config))
        cpu=json.loads((destination/'payload/.config/local-ai/images/cpu-profile.json').read_text())
        self.assertFalse(cpu['enabled']);self.assertFalse(cpu['validated']);self.assertEqual(cpu['sha256'],'')
        self.assertEqual((destination/'payload/.local/state/local-ai/inference.lock').stat().st_mode&0o777,0o600)
        runtime=destination/'payload/.local/share/ai-stack/runtime'
        # The standalone controller's sibling imports must ship in a fresh bundle.
        for name in ['flash_cache_pref.py','engine_diagnostics.py','client_disconnect.py','engine_ledger.py','explicit_route.py','harness_compat.py','harness_registry.py','openai_api.py','request_ledger.py','model_catalog.py','storage_view.py']:
            self.assertEqual((runtime/name).read_bytes(),(PROJECT/'src/snapdragon_ai_lab/runtime'/name).read_bytes())
    def test_existing_destination_and_symlink_parent_cannot_overwrite(self):
        existing=self.root/'existing';existing.mkdir();sentinel=existing/'user-config';sentinel.write_text('retain')
        with self.assertRaisesRegex(ValueError,'fresh destination'):b.stage(existing,self.root/'user',PROJECT)
        alias=self.root/'alias';alias.symlink_to(existing,target_is_directory=True)
        with self.assertRaisesRegex(ValueError,'symlinked'):b.stage(alias/'new',self.root/'user',PROJECT)
        self.assertEqual(sentinel.read_text(),'retain');self.assertFalse((existing/'new').exists())
    def test_stage_verifier_rejects_mutation_extra_file_and_symlink(self):
        destination=self.root/'stage';b.stage(destination,self.root/'user',PROJECT)
        path=destination/'payload/.local/bin/local-ai';old=path.read_bytes();path.write_bytes(old+b'\n')
        with self.assertRaisesRegex(ValueError,'identity changed'):b.verify_stage(destination)
        path.write_bytes(old)
        extra=destination/'payload/extra';extra.write_text('unknown')
        with self.assertRaisesRegex(ValueError,'unexpected'):b.verify_stage(destination)
        extra.unlink();extra.symlink_to(path)
        with self.assertRaisesRegex(ValueError,'symlink'):b.verify_stage(destination)
    def test_mock_fetch_checks_cached_bytes_and_never_overwrites(self):
        payload=b'tiny synthetic asset';wanted=hashlib.sha256(payload).hexdigest();target=self.root/'asset'
        result=b.verified_fetch('https://example.invalid/pinned',target,len(payload),wanted,opener=lambda _:FakeResponse(payload),check=lambda:None)
        self.assertEqual(result['sha256'],wanted);self.assertEqual(target.read_bytes(),payload)
        with self.assertRaisesRegex(ValueError,'fresh destination'):b.verified_fetch('https://example.invalid/pinned',target,len(payload),wanted,opener=lambda _:FakeResponse(b'bad'),check=lambda:None)
        bad=self.root/'bad'
        with self.assertRaisesRegex(ValueError,'size/SHA'):b.verified_fetch('https://example.invalid/pinned',bad,len(payload),'0'*64,opener=lambda _:FakeResponse(payload),check=lambda:None)
        self.assertFalse(bad.exists());self.assertEqual(sorted(p.name for p in self.root.iterdir()),['asset'])
    def test_mock_fetch_stop_cleans_new_partial_only(self):
        target=self.root/'new'
        def stop():raise RuntimeError('synthetic power stop')
        with self.assertRaisesRegex(RuntimeError,'synthetic power stop'):b.verified_fetch('https://example.invalid/pinned',target,4,hashlib.sha256(b'test').hexdigest(),opener=lambda _:FakeResponse(b'test'),check=stop)
        self.assertFalse(target.exists());self.assertEqual(list(self.root.iterdir()),[])
    def test_real_git_patch_on_generated_tiny_clean_repository(self):
        source=self.root/'source';source.mkdir()
        def git(*args):return subprocess.run(['git','-C',str(source),*args],check=True,capture_output=True,text=True)
        git('init','--quiet');git('config','user.email','fixture@example.invalid');git('config','user.name','Synthetic fixture')
        file=source/'fixture.txt';file.write_text('before\n');git('add','fixture.txt');git('commit','--quiet','-m','synthetic fixture')
        commit=git('rev-parse','HEAD').stdout.strip();file.write_text('after\n');patch=self.root/'change.patch';patch.write_text(git('diff','--','fixture.txt').stdout)
        file.write_text('before\n')
        result=b.apply_pinned_patch(source,patch,commit,b.sha(patch))
        self.assertEqual(file.read_text(),'after\n');self.assertFalse(result['built'])
        with self.assertRaisesRegex(ValueError,'clean source'):b.apply_pinned_patch(source,patch,commit,b.sha(patch))
    def test_git_metadata_escape_is_rejected_before_runner(self):
        source=self.root/'source';(source/'.git').mkdir(parents=True)
        (source/'.git/config').write_text('[core]\nworktree = /synthetic/other\n');patch=self.root/'test.patch';patch.write_text('not applied')
        def runner(*args,**kwargs):self.fail('Git must not run')
        with self.assertRaisesRegex(ValueError,'external Git'):b.apply_pinned_patch(source,patch,'a'*40,b.sha(patch),runner)
    def test_path_traversal_and_root_home_rejected(self):
        for path in ['../escape','/absolute','a/../b']:
            with self.assertRaises(ValueError):b.relative(path)
        for path in ['/', '/home/a/../b']:
            with self.assertRaises(ValueError):b.plan(PROJECT,path)

if __name__=='__main__':unittest.main()
