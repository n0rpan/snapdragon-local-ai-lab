"""Synthetic in-memory v2 contracts only; no real qualification/pass artifacts."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from snapdragon_ai_lab.runtime import classic_beta as beta, profile_policy


class CandidateBeta(unittest.TestCase):
    def profile(self, alias=beta.ALIAS, memlat=True):
        policy = beta.POLICY
        profile = {**copy.deepcopy(policy), 'ready':False,'model_alias':alias,
                   'model_path':'/synthetic/model.gguf','server_binary':'/synthetic/engine',
                   'sha256':beta.MODEL_SHA,
                   'expected_bytes':beta.MODEL_BYTES,
                   'server_sha256':beta.SERVER_SHA, 'prefix_cache':{'enabled':False}}
        if memlat:
            profile['memlat_overrides'] = dict(threads=8,threads_batch=12,spec_type='none',small_batch_limit=1)
            profile = profile_policy.resolve(profile,available=True)
        return profile

    def identity(self, profile):
        return dict(bytes=profile['expected_bytes'],inode=123,mtime_ns=456,ctime_ns=789,
                    canonical_path=profile['model_path'],
                    filesystem=dict(uuid='synthetic-uuid',fstype='btrfs',fsroot='/@home',subvolid='259'))

    def report(self, profile):
        return dict(kind=beta.CANDIDATE_KIND,model_alias=profile['model_alias'],
            model_sha256=profile['sha256'],model_bytes=profile['expected_bytes'],
            engine_sha256=profile['server_sha256'],policy_sha256=beta.policy_digest(profile),
            execution_policy_mode=profile.get('execution_policy_mode','stock'),
            model_identity=self.identity(profile),normal13_cases=13,opencode_exit=0,test_exit=0,
            screen_cases=['merge_intervals','parse_port'],
            functional_cases=['ready','chat-record','chat-context-json','structured-tool-call','tool-result-final'],
            guard_failures=[],completed_tools=3,test_commands_completed=1,
            **{k:True for k in ('screen_passed','normal13_passed','protected_unchanged','native_reaped',
                'all_children_reaped','readers_joined','guard_joined','ram_recovered','lease_free_after','post_source_pins_passed')})

    def qualify(self, profile, report=None):
        raw = json.dumps(self.report(profile) if report is None else report).encode()
        sha = hashlib.sha256(raw).hexdigest()
        with patch.object(beta,'stable_model_identity',return_value=self.identity(profile)):
            return beta.qualify_candidate(profile,raw,report_sha256=sha,reviewed_report_sha256=sha)

    def test_each_alias_and_execution_mode_requires_its_exact_new_policy(self):
        for alias in (beta.ALIAS,):
            for memlat in (False,True):
                with self.subTest(alias=alias,memlat=memlat):
                    profile = self.profile(alias,memlat)
                    result = self.qualify(profile)
                    with patch.object(beta,'stable_model_identity',return_value=self.identity(profile)), \
                         patch.object(beta,'verify_engine',side_effect=AssertionError('poll must not hash engine')), \
                         patch.object(Path,'open',side_effect=AssertionError('poll must not open model')):
                        self.assertTrue(beta.is_beta(result))
                        self.assertTrue(beta.eligible(result))
                        result['threads'] += 1
                        self.assertFalse(beta.eligible(result))
                    stale = self.report(self.profile(alias,False))
                    if memlat:
                        with self.assertRaises(ValueError):self.qualify(profile,stale)

    def test_incomplete_or_modified_quality_lifecycle_and_identity_are_rejected(self):
        profile = self.profile()
        changes = {'normal13_cases':12,'normal13_passed':False,'protected_unchanged':False,
                   'opencode_exit':1,'test_exit':1,'completed_tools':0,'test_commands_completed':0,
                   'guard_failures':['stop'],'ram_recovered':False,'all_children_reaped':False,
                   'screen_cases':['parse_port'],'functional_cases':['ready'],
                   'model_sha256':'e'*64,'model_bytes':1,'engine_sha256':'f'*64}
        for key,value in changes.items():
            report = self.report(profile)
            report[key] = value
            with self.subTest(key=key),self.assertRaises(ValueError):self.qualify(profile,report)
        result = self.qualify(profile)
        for key,value in self.identity(profile).items():
            identity = self.identity(profile)
            identity[key] = {'changed':True} if isinstance(value,dict) else str(value)+'changed'
            with self.subTest(identity=key),patch.object(beta,'stable_model_identity',return_value=identity):
                self.assertFalse(beta.eligible(result))
        changed = copy.deepcopy(result)
        changed['validation']['root_reviewed_report_sha256'] = 'f'*64
        with patch.object(beta,'stable_model_identity',return_value=self.identity(profile)):
            self.assertFalse(beta.eligible(changed))

    def test_exact_report_review_and_original_weights_and_retired_alias_rejection(self):
        profile = self.profile()
        raw = json.dumps(self.report(profile)).encode()
        sha = hashlib.sha256(raw).hexdigest()
        for data,review in ((raw+b' ',sha),(raw,'f'*64)):
            with self.assertRaises(ValueError):
                beta.qualify_candidate(profile,data,report_sha256=sha,reviewed_report_sha256=review)
        for mutate in (lambda p:p.update(sha256='f'*64),lambda p:p.update(prefix_cache={'enabled':True}),
                       lambda p:p.update(server_sha256='f'*64)):
            changed = copy.deepcopy(profile)
            mutate(changed)
            with self.assertRaises(ValueError):self.qualify(changed)
        retired = self.profile()
        retired['model_alias'] = 'arete-27b-local'
        with self.assertRaises(ValueError):self.qualify(retired)
        self.assertFalse(beta.is_beta(retired))
        for operation in (lambda: beta.opencode_fragment('arete-27b-local'),
                          lambda: beta.request_defaults({},'/v1/chat/completions','arete-27b-local'),
                          lambda: beta.add_profile({},retired)):
            with self.assertRaises(ValueError):operation()
        qualified = self.qualify(profile)
        qualified['model_alias'] = 'arete-27b-local'
        with patch.object(beta,'stable_model_identity',return_value=self.identity(profile)):
            self.assertFalse(beta.eligible(qualified))

    def test_prefix_binds_capacity_and_records_optional_hit_isolation_evidence(self):
        for alias in (beta.ALIAS,):
            for memlat in (False, True):
                profile = self.profile(alias, memlat)
                profile['prefix_cache'] = dict(enabled=True,directory='/synthetic/prefixes',max_bytes=1024**3)
                report = self.report(profile)
                core_result = self.qualify(profile, report)
                with patch.object(beta,'stable_model_identity',return_value=self.identity(profile)):
                    self.assertTrue(beta.eligible(core_result))
                report.update(prefix_store_hit_verified=True,prefix_cache_isolation_verified=True)
                result = self.qualify(profile, report)
                self.assertEqual(result['prefix_cache'],profile['prefix_cache'])
                with patch.object(beta,'stable_model_identity',return_value=self.identity(profile)):
                    self.assertTrue(beta.eligible(result))
                    moved = copy.deepcopy(result)
                    moved['prefix_cache']['directory'] = '/synthetic/relocated-prefixes'
                    self.assertTrue(beta.eligible(moved))
                    for key,value in (('enabled',False),('max_bytes',2*1024**3)):
                        changed = copy.deepcopy(result)
                        changed['prefix_cache'][key] = value
                        self.assertFalse(beta.eligible(changed))

    def test_persistent_identity_ignores_boot_device_number_but_keeps_file_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'synthetic'
            path.write_bytes(b'only tiny test bytes')
            info = path.lstat()
            filesystem = dict(uuid='fixture',fstype='btrfs',fsroot='/@home',subvolid='259')
            with patch.object(beta,'filesystem_identity',return_value=filesystem):
                before = beta.stable_model_identity(path)
                changed = SimpleNamespace(st_dev=info.st_dev+99,st_mode=info.st_mode,st_size=info.st_size,
                    st_ino=info.st_ino,st_mtime_ns=info.st_mtime_ns,st_ctime_ns=info.st_ctime_ns)
                with patch.object(Path,'lstat',return_value=changed):
                    self.assertEqual(beta.stable_model_identity(path),before)
                self.assertNotIn('device',before)

    def test_ornith_fragments_defaults_and_opt_in_preserve_existing_selection(self):
        profile = self.qualify(self.profile(beta.ALIAS))
        runtime = {'selected_model':'qwen','profiles':{'qwen':{'ready':True}}}
        updated = beta.add_profile(runtime,profile)
        self.assertEqual(updated['selected_model'],'qwen')
        self.assertNotIn(beta.ALIAS,runtime['profiles'])
        model = beta.opencode_fragment(beta.ALIAS)['providers']['local-qwen']['models'][beta.ALIAS]
        self.assertTrue(model['capabilities']['tools'])
        self.assertFalse(model['capabilities']['reasoning']['supported'])
        value = beta.request_defaults({'messages':[]},'/v1/chat/completions',beta.ALIAS)
        self.assertEqual(value['chat_template_kwargs'],{'enable_thinking':False})
        self.assertEqual(value['temperature'],0)
        self.assertEqual(value['max_tokens'],2048)


if __name__ == '__main__':unittest.main()
