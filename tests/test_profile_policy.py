"""Synthetic policy and controller lifecycle tests; never start a native engine."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from snapdragon_ai_lab.runtime import controller, profile_policy
from snapdragon_ai_lab.runtime.prefix_plan import make_plan


class ProfilePolicyTests(unittest.TestCase):
    def profile(self):
        return dict(model_alias='synthetic', ready=False, sha256='a'*64,
                    threads=4, extra_args=['-tb', '10', '--threads-batch=9', '--cpu-mask', '0xff',
                        '--spec-type', 'draft-mtp', '--spec-draft-n-max', '2'],
                    environment={'UNRELATED': 'preserve'},
                    memlat_overrides=dict(threads=6, threads_batch=8, cpu_mask='0x111',
                                         spec_type='draft-mtp', spec_draft_n_max=3, small_batch_limit=4))

    def test_stock_exact_and_memlat_detached_without_identity_bypass(self):
        original = self.profile()
        stock = profile_policy.resolve(original, available=False)
        self.assertEqual(stock, original)
        effective = profile_policy.resolve(original, available=True)
        self.assertEqual(effective['threads'], 6)
        self.assertFalse(effective['ready'])
        self.assertEqual(effective['sha256'], original['sha256'])
        self.assertEqual(effective['extra_args'], ['--threads-batch', '8', '--cpu-mask', '0x111',
                         '--spec-type', 'draft-mtp', '--spec-draft-n-max', '3'])
        self.assertEqual(effective['environment'], {'UNRELATED':'preserve','LLAMA_CPU_SMALL_BATCH_LIMIT':'4'})
        self.assertEqual(original, self.profile())
        effective['environment']['UNRELATED'] = 'changed'
        self.assertEqual(stock['environment']['UNRELATED'], 'preserve')

    def test_memlat_prefix_enable_and_validation_select_only_in_active_mode(self):
        profile = self.profile()
        profile['prefix_cache'] = dict(enabled=False,directory='/synthetic/prefixes',max_bytes=123456)
        profile['validation'] = {'kind':'synthetic-stock-evidence'}
        profile['memlat_validation'] = {'kind':'synthetic-memlat-evidence'}
        profile['memlat_overrides']['prefix_cache_enabled'] = True
        stock = profile_policy.resolve(profile,available=False)
        self.assertEqual(stock,profile)
        effective = profile_policy.resolve(profile,available=True)
        self.assertEqual(effective['prefix_cache'],dict(enabled=True,directory='/synthetic/prefixes',max_bytes=123456))
        self.assertEqual(effective['validation'],profile['memlat_validation'])
        self.assertFalse(effective['ready'])
        effective['validation']['kind'] = 'changed'
        self.assertEqual(profile['memlat_validation']['kind'],'synthetic-memlat-evidence')
        for invalid in (1,0,'true',None):
            changed = self.profile()
            changed['memlat_overrides']['prefix_cache_enabled'] = invalid
            with self.subTest(invalid=invalid),self.assertRaises(ValueError):
                profile_policy.resolve(changed,available=True)

    def test_strict_placement_and_mtp_off_normalize_measured_flags(self):
        profile = self.profile()
        profile['extra_args'] += ['--cpu-strict=0','--cpu-strict-batch','0']
        profile['memlat_overrides'] = dict(threads=8,threads_batch=12,cpu_mask='0xff',
            cpu_mask_batch='0xfff',cpu_strict=1,cpu_strict_batch=1,spec_type='none')
        result = profile_policy.resolve(profile,available=True)
        self.assertNotIn('--spec-draft-n-max',result['extra_args'])
        self.assertEqual(result['extra_args'].count('--cpu-strict'),1)
        self.assertEqual(result['extra_args'].count('--cpu-strict-batch'),1)
        self.assertNotIn('--cpu-strict=0',result['extra_args'])
        for key in ('cpu_strict','cpu_strict_batch'):
            for invalid in (True,2,-1,'1'):
                changed = self.profile()
                changed['memlat_overrides'][key] = invalid
                with self.subTest(key=key,value=invalid),self.assertRaises(ValueError):
                    profile_policy.resolve(changed,available=True)
        profile['memlat_overrides']['spec_draft_n_max'] = 2
        with self.assertRaises(ValueError):profile_policy.resolve(profile,available=True)

    def test_detection_requires_both_remote_domains_and_readable_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ('ddr', 'llcc'):
                node = root/name
                node.mkdir()
                (node/'name').write_text(name)
                (node/'governor').write_text('remote')
            (root/'unrelated').mkdir()
            self.assertTrue(profile_policy.memlat_available(root))
            (root/'llcc/governor').write_text('performance')
            self.assertFalse(profile_policy.memlat_available(root))
            (root/'llcc/governor').unlink()
            self.assertFalse(profile_policy.memlat_available(root))
        self.assertFalse(profile_policy.memlat_available('/nonexistent/synthetic/devfreq'))

    def test_invalid_overrides_never_change_model_or_readiness(self):
        for overrides in ({'ready':True}, {'model_path':'elsewhere'}, {'threads':True},
                          {'threads':0}, {'cpu_mask':'0x0'}, {'spec_type':'unknown'}):
            p = self.profile()
            p['memlat_overrides'] = overrides
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                profile_policy.resolve(p, available=True)

    def test_controller_does_not_inherit_other_models_overrides(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(controller.threading.Thread, 'start'):
            config = Path(tmp)/'runtime.json'
            config.write_text(json.dumps(dict(model_alias='base',selected_model='other',threads=4,
                memlat_overrides={'threads':8},profiles={'other':{'threads':6}})))
            manager = controller.Manager()
            with patch.object(controller, 'CONFIG', config), patch.object(profile_policy, 'memlat_available', return_value=True):
                self.assertEqual(manager.config('base')['threads'], 8)
                other = manager.config('other')
                self.assertEqual(other['threads'], 6)
                self.assertNotIn('execution_policy_mode', other)
                self.assertNotIn('memlat_overrides', other)

    def test_prefix_identity_separates_threads_and_execution_mode(self):
        profile = dict(sha256='a'*64, server_sha256='b'*64, context_size=4096, threads=4)
        payload = {'messages':[{'role':'system','content':'synthetic fixed prefix'}]}
        def key(p):
            return make_plan(payload, p, min_characters=0).key
        self.assertNotEqual(key(profile), key({**profile,'threads':6}))
        self.assertNotEqual(key(profile), key({**profile,'execution_policy_mode':'memlat'}))

    def test_classic_controller_marks_ready_reaps_before_guard_stop_and_lease_close(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(controller.threading.Thread, 'start'):
            root = Path(tmp)
            model = root/'model.gguf'
            model.write_bytes(b'synthetic')
            manager = controller.Manager()
            profile = dict(model_alias='synthetic',ready=True,model_path=str(model),
                           server_binary='/synthetic/never-executed',threads=6,context_size=4096)
            manager.config = Mock(return_value=profile)
            process = Mock(pid=123456)
            process.poll.return_value = None
            events = []
            process.wait.side_effect = lambda **kwargs: events.append('reaped')
            guard = Mock()
            guard.failure.return_value = None
            guard.metadata.return_value = {'reason':None}
            def stop():
                self.assertIsNotNone(manager.lease)
                events.append('guard_stopped')
            guard.stop.side_effect = stop
            response = Mock(status=200)
            response.read.return_value = b'{"data":[]}'
            connection = Mock()
            connection.getresponse.return_value = response
            with patch.object(controller, 'LEASE', root/'lease'), \
                 patch.object(controller, 'ENGINE_DIAGNOSTIC_LOG', root/'engine-stderr.log'), \
                 patch.object(controller, 'available_memory_bytes', return_value=30*1024**3), \
                 patch.object(controller.memory_policy, 'MemoryGuard', return_value=guard) as factory, \
                 patch.object(controller.subprocess, 'Popen', return_value=process) as launch, \
                 patch.object(controller.http.client, 'HTTPConnection', return_value=connection), \
                 patch.object(controller.os, 'killpg'):
                manager.begin(allow_start=True)
                factory.assert_called_once_with(loading_timeout_seconds=min(300.,controller.START_SECONDS))
                guard.start.assert_called_once_with(process)
                guard.mark_ready.assert_called_once_with()
                self.assertIs(manager.loaded_config, profile)
                self.assertIn('6', launch.call_args.args[0])
                # A repeat explicit request must keep this child's effective policy
                # even if detection/configuration changes after it was launched.
                manager.config.return_value = {**profile, 'threads':4}
                manager.begin(allow_start=True, model_alias='synthetic')
                self.assertIs(manager.loaded_config, profile)
                self.assertEqual(launch.call_count, 1)
                manager.status()
                self.assertEqual(launch.call_count, 1)
                self.assertEqual(guard.mark_ready.call_count, 1)
                manager.cleanup()
            self.assertEqual(events, ['reaped','guard_stopped'])
            self.assertIsNone(manager.lease)
            self.assertIsNone(manager.loaded_config)
