"""Synthetic MEMLAT engine/environment override and V2 engine allowlist tests."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from snapdragon_ai_lab.runtime import classic_beta as beta, profile_policy

PATCHED = 'acd47ff19a161f9303be23ae1e99a80050579dad17071302ba77b3a0d6060efc'


class MemlatEngineOverride(unittest.TestCase):
    def profile(self):
        return {**copy.deepcopy(beta.POLICY), 'ready':False, 'model_alias':beta.ALIAS,
                'model_path':'/synthetic/model.gguf', 'server_binary':'/synthetic/old-engine',
                'sha256':beta.MODEL_SHA, 'expected_bytes':beta.MODEL_BYTES,
                'server_sha256':beta.SERVER_SHA,
                'prefix_cache':{'enabled':False, 'directory':'/synthetic/prefixes'},
                'validation':{'kind':'synthetic-stock'}, 'memlat_validation':{'kind':'synthetic-memlat'},
                'memlat_overrides':dict(server_binary='/synthetic/patched-engine', server_sha256=PATCHED,
                    environment={'LLAMA_CPU_SMALL_BATCH_LIMIT':'7', 'GGML_PATCHED_CACHE':'1'},
                    threads=8, small_batch_limit=1, prefix_cache_enabled=True)}

    def test_patched_sha_constant_and_legacy_sha_unchanged(self):
        self.assertEqual(beta.SERVER_SHA, '1bd3e439f357365c50fc8390cc2c7e4cde2f86f14fa1771a65f2323a307457b1')
        self.assertEqual(beta.PATCHED_SERVER_SHA, PATCHED)
        self.assertEqual(beta.candidate_engine_shas(), frozenset((beta.SERVER_SHA, PATCHED)))

    def test_stock_resolution_unchanged(self):
        original = self.profile()
        stock = profile_policy.resolve(original, available=False)
        self.assertEqual(stock, original)
        self.assertEqual(stock['server_sha256'], beta.SERVER_SHA)
        self.assertEqual(stock['server_binary'], '/synthetic/old-engine')
        self.assertEqual(stock['threads'], 4)
        self.assertEqual(stock['environment'], {'LLAMA_CPU_SMALL_BATCH_LIMIT':'3'})
        self.assertFalse(stock['prefix_cache']['enabled'])
        self.assertEqual(stock['validation'], {'kind':'synthetic-stock'})
        beta.require_candidate_profile(stock)

    def test_memlat_resolves_exact_engine_env_and_fresh_validation(self):
        original = self.profile()
        effective = profile_policy.resolve(original, available=True)
        self.assertEqual(effective['server_binary'], '/synthetic/patched-engine')
        self.assertEqual(effective['server_sha256'], PATCHED)
        # Environment merges first; small_batch_limit is applied after and wins.
        self.assertEqual(effective['environment'], {'LLAMA_CPU_SMALL_BATCH_LIMIT':'1', 'GGML_PATCHED_CACHE':'1'})
        self.assertEqual(effective['threads'], 8)
        self.assertTrue(effective['prefix_cache']['enabled'])
        self.assertEqual(effective['validation'], {'kind':'synthetic-memlat'})
        self.assertEqual(effective['execution_policy_mode'], 'memlat')
        self.assertEqual(original, self.profile())
        beta.require_candidate_profile(effective)
        # Engine SHA and environment are bound into the proof.
        stock = profile_policy.resolve(original, available=False)
        self.assertNotEqual(beta.policy_digest(effective), beta.policy_digest(stock))

    def test_environment_only_override_merges_with_base(self):
        profile = self.profile()
        profile['memlat_overrides'] = dict(environment={'EXTRA':'x'})
        effective = profile_policy.resolve(profile, available=True)
        self.assertEqual(effective['environment'], {'LLAMA_CPU_SMALL_BATCH_LIMIT':'3', 'EXTRA':'x'})
        self.assertEqual(effective['server_sha256'], beta.SERVER_SHA)

    def test_malformed_overrides_rejected(self):
        bad = [dict(server_binary='relative/engine', server_sha256=PATCHED),
               dict(server_binary='/x/engine\n', server_sha256=PATCHED),
               dict(server_binary=7, server_sha256=PATCHED),
               dict(server_binary='/x/engine', server_sha256=PATCHED.upper()),
               dict(server_binary='/x/engine', server_sha256=PATCHED[:-1]),
               dict(server_binary='/x/engine', server_sha256=None),
               dict(server_binary='/x/engine'),
               dict(server_sha256=PATCHED),
               dict(environment=['A=1']),
               dict(environment={'A':1}),
               dict(environment={1:'a'}),
               dict(environment={'A':None})]
        for overrides in bad:
            for available in (True, False):
                profile = self.profile()
                profile['memlat_overrides'] = overrides
                with self.subTest(overrides=overrides, available=available), self.assertRaises(ValueError):
                    profile_policy.resolve(profile, available=available)

    def test_candidate_rejects_unlisted_engine_sha(self):
        profile = self.profile()
        profile['memlat_overrides']['server_sha256'] = 'b' * 64
        effective = profile_policy.resolve(profile, available=True)
        with self.assertRaises(ValueError):
            beta.require_candidate_profile(effective)

    def test_patched_engine_requires_v2_proof_at_startup(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'engine'
            path.write_bytes(b'synthetic engine')
            actual = hashlib.sha256(b'synthetic engine').hexdigest()
            v2 = {'server_binary':str(path), 'server_sha256':actual, 'validation':{'kind':beta.CANDIDATE_KIND}}
            legacy = {'server_binary':str(path), 'server_sha256':actual, 'validation':{'kind':'ornith-classic-beta-v1'}}
            with patch.object(beta, 'PATCHED_SERVER_SHA', actual):
                beta.verify_engine(v2)
                with self.assertRaises(ValueError):
                    beta.verify_engine(legacy)
                with self.assertRaises(ValueError):
                    beta.verify_engine({**v2, 'server_sha256':beta.SERVER_SHA})
            with patch.object(beta, 'SERVER_SHA', actual):
                beta.verify_engine(legacy)
                beta.verify_engine(v2)
            with self.assertRaises(ValueError):
                beta.verify_engine(v2)


class MemlatArguments(unittest.TestCase):
    def test_explicit_arguments_preserve_stock_and_are_thread_normalized(self):
        original = {'extra_args':['--stock-only'], 'memlat_overrides':{
            'extra_args':['--load-mode','mmap','--threads-batch','10'], 'threads_batch':12}}
        self.assertEqual(profile_policy.resolve(original, available=False), original)
        resolved = profile_policy.resolve(original, available=True)
        self.assertEqual(resolved['extra_args'], ['--load-mode','mmap','--threads-batch','12'])
        self.assertEqual(original['extra_args'], ['--stock-only'])
        resolved['memlat_overrides']['extra_args'].append('--changed')
        self.assertEqual(original['memlat_overrides']['extra_args'][-1], '10')

    def test_malformed_arguments_are_rejected(self):
        for value in ('--load-mode mmap', [1], ['bad\nargument']):
            with self.subTest(value=value), self.assertRaises(ValueError):
                profile_policy.resolve({'memlat_overrides':{'extra_args':value}}, available=True)

if __name__ == '__main__':
    unittest.main()
