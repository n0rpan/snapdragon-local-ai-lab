"""CACHE-PERFECT P2 (lab): disk state cache usage, budget eviction, TTL, clear scopes, validation,
controller endpoints and engine flags. Model-free, on synthetic trees in a temporary HOME."""
import hashlib
import importlib.util
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from snapdragon_ai_lab.runtime import controller, oxe_runtime, state_cache as sc, storage_view
import stat
import fcntl

import openai_api_stack as stack_module
from openai_api_stack import FakeEngine, Stack
spec = importlib.util.spec_from_file_location('openai_api_check', Path(__file__).resolve().parents[1] / 'tools/openai_api_check.py')
check = importlib.util.module_from_spec(spec); spec.loader.exec_module(check)

M1, M2 = 'a' * 16, 'b' * 16
NOW = 2_000_000_000.0


def h(name):
    return hashlib.sha256(name.encode()).hexdigest()


def put(path, size, mtime=NOW - 86400):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        f.truncate(size)  # sparse: the size is what counts, no disk is used
    os.utime(path, (mtime, mtime))
    return path


def hint_of(name):
    """The manifest's `hint`: 32 hex of SHA-256 of the X-Oxe-Session value (engine scheme)."""
    return h(name)[:32]


def session(root, model, sid, blocks, atime, n_tokens=1000, block_size=1000, snaps=(), pin=False, hint=''):
    """A manifest over `blocks` (names; equal names are shared blocks), plus its block, snap and echo
    files. `hint` names the conversation (default: the sid; forks pass the same one); None = no hint,
    no echo file; a raw value starting with '!' is stored as is (a malformed hint)."""
    d = root / model
    put(d / 'identity.json', 10)
    refs = []
    for name in blocks:
        hh, c = h(name), '0' * 16
        put(d / 'blocks' / hh[:2] / ('%s.%s.kv' % (hh, c)), block_size)
        refs.append([hh, c])
    for key in snaps:
        put(d / 'snaps' / ('%s.%s.st' % (key, '1' * 16)), 500)
    hint = hint_of(sid if hint == '' else hint) if hint is not None and not hint.startswith('!') else hint
    if hint is not None and not hint.startswith('!'):
        put(d / 'echo' / (hint + '.jsonl'), 50)
    manifest = {'version': 1, 'identity_hash': 'f' * 64, 'sid': sid, 'hint': hint[1:] if hint and hint.startswith('!') else hint, 'created': atime, 'atime': atime,
                'n_tokens': n_tokens, 'blocks': refs, **({'pin': True} if pin else {}),
                'snaps': [{'pos': 256, 'class': 'head', 'key': k, 'c': '1' * 16, 'clean': True, 'bytes': 500} for k in snaps]}
    p = d / 'sessions' / (sid + '.json')
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifest))
    return manifest


def files(root):
    return sorted(str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() or p.is_symlink())


class Base(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.home = Path(tmp.name)
        self.root = sc.root(self.home)


class SettingsTests(Base):
    def test_defaults_validation_and_root_under_home(self):
        self.assertEqual(sc.settings({}), {'disk_budget_bytes': 64_000_000_000, 'ttl_secs': 604800, 'invalid': []})
        self.assertEqual(sc.settings({'cache': {'disk_budget_bytes': 0}})['disk_budget_bytes'], 0)
        for bad in (3_999_999_999, -1, 5e9, '64000000000', True, None):
            self.assertFalse(sc.valid_budget(bad), bad)
            self.assertEqual(sc.settings({'cache': {'disk_budget_bytes': bad}})['invalid'], ['disk_budget_bytes'])
        self.assertTrue(sc.valid_budget(4_000_000_000))
        self.assertEqual(sc.settings({'cache': {'ttl_secs': 3600}})['ttl_secs'], 3600)
        self.assertEqual(self.root, self.home / '.cache/local-ai/snapdragon-engine/state')
        outside = tempfile.TemporaryDirectory()
        self.addCleanup(outside.cleanup)
        (self.home / '.cache/local-ai/snapdragon-engine').mkdir(parents=True)
        (self.home / '.cache/local-ai/snapdragon-engine/state').symlink_to(outside.name)
        with self.assertRaises(ValueError):
            sc.root(self.home)

    def test_engine_args_for_moe_not_flash_not_at_zero(self):
        cfg = {'backend_kind': 'oxe', 'cache': {'disk_budget_bytes': 8_000_000_000, 'ttl_secs': 3600}}
        yes = lambda binary: True
        self.assertEqual(oxe_runtime.state_cache_args(cfg, self.home, yes),
                         ['--state-cache', str(self.root), '--state-budget-bytes', '8000000000', '--state-ttl-secs', '3600'])
        self.assertEqual(oxe_runtime.state_cache_args({**cfg, 'oxe_flash': {'cache_gib': 20}}, self.home, yes), [])
        self.assertEqual(oxe_runtime.state_cache_args({'cache': {'disk_budget_bytes': 0}}, self.home, yes), [])
        self.assertIn('--state-budget-bytes', oxe_runtime.state_cache_args({}, self.home, yes))  # default 64 GB

    def test_flags_go_only_to_an_engine_that_knows_them(self):
        """The ordering gate: a lab installed before the engine (an older oxe-server exits on an
        unknown flag) sends no state-cache flags; the same lab sends them once the engine has them."""
        def engine(name, usage):
            path = self.home / name
            path.write_bytes(b'\x7fELF' + b'\0' * (1024**2 - 9) + usage.encode() + b'\0' * 100)  # a name across the read blocks
            return str(path)
        old = engine('oxe-old', 'usage: oxe-server --model PATH [--session-cache DIR]')
        new = engine('oxe-new', 'usage: oxe-server --model PATH [--state-cache DIR] [--state-budget-bytes N] [--state-ttl-secs N]')
        self.assertFalse(oxe_runtime.engine_has_state_cache(old))
        self.assertTrue(oxe_runtime.engine_has_state_cache(new))
        self.assertFalse(oxe_runtime.engine_has_state_cache(str(self.home / 'missing')))
        self.assertFalse(oxe_runtime.engine_has_state_cache(None))
        self.assertFalse(oxe_runtime.engine_has_state_cache(str(self.home)))
        cfg = {'backend_kind': 'oxe', 'cache': {'disk_budget_bytes': 8_000_000_000}}
        self.assertEqual(oxe_runtime.state_cache_args({**cfg, 'server_binary': old}, self.home), [])
        self.assertEqual(oxe_runtime.state_cache_args({**cfg, 'server_binary': new}, self.home)[0], '--state-cache')
        self.assertEqual(oxe_runtime.state_cache_args(cfg, self.home), [])  # no binary named: off

    def test_full_command_has_flags_for_moe_and_raw_flags_are_refused(self):
        profile = {**stack_module.oxe_profile('ornith', 'Ornith'), 'model_alias': 'ornith'}
        with patch.dict(os.environ, {'LOCAL_AI_ENABLE_OXE': '1', 'HOME': str(self.home)}):
            self.assertNotIn('--state-cache', oxe_runtime.command(profile, 18080))  # the synthetic binary does not exist
        with patch.dict(os.environ, {'LOCAL_AI_ENABLE_OXE': '1', 'HOME': str(self.home)}), \
                patch.object(oxe_runtime, 'engine_has_state_cache', lambda binary: True):
            argv = oxe_runtime.command(profile, 18080)
            self.assertEqual(argv[argv.index('--state-cache') + 1], str(self.root))
            self.assertNotIn('--state-cache', oxe_runtime.command({**profile, 'cache': {'disk_budget_bytes': 0}}, 18080))
            raw = {**profile, 'oxe_args': profile['oxe_args'] + ['--state-cache', 'elsewhere']}
            self.assertFalse(oxe_runtime.eligible(raw))
            with self.assertRaises(ValueError):
                oxe_runtime._validate(raw)


class RootOverrideTests(Base):
    """G1 binding: LOCAL_AI_STATE_CACHE_ROOT (or runtime.json cache.root; env wins) moves every
    conversation store (state, legacy prefix and session caches, Flash prefix) below one private
    root inside the home; the weight caches stay put; anything invalid is refused, never ignored."""
    def setUp(self):
        super().setUp()
        self.gate = self.home / 'gates' / 'arm-a'
        env = patch.dict(os.environ, {'HOME': str(self.home), 'LOCAL_AI_ENABLE_OXE': '1'})
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop(sc.ROOT_ENV, None)

    def profile(self, **extra):
        p = {**stack_module.oxe_profile('ornith', 'Ornith'), 'model_alias': 'ornith', **extra}
        p['oxe_args'] = p['oxe_args'] + ['--prefix-cache', 'default', '--session-cache=/old/sessions']
        return p

    def command(self, profile):
        with patch.object(oxe_runtime, 'engine_has_state_cache', lambda binary: True):
            return oxe_runtime.command(profile, 18080)

    def flag(self, argv, name):
        for i, arg in enumerate(argv):
            if arg == name:
                return argv[i + 1]
            if arg.startswith(name + '='):
                return arg.split('=', 1)[1]
        return None

    def test_default_layout_is_unchanged(self):
        self.assertEqual(sc.root(), self.root)
        argv = self.command(self.profile())
        self.assertEqual(self.flag(argv, '--prefix-cache'), 'default')
        self.assertEqual(self.flag(argv, '--session-cache'), '/old/sessions')
        self.assertEqual(self.flag(argv, '--state-cache'), str(self.root))

    def test_env_override_moves_all_conversation_stores_not_the_weights(self):
        with patch.dict(os.environ, {sc.ROOT_ENV: str(self.gate)}):
            self.assertEqual(sc.root(), self.gate / 'state')
            argv = self.command(self.profile())
            self.assertEqual(self.flag(argv, '--state-cache'), str(self.gate / 'state'))
            self.assertEqual(self.flag(argv, '--prefix-cache'), str(self.gate / 'prefix'))
            self.assertEqual(self.flag(argv, '--session-cache'), str(self.gate / 'sessions'))
            self.assertEqual(sc.store_dir('prefix-flash'), self.gate / 'prefix-flash')
            for d in (self.home / 'gates', self.gate, self.gate / 'state', self.gate / 'prefix', self.gate / 'sessions'):
                self.assertEqual(stat.S_IMODE(d.stat().st_mode), 0o700, d)
            # weights: the model file and the engine's arena/sidecar cache root are not moved
            self.assertEqual(self.flag(argv, '--model'), '/synthetic/ornith.gguf')
            self.assertEqual(storage_view.cache_root(self.home), self.home / '.cache/local-ai')
            self.assertFalse(any(str(self.gate) in a for a in argv if not a.startswith(str(self.gate))
                                 and a.split('=', 1)[0] not in ('--prefix-cache', '--session-cache')))
            # GC/clear/usage use the same root
            session(self.gate / 'state', M1, 's1', ['a'], NOW - 9 * 86400)
            self.assertEqual(sc.usage(sc.root(), 1, 604800, NOW)['sessions']['count'], 1)

    def test_runtime_json_root_and_env_wins(self):
        other = self.home / 'from-config'
        profile = self.profile(cache={'root': str(other)})
        self.assertEqual(sc.root(runtime=profile), other / 'state')
        self.assertEqual(self.flag(self.command(profile), '--prefix-cache'), str(other / 'prefix'))
        with patch.dict(os.environ, {sc.ROOT_ENV: str(self.gate)}):
            self.assertEqual(sc.root(runtime=profile), self.gate / 'state')

    def test_invalid_overrides_are_refused_not_ignored(self):
        outside = tempfile.TemporaryDirectory()
        self.addCleanup(outside.cleanup)
        escape = self.home / 'escape'
        escape.symlink_to(outside.name)
        for bad in ('relative/root', outside.name, str(escape / 'arm'), str(self.home), '/'):
            with patch.dict(os.environ, {sc.ROOT_ENV: bad}), self.subTest(bad=bad):
                with self.assertRaisesRegex(ValueError, 'state_cache_root_override_invalid'):
                    sc.root()
                with self.assertRaisesRegex(ValueError, 'state_cache_root_override_invalid'):
                    self.command(self.profile())
        with self.assertRaisesRegex(ValueError, 'state_cache_root_override_invalid'):
            sc.root(runtime={'cache': {'root': 7}})
        self.assertEqual(os.listdir(outside.name), [])

    def test_controller_reports_a_bad_override(self):
        stack = Stack(1)  # no engine needed: both routes fail before any backend call
        self.addCleanup(stack.close)
        with patch.dict(os.environ, {sc.ROOT_ENV: 'relative'}):
            status, _, _ = check.call(stack.base, 'GET', '/_local/cache')
            self.assertEqual(status, 503)
            self.assertEqual(check.call(stack.base, 'POST', '/_local/cache/clear', {'scope': 'all'})[0], 503)


class UsageGcTests(Base):
    def test_usage_on_a_synthetic_tree(self):
        session(self.root, M1, 's1', ['x', 'y'], NOW - 3600)
        session(self.root, M2, 's2', ['z'], NOW - 60, snaps=['k1'])
        put(self.root / M1 / 'blocks' / 'ab' / 'fresh.pending', 999, NOW - 5)       # in flight: ignored
        put(self.root / M1 / 'snaps' / 'old.pending', 7, NOW - 3600)                 # stale: counted
        put(self.root / M1 / 'notes.txt', 12345)                                       # not a store name
        (self.root / 'not-a-model').mkdir()
        got = sc.usage(self.root, 64_000_000_000, 604800, NOW)
        m1 = 10 + 2000 + 50 + (self.root / M1 / 'sessions/s1.json').stat().st_size + 7
        m2 = 10 + 1000 + 500 + 50 + (self.root / M2 / 'sessions/s2.json').stat().st_size
        self.assertEqual(got['models'], {M1: m1, M2: m2})
        self.assertEqual(got['used_bytes'], m1 + m2)
        self.assertEqual(got['sessions'], {'count': 2, 'oldest_atime': NOW - 3600, 'newest_atime': NOW - 60})
        self.assertEqual(sc.usage(self.home / 'missing', 1, 1, NOW)['used_bytes'], 0)

    def test_budget_lowering_evicts_lowest_value_first_keeps_pinned_and_shared(self):
        # cheap: few tokens per byte, old; dear: many tokens, recent; pinned: used 2 min ago
        session(self.root, M1, 'cheap', ['shared', 'c1', 'c2'], NOW - 48 * 3600, n_tokens=256)
        session(self.root, M1, 'dear', ['shared', 'd1'], NOW - 3600, n_tokens=60000)
        session(self.root, M2, 'pinned', ['p1', 'p2', 'p3'], NOW - 120, n_tokens=10)
        before = sc.usage(self.root, 0, 604800, NOW)['used_bytes']
        freed = sc.gc(self.root, before - 1500, 604800, NOW)  # one session's worth must go
        left = files(self.root)
        self.assertFalse((self.root / M1 / 'sessions/cheap.json').exists())
        self.assertFalse((self.root / M1 / 'echo' / (hint_of('cheap') + '.jsonl')).exists())
        self.assertTrue((self.root / M1 / 'echo' / (hint_of('dear') + '.jsonl')).exists())
        self.assertTrue((self.root / M1 / 'sessions/dear.json').exists())
        self.assertTrue((self.root / M2 / 'sessions/pinned.json').exists())
        shared = h('shared')
        self.assertIn('%s/blocks/%s/%s.%s.kv' % (M1, shared[:2], shared, '0' * 16), left)  # still referenced by dear
        for gone in ('c1', 'c2'):
            self.assertNotIn('%s/blocks/%s/%s.%s.kv' % (M1, h(gone)[:2], h(gone), '0' * 16), left)
        self.assertEqual(freed, before - sc.usage(self.root, 0, 604800, NOW)['used_bytes'])
        # far below everything: all but the pinned session go, and its blocks stay
        sc.gc(self.root, 1, 604800, NOW)
        self.assertEqual(sorted(p.name for p in self.root.rglob('*.json') if p.parent.name == 'sessions'), ['pinned.json'])
        self.assertEqual(len(list((self.root / M1).rglob('*.kv'))), 0)
        self.assertEqual(len(list((self.root / M2).rglob('*.kv'))), 3)
        # budget 0 (tier off) keeps nothing, pins included
        sc.gc(self.root, 0, 604800, NOW)
        self.assertEqual(list(self.root.rglob('*.kv')), [])

    def test_ttl_expired_sessions_go_even_under_budget(self):
        session(self.root, M1, 'stale', ['a'], NOW - 8 * 86400, n_tokens=100000)
        session(self.root, M1, 'fresh', ['b'], NOW - 86400)
        self.assertGreater(sc.gc(self.root, 64_000_000_000, 7 * 86400, NOW), 0)
        self.assertEqual(sorted(p.name for p in (self.root / M1 / 'sessions').iterdir()), ['fresh.json'])

    def test_unreadable_manifest_makes_the_model_opaque_and_young_orphans_stay(self):
        session(self.root, M1, 'old', ['a'], NOW - 9 * 86400)
        (self.root / M1 / 'sessions/v2.json').write_text('{"version": 2}')
        sc.gc(self.root, 64_000_000_000, 7 * 86400, NOW)
        self.assertTrue((self.root / M1 / 'blocks' / h('a')[:2]).exists())  # nothing swept in an opaque model
        session(self.root, M2, 'live', ['k'], NOW - 60)
        young = put(self.root / M2 / 'blocks' / h('new')[:2] / ('%s.%s.kv' % (h('new'), '2' * 16)), 5, time.time())
        sc.gc(self.root, 64_000_000_000, 7 * 86400, time.time())
        self.assertTrue(young.exists())  # an in-flight block written before its manifest

    def test_symlinks_are_never_followed_or_deleted(self):
        session(self.root, M1, 'old', ['a'], NOW - 9 * 86400)
        victim = self.home / 'precious.kv'
        victim.write_bytes(b'keep')
        hh = h('link')
        link = self.root / M1 / 'blocks' / hh[:2] / ('%s.%s.kv' % (hh, '0' * 16))
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(victim)
        other = self.home / 'elsewhere'
        other.mkdir()
        put(other / 'sessions' / 'x.json', 3)
        (self.root / M2).symlink_to(other)
        sc.clear(self.root, 'all')
        self.assertEqual(victim.read_bytes(), b'keep')
        self.assertTrue((other / 'sessions/x.json').exists())
        self.assertTrue(link.is_symlink())


class ClearTests(Base):
    def test_clear_scopes(self):
        session(self.root, M1, 's1', ['shared', 'a'], NOW - 60)
        session(self.root, M1, 's2', ['shared', 'b'], NOW - 60)
        session(self.root, M2, 's3', ['c'], NOW - 60)
        freed = sc.clear(self.root, 'session', session='s1', now=NOW)
        self.assertGreater(freed, 0)
        left = files(self.root)
        self.assertNotIn('%s/sessions/s1.json' % M1, left)
        self.assertNotIn('%s/blocks/%s/%s.%s.kv' % (M1, h('a')[:2], h('a'), '0' * 16), left)
        self.assertIn('%s/blocks/%s/%s.%s.kv' % (M1, h('shared')[:2], h('shared'), '0' * 16), left)
        sc.clear(self.root, 'model', model=M1)
        self.assertFalse((self.root / M1).exists())
        self.assertTrue((self.root / M2 / 'sessions/s3.json').exists())
        put(self.root / M2 / 'unrelated.txt', 3)
        sc.clear(self.root, 'all')
        self.assertEqual(files(self.root), ['%s/unrelated.txt' % M2])
        for bad in ({'scope': 'nope'}, {'scope': 'model'}, {'scope': 'model', 'model': '../x'},
                    {'scope': 'session', 'session': '../../x'}):
            with self.assertRaises(ValueError, msg=bad):
                sc.clear(self.root, **bad)


class EchoFileTests(Base):
    """P2b: echo/<hint>.jsonl belongs to every manifest carrying that hint (forks of one conversation)."""
    def echo(self, model, name):
        return self.root / model / 'echo' / (hint_of(name) + '.jsonl')

    def test_forks_share_the_echo_file_until_the_last_one_goes(self):
        session(self.root, M1, 'fork-a', ['a'], NOW - 60, hint='conv')
        session(self.root, M1, 'fork-b', ['b'], NOW - 60, hint='conv')
        session(self.root, M2, 'other', ['c'], NOW - 60, hint='conv')  # same hint, another model
        sc.clear(self.root, 'session', model=M1, session='fork-a', now=NOW)
        self.assertFalse((self.root / M1 / 'sessions/fork-a.json').exists())
        self.assertTrue(self.echo(M1, 'conv').exists())  # fork-b still carries the hint
        sc.clear(self.root, 'session', model=M1, session='fork-b', now=NOW)
        self.assertFalse(self.echo(M1, 'conv').exists())  # the last one took it
        self.assertTrue(self.echo(M2, 'conv').exists())   # a hint is per model

    def test_eviction_keeps_a_shared_echo_and_counts_it_once(self):
        session(self.root, M1, 'old', ['a'], NOW - 9 * 86400, hint='conv')
        session(self.root, M1, 'new', ['b'], NOW - 86400, hint='conv')
        sc.gc(self.root, 64_000_000_000, 7 * 86400, NOW)  # TTL: only 'old' goes
        self.assertFalse((self.root / M1 / 'sessions/old.json').exists())
        self.assertTrue(self.echo(M1, 'conv').exists())
        # a budget plan that must free 'new' accounts for its echo file (now exclusive)
        models = sc.scan(self.root)
        used = sc.usage(self.root, 0, 604800, NOW)['used_bytes']
        self.assertEqual(sc.plan(models, used - 1, 604800, NOW), [(M1, 'new')])
        freed = sc.gc(self.root, used - 1, 604800, NOW)
        self.assertFalse(self.echo(M1, 'conv').exists())
        self.assertEqual(freed, used - sc.usage(self.root, 0, 604800, NOW)['used_bytes'])

    def test_null_and_malicious_hints_delete_no_echo_file(self):
        keep = put(self.root / M1 / 'echo' / (hint_of('x') + '.jsonl'), 50)
        session(self.root, M1, 'nohint', ['a'], NOW - 60, hint=None)
        session(self.root, M1, 'evil', ['b'], NOW - 60, hint='!../x')
        session(self.root, M1, 'upper', ['c'], NOW - 60, hint='!' + hint_of('x').upper())
        outside = put(self.root / M1 / 'x.jsonl', 5)
        for sid in ('nohint', 'evil', 'upper'):
            sc.clear(self.root, 'session', model=M1, session=sid, now=NOW)
            self.assertFalse((self.root / M1 / 'sessions' / (sid + '.json')).exists())
        self.assertTrue(keep.exists())
        self.assertTrue(outside.exists())
        self.assertIsNone(sc.parse_manifest({'version': 1, 'blocks': [], 'atime': 1, 'hint': '../x'})['hint'])


class EndpointTests(Base):
    @classmethod
    def setUpClass(cls):
        cls.engine = FakeEngine()
        cls.addClassCleanup(cls.engine.close)

    def stack(self, **config):
        env = patch.dict(os.environ, {'HOME': str(self.home)})
        env.start()
        self.addCleanup(env.stop)
        stack = Stack(self.engine.port, **config)
        self.addCleanup(stack.close)
        return stack

    def test_get_usage_budget_and_clear_without_an_engine(self):
        now = time.time()
        session(self.root, M1, 'old', ['a', 'b'], now - 3 * 86400, block_size=3_000_000_000 // 2)
        session(self.root, M1, 'new', ['c'], now - 60, block_size=1000)
        stack = self.stack()
        status, _, value = check.call(stack.base, 'GET', '/_local/cache')
        self.assertEqual(status, 200)
        self.assertEqual((value['budget_bytes'], value['sessions']['count']), (64_000_000_000, 2))
        self.assertGreater(value['used_bytes'], 3_000_000_000 - 1)
        for bad in ({'bytes': 1000}, {'bytes': '5000000000'}, {'bytes': 5e9}, {'bytes': True}, {'size': 5_000_000_000}):
            self.assertEqual(check.call(stack.base, 'POST', '/_local/cache/budget', bad)[0], 400, bad)
        status, _, got = check.call(stack.base, 'POST', '/_local/cache/budget', {'bytes': 4_000_000_000})
        self.assertEqual((status, got['budget_bytes'], got['freed_bytes']), (200, 4_000_000_000, 0))
        self.assertEqual(json.loads(stack.config_path.read_text())['cache'], {'disk_budget_bytes': 4_000_000_000})
        status, _, got = check.call(stack.base, 'POST', '/_local/cache/budget', {'bytes': 0})
        self.assertEqual(status, 200)
        self.assertGreater(got['freed_bytes'], 3_000_000_000 - 1)
        self.assertEqual(got['used_bytes'], 10)  # identity.json only
        self.assertEqual(check.call(stack.base, 'POST', '/_local/cache/clear', {'scope': 'model'})[0], 400)
        status, _, got = check.call(stack.base, 'POST', '/_local/cache/clear', {'scope': 'model', 'model': M1})
        self.assertEqual((status, got['state_cache']['freed_bytes']), (200, 10))
        self.assertEqual(check.call(stack.base, 'POST', '/_local/cache/clear')[0], 200)  # no body: as before + all

    def test_clear_is_refused_while_a_request_runs(self):
        stack = self.stack()
        stack.manager.active = 1
        try:
            self.assertEqual(check.call(stack.base, 'POST', '/_local/cache/clear', {'scope': 'all'})[0], 409)
        finally:
            stack.manager.active = 0

    def test_loaded_engine_gets_budget_and_clear_forwarded(self):
        stack = self.stack()
        calls = []

        def engine(method, path, payload=None):
            calls.append((method, path, payload))
            return {'budget_bytes': 5_000_000_000, 'used_bytes': 7, 'freed_bytes': 11}
        stack.manager.state_engine = True
        stack.manager.process = type('P', (), {'poll': lambda self: None, 'pid': None})()
        session(self.root, M1, 'old', ['a'], time.time() - 30 * 86400)
        with patch.object(stack.manager, 'state_engine_json', engine):
            status, _, got = check.call(stack.base, 'POST', '/_local/cache/budget', {'bytes': 5_000_000_000})
            self.assertEqual((status, got['freed_bytes'], got['applied_by']), (200, 11, 'engine'))
            status, _, got = check.call(stack.base, 'POST', '/_local/cache/clear', {'scope': 'session', 'session': 's9'})
            self.assertEqual((status, got['state_cache']['applied_by']), (200, 'engine'))
            self.assertEqual(check.call(stack.base, 'POST', '/_local/cache/clear', {'scope': 'model', 'model': M1})[0], 409)
            self.assertEqual(check.call(stack.base, 'GET', '/_local/cache')[2]['engine'], {'budget_bytes': 5_000_000_000, 'used_bytes': 7})
        self.assertEqual(calls[:2], [('POST', '/state-cache/budget', {'bytes': 5_000_000_000}),
                                     ('POST', '/state-cache/clear', {'scope': 'session', 'session': 's9'})])
        self.assertTrue((self.root / M1 / 'sessions/old.json').exists())  # Python never evicts under a live engine
        stack.manager.process = None


class LeaseAndGuardTests(Base):
    """Codex review item 3: the budget write is TEST-ISO guarded before it mutates, and the
    unloaded Python GC/clear hold the canonical inference lease across planning and deletion."""
    def setUp(self):
        super().setUp()
        self.manager = controller.Manager()
        self.manager.stop_event.set(); self.manager.thread.join(2)
        self.addCleanup(self.manager.close)
        self.config = self.home / 'runtime.json'
        self.config.write_text('{}')
        self.lease = self.home / 'state' / 'inference.lock'
        for p in (patch.dict(os.environ, {'HOME': str(self.home)}), patch.object(controller, 'CONFIG', self.config),
                  patch.object(controller, 'LEASE', self.lease)):
            p.start()
            self.addCleanup(p.stop)

    def hold_lease(self):
        self.lease.parent.mkdir(parents=True, exist_ok=True)
        holder = open(self.lease, 'a')
        fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)  # another process's job (flock is per open file)
        self.addCleanup(holder.close)
        return holder

    def test_budget_write_refuses_a_production_config_before_writing(self):
        with patch.object(controller, 'REAL_PRODUCTION_PATHS', (self.home,)):
            with self.assertRaisesRegex(RuntimeError, 'production path'):
                self.manager.set_cache_budget(4_000_000_000)
        self.assertEqual(self.config.read_text(), '{}')
        self.assertEqual(sorted(p.name for p in self.home.iterdir() if p.name.startswith('runtime')), ['runtime.json'])

    def test_lease_guard_refuses_a_production_lease(self):
        session(self.root, M1, 'old', ['a'], NOW - 9 * 86400)
        with patch.object(controller, 'REAL_PRODUCTION_PATHS', (self.lease.parent,)), self.manager.mu:
            with self.assertRaisesRegex(RuntimeError, 'production path'):
                self.manager.clear_state_cache('all')
        self.assertFalse(self.lease.parent.exists())
        self.assertTrue((self.root / M1 / 'sessions/old.json').exists())

    def test_gc_and_clear_refuse_while_another_job_holds_the_lease(self):
        session(self.root, M1, 'old', ['a'], NOW - 9 * 86400)
        holder = self.hold_lease()
        with self.assertRaisesRegex(controller.Unavailable, 'budget was saved'):
            self.manager.set_cache_budget(0)
        self.assertEqual(json.loads(self.config.read_text())['cache'], {'disk_budget_bytes': 0})
        with self.manager.mu:
            for scope, kw in (('all', {}), ('session', {'session': 'old'}), ('model', {'model': M1})):
                with self.assertRaises(controller.Unavailable, msg=scope):
                    self.manager.clear_state_cache(scope, **kw)
        self.assertTrue((self.root / M1 / 'sessions/old.json').exists())  # nothing planned or deleted
        holder.close()
        self.assertEqual(self.manager.set_cache_budget(0)['applied_by'], 'controller')
        self.assertFalse((self.root / M1 / 'sessions/old.json').exists())

    def test_the_lease_is_held_during_the_deletion_and_released_after(self):
        session(self.root, M1, 'old', ['a'], NOW - 9 * 86400)
        seen = []

        def clear(*args, **kw):
            probe = open(self.lease, 'a')
            try:
                fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
                seen.append('free')
            except BlockingIOError:
                seen.append('held')
            finally:
                probe.close()
            return 0
        with patch.object(sc, 'clear', clear), patch.object(sc, 'gc', clear), self.manager.mu:
            self.manager.clear_state_cache('all')
        with patch.object(sc, 'gc', clear):
            self.manager.set_cache_budget(4_000_000_000)
        self.assertEqual(seen, ['held', 'held'])
        self.hold_lease()  # released: a new holder gets it at once

    def test_a_loaded_model_owns_the_lease_already(self):
        session(self.root, M1, 'old', ['a'], NOW - 9 * 86400)
        self.hold_lease()  # stands for the lease this controller took at load
        self.manager.lease = object()
        try:
            with self.manager.mu:
                self.assertEqual(self.manager.clear_state_cache('all')['applied_by'], 'controller')
        finally:
            self.manager.lease = None
        self.assertFalse((self.root / M1 / 'sessions/old.json').exists())

class LegacyStoreTests(Base):
    """Engine contract mirrored on the unloaded path: budget 0 means all conversation disk
    persistence off (no legacy --prefix-cache/--session-cache at the next launch) and clear
    all also empties the legacy Rust prefix store."""
    def setUp(self):
        super().setUp()
        for p in (patch.dict(os.environ, {'HOME': str(self.home), 'LOCAL_AI_ENABLE_OXE': '1'}),):
            p.start()
            self.addCleanup(p.stop)
        os.environ.pop(sc.ROOT_ENV, None)

    def test_budget_zero_drops_the_legacy_stores_from_a_moe_launch(self):
        p = {**stack_module.oxe_profile('ornith', 'Ornith'), 'model_alias': 'ornith'}
        p['oxe_args'] = p['oxe_args'] + ['--prefix-cache', 'default', '--prefix-cache-max-bytes', '9',
                                         '--session-cache=/s', '--ctx-checkpoints', '4']
        with patch.object(oxe_runtime, 'engine_has_state_cache', lambda binary: True):
            on = oxe_runtime.command(p, 18080)
            off = oxe_runtime.command({**p, 'cache': {'disk_budget_bytes': 0}}, 18080)
        self.assertIn('--prefix-cache', on)
        self.assertFalse({'--prefix-cache', '--prefix-cache-max-bytes', '--session-cache=/s', '--state-cache', '9'} & set(off))
        self.assertEqual(off[-2:], ['--ctx-checkpoints', '4'])  # everything else kept, in order

    def test_unloaded_clear_all_empties_the_legacy_rust_prefix_store(self):
        def entry(directory, name, model='a' * 64):
            identity = ('model=sha256:%s server=sha256:%s ' % (model, 'b' * 64)).encode()
            path = directory / (name * 64 + '.bin')
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'OXEPFX01' + len(identity).to_bytes(4, 'little') + identity + b'\0' * 64)
            return path
        prefix = self.home / '.cache/local-ai/snapdragon-engine/prefix'
        custom = self.home / 'custom-prefix'
        e1, e2 = entry(prefix, '1'), entry(custom, '2')
        other = prefix / 'notes.txt'
        other.write_text('keep')
        arena = self.home / '.cache/local-ai/snapdragon-engine/weights.arena'
        arena.write_bytes(b'w')
        config = self.home / 'runtime.json'
        profile = {**stack_module.oxe_profile('ornith', 'Ornith'), 'model_alias': 'ornith'}
        profile['oxe_args'] = profile['oxe_args'] + ['--prefix-cache', str(custom)]
        config.write_text(json.dumps({'profiles': {'ornith': profile}}))
        session(self.root, M1, 's1', ['a'], NOW - 60)
        manager = controller.Manager()
        manager.stop_event.set(); manager.thread.join(2)
        self.addCleanup(manager.close)
        with patch.object(controller, 'CONFIG', config), patch.object(controller, 'LEASE', self.home / 'lease'), manager.mu:
            manager.clear_state_cache('session', session='s1')
            self.assertTrue(e1.exists() and e2.exists())  # session/model scopes keep legacy entries
            got = manager.clear_state_cache('all')
        self.assertFalse(e1.exists() or e2.exists())
        self.assertTrue(other.exists() and arena.exists())
        self.assertGreater(got['freed_bytes'], 0)

class PreloadTests(Base):
    """P2b: an explicit load (POST /_local/load) of an oxe engine with the state cache and the
    preload route sends one POST /state-cache/preload in the background; nothing else does."""
    def setUp(self):
        super().setUp()
        self.engine = FakeEngine()
        self.addCleanup(self.engine.close)
        self.ledger = self.home / 'ledger'  # private: no other test's rows can name a session here
        env = patch.dict(os.environ, {'HOME': str(self.home), 'LOCAL_AI_LEDGER_DIR': str(self.ledger)})
        env.start()
        self.addCleanup(env.stop)

    def stack(self, capable=True):
        stack = Stack(self.engine.port)
        self.addCleanup(stack.close)
        manager, plain = stack.manager, stack.manager.begin

        def begin(**kw):  # what the real begin does at launch: count it, record the probe
            loads = manager.loads
            process = plain(**kw)
            if manager.loads != loads:
                manager.launches += 1
                manager.state_engine = manager.state_preload = capable
            return process
        manager.begin = begin
        return stack

    def preloads(self, wait=1.0):
        deadline = time.monotonic() + wait
        while time.monotonic() < deadline and not any(r['path'] == '/state-cache/preload' for r in self.engine.requests):
            time.sleep(0.02)
        return [r['body'] for r in self.engine.requests if r['path'] == '/state-cache/preload']

    def chat(self, stack, session_id):
        return check.call(stack.base, 'POST', '/v1/chat/completions',
                          {'model': stack_module.ORNITH, 'messages': [{'role': 'user', 'content': 'hi'}], 'max_tokens': 4},
                          headers={'X-Session-Id': session_id})

    def test_explicit_load_fires_exactly_one_preload(self):
        stack = self.stack()
        self.assertEqual(check.call(stack.base, 'POST', '/_local/load', {})[0], 200)
        self.assertEqual(self.preloads(), [{}])  # no session known yet
        self.assertEqual(check.call(stack.base, 'POST', '/_local/load', {})[0], 200)  # already up: no relaunch
        time.sleep(0.2)
        self.assertEqual(self.preloads(), [{}])

    def test_organic_request_auto_load_fires_none_and_its_hint_feeds_the_next_load(self):
        stack = self.stack()
        status = self.chat(stack, 'conv-1')[0]
        self.assertEqual(status, 200)
        self.assertEqual(self.preloads(wait=0.3), [])
        hint = self.engine.requests[-1]['headers'].get('X-Oxe-Session')
        self.assertRegex(hint, r'^[0-9a-f]{16}$')
        self.assertEqual(check.call(stack.base, 'POST', '/_local/unload', {})[0], 200)
        self.assertEqual(check.call(stack.base, 'POST', '/_local/load', {})[0], 200)
        self.assertEqual(self.preloads(), [{'session': hint}])

    def test_engine_without_the_route_fires_none(self):
        stack = self.stack(capable=False)
        self.assertEqual(check.call(stack.base, 'POST', '/_local/load', {})[0], 200)
        self.assertEqual(self.preloads(wait=0.3), [])

    def test_a_failing_engine_only_logs(self):
        stack = self.stack()
        self.engine.close()  # nothing listens: the background POST fails, the load still answers
        with self.assertLogs('local_ai.controller', 'WARNING'):
            self.assertEqual(check.call(stack.base, 'POST', '/_local/load', {})[0], 200)
            time.sleep(0.5)

    def test_hint_from_the_request_ledger_after_a_restart(self):
        from snapdragon_ai_lab.runtime import request_ledger
        d = self.ledger
        d.mkdir(exist_ok=True)
        rows = [{'model': stack_module.ORNITH, 'session': 'a' * 16}, {'model': stack_module.QWEN, 'session': 'b' * 16},
                {'model': stack_module.ORNITH, 'session': '../x'}]
        (d / 'requests-20261111.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows) + 'not json\n')
        self.assertEqual(request_ledger.last_session(stack_module.ORNITH), 'a' * 16)
        self.assertEqual(request_ledger.last_session(stack_module.QWEN), 'b' * 16)
        self.assertIsNone(request_ledger.last_session('other'))

    def test_probe_looks_for_the_capability_marker(self):
        old = self.home / 'oxe-old'
        old.write_bytes(b'\x7fELF' + b'\0' * 4096 + b'/state-cache/budget\0/state-cache/clear\0')
        # a binary with the route string but no marker (as an LTO build may lay it out): no
        route_only = self.home / 'oxe-route'
        route_only.write_bytes(b'\x7fELF' + b'\0' * 4096 + b'/state-cache/preload\0')
        new = self.home / 'oxe-new'
        marker = oxe_runtime.STATE_PRELOAD_MARKER.encode()
        new.write_bytes(b'\x7fELF' + b'\0' * (1024**2 - 10) + marker + b' (POST /state-cache/preload)\n')  # across the read blocks
        self.assertEqual(marker, b'oxe-capability state-preload-v1', 'the engine usage text and its CI grep use this exact literal')
        self.assertFalse(oxe_runtime.engine_has_state_preload(str(old)))
        self.assertFalse(oxe_runtime.engine_has_state_preload(str(route_only)))
        self.assertTrue(oxe_runtime.engine_has_state_preload(str(new)))
        self.assertFalse(oxe_runtime.engine_has_state_preload(None))
        self.assertFalse(oxe_runtime.engine_has_state_cache(str(new)))  # separate needles, separate answers


if __name__ == '__main__':
    unittest.main()
