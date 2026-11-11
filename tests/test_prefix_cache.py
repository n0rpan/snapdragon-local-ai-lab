"""Private synthetic prefix-state storage tests; no model or user data."""
import copy
import json
import os
from pathlib import Path
import stat
import tempfile
import unittest

from snapdragon_ai_lab.runtime.prefix_plan import make_plan
from snapdragon_ai_lab.runtime.prefix_store import PrefixStore


class PrefixCache(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / 'private-prefixes'
        self.profile = {'sha256': 'a' * 64, 'server_sha256': 'b' * 64,
                        'context_size': 16384, 'model_alias': 'synthetic-local',
                        'extra_args': ['--ctx-checkpoints', '4'],
                        'environment': {'LLAMA_CPU_SMALL_BATCH_LIMIT': '1'}}
        self.payload = {'messages': [{'role': 'system', 'content': 'Fixed synthetic coding instructions'},
                                     {'role': 'user', 'content': 'PRIVATE_USER_SENTINEL'},
                                     {'role': 'tool', 'content': 'PRIVATE_TOOL_SENTINEL'}],
                        'tools': [{'type': 'function', 'function': {'name': 'synthetic_read',
                                   'parameters': {'type': 'object', 'properties': {}}}}],
                        'chat_template_kwargs': {'enable_thinking': False}}

    def plan(self, payload=None, profile=None):
        result = make_plan(payload or self.payload, profile or self.profile, min_characters=0)
        self.assertIsNotNone(result)
        return result

    def commit(self, store, plan, content=b'opaque synthetic prefix state'):
        (store.root / store.pending(plan)).write_bytes(content)
        store.commit(plan)

    def test_private_state_survives_store_recreation_without_request_text(self):
        plan = self.plan()
        rendered = json.dumps(plan.primer)
        self.assertNotIn('PRIVATE_USER_SENTINEL', rendered)
        self.assertNotIn('PRIVATE_TOOL_SENTINEL', rendered)
        store = PrefixStore(self.root, 128)
        self.commit(store, plan)
        restored = PrefixStore(self.root, 128)
        self.assertEqual(restored.lookup(plan), plan.key + '.bin')
        self.assertEqual(stat.S_IMODE(self.root.stat().st_mode), 0o700)
        for path in self.root.iterdir():
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
        metadata = (self.root / (plan.key + '.json')).read_text()
        self.assertNotIn('Fixed synthetic coding instructions', metadata)
        self.assertNotIn('PRIVATE_USER_SENTINEL', metadata)

    def test_model_engine_settings_template_and_tools_use_distinct_cache_keys(self):
        original = self.plan()
        variants = []
        for key, value in [('sha256', 'c' * 64), ('server_sha256', 'd' * 64),
                           ('context_size', 8192), ('extra_args', ['--ctx-checkpoints', '2']),
                           ('environment', {'LLAMA_CPU_SMALL_BATCH_LIMIT': '3'})]:
            profile = copy.deepcopy(self.profile)
            profile[key] = value
            variants.append(self.plan(profile=profile))
        for field in ('chat_template_kwargs', 'tools'):
            payload = copy.deepcopy(self.payload)
            payload[field] = {'enable_thinking': True} if field == 'chat_template_kwargs' else []
            variants.append(self.plan(payload=payload))
        store = PrefixStore(self.root, 128)
        self.commit(store, original)
        self.assertEqual(len({original.key, *(plan.key for plan in variants)}), 8)
        for plan in variants:
            self.assertIsNone(store.lookup(plan))
        self.assertIsNotNone(store.lookup(original))

    def test_corruption_misses_and_discards_only_the_bad_entry(self):
        plan = self.plan()
        store = PrefixStore(self.root, 128)
        self.commit(store, plan)
        unknown = self.root / 'unrelated-synthetic-note'
        unknown.write_bytes(b'keep this fixture')
        (self.root / (plan.key + '.bin')).write_bytes(b'corrupt synthetic state')
        self.assertIsNone(store.lookup(plan))
        self.assertFalse((self.root / (plan.key + '.bin')).exists())
        self.assertFalse((self.root / (plan.key + '.json')).exists())
        self.assertEqual(unknown.read_bytes(), b'keep this fixture')

    def test_interrupted_pending_state_is_never_restored(self):
        plan = self.plan()
        store = PrefixStore(self.root, 128)
        pending = store.root / store.pending(plan)
        pending.write_bytes(b'incomplete synthetic checkpoint')
        restored = PrefixStore(self.root, 128)
        self.assertFalse(pending.exists())
        self.assertIsNone(restored.lookup(plan))

    def test_disk_budget_prunes_old_state_and_clear_preserves_unknown_files(self):
        first = self.plan()
        payload = copy.deepcopy(self.payload)
        payload['messages'][0]['content'] = 'Other fixed synthetic instructions'
        second = self.plan(payload=payload)
        store = PrefixStore(self.root, 40)
        self.commit(store, first, b'A' * 24)
        os.utime(self.root / (first.key + '.json'), (1, 1))
        self.commit(store, second, b'B' * 24)
        self.assertIsNone(store.lookup(first))
        self.assertIsNotNone(store.lookup(second))
        self.assertLessEqual(store.stats()['bytes'], 40)
        unknown = self.root / 'unrelated-synthetic-note'
        unknown.write_bytes(b'keep')
        store.clear()
        self.assertEqual(store.stats()['entries'], 0)
        self.assertEqual(unknown.read_bytes(), b'keep')


if __name__ == '__main__':
    unittest.main()
