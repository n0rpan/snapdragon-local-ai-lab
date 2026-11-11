import argparse
import contextlib
import fcntl
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
sys.path.insert(0, str(TOOLS))
import storage_cleanup as cleanup
import repack_cache


class StorageCleanupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.home = Path(self.temp.name)
        self.cache = self.home / '.cache/local-ai'
        self.work = self.home / 'Work/local-ai'
        self.cache.mkdir(parents=True)
        self.work.mkdir(parents=True)
        self.now = time.time()

    def old(self, path):
        for root, dirs, files in os.walk(path, followlinks=False):
            for name in files + dirs:
                os.utime(Path(root) / name, (self.now - 9 * 86400,) * 2, follow_symlinks=False)
        os.utime(path, (self.now - 9 * 86400,) * 2)
        return path

    def cache_dir(self, name):
        path = self.cache / name
        path.mkdir()
        (path / 'payload').write_bytes(b'generated')
        return self.old(path)

    def plan(self, refs=None, protected=None, busy=None):
        return cleanup.plan(self.cache, self.work, refs or {}, protected or set(), busy or set(), self.now)

    def test_active_cache_child_protects_parent(self):
        parent = self.cache_dir('repack-parent')
        child = parent / 'live'
        child.mkdir()
        self.old(parent)
        actions, skipped = self.plan(refs={child: {}})
        self.assertFalse(actions)
        self.assertEqual(skipped[0]['reason'], 'referenced-or-in-use')

    def test_runtime_union_includes_stock_memlat_fallback_and_oxe_binary(self):
        cache = str(self.cache / 'repack')
        doc = dict(model_path='/models/main', server_binary='/bin/stock', sha256='a'*64,
                   server_sha256='b'*64, environment={'GGML_CPU_REPACK_CACHE': cache},
                   memlat_overrides=dict(server_binary='/bin/memlat', server_sha256='c'*64,
                                         oxe=dict(server_binary='/bin/rust', environment={})),
                   profiles={'fallback':dict(model_path='/models/fallback', sha256='d'*64)})
        refs, protected = cleanup.runtime_references(doc)
        self.assertEqual(refs[Path(cache)]['engines'], {'b'*64, 'c'*64})
        self.assertEqual(refs[Path(cache)]['models'], {'a'*64, 'd'*64})
        self.assertTrue({Path('/bin/rust').resolve(), Path('/models/fallback')} <= protected)

    def test_runtime_logical_symlink_path_is_preserved(self):
        parent = self.cache_dir('repack-parent')
        target = self.home / 'elsewhere'
        target.mkdir()
        link = parent / 'live'
        link.symlink_to(target, target_is_directory=True)
        self.old(parent)
        refs, protected = cleanup.runtime_references({'environment': {'GGML_CPU_REPACK_CACHE': str(link)}})
        self.assertIn(target, refs)
        self.assertFalse(self.plan(refs, protected)[0])
        self.assertTrue(link.is_symlink())

    def test_selectable_fallback_without_cache_keeps_its_model_identity(self):
        cache = self.cache/'repack'
        doc = dict(sha256='a'*64, server_sha256='b'*64,
                   environment={'GGML_CPU_REPACK_CACHE': str(cache)},
                   profiles={'fallback': dict(sha256='c'*64, environment={})})
        refs, _ = cleanup.runtime_references(doc)
        self.assertEqual(refs[cache]['models'], {'a'*64, 'c'*64})

    def test_unreadable_tree_is_skipped_not_partially_deleted(self):
        path = self.cache_dir('repack-old')
        with mock.patch.object(cleanup, 'tree_info', side_effect=PermissionError):
            actions, skipped = self.plan()
        self.assertFalse(actions)
        self.assertEqual(skipped[0]['reason'], 'unreadable-or-changed-tree')
        self.assertTrue(path.exists())

    def test_active_build_and_models_are_preserved(self):
        project = self.work / 'engine'
        target = project / 'target'
        target.mkdir(parents=True)
        (project / 'Cargo.toml').write_text('[workspace]')
        binary = target / 'executable'
        binary.write_bytes(b'built')
        models = project / 'models'
        models.mkdir()
        (models / 'keep.gguf').write_bytes(b'model')
        self.old(target)
        self.assertFalse(self.plan(busy={binary})[0])
        actions, _ = self.plan()
        self.assertEqual([Path(a['path']) for a in actions], [target])
        self.assertTrue((models / 'keep.gguf').exists())

    def test_configured_image_binary_and_split_model_protect_builds(self):
        for name in ('image-engine', 'split-model'):
            project = self.work/name
            target = project/'target'
            target.mkdir(parents=True)
            (project/'Cargo.toml').write_text('[workspace]')
            (target/'asset').write_bytes(b'keep')
            self.old(target)
        protected = cleanup.path_references({'binary': str(self.work/'image-engine/target/asset'),
                                             'shards': [{'path': str(self.work/'split-model/target/asset')}]})
        self.assertFalse(self.plan(protected=protected)[0])

    def test_oracle_keeps_complete_fixture_set_and_unknown_when_unindexed(self):
        root = self.cache / 'engine-oracle'
        root.mkdir()
        fixtures = []
        for i in range(3):
            case = {}
            for kind in ('control', 'capture'):
                path = root / f'fixture-{i}-{kind}'
                path.mkdir()
                (path / 'summary.json').write_text('{}')
                case[kind] = str(path)
            fixtures.append(case)
        old = root / 'model-production-old-run'
        old.mkdir()
        (old / 'summary.json').write_text('{}')
        self.old(old)
        unknown = root / 'other-production-run'
        unknown.mkdir()
        (unknown / 'summary.json').write_text('{}')
        self.old(unknown)
        recent = root / 'model-production-recent'
        recent.mkdir()
        (recent / 'summary.json').write_text('{}')
        self.assertFalse(self.plan()[0])
        (root / 'model-production-summary.json').write_text(json.dumps(dict(passed=True, fixtures=fixtures)))
        actions, _ = self.plan()
        self.assertEqual([Path(a['path']) for a in actions], [old])
        (root / 'model-production-summary.json').write_text('{"passed":false}')
        with self.assertRaises(ValueError): self.plan()

    def test_repack_scan_never_follows_symlinks(self):
        root = self.cache_dir('repack-test')
        outside = self.home / 'outside.rpk'
        outside.write_bytes(b'not a cache')
        (root / 'link.rpk').symlink_to(outside)
        self.assertEqual(repack_cache.scan(root), ([], []))
        link = self.home / 'cache-link'
        link.symlink_to(root, target_is_directory=True)
        with self.assertRaises(ValueError): repack_cache.scan(link)
        self.assertEqual(outside.read_bytes(), b'not a cache')

    def test_prune_keeps_runtime_identity_union_and_io_errors(self):
        root = self.cache_dir('repack-test')
        entries = []
        for name, engine, model, status in [('live1','e1','m1','ok'),('live2','e2','m2','ok'),
                                            ('obsolete','old','m1','ok'),('unreadable1','','','io-error: denied'),
                                            ('unreadable2','','','io-error: denied')]:
            path = root / (name + '.rpk')
            path.write_bytes(b'generated')
            entries.append(dict(path=str(path),status=status,key=dict(engine=engine,model_sha256=model,tensor='w'),file_size=9,last_use=self.now))
        args = argparse.Namespace(dir=str(root),older_than=None,build=None,patch=None,
                                  engine=['e1','e2'],model_sha=['m1','m2'],keep_latest=True,dry_run=False)
        with mock.patch.object(repack_cache, 'scan', return_value=(entries, [])), contextlib.redirect_stdout(io.StringIO()):
            repack_cache.cmd_prune(args)
        self.assertFalse((root/'obsolete.rpk').exists())
        for name in ['live1','live2','unreadable1','unreadable2']:
            self.assertTrue((root/(name+'.rpk')).exists())

    def cli(self, *args):
        return subprocess.run([sys.executable, '-B', str(TOOLS/'storage_cleanup.py'), *args],
                              env=dict(os.environ, HOME=str(self.home)), text=True, capture_output=True, check=True)

    def test_busy_lease_skips_before_reading_config(self):
        lease = self.home / '.local/state/local-ai/inference.lock'
        lease.parent.mkdir(parents=True)
        with lease.open('w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            result = json.loads(self.cli('--apply').stdout)
        self.assertEqual(result['freed_bytes'], 0)
        self.assertIn('busy', result['skipped'])

    def test_cli_dry_run_and_apply_preserve_active_cache_and_models(self):
        obsolete = self.cache_dir('repack-obsolete')
        active = self.cache_dir('repack-active')
        model = self.work / 'models/keep.gguf'
        model.parent.mkdir()
        model.write_bytes(b'model')
        config = self.home / '.config/local-ai/llm/runtime.json'
        config.parent.mkdir(parents=True)
        config.write_text(json.dumps(dict(model_path=str(model), environment={'GGML_CPU_REPACK_CACHE':str(active)})))
        dry = json.loads(self.cli().stdout)
        self.assertFalse(dry['apply'])
        self.assertTrue(obsolete.exists())
        applied = json.loads(self.cli('--apply').stdout)
        self.assertTrue(applied['apply'])
        self.assertFalse(obsolete.exists())
        self.assertTrue(active.is_dir())
        self.assertEqual(model.read_bytes(), b'model')
        self.assertTrue((self.home/'.local/state/local-ai/storage-cleanup.jsonl').exists())

    def test_changed_tree_is_preserved_at_apply(self):
        obsolete = self.cache_dir('repack-old')
        info = cleanup.tree_info(obsolete)
        action = dict(path=str(obsolete), kind='unreferenced-repack', **info)
        action['inode'] += 1
        config = self.home/'runtime.json'
        config.write_text('{}')
        with mock.patch.object(cleanup, 'plan', return_value=([action], [])), \
             mock.patch.object(cleanup, 'active_paths', return_value=set()), \
             mock.patch.object(cleanup.Path, 'home', return_value=self.home), \
             mock.patch.object(sys, 'argv', ['storage_cleanup', '--apply', '--runtime', str(config)]), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            cleanup.main()
        self.assertTrue(obsolete.exists())
        self.assertEqual(json.loads(output.getvalue())['skipped'][0]['reason'], 'changed-since-plan')


if __name__ == '__main__': unittest.main()
