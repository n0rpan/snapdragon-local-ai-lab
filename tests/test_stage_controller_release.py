"""Controller-only stage: unit re-pointed, runtime.json untouched, guarded rollback. Starts nothing."""
import importlib.util
import json
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
spec = importlib.util.spec_from_file_location('stage_controller_release', ROOT / 'tools/stage_controller_release.py')
tool = importlib.util.module_from_spec(spec); spec.loader.exec_module(tool)
import install_staged_release as installer  # noqa: E402

UNIT = '''[Service]
WorkingDirectory=/synthetic/releases/old
Environment="LOCAL_LLM_CONFIG=/synthetic/runtime.json"
ExecStart=/usr/bin/python3 "/synthetic/releases/old/controller.py"
MemoryMax=28G
MemorySwapMax=0
'''


class Stage(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.unit = self.root / 'local-ai-chat.service'
        self.unit.write_text(UNIT)
        self.stage_dir = self.root / 'stage'

    def run_stage(self, **kw):
        return tool.stage(self.stage_dir, unit_path=self.unit, unit_sha=tool.base.sha(UNIT.encode()),
                          releases_root=self.root / 'releases', **kw)

    def test_stage_contains_the_whole_runtime_and_changes_only_the_unit(self):
        manifest = self.run_stage()
        self.assertEqual(list(manifest['paths']), ['unit'])
        self.assertFalse(manifest['runtime_json_changed'])
        self.assertFalse((self.stage_dir / 'runtime.json').exists())
        staged = (self.stage_dir / 'local-ai-chat.service').read_text()
        self.assertIn('ExecStart=/usr/bin/python3 "%s/controller.py"' % manifest['release_dir'], staged)
        self.assertEqual([l for l in staged.splitlines() if not l.startswith(('WorkingDirectory=', 'ExecStart='))],
                         [l for l in UNIT.splitlines() if not l.startswith(('WorkingDirectory=', 'ExecStart='))])
        self.assertEqual(self.unit.read_text(), UNIT)  # staging never touches the live file
        source = ROOT / 'src/snapdragon_ai_lab/runtime'
        for name in tool.REQUIRED + ('local-ai',):
            self.assertEqual((self.stage_dir / 'runtime-source' / name).read_bytes(), (source / name).read_bytes(), name)
        # every sibling module the standalone controller imports is in the release
        text = (source / 'controller.py').read_text()
        standalone = text[text.index('\nelse:\n'):text.index('\nROOT = ')]
        for line in standalone.splitlines():
            words = line.split()
            if words[:1] in (['import'], ['from']):
                self.assertTrue((self.stage_dir / 'runtime-source' / (words[1] + '.py')).is_file(), words[1])
        # and the staged controller imports as the unit runs it (no package, siblings by name)
        done = subprocess.run([sys.executable, '-c', 'import controller, sys; sys.stdout.write(controller.CAPABILITIES[0])'],
                              cwd=self.stage_dir / 'runtime-source', capture_output=True, text=True,
                              env={'PATH': '/usr/bin:/bin', 'HOME': str(self.root), 'PYTHONDONTWRITEBYTECODE': '1'})
        self.assertEqual((done.returncode, done.stdout), (0, 'models'), done.stderr[-400:])
        self.assertEqual(stat.S_IMODE(self.stage_dir.stat().st_mode), 0o700)
        self.assertEqual(stat.S_IMODE((self.stage_dir / 'manifest.json').stat().st_mode), 0o600)
        self.assertEqual(json.loads((self.stage_dir / 'manifest.json').read_text())['release_id'], manifest['release_id'])

    def test_install_dry_run_then_apply_and_guarded_rollback(self):
        self.run_stage()
        self.assertEqual(installer.run(self.stage_dir, apply=False, out=open('/dev/null', 'w')), 0)
        self.assertEqual(self.unit.read_text(), UNIT)
        self.assertEqual(installer.run(self.stage_dir, apply=True, out=open('/dev/null', 'w')), 0)
        self.assertEqual(self.unit.read_bytes(), (self.stage_dir / 'local-ai-chat.service').read_bytes())
        check = subprocess.run([sys.executable, str(self.stage_dir / 'rollback/rollback.py'), '--check'],
                               capture_output=True, text=True)
        self.assertEqual(check.returncode, 0, check.stderr)
        self.assertIn('would restore unit', check.stdout)
        self.assertEqual(self.unit.read_bytes(), (self.stage_dir / 'local-ai-chat.service').read_bytes())

    def test_refuses_a_changed_unit_an_existing_stage_and_an_incomplete_or_broken_release(self):
        with self.assertRaises(ValueError):
            tool.stage(self.stage_dir, unit_path=self.unit, unit_sha='0' * 64, releases_root=self.root / 'releases')
        self.assertFalse(self.stage_dir.exists())
        bare = self.root / 'bare'; bare.mkdir()
        (bare / 'controller.py').write_text('pass\n')
        with self.assertRaisesRegex(ValueError, 'release_is_missing'):
            self.run_stage(source_dir=bare)
        broken = self.root / 'broken'; broken.mkdir()
        for name in tool.REQUIRED:
            (broken / name).write_text('pass\n')
        (broken / 'openai_api.py').write_text('def broken(:\n')
        with self.assertRaises(Exception):
            self.run_stage(source_dir=broken)
        self.assertFalse(self.stage_dir.exists())
        self.run_stage()
        with self.assertRaises(ValueError):
            self.run_stage()


if __name__ == '__main__':
    unittest.main()
