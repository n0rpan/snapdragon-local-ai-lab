"""Runs first under `unittest discover`: isolates HOME/XDG for the whole suite (tests/_isolation.py)."""
import os
import json
import tempfile
from unittest.mock import Mock, patch
import unittest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _isolation  # noqa: E402

_isolation.isolate()

from snapdragon_ai_lab.runtime import controller  # noqa: E402  (after isolation on purpose)


class IsolationTests(unittest.TestCase):
    def test_home_and_xdg_are_private(self):
        root = _isolation.ROOT
        for name in ('HOME', 'XDG_CONFIG_HOME', 'XDG_STATE_HOME'):
            self.assertTrue(Path(os.environ[name]).is_relative_to(root), name)

    def test_controller_never_defaults_to_production_paths(self):
        for path in (controller.CONFIG, controller.LEASE):
            self.assertFalse(any(path.resolve().is_relative_to(real.resolve())
                                 for real in controller.REAL_PRODUCTION_PATHS), path)

    def test_guard_refuses_every_production_path(self):
        home = _isolation.REAL_HOME
        for path in (home / '.config/local-ai/llm/runtime.json', home / '.local/state/local-ai/inference.lock',
                     home / '.local/state/local-ai/other', home / '.config/omarchy/plugins/snapdragon.local-ai/Model.js'):
            with self.subTest(path=path), self.assertRaisesRegex(RuntimeError, 'production path'):
                controller.guard_production_path(path)
        controller.guard_production_path(_isolation.ROOT / 'runtime.json')  # a temp path is fine


    def test_protected_lease_is_rejected_before_creating_its_directory(self):
        manager = controller.Manager()
        manager.stop_event.set(); manager.thread.join(2)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                config = root / 'runtime.json'; config.write_text('{}')
                protected = root / 'protected'; lease = protected / 'new' / 'lease'
                with patch.object(controller, 'REAL_PRODUCTION_PATHS', (protected,)), \
                     patch.object(controller, 'CONFIG', config), patch.object(controller, 'LEASE', lease), \
                     patch.object(controller.storage_view, 'delete') as delete:
                    with self.assertRaisesRegex(RuntimeError, 'production path'):
                        manager.delete_storage('unused')
                    self.assertFalse(protected.exists(), 'rejected test created a production directory')
                    delete.assert_not_called()
        finally:
            manager.close()

    def test_protected_preference_is_rejected_before_stopping_the_model(self):
        manager = controller.Manager()
        manager.stop_event.set(); manager.thread.join(2)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                protected = Path(tmp)
                config = protected / 'runtime.json'
                original = json.dumps({'profiles': {'qwen3.8-flash-next-local': {
                    'memlat_overrides': {'oxe_flash': {'cache_pref': '14'}}}}})
                config.write_text(original)
                manager.loaded_config = {'model_alias': 'qwen3.8-flash-next-local'}
                settings = {'enabled': True, 'editable': True, 'choices': ['14', '18'], 'pref': '14'}
                with patch.object(controller, 'REAL_PRODUCTION_PATHS', (protected,)), \
                     patch.object(controller, 'CONFIG', config), \
                     patch.object(manager, 'flash_cache_setting', return_value=settings), \
                     patch.object(manager, 'cleanup') as cleanup, \
                     patch.object(controller.tempfile, 'NamedTemporaryFile', side_effect=AssertionError('wrote before guard')):
                    with self.assertRaisesRegex(RuntimeError, 'production path'):
                        manager.set_flash_cache_pref('18')
                    cleanup.assert_not_called()
                    self.assertEqual(config.read_text(), original)
        finally:
            manager.close()


if __name__ == '__main__':
    unittest.main()
