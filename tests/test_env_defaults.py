"""Audit N8: empty or relative XDG directories and bad time limits fall back to safe defaults."""
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from snapdragon_ai_lab.runtime import ctl_env


class XdgAndSeconds(unittest.TestCase):
    def test_empty_or_relative_xdg_state_home_is_ignored(self):
        for value in ('', 'relative/state', '.'):
            with self.subTest(value=value), patch.dict(os.environ, {'XDG_STATE_HOME': value}):
                path = ctl_env.xdg_dir('XDG_STATE_HOME', '.local/state')
                self.assertTrue(path.is_absolute())
                self.assertEqual(path, Path.home() / '.local/state')
        with patch.dict(os.environ, {'XDG_STATE_HOME': '/synthetic/state'}):
            self.assertEqual(ctl_env.xdg_dir('XDG_STATE_HOME', '.local/state'), Path('/synthetic/state'))
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop('XDG_CONFIG_HOME', None)
            self.assertEqual(ctl_env.xdg_dir('XDG_CONFIG_HOME', '.config'), Path.home() / '.config')

    def test_seconds_must_be_finite_and_positive(self):
        for value in ('nan', 'inf', '-inf', '-5', '0', 'soon', ''):
            with self.subTest(value=value), patch.dict(os.environ, {'LOCAL_LLM_IDLE_SECONDS': value}):
                self.assertEqual(ctl_env.env_seconds('LOCAL_LLM_IDLE_SECONDS', 600), 600.0)
        with patch.dict(os.environ, {'LOCAL_LLM_IDLE_SECONDS': '90.5'}):
            self.assertEqual(ctl_env.env_seconds('LOCAL_LLM_IDLE_SECONDS', 600), 90.5)

    def test_module_paths_are_absolute(self):
        for path in (ctl_env.CONFIG, ctl_env.LEASE, ctl_env.ENGINE_DIAGNOSTIC_LOG, Path(ctl_env.HARNESS_STATE or '/')):
            self.assertTrue(Path(path).is_absolute(), path)


if __name__ == '__main__':
    unittest.main()
