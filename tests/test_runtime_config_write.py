"""Audit S6: runtime.json rewrites are private regardless of the umask, fsynced and atomic."""
import json
import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch

from snapdragon_ai_lab.runtime import controller, lan_share


class RuntimeConfigWrite(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.config = self.root / 'runtime.json'
        self.old_umask = os.umask(0o022)
        self.addCleanup(os.umask, self.old_umask)

    def test_a_new_file_is_owner_only_under_a_permissive_umask(self):
        lan_share.write_config(self.config, {'a': 1})
        self.assertEqual(stat.S_IMODE(self.config.stat().st_mode), 0o600)
        self.assertEqual(json.loads(self.config.read_text()), {'a': 1})
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), ['runtime.json'])

    def test_the_rewrite_fsyncs_and_keeps_the_existing_mode(self):
        self.config.write_text('{}'); os.chmod(self.config, 0o600)
        synced = []
        real = os.fsync
        with patch.object(lan_share.os, 'fsync', side_effect=lambda fd: (synced.append(fd), real(fd))):
            lan_share.write_config(self.config, {'b': 2})
        self.assertGreaterEqual(len(synced), 2)  # the file, then its directory
        self.assertEqual(stat.S_IMODE(self.config.stat().st_mode), 0o600)

    def test_a_failed_rename_leaves_the_old_file_and_no_temporary(self):
        self.config.write_text('{"old": true}\n'); os.chmod(self.config, 0o600)
        with patch.object(lan_share.os, 'replace', side_effect=OSError('synthetic')), self.assertRaises(OSError):
            lan_share.write_config(self.config, {'new': True})
        self.assertEqual(json.loads(self.config.read_text()), {'old': True})
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), ['runtime.json'])

    def test_select_writes_through_the_private_writer(self):
        self.config.write_text(json.dumps({'model_alias': 'known', 'selected_model': 'known',
                                           'profiles': {'known': {}, 'other': {}}}))
        os.chmod(self.config, 0o600)
        with patch.object(controller, 'CONFIG', self.config), patch.object(controller.threading.Thread, 'start'), \
                patch.object(controller.Manager, 'eligible', return_value=True), \
                patch.object(lan_share, 'write_config', wraps=lan_share.write_config) as writer:
            manager = controller.Manager()
            self.assertEqual(manager.select('other'), 'other')
        writer.assert_called_once()
        self.assertEqual(json.loads(self.config.read_text())['selected_model'], 'other')
        self.assertEqual(stat.S_IMODE(self.config.stat().st_mode), 0o600)


if __name__ == '__main__':
    unittest.main()
