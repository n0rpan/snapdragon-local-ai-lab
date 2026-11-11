"""The panel's view components load and lay out offscreen against stand-ins for Omarchy's kit.

This catches QML load errors, missing properties and binding warnings. It does not prove the
look under the real kit and theme; that needs the running shell.
"""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
RENDER = ROOT / 'tests/panel_preview/render.sh'
FIXTURES = ROOT / 'tests/fixtures/panel'


@unittest.skipUnless(shutil.which('qml6'), 'qml6 is not installed')
class PanelPreview(unittest.TestCase):
    def test_views_render_without_qml_warnings(self):
        for name in ('status_production', 'status_full', 'preview_ready', 'preview_sleeping_flash'):
            with self.subTest(name), tempfile.TemporaryDirectory() as directory:
                image = Path(directory) / 'out.png'
                result = subprocess.run([str(RENDER), str(FIXTURES / f'{name}.json'), str(image)],
                                        capture_output=True, text=True, timeout=60)
                noise = [line for line in result.stderr.splitlines()
                         if line and not line.startswith(('qml: Qt ', 'qml: Using built-in'))]
                self.assertEqual(noise, [])
                self.assertEqual(result.returncode, 0)
                self.assertGreater(image.stat().st_size, 10000)

    def test_plugin_qml_has_no_rich_text_sink_or_hex_colour(self):
        import re
        for path in sorted((ROOT / 'src/snapdragon_ai_lab/omarchy_plugin').glob('*.qml')):
            text = path.read_text()
            with self.subTest(path.name):
                self.assertEqual(len(re.findall(r'^\s*(?:Ui\.)?Text \{', text, re.M)),
                                 text.count('textFormat: Text.PlainText'))
                self.assertIsNone(re.search(r'"#[0-9a-fA-F]{3,8}"', text))
                self.assertNotIn('/home/', text)
