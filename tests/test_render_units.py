"""S9: the example units are generated from one template, and each unit sets only variables
that its own program (or a module it imports) reads."""
import importlib.util
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / 'src/snapdragon_ai_lab/runtime'
spec = importlib.util.spec_from_file_location('render_units', ROOT / 'tools/render_units.py')
render_units = importlib.util.module_from_spec(spec)
spec.loader.exec_module(render_units)


def env_names(text):
    return re.findall(r'^Environment=([A-Z_][A-Z0-9_]*)=', text, re.M)


class RenderUnitsTests(unittest.TestCase):
    def test_examples_match_the_template(self):
        self.assertEqual(render_units.main(['--check']), 0,
                         'examples/runtime/units drifted: run python3 tools/render_units.py --write')

    def test_every_variable_has_a_reader_in_its_program(self):
        controller_side = ''.join(p.read_text() for p in RUNTIME.glob('*.py'))
        studio = (RUNTIME / 'studio').read_text()
        for name, source in (('local-ai-chat.service', controller_side), ('prompt-studio-app.service', studio)):
            for variable in set(env_names(render_units.render(name))) - {'PYTHONDONTWRITEBYTECODE'}:  # read by python3
                with self.subTest(unit=name, variable=variable):
                    self.assertIn(variable, source, f'{name} sets {variable} but nothing it runs reads it')

    def test_chat_unit_has_no_studio_only_variables(self):
        names = set(env_names(render_units.render('local-ai-chat.service')))
        self.assertFalse(names & {'SD_HOME', 'SD_CLI', 'IMG_OUTDIR', 'LOCAL_IMAGE_CONFIG_DIR'}, names)


if __name__ == '__main__':
    unittest.main()
