"""Selected optional asset planning only; no network/lease/native commands."""
import json
from pathlib import Path
import unittest
from snapdragon_ai_lab.bootstrap import DEFAULT_IDS
from snapdragon_ai_lab.bootstrap_download import selected_assets
PROJECT=Path(__file__).resolve().parents[1]
class Selected(unittest.TestCase):
    def test_default_is_only_four_core_assets(self):
        assets=selected_assets(PROJECT);self.assertEqual({x['id'] for x in assets},set(DEFAULT_IDS));self.assertEqual(len(assets),4)
    def test_optional_paths_are_explicit_complete_and_unqualified(self):
        assets=selected_assets(PROJECT,include_dense=True,include_q4=True,include_flash=True)
        flash=[x for x in assets if x.get('selection_id')=='qwen38-flash-next-ud-iq3-xxs']
        self.assertEqual(len(flash),4);self.assertEqual(sum(x['bytes'] for x in flash),83626438426)
        self.assertTrue(all(x['stack_relative_path'].startswith('models/llm/Qwen3.8-Flash-Next/UD-IQ3_XXS/') and x['qualified'] is False for x in flash))
        q4=next(x for x in assets if x['id']=='z-image-turbo-q4');self.assertEqual(q4['stack_relative_path'],'models/images/z_image_turbo-Q4_K.gguf')
        self.assertEqual(len({x['stack_relative_path'] for x in assets}),len(assets))
    def test_retired_provenance_is_preserved_but_not_selectable(self):
        active=json.loads((PROJECT/'dependencies/model-artifacts.json').read_text())
        historical=json.loads((PROJECT/'dependencies/retired-model-artifacts.json').read_text())
        self.assertNotIn('arete27b-q4-k-m-candidate',{x['id'] for x in active['models']})
        retired=next(x for x in historical['models'] if x['id']=='arete27b-q4-k-m-candidate')
        self.assertEqual(retired['status'],'retired');self.assertEqual(retired['license'],'Apache-2.0')
        self.assertEqual(retired['files'][0]['sha256'],'6f186a1fe0c4f741acb3a35b5df96bc2fc7279651f48500246e56079418f091b')
        self.assertNotIn(retired['id'],{x.get('selection_id',x['id']) for x in selected_assets(PROJECT,True,True,True)})
if __name__=='__main__':unittest.main()
