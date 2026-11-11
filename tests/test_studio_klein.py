"""FLUX.2 klein 4B text-to-image in the Generate flow; the sd-cli engine is always stubbed.

No image model, sd-cli, service or live endpoint is used.
"""
import importlib.machinery
import importlib.util
from http.server import ThreadingHTTPServer
import json
import os
from pathlib import Path
import struct
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from urllib.request import Request, urlopen

SOURCE = Path(__file__).resolve().parents[1] / 'src/snapdragon_ai_lab/runtime/studio'


def option(args, name):
    return args[args.index(name) + 1]


def fake_png(path, width, height):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ihdr = struct.pack('>II', width, height) + b'\x08\x02\x00\x00\x00'
    path.write_bytes(b'\x89PNG\r\n\x1a\n' + struct.pack('>I', 13) + b'IHDR' + ihdr + b'\x00' * 4)


class KleinCase(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        environment = {'IMG_OUTDIR': str(self.root / 'output'), 'LOCAL_IMAGE_CONFIG_DIR': str(self.root / 'config'),
                       'LOCAL_AI_LEASE': str(self.root / 'inference.lock'), 'SD_HOME': str(self.root / 'stack')}
        with patch.dict(os.environ, environment):
            loader = importlib.machinery.SourceFileLoader('klein_studio', str(SOURCE))
            spec = importlib.util.spec_from_loader(loader.name, loader)
            self.studio = importlib.util.module_from_spec(spec)
            loader.exec_module(self.studio)
        s = self.studio
        # Klein Q8 and the shared Qwen3/VAE files are present; klein Q4 is not.
        for path in (s.KLEIN_FILES['q8'], s.KLEIN_LLM, s.KLEIN_VAE, s.MODEL_FILES['q8'], s.LLM, s.VAE):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'stub')
        s.PROJECT.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        self.temporary.cleanup()


class CommandLine(KleinCase):
    def test_klein_t2i_argv_has_no_reference_and_fixed_distilled_settings(self):
        s = self.studio
        plan = s.plan_generation({'prompt': 'a secret lighthouse', 'model': 'klein-q8', 'width': 512,
                                  'height': 768, 'steps': 8, 'seed': 7})
        step = plan['steps'][0]
        args = step['args']
        out = option(args, '-o')
        self.assertEqual(args, [str(s.SDCLI), '--diffusion-model', str(s.KLEIN_FILES['q8']),
                                '--llm', str(s.KLEIN_LLM), '--vae', str(s.KLEIN_VAE),
                                '--cfg-scale', '1.0', '--sampling-method', 'euler', '--steps', '4',
                                '-W', '512', '-H', '768', '-o', out, '--threads', '6',
                                '--flash-attn', 'off,diffusion=off', '--vae-tiling', '--vae-tile-size', '512',
                                '--disable-image-metadata', '--backend', 'cpu', '--seed', '7'])
        self.assertNotIn('-r', args)
        self.assertNotIn('-i', args)
        self.assertNotIn('--mask', args)
        self.assertNotIn('--image-preprocess', args)
        self.assertEqual(step['outputs'], [Path(out)])
        self.assertTrue(step['prompt'])
        self.assertEqual(plan['prompt'], 'a secret lighthouse')
        self.assertNotIn('secret lighthouse', ' '.join(args))

    def test_klein_q8_is_the_default_and_no_seed_means_no_flag(self):
        s = self.studio
        plan = s.plan_generation({'prompt': 'x', 'model': 'klein-q8', 'width': 256, 'height': 256})
        args = plan['steps'][0]['args']
        self.assertNotIn('--seed', args)
        self.assertNotIn('-b', args)
        self.assertEqual(plan['results'][0]['seed'], 42)
        self.assertEqual(option(args, '--diffusion-model'), str(s.KLEIN_FILES['q8']))

    def test_klein_q4_uses_the_q4_diffusion_file(self):
        s = self.studio
        s.KLEIN_FILES['q4'].write_bytes(b'stub')
        plan = s.plan_generation({'prompt': 'x', 'model': 'klein-q4', 'width': 256, 'height': 256})
        self.assertEqual(option(plan['steps'][0]['args'], '--diffusion-model'), str(s.KLEIN_FILES['q4']))
        self.assertEqual(plan['results'][0]['model'], 'klein-q4')

    def test_klein_uses_the_base_binary_not_the_repack_profile(self):
        s = self.studio
        binary = s.STACK / 'releases/sd-cpu-repack/sd-cli'
        binary.parent.mkdir(parents=True)
        binary.write_bytes(b'bin')
        binary.chmod(0o700)
        import hashlib
        profile = {'enabled': True, 'validated': True, 'binary': str(binary), 'models': ['q8'],
                   'sha256': hashlib.sha256(b'bin').hexdigest(), 'dimensions': [[512, 512]],
                   'img2img_validated': True, 'img2img_dimensions': [[512, 512]]}
        (s.PROJECT / 'cpu-profile.json').write_text(json.dumps(profile))
        plan = s.plan_generation({'prompt': 'x', 'model': 'klein-q8', 'width': 512, 'height': 512})
        self.assertEqual(plan['steps'][0]['args'][0], str(s.SDCLI))
        self.assertIsNone(plan['environment'])

    def test_klein_variations_use_batch_seed_and_sequence_output(self):
        s = self.studio
        plan = s.plan_generation({'prompt': 'x', 'model': 'klein-q8', 'width': 256, 'height': 256, 'variations': 3})
        args = plan['steps'][0]['args']
        self.assertEqual(option(args, '-b'), '3')
        self.assertEqual(option(args, '--seed'), '42')
        self.assertTrue(option(args, '-o').endswith('-%d.png'))
        self.assertEqual([Path(p).name[-6:] for p in plan['steps'][0]['outputs']], ['-0.png', '-1.png', '-2.png'])
        self.assertEqual([r['seed'] for r in plan['results']], [42, 43, 44])
        plan = s.plan_generation({'prompt': 'x', 'model': 'klein-q8', 'width': 256, 'height': 256,
                                  'variations': 2, 'seed': 10})
        self.assertEqual(option(plan['steps'][0]['args'], '--seed'), '10')
        self.assertEqual([r['seed'] for r in plan['results']], [10, 11])
        self.assertEqual(option(plan['steps'][0]['args'], '-b'), '2')


class Validation(KleinCase):
    def test_sizes_must_be_multiples_of_16_between_256_and_1024(self):
        s = self.studio
        base = {'prompt': 'x', 'model': 'klein-q8'}
        for width, height in ((500, 512), (512, 240), (1040, 512), (512, 1040), (1000, 512), (512, 1032),
                              ('512', 512), (512.0, 512), (True, 512)):
            with self.assertRaises(s.StudioError, msg=(width, height)):
                s.plan_generation({**base, 'width': width, 'height': height})
        for width, height in ((256, 256), (1024, 1024), (512, 768), (1008, 256)):
            s.plan_generation({**base, 'width': width, 'height': height})

    def test_steps_are_ignored_and_prompt_is_required(self):
        s = self.studio
        plan = s.plan_generation({'prompt': 'x', 'model': 'klein-q8', 'width': 256, 'height': 256, 'steps': 50})
        self.assertEqual(option(plan['steps'][0]['args'], '--steps'), '4')
        with self.assertRaises(s.StudioError):
            s.plan_generation({'prompt': '   ', 'model': 'klein-q8', 'width': 256, 'height': 256})

    def test_missing_files_are_refused_with_the_file_name(self):
        s = self.studio
        with self.assertRaises(s.StudioError) as caught:
            s.plan_generation({'prompt': 'x', 'model': 'klein-q4', 'width': 256, 'height': 256})
        self.assertIn('flux-2-klein-4b-Q4_0.gguf', str(caught.exception))
        s.KLEIN_LLM.unlink()
        with self.assertRaises(s.StudioError) as caught:
            s.plan_generation({'prompt': 'x', 'model': 'klein-q8', 'width': 256, 'height': 256})
        self.assertIn('Qwen3-4B-Q8_0.gguf', str(caught.exception))

    def test_photo_and_gpu_are_refused_for_klein_text_to_image(self):
        s = self.studio
        photo = s.UPLOADS / 'p.png'
        fake_png(photo, 256, 256)
        with self.assertRaises(s.StudioError):
            s.plan_generation({'prompt': 'x', 'model': 'klein-q8', 'width': 256, 'height': 256, 'init': str(photo)})
        with self.assertRaises(s.StudioError):
            s.plan_generation({'prompt': 'x', 'model': 'klein-q8', 'width': 256, 'height': 256, 'cpu': False})

    def test_variations_and_seed_are_checked(self):
        s = self.studio
        base = {'prompt': 'x', 'model': 'klein-q8', 'width': 256, 'height': 256}
        for bad in ({'variations': 0}, {'variations': 5}, {'seed': -1}, {'seed': 1.5}):
            with self.assertRaises(s.StudioError, msg=bad):
                s.plan_generation({**base, **bad})


class Estimates(KleinCase):
    def test_t2i_halves_sampling_and_drops_vae_encode(self):
        s = self.studio
        e = s.edit_estimate_settings()
        mpx = 512 * 512 / 1e6
        expected = e['fixed_s'] + 0.5 * e['s_per_mpx_step'] * mpx ** e['pixel_exponent'] * 4
        self.assertAlmostEqual(s.edit_estimate_seconds(512, 512, t2i=True), expected, delta=0.1)
        self.assertLess(s.edit_estimate_seconds(512, 512, t2i=True), s.edit_estimate_seconds(512, 512))
        self.assertAlmostEqual(s.edit_estimate_seconds(512, 512, t2i=True, count=2),
                               e['fixed_s'] + 2 * 0.5 * e['s_per_mpx_step'] * mpx ** e['pixel_exponent'] * 4, delta=0.1)



class Draft512Preset(KleinCase):
    def test_default_and_example_preset(self):
        s = self.studio
        self.assertEqual(s.image_presets()['draft512'], {'model': 'klein-q8', 'width': 512, 'height': 512, 'steps': 4})
        # The page still opens on the Draft preset.
        self.assertEqual(s.image_presets()['draft']['width'], 256)
        example = SOURCE.parents[3] / 'examples/runtime/images/presets.json'
        (s.PROJECT / 'presets.json').write_text(example.read_text())
        presets = s.image_presets()
        self.assertEqual(presets['draft512'], {'model': 'klein-q8', 'width': 512, 'height': 512, 'steps': 4})
        self.assertEqual(presets['draft'], {'model': 'q8', 'width': 256, 'height': 256, 'steps': 8})

    def test_invalid_draft512_falls_back_to_default(self):
        s = self.studio
        for bad in ({'model': 'q8', 'width': 512, 'height': 512, 'steps': 4},
                    {'model': 'klein-q8', 'width': 520, 'height': 512, 'steps': 4},
                    {'model': 'klein-q8', 'width': 512, 'height': 512, 'steps': 8},
                    {'model': 'klein-q8', 'width': 2048, 'height': 512, 'steps': 4}):
            (s.PROJECT / 'presets.json').write_text(json.dumps({'draft512': bad}))
            self.assertEqual(s.image_presets()['draft512']['width'], 512, bad)
            self.assertEqual(s.image_presets()['draft512']['steps'], 4, bad)
            self.assertEqual(s.image_presets()['draft512']['model'], 'klein-q8', bad)
        (s.PROJECT / 'presets.json').write_text(json.dumps({'draft512': {'model': 'klein-q4', 'width': 768,
                                                                        'height': 768, 'steps': 4}}))
        self.assertEqual(s.image_presets()['draft512'], {'model': 'klein-q4', 'width': 768, 'height': 768, 'steps': 4})


class Endpoints(KleinCase):
    def setUp(self):
        super().setUp()
        self.calls = []
        fixture = self

        def engine(args, job, environment=None, prompt=None):
            fixture.calls.append({'args': list(args), 'prompt': prompt})
            out = option(args, '-o')
            if '-M' in args:
                w, h = fixture.studio.png_size(option(args, '-i'))
                fake_png(out, w * 4, h * 4)
                return 0
            w, h = int(option(args, '-W')), int(option(args, '-H'))
            count = int(option(args, '-b')) if '-b' in args else 1
            for i in range(count):
                fake_png(out.replace('%d', str(i)), w, h)
            job['state'] = 'generating 4/4'
            return 0

        self.engine = patch.object(self.studio, 'run_engine', side_effect=engine)
        self.engine.start()
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), self.studio.H)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(2)
        self.engine.stop()
        super().tearDown()

    def request(self, path, body=None):
        url = f'http://127.0.0.1:{self.server.server_port}{path}'
        data = None if body is None else json.dumps(body).encode()
        try:
            with urlopen(Request(url, data=data), timeout=10) as response:
                return json.loads(response.read())
        except Exception as error:
            if hasattr(error, 'read'):
                with error:
                    return json.loads(error.read())
            raise

    def finish(self, path, body):
        started = self.request(path, body)
        self.assertIn('id', started, started)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            job = self.request('/job?id=' + started['id'])
            if job['state'] in ('done', 'error'):
                return job
            time.sleep(0.02)
        self.fail('job did not finish')

    def test_models_reports_klein_readiness_and_uncalibrated_estimates(self):
        models = self.request('/models')
        klein = models['klein']
        self.assertEqual(klein['models'], {'klein-q8': True, 'klein-q4': False})
        self.assertTrue(klein['ready'])
        self.assertFalse(klein['estimates_calibrated'])
        self.assertEqual(klein['estimate_label'], 'uncalibrated estimate')
        self.assertEqual(set(klein['estimates']), {'512', '768', '1024'})
        self.assertLess(klein['estimates']['512'], klein['estimates']['768'])
        self.assertLess(klein['estimates']['512'], models['edit']['estimates']['512'])
        self.assertEqual(klein['steps'], 4)

    def test_estimate_endpoint_accepts_klein(self):
        result = self.request('/estimate?model=klein-q8&width=512&height=512&steps=8&variations=1')
        self.assertEqual(result['seconds'], self.studio.edit_estimate_seconds(512, 512, t2i=True))
        self.assertFalse(result['calibrated'])
        self.assertIn('error', self.request('/estimate?model=klein-q8&width=500&height=512'))

    def test_generate_with_klein_registers_the_image_and_keeps_prompt_out_of_job(self):
        job = self.finish('/generate', {'prompt': 'secret klein prompt', 'model': 'klein-q8',
                                        'width': 512, 'height': 512, 'steps': 8, 'cpu': True})
        self.assertEqual(job['state'], 'done', job)
        self.assertNotIn('secret klein prompt', json.dumps(job))
        self.assertEqual(self.calls[0]['prompt'], 'secret klein prompt')
        image = job['images'][0]
        self.assertEqual((image['width'], image['height'], image['kind']), (512, 512, 'image'))
        self.assertEqual(self.studio.png_size(image['path']), (512, 512))
        self.assertIsNone(image['actions']['upscale'])            # no ESRGAN model installed here
        self.assertEqual(sorted(image['actions']['refine']), ['1024', '768'])

    def test_refine_of_a_klein_image_uses_z_image_at_eight_steps(self):
        job = self.finish('/generate', {'prompt': 'p', 'model': 'klein-q8', 'width': 256, 'height': 256})
        image = job['images'][0]
        refined = self.finish('/refine', {'image': image['id'], 'target': 768, 'strength': 0.45})
        self.assertEqual(refined['state'], 'done', refined)
        args = self.calls[-1]['args']
        self.assertEqual(option(args, '--diffusion-model'), str(self.studio.MODEL_FILES['q8']))
        self.assertEqual(option(args, '--steps'), '8')
        self.assertEqual(option(args, '-i'), image['path'])
        self.assertEqual(self.calls[-1]['prompt'], 'p')

    def test_quick_upscale_works_on_a_klein_image(self):
        s = self.studio
        esrgan = s.MODELS / 'upscale/RealESRGAN_x4plus.pth'
        esrgan.parent.mkdir(parents=True, exist_ok=True)
        esrgan.write_bytes(b'stub')
        job = self.finish('/generate', {'prompt': 'p', 'model': 'klein-q8', 'width': 256, 'height': 256})
        image = job['images'][0]
        self.assertIsNotNone(image['actions']['upscale'])
        up = self.finish('/upscale', {'image': image['id']})
        self.assertEqual(up['state'], 'done', up)
        self.assertEqual((up['images'][0]['width'], up['images'][0]['height']), (1024, 1024))
        self.assertIsNone(self.calls[-1]['prompt'])


if __name__ == '__main__':
    unittest.main()
