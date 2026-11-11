"""Studio draft/upscale/refine/edit tests; the sd-cli engine is always stubbed.

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


def fake_png(path, width, height):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ihdr = struct.pack('>II', width, height) + b'\x08\x02\x00\x00\x00'
    path.write_bytes(b'\x89PNG\r\n\x1a\n' + struct.pack('>I', 13) + b'IHDR' + ihdr + b'\x00' * 4)


def option(args, name):
    return args[args.index(name) + 1]


class StudioCase(unittest.TestCase):
    def setUp(self):
        # A parent directory containing x2 must not change the stub model scale.
        self.temporary = tempfile.TemporaryDirectory(prefix='studio-x2-fixture-')
        self.root = Path(self.temporary.name)
        environment = {'IMG_OUTDIR': str(self.root / 'output'), 'LOCAL_IMAGE_CONFIG_DIR': str(self.root / 'config'),
                       'LOCAL_AI_LEASE': str(self.root / 'inference.lock'), 'SD_HOME': str(self.root / 'stack')}
        with patch.dict(os.environ, environment):
            loader = importlib.machinery.SourceFileLoader('upscale_studio', str(SOURCE))
            spec = importlib.util.spec_from_loader(loader.name, loader)
            self.studio = importlib.util.module_from_spec(spec)
            loader.exec_module(self.studio)
        self.models = self.studio.MODELS
        for path in (self.studio.MODEL_FILES['q8'], self.studio.LLM, self.studio.VAE):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'stub')
        self.studio.PROJECT.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        self.temporary.cleanup()

    def install_esrgan(self, name='upscale/RealESRGAN_x4plus.pth'):
        path = self.models / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'stub')
        return path

    def write_config(self, value):
        (self.studio.PROJECT / 'upscale.json').write_text(json.dumps(value))


class CommandLines(StudioCase):
    def test_single_draft_command_matches_original_invocation(self):
        s = self.studio
        args = s.build_generation_args('sd-cli', s.MODEL_FILES['q8'], 256, 256, 8, '/o.png', 10, 'on')
        self.assertEqual(args, ['sd-cli', '--diffusion-model', str(s.MODEL_FILES['q8']), '--llm', str(s.LLM),
                                '--vae', str(s.VAE), '--cfg-scale', '1.0', '--steps', '8', '-W', '256', '-H', '256',
                                '-o', '/o.png', '--threads', '10', '--flash-attn', 'off,diffusion=on',
                                '--vae-tiling', '--vae-tile-size', '512', '--disable-image-metadata',
                                '--backend', 'cpu'])

    def test_variations_use_batch_explicit_seed_and_sequence_output(self):
        s = self.studio
        plan = s.plan_generation({'prompt': 'a red bicycle', 'model': 'q8', 'width': 256, 'height': 256,
                                  'steps': 8, 'variations': 3})
        args = plan['steps'][0]['args']
        self.assertEqual(option(args, '-b'), '3')
        self.assertEqual(option(args, '--seed'), '42')
        self.assertTrue(option(args, '-o').endswith('-%d.png'))
        self.assertEqual([Path(p).name[-6:] for p in plan['steps'][0]['outputs']], ['-0.png', '-1.png', '-2.png'])
        self.assertEqual([r['seed'] for r in plan['results']], [42, 43, 44])
        self.assertNotIn('a red bicycle', ' '.join(args))

    def test_single_draft_without_seed_keeps_sd_cli_default(self):
        plan = self.studio.plan_generation({'prompt': 'x', 'model': 'q8', 'width': 256, 'height': 256})
        args = plan['steps'][0]['args']
        self.assertNotIn('--seed', args)
        self.assertNotIn('-b', args)
        self.assertEqual(plan['results'][0]['seed'], 42)

    def test_upscale_mode_command(self):
        args = self.studio.build_upscale_args('sd-cli', '/m/x4.pth', '/in.png', '/out.png', 10, 128)
        self.assertEqual(args, ['sd-cli', '-M', 'upscale', '-i', '/in.png', '--upscale-model', '/m/x4.pth',
                                '--upscale-tile-size', '128', '--threads', '10', '--backend', 'cpu',
                                '-o', '/out.png', '--disable-image-metadata'])

    def test_refine_command_with_and_without_esrgan(self):
        s = self.studio
        draft = s.OUT / 'd.png'
        fake_png(draft, 256, 256)
        image = s.register_image({'path': str(draft), 'prompt': 'p', 'seed': 7, 'model': 'q8',
                                  'width': 256, 'height': 256, 'steps': 8, 'kind': 'image'})
        plan = s.plan_refine({'image': image})
        self.assertEqual(len(plan['steps']), 1)
        args = plan['steps'][0]['args']
        self.assertEqual(option(args, '-i'), str(draft))
        self.assertEqual((option(args, '-W'), option(args, '-H')), ('1024', '1024'))
        self.assertEqual(option(args, '--strength'), '0.45')
        self.assertEqual(option(args, '--seed'), '7')
        self.assertEqual(option(args, '--image-preprocess'), 'target=init,mode=stretch,filter=lanczos')
        self.assertEqual(plan['prompt'], 'p')
        model = self.install_esrgan()
        plan = s.plan_refine({'image': image, 'target': 768, 'strength': 0.3})
        up, refine = plan['steps']
        self.assertEqual(option(up['args'], '-M'), 'upscale')
        self.assertEqual(option(up['args'], '--upscale-model'), str(model.resolve()))
        self.assertEqual(up['expect_size'], (1024, 1024))
        self.assertFalse(up['prompt'])
        self.assertEqual(option(refine['args'], '-i'), option(up['args'], '-o'))
        self.assertEqual((option(refine['args'], '-W'), option(refine['args'], '--strength')), ('768', '0.3'))
        self.assertEqual(plan['scratch'], [option(up['args'], '-o')])

    def test_refine_post_upscale_requires_x2_model(self):
        s = self.studio
        fake_png(s.OUT / 'd.png', 256, 256)
        image = s.register_image({'path': str(s.OUT / 'd.png'), 'prompt': 'p', 'seed': 1, 'model': 'q8',
                                  'width': 256, 'height': 256, 'steps': 8, 'kind': 'image'})
        with self.assertRaises(s.StudioError):
            s.plan_refine({'image': image, 'post_upscale': True})
        self.install_esrgan('upscale/x2.pth')
        self.write_config({'esrgan_x2_model': 'upscale/x2.pth'})
        plan = s.plan_refine({'image': image, 'post_upscale': True})
        self.assertEqual(plan['steps'][-1]['expect_size'], (2048, 2048))
        self.assertEqual(plan['results'][0]['width'], 2048)

    def test_edit_mask_is_passed_only_with_photo(self):
        s = self.studio
        photo, mask = s.UPLOADS / 'photo.png', s.UPLOADS / 'mask.png'
        fake_png(photo, 512, 512)
        fake_png(mask, 512, 512)
        plan = s.plan_generation({'prompt': 'x', 'model': 'q8', 'width': 512, 'height': 512,
                                  'init': str(photo), 'mask': str(mask)})
        args = plan['steps'][0]['args']
        self.assertEqual(option(args, '--mask'), str(mask))
        self.assertEqual(option(args, '--strength'), '0.3')
        self.assertNotIn('--image-preprocess', args)
        plan = s.plan_generation({'prompt': 'x', 'model': 'q8', 'width': 512, 'height': 512, 'mask': str(mask)})
        self.assertNotIn('--mask', plan['steps'][0]['args'])
        with self.assertRaises(s.StudioError):
            s.plan_generation({'prompt': 'x', 'model': 'q8', 'width': 512, 'height': 512,
                               'init': str(photo), 'mask': str(self.root / 'elsewhere.png')})


class Validation(StudioCase):
    def test_edit_preset_strength_mapping(self):
        f = self.studio.edit_strength
        self.assertEqual(f({}), 0.3)
        self.assertEqual(f({'edit_preset': 'light'}), 0.3)
        self.assertEqual(f({'edit_preset': 'restyle'}), 0.55)
        self.assertEqual(f({'edit_preset': 'reimagine'}), 0.8)
        self.assertEqual(f({'edit_preset': 'restyle', 'strength': 0.9}), 0.55)
        self.assertEqual(f({'edit_preset': 'custom', 'strength': 0.65}), 0.65)
        self.assertEqual(f({'strength': 0.65}), 0.65)
        for bad in ({'edit_preset': 'redraw'}, {'edit_preset': 'custom', 'strength': 1.2},
                    {'strength': 0}, {'strength': True}, {'strength': 'lots'}):
            with self.assertRaises(ValueError):
                f(bad)

    def test_generation_validation(self):
        s = self.studio
        base = {'prompt': 'x', 'model': 'q8', 'width': 256, 'height': 256}
        for bad in ({'variations': 0}, {'variations': 5}, {'variations': '2'}, {'seed': -1}, {'seed': 1.5},
                    {'width': 300}, {'steps': 0}, {'model': 'q2'}):
            with self.assertRaises(s.StudioError):
                s.plan_generation({**base, **bad})
        with self.assertRaises(s.StudioError) as caught:
            s.plan_generation({**base, 'prompt': ' '})
        self.assertEqual(caught.exception.code, 200)

    def test_refine_validation(self):
        s = self.studio
        fake_png(s.OUT / 'd.png', 256, 256)
        image = s.register_image({'path': str(s.OUT / 'd.png'), 'prompt': 'p', 'seed': 1, 'model': 'q8',
                                  'width': 256, 'height': 256, 'steps': 8, 'kind': 'image'})
        for bad in ({'image': 'nope'}, {'image': image, 'target': 900}, {'image': image, 'target': 512},
                    {'image': image, 'strength': 0}, {'image': image, 'strength': 1.0},
                    {'image': image, 'strength': '0.4'}):
            with self.assertRaises(s.StudioError):
                s.plan_refine(bad)

    def test_sampled_steps_follow_pinned_img2img_schedule(self):
        f = self.studio.sampled_steps
        self.assertEqual([f(8, x) for x in (0.3, 0.45, 0.55, 0.8, 0.99)], [3, 4, 5, 7, 8])
        self.assertEqual(f(8), 8)

    def test_refine_size_keeps_aspect_on_grid(self):
        f = self.studio.refine_size
        self.assertEqual(f(256, 256, 1024), (1024, 1024))
        self.assertEqual(f(384, 256, 768), (768, 512))
        with self.assertRaises(ValueError):
            f(256, 256, 1536)

    def test_upscale_settings_degrade_and_confine_models(self):
        s = self.studio
        self.assertIsNone(s.upscale_settings()['esrgan_path'])
        outside = self.root / 'evil.pth'
        outside.write_bytes(b'x')
        self.write_config({'esrgan_model': str(outside), 'refine_strength': 3, 'refine_targets': [100],
                           'estimate': {'q8': {'fixed_s': -1}, 'esrgan_fixed_s': 'x'}})
        settings = s.upscale_settings()
        self.assertIsNone(settings['esrgan_path'])
        self.assertEqual(settings['refine_strength'], 0.45)
        self.assertEqual(settings['refine_targets'], [768, 1024])
        self.assertEqual(settings['estimate']['q8']['fixed_s'], 12.0)
        self.write_config({'esrgan_model': 'upscale/m.txt'})
        (self.models / 'upscale').mkdir(parents=True)
        (self.models / 'upscale/m.txt').write_text('x')
        self.assertIsNone(s.upscale_settings()['esrgan_path'])
        model = self.install_esrgan()
        self.write_config({'refine_target': 768, 'refine_targets': [1024]})
        settings = s.upscale_settings()
        self.assertEqual(settings['esrgan_path'], model.resolve())
        self.assertEqual(settings['refine_targets'], [768, 1024])

    def test_estimates_reproduce_measured_anchors_and_flag_uncalibrated(self):
        s = self.studio
        settings = s.upscale_settings()
        self.assertFalse(settings['estimate']['calibrated'])
        self.assertAlmostEqual(s.estimate_seconds(settings, 'txt2img', 'q8', 256, 256), 20.07, delta=2)
        self.assertAlmostEqual(s.estimate_seconds(settings, 'txt2img', 'q8', 1024, 1024), 526.61, delta=15)
        self.assertAlmostEqual(s.estimate_seconds(settings, 'txt2img', 'q4', 1024, 1024), 615.04, delta=20)
        one = s.estimate_seconds(settings, 'txt2img', 'q8', 256, 256)
        # One engine load is shared by the batch; sampling repeats per variation.
        self.assertAlmostEqual(s.estimate_seconds(settings, 'txt2img', 'q8', 256, 256, count=4), 12 + 4 * (one - 12), delta=0.3)
        refine = s.estimate_seconds(settings, 'img2img', 'q8', 1024, 1024, strength=0.45)
        self.assertLess(refine, s.estimate_seconds(settings, 'txt2img', 'q8', 1024, 1024))
        self.assertGreater(s.estimate_seconds(settings, 'img2img', 'q8', 1024, 1024, strength=0.45, optimized=False), refine)
        self.assertAlmostEqual(s.estimate_seconds(settings, 'esrgan', input_size=(256, 256), scale=4), 3.0 + 20.0 * 1.048576, delta=0.1)

    def test_img2img_repack_profile_is_opt_in(self):
        s = self.studio
        binary = s.STACK / 'releases/sd-cpu-repack/sd-cli'
        binary.parent.mkdir(parents=True)
        binary.write_bytes(b'bin')
        binary.chmod(0o700)
        import hashlib
        profile = {'enabled': True, 'validated': True, 'binary': str(binary), 'models': ['q8'],
                   'sha256': hashlib.sha256(b'bin').hexdigest(), 'dimensions': [[256, 256], [1024, 1024]]}
        (s.PROJECT / 'cpu-profile.json').write_text(json.dumps(profile))
        self.assertIsNotNone(s.optimized_cpu_profile('q8', 1024, 1024, 8, False))
        self.assertIsNone(s.optimized_cpu_profile('q8', 1024, 1024, 8, True))
        profile.update(img2img_validated=True, img2img_dimensions=[[1024, 1024]])
        (s.PROJECT / 'cpu-profile.json').write_text(json.dumps(profile))
        self.assertIsNotNone(s.optimized_cpu_profile('q8', 1024, 1024, 8, True))
        self.assertIsNone(s.optimized_cpu_profile('q8', 768, 768, 8, True))


class StubbedFlow(StudioCase):
    """Draft -> choose -> quick upscale / refine through the real HTTP handler."""

    def setUp(self):
        super().setUp()
        self.calls = []
        self.esrgan_broken = False
        fixture = self

        def engine(args, job, environment=None, prompt=None):
            fixture.calls.append({'args': list(args), 'prompt': prompt, 'environment': environment})
            out = option(args, '-o')
            if '-M' in args:
                source = option(args, '-i')
                w, h = fixture.studio.png_size(source)
                scale = 1 if fixture.esrgan_broken else (2 if 'x2' in Path(option(args, '--upscale-model')).name else 4)
                fake_png(out, w * scale, h * scale)
                return 0
            w, h = int(option(args, '-W')), int(option(args, '-H'))
            count = int(option(args, '-b')) if '-b' in args else 1
            for i in range(count):
                fake_png(out.replace('%d', str(i)), w, h)
            job['state'] = 'generating 8/8'
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
        except Exception as error:  # HTTPError carries a JSON body too
            if hasattr(error, 'read'):
                with error:
                    return json.loads(error.read())
            raise

    def finish(self, body, path='/generate'):
        started = self.request(path, body)
        self.assertIn('id', started, started)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            job = self.request('/job?id=' + started['id'])
            if job['state'] in ('done', 'error'):
                return job
            time.sleep(0.02)
        self.fail('job did not finish')

    def draft(self, variations=2):
        job = self.finish({'prompt': 'secret lighthouse prompt', 'model': 'q8', 'width': 256, 'height': 256,
                           'steps': 8, 'variations': variations})
        self.assertEqual(job['state'], 'done', job)
        self.assertNotIn('secret lighthouse', json.dumps(job))
        return job

    def test_draft_choose_quick_upscale(self):
        self.install_esrgan()
        job = self.draft()
        self.assertEqual([im['seed'] for im in job['images']], [42, 43])
        chosen = job['images'][1]
        self.assertIsNotNone(chosen['actions']['upscale'])
        self.assertEqual(sorted(chosen['actions']['refine']), ['1024', '768'])
        self.assertEqual(self.calls[0]['prompt'], 'secret lighthouse prompt')
        up = self.finish({'image': chosen['id']}, '/upscale')
        self.assertEqual(up['state'], 'done', up)
        self.assertEqual((up['images'][0]['width'], up['images'][0]['height']), (1024, 1024))
        self.assertEqual(self.studio.png_size(up['path']), (1024, 1024))
        call = self.calls[-1]
        self.assertIsNone(call['prompt'])
        self.assertEqual(option(call['args'], '-i'), chosen['path'])

    def test_draft_choose_refine_with_esrgan(self):
        self.install_esrgan()
        job = self.draft()
        chosen = job['images'][1]
        refined = self.finish({'image': chosen['id'], 'target': 1024, 'strength': 0.45}, '/refine')
        self.assertEqual(refined['state'], 'done', refined)
        esrgan, img2img = self.calls[-2:]
        self.assertEqual(option(esrgan['args'], '-i'), chosen['path'])
        self.assertIsNone(esrgan['prompt'])
        self.assertEqual(option(img2img['args'], '-i'), option(esrgan['args'], '-o'))
        self.assertEqual(option(img2img['args'], '--seed'), '43')
        self.assertEqual(img2img['prompt'], 'secret lighthouse prompt')
        self.assertEqual(self.studio.png_size(refined['path']), (1024, 1024))
        self.assertFalse(Path(option(esrgan['args'], '-o')).exists(), 'intermediate ESRGAN file is removed')
        self.assertEqual(refined['images'][0]['seed'], 43)

    def test_missing_esrgan_hides_quick_upscale_and_refine_resizes(self):
        models = self.request('/models')
        self.assertFalse(models['upscale']['esrgan'])
        self.assertEqual(models['edit_presets']['light'], 0.3)
        job = self.draft(variations=1)
        chosen = job['images'][0]
        self.assertIsNone(chosen['actions']['upscale'])
        refused = self.request('/upscale', {'image': chosen['id']})
        self.assertIn('ESRGAN', refused['error'])
        refined = self.finish({'image': chosen['id'], 'target': 768}, '/refine')
        self.assertEqual(refined['state'], 'done', refined)
        self.assertNotIn('-M', self.calls[-1]['args'])
        self.assertEqual(option(self.calls[-1]['args'], '-i'), chosen['path'])
        self.assertEqual(self.studio.png_size(refined['path']), (768, 768))

    def test_esrgan_that_silently_fails_is_reported(self):
        self.install_esrgan()
        chosen = self.draft(variations=1)['images'][0]
        self.esrgan_broken = True
        up = self.finish({'image': chosen['id']}, '/upscale')
        self.assertEqual(up['state'], 'error')
        self.assertIn('expected 1024×1024', up['error'])

    def test_estimate_endpoint(self):
        result = self.request('/estimate?model=q8&width=256&height=256&steps=8&variations=2')
        self.assertFalse(result['calibrated'])
        self.assertGreater(result['seconds'], 20)
        self.assertIn('error', self.request('/estimate?model=q8&width=99999'))


if __name__ == '__main__':
    unittest.main()
