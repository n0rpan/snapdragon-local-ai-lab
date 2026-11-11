"""Studio Edit (instruction) mode tests; the sd-cli engine is always stubbed.

No image model, sd-cli, service or live endpoint is used.
"""
import importlib.machinery
import importlib.util
from http.server import ThreadingHTTPServer
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from urllib.request import Request, urlopen

SOURCE = Path(__file__).resolve().parents[1] / 'src/snapdragon_ai_lab/runtime/studio'


def option(args, name):
    return args[args.index(name) + 1]


class EditCase(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        environment = {'IMG_OUTDIR': str(self.root / 'output'), 'LOCAL_IMAGE_CONFIG_DIR': str(self.root / 'config'),
                       'LOCAL_AI_LEASE': str(self.root / 'inference.lock'), 'SD_HOME': str(self.root / 'stack')}
        with patch.dict(os.environ, environment):
            loader = importlib.machinery.SourceFileLoader('edit_studio', str(SOURCE))
            spec = importlib.util.spec_from_loader(loader.name, loader)
            self.studio = importlib.util.module_from_spec(spec)
            loader.exec_module(self.studio)
        s = self.studio
        for path in (s.KLEIN_FILES['q8'], s.KLEIN_LLM, s.KLEIN_VAE):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'stub')
        s.PROJECT.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        self.temporary.cleanup()

    def photo(self, name, width, height, colour=(10, 20, 30)):
        path = self.studio.UPLOADS / name
        self.studio.write_png(path, width, height, bytes(colour) * (width * height))
        return path

    def gray_mask(self, name, width, height, white_box):
        """Mask PNG (stored RGB): white inside white_box=(x0,y0,x1,y1), black elsewhere."""
        x0, y0, x1, y1 = white_box
        pixels = bytearray()
        for y in range(height):
            for x in range(width):
                pixels += b'\xff\xff\xff' if x0 <= x < x1 and y0 <= y < y1 else b'\x00\x00\x00'
        path = self.studio.UPLOADS / name
        self.studio.write_png(path, width, height, pixels)
        return path


class CommandLine(EditCase):
    def test_klein_command_is_the_documented_edit_invocation(self):
        s = self.studio
        args = s.build_edit_args('sd-cli', s.KLEIN_FILES['q8'], '/in.png', '/o.png', 512, 512, 4, 10, seed=5)
        self.assertEqual(args, ['sd-cli', '--diffusion-model', str(s.KLEIN_FILES['q8']), '--llm', str(s.KLEIN_LLM),
                                '--vae', str(s.KLEIN_VAE), '-r', '/in.png', '--cfg-scale', '1.0',
                                '--sampling-method', 'euler', '--steps', '4', '-W', '512', '-H', '512',
                                '-o', '/o.png', '--threads', '10', '--flash-attn', 'off,diffusion=off',
                                '--vae-tiling', '--vae-tile-size', '512', '--disable-image-metadata',
                                '--backend', 'cpu', '--seed', '5'])

    def test_plan_without_mask_passes_the_upload_and_keeps_instruction_off_argv(self):
        s = self.studio
        photo = self.photo('p.png', 1000, 750)
        plan = s.plan_edit({'instruction': 'change the jacket colour to green', 'init': str(photo)})
        step = plan['steps'][0]
        args = step['args']
        self.assertEqual(option(args, '-r'), str(photo))
        self.assertEqual((option(args, '-W'), option(args, '-H')), ('512', '384'))
        self.assertEqual((option(args, '--steps'), option(args, '--cfg-scale')), ('4', '1.0'))
        self.assertEqual(option(args, '--seed'), '42')
        self.assertNotIn('--mask', args)
        self.assertNotIn('jacket', ' '.join(args))
        self.assertEqual(plan['prompt'], 'change the jacket colour to green')
        self.assertEqual(step['expect_size'], (512, 384))
        self.assertNotIn('pre', step)
        self.assertEqual(plan['results'][0]['engine'], 'klein')

    def test_768_and_jpeg_dimensions(self):
        s = self.studio
        # Smallest valid JPEG header with an SOF0 of 640x480.
        jpeg = s.UPLOADS / 'p.jpg'
        jpeg.write_bytes(b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00'
                         b'\xff\xc0\x00\x11\x08\x01\xe0\x02\x80\x03\x01\x22\x00\x02\x11\x01\x03\x11\x01')
        self.assertEqual(s.image_size(jpeg), (640, 480))
        plan = s.plan_edit({'instruction': 'x', 'init': str(jpeg), 'size': 768})
        self.assertEqual((option(plan['steps'][0]['args'], '-W'), option(plan['steps'][0]['args'], '-H')), ('768', '576'))

    def test_work_size_is_on_the_16_grid(self):
        s = self.studio
        self.assertEqual(s.edit_work_size(1000, 750, 512), (512, 384))
        self.assertEqual(s.edit_work_size(333, 500, 768), (512, 768))
        for w, h in ((1234, 987), (100, 3000), (64, 64)):
            ww, hh = s.edit_work_size(w, h, 512)
            self.assertEqual((ww % 16, hh % 16), (0, 0))


class Validation(EditCase):
    def test_rejections(self):
        s = self.studio
        photo = self.photo('p.png', 256, 256)
        good = {'instruction': 'make it night', 'init': str(photo)}
        s.plan_edit(good)
        bad = [dict(good, instruction='  '), dict(good, instruction='x' * 1001), dict(good, size=1024),
               dict(good, size='512'), dict(good, model='q2'), dict(good, steps=0), dict(good, steps=9),
               dict(good, seed=-1), dict(good, seed=1.5), dict(good, init=str(self.root / 'elsewhere.png')),
               {'instruction': 'x'}, dict(good, mask=str(self.root / 'elsewhere.png'))]
        for body in bad:
            with self.assertRaises(s.StudioError, msg=body):
                s.plan_edit(body)

    def test_unknown_image_format_and_tiny_images(self):
        s = self.studio
        junk = s.UPLOADS / 'junk.png'
        junk.write_bytes(b'not an image')
        with self.assertRaises(s.StudioError):
            s.plan_edit({'instruction': 'x', 'init': str(junk)})
        with self.assertRaises(s.StudioError):
            s.plan_edit({'instruction': 'x', 'init': str(self.photo('tiny.png', 32, 32))})

    def test_missing_model_files_name_what_is_missing(self):
        s = self.studio
        s.KLEIN_VAE.unlink()
        with self.assertRaises(s.StudioError) as caught:
            s.plan_edit({'instruction': 'x', 'init': str(self.photo('p.png', 256, 256))})
        self.assertIn('flux2-vae.safetensors', str(caught.exception))
        with self.assertRaises(s.StudioError):
            s.plan_edit({'instruction': 'x', 'model': 'q4', 'init': str(self.photo('p.png', 256, 256))})

    def test_mask_needs_png_and_size_cap(self):
        s = self.studio
        photo = self.photo('p.png', 256, 256)
        jpeg = s.UPLOADS / 'm.jpg'
        jpeg.write_bytes(b'\xff\xd8\xff\xd9')
        with self.assertRaises(s.StudioError):
            s.plan_edit({'instruction': 'x', 'init': str(photo), 'mask': str(jpeg)})
        with patch.object(s, 'MAX_MASK_PIXELS', 1000):
            with self.assertRaises(s.StudioError):
                s.plan_edit({'instruction': 'x', 'init': str(photo),
                             'mask': str(self.gray_mask('m.png', 64, 64, (0, 0, 8, 8)))})

    def test_estimates_are_marked_uncalibrated(self):
        s = self.studio
        info = s.edit_info()
        self.assertFalse(info['estimates_calibrated'])
        self.assertTrue(info['models']['q8'])
        self.assertFalse(info['models']['q4'])
        self.assertLess(info['estimates']['512'], info['estimates']['768'])
        self.assertGreater(s.edit_estimate_seconds(512, 512, masked=True), s.edit_estimate_seconds(512, 512))
        (s.PROJECT / 'edit.json').write_text(json.dumps({'estimate': {'calibrated': True, 'fixed_s': 1}}))
        self.assertTrue(s.edit_info()['estimates_calibrated'])


class Codec(EditCase):
    def test_png_roundtrip_and_resize(self):
        s = self.studio
        rgb = bytearray()
        for y in range(8):
            for x in range(8):
                rgb += bytes((x * 30, y * 30, 7))
        path = self.root / 'a.png'
        s.write_png(path, 8, 8, rgb)
        self.assertEqual(s.png_size(path), (8, 8))
        width, height, back = s.read_png(path)
        self.assertEqual((width, height, bytes(back)), (8, 8, bytes(rgb)))
        _, _, gray = s.read_png(path, gray=True)
        self.assertEqual(len(gray), 64)
        self.assertEqual(bytes(s.resize_pixels(bytes([100]) * 16 * 16, 16, 16, 4, 4, 1)), bytes([100]) * 16)
        self.assertEqual(len(s.resize_pixels(rgb, 8, 8, 12, 6, 3)), 12 * 6 * 3)

    def test_composite_keeps_base_where_mask_is_black(self):
        s = self.studio
        base = bytes([1, 2, 3]) * 4
        edited = bytes([201, 202, 203]) * 4
        out = s.composite_masked(base, edited, bytes([0, 255, 128, 0]))
        self.assertEqual(out[0:3], base[0:3])
        self.assertEqual(out[9:12], base[9:12])
        self.assertEqual(out[3:6], edited[3:6])
        self.assertEqual(list(out[6:9]), [(1 * 127 + 201 * 128 + 127) // 255, (2 * 127 + 202 * 128 + 127) // 255,
                                          (3 * 127 + 203 * 128 + 127) // 255])


class StubbedFlow(EditCase):
    def setUp(self):
        super().setUp()
        self.calls = []
        fixture = self

        def engine(args, job, environment=None, prompt=None):
            ref = option(args, '-r') if '-r' in args else None
            fixture.calls.append({'args': list(args), 'prompt': prompt,
                                  'ref_size': fixture.studio.png_size(ref) if ref else None})
            w, h = int(option(args, '-W')), int(option(args, '-H'))
            fixture.studio.write_png(option(args, '-o'), w, h, bytes((250, 0, 0)) * (w * h))  # edited image is all red
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

    def finish(self, body):
        started = self.request('/edit', body)
        self.assertIn('id', started, started)
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            job = self.request('/job?id=' + started['id'])
            if job['state'] in ('done', 'error'):
                return job
            time.sleep(0.02)
        self.fail('job did not finish')

    def test_unmasked_edit_returns_engine_output_and_never_leaks_instruction(self):
        photo = self.photo('p.png', 640, 480)
        job = self.finish({'instruction': 'secret jacket instruction', 'init': str(photo), 'seed': 9})
        self.assertEqual(job['state'], 'done', job)
        self.assertNotIn('secret jacket', json.dumps(job))
        self.assertEqual(self.calls[0]['prompt'], 'secret jacket instruction')
        image = job['images'][0]
        self.assertEqual((image['width'], image['height'], image['seed'], image['kind']), (512, 384, 9, 'edit'))
        self.assertEqual(self.studio.png_size(image['path']), (512, 384))
        self.assertEqual(image['actions']['refine'], {})   # refine is Z-Image only
        refused = self.request('/refine', {'image': image['id']})
        self.assertIn('error', refused)

    def test_masked_edit_composites_and_keeps_outside_pixels(self):
        s = self.studio
        # Left half is a gradient so "identical" is meaningful; the mask covers a box in the right half.
        w, h = 320, 240
        pixels = bytearray()
        for y in range(h):
            for x in range(w):
                pixels += bytes((x % 256, y % 256, (x + y) % 256))
        photo = s.UPLOADS / 'p.png'
        s.write_png(photo, w, h, pixels)
        mask = self.gray_mask('m.png', w, h, (160, 60, 300, 180))
        job = self.finish({'instruction': 'make it red', 'init': str(photo), 'mask': str(mask), 'crop': False})
        self.assertEqual(job['state'], 'done', job)
        call = self.calls[0]['args']
        ref = option(call, '-r')
        self.assertNotEqual(ref, str(photo))                       # working-size reference, not the upload
        self.assertEqual(self.calls[0]['ref_size'], (512, 384))
        self.assertNotIn('--mask', call)
        final = job['images'][0]['path']
        _, _, out = s.read_png(final)
        # Scratch files are removed after the job; rebuild the working reference to compare.
        _, _, rgb = s.read_png(photo)
        expected_ref = s.resize_pixels(rgb, w, h, 512, 384, 3)
        _, _, gray = s.read_png(mask, gray=True)
        expected_mask = s.resize_pixels(gray, w, h, 512, 384, 1)
        for i, a in enumerate(expected_mask):
            pixel = out[i * 3:i * 3 + 3]
            if a == 0:
                self.assertEqual(pixel, expected_ref[i * 3:i * 3 + 3])
            elif a == 255:
                self.assertEqual(bytes(pixel), bytes((250, 0, 0)))
        self.assertEqual(out[0:3], expected_ref[0:3])
        index = 200 * 512 + 400   # well inside the white box at working scale
        self.assertEqual(bytes(out[index * 3:index * 3 + 3]), bytes((250, 0, 0)))
        self.assertFalse(Path(ref).exists())                       # scratch cleaned up

    def test_models_endpoint_reports_edit_readiness(self):
        info = self.request('/models')['edit']
        self.assertTrue(info['ready'])
        self.assertFalse(info['estimates_calibrated'])
        self.assertEqual(info['sizes'], [512, 768])


if __name__ == '__main__':
    unittest.main()
