"""Crop-edit-paste for masked klein edits at native resolution; the sd-cli engine is stubbed.

No image model, sd-cli, service or live endpoint is used.
"""
from http.server import ThreadingHTTPServer
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'src/snapdragon_ai_lab/runtime/studio'
CHECKER = ROOT / 'tools/studio_mask_check.py'


def option(args, name):
    return args[args.index(name) + 1]


def load_studio(root):
    environment = {'IMG_OUTDIR': str(root / 'output'), 'LOCAL_IMAGE_CONFIG_DIR': str(root / 'config'),
                   'LOCAL_AI_LEASE': str(root / 'inference.lock'), 'SD_HOME': str(root / 'stack')}
    with patch.dict(os.environ, environment):
        loader = importlib.machinery.SourceFileLoader('crop_studio', str(SOURCE))
        spec = importlib.util.spec_from_loader(loader.name, loader)
        module = importlib.util.module_from_spec(spec)
        loader.exec_module(module)
    return module


class CropGeometry(unittest.TestCase):
    """Pure geometry: crop_box_for_mask(mask_bbox, image_w, image_h, work_size)."""

    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.s = load_studio(Path(cls.temporary.name))

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def assertInside(self, box, bbox, image_w, image_h):
        x0, y0, x1, y1 = box[:4]
        self.assertTrue(0 <= x0 <= bbox[0] and bbox[2] <= x1 <= image_w, (box, bbox))
        self.assertTrue(0 <= y0 <= bbox[1] and bbox[3] <= y1 <= image_h, (box, bbox))

    def test_box_at_top_left_corner(self):
        self.assertEqual(self.s.crop_box_for_mask((0, 0, 20, 20), 1000, 1000, 512), (0, 0, 256, 256, 256, 256))

    def test_box_at_bottom_right_corner_shifts_inside(self):
        self.assertEqual(self.s.crop_box_for_mask((980, 980, 1000, 1000), 1000, 1000, 512),
                         (744, 744, 1000, 1000, 256, 256))

    def test_box_on_left_edge(self):
        # x: padded to [0, 42), widened to 256 at the edge. y: padded to [368, 442), widened to 256 centred.
        self.assertEqual(self.s.crop_box_for_mask((0, 400, 10, 410), 1000, 1000, 512), (0, 277, 256, 533, 256, 256))

    def test_tiny_box_is_padded_to_a_small_crop(self):
        self.assertEqual(self.s.crop_box_for_mask((500, 500, 501, 501), 1000, 1000, 512), (373, 373, 629, 629, 256, 256))

    def test_non_square_box_is_edited_at_one_to_one(self):
        self.assertEqual(self.s.crop_box_for_mask((100, 100, 300, 200), 1200, 600, 512), (48, 22, 352, 278, 304, 256))

    def test_box_larger_than_work_size_is_scaled_on_the_grid(self):
        box = self.s.crop_box_for_mask((100, 100, 1100, 600), 2000, 2000, 512)
        self.assertEqual(box, (0, 0, 1350, 886, 512, 336))
        x0, y0, x1, y1, work_w, work_h = box
        self.assertEqual(max(work_w, work_h), 512)
        self.assertEqual(min(work_w, work_h) % 16, 0)
        self.assertEqual(work_h % 16, 0)

    def test_scaled_work_sizes_are_on_the_grid_for_many_boxes(self):
        s = self.s
        image_w, image_h = 3000, 2000
        boxes = [(10, 10, 2900, 1900), (1500, 40, 1600, 1990), (200, 700, 2500, 1100), (0, 0, 3000, 2000)]
        for work in (512, 768, 1024):
            for bbox in boxes:
                x0, y0, x1, y1, work_w, work_h = s.crop_box_for_mask(bbox, image_w, image_h, work)
                self.assertEqual(max(work_w, work_h), work, (bbox, work))
                self.assertEqual(min(work_w, work_h) % 16, 0, (bbox, work))
                self.assertTrue(0 <= x0 < x1 <= image_w and 0 <= y0 < y1 <= image_h, (bbox, work))
                self.assertInside((x0, y0, x1, y1), bbox, image_w, image_h)

    def test_one_to_one_sizes_are_multiples_of_16_and_contain_the_box(self):
        s = self.s
        for bbox in ((5, 5, 40, 90), (300, 200, 301, 260), (700, 100, 800, 180), (100, 800, 200, 900)):
            x0, y0, x1, y1, work_w, work_h = s.crop_box_for_mask(bbox, 1024, 1024, 768)
            self.assertEqual((work_w, work_h), (x1 - x0, y1 - y0))
            self.assertEqual((work_w % 16, work_h % 16), (0, 0), bbox)
            self.assertInside((x0, y0, x1, y1), bbox, 1024, 1024)

    def test_image_smaller_than_a_16_multiple(self):
        s = self.s
        # 100x90 photo: the widened span is capped at 96 (x) and 80 (y), the largest 16-multiples inside.
        self.assertEqual(s.crop_box_for_mask((20, 20, 30, 30), 100, 90, 512), (0, 0, 96, 80, 96, 80))
        # Rounding up to 96 fits a 100-px image here.
        self.assertEqual(s.crop_box_for_mask((40, 40, 70, 70), 100, 100, 512), (4, 4, 100, 100, 96, 96))

    def test_tiny_mask_in_large_photo_gets_a_centred_256_square(self):
        x0, y0, x1, y1, work_w, work_h = self.s.crop_box_for_mask((1000, 1000, 1001, 1001), 2000, 2000, 768)
        self.assertEqual((x1 - x0, y1 - y0, work_w, work_h), (256, 256, 256, 256))
        self.assertEqual((x0, y0), (873, 873))            # centre 1001 = mask centre
        self.assertEqual(x0 + x1, 2 * 1001)

    def test_photo_smaller_than_256_on_one_side_uses_the_whole_axis(self):
        # 240 px wide is a 16-multiple: the whole axis is used. y is padded and widened as usual.
        self.assertEqual(self.s.crop_box_for_mask((10, 400, 20, 410), 240, 1000, 512), (0, 277, 240, 533, 240, 256))
        # 200 px wide is not a 16-multiple: the largest one inside (192) is used, not the whole 200 px.
        self.assertEqual(self.s.crop_box_for_mask((10, 400, 20, 410), 200, 1000, 512), (0, 277, 192, 533, 192, 256))

    def test_box_that_cannot_be_held_at_a_16_multiple_is_refused(self):
        with self.assertRaises(ValueError):
            self.s.crop_box_for_mask((0, 0, 97, 97), 100, 100, 512)

    def test_invalid_inputs_are_refused(self):
        with self.assertRaises(ValueError):
            self.s.crop_box_for_mask((0, 0, 10, 10), 100, 100, 500)
        with self.assertRaises(ValueError):
            self.s.crop_box_for_mask((90, 90, 120, 120), 100, 100, 512)


class Helpers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.s = load_studio(Path(cls.temporary.name))

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_mask_bounding_box(self):
        s = self.s
        gray = bytearray(10 * 8)
        gray[3 * 10 + 2] = 255          # single pixel at (2, 3)
        gray[6 * 10 + 7] = 9            # any value > 0 counts
        self.assertEqual(s.mask_bounding_box(gray, 10, 8), (2, 3, 8, 7))
        self.assertIsNone(s.mask_bounding_box(bytearray(80), 10, 8))

    def test_box_blur_keeps_flat_areas_and_spreads_an_edge(self):
        s = self.s
        flat = bytearray([100]) * 36
        self.assertEqual(bytes(s.box_blur(flat, 6, 6, 2)), bytes(flat))
        spike = bytearray(36)
        spike[3 * 6 + 3] = 255
        blurred = s.box_blur(spike, 6, 6, 1)
        self.assertEqual(blurred[3 * 6 + 3], (255 + 9 // 2) // 9)  # 3x3 average of one bright pixel
        self.assertEqual(blurred[0], 0)
        self.assertEqual(bytes(s.box_blur(spike, 6, 6, 0)), bytes(spike))

    def test_crop_pixels_extracts_rectangle(self):
        s = self.s
        image = bytearray(range(4 * 3 * 3))  # 4x3, RGB
        crop = s.crop_pixels(image, 4, 1, 1, 2, 2, 3)
        expected = bytearray()
        for y in (1, 2):
            for x in (1, 2):
                expected += image[(y * 4 + x) * 3:(y * 4 + x) * 3 + 3]
        self.assertEqual(bytes(crop), bytes(expected))


class StubbedCrop(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.s = load_studio(self.root)
        s = self.s
        for path in (s.KLEIN_FILES['q8'], s.KLEIN_LLM, s.KLEIN_VAE):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'stub')
        s.PROJECT.mkdir(parents=True, exist_ok=True)
        self.calls = []
        fixture = self

        def engine(args, job, environment=None, prompt=None):
            ref = option(args, '-r')
            fixture.calls.append({'ref_size': fixture.s.png_size(ref), 'prompt': prompt,
                                  'work': (int(option(args, '-W')), int(option(args, '-H')))})
            w, h = int(option(args, '-W')), int(option(args, '-H'))
            fixture.s.write_png(option(args, '-o'), w, h, bytes((250, 0, 0)) * (w * h))  # edited is solid red
            job['state'] = 'generating 4/4'
            return 0

        self.engine = patch.object(self.s, 'run_engine', side_effect=engine)
        self.engine.start()
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), self.s.H)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(2)
        self.engine.stop()
        self.temporary.cleanup()

    def gradient(self, name, width, height):
        pixels = bytearray()
        for y in range(height):
            for x in range(width):
                pixels += bytes((x % 256, y % 256, (x * 3 + y) % 256))
        path = self.s.UPLOADS / name
        self.s.write_png(path, width, height, pixels)
        return path

    def mask(self, name, width, height, box):
        x0, y0, x1, y1 = box
        pixels = bytearray()
        for y in range(height):
            for x in range(width):
                pixels += b'\xff\xff\xff' if x0 <= x < x1 and y0 <= y < y1 else b'\x00\x00\x00'
        path = self.s.UPLOADS / name
        self.s.write_png(path, width, height, pixels)
        return path

    def request(self, path, body=None):
        url = f'http://127.0.0.1:{self.server.server_port}{path}'
        data = None if body is None else json.dumps(body).encode()
        try:
            with urlopen(Request(url, data=data), timeout=20) as response:
                return json.loads(response.read())
        except Exception as error:
            if hasattr(error, 'read'):
                with error:
                    return json.loads(error.read())
            raise

    def finish(self, body):
        started = self.request('/edit', body)
        self.assertIn('id', started, started)
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            job = self.request('/job?id=' + started['id'])
            if job['state'] in ('done', 'error'):
                return job
            time.sleep(0.05)
        self.fail('job did not finish')

    def expected_alpha(self, mask_path, width, height, box, radius_from_crop=True):
        """Feathered alpha over the crop using the Studio helpers (independent of the paste code)."""
        s = self.s
        _, _, gray = s.read_png(mask_path, gray=True)
        x0, y0, x1, y1 = box
        cw, ch = x1 - x0, y1 - y0
        return s.box_blur(s.crop_pixels(gray, width, x0, y0, cw, ch, 1), cw, ch, max(2, max(cw, ch) // 128))

    def assert_outside_unchanged(self, original, final, mask_path, box, width, height):
        s = self.s
        _, _, base = s.read_png(original)
        _, _, out = s.read_png(final)
        x0, y0, x1, y1 = box
        cw = x1 - x0
        alpha = self.expected_alpha(mask_path, width, height, box)
        for y in range(height):
            for x in range(width):
                inside = x0 <= x < x1 and y0 <= y < y1 and alpha[(y - y0) * cw + (x - x0)] > 0
                if not inside:
                    i = (y * width + x) * 3
                    self.assertEqual(bytes(out[i:i + 3]), bytes(base[i:i + 3]), (x, y))

    def test_crop_edit_keeps_native_size_outside_mask_and_pastes_the_centre(self):
        s = self.s
        w, h = 320, 240
        photo = self.gradient('p.png', w, h)
        mask = self.mask('m.png', w, h, (160, 60, 300, 180))
        job = self.finish({'instruction': 'make it red', 'init': str(photo), 'mask': str(mask)})
        self.assertEqual(job['state'], 'done', job)
        self.assertNotIn('make it red', json.dumps(job))
        self.assertEqual(self.calls[0]['ref_size'], (256, 240))     # crop edited at 1:1, on the 16 grid
        self.assertEqual(self.calls[0]['work'], (256, 240))
        image = job['images'][0]
        self.assertEqual((image['width'], image['height']), (w, h))
        self.assertEqual(image['crop'], {'box': [64, 0, 320, 240], 'work': [256, 240], 'scale': 1.0})
        final = image['path']
        self.assertEqual(s.png_size(final), (w, h))
        self.assert_outside_unchanged(photo, final, mask, (64, 0, 320, 240), w, h)
        _, _, out = s.read_png(final)
        centre = (120 * w + 230) * 3
        self.assertEqual(bytes(out[centre:centre + 3]), bytes((250, 0, 0)))

    def test_scaled_crop_edit_returns_native_size_and_records_scale(self):
        s = self.s
        w, h = 800, 600
        photo = self.gradient('big.png', w, h)
        mask = self.mask('big-m.png', w, h, (60, 60, 540, 440))
        job = self.finish({'instruction': 'make it red', 'init': str(photo), 'mask': str(mask)})
        self.assertEqual(job['state'], 'done', job)
        image = job['images'][0]
        self.assertEqual((image['width'], image['height']), (w, h))
        box = image['crop']['box']
        work = image['crop']['work']
        self.assertEqual(max(work), 512)
        self.assertEqual(min(work) % 16, 0)
        self.assertEqual(image['crop']['scale'], round(512 / max(box[2] - box[0], box[3] - box[1]), 4))
        self.assertEqual(self.calls[0]['work'], tuple(work))
        self.assertEqual(s.png_size(image['path']), (w, h))
        self.assert_outside_unchanged(photo, image['path'], mask, tuple(box), w, h)
        _, _, out = s.read_png(image['path'])
        centre = (300 * w + 300) * 3
        self.assertEqual(bytes(out[centre:centre + 3]), bytes((250, 0, 0)))

    def test_plan_defaults_to_crop_and_crop_false_keeps_whole_image(self):
        s = self.s
        photo = self.gradient('p.png', 320, 240)
        mask = self.mask('m.png', 320, 240, (160, 60, 300, 180))
        plan = s.plan_edit({'instruction': 'x', 'init': str(photo), 'mask': str(mask)})
        self.assertEqual(option(plan['steps'][0]['args'], '-W'), '256')
        self.assertEqual(plan['results'][0]['crop']['box'], [64, 0, 320, 240])
        self.assertEqual(plan['steps'][0]['expect_size'], (256, 240))
        self.assertIn('estimate', plan)
        old = s.plan_edit({'instruction': 'x', 'init': str(photo), 'mask': str(mask), 'crop': False})
        self.assertNotIn('crop', old['results'][0])
        self.assertEqual(old['results'][0]['width'], 512)
        self.assertEqual((option(old['steps'][0]['args'], '-W'), option(old['steps'][0]['args'], '-H')), ('512', '384'))

    def test_empty_mask_and_bad_crop_flag_are_refused(self):
        s = self.s
        photo = self.gradient('p.png', 320, 240)
        empty = self.mask('empty.png', 320, 240, (0, 0, 0, 0))
        with self.assertRaises(s.StudioError):
            s.plan_edit({'instruction': 'x', 'init': str(photo), 'mask': str(empty)})
        mask = self.mask('m.png', 320, 240, (160, 60, 300, 180))
        with self.assertRaises(s.StudioError):
            s.plan_edit({'instruction': 'x', 'init': str(photo), 'mask': str(mask), 'crop': 'no'})

    def test_masked_size_limit_is_checked_before_reading(self):
        s = self.s
        photo = self.gradient('p.png', 320, 240)
        mask = self.mask('m.png', 320, 240, (160, 60, 300, 180))
        with patch.object(s, 'MAX_MASK_PIXELS', 1000):
            with self.assertRaises(s.StudioError):
                s.plan_edit({'instruction': 'x', 'init': str(photo), 'mask': str(mask)})


class CheckerScript(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.s = load_studio(self.root)

    def tearDown(self):
        self.temporary.cleanup()

    def run_checker(self, *paths):
        return subprocess.run([sys.executable, '-I', str(CHECKER), *map(str, paths)],
                              capture_output=True, text=True, timeout=120)

    def make_case(self, changed_outside=0):
        s = self.s
        w, h = 320, 240
        base = bytearray()
        for y in range(h):
            for x in range(w):
                base += bytes((x % 256, y % 256, 7))
        result = bytearray(base)
        for y in range(h):
            for x in range(w):
                if 160 <= x < 300 and 60 <= y < 180:
                    result[(y * w + x) * 3:(y * w + x) * 3 + 3] = bytes((250, 0, 0))
        for k in range(changed_outside):
            result[k * 3] = (result[k * 3] + 1) % 256   # pixel 0.. is far outside the mask
        mask_pixels = bytearray()
        for y in range(h):
            for x in range(w):
                mask_pixels += b'\xff\xff\xff' if 160 <= x < 300 and 60 <= y < 180 else b'\x00\x00\x00'
        paths = {}
        for name, pixels in (('original.png', base), ('result.png', result), ('mask.png', mask_pixels)):
            paths[name] = self.root / name
            s.write_png(paths[name], w, h, pixels)
        return paths

    def test_identical_outside_mask_exits_zero(self):
        paths = self.make_case()
        done = self.run_checker(paths['original.png'], paths['result.png'], paths['mask.png'])
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn('0', done.stdout)

    def test_a_changed_outside_pixel_exits_one(self):
        paths = self.make_case(changed_outside=1)
        done = self.run_checker(paths['original.png'], paths['result.png'], paths['mask.png'])
        self.assertEqual(done.returncode, 1, done.stdout + done.stderr)
        self.assertIn('1', done.stdout)

    def test_bad_input_exits_two(self):
        paths = self.make_case()
        self.assertEqual(self.run_checker(paths['original.png'], paths['result.png']).returncode, 2)
        empty = self.root / 'empty.png'
        self.s.write_png(empty, 320, 240, bytes(320 * 240 * 3))
        self.assertEqual(self.run_checker(paths['original.png'], paths['result.png'], empty).returncode, 2)


if __name__ == '__main__':
    unittest.main()
