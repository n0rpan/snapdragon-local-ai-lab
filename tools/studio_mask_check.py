#!/usr/bin/env python3
"""Check that a crop-edit-paste result changed nothing outside its feathered mask.

  python3 -I tools/studio_mask_check.py ORIGINAL.png RESULT.png MASK.png [--work-size 512]

Uses the Studio crop geometry and feather rule (crop_box_for_mask, box_blur with radius
max(2, crop long edge // 128)). Prints the number of pixels that differ from ORIGINAL where the
feathered mask is 0. Exits 0 only when that count is 0; 1 when it is not; 2 on bad input.
The Studio module is loaded from this checkout with temporary directories, so nothing under
the user's home or the Studio state directories is touched.
"""
import importlib.machinery
import importlib.util
import os
from pathlib import Path
import sys
import tempfile

SOURCE = Path(__file__).resolve().parents[1] / 'src/snapdragon_ai_lab/runtime/studio'


def load_studio(root):
    names = {'IMG_OUTDIR': 'output', 'LOCAL_IMAGE_CONFIG_DIR': 'config',
             'LOCAL_AI_LEASE': 'inference.lock', 'SD_HOME': 'stack'}
    for key, name in names.items():
        os.environ[key] = str(Path(root) / name)
    loader = importlib.machinery.SourceFileLoader('studio_mask_check', str(SOURCE))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def check(studio, original, result, mask, work_size):
    """Return the count of pixels outside the feathered mask that differ, or raise ValueError."""
    width, height, base = studio.read_png(original)
    rw, rh, out = studio.read_png(result)
    if (rw, rh) != (width, height):
        raise ValueError(f'result is {rw}x{rh}, original is {width}x{height}')
    mw, mh, gray = studio.read_png(mask, gray=True)
    if (mw, mh) != (width, height):
        gray = studio.resize_pixels(gray, mw, mh, width, height, 1)
    bbox = studio.mask_bounding_box(gray, width, height)
    if bbox is None:
        raise ValueError('the mask has no white pixels')
    x0, y0, x1, y1, _, _ = studio.crop_box_for_mask(bbox, width, height, work_size)
    cw, ch = x1 - x0, y1 - y0
    alpha = studio.box_blur(studio.crop_pixels(gray, width, x0, y0, cw, ch, 1), cw, ch,
                            max(2, max(cw, ch) // 128))
    differing = 0
    for y in range(height):
        for x in range(width):
            inside = x0 <= x < x1 and y0 <= y < y1 and alpha[(y - y0) * cw + (x - x0)] > 0
            if inside:
                continue
            i = (y * width + x) * 3
            if base[i:i + 3] != out[i:i + 3]:
                differing += 1
    return differing


def main(argv):
    args = list(argv)
    work_size = 512
    if '--work-size' in args:
        at = args.index('--work-size')
        try:
            work_size = int(args[at + 1])
        except (IndexError, ValueError):
            print('--work-size needs a number', file=sys.stderr)
            return 2
        del args[at:at + 2]
    if len(args) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    with tempfile.TemporaryDirectory() as root:
        studio = load_studio(root)
        try:
            count = check(studio, *args, work_size)
        except (ValueError, OSError) as error:
            print(f'error: {error}', file=sys.stderr)
            return 2
    print(f'pixels changed outside the feathered mask: {count}')
    return 0 if count == 0 else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
