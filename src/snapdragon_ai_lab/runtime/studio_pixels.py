"""Prompt studio: pixel work with no third-party package and no Studio state. Image sizes from PNG
and JPEG headers, a minimal PNG codec, resizing, and the mask box, crop, blur and composite
steps of a masked edit. Pure functions over paths and flat byte buffers.
"""
from pathlib import Path
import math
import struct
import zlib


def png_size(path):
    """(width, height) from a PNG header, or None."""
    try:
        with open(path, 'rb') as stream:
            head = stream.read(24)
    except OSError:
        return None
    if len(head) < 24 or head[:8] != b'\x89PNG\r\n\x1a\n' or head[12:16] != b'IHDR':
        return None
    return struct.unpack('>II', head[16:24])


def jpeg_size(path):
    """(width, height) from a JPEG's SOF marker, or None."""
    try:
        data = Path(path).read_bytes()[:1 << 20]
    except OSError:
        return None
    if data[:2] != b'\xff\xd8':
        return None
    i = 2
    while i + 9 < len(data):
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7 or marker == 0xFF:
            i += 1 if marker == 0xFF else 2
            continue
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            h, w = struct.unpack('>HH', data[i + 5:i + 9])
            return (w, h)
        i += 2 + struct.unpack('>H', data[i + 2:i + 4])[0]
    return None


def image_size(path):
    return png_size(path) or jpeg_size(path)


# Minimal PNG codec (8/16-bit, non-interlaced) so the mask composite needs no
# third-party package. Pixels are flat bytearrays: RGB (3 channels) or gray (1).
def read_png(path, gray=False):
    data = Path(path).read_bytes()
    if data[:8] != b'\x89PNG\r\n\x1a\n':
        raise ValueError('The mask composite needs PNG images (convert JPEG to PNG first)')
    pos, idat, palette, header = 8, [], None, None
    while pos + 8 <= len(data):
        length, kind = struct.unpack('>I4s', data[pos:pos + 8])
        body = data[pos + 8:pos + 8 + length]
        pos += 12 + length
        if kind == b'IHDR':
            header = struct.unpack('>IIBBBBB', body)
        elif kind == b'PLTE':
            palette = body
        elif kind == b'IDAT':
            idat.append(body)
        elif kind == b'IEND':
            break
    if header is None or not idat:
        raise ValueError('Unreadable PNG')
    width, height, depth, ctype, _, _, interlace = header
    if interlace or depth not in (8, 16) or ctype not in (0, 2, 3, 4, 6) or (ctype == 3 and (depth != 8 or palette is None)):
        raise ValueError('Unsupported PNG (use 8-bit, non-interlaced)')
    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[ctype]
    step = channels * depth // 8
    stride = width * step
    raw = zlib.decompress(b''.join(idat))
    if len(raw) < (stride + 1) * height:
        raise ValueError('Truncated PNG')
    pixels = bytearray()
    previous = bytearray(stride)
    for y in range(height):
        base = y * (stride + 1)
        mode = raw[base]
        row = bytearray(raw[base + 1:base + 1 + stride])
        if mode == 1:
            for i in range(step, stride):
                row[i] = (row[i] + row[i - step]) & 255
        elif mode == 2:
            row = bytearray((a + b) & 255 for a, b in zip(row, previous))
        elif mode == 3:
            for i in range(stride):
                left = row[i - step] if i >= step else 0
                row[i] = (row[i] + ((left + previous[i]) >> 1)) & 255
        elif mode == 4:
            for i in range(stride):
                a = row[i - step] if i >= step else 0
                b, c = previous[i], (previous[i - step] if i >= step else 0)
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                row[i] = (row[i] + (a if pa <= pb and pa <= pc else b if pb <= pc else c)) & 255
        elif mode != 0:
            raise ValueError('Corrupt PNG filter')
        previous = row
        if depth == 16:
            row = row[0::2]
        out = row
        if ctype == 3:
            out = bytearray()
            for index in row:
                out += palette[index * 3:index * 3 + 3]
            channels_now = 3
        else:
            channels_now = channels
        if channels_now == 1 and not gray:
            out = bytearray(v for v in out for _ in range(3))
        elif channels_now == 2:
            out = bytearray(out[0::2]) if gray else bytearray(v for v in out[0::2] for _ in range(3))
        elif channels_now == 3 and gray:
            out = bytearray((299 * r + 587 * g + 114 * b) // 1000 for r, g, b in zip(out[0::3], out[1::3], out[2::3]))
        elif channels_now == 4:
            rgb = bytearray()
            for i in range(0, len(out), 4):
                rgb += out[i:i + 3]
            out = bytearray((299 * r + 587 * g + 114 * b) // 1000 for r, g, b in zip(rgb[0::3], rgb[1::3], rgb[2::3])) if gray else rgb
        pixels += out
    return width, height, pixels


def write_png(path, width, height, rgb):
    def chunk(kind, body):
        return struct.pack('>I', len(body)) + kind + body + struct.pack('>I', zlib.crc32(kind + body))
    stride = width * 3
    raw = b''.join(b'\x00' + bytes(rgb[y * stride:(y + 1) * stride]) for y in range(height))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0))
                           + chunk(b'IDAT', zlib.compress(raw, 6)) + chunk(b'IEND', b''))


def _taps(source, target):
    scale = source / target
    support = max(scale, 1.0)
    taps = []
    for i in range(target):
        center = (i + 0.5) * scale
        lo, hi = max(int(center - support), 0), min(int(center + support) + 1, source)
        weights = [max(0.0, 1 - abs((j + 0.5 - center) / support)) for j in range(lo, hi)]
        total = sum(weights) or 1.0
        taps.append((lo, [w / total for w in weights]))
    return taps


def resize_pixels(pixels, width, height, new_width, new_height, channels):
    """Separable triangle-filter resize (area-aware when shrinking)."""
    if (width, height) == (new_width, new_height):
        return bytearray(pixels)
    stride = width * channels
    xt = _taps(width, new_width)
    rows = []
    for y in range(height):
        row = pixels[y * stride:(y + 1) * stride]
        out = [0.0] * (new_width * channels)
        for c in range(channels):
            plane = row[c::channels]
            out[c::channels] = [sum(plane[lo + k] * w for k, w in enumerate(ws)) for lo, ws in xt]
        rows.append(out)
    result = bytearray()
    for lo, ws in _taps(height, new_height):
        acc = [0.0] * (new_width * channels)
        for k, w in enumerate(ws):
            acc = [a + w * v for a, v in zip(acc, rows[lo + k])]
        result += bytes(min(255, max(0, int(v + 0.5))) for v in acc)
    return result


def _ceil16(value):
    return int(math.ceil(value / 16)) * 16


def _floor16(value):
    return int(math.floor(value / 16)) * 16


def mask_bounding_box(gray, width, height):
    """(x0, y0, x1, y1) with exclusive ends around the pixels > 0 of a flat gray mask, or None."""
    x0 = y0 = x1 = y1 = None
    for y in range(height):
        row = gray[y * width:(y + 1) * width]
        if not any(row):
            continue
        left = len(row) - len(row.lstrip(b'\x00'))
        right = len(row.rstrip(b'\x00'))
        x0 = left if x0 is None else min(x0, left)
        x1 = right if x1 is None else max(x1, right)
        if y0 is None:
            y0 = y
        y1 = y + 1
    if y0 is None:
        return None
    return x0, y0, x1, y1


def _grow_axis(start, end, image_len):
    """Grow [start, end) to a multiple of 16 inside [0, image_len), centred, shifting to fit.

    If the rounded-up length is longer than the image, round down to the largest multiple
    that fits; if the span itself does not fit, that is refused.
    """
    length = end - start
    target = _ceil16(length)
    if target > image_len:
        target = _floor16(image_len)
        if target < length:
            raise ValueError('The masked area cannot be cropped to a 16-pixel multiple inside this photo')
    new_start = max(0, min(start - (target - length) // 2, image_len - target))
    return new_start, new_start + target


def _fit_short(start, end, image_len, scale):
    """Native span for the short axis so that its scaled length is a multiple of 16.

    Returns (new_start, new_end, work_length). When the image is too short to grow to that
    span, the whole axis is used and its work length rounded down (aspect changes slightly).
    """
    length = end - start
    work = _ceil16(length * scale)
    need = int(math.ceil(work / scale))
    if need > image_len:
        need = image_len
        work = max(16, _floor16(image_len * scale))
    new_start = max(0, min(start - (need - length) // 2, image_len - need))
    return new_start, new_start + need, work


MIN_CROP_SIDE = 256              # smallest crop side handed to klein (a 16x16-token latent)


def _widen_axis(start, end, image_len, minimum):
    """Widen [start, end) to at least `minimum`, centred and shifted to fit inside the image.

    The widened span is capped at the largest multiple of 16 inside the image: a whole axis of
    200 px cannot become a 16-multiple, so it would be refused. A span already longer than the
    cap is left alone so that _grow_axis decides.
    """
    length = end - start
    cap = image_len if image_len % 16 == 0 else _floor16(image_len)
    target = max(length, min(minimum, cap))
    new_start = max(0, min(start - (target - length) // 2, image_len - target))
    return new_start, new_start + target


def crop_box_for_mask(mask_bbox, image_w, image_h, work_size):
    """Crop geometry for crop-edit-paste: returns (x0, y0, x1, y1, work_w, work_h).

    mask_bbox is (x0, y0, x1, y1) with exclusive ends. The box is padded by 25% of its larger
    side (at least 32 px) and clamped to the image. If the padded long edge fits work_size, the
    crop is edited 1:1 with both sides rounded up to multiples of 16 (grown inside the image).
    Otherwise the crop is scaled so its long edge equals work_size, and the short side is chosen
    so its scaled length is a multiple of 16. Raises ValueError when no crop can hold the box.
    """
    bx0, by0, bx1, by1 = mask_bbox
    if not (0 <= bx0 < bx1 <= image_w and 0 <= by0 < by1 <= image_h):
        raise ValueError('The masked area lies outside the photo')
    if work_size <= 0 or work_size % 16:
        raise ValueError('Crop work size must be a multiple of 16')
    pad = max(32, -(-max(bx1 - bx0, by1 - by0) // 4))
    x0, x1 = max(0, bx0 - pad), min(image_w, bx1 + pad)
    y0, y1 = max(0, by0 - pad), min(image_h, by1 + pad)
    # klein needs some picture to work with: never hand it less than MIN_CROP_SIDE per side
    # (or the whole axis when the photo is smaller than that).
    x0, x1 = _widen_axis(x0, x1, image_w, min(MIN_CROP_SIDE, work_size))
    y0, y1 = _widen_axis(y0, y1, image_h, min(MIN_CROP_SIDE, work_size))
    if max(x1 - x0, y1 - y0) <= work_size:
        x0, x1 = _grow_axis(x0, x1, image_w)
        y0, y1 = _grow_axis(y0, y1, image_h)
        return x0, y0, x1, y1, x1 - x0, y1 - y0
    scale = work_size / max(x1 - x0, y1 - y0)
    if x1 - x0 >= y1 - y0:
        y0, y1, short = _fit_short(y0, y1, image_h, scale)
        return x0, y0, x1, y1, work_size, short
    x0, x1, short = _fit_short(x0, x1, image_w, scale)
    return x0, y0, x1, y1, short, work_size


def crop_pixels(buffer, full_width, x0, y0, width, height, channels):
    """Rectangle of a flat buffer (channels per pixel) as a new flat buffer."""
    stride = full_width * channels
    span = width * channels
    out = bytearray()
    for row in range(height):
        base = (y0 + row) * stride + x0 * channels
        out += buffer[base:base + span]
    return out


def box_blur(values, width, height, radius):
    """Separable box blur of a flat gray buffer with clamped edges (radius 0 copies)."""
    if radius <= 0:
        return bytearray(values)
    size = 2 * radius + 1

    def blur_line(line):
        padded = [line[0]] * radius + list(line) + [line[-1]] * radius
        prefix = [0]
        for v in padded:
            prefix.append(prefix[-1] + v)
        return [(prefix[i + size] - prefix[i] + size // 2) // size for i in range(len(line))]

    horizontal = bytearray()
    for y in range(height):
        horizontal += bytes(blur_line(values[y * width:(y + 1) * width]))
    out = bytearray(width * height)
    for x in range(width):
        out[x::width] = bytes(blur_line(horizontal[x::width]))
    return out


def composite_crop(base_rgb, full_width, full_height, box, mask_gray, raw_png, work_size, final_png):
    """Paste the edited crop into the native photo through the feathered native mask.

    Pixels whose feathered alpha is 0 are copied from base_rgb unchanged.
    """
    x0, y0, cw, ch = box
    w, h, edited = read_png(raw_png)
    if (w, h) != tuple(work_size):
        raise ValueError('Edited image has an unexpected size')
    if (w, h) != (cw, ch):
        edited = resize_pixels(edited, w, h, cw, ch, 3)
    crop_mask = crop_pixels(mask_gray, full_width, x0, y0, cw, ch, 1)
    alpha = box_blur(crop_mask, cw, ch, max(2, max(cw, ch) // 128))
    out = bytearray(base_rgb)
    for row in range(ch):
        line = alpha[row * cw:(row + 1) * cw]
        if not any(line):
            continue
        start = ((y0 + row) * full_width + x0) * 3
        out[start:start + cw * 3] = composite_masked(out[start:start + cw * 3],
                                                    edited[row * cw * 3:(row + 1) * cw * 3], line)
    write_png(final_png, full_width, full_height, out)


def composite_masked(base, edited, mask):
    """Blend edited over base by mask (255 = edited, 0 = base). Equal-sized flat buffers."""
    out = bytearray(base)
    for i, a in enumerate(mask):
        if a == 255:
            out[i * 3:i * 3 + 3] = edited[i * 3:i * 3 + 3]
        elif a:
            for c in range(3):
                j = i * 3 + c
                out[j] = (base[j] * (255 - a) + edited[j] * a + 127) // 255
    return out


def composite_files(ref, mask_gray, raw_png, final_png, size):
    w, h, edited = read_png(raw_png)
    if (w, h) != tuple(size):
        raise ValueError('Edited image has an unexpected size')
    write_png(final_png, w, h, composite_masked(ref, edited, mask_gray))


def read_mask_at(mask, width, height):
    """Mask as a flat gray buffer at the photo's native size (resized when the sizes differ)."""
    mw, mh, gray = read_png(mask, gray=True)
    if (mw, mh) != (width, height):
        gray = resize_pixels(gray, mw, mh, width, height, 1)
    return gray


def refine_size(width, height, target):
    """Scale so the long edge is `target`, keeping aspect, on the 64-pixel grid."""
    scale = target / max(width, height)
    w = max(256, int(round(width * scale / 64)) * 64)
    h = max(256, int(round(height * scale / 64)) * 64)
    if w > 1536 or h > 1536 or w * h > 1536 * 1024:
        raise ValueError('Refine target is larger than 1536×1024 pixels')
    return w, h


def edit_work_size(width, height, long_edge):
    """Working size: long edge `long_edge`, aspect kept, both sides a multiple of 16 (FLUX.2 latent grid)."""
    scale = long_edge / max(width, height)
    w = max(64, int(round(width * scale / 16)) * 16)
    h = max(64, int(round(height * scale / 16)) * 16)
    return w, h
