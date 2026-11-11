#!/usr/bin/env python3
"""Opt-in Yoga MEMLAT UKI preparation and independently removable Limine entry.

No command changes the default entry, selects a boot, or reboots. Installation
and removal are explicit persistent operations; obtain the machine owner's approval.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import struct
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
OVERLAY = ROOT / 'benchmarks/hardware/memlat/yoga-memlat.dtso'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def sections(data):
    """Parse only bounded, unsigned PE32+ images; reject ambiguous/overlapping slots."""
    def unpack(fmt, offset):
        size = struct.calcsize(fmt)
        if offset < 0 or offset + size > len(data):
            raise ValueError('truncated PE image')
        return struct.unpack_from(fmt, data, offset)
    if data[:2] != b'MZ':
        raise ValueError('not a PE image')
    pe, = unpack('<I', 0x3c)
    if data[pe:pe + 4] != b'PE\0\0':
        raise ValueError('bad PE signature')
    count, = unpack('<H', pe + 6)
    size, = unpack('<H', pe + 20)
    opt = pe + 24
    if size < 152 or unpack('<H', opt)[0] != 0x20b:
        raise ValueError('PE32+ optional header required')
    if unpack('<I', opt + 64)[0] or unpack('<2I', opt + 144) != (0, 0):
        raise ValueError('signed/checksummed UKI unsupported; signing is not implemented')
    result = []
    for i in range(count):
        h = opt + size + i * 40
        unpack('<40s', h)
        vs, va, rs, rp = unpack('<4I', h + 8)
        if rp + rs > len(data) or vs > rs:
            raise ValueError('invalid PE section bounds')
        result.append(dict(header=h, name=data[h:h + 8].rstrip(b'\0'),
                           virtual_size=vs, virtual_addr=va, raw_size=rs, raw_offset=rp))
    table_end = opt + size + count * 40
    spans = sorted((s['raw_offset'], s['raw_offset'] + s['raw_size']) for s in result if s['raw_size'])
    if any(a < table_end for a, b in spans) or any(b > c for (a, b), (c, d) in zip(spans, spans[1:])):
        raise ValueError('overlapping PE sections/headers')
    return result


def patch_uki(data, stock, changed):
    all_sections = sections(data)
    selected = [s for s in all_sections if s['name'] == b'.dtbauto' and
                data[s['raw_offset']:s['raw_offset'] + s['virtual_size']] == stock]
    if len(selected) != 1:
        raise ValueError('expected exactly one .dtbauto matching the complete stock DTB')
    s = selected[0]
    if stock[:4] != b'\xd0\x0d\xfe\xed' or changed[:4] != stock[:4]:
        raise ValueError('DTB magic required')
    next_va = [q['virtual_addr'] for q in all_sections if q['virtual_addr'] > s['virtual_addr']]
    if len(changed) > s['raw_size'] or (next_va and s['virtual_addr'] + len(changed) > min(next_va)):
        raise ValueError('changed DTB does not fit the existing section')
    out = bytearray(data)
    struct.pack_into('<I', out, s['header'] + 8, len(changed))
    p, n = s['raw_offset'], s['raw_size']
    out[p:p + n] = changed + bytes(n - len(changed))
    return bytes(out)


def secure_boot():
    files = list(Path('/sys/firmware/efi/efivars').glob('SecureBoot-*'))
    if len(files) != 1:
        return 'unknown'
    try:
        value = files[0].read_bytes()
        return {0: 'disabled', 1: 'enabled'}.get(value[4], 'unknown')
    except (OSError, IndexError):
        return 'unknown'


def read_text(path):
    try:
        return Path(path).read_text().strip('\0\n ')
    except OSError:
        return 'unknown'


def doctor():
    model = read_text('/sys/firmware/devicetree/base/model')
    protocol = Path('/sys/firmware/devicetree/base/firmware/scmi/protocol@80').exists()
    domains = sorted(p.name for p in Path('/sys/class/devfreq').glob('*') if any(x in p.name.lower() for x in ('ddr', 'llcc')))
    config = Path('/usr/lib/modules') / platform.release() / 'build/.config'
    driver = 'CONFIG_SCMI_QCOM_MEMLAT_DEVFREQ=y' in read_text(config)
    result = dict(model=model, bios=read_text('/sys/class/dmi/id/bios_version'),
                  kernel=platform.release(), secure_boot=secure_boot(),
                  driver_builtin=driver, protocol80=protocol, devfreq_domains=domains)
    result['memlat_active'] = protocol and {'ddr', 'llcc'}.issubset(domains)
    result['guidance'] = ('MEMLAT protocol and DDR/LLCC domains present; measure benefit with matched controls.' if result['memlat_active'] else
        'Optional experiment only: built-in driver alone does not enable firmware MEMLAT. '
        'Validated profile: Lenovo 83ED, NHCN62WW, 7.3.0-40-omarchy. Unsupported machines should keep stock boot. '
        'Build a separate UKI, review it, and obtain approval before installing a test entry.')
    print(json.dumps(result, indent=2))


def run(*args):
    subprocess.run([str(a) for a in args], check=True)


def build(args):
    kernel = Path(args.kernel_build)
    dtc = args.dtc or shutil.which('dtc') or str(kernel / 'scripts/dtc/dtc')
    overlay_tool = args.fdtoverlay or shutil.which('fdtoverlay') or str(kernel / 'scripts/dtc/fdtoverlay')
    with tempfile.TemporaryDirectory(prefix='memlat-') as directory:
        tmp = Path(directory)
        stock = tmp / 'stock.dtb'
        if args.stock_dtb:
            stock.write_bytes(Path(args.stock_dtb).read_bytes())
        else:
            dts = Path(args.stock_dts) if args.stock_dts else kernel / 'arch/arm64/boot/dts/qcom/x1e80100-lenovo-yoga-slim7x.dts'
            pp = tmp / 'stock.dts'
            run(args.cpp, '-nostdinc', '-I', kernel / 'scripts/dtc/include-prefixes', '-I', dts.parent,
                '-undef', '-D__DTS__', '-x', 'assembler-with-cpp', dts, '-o', pp)
            run(dtc, '-@', '-I', 'dts', '-O', 'dtb', '-o', stock, pp)
        compiled = tmp / 'overlay.dtbo'
        changed = tmp / 'changed.dtb'
        run(dtc, '-@', '-I', 'dts', '-O', 'dtb', '-o', compiled, OVERLAY)
        run(overlay_tool, '-i', stock, '-o', changed, compiled)
        original = Path(args.uki).read_bytes()
        output = patch_uki(original, stock.read_bytes(), changed.read_bytes())
        destination = Path(args.output)
        with destination.open('xb') as handle:
            handle.write(output)
        print(json.dumps(dict(output=str(destination), source_sha256=digest(original),
                              output_sha256=digest(output), stock_dtb_sha256=digest(stock.read_bytes()),
                              changed_dtb_sha256=digest(changed.read_bytes())), indent=2))



def validate_default(config, *, removing_entry=None):
    """Conservative Limine subset: explicit numeric default and direct headers.

    Limine numbers menu headers, including submenu parents. Appending is safe
    only when the old default already resolves to a bootable leaf. Removal must
    not delete that target or renumber it. Unsupported syntax fails closed.
    """
    try:
        text = config.decode('utf-8')
    except UnicodeDecodeError as error:
        raise ValueError('Limine configuration must be UTF-8') from error
    defaults = re.findall(r'^\s*default_entry\s*:\s*(.*?)\s*$', text, re.M | re.I)
    if len(defaults) != 1 or not re.fullmatch(r'[1-9][0-9]*', defaults[0]):
        raise ValueError('require one explicit numeric default_entry; named/implicit defaults need manual review')
    if re.search(r'^\s*(?:include|config_path)\s*:', text, re.M | re.I):
        raise ValueError('included Limine menus require manual default review')
    headers = list(re.finditer(r'^(/+)([^/\r\n][^\r\n]*)$', text, re.M))
    index = int(defaults[0])
    if index > len(headers):
        raise ValueError('default_entry does not resolve before the change')
    selected = headers[index - 1]
    end = headers[index].start() if index < len(headers) else len(text)
    body = text[selected.end():end]
    if (index < len(headers) and len(headers[index].group(1)) > len(selected.group(1)) or
            not re.search(r'^\s*protocol\s*:\s*\S+', body, re.M | re.I) or
            not re.search(r'^\s*(?:path|kernel_path)\s*:\s*\S+', body, re.M | re.I)):
        raise ValueError('default_entry must already select a bootable leaf')
    if removing_entry is not None:
        # The owned entry is one direct top-level header, appended at install.
        own_offset = text.index(removing_entry.decode()) + 1
        owned = [i + 1 for i, h in enumerate(headers) if h.start() == own_offset]
        if len(owned) != 1 or index >= owned[0]:
            raise ValueError('removal would delete or renumber the persistent default; choose an earlier stock default first')


def replace_config(path, expected, replacement):
    """Stage a complete file before replacing it; refuse observed intervening edits."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.memlat-', delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(replacement)
            handle.flush()
            os.fsync(handle.fileno())
        shutil.copystat(path, temporary)
        if path.read_bytes() != expected:
            raise ValueError('configuration changed during operation; retry after review')
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def install(args):
    if secure_boot() != 'disabled':
        raise ValueError('installation requires verified Secure Boot disabled; signed boot is unsupported')
    source, dest, config, manifest = map(Path, (args.image, args.destination, args.config, args.manifest))
    config = config.resolve(strict=True)
    if len({p.resolve() for p in (source, dest, config, manifest)}) != 4:
        raise ValueError('image, destination, config, and manifest must be distinct paths')
    data = source.read_bytes()
    sections(data)
    if any(p.exists() for p in (dest, manifest)):
        raise ValueError('destination or manifest already exists; refusing to adopt existing artifacts')
    if not args.efi_path.startswith('boot():/') or any(c in args.efi_path for c in '\r\n'):
        raise ValueError('explicit Limine boot():/ EFI path required')
    entry = f'\n/Local AI MEMLAT opt-in ({digest(data)[:12]})\n    protocol: efi\n    path: {args.efi_path}\n'
    original = config.read_bytes()
    validate_default(original)
    if entry.encode() in original:
        raise ValueError('entry already exists')
    record = dict(version=1, destination=str(dest.resolve()), config=str(config.resolve()),
                  image_sha256=digest(data), entry=entry, entry_sha256=digest(entry.encode()))
    # No directories are created implicitly; caller chooses existing EFI and state directories.
    with dest.open('xb') as handle:
        handle.write(data)
    try:
        with manifest.open('x') as handle:
            json.dump(record, handle, indent=2)
        replace_config(config, original, original + entry.encode())
    except Exception:
        if config.read_bytes() == original:
            dest.unlink()
            manifest.unlink(missing_ok=True)
        raise
    print('Installed separate entry; default unchanged. No boot selected or reboot requested.')


def uninstall(args):
    manifest = Path(args.manifest)
    record = json.loads(manifest.read_text())
    dest, config = Path(record['destination']), Path(record['config'])
    entry = record['entry'].encode()
    data = config.read_bytes()
    if record['version'] != 1 or digest(entry) != record['entry_sha256']:
        raise ValueError('invalid ownership manifest')
    if data.count(entry) != 1 or digest(dest.read_bytes()) != record['image_sha256']:
        raise ValueError('owned image or entry changed; refusing removal; inspect manually')
    validate_default(data, removing_entry=entry)
    replace_config(config, data, data.replace(entry, b'', 1))
    dest.unlink()
    manifest.unlink()
    print('Removed owned entry and image; unrelated configuration preserved.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('doctor')
    p = sub.add_parser('build')
    p.add_argument('--uki', required=True)
    p.add_argument('--output', required=True)
    group = p.add_mutually_exclusive_group()
    group.add_argument('--stock-dtb')
    group.add_argument('--stock-dts')
    p.add_argument('--kernel-build', default=f'/usr/lib/modules/{platform.release()}/build')
    p.add_argument('--cpp', default='cpp')
    p.add_argument('--dtc')
    p.add_argument('--fdtoverlay')
    p = sub.add_parser('install', help='persistent: requires owner approval before running')
    for name in ('image', 'destination', 'config', 'manifest', 'efi-path'):
        p.add_argument('--' + name, required=True)
    p = sub.add_parser('uninstall', help='persistent: requires owner approval before running')
    p.add_argument('--manifest', required=True)
    args = parser.parse_args()
    try:
        globals()[args.command](args) if args.command != 'doctor' else doctor()
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, f'memlat: {error}\n')


if __name__ == '__main__':
    main()
