#!/usr/bin/env python3
"""Explicit MEMLAT default selection and fail-safe stock fallback (no reboot).

Install a root-owned copy and policy outside the checkout before selecting it.
The ALPM pre-transaction hook uses prepare-update, not check: the installed
kernel has not changed yet. Post hooks rebuild and restore the approved default.
"""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time


def sha256(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def entries(text):
    """Resolve the direct menu subset used here; reject ambiguous/include syntax."""
    if re.search(r'^\s*(include|config_path)\s*:', text, re.M | re.I):
        raise ValueError('included menus need manual review')
    headers = list(re.finditer(r'^\s*(/+)([^/\r\n][^\r\n]*)$', text, re.M))
    stack, result = [], {}
    for i, header in enumerate(headers):
        depth, name = len(header[1]), header[2].lstrip('+')
        if any(c in name for c in '\\/#') or depth > len(stack) + 1:
            raise ValueError('unsupported menu name or nesting')
        stack = stack[:depth - 1] + [name]
        key = '/'.join(stack)
        if key in result:
            raise ValueError('duplicate menu path')
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        body = text[header.end():end]
        paths = re.findall(r'^\s*path\s*:\s*(\S+)\s*$', body, re.M | re.I)
        protocol = re.findall(r'^\s*protocol\s*:\s*(\S+)\s*$', body, re.M | re.I)
        cmdlines = re.findall(r'^[ \t]*cmdline[ \t]*:[ \t]*(.*?)[ \t]*$', body, re.M | re.I)
        if len(cmdlines) > 1:
            raise ValueError('duplicate entry command lines need manual review')
        result[key] = {'index': i + 1, 'path': paths[0] if len(paths) == 1 else None,
                       'efi': protocol == ['efi'], 'cmdline': cmdlines[0] if cmdlines else None}
    return result


def selected(text, menu):
    defaults = re.findall(r'^\s*default_entry\s*:\s*(\S.*?)\s*$', text, re.M | re.I)
    if len(defaults) != 1 or re.search(r'^\s*remember_last_entry\s*:\s*yes', text, re.M | re.I):
        raise ValueError('require one explicit default with remember_last_entry disabled')
    value = defaults[0]
    if value.isdigit():
        names = [name for name, entry in menu.items() if entry['index'] == int(value)]
        if len(names) != 1:
            raise ValueError('unresolved numeric default')
        return names[0]
    return value


def verify_entry(menu, name, image, expected_sha=None):
    entry = menu.get(name)
    if not entry or not entry['efi'] or not entry['path']:
        raise ValueError(f'no unambiguous EFI leaf: {name}')
    path, separator, checksum = entry['path'].partition('#')
    # Policy paths refer to this machine's /boot ESP, never an arbitrary disk.
    expected_path = 'boot():/' + image.relative_to('/boot').as_posix()
    if path != expected_path or not image.is_file():
        raise ValueError(f'EFI image/path mismatch: {name}')
    if separator and (not re.fullmatch('[0-9a-fA-F]{128}', checksum) or
                      hashlib.blake2b(image.read_bytes()).hexdigest() != checksum.lower()):
        raise ValueError(f'Limine EFI hash mismatch: {name}')
    if expected_sha and sha256(image) != expected_sha:
        raise ValueError(f'EFI image changed since qualification: {name}')


def replace(path, before, after):
    temporary = None
    try:
        info = path.stat()
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.memlat-default-', delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(after.encode())
            handle.flush()
            os.fchmod(handle.fileno(), info.st_mode & 0o777)
            if os.geteuid() == 0:
                os.fchown(handle.fileno(), info.st_uid, info.st_gid)
            os.fsync(handle.fileno())
        if path.read_text() != before:
            raise ValueError('boot configuration changed concurrently')
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary:
            temporary.unlink(missing_ok=True)


def apply(policy, action):
    config = Path(policy['config'])
    before = config.read_text()
    menu = entries(before)
    current = selected(before, menu)
    own, stock = policy['memlat_entry'], policy['stock_entry']
    if action not in ('enable',) and current != own:
        return {'action': action, 'changed': False, 'default': current}
    verify_entry(menu, stock, Path(policy['stock_image']))
    if action == 'enable':
        if not Path(policy['modules']).is_dir():
            raise ValueError('qualified kernel modules missing')
        verify_entry(menu, own, Path(policy['memlat_image']), policy['memlat_sha256'])
        verify_entry(menu, stock, Path(policy['stock_image']), policy['stock_sha256'])
        target = own
    elif action == 'check':
        valid = (Path(policy['modules']).is_dir() and
                 sha256(Path(policy['stock_image'])) == policy['stock_sha256'])
        try:
            verify_entry(menu, own, Path(policy['memlat_image']), policy['memlat_sha256'])
        except (OSError, ValueError):
            valid = False
        target = own if valid else stock
    else:
        target = stock
    if action == 'prepare-update':
        pending = Path(policy['pending'])
        pending.parent.mkdir(parents=True, exist_ok=True)
        pending.write_text('Rebuild approved MEMLAT default after stock update.\n')
    after = re.sub(r'^(\s*default_entry\s*:)\s*.*$', lambda m: m[1] + ' ' + target,
                   before, count=1, flags=re.M | re.I)
    if after != before:
        replace(config, before, after)
    return {'action': action, 'changed': after != before, 'default': target}


def build_image(policy, output):
    """Use the installed stock DTB, requiring an exact embedded-payload match."""
    spec = importlib.util.spec_from_file_location('local_ai_memlat', policy['builder'])
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    if builder.secure_boot() != 'disabled':
        raise ValueError('unsigned MEMLAT rebuild requires Secure Boot disabled')
    stock = Path(policy['stock_image'])
    original = stock.read_bytes()
    uname = [s for s in builder.sections(original) if s['name'] == b'.uname']
    if len(uname) != 1:
        raise ValueError('expected one kernel version in stock UKI')
    section = uname[0]
    version = original[section['raw_offset']:section['raw_offset'] + section['virtual_size']].rstrip(b'\0').decode('ascii')
    if not re.fullmatch(r'[a-zA-Z0-9._+-]+', version):
        raise ValueError('invalid kernel version')
    modules = Path('/usr/lib/modules') / version
    kernel = modules / 'build'
    if not modules.is_dir() or 'CONFIG_SCMI_QCOM_MEMLAT_DEVFREQ=y' not in (kernel / '.config').read_text():
        raise ValueError('matching kernel modules/headers or built-in MEMLAT driver missing')
    builder.OVERLAY = Path(policy['overlay'])
    builder.build(argparse.Namespace(kernel_build=str(kernel), stock_dtb=policy['stock_dtb'],
        stock_dts=None, dtc=None, fdtoverlay=None, cpp='cpp', uki=str(stock), output=str(output)))
    # Recompute the expected payload edit independently of the build entry point.
    patched = output.read_bytes()
    old_sections, new_sections = builder.sections(original), builder.sections(patched)
    changed = [i for i, (a, b) in enumerate(zip(old_sections, new_sections))
               if original[a['raw_offset']:a['raw_offset'] + a['virtual_size']] !=
                  patched[b['raw_offset']:b['raw_offset'] + b['virtual_size']]]
    if len(changed) != 1 or old_sections[changed[0]]['name'] != b'.dtbauto':
        raise ValueError('expected exactly the Yoga DTB section to change')
    return dict(kernel=version, modules=str(modules), stock_sha256=sha256(stock),
                memlat_sha256=sha256(output), stock_dtb_sha256=sha256(Path(policy['stock_dtb'])))


def notify_failure(policy, error):
    # No prompts/logs are included; notify the configured desktop over its session bus.
    message = 'MEMLAT update failed; stock Omarchy selected. ' + str(error)[:300]
    print(message, flush=True)
    try:
        subprocess.run(['/usr/bin/runuser', '-u', policy['notification_user'], '--', '/usr/bin/env',
            'DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/' + str(policy['notification_uid']) + '/bus',
            '/usr/bin/notify-send', '-u', 'critical', 'Local AI: MEMLAT fallback', message],
            timeout=5, check=False)
    except (OSError, subprocess.TimeoutExpired):
        pass


def remove_owned_block(text, policy):
    expected = policy.get('automatic_entry_block', '')
    if not expected:
        return text
    if text.count(expected) == 1:
        return text.replace(expected, '', 1)
    # Boot-menu generators may normalize indentation/order. Accept formatting
    # changes only; never replace semantically modified or additional fields.
    name = policy['memlat_entry']
    headers = list(re.finditer(r'^\s*(/+)([^/\r\n][^\r\n]*)$', text, re.M))
    matches = [(i, h) for i, h in enumerate(headers) if h[1] == '/' and h[2] == name]
    if len(matches) != 1:
        raise ValueError('owned automatic menu entry missing or duplicated')
    index, header = matches[0]
    end = headers[index + 1].start() if index + 1 < len(headers) else len(text)
    actual = text[header.start():end]
    def fields(block):
        result = []
        for line in block.splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('/'):
                result.append(('header', line))
                continue
            key, separator, value = line.partition(':')
            if not separator:
                raise ValueError('unsupported automatic menu field')
            result.append((key.lower().strip(), value.strip()))
        return sorted(result)
    if fields(expected) != fields(actual):
        raise ValueError('owned automatic menu block changed semantically')
    return text[:header.start()] + text[end:]


def rebuild(policy_path, policy, *, dry_run=False, force=False):
    pending = Path(policy['pending'])
    if not (pending.exists() or dry_run or force):
        return {'action': 'rebuild', 'changed': False, 'reason': 'no update requested'}
    try:
        if not dry_run:
            # Always retain a bootable default during build/install/policy writes.
            apply(policy, 'fallback')
        with tempfile.TemporaryDirectory(prefix='memlat-rebuild-') as directory:
            image = Path(directory) / 'candidate.efi'
            evidence = build_image(policy, image)
            if dry_run:
                return dict(action='rebuild-dry-run', changed=False, passed=True, **evidence)
            config = Path(policy['config'])
            before = config.read_text()
            menu = entries(before)
            if selected(before, menu) != policy['stock_entry']:
                raise ValueError('default changed during rebuild; refusing promotion')
            verify_entry(menu, policy['stock_entry'], Path(policy['stock_image']), evidence['stock_sha256'])
            destination = Path(policy.get('image_directory', '/boot/EFI/LocalAI')) / ('omarchy-memlat-' + evidence['kernel'] + '-' + evidence['memlat_sha256'][:12] + '.efi')
            if destination.exists():
                if sha256(destination) != evidence['memlat_sha256']:
                    raise ValueError('existing rebuilt image hash mismatch')
            else:
                temporary = destination.with_suffix('.staging')
                if temporary.exists():
                    raise ValueError('stale image staging file; inspect before retry')
                try:
                    with temporary.open('xb') as handle:
                        with image.open('rb') as source:
                            shutil.copyfileobj(source, handle)
                        handle.flush()
                        os.fsync(handle.fileno())
                    os.replace(temporary, destination)
                finally:
                    temporary.unlink(missing_ok=True)
            name = 'Local AI memlat (' + evidence['kernel'] + ')'
            # Keep the original tested entry/image as rollback. Replace only our
            # exact earlier auto-generated block, never arbitrary menu sections.
            body = remove_owned_block(before, policy)
            if name in entries(body):
                raise ValueError('new automatic entry name already exists')
            entry = ('\n/' + name + '\ncomment: Auto-rebuilt Yoga MEMLAT; stock fallback Omarchy/linux-omarchy\n'
                'protocol: efi\npath: boot():/' + destination.relative_to(policy.get('boot_root', '/boot')).as_posix() + '#' +
                hashlib.blake2b(image.read_bytes()).hexdigest() + '\n')
            # Preserve the STOCK entry's outer command line too. It may override
            # and differ from the UKI's embedded .cmdline (as on this Yoga).
            cmdline = menu[policy['stock_entry']]['cmdline']
            if cmdline is not None:
                entry += 'cmdline: ' + cmdline + '\n'
            after = body.rstrip('\n') + '\n' + entry
            updated = dict(policy, **evidence, memlat_entry=name, memlat_image=str(destination), automatic_entry_block=entry)
            record = json.dumps(updated, indent=2) + '\n'
            replace(policy_path, policy_path.read_text(), record)
            replace(config, before, after)
            result = apply(updated, 'enable')
            pending.unlink(missing_ok=True)
            return dict(result, **evidence, image=str(destination))
    except Exception as error:
        if not dry_run:
            # A crash before the final enable leaves stock selected. If policy
            # was updated, that newer policy still names the same stock image.
            latest = json.loads(policy_path.read_text())
            try:
                apply(latest, 'fallback')
            finally:
                notify_failure(latest, error)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('enable', 'fallback', 'check', 'prepare-update', 'rebuild'))
    parser.add_argument('--policy', default='/etc/local-ai/memlat-boot.json')
    parser.add_argument('--no-lock', action='store_true', help='only inside limine-entry-tool mutex')
    parser.add_argument('--dry-run', action='store_true', help='build and verify in temporary files; do not select/install')
    parser.add_argument('--force', action='store_true', help='explicitly rebuild even without an update marker')
    args = parser.parse_args()
    try:
        with open('/run/lock/boot-partition.lock', 'a') as lock:
            if not args.no_lock:
                deadline = time.monotonic() + 10
                while True:
                    try:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        if time.monotonic() >= deadline:
                            raise ValueError('boot-partition mutex busy; retry after boot tooling finishes')
                        time.sleep(0.1)
            policy_path = Path(args.policy)
            policy = json.loads(policy_path.read_text())
            if args.action == 'rebuild':
                result = rebuild(policy_path, policy, dry_run=args.dry_run, force=args.force)
            elif args.dry_run or args.force:
                raise ValueError('--dry-run/--force are only valid for rebuild')
            else:
                result = apply(policy, args.action)
                if args.action == 'fallback' and policy.get('pending'):
                    # Explicit manual rollback cancels a queued automatic rebuild.
                    Path(policy['pending']).unlink(missing_ok=True)
            print(json.dumps(result))
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f'memlat-boot: {error}\n')


if __name__ == '__main__':
    main()
