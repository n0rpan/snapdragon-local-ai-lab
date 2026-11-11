#!/usr/bin/env python3
"""Read-only fio matrix; run under the coordinator's sole lease/resource guard.
No --run: metadata/argv preview only. 120 arms, default2s (240s timed I/O).
Root guard owns MemAvailable1.5GiB, PSI full10%/10s, ownswap0 and battery/temps.
Buffered GB/s may be RAM hits: never label it NVMe peak. No global cache drop.
"""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import signal
import stat
import subprocess
import tempfile
import time

DEFAULT = Path.home()/'.local/share/ai-stack/models/llm/Qwen3.8-Flash-Next/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00002-of-00004.gguf'
BLOCKS = (256*1024, 512*1024, 1024**2, 2*1024**2, 4*1024**2)
DEPTHS = (1, 4, 16, 32)
KEYS = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns', 'st_uid')


def identity(value):
    return {key: getattr(value, key) for key in KEYS}


def owned_file(path, minimum):
    value = path.lstat()
    if path.resolve() != path or not stat.S_ISREG(value.st_mode) or value.st_uid != os.getuid() or value.st_size <= minimum:
        raise ValueError('explicit_owned_regular_file_larger_than_RAM_required')
    return identity(value)


def advice(fd, path, expected):
    if identity(os.fstat(fd)) != expected or owned_file(path, 0) != expected:
        raise ValueError('owned_file_identity_changed_before_advice')
    os.posix_fadvise(fd, 0, expected['st_size'], os.POSIX_FADV_DONTNEED)
    return 'exact_held_owned_file_FADV_DONTNEED_advisory_not_guaranteed_cold'


def matrix():
    for rep in range(3):
        for bi, block in enumerate(BLOCKS):
            for qi, depth in enumerate(DEPTHS):
                for direct in ((1, 0) if rep % 2 == 0 else (0, 1)):
                    yield dict(rep=rep+1, block_bytes=block, qd=depth, direct=direct,
                               seed=20261109+rep*100+bi*4+qi)


def command(fio, path, size, engine, seconds, arm):
    if engine not in ('io_uring', 'libaio') or seconds not in (1, 2):
        raise ValueError('reviewed_engine_and_one_or_two_seconds_required')
    return [fio, '--readonly', '--name=nvme-read', '--filename='+str(path), '--rw=randread',
            '--allow_file_create=0', '--create_on_open=0', '--unlink=0', '--invalidate=0',
            '--fadvise_hint=0', '--ioengine='+engine, '--thread=1', '--numjobs=1',
            '--direct='+str(arm['direct']), '--bs='+str(arm['block_bytes']),
            '--iodepth='+str(arm['qd']), '--size='+str(size//arm['block_bytes']*arm['block_bytes']),
            '--time_based=1', '--runtime='+str(seconds), '--randrepeat=1',
            '--randseed='+str(arm['seed']), '--norandommap=1', '--group_reporting=1',
            '--output-format=json', '--eta=never']


def diskstats():
    return {r[2]: int(r[5])*512 for line in Path('/proc/diskstats').read_text().splitlines()
            if len(r := line.split()) >= 14 and re.fullmatch(r'nvme\d+n\d+(?:p\d+)?', r[2])}


def process_io(pid):
    try: return {k: int(v) for k, v in (line.split(':') for line in Path(f'/proc/{pid}/io').read_text().splitlines())}
    except (FileNotFoundError, ProcessLookupError): return {}


def sensors():
    out = {}
    for path in [*Path('/sys/class/thermal').glob('thermal_zone*/temp'),
                 *Path('/sys/class/hwmon').glob('hwmon*/temp*_input'),
                 *Path('/sys/class/power_supply').glob('*/*')]:
        if path.name not in ('temp', 'capacity', 'online', 'power_now', 'current_now', 'voltage_now', 'energy_now', 'energy_full', 'charge_now', 'charge_full') and not re.fullmatch(r'temp\d+_input', path.name):
            continue
        if path.name == 'online':
            try:
                if (path.parent/'type').read_text().strip() == 'Battery':
                    continue  # A battery being online does not mean external power.
            except OSError:
                pass
        try: out[path.parent.name+'/'+path.name] = int(path.read_text().strip())
        except (OSError, ValueError): pass
    for supply in Path('/sys/class/power_supply').iterdir():
        try:
            if (supply/'type').read_text().strip() != 'Battery':
                continue
        except OSError:
            continue
        prefix = supply.name+'/'
        if prefix+'capacity' in out:
            continue
        for unit in ('energy', 'charge'):
            now, full = out.get(prefix+unit+'_now'), out.get(prefix+unit+'_full')
            if now is not None and full and full > 0:
                out[prefix+'capacity'] = max(0, min(100, 100*now//full))
                break
    return out


def fio_metrics(data):
    jobs = data['jobs']
    if len(jobs) != 1 or jobs[0]['error'] != 0:
        raise ValueError('one_successful_readonly_fio_job_required')
    job = jobs[0]; read = job['read']
    if any(job.get(k, {}).get('io_bytes', 0) for k in ('write', 'trim')):
        raise ValueError('unexpected_nonread_IO')
    bw = read['bw_bytes'] if 'bw_bytes' in read else read['bw']*1024
    return dict(bandwidth_bytes_s=bw, GB_s=bw/1e9, GiB_s=bw/1024**3, IOPS=read['iops'],
                read_bytes=read['io_bytes'], read_runtime_ms=read['runtime'],
                latency_ns=read.get('lat_ns'), completion_latency_ns=read.get('clat_ns'),
                achieved_iodepth_percent=job.get('iodepth_level'))


def run_arm(argv, timeout):
    with tempfile.TemporaryFile() as output:
        child = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=output, stderr=output, start_new_session=True)
        before = process_io(child.pid); latest = before; began = time.monotonic()
        try:
            while child.poll() is None:
                latest = process_io(child.pid) or latest
                if time.monotonic()-began > timeout: raise TimeoutError('fio_arm_deadline')
                time.sleep(.05)
            output.seek(0); raw = output.read(1024*1024+1)
            if child.returncode or len(raw) > 1024*1024:
                raise ValueError('fio_returncode='+str(child.returncode)+':'+raw[:600].decode(errors='replace'))
            return fio_metrics(json.loads(raw)), {k: latest[k]-before.get(k, 0) for k in latest}
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try: child.wait(timeout=2)
                except subprocess.TimeoutExpired: os.killpg(child.pid, signal.SIGKILL); child.wait(timeout=2)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', action='store_true'); p.add_argument('--file', type=Path, default=DEFAULT)
    p.add_argument('--seconds', type=int, choices=(1, 2), default=2)
    p.add_argument('--engine', choices=('io_uring', 'libaio'))
    p.add_argument('--advise-owned-file', action='store_true'); args = p.parse_args()
    ram = int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemTotal:')))*1024
    expected = owned_file(args.file, ram); fio = shutil.which('fio')
    mount = json.loads(subprocess.check_output(['findmnt', '--json', '--target', str(args.file),
        '--output', 'SOURCE,UUID,FSTYPE,FSROOT,TARGET'], text=True, timeout=3))
    engine = args.engine; version = None
    if fio:
        version = subprocess.check_output([fio, '--version'], text=True, timeout=3).strip()
        if not re.fullmatch(r'fio-\d+(?:\.\d+)+(?:[\w.+-]*)', version): raise ValueError('Flexible_IO_tester_fio_required')
        engines = subprocess.check_output([fio, '--enghelp'], text=True, timeout=3)
        if engine is None: engine = next((v for v in ('io_uring', 'libaio') if re.search(r'\b'+v+r'\b', engines)), None)
        if not engine or not re.search(r'\b'+engine+r'\b', engines): raise ValueError('installed_async_read_engine_required')
    if not args.run:
        print(json.dumps(dict(metadata_only=True, fio=fio, fio_version=version, engine=engine,
              file=str(args.file), identity=expected, mount=mount, RAM_bytes=ram, arms=120, timed_IO_seconds=120*args.seconds,
              first_argv=command(fio or 'fio', args.file, expected['st_size'], engine or 'io_uring', args.seconds, next(matrix())))))
        return 0
    if not fio: raise ValueError('fio_not_installed_no_benchmark_started')
    fd = os.open(args.file, os.O_RDONLY|os.O_NOFOLLOW|os.O_CLOEXEC); began = time.monotonic()
    try:
        for arm in matrix():
            if time.monotonic()-began > 450: raise TimeoutError('matrix_eight_minute_budget')
            row = dict(kind='nvme_random_read', **arm, file=str(args.file), identity=expected, mount=mount,
                       fio_version=version, engine=engine, seconds=args.seconds,
                       throughput_scope='direct_logical_IO' if arm['direct'] else 'buffered_may_include_RAM_NOT_NVMe_peak')
            try:
                if owned_file(args.file, ram) != expected or identity(os.fstat(fd)) != expected: raise ValueError('target_identity_changed')
                row['cache_advice'] = advice(fd, args.file, expected) if args.advise_owned_file else 'none'
                argv = command(fio, args.file, expected['st_size'], engine, args.seconds, arm)
                row.update(argv=argv, sensors_before=sensors()); before = diskstats(); start = time.monotonic()
                metrics, proc = run_arm(argv, min(10, 475-(time.monotonic()-began)))
                elapsed = time.monotonic()-start; after = diskstats()
                row.update(metrics=metrics, wall_seconds=elapsed, process_IO_delta_last_live_sample_lower_bound=proc,
                           disk_read_bytes_delta={k: after[k]-v for k, v in before.items() if k in after},
                           disk_counter_scope='systemwide_per_device_do_not_sum_parent_and_partition', sensors_after=sensors())
                row['physical_GB_s_systemwide'] = {k: v/elapsed/1e9 for k, v in row['disk_read_bytes_delta'].items()}
                row['physical_to_fio_read_ratio_systemwide'] = {k: v/metrics['read_bytes'] if metrics['read_bytes'] else None for k, v in row['disk_read_bytes_delta'].items()}
                if owned_file(args.file, ram) != expected or identity(os.fstat(fd)) != expected: raise ValueError('target_changed_after_fio')
                row['passed'] = True
            except BaseException as error:
                row.update(passed=False, failure=type(error).__name__+':'+str(error)); raise
            finally: print(json.dumps(row, separators=(',', ':')), flush=True)
    finally: os.close(fd)
    return 0


if __name__ == '__main__': raise SystemExit(main())
