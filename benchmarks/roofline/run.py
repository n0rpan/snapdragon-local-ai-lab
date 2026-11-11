#!/usr/bin/env python3
"""Serial roofline measurements, one inference lease, existing memory policy."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src/snapdragon_ai_lab/runtime'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import memory_policy as memory
from nvme_read import sensors


def devfreq():
    return {path.name: {name: (path/name).read_text().strip()
                       for name in ('governor', 'cur_freq', 'min_freq', 'max_freq')
                       if (path/name).exists()}
            for path in Path('/sys/class/devfreq').iterdir()}


def run(argv, output, label, limit=500, ready_log=None, load_timeout=120):
    if ready_log is not None and ready_log.exists():
        raise ValueError('fresh_owned_benchmark_log_required')
    guard = memory.MemoryGuard(loading_timeout_seconds=load_timeout if ready_log else None)
    ready_seen = ready_log is None
    before = sensors()
    freq_before = devfreq()
    started = time.monotonic()
    sensor_peak = dict(before)
    sensor_minimum = dict(before)
    with tempfile.TemporaryFile() as raw:
        child = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=raw,
                                 stderr=raw, start_new_session=True)
        guard.start(child)
        failure = None
        try:
            while child.poll() is None:
                if not ready_seen and ready_log.exists():
                    info = ready_log.lstat()
                    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_size > 512*1024:
                        raise ValueError('invalid_owned_benchmark_log')
                    ready_text = ready_log.read_text(errors='replace')
                    if ': warmup ' in ready_text or 'LOCAL_AI_MODEL_READY\n' in ready_text:
                        guard.mark_ready()
                        ready_seen = True
                reason = guard.failure()
                power = sensors()
                for key, value in power.items():
                    sensor_peak[key] = max(sensor_peak.get(key, value), value)
                    sensor_minimum[key] = min(sensor_minimum.get(key, value), value)
                batteries = [v for k, v in power.items() if k.endswith('/capacity')]
                plugged = any(v for k, v in power.items() if k.endswith('/online'))
                if reason or (batteries and min(batteries) < 20 and not plugged):
                    raise RuntimeError(reason or 'unplugged_battery_below20')
                if time.monotonic()-started > limit:
                    raise TimeoutError('bounded_roofline_deadline')
                time.sleep(.25)
            if child.returncode:
                raise RuntimeError(guard.failure() or 'child_returncode_'+str(child.returncode))
        except BaseException as error:
            failure = type(error).__name__+':'+str(error)
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait(timeout=3)
        finally:
            guard.stop()
        raw.seek(0)
        text = raw.read(4*1024*1024).decode(errors='replace')
        parsed = []
        for line in text.splitlines():
            try:
                parsed.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        common = dict(label=label, argv=argv, utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      wall_seconds=time.monotonic()-started, sensors_before=before,
                      sensors_after=sensors(), returncode=child.returncode,
                      boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                      kernel=os.uname().release, devfreq_before=freq_before, devfreq_after=devfreq(),
                      whole_process_sensor_max=sensor_peak, whole_process_sensor_min=sensor_minimum,
                      minimum_available_bytes=guard.metadata()['minimum_available_bytes'],
                      memory_guard=guard.metadata(), benchmark_ready_observed=ready_seen)
        with output.open('a') as log:
            for row in parsed or [{}]:
                log.write(json.dumps({**common, **row, 'run_failure':failure})+'\n')
                print(json.dumps({k:v for k,v in {**common,**row,'run_failure':failure}.items()
                                  if k in ('label','threads','gb_s','backend','median_gflops','run_failure')}), flush=True)
        if failure:
            print(text[-2500:], file=sys.stderr)
            raise RuntimeError(failure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('kind', choices=('ram','nvme','command'))
    parser.add_argument('--binary', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--timeout-seconds', type=int, default=120)
    parser.add_argument('--bench-ready-log', '--native-ready-log', dest='bench_ready_log', type=Path)
    parser.add_argument('--load-timeout-seconds', type=int, default=120)
    args, command = parser.parse_known_args()
    if not 30 <= args.timeout_seconds <= 900:
        raise ValueError('timeout must be30..900seconds')
    path = Path.home()/'.local/state/local-ai/inference.lock'
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_uid != os.getuid():
        raise RuntimeError('owned_regular_lease_required')
    fd = os.open(path, os.O_RDWR|os.O_NOFOLLOW|os.O_CLOEXEC)
    try:
        current = os.fstat(fd)
        if (before.st_dev,before.st_ino) != (current.st_dev,current.st_ino):
            raise RuntimeError('lease_replaced')
        fcntl.flock(fd, fcntl.LOCK_EX|fcntl.LOCK_NB)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        if args.kind == 'ram':
            if not args.binary:
                raise ValueError('--binary required')
            packed = list(range(12))
            spread = [0,4,8,1,5,9,2,6,10,3,7,11]
            for repeat in range(3):
                for threads in (2,4,6,8,12):
                    arms = [('packed',packed),('spread',spread)]
                    for name, cpus in arms[::1 if repeat%2 == 0 else -1]:
                        run([str(args.binary),'--threads',str(threads),'--cpus',','.join(map(str,cpus[:threads])),
                             '--mib','2048','--passes','32'],args.output,f'ram-{repeat+1}-{name}',60)
        elif args.kind == 'nvme':
            run([sys.executable,'-B',str(Path(__file__).with_name('nvme_read.py')),
                 '--run','--seconds','2','--advise-owned-file'],args.output,'nvme',500)
        else:
            if command[:1] == ['--']:
                command = command[1:]
            if not command:
                raise ValueError('command required')
            run(command,args.output,'command',args.timeout_seconds,args.bench_ready_log,args.load_timeout_seconds)
    finally:
        os.close(fd)


if __name__ == '__main__':
    main()
