"""Ordered orchestration of already-guarded runners; no model-loading code.

Hardware runners own their build/inference leases, scopes and MemoryGuard.
This layer verifies frozen artifacts, records every arm, and refuses to turn
missing guard evidence or a failed warm-up into a successful measurement.
"""
from datetime import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import stat
import statistics
import subprocess
import sys
import threading
import time

from . import system_counters


BUILD_NAMES = {'cargo', 'rustc', 'cc', 'cc1', 'cc1plus', 'clang', 'clang++',
               'cmake', 'ninja', 'make', 'ld', 'lld', 'rust-lld'}
ID = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,63}')


def instant(value):
    result = datetime.fromisoformat(value)
    if result.tzinfo is None:
        raise ValueError('batch_times_require_timezone')
    return result


def load_plan(path):
    text = Path(path).read_text()
    if not text.lstrip().startswith('{'):
        matches = re.findall(r'^```json\s*\n(.*?)^```\s*$', text, re.M | re.S)
        if len(matches) != 1:
            raise ValueError('plan_requires_one_fenced_json_manifest')
        text = matches[0]
    plan = json.loads(text)
    validate_plan(plan)
    return plan


def validate_plan(plan):
    if plan.get('schema') != 1 or not isinstance(plan.get('steps'), list) or not plan['steps']:
        raise ValueError('batch_schema_and_steps_required')
    if instant(plan['not_before']) >= instant(plan['stop_at']):
        raise ValueError('batch_time_order')
    if plan.get('receipt_exporter'):
        item = plan['receipt_exporter']
        if (not Path(item['path']).expanduser().is_absolute()
                or not re.fullmatch('[a-f0-9]{64}', item['sha256'])):
            raise ValueError('absolute_pinned_exporter_required')
    seen = set()
    for step in plan['steps']:
        key = step.get('id')
        if not isinstance(key, str) or not ID.fullmatch(key) or key in seen:
            raise ValueError('unique_safe_step_id_required')
        seen.add(key)
        if step.get('kind') not in ('offline', 'controller', 'native'):
            raise ValueError('unknown_runner_kind')
        if step['kind'] != 'offline' and step.get('lease_owner') != 'runner':
            raise ValueError('hardware_runner_must_own_leases_and_guards')
        if type(step.get('timeout_s')) not in (int, float) or not 0 < step['timeout_s'] <= 14400:
            raise ValueError('bounded_runner_timeout_required')
        if step.get('timeout_verdict', 'FAIL') not in ('FAIL', 'BLOCKED'):
            raise ValueError('unknown_timeout_verdict')
        if not isinstance(step.get('receipt'), str) or not step['receipt']:
            raise ValueError('runner_receipt_required')
        if not step['receipt'].startswith('{run_dir}/') or '..' in Path(step['receipt']).parts:
            raise ValueError('receipt_must_be_fresh_inside_run_dir')
        if step.get('latest_start'):
            instant(step['latest_start'])
        environment = step.get('environment', {})
        if not isinstance(environment, dict) or any(
                not isinstance(k, str) or not re.fullmatch('[A-Za-z_][A-Za-z0-9_]*', k)
                or not isinstance(v, str) or '\x00' in v for k, v in environment.items()):
            raise ValueError('environment_requires_string_pairs')
        clean_arm_environment(environment)
        if 'flash_cache_gib' in step and (type(step['flash_cache_gib']) not in (int, float)
                                        or not 0 < step['flash_cache_gib'] <= 21):
            raise ValueError('invalid_flash_cache_budget')
        commands = expanded_runs(step)
        if not commands:
            raise ValueError('runner_command_required')
        for _, argv, _ in commands:
            if not isinstance(argv, list) or not argv or not all(
                    isinstance(x, str) and '\x00' not in x for x in argv):
                raise ValueError('argv_list_required_no_shell')
        if step['kind'] != 'offline' and not step.get('artifacts'):
            raise ValueError('hardware_runner_artifacts_required')
        for item in step.get('artifacts', []):
            if not Path(item['path']).expanduser().is_absolute() or not re.fullmatch('[a-f0-9]{64}', item['sha256']):
                raise ValueError('absolute_pinned_artifact_required')


def expanded_runs(step):
    if 'argv' in step:
        if 'arms' in step:
            raise ValueError('choose_command_or_arms')
        return [('run', step['argv'], False)]
    arms = step.get('arms', {})
    order = step.get('order', [])
    drop = step.get('drop_first', 1)
    if not isinstance(arms, dict) or not arms or not order or type(drop) is not int or drop < 0:
        raise ValueError('arms_order_and_drop_required')
    counts, result = {}, []
    for name in order:
        if not isinstance(name, str) or not ID.fullmatch(name) or name not in arms:
            raise ValueError('unknown_or_unsafe_arm')
        counts[name] = counts.get(name, 0) + 1
        result.append((name, arms[name], counts[name] <= drop))
    if set(counts) != set(arms) or any(n <= drop for n in counts.values()):
        raise ValueError('every_arm_needs_a_retained_run')
    return result


def verify_artifacts(items):
    for item in items:
        path = Path(item['path']).expanduser()
        with path.open('rb') as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= 256 * 1024**2:
                raise ValueError('artifact_is_not_bounded_code')
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            after = os.fstat(stream.fileno())
        keys = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
        if digest != item['sha256'] or any(getattr(before, k) != getattr(after, k) for k in keys):
            raise ValueError('frozen_artifact_identity_changed')


def quiet_machine(lock_paths, proc_root=Path('/proc')):
    """Probe both build slots, then compiler names; never kill a process."""
    held = []
    try:
        for path in lock_paths:
            handle = Path(path).open('a')
            held.append(handle)
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return False
        for path in Path(proc_root).glob('[0-9]*/comm'):
            try:
                if path.read_text().strip() in BUILD_NAMES:
                    return False
            except (FileNotFoundError, PermissionError):
                pass
        return True
    finally:
        for handle in held:
            handle.close()


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'w') as stream:
        json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False); stream.write('\n')
    os.replace(temporary, path)


def guard_failure(receipt, step):
    if step['kind'] == 'offline':
        return None
    guards = receipt.get('guards')
    if guards is None:
        guards = [receipt.get('guard')]
    if not isinstance(guards, list) or not guards or any(not isinstance(g, dict) for g in guards):
        return 'missing_guard_evidence'
    for guard in guards:
        if guard.get('reason') is not None:
            return 'guard:' + str(guard['reason'])
        if type(guard.get('swap_current_bytes')) is not int or guard['swap_current_bytes'] != 0:
            return 'model_swap_or_missing_evidence'
        events = guard.get('memory_events', {})
        if any(type(events.get(key)) is not int or events[key] != 0 for key in ('oom', 'oom_kill')):
            return 'oom_or_missing_evidence'
    if receipt.get('host_oom_kills', 0) != 0:
        return 'host_oom_kill'
    if 'flash_cache_gib' in step and receipt.get('cache_gib') != step['flash_cache_gib']:
        return 'flash_cache_mismatch_or_missing_evidence'
    return None


def template(text, values):
    for key, value in values.items():
        text = text.replace('{' + key + '}', str(value))
    return text


def clean_arm_environment(overrides):
    """No inherited engine experiments: only explicit per-arm OXE_* values."""
    result = {k: v for k, v in os.environ.items() if not k.startswith('OXE_')}
    result.update(overrides)
    for name, low, high in [('OXE_PF_TILES', 0, 0x7fffffff), ('OXE_PF_GROUP', 1, 16)]:
        if name in result and (not re.fullmatch('[0-9]+', result[name]) or not low <= int(result[name]) <= high):
            raise ValueError('invalid_explicit_' + name)
    return result


def effective_oxe(receipt, *, required=False):
    """Observed native settings, not a guess from the wrapper's environment.

    A runner with multiple native arms supplies one pair per result row. Flash
    and legacy engine runners use applies:false; MoE groups require the pairs.
    """
    value = receipt.get('effective_oxe')
    if value is None and not required:
        return {'observed': False}
    rows = value if isinstance(value, list) else [value]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError('native_effective_oxe_settings_missing')
    clean = []
    for row in rows:
        if not required and row.get('applies') is False:
            clean.append({'applies': False})
            continue
        for key, low, high in [('OXE_PF_TILES', 0, 0x7fffffff), ('OXE_PF_GROUP', 1, 16)]:
            if type(row.get(key)) is not int or not low <= row[key] <= high:
                raise ValueError('invalid_native_effective_' + key)
        item = {k: row[k] for k in ('OXE_PF_TILES', 'OXE_PF_GROUP')}
        if isinstance(row.get('arm'), str) and ID.fullmatch(row['arm']):
            item['arm'] = row['arm']
        clean.append(item)
    return clean if isinstance(value, list) else clean[0]


def _stop_process(process):
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def _run_command(argv, directory, timeout_s, environment, interval_s=10):
    """Runner owns its descendants/guards; send TERM before bounded KILL."""
    before = system_counters.snapshot()
    stop = threading.Event()
    errors = []
    def sample():
        try:
            with (directory / 'counter-samples.jsonl').open('a') as stream:
                os.fchmod(stream.fileno(), 0o600)
                while not stop.wait(interval_s):
                    stream.write(json.dumps(system_counters.snapshot()) + '\n'); stream.flush()
        except Exception as error:
            errors.append(type(error).__name__)
    sampler = threading.Thread(target=sample, name='batch-counters', daemon=True)
    process = None
    timed_out = False
    started = time.monotonic()
    try:
        with (directory / 'runner.log').open('w') as log:
            os.fchmod(log.fileno(), 0o600)
            process = subprocess.Popen(argv, stdout=log, stderr=log, start_new_session=True,
                                       env=clean_arm_environment(environment), umask=0o077)
            sampler.start()
            try:
                process.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                timed_out = True
    finally:
        _stop_process(process)
        stop.set()
        if sampler.ident is not None:
            sampler.join(2)
        after = system_counters.snapshot()
        write_json(directory / 'counters-before.json', before)
        write_json(directory / 'counters-after.json', after)
        write_json(directory / 'counter-delta.json', system_counters.difference(before, after))
    return {'returncode': process.returncode, 'timed_out': timed_out, 'wall_s': time.monotonic() - started,
            'counter_errors': errors, 'counter_delta': system_counters.difference(before, after)}


def export_receipt(exporter, receipt, directory):
    """Run the release lane's pinned, fail-closed exporter; never publish here."""
    public = directory / 'public-extract.json'
    try:
        verify_artifacts([exporter])
        with (directory / 'export.log').open('w') as log:
            os.fchmod(log.fileno(), 0o600)
            result = subprocess.run(
                [sys.executable, str(Path(exporter['path']).expanduser()),
                 'export', str(receipt), '--out', str(public),
                 '--no-prompt-hash', '--no-source-hash'],
                stdout=log, stderr=log, timeout=30, umask=0o077)
        verify_artifacts([exporter])
        if result.returncode != 0 or not public.is_file():
            raise ValueError('public_export_refused')
        return {'passed': True, 'path': str(public), 'exporter_sha256': exporter['sha256']}
    except Exception as error:
        public.unlink(missing_ok=True)
        return {'passed': False, 'error': type(error).__name__,
                'note': 'Private measurement retained; no public extract available.'}


def run_plan(plan, out, *, now=None, before_step=None, available_bytes=None):
    validate_plan(plan)
    now = now or (lambda: datetime.now().astimezone())
    if not instant(plan['not_before']) <= now() < instant(plan['stop_at']):
        raise ValueError('outside_authorized_batch_window')
    out = Path(out); out.mkdir(mode=0o700, parents=True, exist_ok=False)
    report = {'schema': 1, 'start': now().isoformat(), 'steps': [], 'stopped': False}
    write_json(out / 'plan.json', plan)
    def save():
        write_json(out / 'batch-summary.json', report)
    for step in plan['steps']:
        if now() >= instant(plan['stop_at']) or (step.get('latest_start') and now() >= instant(step['latest_start'])):
            report['steps'].append({'id': step['id'], 'status': 'skipped_clock'}); save(); continue
        status = {'id': step['id'], 'status': 'running', 'runs': []}
        report['steps'].append(status)
        try:
            verify_artifacts(step.get('artifacts', []))
            if before_step:
                before_step(step)
            if 'flash_cache_gib' in step:
                available = available_bytes() if available_bytes else system_counters.snapshot()['memory_bytes']['value']['MemAvailable']
                if available < (step['flash_cache_gib'] + 5.87) * 2**30:
                    status['status'] = 'skipped_admission'; save(); continue
            for number, (arm, argv, dropped) in enumerate(expanded_runs(step), 1):
                if now() >= instant(plan['stop_at']):
                    status['status'] = 'incomplete_clock'; break
                if before_step:
                    before_step(step)
                if 'flash_cache_gib' in step:
                    available = available_bytes() if available_bytes else system_counters.snapshot()['memory_bytes']['value']['MemAvailable']
                    if available < (step['flash_cache_gib'] + 5.87) * 2**30:
                        status['status'] = 'incomplete_admission'; break
                verify_artifacts(step.get('artifacts', []))
                directory = out / step['id'] / f'{number:02d}-{arm}'
                directory.mkdir(parents=True, mode=0o700)
                values = {'outdir': out, 'run_dir': directory, 'arm': arm, 'rep': number}
                command = [template(value, values) for value in argv]
                run = {'arm': arm, 'number': number, 'dropped_warmup': dropped,
                       'start': now().isoformat(), 'argv': command}
                status['runs'].append(run); save()
                receipt_path = Path(template(step['receipt'], values))
                try:
                    # Required blocks (T1) finish their bounded gate, even if later
                    # optional blocks must be dropped from the end of the batch.
                    timeout = step['timeout_s']
                    if not step.get('required'):
                        timeout = min(timeout, max(.01, (instant(plan['stop_at']) - now()).total_seconds()))
                    run.update(_run_command(command, directory, timeout, step.get('environment', {})))
                    verify_artifacts(step.get('artifacts', []))
                    if not receipt_path.resolve().is_relative_to(directory.resolve()):
                        raise ValueError('receipt_escaped_run_directory')
                    receipt = json.loads(receipt_path.read_text())
                    run['runner_receipt'] = str(receipt_path)
                    reason = guard_failure(receipt, step)
                    oom = run['counter_delta']['vmstat'].get('delta', {}).get('oom_kill', 0)
                    if reason or oom:
                        run['guard_failure'] = reason or 'host_oom_during_runner'
                        raise RuntimeError('stop_on_guard')
                    run['guard_verified'] = True
                    run['effective_oxe'] = effective_oxe(receipt, required=step.get('require_effective_oxe') is True)
                    run['blocked'] = receipt.get('verdict') == 'BLOCKED'
                    run['passed'] = (not run.get('timed_out') and not run['blocked']
                                     and run['returncode'] == 0 and receipt.get('passed') is True)
                    run['metrics'] = {k: v for k, v in receipt.get('metrics', {}).items()
                                      if type(v) in (int, float) and math.isfinite(v)}
                    if not run['passed']:
                        raise ValueError('runner_failed')
                finally:
                    if (plan.get('receipt_exporter') and receipt_path.is_file()
                            and receipt_path.resolve().is_relative_to(directory.resolve())):
                        run['public_export'] = export_receipt(plan['receipt_exporter'], receipt_path, directory)
                    run['end'] = now().isoformat(); save()
            else:
                status['status'] = 'completed'
            kept = [r for r in status['runs'] if r.get('passed') and not r['dropped_warmup']]
            status['retained_runs'] = {arm: sum(r['arm'] == arm for r in kept)
                                       for arm in {r['arm'] for r in kept}}
            status['median_metrics'] = {
                arm: {key: statistics.median(r['metrics'][key] for r in kept if r['arm'] == arm and key in r['metrics'])
                      for key in {k for r in kept if r['arm'] == arm for k in r['metrics']}}
                for arm in status['retained_runs']}
        except Exception as error:
            timed_out = bool(status['runs'] and status['runs'][-1].get('timed_out'))
            blocked = bool(status['runs'] and status['runs'][-1].get('blocked'))
            status['status'] = ('blocked_timeout' if timed_out and step.get('timeout_verdict') == 'BLOCKED'
                                else 'blocked' if blocked else 'failed')
            status['error'] = type(error).__name__ + ':' + str(error)
            fatal = any(r.get('guard_failure') for r in status['runs']) or (
                step['kind'] != 'offline' and bool(status['runs'])
                and not status['runs'][-1].get('guard_verified'))
            if fatal or step.get('required') or not step.get('continue_on_failure', False):
                report['stopped'] = True; save(); break
        save()
    report['end'] = now().isoformat()
    report['all_steps_completed'] = all(s['status'] == 'completed' for s in report['steps']) and len(report['steps']) == len(plan['steps'])
    if plan.get('receipt_exporter'):
        report['all_public_extracts_ready'] = all(
            r.get('public_export', {}).get('passed') is True
            for step in report['steps'] for r in step.get('runs', [])) and report['all_steps_completed']
    save()
    return report
