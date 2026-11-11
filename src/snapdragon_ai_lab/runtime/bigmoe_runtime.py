"""Isolated candidate's Flash identity/admission/private transport adapter.

No service or model is started here. All writes happen only when Manager has
the canonical lease and explicitly prepares a qualified Flash load.
"""
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import re
import secrets
import signal
import stat
import subprocess
import sys
import threading
import time

if __package__:
    from . import bigmoe_adapter as api
    from . import slim_request
else:
    import bigmoe_adapter as api
    import slim_request

GIB = 1024 ** 3
KIND = api.ENGINE_KIND
REVISION = '084c631ac221016019d6e0a863afe92d90edd3b0'
PINNED = (
    (10946618, '9d5644b371ffac921427c8021cc7242dab8e3ba37d7b48d1e79fe3770fbf9915'),
    (49946027488, 'aacaff3bdb9032c2c3be428167930809735f65d3cfb204eba07d85e421e7ddf4'),
    (4869325888, 'ef8f14e790ab471b363f146d5e8664eefbaf03e5776f317a1fb96a2bc7707b07'),
    (28800138432, '5de8b5dbf89409acc39468decafd4a4a95b13c29d88a14b0356107897829ec36'),
)


def fail(code):
    raise api.AdapterError(code)


def regular(path, private=False):
    path = Path(path)
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
        fail('unqualified_file_type_or_owner')
    if private and stat.S_IMODE(info.st_mode) & 0o077:
        fail('private_file_permissions_required')
    return info


def read_json(path):
    info = regular(path, private=True)
    if info.st_size > 131072:
        fail('metadata_size_limit')
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        fail('invalid_metadata_schema')
    return value


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while data := stream.read(1024 * 1024):
            value.update(data)
    return value.hexdigest()


def qualification(config, *, verify_binary=False):
    """Ready alone never admits a streamed model or a10MB metadata shard."""
    if config.get('ready') is not True or config.get('backend_kind') != KIND:
        fail('flash_profile_not_ready')
    if config.get('prefix_cache', {}).get('enabled') is not False:
        fail('flash_llama_prefix_cache_must_be_disabled')
    if config.get('context_size') not in (8192, 16384) or type(config.get('context_size')) is not int:
        fail('unqualified_flash_context')
    if config.get('threads') != 4 or config.get('gpu_layers') != 0:
        fail('unqualified_flash_device_policy')
    if config.get('extra_args') not in (None, []) or config.get('environment') not in (None, {}):
        fail('unqualified_flash_launch_overrides')
    settings = config.get('bigmoe')
    if not isinstance(settings, dict) or settings.get('revision') != REVISION or type(settings.get('cache_mib')) is not int or settings.get('cache_mib') not in (8192,12288):
        fail('unqualified_flash_baseline')
    cache_mib=settings['cache_mib']
    for key, expected in (('io_threads', 4), ('overlap', True), ('mapping_prefetch', False)):
        if type(settings.get(key)) is not type(expected) or settings[key] != expected:
            fail('unqualified_flash_storage_policy')
    slim=settings.get('slim_request',False)
    if type(slim) is not bool:fail('typed_slim_request_required')
    policy = settings.get('thp_policy', 'inherit')
    if cache_mib==12288 and policy!='disabled':fail('twelve_gib_requires_scoped_thp_disable')
    microbatch = settings.get('microbatch_size')
    if type(microbatch) is not int or microbatch not in (512, 4096) or policy not in ('inherit', 'disabled'):
        fail('unqualified_flash_microbatch_or_thp_policy')
    alias = config.get('model_alias')
    if not isinstance(alias, str) or re.fullmatch('[a-z0-9][a-z0-9._-]{0,95}', alias) is None:
        fail('invalid_flash_alias')
    shards = settings.get('shards')
    if not isinstance(shards, list) or len(shards) != 4:
        fail('four_verified_shards_required')
    model_root = Path(settings['models_dir'])
    if not model_root.is_absolute() or '..' in model_root.parts:
        fail('invalid_model_root')
    for index, shard in enumerate(shards, 1):
        path = Path(shard['path'])
        expected = PINNED[index - 1]
        if ((shard.get('bytes'), shard.get('sha256')) != expected or path.parent != model_root
                or path.name != f'Qwen3.8-Flash-Next-UD-IQ3_XXS-{index:05}-of-00004.gguf'):
            fail('shard_identity_mismatch')
        receipt = read_json(shard['verified_marker'])
        info = regular(path)
        when = receipt.get('verified_at')
        if (receipt.get('path') != str(path) or receipt.get('sha256') != shard['sha256']
                or receipt.get('bytes') != shard['bytes'] or info.st_size != shard['bytes']
                or type(when) not in (int, float) or not math.isfinite(when)
                or when <= 0 or info.st_mtime > when):
            fail('shard_verification_receipt_mismatch')
    if config.get('model_path') != shards[0]['path']:
        fail('metadata_shard_mismatch')
    binary = Path(config['server_binary'])
    regular(binary)
    digest = config.get('server_sha256')
    if not isinstance(digest, str) or re.fullmatch('[0-9a-f]{64}', digest) is None:
        fail('server_identity_required')
    if verify_binary and sha256(binary) != digest:
        fail('server_identity_mismatch')
    proof = read_json(settings['qualification_path'])
    required = {'passed': True, 'cpu_only': True, 'repack': False, 'engine_kind': KIND,
                'native_http_passed': True, 'tool_calls_passed': True,
                'agent_repair_passed': True, 'unload_ram_recovered': True,
                'revision': REVISION, 'server_sha256': digest,
                'shard_sha256': [row['sha256'] for row in shards],
                'context_size': config['context_size'], 'expert_cache_mib': cache_mib, 'slim_request':slim}
    required.update(microbatch_size=microbatch, thp_policy=policy, io_threads=4, overlap=True,
                    mapping_prefetch=False,
                    source_manifest_sha256='1fa9c70517f5543f8456afa3f69c2fad225c9bda9889d9f76c373ca2d635f2c5',
                    build_manifest_sha256='fa605e67eafbdb4e5965131f94e0577a68319fd67519263e729348856f2734b8')
    if slim:
        helper_sha=sha256(Path(__file__).with_name('slim_request.py'))
        system_sha=hashlib.sha256(slim_request.SYSTEM.encode()).hexdigest()
        if settings.get('slim_request_sha256')!=helper_sha:fail('slim_helper_identity_mismatch')
        required.update(slim_request_sha256=helper_sha,agent_system_sha256=system_sha)
    if policy == 'disabled':
        wrapper = wrapper_path()
        regular(wrapper)
        actual = sha256(wrapper)
        if settings.get('thp_wrapper_sha256') != actual:
            fail('thp_wrapper_identity_mismatch')
        required['thp_wrapper_sha256'] = actual
    for key, expected in required.items():
        if isinstance(expected, bool):
            matches = proof.get(key) is expected
        else:
            matches = type(proof.get(key)) is type(expected) and proof.get(key) == expected
        if not matches:
            fail('native_qualification_mismatch')
    admission = settings.get('admission', {})
    keys = ('resident_estimate_bytes', 'measured_peak_bytes', 'admitted_budget_bytes')
    if any(type(admission.get(key)) is not int or admission[key] <= 0 for key in keys):
        fail('measured_streamed_admission_required')
    if any(type(proof.get(key)) is not int or proof[key] != admission[key] for key in keys):
        fail('measured_admission_proof_mismatch')
    # Conservative static dense lower bound comes from the complete pinned GGUF headers.
    if admission['resident_estimate_bytes'] < 4522949120 + cache_mib * 1024**2:
        fail('selected_cache_missing_from_residency_estimate')
    if cache_mib==12288 or slim:
        agent_evidence(config,settings,proof,admission)
    if admission['admitted_budget_bytes'] > 24 * GIB:
        fail('budget_exceeds_core_overhead_margin')
    if max(admission[key] for key in keys[:2]) > admission['admitted_budget_bytes']:
        fail('streamed_budget_too_small')
    return settings, api.StreamedAdmission(
        model_disk_bytes=sum(row['bytes'] for row in shards), metadata_shard_bytes=shards[0]['bytes'],
        resident_estimate_bytes=admission['resident_estimate_bytes'],
        measured_peak_bytes=admission['measured_peak_bytes'],
        admitted_budget_bytes=admission['admitted_budget_bytes'], os_headroom_bytes=4 * GIB)


def agent_evidence(config,settings,proof,admission):
    """Bind an existing successful own synthetic repair artifact; never create one."""
    path=Path(settings.get('agent_repair_path',''))
    info=regular(path,private=True)
    if info.st_size>8*1024**2:fail('agent_repair_evidence_limit')
    raw=path.read_bytes();digest=hashlib.sha256(raw).hexdigest()
    if proof.get('agent_repair_sha256')!=digest:fail('agent_repair_evidence_hash_mismatch')
    value=json.loads(raw)
    if not isinstance(value,dict):fail('invalid_agent_repair_evidence')
    checks=('passed','task_passed','final_response_complete','ram_recovered')
    if any(value.get(key) is not True for key in checks) or value.get('client_exit')!=0:
        fail('agent_repair_not_successful')
    if (value.get('server_sha256')!=config['server_sha256']
        or [row.get('sha256') for row in value.get('model_shards',[])]!=[row['sha256'] for row in settings['shards']]
        or value.get('chosen_microbatch')!=settings['microbatch_size']
        or value.get('expert_cache_mib')!=settings['cache_mib']
        or value.get('slim_research_request') is not settings.get('slim_request',False)):
        fail('agent_repair_selected_policy_mismatch')
    if value.get('privacy_audit',{}).get('passed') is not True:fail('agent_repair_privacy_unverified')
    cleanup=value.get('cleanup',{})
    if not isinstance(cleanup,dict) or not cleanup or not all(item is True for item in cleanup.values()):fail('agent_repair_cleanup_unverified')
    if settings.get('thp_policy')=='disabled' and value.get('child_thp_disabled_verified') is not True:fail('agent_repair_thp_unverified')
    independent=value.get('independent',{})
    if (independent.get('test_exit')!=0 or independent.get('tests_run')!=12
        or independent.get('protected_unchanged') is not True
        or set(independent.get('changed_modules',[]))!={'validation.py','inventory.py','reservations.py'}
        or independent.get('unexpected_files')!=[]):fail('complete_independent_agent_checks_required')
    resolved=value.get('resolved_config',{})
    if (resolved.get('valid') is not True or resolved.get('auto_plan') is not False
        or resolved.get('reload_required') is not False):fail('agent_native_config_unverified')
    expected=dict(api.baseline_policy(settings['microbatch_size']),**{'ctx-size':config['context_size'],'cache-mb':settings['cache_mib']})
    if any(not api._same(resolved.get('values',{}).get(key),wanted) for key,wanted in expected.items()):fail('agent_native_policy_mismatch')
    peaks=[value.get('sampled_peak_rss_bytes'),value.get('cgroup_peak_bytes')]
    if any(type(peak) is not int or peak<=0 for peak in peaks) or max(peaks)>admission['measured_peak_bytes']:fail('agent_peak_exceeds_measured_admission')
    if settings.get('slim_request') is True:
        if value.get('harness_source_sha256',{}).get('slim_request.py')!=proof.get('slim_request_sha256'):fail('agent_slim_source_mismatch')
    return digest


def shape_chat(config,spec,payload):
    # Exact research filter preserves messages/tool arguments/limits; only schema list is narrowed.
    if config['bigmoe'].get('slim_request',False):
        try:payload=slim_request.request(payload)
        except (ValueError,TypeError):fail('slim_request_rejected')
    value=spec.chat_body(payload)
    cap=value.get('max_tokens',2048)
    if type(cap) is not int or not 1<=cap<=2048:fail('flash_output_limit_unqualified')
    if 'max_completion_tokens' in value:fail('flash_requires_max_tokens_field')
    temperature=value.get('temperature',0)
    if type(temperature) not in (int,float) or not math.isfinite(temperature) or temperature!=0:
        fail('flash_temperature_unqualified')
    value['max_tokens']=cap  # Preserve a caller's smaller cap; never silently raise it.
    value['temperature']=0
    return value


def eligible(config):
    try:
        qualification(config)
        return True
    except (api.AdapterError, OSError, ValueError, KeyError, TypeError):
        return False


def make_spec(config, port):
    settings = config['bigmoe']
    return api.Spec(binary=config['server_binary'], model_path=config['model_path'],
                    models_dir=settings['models_dir'], data_dir=settings['data_dir'],
                    public_model_id=config['model_alias'], port=port, context=config['context_size'], cache_mib=settings['cache_mib'],
                    microbatch=settings['microbatch_size'],
                    api_key_file=str(Path(settings['data_dir']) / 'backend-api.key'))


def normalize_chat(config, payload, port):
    spec = make_spec(config, port)
    spec.validate()
    return shape_chat(config,spec,payload)


def wrapper_path():
    return Path(__file__).resolve().with_name('exec_no_thp.py')


def power_snapshot(root=Path('/sys/class/power_supply')):
    """Read local power telemetry; no power settings or user configuration."""
    supplies = []
    try:
        for device in root.iterdir():
            row = {'ucsi_supply': device.name.startswith('ucsi-source-psy-')}
            for key in ('type', 'scope', 'online', 'status', 'capacity', 'energy_now',
                        'energy_full', 'charge_now', 'charge_full'):
                try:
                    row[key] = (device / key).read_text().strip()
                except OSError:
                    pass
            supplies.append(row)
    except OSError:
        pass
    batteries = [x for x in supplies if x.get('type') == 'Battery' and x.get('scope') != 'Device']
    battery = next((x for x in batteries if 'energy_now' in x or 'charge_now' in x),
                   batteries[0] if batteries else {})
    percent = None
    try:
        if battery.get('capacity', '').strip():
            percent = float(battery['capacity'])
        else:
            for kind in ('energy', 'charge'):
                try:
                    full = float(battery.get(kind + '_full', 0))
                    now = float(battery[kind + '_now'])
                    if math.isfinite(full) and full > 0 and math.isfinite(now) and 0 <= now <= full:
                        percent = 100 * now / full
                        break
                except (ValueError, KeyError, OverflowError):
                    continue
    except (ValueError, KeyError, OverflowError):
        pass
    if percent is not None and (not math.isfinite(percent) or not 0 <= percent <= 100):
        percent = None
    return {'external_online': any((x.get('type') in ('Mains', 'Wireless') or x.get('type', '').startswith('USB')
                                   or x['ucsi_supply']) and x.get('online') == '1' for x in supplies),
            'battery_status': battery.get('status'), 'battery_percent': percent}


def power_failure(value, *, starting=False):
    if not isinstance(value, dict):
        return 'flash_power_telemetry_unavailable'
    percent = value.get('battery_percent')
    # A readable USB-C supply is independent evidence of external power when
    # battmgr temporarily provides no reserve reading. Never invent a percentage.
    if percent is None and value.get('external_online') is True:
        return None
    if type(percent) not in (int, float) or not math.isfinite(percent) or not 0 <= percent <= 100:
        return 'flash_power_telemetry_unavailable'
    if percent <= 20:
        return 'flash_battery_stop_reserve'
    if starting and percent < 30:
        return 'flash_battery_start_reserve'
    if value.get('battery_status') not in ('Charging', 'Full', 'Not charging', 'Discharging'):
        return 'flash_power_telemetry_unavailable'
    return None


class PowerTelemetry:
    """Log transitions into degraded reserve telemetry, without flooding guard logs."""
    def __init__(self, reader=None, *, clock=time.monotonic, log=None):
        self.reader = power_snapshot if reader is None else reader
        self.clock = clock
        self.log = (lambda line: print(line, file=sys.stderr, flush=True)) if log is None else log
        self.degraded_since = None
        self.degraded_samples = 0
        self.longest_degraded_seconds = 0.0
        self.warned_ten_minutes = False

    def sample(self):
        value = self.reader()
        now = self.clock()
        degraded = (isinstance(value, dict) and value.get('battery_percent') is None
                    and value.get('external_online') is True)
        if degraded:
            self.degraded_samples += 1
            if self.degraded_since is None:
                self.degraded_since = now
                self.warned_ten_minutes = False
                self.log('power: degraded telemetry; external supply online, battery reserve unreadable')
            seconds = now - self.degraded_since
            self.longest_degraded_seconds = max(self.longest_degraded_seconds, seconds)
            if seconds >= 600 and not self.warned_ten_minutes:
                self.warned_ten_minutes = True
                self.log('power: degraded battery telemetry persisted at least 10 minutes; follow up battmgr')
        elif self.degraded_since is not None:
            self.longest_degraded_seconds = max(self.longest_degraded_seconds, now - self.degraded_since)
            self.log('power: reserve telemetry recovered' if isinstance(value, dict)
                     and value.get('battery_percent') is not None else 'power: external fallback no longer available')
            self.degraded_since = None
        return value

    def metadata(self):
        return {'degraded_samples': self.degraded_samples,
                'longest_degraded_seconds': self.longest_degraded_seconds,
                'degraded_now': self.degraded_since is not None}


class BackendGuard:
    """Watch only our child; Manager keeps its lease until cleanup joins us."""
    def __init__(self, process, available, *, power=None, interval=1, reserve_bytes=4 * GIB):
        if type(reserve_bytes) is not int or not GIB <= reserve_bytes <= 4 * GIB:
            fail('flash_guard_reserve_unqualified')
        self.process, self.available, self.reserve_bytes = process, available, reserve_bytes
        self.power_telemetry = PowerTelemetry(power)
        self.power = self.power_telemetry.sample
        self.interval = interval
        self.failure = None
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self.watch, daemon=True)

    def check(self):
        reason = power_failure(self.power())
        if reason:
            return reason
        available = self.available()
        if type(available) is not int or available < self.reserve_bytes:
            return 'flash_available_memory_below_reserve'
        return None

    def start(self):
        self.thread.start()

    def watch(self):
        while not self.stopped.is_set() and self.process.poll() is None:
            try:
                failure = self.check()
            except (OSError, ValueError):
                failure = 'flash_guard_observation_failed'
            if failure:
                self.failure = failure
                try:
                    os.killpg(self.process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(self.process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    self.process.wait()
                return
            self.stopped.wait(self.interval)

    def stop(self):
        self.stopped.set()
        self.thread.join(timeout=12)
        if self.thread.is_alive():
            fail('flash_guard_cleanup_incomplete')


class Transport:
    def __init__(self, port, key):
        self.port, self.key = port, key

    def request(self, method, path, body=None):
        connection = http.client.HTTPConnection('127.0.0.1', self.port, timeout=3)
        try:
            connection.request(method, path, body=None if body is None else json.dumps(body).encode(),
                               headers={'Authorization': 'Bearer ' + self.key,
                                        'Content-Type': 'application/json',
                                        'Host': '127.0.0.1:' + str(self.port)})
            response = connection.getresponse()
            data = response.read(131073)
            if len(data) > 131072:
                fail('backend_body_limit')
            return api.Response(response.status, data, response.getheader('Content-Type', ''))
        finally:
            connection.close()


class Runtime:
    def __init__(self, config, port):
        self.config = config
        self.settings, self.admission = qualification(config, verify_binary=True)
        self.key = None
        self.guard = None
        self.native_environment = None
        self.spec = make_spec(config, port)
        self.spec.validate()

    def require_memory(self, available, *, lease_owned):
        self.admission.require(available, lease_owned=lease_owned)
        reason = power_failure(power_snapshot(), starting=True)
        if reason:
            fail(reason)

    def start_guard(self, process, available, *, lease_owned, reserve_bytes=4 * GIB):
        if lease_owned is not True:
            fail('lease_required')
        self.guard = BackendGuard(process, available, reserve_bytes=reserve_bytes)
        self.guard.start()

    def guard_failure(self):
        return self.guard.failure if self.guard is not None else None

    def stop_guard(self):
        if self.guard is not None:
            self.guard.stop()

    def environment(self, _parent):
        if self.native_environment is None:
            fail('private_native_environment_not_prepared')
        return dict(self.native_environment)

    def prepare(self, *, lease_owned):
        if lease_owned is not True:
            fail('lease_required')
        directory = Path(self.spec.data_dir)
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        info = directory.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
            fail('private_backend_directory_required')
        environment = {'PATH':'/usr/bin:/bin','LANG':'C.UTF-8',
                       'OMP_NUM_THREADS':'6','OMP_WAIT_POLICY':'PASSIVE','OMP_DYNAMIC':'FALSE'}
        for key,name in (('HOME','native-home'),('XDG_CONFIG_HOME','native-config'),
                         ('XDG_DATA_HOME','native-data'),('XDG_STATE_HOME','native-state'),
                         ('XDG_CACHE_HOME','native-cache')):
            path = directory / name
            try:
                path.mkdir(mode=0o700)
            except FileExistsError:
                pass
            current = path.lstat()
            if not stat.S_ISDIR(current.st_mode) or current.st_uid != os.getuid() or stat.S_IMODE(current.st_mode) & 0o077:
                fail('private_native_directory_required')
            environment[key] = str(path)
        self.native_environment = environment
        settings = directory / 'settings.json'
        if settings.exists():
            api.validate_settings(read_json(settings), self.spec.microbatch)
        temporary = directory / ('.settings-' + secrets.token_hex(8))
        with temporary.open('x') as stream:
            os.fchmod(stream.fileno(), 0o600)
            json.dump(api.startup_settings(self.spec.microbatch), stream)
            stream.flush(); os.fsync(stream.fileno())
        temporary.replace(settings)
        key_path = Path(self.spec.api_key_file)
        if not key_path.exists():
            with key_path.open('x') as stream:
                os.fchmod(stream.fileno(), 0o600)
                stream.write(secrets.token_urlsafe(32) + '\n')
        regular(key_path, private=True)
        key = key_path.read_text().strip()
        if re.fullmatch('[A-Za-z0-9_-]{32,128}', key) is None:
            fail('invalid_private_backend_key')
        self.key = key
        self.adapter = api.Adapter(self.spec, Transport(self.spec.port, key))
        args = self.spec.launch_argv(api.startup_settings(self.spec.microbatch))
        if self.settings.get('thp_policy', 'inherit') == 'disabled':
            args = [sys.executable, '-I', '-B', str(wrapper_path()), '--sha256',
                    self.config['server_sha256'], '--', *args]
        return args

    def verify_child(self, process):
        if self.settings.get('thp_policy', 'inherit') != 'disabled':
            return
        if process.poll() is not None:
            fail('native_child_exited')
        status = {key: tail.strip() for key, separator, tail in
                  (line.partition(':') for line in Path('/proc', str(process.pid), 'status').read_text().splitlines())
                  if separator}
        if (int(status.get('PPid', '-1')) != os.getpid()
                or int(status.get('Uid', '-1').split()[0]) != os.getuid()
                or status.get('THP_enabled') != '0'):
            fail('own_native_child_thp_disable_unverified')

    def headers(self):
        if self.key is None:
            fail('private_backend_not_prepared')
        return {'Authorization': 'Bearer ' + self.key}

    def readiness(self, *, lease_owned):
        return self.adapter.readiness(lease_owned=lease_owned)

    def chat_body(self, payload):
        return shape_chat(self.config,self.spec,payload)
