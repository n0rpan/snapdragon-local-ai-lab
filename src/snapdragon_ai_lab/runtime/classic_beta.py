"""Ornith metadata qualification; no model reads, inference or installation.

Ornith is the production default (Rust engine on MEMLAT since 2026-11-10).
The module name and the stored ``release_channel`` value predate that; they
are qualification data, not UI labels. Promoted profiles may say
``production`` instead of ``beta``; the proof requirements are identical.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess

ALIAS = 'ornith-35b-a3b-local'
MODEL_SHA = 'a7373f724cf3693e8a6a5f8209ae9a06ff8071ae2af60659bdbdf8ff6f6c10df'
MODEL_BYTES = 17437861152
SERVER_SHA = '1bd3e439f357365c50fc8390cc2c7e4cde2f86f14fa1771a65f2323a307457b1'
# Exact patched engine admitted ONLY for V2 candidate proofs (never arbitrary hashes).
PATCHED_SERVER_SHA = 'acd47ff19a161f9303be23ae1e99a80050579dad17071302ba77b3a0d6060efc'
SCREEN_SOURCE_SHA = '6d50ffd5dc13d00206281d1656c31090d6fe8dd750631d92475c4bfec067b616'
AGENT_SOURCE_SHA = '24fad6e86bbca6718d9235d5ffe3e648191bc65aeaf89865673b83e3d01d31f3'
POLICY = dict(context_size=16384, threads=4, gpu_layers=0, backend='cpu',
    extra_args=['--device','none','--no-op-offload','--no-host','--parallel','1',
        '--threads-batch','10','--batch-size','256','--ubatch-size','256','--repack',
        '--flash-attn','off','--cache-type-k','f16','--cache-type-v','f16','--cache-ram','0',
        '--ctx-checkpoints','4','--no-warmup','--jinja','--reasoning-budget','768',
        '--log-disable','--chat-template-kwargs','{"enable_thinking":false}',
        '--spec-type','draft-mtp','--spec-draft-n-max','2'],
    environment={'LLAMA_CPU_SMALL_BATCH_LIMIT':'3'})

RETIRED_ALIASES = frozenset({'arete-27b-local'})
DISPLAY_NAME = 'Ornith 1.5 35B A3B'
# Accepted stored channel values; neither relaxes any proof check.
RELEASE_CHANNELS = frozenset({'beta', 'production'})
CANDIDATE_KIND = 'classic-beta-candidate-v2'
_FILESYSTEM_CACHE = {}


def candidate_engine_shas():
    """V2 candidate engine allowlist; legacy V1 proofs remain bound to SERVER_SHA."""
    return frozenset((SERVER_SHA, PATCHED_SERVER_SHA))


def _allowed_engine_shas(profile):
    if isinstance(profile.get('validation'), dict) and profile['validation'].get('kind') == CANDIDATE_KIND:
        return candidate_engine_shas()
    return frozenset((SERVER_SHA,))


def is_beta(profile):
    return profile.get('model_alias') == ALIAS


def is_retired(profile):
    return profile.get('model_alias') in RETIRED_ALIASES


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def valid_sha(value):
    return isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value) is not None


def require_profile(profile):
    if profile.get('backend_kind') not in (None,'llama'):raise ValueError('Ornith beta uses the classic backend')
    if any(profile.get(k)!=v for k,v in POLICY.items()) or profile.get('sha256')!=MODEL_SHA or profile.get('expected_bytes')!=MODEL_BYTES or profile.get('server_sha256')!=SERVER_SHA:
        raise ValueError('Ornith requires the qualified classic model and policy')
    for name in ('model_path','server_binary'):
        value=profile.get(name)
        if not isinstance(value,str) or not Path(value).is_absolute() or any(c in value for c in '\n\r\0'):
            raise ValueError('Absolute qualified model and engine paths required')


def model_identity(path):
    info=Path(path).lstat()
    if not stat.S_ISREG(info.st_mode):raise ValueError('Qualified model must be a regular file')
    return dict(bytes=info.st_size,device=info.st_dev,inode=info.st_ino,mtime_ns=info.st_mtime_ns)


def eligible(profile):
    """Nonwaking metadata-only UI/load gate; identity drift requires reproof."""
    try:
        if profile.get('validation', {}).get('kind') == CANDIDATE_KIND:
            return _candidate_eligible(profile)
        if profile.get('model_alias') != ALIAS:
            return False
        require_profile(profile)
        proof=profile['validation']
        if profile.get('ready') is not True or profile.get('release_channel') not in RELEASE_CHANNELS or profile.get('prefix_cache',{}).get('enabled') is not False:return False
        if proof.get('kind')!='ornith-classic-beta-v1' or proof.get('normal_cases')!=13 or proof.get('policy_sha256')!=digest(POLICY):return False
        for key in ('normal13_sha256','screen_sha256','inventory_sha256','root_reviewed_normal13_sha256'):
            if not valid_sha(proof.get(key)):return False
        if proof['root_reviewed_normal13_sha256']!=proof['normal13_sha256']:return False
        return proof['model_identity']==model_identity(profile['model_path']) and proof['model_identity']['bytes']==MODEL_BYTES
    except (OSError,ValueError,TypeError,KeyError,AttributeError):return False


def verify_engine(profile):
    """Startup only, under the existing lease; never invoked by catalog polling."""
    descriptor=os.open(profile['server_binary'],os.O_RDONLY|os.O_NOFOLLOW|os.O_CLOEXEC)
    with os.fdopen(descriptor,'rb') as stream:
        before=os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_uid not in (0,os.getuid()) or not 0<before.st_size<=64*1024**2:raise ValueError('Bounded qualified engine required')
        check=hashlib.sha256()
        while data:=stream.read(1024**2):check.update(data)
        after=os.fstat(stream.fileno())
    expected=profile.get('server_sha256')
    if (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)!=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns) or expected not in _allowed_engine_shas(profile) or check.hexdigest()!=expected:raise ValueError('Qualified beta engine identity changed')


def qualify(agent,screen,inventory,*,normal_sha,screen_sha,inventory_sha,reviewed_normal_sha):
    """Validate explicitly hashed actual receipts before preparing an opt-in profile."""
    if not all(valid_sha(v) for v in (normal_sha,screen_sha,inventory_sha,reviewed_normal_sha)) or reviewed_normal_sha!=normal_sha:raise ValueError('Explicit root-reviewed normal-task receipt required')
    profile=agent.get('profile',{});require_profile(profile)
    if agent.get('passed') is not True or agent.get('stage')!='normal-agent' or agent.get('model_role')!='ornith' or agent.get('source_manifest_sha256')!=AGENT_SOURCE_SHA or agent.get('candidate_screen_quality_passed') is not True or agent.get('matched_screen_sha256',{}).get('ornith')!=screen_sha or agent.get('inventory_receipt_sha256')!=inventory_sha:raise ValueError('Actual qualified Ornith normal-task evidence required')
    for key in ('owned_native_reaped','all_children_reaped','reader_joined','ram_recovered','lease_free_after','post_source_pins_passed'):
        if agent.get(key) is not True:raise ValueError('Complete normal-task lifecycle required')
    inner=agent.get('measurement',{});task=inner.get('agent',{})
    if agent.get('guard_failures')!=[] or 'error' in agent or inner.get('passed') is not True or inner.get('server_reaped') is not True or inner.get('server_reader_joined') is not True or task.get('passed') is not True or task.get('expected_test_cases')!=13 or task.get('independent_test_cases_run')!=13 or task.get('complete') is not True or task.get('protected_unchanged') is not True or task.get('changed_modules')!=['parse_port.py'] or task.get('unexpected_files')!=[] or task.get('opencode_exit')!=0 or task.get('test_exit')!=0 or task.get('readers_joined') is not True or task.get('completed_tools',0)<1 or task.get('test_commands_completed',0)<1:raise ValueError('All protected normal 13 cases and terminal task completion required')
    if screen.get('passed') is not True or screen.get('stage')!='screen' or screen.get('model_role')!='ornith' or screen.get('profile')!=profile or screen.get('source_manifest_sha256')!=SCREEN_SOURCE_SHA or screen.get('workload_version')!='ornith-matched-classic-v3-tool-json-schema' or screen.get('inventory_receipt_sha256')!=inventory_sha:raise ValueError('Actual V3 candidate screen required')
    cases=screen.get('cases',[]);functional=screen.get('functional',[])
    if [x.get('case') for x in cases]!=['merge_intervals','parse_port'] or not all(x.get('passed') is True for x in cases) or [x.get('id') for x in functional]!=['ready','chat-record','chat-context-json','structured-tool-call','tool-result-final'] or not all(x.get('passed') is True for x in functional):raise ValueError('All coding and chat/tool quality gates required')
    for key in ('owned_native_reaped','guard_joined','ram_recovered','lease_free_after'):
        if screen.get(key) is not True:raise ValueError('Complete screen lifecycle required')
    if screen.get('guard_failures')!=[] or screen.get('native_returncode') not in (0,-15) or 'error' in screen:raise ValueError('Clean completed screen required')
    data=inventory.get('inventory',{});identity=data.get('file_identity')
    if inventory.get('passed') is not True or data.get('tensor_inventory_complete') is not True or data.get('tensor_payload_read') is not False or data.get('header_bytes_read',16*1024**2+1)>16*1024**2 or not isinstance(identity,dict) or identity.get('bytes')!=MODEL_BYTES:raise ValueError('Complete verified inventory required')
    actual=dict(bytes=agent['identity']['model_bytes'],device=agent['identity']['model_device'],inode=agent['identity']['model_inode'],mtime_ns=agent['identity']['model_mtime_ns'])
    if identity!=actual:raise ValueError('Inventory and task model identity differ; reproof required')
    if any(screen.get('identity',{}).get(key)!=agent['identity'].get(key) for key in ('model_bytes','model_device','model_inode','model_mtime_ns')):raise ValueError('Screen and task model identities differ')
    result=copy.deepcopy(profile)
    result.update(ready=True,model_alias=ALIAS,display_name=DISPLAY_NAME,release_channel='beta',prefix_cache={'enabled':False},validation=dict(kind='ornith-classic-beta-v1',normal_cases=13,normal13_sha256=normal_sha,screen_sha256=screen_sha,inventory_sha256=inventory_sha,root_reviewed_normal13_sha256=reviewed_normal_sha,policy_sha256=digest(POLICY),model_identity=identity))
    return result


def add_profile(runtime,profile):
    """Copy-on-write opt-in; Qwen profiles and selected default stay unchanged."""
    alias = profile.get('model_alias', ALIAS)
    if alias != ALIAS:raise ValueError('Unknown beta alias')
    if alias in runtime.get('profiles',{}):raise ValueError('Beta profile already exists; preserve it for review')
    result=copy.deepcopy(runtime);result.setdefault('profiles',{})[alias]=copy.deepcopy(profile)
    return result


def request_defaults(payload,path,alias=ALIAS):
    """Ornith's qualified request policy for text generation.

    Production qualification (classic and Rust, 13/13 OpenCode) ran with these
    defaults, so they stay; explicit client choices always survive.
    """
    if alias != ALIAS:raise ValueError('Unknown Ornith alias')
    paths={'/completion','/completions','/v1/completions','/chat/completions','/v1/chat/completions'}
    if path not in paths or not isinstance(payload,dict):return payload
    result=copy.deepcopy(payload)
    result.setdefault('temperature',0)
    result.setdefault('cache_prompt',True)
    if not any(key in result for key in ('max_tokens','max_completion_tokens','n_predict')):
        result['n_predict' if path=='/completion' else 'max_tokens']=2048
    if path in ('/chat/completions','/v1/chat/completions'):
        kwargs=result.setdefault('chat_template_kwargs',{})
        if isinstance(kwargs,dict):kwargs.setdefault('enable_thinking',False)
    return result


def opencode_fragment(alias=ALIAS):
    if alias != ALIAS:raise ValueError('Unknown Ornith alias')
    name = DISPLAY_NAME
    return {'providers':{'local-qwen':{'models':{alias:{'name':name,'type':'chat','family':'qwen','capabilities':{'tools':True,'input':['text'],'output':['text'],'temperature':True,'reasoning':{'supported':False}},'limit':{'context':16384,'input':12288,'output':2048},'compatibility':{'maxTokensField':'max_tokens'}}}}}}


def filesystem_identity(path):
    """Persistent filesystem/subvolume metadata, never boot-local st_dev.

    Reuses the reviewed reboot fix's findmnt identity without its header/model
    verification path: catalog polling never opens model bytes.
    """
    key = (str(path), hashlib.sha256(Path('/proc/self/mountinfo').read_bytes()).hexdigest())
    if key in _FILESYSTEM_CACHE:
        return dict(_FILESYSTEM_CACHE[key])
    result = subprocess.run(['findmnt', '--json', '--target', str(path), '--output',
                             'UUID,FSTYPE,FSROOT,OPTIONS'], capture_output=True,
                            text=True, timeout=3, check=True)
    rows = json.loads(result.stdout)['filesystems']
    if len(rows) != 1:
        raise ValueError('One model filesystem required')
    row = rows[0]
    if not all(row.get(k) for k in ('uuid', 'fstype', 'fsroot')):
        raise ValueError('Persistent model filesystem identity required')
    options = dict(item.split('=', 1) for item in row.get('options', '').split(',') if '=' in item)
    subvolume = options.get('subvolid') if row['fstype'] == 'btrfs' else None
    if row['fstype'] == 'btrfs' and (not subvolume or not subvolume.isdecimal()):
        raise ValueError('Btrfs subvolume identity required')
    value = dict(uuid=row['uuid'], fstype=row['fstype'], fsroot=row['fsroot'], subvolid=subvolume)
    if len(_FILESYSTEM_CACHE) >= 32:
        _FILESYSTEM_CACHE.clear()
    _FILESYSTEM_CACHE[key] = value
    return dict(value)


def stable_model_identity(path):
    """Metadata-only qualification identity stable across device-number changes."""
    source = Path(path)
    info = source.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError('Qualified model must be a regular file')
    canonical = source.resolve(strict=True)
    return dict(bytes=info.st_size, inode=info.st_ino, mtime_ns=info.st_mtime_ns,
                ctime_ns=info.st_ctime_ns, canonical_path=str(canonical),
                filesystem=filesystem_identity(canonical))


def effective_policy(profile):
    """Exact execution/template policy; declarative unused overrides are excluded."""
    cache = profile.get('prefix_cache', {})
    return {**{k: copy.deepcopy(profile.get(k)) for k in POLICY},
            'execution_policy_mode': profile.get('execution_policy_mode', 'stock'),
            'prefix_cache': {'enabled': cache.get('enabled'),
                             'max_bytes': cache.get('max_bytes', 3 * 1024**3)}}


def policy_digest(profile):
    return digest(effective_policy(profile))


def _flag(args, flag):
    values = []
    for i, item in enumerate(args):
        if item == flag:
            if i + 1 == len(args):raise ValueError('Missing candidate flag value')
            values.append(args[i + 1])
        elif item.startswith(flag + '='):
            values.append(item.split('=', 1)[1])
    if len(values) != 1:
        raise ValueError('One explicit candidate flag required: ' + flag)
    return values[0]


def require_candidate_profile(profile):
    alias = profile.get('model_alias')
    if alias != ALIAS:raise ValueError('Unknown beta candidate')
    sha, size = MODEL_SHA, MODEL_BYTES
    if (profile.get('sha256') != sha or profile.get('expected_bytes') != size or
            profile.get('server_sha256') not in candidate_engine_shas() or
            profile.get('backend_kind') not in (None, 'llama') or
            profile.get('backend') != 'cpu' or profile.get('gpu_layers') != 0):
        raise ValueError('Pinned original CPU model and allowlisted engine required')
    for key in ('context_size', 'threads'):
        if type(profile.get(key)) is not int or not 1 <= profile[key] <= (262144 if key == 'context_size' else 256):
            raise ValueError('Invalid candidate execution policy')
    for key in ('model_path', 'server_binary'):
        value = profile.get(key)
        if not isinstance(value, str) or not Path(value).is_absolute() or any(c in value for c in '\n\r\0'):
            raise ValueError('Absolute candidate paths required')
    args = profile.get('extra_args')
    if not isinstance(args, list) or not all(isinstance(x, str) for x in args):
        raise ValueError('Explicit candidate flags required')
    if _flag(args, '--device') != 'none' or '--no-op-offload' not in args or '--no-host' not in args:
        raise ValueError('CPU-only candidate required')
    if _flag(args, '--spec-type') not in ('none', 'draft-mtp'):
        raise ValueError('Unqualified candidate speculation')
    if json.loads(_flag(args, '--chat-template-kwargs')).get('enable_thinking') is not False:
        raise ValueError('Explicit nonthinking template policy required')
    if profile.get('execution_policy_mode', 'stock') not in ('stock', 'memlat'):
        raise ValueError('Unknown execution policy mode')
    cache = profile.get('prefix_cache')
    if not isinstance(cache, dict) or type(cache.get('enabled')) is not bool:
        raise ValueError('Explicit candidate prefix cache policy required')
    if type(cache.get('max_bytes', 3 * 1024**3)) is not int or not 0 < cache.get('max_bytes', 3 * 1024**3) <= 64 * 1024**3:
        raise ValueError('Bounded candidate prefix cache capacity required')
    if cache['enabled']:
        directory = cache.get('directory')
        if not isinstance(directory, str) or not Path(directory).is_absolute() or any(c in directory for c in '\n\r\0'):
            raise ValueError('Absolute candidate prefix directory required')
    if not isinstance(profile.get('environment'), dict):
        raise ValueError('Explicit candidate environment required')


def _check_candidate_report(profile, report):
    require_candidate_profile(profile)
    exact = dict(kind=CANDIDATE_KIND, model_alias=profile['model_alias'],
                 model_sha256=profile['sha256'], model_bytes=profile['expected_bytes'],
                 engine_sha256=profile['server_sha256'], policy_sha256=policy_digest(profile),
                 execution_policy_mode=profile.get('execution_policy_mode', 'stock'),
                 normal13_cases=13, opencode_exit=0, test_exit=0,
                 screen_cases=['merge_intervals', 'parse_port'],
                 functional_cases=['ready', 'chat-record', 'chat-context-json',
                                   'structured-tool-call', 'tool-result-final'], guard_failures=[])
    if any(report.get(k) != v or type(report.get(k)) is not type(v) for k, v in exact.items()):
        raise ValueError('Fresh actual matching screen/OpenCode policy evidence required')
    for key in ('screen_passed', 'normal13_passed', 'protected_unchanged',
                'native_reaped', 'all_children_reaped', 'readers_joined',
                'ram_recovered', 'lease_free_after', 'post_source_pins_passed'):
        if report.get(key) is not True:raise ValueError('Complete candidate quality and lifecycle required')
    for key in ('completed_tools', 'test_commands_completed'):
        if type(report.get(key)) is not int or report[key] < 1:
            raise ValueError('Actual completed OpenCode tools/tests required')
    # Controller guard-join and prefix hit/isolation observations are promotion
    # checks. Direct-engine quality runs cannot observe the controller guard;
    # do not require a fabricated observation before its first private launch.
    if 'error' in report or report.get('model_identity') != stable_model_identity(profile['model_path']):
        raise ValueError('Candidate model identity changed')
    if report['model_identity']['bytes'] != profile['expected_bytes']:
        raise ValueError('Candidate model size changed')


def qualify_candidate(profile, report_bytes, *, report_sha256, reviewed_report_sha256):
    """Prepare opt-in beta from ONE flat actual JSON report; never run/read models.

    report_sha256 hashes the exact supplied JSON bytes, not normalized JSON.
    The caller must obtain separate review of this exact completed report.
    Both stock and MEMLAT effective policies require their own report.
    """
    if (not isinstance(report_bytes, bytes) or len(report_bytes) > 8*1024**2 or
            not valid_sha(report_sha256) or reviewed_report_sha256 != report_sha256 or
            hashlib.sha256(report_bytes).hexdigest() != report_sha256):
        raise ValueError('Exact reviewed candidate report bytes required')
    report = json.loads(report_bytes)
    if not isinstance(report, dict) or set(report) & {'report_sha256', 'root_reviewed_report_sha256', 'report_content_sha256'}:
        raise ValueError('Flat candidate report without reserved proof fields required')
    _check_candidate_report(profile, report)
    result = copy.deepcopy(profile)
    result.update(ready=True, release_channel='beta',
                  display_name=DISPLAY_NAME)
    result['validation'] = {**report, 'report_sha256':report_sha256,
                            'root_reviewed_report_sha256':reviewed_report_sha256,
                            'report_content_sha256':digest(report)}
    return result


def _candidate_eligible(profile):
    try:
        if profile.get('ready') is not True or profile.get('release_channel') not in RELEASE_CHANNELS:
            return False
        proof = profile['validation']
        report = {k:v for k,v in proof.items() if k not in ('report_sha256', 'root_reviewed_report_sha256', 'report_content_sha256')}
        if (not valid_sha(proof.get('report_sha256')) or
                proof.get('root_reviewed_report_sha256') != proof['report_sha256'] or
                proof.get('report_content_sha256') != digest(report)):
            return False
        _check_candidate_report(profile, report)
        return True
    except (OSError, ValueError, KeyError, TypeError, AttributeError, subprocess.SubprocessError):
        return False
