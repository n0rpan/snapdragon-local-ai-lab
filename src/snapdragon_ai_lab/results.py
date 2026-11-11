"""Allowlist public screen metrics; discard raw responses and local paths."""
import math
import re

CONFIG_KEYS={'threads','batch_threads','batch_size','ubatch_size','cpu_mask','group','id'}
TIMING_KEYS={'cache_n','prompt_n','prompt_ms','prompt_per_second','predicted_n','predicted_ms','predicted_per_second'}

def number(value):
    return value if type(value) in (int,float) and math.isfinite(value) else None

def label(value):
    if not isinstance(value,str) or re.fullmatch(r'[A-Za-z0-9_.-]{1,100}',value) is None:
        raise ValueError('Public labels must be short identifiers without paths')
    return value

def export_screen(source):
    identities={}
    for key in ('target_sha256','binary_sha256'):
        value=source.get(key)
        if value is not None:
            if not isinstance(value,str) or re.fullmatch(r'[0-9a-f]{64}',value) is None:
                raise ValueError('Engine and model identities must be SHA256 digests')
            identities[key]=value
    rows=[]
    for raw in source.get('rows',[]):
        config={}
        for key,value in raw.get('configuration',{}).items():
            if key not in CONFIG_KEYS:continue
            config[key]=label(value) if isinstance(value,str) else number(value)
        timings={}
        for name,values in raw.get('case_timings',{}).items():
            timings[label(name)]={k:number(v) for k,v in (values or {}).items() if k in TIMING_KEYS}
        rows.append({'configuration':config,'passed':raw.get('passed') is True,
                     'power_conditions_verified':False,
                     'request_seconds_total':number(raw.get('request_seconds')),
                     'load_seconds':number(raw.get('load_seconds')),
                     'server_exit':number(raw.get('server_exit')),'case_timings':timings})
    return {'schema_version':1,'workload':'synthetic-code-screen-v1','preliminary':True,
            'identities':identities,
            'scope':'Two synthetic code requests per condition. Request timing excludes process load and independent checks; this is not a finished-agent benchmark.',
            'comparison_caveat':'The first control predates a reboot and charger change. Verify finalists under matched power conditions before making release claims.',
            'rows':rows}

def export_controller_gate(source):
    """Allowlist a real synthetic coding gate; never retain agent output/text."""
    result=source.get('agent_result',{})
    warm=result.get('warm_session',{})
    profile=result.get('profile',{})
    identities={}
    for key in ('server_sha256','target_model_sha256'):
        value=profile.get(key)
        if not isinstance(value,str) or re.fullmatch(r'[0-9a-f]{64}',value) is None:
            raise ValueError('Verified engine/model SHA256 identities are required')
        identities[key]=value
    cold=number(result.get('verified_completion_seconds'));reloaded=number(warm.get('seconds'))
    restored=warm.get('controller_cache',{}).get('restores')
    new=number(warm.get('first_turn_new_tokens'));cached=number(warm.get('first_turn_cached_tokens'))
    reuse=cached/(new+cached) if new is not None and cached is not None and min(new,cached)>=0 and new+cached>0 else None
    checks={'cold_repair_passed':result.get('passed') is True,
            'reloaded_repair_passed':warm.get('passed') is True,
            'protected_test_unchanged':warm.get('test_file_unchanged') is True,
            'full_backend_reload':warm.get('full_backend_reload') is True,
            'cached_prefix_reused':warm.get('prefix_reused') is True and reuse is not None and reuse>=.8,
            'controller_restore_recorded':type(restored) is int and restored>0}
    return {'schema_version':1,'workload':'synthetic-port-repair-controller-cache-v1',
        'passed':source.get('passed') is True and all(checks.values()) and cold is not None and cold>0 and reloaded is not None and reloaded>0,
        'checks':checks,'identities':identities,
        'cold_verified_seconds_including_load':cold,'reloaded_verified_seconds_including_load':reloaded,
        'first_turn_new_tokens':number(warm.get('first_turn_new_tokens')),
        'first_turn_cached_tokens':number(warm.get('first_turn_cached_tokens')),
        'first_turn_cache_fraction':reuse,
        'configuration':{k:number(profile.get(k)) for k in ('threads','batch_threads','batch_size','ubatch_size')},
        'power_conditions_verified':False,
        'scope':'Two fresh archived OpenCode repairs, original bug reset between them; second follows full unload/reload. One fixed fixture, no broad quality claim. Cold-to-reload benefit is caching, not a kernel speedup.'}
