#!/usr/bin/env python3
"""Summarize private synthetic qualification runs without exporting text or IDs."""
import argparse
import json
from pathlib import Path
from statistics import median


def summarize_run(path):
    d=json.loads(path.read_text())
    if not d.get('passed') or not d.get('cleanup_passed'):
        raise ValueError('completed passing run required: '+path.parent.name)
    result={k:d.get(k) for k in ('model','arm','engine_sha256','source_commit','protected_task_seconds',
        'model_calls','uncached_prompt_tokens','generated_tokens','cached_prompt_tokens',
        'cgroup_memory_peak','cgroup_swap_bytes','native_process_peak_bytes','diagnostic_build','spec_off')}
    result['name']=path.parent.name
    result['native_reference_cases_passed']=sum(x['passed'] for x in d.get('oracle_cases',[]) if x.get('exactness_gate',d['arm']=='oxe'))
    records=d.get('request_timings',[])
    req=[x for x in records if x['kind']=='request']
    if req:
        if len(req)!=d['model_calls'] or any(not x.get('timings') for x in req):
            raise ValueError('complete per-request timers required')
        prompt=sum(x['timings']['prompt_ms'] for x in req)/1000
        decode=sum(x['timings']['predicted_ms'] for x in req)/1000
        backend=sum(x['backend_wall_ms'] for x in req)/1000
        startup=sum(x['milliseconds'] for x in records if x['kind']=='startup')/1000
        result['phases']=dict(prompt_seconds=prompt,decode_seconds=decode,
            processed_prompt_tokens=sum(x['timings']['prompt_n'] for x in req),
            reused_prompt_tokens=sum(x['timings']['cache_n'] for x in req),
            weighted_reported_decode_tok_s=sum(x['timings']['predicted_per_second']*x['timings']['predicted_ms']/1000 for x in req)/decode,
            emitted_tokens_per_decode_second=d['generated_tokens']/decode,
            backend_http_overhead_seconds=backend-prompt-decode,startup_seconds=startup,
            client_tools_other_seconds=d['protected_task_seconds']-backend-startup)
        native=d.get('native_timings',[])
        if native:
            result['phases'].update(checkpoint_seconds=sum(x.get('checkpoint_ms',0) for x in native)/1000,
                cache_prepare_seconds=sum(x.get('cache_prepare_ms',0) for x in native)/1000,
                mtp_seed_seconds=sum(x.get('seed_ms',0) for x in native)/1000,
                mtp_main_prefill_seconds=sum(x.get('main_ms',0) for x in native)/1000)
        result['requests']=[{k:v for k,v in x.items() if k in ('timings','usage','cached_tokens','backend_wall_ms','request_body_sha256')} for x in req]
        result['native_diagnostics']=native
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--runs',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    groups=[]
    for model in ['ornith','qwen']:
        for arm in ['llama','oxe']:
            prefix='controller-integrated-'+model+'-'+arm
            names=[prefix+('-matched-warmup' if arm=='llama' else ''),prefix+'-repeat2',prefix+'-repeat3']
            rows=[summarize_run(a.runs/name/'qualification.json') for name in names]
            groups.append(dict(model=model,arm=arm,samples=3,passed=True,
                task_seconds_median=median(x['protected_task_seconds'] for x in rows),
                task_seconds_range=[min(x['protected_task_seconds'] for x in rows),max(x['protected_task_seconds'] for x in rows)],
                phase_samples=sum('phases' in x for x in rows),runs=rows))
    comparison=[]
    for model in ['ornith','qwen']:
        before=next(x for x in groups if x['model']==model and x['arm']=='llama')['task_seconds_median']
        after=next(x for x in groups if x['model']==model and x['arm']=='oxe')['task_seconds_median']
        comparison.append(dict(model=model,llama_seconds_median=before,rust_seconds_median=after,less_task_time_percent=100*(1-after/before)))
    extra=summarize_run(a.runs/'controller-integrated-qwen-oxe-diagnostic-off/qualification.json')
    report=dict(scope='synthetic-protected-OpenCode-normal13',promotion=False,groups=groups,comparison=comparison,qwen_mtp_off=extra,
        limitations=['One protected repair task; native and KleidiAI trajectories differ.','First sample lacks phase timings; phase analysis uses the two diagnostic repeats.','Same three 128-token warmups followed by fresh process, external prefix-file restore off; OS page cache uncontrolled.','Checkpoint and MTP seed timers are subsets of prompt/decode, never additive overhead.','Backend overhead includes HTTP, template/tokenizer and serialization; client/tools residual is separate.','Reported decode rates retain each server first-token convention; emitted/time is also provided.','Cgroup memory counts can exclude shared file pages charged earlier; process RSS supplied where sampled.'])
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(comparison))

if __name__=='__main__':main()
