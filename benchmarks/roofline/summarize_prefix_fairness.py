#!/usr/bin/env python3
"""Matched cold/warm-prefix sessions with installed classic profiles; metadata only."""
import argparse
import json
from pathlib import Path
from summarize_integrated_server import summarize_run


def summarize(root, upgrade):
    results = []
    for model in ('ornith', 'qwen'):
        cpp = {}
        for state in ('cold', 'warm'):
            path = root/f'controller-integrated-{model}-llama-production-prefix-{state}'/'qualification.json'
            raw = json.loads(path.read_text())
            row = summarize_run(path)
            if not raw.get('production_classic') or not raw.get('prefix_store_enabled') or raw.get('normal13_cases') != 13:
                raise ValueError('passing production-settings prefix control required')
            requests = [r for r in raw['request_timings'] if r['kind'] == 'request' and r.get('origin','client') == 'client']
            prepares = [r for r in raw['request_timings'] if r['kind'] == 'prefix_prepare']
            if len(prepares) != len(requests): raise ValueError('one prefix preparation per client request required')
            row['first_request'] = requests[0]
            row['first_prefix_prepare_ms'] = prepares[0]['milliseconds']
            row['first_prompt_pipeline_ms'] = prepares[0]['milliseconds'] + requests[0]['timings']['prompt_ms']
            row['phases']['prefix_prepare_seconds'] = sum(r['milliseconds'] for r in prepares)/1000
            row['phases']['client_tools_other_seconds'] -= row['phases']['prefix_prepare_seconds']
            row['prefix_stats'] = raw['status_after']['prefix_cache']
            if state == 'cold' and row['prefix_stats']['primes'] < 1: raise ValueError('cold store must prime')
            if state == 'warm' and row['prefix_stats']['restores'] < 1: raise ValueError('warm store must restore')
            config = json.loads((path.parent/'runtime.json').read_text())
            row['configured_extra_args'] = config['extra_args']
            store = root/'prefix-store-production-qualification'/model
            files = [p for p in store.iterdir() if p.is_file()]
            row['prefix_disk_bytes_after_pair'] = sum(p.stat().st_size for p in files)
            cpp[state] = row
        rust = next(r for r in upgrade['prefix_session_pairs'] if r['model']==model)
        comparison = {}
        for state in ('cold','warm'):
            a,b = cpp[state]['protected_task_seconds'],rust[state]['protected_task_seconds']
            comparison[state] = dict(cpp_seconds=a,rust_seconds=b,less_task_time_percent=100*(1-b/a))
        results.append(dict(model=model,cpp=cpp,rust=rust,comparison=comparison))
    return dict(kind='production-prefix-fairness',models=results,production_changed=False,samples_per_engine_state_model=1,
                scope='Same synthetic OpenCode task; three short model warmups, then a fresh native process per timed session. First session populates an isolated store; second restores it. Uses actual production classic profile settings; Qwen classic is 1bd3e439/MTP2, Ornith acd47ff1/MTP-off, Rust 649ac806/MTP-off. Experimental aliases and fresh cache directories isolate user caches. Prefix-cold is not SSD-cold model residency. Classic prefix_prepare includes the primer/save or restore before the visible backend request: do not compare its tiny native prompt_ms alone with Rust whole prompt_ms. Whole task includes startup and preparation. Native/KleidiAI trajectories differ; all protected checks must pass. Not a decode-only benchmark.')


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',type=Path,required=True)
    p.add_argument('--upgrade',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    result=summarize(a.runs,json.loads(a.upgrade.read_text()))
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps([{k:v for k,v in r.items() if k in ('model','comparison')} for r in result['models']]))
