#!/usr/bin/env python3
"""Metadata-only summary of the integrated Rust upgrade qualification."""
import argparse
import json
from pathlib import Path
from statistics import median
from summarize_integrated_server import summarize_run


def load(path):
    row = summarize_run(path)
    raw = json.loads(path.read_text())
    if not raw.get('spec_off') or row['native_reference_cases_passed'] != 3:
        raise ValueError('MTP-off and three native fixtures required')
    if raw.get('normal13_cases') != 13 or raw.get('native_process_peak_bytes', {}).get('VmSwap'):
        raise ValueError('protected checks and zero model swap required')
    req = [v for v in raw['request_timings'] if v['kind'] == 'request']
    row['first_request'] = {k: req[0][k] for k in ('timings','request_body_sha256','sampling','backend_first_useful_delta_ms') if k in req[0]}
    row['prefix_store_enabled'] = raw.get('prefix_store_enabled', False)
    row['actual_environment'] = raw['actual_process'].get('environment', {})
    if row['engine_sha256'].startswith('649ac806') and row['actual_environment'].get('OXE_ATTN_TILED_MIN_KV') != '0':
        raise ValueError('actual candidate threshold 0 required')
    for request in req:
        value = request.get('backend_first_useful_delta_ms')
        if value is not None and not 0 <= value <= request['backend_wall_ms']:
            raise ValueError('invalid first-useful-delta timer')
    return row


def summarize(root):
    groups, comparison, prefix = [], [], []
    for model in ('ornith', 'qwen'):
        by_variant = {}
        for variant in ('old', 'next'):
            rows = []
            for trial in (1, 2, 3):
                tag = f'upgrade-{variant}-r{trial}'
                if model == 'ornith' and variant == 'old' and trial == 1: tag += '-retry1'
                rows.append(load(root / f'controller-integrated-{model}-oxe-{tag}' / 'qualification.json'))
            result = dict(model=model,variant=variant,runs=rows,samples=3,
                          task_seconds_median=median(r['protected_task_seconds'] for r in rows),
                          phase_medians={key:median(r['phases'][key] for r in rows)
                                         for key in ('prompt_seconds','decode_seconds','emitted_tokens_per_decode_second')})
            groups.append(result)
            by_variant[variant] = result
        a, b = (by_variant[v]['task_seconds_median'] for v in ('old', 'next'))
        comparison.append(dict(model=model,old_seconds_median=a,next_seconds_median=b,less_task_time_percent=100*(1-b/a)))
        cold, warm = [load(root/f'controller-integrated-{model}-oxe-upgrade-prefix-{kind}'/'qualification.json') for kind in ('cold','warm')]
        if not cold['prefix_store_enabled'] or not warm['prefix_store_enabled']:
            raise ValueError('persistent prefix arm must be enabled')
        prefix.append(dict(model=model,cold=cold,warm=warm,
                           first_prompt_seconds_saved=(cold['first_request']['timings']['prompt_ms']-warm['first_request']['timings']['prompt_ms'])/1000,
                           warm_reuse_verified=warm['first_request']['timings']['cache_n']>0 and warm['first_request']['timings']['prompt_n']<cold['first_request']['timings']['prompt_n']))
    return dict(kind='integrated-rust-upgrade-qualification',groups=groups,comparison=comparison,prefix_session_pairs=prefix,
                production_changed=False,
                scope='Three protected-task trials per arm/model, same boot, MTP off. Same three native warmups then a fresh process per task. Candidate threshold 0 verified in process environment. Persistent prefix off for paired trials; isolated prefix store for two extra fresh-process sessions per model. First-useful SSE delta is an API-observed latency, not a kernel first-token timer. OS file residency is uncontrolled. One synthetic repair task is not a broad quality benchmark. Initial pre-inference lease refusal is retained separately, excluded from task samples.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args = p.parse_args()
    result = summarize(args.runs)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['comparison']))
