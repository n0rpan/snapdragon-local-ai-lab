#!/usr/bin/env python3
"""Summarize fixed-workload legacy/tiled attention pairs; no model access."""
import argparse
import json
import statistics
from pathlib import Path


def summarize(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    groups = {}
    for row in rows:
        if row.get('run_failure') or not row.get('passed'):
            raise ValueError('failed attention arm: ' + row.get('label', '?'))
        if row['cgroup_swap_bytes'] or any(int(line.split()[1]) for line in row['cgroup_events'].splitlines()
                                          if line.split()[0] in ('oom', 'oom_kill', 'max')):
            raise ValueError('attention arm exceeded resource policy')
        kind = row['kind'].removeprefix('rust-attention-')
        context = row['prompt_tokens'] if kind == 'decode' else row['prefix_tokens']
        group = groups.setdefault((row['model_alias'], kind, context), {})
        group.setdefault(row['attn'], []).append(row)
    pairs = []
    for (model, kind, context), arms in groups.items():
        if set(arms) != {'legacy', 'tiled'}:
            raise ValueError('incomplete attention pair')
        if len(arms['legacy']) != len(arms['tiled']):
            raise ValueError('unbalanced attention repetitions')
        a, b = arms['legacy'][0], arms['tiled'][0]
        all_arms = arms['legacy'] + arms['tiled']
        identity = 'generated_ids' if kind == 'decode' else 'first_token'
        if any(r[identity] != a[identity] for r in all_arms):
            raise ValueError('attention output mismatch')
        for key in ('source_commit', 'engine_sha256', 'model_sha256', 'cpus', 'threads', 'prompt_tokens'):
            if any(r[key] != a[key] for r in all_arms):
                raise ValueError('unmatched ' + key)
        metric = 'tok_s' if kind == 'decode' else 'tok_s_median'
        samples = {mode: [r[metric] for r in values] for mode, values in arms.items()}
        medians = {mode: statistics.median(values) for mode, values in samples.items()}
        peaks = [max(v for r in arms[m] for k, v in r['whole_process_sensor_max'].items()
                     if ('thermal_zone' in k and k.endswith('/temp'))) / 1000 for m in ('legacy', 'tiled')]
        pairs.append(dict(model=model, kind=kind, context=context,
                          legacy_tok_s=medians['legacy'], tiled_tok_s=medians['tiled'],
                          speedup=medians['tiled'] / medians['legacy'],
                          runs_per_arm=len(arms['legacy']), tok_s_samples=samples,
                          whole_arm_major_faults={m: [r.get('child_major_faults') for r in rs] for m, rs in arms.items()},
                          whole_arm_cpu_frequency_khz={m: [r.get('cpu_frequency_khz') for r in rs] for m, rs in arms.items()},
                          identical_output=True, equality_scope=identity,
                          threads=a['threads'], cpus=a['cpus'],
                          legacy_peak_c=peaks[0], tiled_peak_c=peaks[1],
                          legacy_rss_peak_gib=max(r['memory_kb']['VmHWM'] for r in arms['legacy']) / 1048576,
                          tiled_rss_peak_gib=max(r['memory_kb']['VmHWM'] for r in arms['tiled']) / 1048576,
                          source_commit=a['source_commit'], engine_sha256=a['engine_sha256'],
                          model_sha256=a['model_sha256']))
    return dict(kind='rust-attention-fixed-workload-screen', pairs=pairs,
                production_changed=False,
                scope='Each arm decode metric is the median of 64 token steps after eight warmup steps; prefill is the median of three timed repetitions after one warmup. Repeated arms are summarized as the median of their metrics. Prompt prefill is batched. Not a full-task or controller qualification. No automatic promotion.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('receipts', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.write_text(json.dumps(summarize(args.receipts), indent=2) + '\n')
