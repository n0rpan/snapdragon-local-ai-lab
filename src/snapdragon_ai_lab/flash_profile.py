"""Numeric-only historical attribution; never starts an engine or reads user content."""
import math
from pathlib import Path
import os
import csv


def classify_cpu_symbol(symbol):
    """Conservative leaf-PC category; this does not isolate inlined operations."""
    if 'ggml_vec_dot_' in symbol:
        return 'quantized_dot_including_unpack'
    if 'dequantize_row_' in symbol:
        return 'standalone_dequantization'
    if 'ggml_compute_forward_flash_attn' in symbol:
        return 'attention'
    if 'ggml_compute_forward_mul_mat' in symbol:
        return 'matrix_kernel'
    if 'ggml_compute_forward_' in symbol:
        return 'other_ggml_kernel'
    return 'other_or_unresolved'


def summarize_io_trace(path, max_bytes=512*1024**2):
    """Stream numeric attribution CSV; lane latency is deliberately nonadditive."""
    layers = {}
    frames = []
    read_lines = 0
    with Path(path).open() as handle:
        while line := handle.readline(4097):
            read_lines += len(line.encode())
            if read_lines > max_bytes or len(line) > 4096:
                raise ValueError('trace size limit')
            row = next(csv.reader([line.strip()]))
            if not row:
                continue
            if row[0] == '# frame':
                values = [int(x) for x in row[1:]]
                if len(values) != 5 or values[1] not in (0,1,2) or not 0 <= values[3] <= values[4] or len(frames) >= 65536:
                    raise ValueError('invalid frame')
                frames.append(dict(zip(('turn','phase','step','begin_ns','end_ns'),values)))
                continue
            if row[0] == '# cache':
                values = [int(x) for x in row[1:]]
                if len(values) != 6 or values[1] not in (0,1,2) or not 0 <= values[5] <= values[4]:
                    raise ValueError('invalid cache counts')
                turn,phase,step,layer,lookups,hits=values
                key=(turn,phase,layer)
                if key not in layers and len(layers) >= 16384:
                    raise ValueError('layer aggregate limit')
                item=layers.setdefault(key,dict(lookups=0,hits=0,requested_bytes=0,read_bytes=0,lane_busy_ns=0))
                item['lookups']+=lookups;item['hits']+=hits
                continue
            if row[0].startswith('#') or row[0] == 'turn':
                continue
            values = [int(x) for x in row]
            if len(values) != 12:
                raise ValueError('invalid I/O row')
            turn,phase,step,layer,expert,projection,lane,spec,offset,requested,read,latency=values
            if phase not in (0,1,2) or not 0 <= requested <= read or min(offset,latency) < 0:
                raise ValueError('invalid read accounting')
            if spec:
                raise ValueError('Phase 1 requires speculation off')
            key=(turn,phase,layer)
            if key not in layers and len(layers) >= 16384:
                raise ValueError('layer aggregate limit')
            item=layers.setdefault(key,dict(lookups=0,hits=0,requested_bytes=0,read_bytes=0,lane_busy_ns=0))
            item['requested_bytes']+=requested;item['read_bytes']+=read;item['lane_busy_ns']+=latency
    return {'schema_version':1,'frames':frames,'layers':[
        dict(turn=k[0],phase=k[1],layer=k[2],**v,
             hit_percent=100*v['hits']/v['lookups'] if v['lookups'] else None)
        for k,v in sorted(layers.items())],
        'lane_busy_time_is_wall_additive':False,
        'bytes_are_application_aligned_reads_not_physical_nand':True}


def cpu_counters(text):
    """Host per-core counters; guest times are already included in user/nice."""
    result = {}
    for line in text.splitlines():
        fields = line.split()
        if not fields or not fields[0].startswith('cpu') or not fields[0][3:].isdigit():
            continue
        values = [int(x) for x in fields[1:9]]
        if len(values) < 4 or any(x < 0 for x in values):
            raise ValueError('invalid CPU counters')
        idle = values[3] + (values[4] if len(values) > 4 else 0)
        result[fields[0]] = (sum(values), idle)
    return result


def cpu_utilization(before, after):
    result = {}
    for core in before.keys() & after.keys():
        total = after[core][0] - before[core][0]
        idle = after[core][1] - before[core][1]
        if total < 0 or idle < 0 or idle > total:
            raise ValueError('CPU counter reset')
        result[core] = None if total == 0 else 100*(total-idle)/total
    return result


def process_identity(pid, proc=Path('/proc')):
    if type(pid) is not int or pid <= 0:
        raise ValueError('positive owned child PID required')
    fields = (proc/str(pid)/'stat').read_text().rsplit(')', 1)[1].split()
    return int(fields[19])


def sample_owned_child(pid, start_ticks, proc=Path('/proc')):
    """Caller supplies its Popen PID/start identity; read metadata only, never cmdline."""
    if process_identity(pid, proc) != start_ticks:
        raise ValueError('child PID was reused')
    hz = os.sysconf('SC_CLK_TCK')
    threads = {}; users = {}; systems = {}
    for path in (proc/str(pid)/'task').glob('[0-9]*/stat'):
        try:
            fields = path.read_text().rsplit(')', 1)[1].split()
            threads[path.parent.name] = (int(fields[11])+int(fields[12]))/hz
            users[path.parent.name] = int(fields[11])/hz
            systems[path.parent.name] = int(fields[12])/hz
        except FileNotFoundError:
            continue  # Thread exited during snapshot.
    io = {}
    for line in (proc/str(pid)/'io').read_text().splitlines():
        key, value = line.split(':', 1)
        if key in ('read_bytes', 'write_bytes', 'syscr', 'rchar'):
            io[key] = int(value)
    cores = cpu_counters((proc/'stat').read_text())
    process_fields = (proc/str(pid)/'stat').read_text().rsplit(')',1)[1].split()
    if process_identity(pid, proc) != start_ticks:
        raise ValueError('child PID changed during snapshot')
    return {'host_core_counters': cores, 'owned_thread_cpu_seconds': threads,
            'owned_thread_user_cpu_seconds':users,'owned_thread_system_cpu_seconds':systems,
            'owned_process_major_faults':int(process_fields[9]),
            'owned_process_minor_faults':int(process_fields[7]),
            'owned_process_io': io, 'child_start_ticks': start_ticks}


def number(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError('finite nonnegative measurement required')
    return value


def summarize(receipt):
    rows = receipt['native_requests']
    if not rows:
        raise ValueError('native requests required')
    wall = number(receipt['timings']['total_wall_seconds'])
    prefill = decode = tokens = wait = residual = management = lane = read = 0
    for row in rows:
        n, rate = number(row['tokens']), number(row['tok_s'])
        if not n or not rate:
            raise ValueError('nonempty completed generation required')
        tokens += n
        prefill += number(row['prefill_s'])
        decode += n / rate
        wait += n * number(row['flash_wait_s_tok'])
        residual += n * number(row['compute_s_tok'])
        management += n * number(row['mgmt_s_tok'])
        lane += n * number(row['io_s_tok'])
        read += number(row['read_mib']) * 1024**2
    if abs(decode - wait - residual - management) > max(.01, decode * .001):
        raise ValueError('decode accounting mismatch')
    if wall < prefill + decode - .01:
        raise ValueError('native time exceeds total')
    return dict(schema_version=1, scope='historical_summary_not_new_instrumented_run',
                total_wall_seconds=wall, request_count=len(rows), generated_tokens=tokens,
                prefill_wall_seconds=prefill, decode_wall_seconds=decode,
                other_wall_seconds=wall-prefill-decode,
                decode_exposed_ssd_wait_seconds=wait,
                decode_compute_residual_seconds=residual,
                decode_management_seconds=management,
                decode_io_lane_busy_seconds_nonadditive=lane,
                decode_bytes_read=read, decode_bytes_per_token=read/tokens,
                aggregate_decode_tokens_per_second=tokens/decode,
                direct_prefill_compute_seconds=None, direct_decode_compute_seconds=None,
                per_layer_hit_rate=None, raw_nvme_bytes_per_second=None,
                per_core_utilization=None,
                attribution_complete=False)
