#!/usr/bin/env python3
"""LIVE forced-compaction check for real OpenCode 2.2.7 with the MoE launch preset.

Drives `opencode run --standalone` against the local controller, with the overlay produced by
runtime/launch_local_plan.py for a MoE model (moe or ornith). The MoE models' `input` limit is
cut to 16K or 32K for this run only, so OpenCode's auto compaction fires after a few turns.
Context (131072), output (16384), compaction.keep (8000) and the small-model isolation are the
production preset's values.

Why each CLI piece (checked in the 2.2.7 help and its bundled source, not guessed):
- `--standalone`: a private server started by this process, so it reads this process's
  OPENCODE_CONFIG_CONTENT. The default background service would ignore the overlay.
- `run --format json`: one JSON event per line, each with a top-level `sessionID`;
  `--session <id>` continues that session and `--archived` keeps it out of session lists.
- `models`: prints the server's `model.list`. That is the enabled (available) catalog, the same
  list the TUI model picker uses.
- `session export <id>`: the session as JSON. A compaction is a part
  {type: "compaction", auto, overflow?, tail_start_id?}; step-finish parts carry token counts.
- Trigger: OpenCode with no compaction.buffer compacts once the estimate reaches
  o - max(floor(0.15*o), 16000 if o >= 32000 else 0), where o = min(input, context).
  16K -> 13927, 32K -> 16768. keep.tokens bounds the verbatim tail kept after the summary.

Assertions (numbers and labels only; prompts are synthetic filler, nothing is printed from replies):
 1. isolation: `opencode models` lists both MoE models and none of the small-context models;
 2. compaction: an auto (not overflow) compaction part appears before --max-turns;
 3. threshold: it did not fire early (last measured context + next prompt estimate >= trigger
    - slack) or late (no earlier turn measured >= trigger + slack);
 4. keep: the estimated verbatim tail (chars/4 from tail_start_id) is <= keep * 1.1;
 5. cache (controller ledger, rows since the start): a `compaction` turn, then the first
    `post-compaction` row has compact_layout 'ok', cache_n >= --min-prefix-hit and a
    miss_class other than 'unknown'; the next row is 'append-ok' (C3 holds again); no row in
    the window is 'unknown'.

--mode fast|deep (default fast, the production default since O-105) picks the preset the window
is cut from. --controller-url (loopback only, default the installed 8092) is used for both the
OpenCode provider and the status preflight; every request carries X-Local-AI-Origin: lab (REAL-1).
--api-key-env names an environment variable with a fresh token; the key is never printed or
written to the receipt. An inherited OPENCODE_CONFIG_CONTENT is kept as the base of the overlay.

--dry-run prints the overlay (key redacted), trigger and exact commands and runs nothing. A live run needs the
quiet hardware window. The chosen MoE model must already be selected in Local AI, and no other
client may use the controller during the run.
"""
import argparse
import importlib.util
import ipaddress
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
OPENCODE_VERSION = '2.2.7'
WINDOWS = {'16k': 16384, '32k': 32768}
PROVIDER = 'local-qwen'
CHARS_PER_TOKEN = 4
KEEP_SLACK = 0.10
# REAL-1: every request this recipe causes is lab traffic, never an organic ledger row.
ORIGIN_HEADER, ORIGIN_LAB = 'X-Local-AI-Origin', 'lab'
DEFAULT_CONTROLLER_URL = 'http://127.0.0.1:8092/v1'  # the installed controller
REDACTED = '<redacted>'


def controller_url(url):
    """A loopback http(s) controller base URL ending in /v1, normalised; ValueError otherwise.
    A pre-install gate points this at a test controller (e.g. 127.0.0.1:18092) so it can never
    reach the installed one by accident."""
    if not isinstance(url, str) or any(c in url for c in '\r\n\0 '):
        raise ValueError('controller_url_invalid')
    parts = urllib.parse.urlsplit(url)
    host = parts.hostname or ''
    try:
        loopback = host == 'localhost' or ipaddress.ip_address(host).is_loopback
    except ValueError:
        loopback = False
    if (parts.scheme not in ('http', 'https') or not loopback or parts.username or parts.password
            or parts.query or parts.fragment or parts.port is None):
        raise ValueError('controller_url_must_be_loopback_with_port')
    path = parts.path.rstrip('/') or '/v1'
    if path != '/v1':
        raise ValueError('controller_url_path_must_be_v1')
    netloc = f'[{host}]:{parts.port}' if ':' in host else f'{host}:{parts.port}'
    return urllib.parse.urlunsplit((parts.scheme, netloc, '/v1', '', ''))


def api_key_from(env_name, environment=None):
    """The optional fresh token from the named environment variable (never printed)."""
    if not env_name:
        return None
    value = (os.environ if environment is None else environment).get(env_name)
    if not value:
        raise ValueError('api_key_env_not_set')
    return value


def load_launcher(path=ROOT / 'src/snapdragon_ai_lab/runtime/launch_local_plan.py'):
    spec = importlib.util.spec_from_file_location('compaction_live_launcher', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def opencode_trigger(input_limit, context=None):
    o = min(input_limit, context) if input_limit and context else (input_limit or context)
    if not o or o <= 0:
        raise ValueError('invalid_limit')
    return o - max(int(o * 0.15), 16000 if o >= 32000 else 0)


def model_id(launcher, model):
    return launcher.CLASSIC_MODELS[model].split('/', 1)[1]


def build_overlay(launcher, model, window, mode='fast', base=None, url=DEFAULT_CONTROLLER_URL, api_key=None):
    """Production MoE preset (`mode`, default fast as in production, O-105) merged onto the
    inherited OPENCODE_CONFIG_CONTENT `base` (as the launcher does), with the MoE models' input
    cut to the test window, the provider pointed at `url` and every request marked lab (REAL-1)."""
    if model not in launcher.MOE_PRESET_MODELS:
        raise ValueError('moe_model_required')
    if window not in WINDOWS:
        raise ValueError('unsupported_window')
    if mode not in launcher.OPENCODE_MODE_INPUT:
        raise ValueError('unsupported_opencode_mode')
    environment = {} if not base else {'OPENCODE_CONFIG_CONTENT': base}
    overlay = json.loads(launcher.environment_for_classic_model(environment, model, mode)['OPENCODE_CONFIG_CONTENT'])
    provider = overlay['providers'][PROVIDER]
    settings = provider.setdefault('settings', {})
    if not isinstance(settings, dict):
        raise ValueError('invalid_provider_settings')
    settings['baseURL'] = controller_url(url)
    headers = settings.setdefault('headers', {})
    if not isinstance(headers, dict):
        raise ValueError('invalid_provider_headers')
    headers[ORIGIN_HEADER] = ORIGIN_LAB
    if api_key is not None:
        settings['apiKey'] = api_key
    models = provider['models']
    for key in launcher.MOE_PRESET_MODELS:
        models[model_id(launcher, key)]['limit']['input'] = WINDOWS[window]
    for small in launcher.SMALL_CONTEXT_MODEL_IDS:
        if models.get(small, {}).get('disabled') is not True:
            raise ValueError('isolation_missing')
    if overlay.get('compaction', {}).get('keep', {}).get('tokens') != launcher.MOE_KEEP_TOKENS:
        raise ValueError('keep_missing')
    return overlay


def filler_prompt(turn, tokens):
    words = ('alpha', 'cobalt', 'harbor', 'lattice', 'meadow', 'quartz', 'river', 'signal', 'timber', 'vector')
    body, i = [], 0
    while sum(len(w) + 1 for w in body) < tokens * CHARS_PER_TOKEN:
        body.append(f'{words[(turn * 7 + i) % len(words)]}{i % 97}')
        i += 1
    return (f'Turn {turn}. The notes below are filler for a context test. Do not use any tools. '
            f'Reply with exactly: ACK {turn}\n\n' + ' '.join(body))


def commands(opencode, model_ref, session=None, prompt='<prompt>', title='compaction-live'):
    run = [opencode, 'run', '--standalone', '--format', 'json', '-m', model_ref]
    run += ['--session', session] if session else ['--archived', '--title', title]
    return {'version': [opencode, '--version'], 'models': [opencode, 'models', '--standalone'],
            'run': run + [prompt], 'export': [opencode, 'session', 'export', '--standalone', session or '<session>']}


def session_from_events(stdout):
    session, errors = None, []
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if not isinstance(event, dict):
            continue
        if event.get('type') == 'error':
            errors.append((event.get('error') or {}).get('message', 'error'))
        if not session and isinstance(event.get('sessionID'), str) and event['sessionID']:
            session = event['sessionID']
    if errors:
        raise RuntimeError('opencode_run_error: ' + '; '.join(map(str, errors))[:300])
    if not session:
        raise RuntimeError('no_session_id')
    return session


def walk(value):
    """Every dict in document order (depth first)."""
    if isinstance(value, dict):
        yield value
        for item in value.values():
            yield from walk(item)
    elif isinstance(value, list):
        for item in value:
            yield from walk(item)


def measured(tokens):
    cache = tokens.get('cache') or {}
    return sum(int(x or 0) for x in (tokens.get('input'), cache.get('read'), cache.get('write'),
                                      tokens.get('output'), tokens.get('reasoning')))


def analyse_export(document):
    """Steps (measured context per finished step), compaction parts and per-message text sizes,
    all in document order. Only sizes leave this function."""
    events, message_order = [], []
    messages = document.get('messages', []) if isinstance(document, dict) else []
    # OpenCode 2.2.7 exports typed messages directly. The older parts export uses
    # step-finish + compaction(auto); its shape is retained below for compatibility.
    if any(isinstance(m, dict) and m.get('type') in ('user', 'assistant', 'compaction') for m in messages):
        for node in messages:
            if not isinstance(node, dict):
                continue
            kind, mid = node.get('type'), node.get('id')
            if isinstance(mid, str) and mid not in message_order:
                message_order.append(mid)
            if kind == 'assistant' and isinstance(node.get('tokens'), dict) and node.get('finish'):
                events.append(('step', measured(node['tokens']), mid))
            if kind == 'compaction' and node.get('status') == 'completed':
                metadata = node.get('metadata') or {}
                part = {'auto': node.get('reason') == 'auto', 'overflow': node.get('reason') == 'overflow',
                        'export_format': 'typed-v2'}
                if isinstance(metadata, dict) and 'keepID' in metadata:
                    part['tail_start_id'] = metadata['keepID']
                    part['tail_empty'] = metadata['keepID'] is None
                events.append(('compaction', part, mid))
                if isinstance(node.get('summary'), str):
                    events.append(('text', len(node['summary']), mid))
            if kind == 'user' and isinstance(node.get('text'), str):
                events.append(('text', len(node['text']), mid))
            if kind == 'assistant':
                for part in node.get('content') or []:
                    if isinstance(part, dict) and part.get('type') in ('text', 'reasoning') and isinstance(part.get('text'), str):
                        events.append(('text', len(part['text']), mid))
        return events, message_order
    for node in walk(document):
        kind = node.get('type')
        mid = node.get('messageID')
        if isinstance(mid, str) and mid not in message_order:
            message_order.append(mid)
        if kind == 'step-finish' and isinstance(node.get('tokens'), dict):
            events.append(('step', measured(node['tokens']), mid))
        elif kind == 'compaction' and 'auto' in node:
            events.append(('compaction', node, mid))
        elif kind == 'text' and isinstance(node.get('text'), str):
            events.append(('text', len(node['text']), mid))
    return events, message_order


def compaction_report(events, message_order, trigger, keep, prompt_tokens, slack):
    index = next((i for i, e in enumerate(events) if e[0] == 'compaction'), None)
    if index is None:
        return {'compacted': False, 'failures': ['no_compaction']}
    part, failures = events[index][1], []
    if part.get('auto') is not True or part.get('overflow') is True:
        failures.append('not_auto_threshold_compaction')
    steps = [e[1] for e in events[:index] if e[0] == 'step']
    last = steps[-1] if steps else 0
    if last + prompt_tokens < trigger - slack:
        failures.append('compacted_early')
    if any(s >= trigger + slack for s in steps[:-1]):
        failures.append('compacted_late')
    tail_chars = 0
    start = part.get('tail_start_id')
    if part.get('export_format') == 'typed-v2' and not (part.get('tail_empty') is True or start in message_order):
        failures.append('compaction_tail_boundary_unknown')
    if isinstance(start, str) and start in message_order:
        first = message_order.index(start)
        tail_ids = set(message_order[first:])
        tail_chars = sum(e[1] for e in events[:index] if e[0] == 'text' and e[2] in tail_ids)
    tail_tokens = -(-tail_chars // CHARS_PER_TOKEN)
    if tail_tokens > keep * (1 + KEEP_SLACK):
        failures.append('tail_exceeds_keep')
    return {'compacted': True, 'last_measured_before': last, 'max_measured_before': max(steps, default=0),
            'trigger': trigger, 'tail_tokens_estimate': tail_tokens, 'keep': keep,
            'tail_start_known': isinstance(start, str),
            'tail_explicitly_empty': part.get('tail_empty') is True, 'failures': failures}


def ledger_rows(directory, since, alias):
    rows = []
    for path in sorted(Path(directory).glob('requests-*.jsonl')):
        for line in path.read_text(errors='replace').splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if (isinstance(row, dict) and isinstance(row.get('t'), (int, float)) and row['t'] >= since
                    and row.get('model') == alias and str(row.get('path', '')).split('?')[0].endswith('/chat/completions')):
                rows.append(row)
    return sorted(rows, key=lambda r: r['t'])


def ledger_report(rows, min_prefix_hit):
    failures = []
    kinds = [r.get('turn_kind') for r in rows]
    if 'compaction' not in kinds:
        return {'failures': ['ledger_no_compaction_turn'], 'rows': len(rows)}
    cut = kinds.index('compaction')
    after = [r for r in rows[cut + 1:] if r.get('turn_kind') == 'post-compaction']
    first = after[0] if after else None
    if first is None:
        failures.append('ledger_no_post_compaction_turn')
    else:
        if first.get('compact_layout') != 'ok':
            failures.append('post_compaction_layout_not_ok')
        if not isinstance(first.get('cache_n'), (int, float)) or first['cache_n'] < min_prefix_hit:
            failures.append('post_compaction_prefix_miss')
        if first.get('miss_class') == 'unknown':
            failures.append('post_compaction_unknown_miss')
    second = after[1] if len(after) > 1 else None
    if second is None:
        failures.append('ledger_no_second_post_compaction_turn')
    elif second.get('miss_class') != 'append-ok':
        failures.append('c3_not_restored')
    if any(r.get('miss_class') == 'unknown' for r in rows):
        failures.append('unknown_miss_in_window')
    return {'rows': len(rows), 'compaction_turns': kinds.count('compaction'),
            'post_compaction_cache_n': first.get('cache_n') if first else None,
            'post_compaction_miss_class': first.get('miss_class') if first else None,
            'next_miss_class': second.get('miss_class') if second else None, 'failures': failures}


def isolation_report(models_stdout, launcher):
    listed = {line.strip() for line in models_stdout.splitlines() if line.strip()}
    moe = {f'{PROVIDER}/{model_id(launcher, k)}' for k in launcher.MOE_PRESET_MODELS}
    small = {f'{PROVIDER}/{m}' for m in launcher.SMALL_CONTEXT_MODEL_IDS}
    failures = []
    if not moe <= listed:
        failures.append('moe_model_not_listed')
    if small & listed:
        failures.append('small_context_model_selectable')
    return {'moe_listed': sorted(moe & listed), 'small_listed': sorted(small & listed), 'failures': failures}


def redacted(overlay):
    """The overlay as printed (dry run): the API key never leaves the process."""
    copy = json.loads(json.dumps(overlay))
    settings = copy.get('providers', {}).get(PROVIDER, {}).get('settings', {})
    if isinstance(settings, dict) and 'apiKey' in settings:
        settings['apiKey'] = REDACTED
    return copy


def plan(launcher, args, environment=None):
    environment = os.environ if environment is None else environment
    url = controller_url(args.controller_url)
    key = api_key_from(args.api_key_env, environment)
    overlay = build_overlay(launcher, args.model, args.window, args.mode,
                            base=environment.get('OPENCODE_CONFIG_CONTENT') or None, url=url, api_key=key)
    ref = launcher.CLASSIC_MODELS[args.model]
    limit = overlay['providers'][PROVIDER]['models'][model_id(launcher, args.model)]['limit']
    return {'model': args.model, 'model_ref': ref, 'window': args.window, 'mode': args.mode,
            'controller_url': url, 'api_key_env': args.api_key_env or None,
            'trigger': opencode_trigger(limit['input'], limit['context']),
            'keep': overlay['compaction']['keep']['tokens'], 'overlay': overlay,
            'commands': commands(args.opencode, ref)}


def controller_status(url, token=None, timeout=5):
    """GET <controller>/_local/status at the SAME target as the overlay, marked lab."""
    root = url[:-len('/v1')]
    request = urllib.request.Request(root + '/_local/status', headers={ORIGIN_HEADER: ORIGIN_LAB})
    if token:
        request.add_header('Authorization', 'Bearer ' + token)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read(1 << 20) or b'{}')
    except (urllib.error.URLError, OSError, ValueError) as error:
        raise RuntimeError('controller_status_unavailable') from error


def run(command, env, cwd, timeout):
    return subprocess.run(command, env=env, cwd=cwd, capture_output=True, text=True, timeout=timeout)


def live(launcher, args, runner=run, status=controller_status, environment=None):
    environment = dict(os.environ if environment is None else environment)
    p = plan(launcher, args, environment)
    env = dict(environment, OPENCODE_CONFIG_CONTENT=json.dumps(p['overlay'], separators=(',', ':')))
    cwd = tempfile.mkdtemp(prefix='oc-compaction-')
    version = runner(p['commands']['version'], env, cwd, 60).stdout
    if OPENCODE_VERSION not in version:
        raise RuntimeError('opencode_version_mismatch')
    current = status(p['controller_url'], api_key_from(args.api_key_env, environment))
    if current.get('selected_model', current.get('model')) not in (args.model, model_id(launcher, args.model)):
        raise RuntimeError('select_the_model_in_local_ai_first')
    receipt = {'tool': 'opencode_compaction_live', 'model': args.model, 'window': args.window,
               'mode': p['mode'], 'controller_url': p['controller_url'], 'origin': ORIGIN_LAB,
               'trigger': p['trigger'], 'keep': p['keep']}
    receipt['isolation'] = isolation_report(runner(p['commands']['models'], env, cwd, 120).stdout, launcher)
    since, session, report, extra = time.time() - 1, None, None, 0
    for turn in range(1, args.max_turns + 1):
        command = commands(args.opencode, p['model_ref'], session, filler_prompt(turn, args.filler_tokens))['run']
        result = runner(command, env, cwd, args.turn_timeout)
        session = session_from_events(result.stdout) if session is None else session
        if result.returncode != 0:
            raise RuntimeError(f'opencode_run_failed_turn_{turn}')
        exported = runner(commands(args.opencode, p['model_ref'], session)['export'], env, cwd, 120)
        events, order = analyse_export(json.loads(exported.stdout))
        report = compaction_report(events, order, p['trigger'], p['keep'], args.filler_tokens, args.slack)
        if report['compacted']:
            extra += 1
            if extra > 2:  # two turns after the compaction: post-compaction hit and C3 check
                break
    receipt['turns'] = turn
    receipt['compaction'] = report
    receipt['ledger'] = ledger_report(ledger_rows(args.ledger_dir, since, model_id(launcher, args.model)),
                                      args.min_prefix_hit)
    receipt['failures'] = receipt['isolation']['failures'] + report['failures'] + receipt['ledger']['failures']
    receipt['ok'] = not receipt['failures']
    return receipt


def parser():
    a = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    a.add_argument('--model', choices=('moe', 'ornith'), required=True)
    a.add_argument('--window', choices=tuple(WINDOWS), required=True)
    a.add_argument('--mode', choices=('fast', 'deep'), default='fast',
                   help='MoE preset the window is cut from (default fast, the production default)')
    a.add_argument('--controller-url', default=DEFAULT_CONTROLLER_URL,
                   help='loopback controller base URL ending in /v1 (default: the installed 8092)')
    a.add_argument('--api-key-env', help='name of an environment variable holding a fresh controller token')
    a.add_argument('--dry-run', action='store_true')
    a.add_argument('--max-turns', type=int, default=16)
    a.add_argument('--filler-tokens', type=int, default=1500)
    a.add_argument('--slack', type=int, default=1024)
    a.add_argument('--min-prefix-hit', type=int, default=4000)
    a.add_argument('--turn-timeout', type=int, default=900)
    a.add_argument('--opencode', default='opencode')
    a.add_argument('--ledger-dir', default=os.environ.get('LOCAL_AI_LEDGER_DIR',
                                                          str(Path.home() / '.local/state/local-ai/ledger')))
    a.add_argument('--out')
    return a


def main(argv=None, runner=run, status=controller_status, environment=None):
    args = parser().parse_args(argv)
    launcher = load_launcher()
    if args.dry_run:
        p = plan(launcher, args, environment)
        p['overlay'] = redacted(p['overlay'])
        print(json.dumps(p, indent=2))
        return 0
    receipt = live(launcher, args, runner, status, environment)
    text = json.dumps(receipt, indent=2)
    if args.out:
        Path(args.out).write_text(text + '\n')
    print(text)
    return 0 if receipt['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
