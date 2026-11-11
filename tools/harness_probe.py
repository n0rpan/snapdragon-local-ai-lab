#!/usr/bin/env python3
"""Offline probe of what an agent harness sends to an OpenAI-compatible server.

No model is involved. Two subcommands:

  mock     a scripted upstream: answers every chat request with a tool call for the first
           `--rounds` tool rounds after a user message, then with plain text. Put
           `session_replay.py record` in front of it and point the harness at the recorder.
  interleave  merge recordings prompt by prompt, for replaying a harness switch.
  analyze  read one or more recordings and report, per request: system prompt and tool
           sizes, sampling parameters, streaming, message shapes, whether history is re-sent
           verbatim and (with `--server`, a tokenizer-only `oxe-server --engine stub`)
           whether turn N+1's prompt tokens extend turn N's prompt + reply tokens, which is
           what the engine's prefix cache needs.

Stdlib only. Binds loopback only. Recordings hold request bodies: keep them out of git.
"""
import argparse
import hashlib
import json
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parent))
import session_replay as sr  # noqa: E402

CHAT_PATHS = ('/v1/chat/completions', '/chat/completions')
TOOL_PREFERENCE = ('bash', 'shell', 'exec', 'run_command', 'execute_command', 'read', 'read_file', 'ls', 'list', 'glob')
SAMPLING_KEYS = ('temperature', 'top_p', 'top_k', 'min_p', 'seed', 'presence_penalty', 'frequency_penalty',
                 'repeat_penalty', 'stop')
LENGTH_KEYS = ('max_tokens', 'max_completion_tokens', 'n_predict')
KNOWN_KEYS = frozenset(('model', 'messages', 'tools', 'tool_choice', 'stream', 'stream_options')
                       + SAMPLING_KEYS + LENGTH_KEYS)
# request keys oxe-server reads (crates/chat/src/request.rs, crates/server/src/api.rs); others are ignored
OXE_KEYS = KNOWN_KEYS | {'cache_prompt', 'chat_template_kwargs', 'ignore_eos', 'parallel_tool_calls',
                         'reasoning_format', 'response_format', 'n_probs', 'logprobs', 'top_logprobs'}


# ---------------------------------------------------------------- mock upstream

def _argument_for(name, schema):
    lowered = name.lower()
    kind = schema.get('type') if isinstance(schema, dict) else None
    if isinstance(kind, list):
        kind = next((k for k in kind if k != 'null'), None)
    if kind in (None, 'string'):
        if isinstance(schema, dict) and isinstance(schema.get('enum'), list) and schema['enum']:
            return schema['enum'][0]
        if 'command' in lowered or lowered in ('cmd', 'script'):
            return 'ls'
        if 'path' in lowered or 'file' in lowered:
            return 'README.md'
        if 'pattern' in lowered or 'glob' in lowered:
            return '*'
        return 'probe'
    return {'integer': 1, 'number': 1, 'boolean': False, 'array': [], 'object': {}}.get(kind, 'probe')


def scripted_arguments(tool):
    params = tool.get('function', {}).get('parameters') or {}
    props = params.get('properties') if isinstance(params, dict) else None
    if not isinstance(props, dict):
        return {}
    required = params.get('required') if isinstance(params.get('required'), list) else list(props)[:1]
    return {key: _argument_for(key, props.get(key, {})) for key in required if key in props}


def pick_tool(tools, round_index):
    """Deterministic: the round-th distinct tool in TOOL_PREFERENCE order, else the first tool."""
    named = {}
    for tool in tools or []:
        if isinstance(tool, dict) and isinstance(tool.get('function'), dict):
            named.setdefault(str(tool['function'].get('name', '')), tool)
    ranked = [named[n] for pref in TOOL_PREFERENCE for n in named if n.lower() == pref]
    if not ranked:
        ranked = list(named.values())
    return ranked[round_index % len(ranked)] if ranked else None


def tool_rounds(messages):
    """Tool-result messages since the last user message."""
    count = 0
    for message in reversed(messages or []):
        if not isinstance(message, dict) or message.get('role') == 'user':
            break
        if message.get('role') == 'tool':
            count += 1
    return count


def scripted_reply(body, rounds):
    """Return (content, tool_call or None) for a chat body."""
    tools = body.get('tools') if isinstance(body.get('tools'), list) else []
    done = tool_rounds(body.get('messages'))
    if tools and body.get('tool_choice') != 'none' and done < rounds:
        tool = pick_tool(tools, done)
        if tool is not None:
            name = tool['function']['name']
            return None, {'id': 'call_probe%04d' % done, 'name': name,
                          'arguments': json.dumps(scripted_arguments(tool))}
    return 'Probe complete.', None


class _MockHandler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, *args):
        pass

    def _json(self, code, value):
        data = json.dumps(value).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = urlsplit(self.path).path
        if path in ('/v1/models', '/models'):
            return self._json(200, {'object': 'list', 'data': [
                {'id': self.server.model, 'object': 'model', 'created': 0, 'owned_by': 'local'}]})
        if path in ('/health', '/v1/health'):
            return self._json(200, {'status': 'ok'})
        if path == '/props':
            return self._json(200, {'model_path': '/models/probe.gguf', 'build_info': 'harness-probe-mock',
                                    'total_slots': 1, 'default_generation_settings': {'n_ctx': 32768}})
        return self._json(404, {'error': 'not found'})

    def do_POST(self):
        path = urlsplit(self.path).path
        try:
            body = json.loads(self.rfile.read(int(self.headers.get('Content-Length', '0'))) or b'{}')
        except ValueError:
            return self._json(400, {'error': 'invalid json'})
        with self.server.lock:
            self.server.requests.append({'path': path, 'headers': {
                k.lower(): v for k, v in self.headers.items()
                if k.lower() not in ('authorization', 'x-api-key', 'api-key', 'cookie')}})
        if path not in CHAT_PATHS or not isinstance(body, dict):
            return self._json(404, {'error': 'only chat completions are scripted'})
        content, call = scripted_reply(body, self.server.rounds)
        prompt_tokens = len(json.dumps(body.get('messages'))) // 4
        completion_tokens = len(content or call['arguments']) // 4 + 1
        usage = {'prompt_tokens': prompt_tokens, 'completion_tokens': completion_tokens,
                 'total_tokens': prompt_tokens + completion_tokens}
        base = {'id': 'chatcmpl-probe', 'created': int(time.time()), 'model': body.get('model', self.server.model)}
        finish = 'tool_calls' if call else 'stop'
        if not body.get('stream'):
            message = {'role': 'assistant', 'content': content}
            if call:
                message['tool_calls'] = [{'id': call['id'], 'type': 'function',
                                          'function': {'name': call['name'], 'arguments': call['arguments']}}]
            return self._json(200, {**base, 'object': 'chat.completion', 'usage': usage,
                                    'choices': [{'index': 0, 'finish_reason': finish, 'message': message}]})
        chunks = [{'role': 'assistant', 'content': None}]
        if call:
            chunks.append({'tool_calls': [{'index': 0, 'id': call['id'], 'type': 'function',
                                           'function': {'name': call['name'], 'arguments': ''}}]})
            chunks.append({'tool_calls': [{'index': 0, 'function': {'arguments': call['arguments']}}]})
        else:
            chunks.append({'content': content})
        events = [{**base, 'object': 'chat.completion.chunk',
                   'choices': [{'index': 0, 'finish_reason': None, 'delta': delta}]} for delta in chunks]
        events.append({**base, 'object': 'chat.completion.chunk',
                       'choices': [{'index': 0, 'finish_reason': finish, 'delta': {}}]})
        if isinstance(body.get('stream_options'), dict) and body['stream_options'].get('include_usage'):
            events.append({**base, 'object': 'chat.completion.chunk', 'choices': [], 'usage': usage})
        data = b''.join(b'data: ' + json.dumps(e).encode() + b'\n\n' for e in events) + b'data: [DONE]\n\n'
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class MockUpstream(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, port=0, rounds=2, model='probe-local'):
        super().__init__(('127.0.0.1', port), _MockHandler)
        self.rounds, self.model = rounds, model
        self.lock = threading.Lock()
        self.requests = []
        self.url = 'http://127.0.0.1:%d' % self.server_address[1]


def cmd_mock(args):
    server = MockUpstream(args.port, args.rounds, args.model)
    print('harness-probe mock on %s (rounds %d)' % (server.url, args.rounds), flush=True)
    def stop(*_):
        raise KeyboardInterrupt()
    # a backgrounded process ignores SIGINT: stop (and write --headers-out) on SIGTERM too
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        server.serve_forever(poll_interval=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        if args.headers_out:
            Path(args.headers_out).write_text(json.dumps(server.requests, indent=1))
        server.server_close()
    return 0


# ---------------------------------------------------------------- analysis

def _sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=False).encode()).hexdigest()[:12]


def _text(content):
    return sr.content_text(content) if not isinstance(content, str) else content


def leading_system(messages):
    out = []
    for message in messages:
        if not isinstance(message, dict) or message.get('role') not in ('system', 'developer'):
            break
        out.append(message)
    return out


def assistant_from_response(response):
    """The assistant message a client is expected to send back for a recorded reply."""
    calls = [{'name': tc.get('name'), 'arguments': _canonical_json(tc.get('arguments'))}
             for tc in response.get('tool_calls') or []]
    return {'content': response.get('content') or '', 'tool_calls': calls}


def _canonical_json(text):
    try:
        return json.dumps(json.loads(text), sort_keys=True) if isinstance(text, str) else json.dumps(text, sort_keys=True)
    except ValueError:
        return text


def assistant_normal_form(message):
    calls = [{'name': (tc.get('function') or {}).get('name'),
              'arguments': _canonical_json((tc.get('function') or {}).get('arguments'))}
             for tc in message.get('tool_calls') or [] if isinstance(tc, dict)]
    return {'content': _text(message.get('content')) or '', 'tool_calls': calls}


def describe_turn(turn):
    body = turn.get('body') if isinstance(turn.get('body'), dict) else {}
    messages = body.get('messages') if isinstance(body.get('messages'), list) else []
    tools = body.get('tools') if isinstance(body.get('tools'), list) else []
    systems = leading_system(messages)
    system_text = '\n'.join(_text(m.get('content')) or '' for m in systems)
    shapes = set()
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get('content')
        kind = 'null' if content is None else 'string' if isinstance(content, str) else 'parts'
        shapes.add('%s:%s' % (message.get('role'), kind))
    later_system = sum(1 for m in messages[len(systems):]
                       if isinstance(m, dict) and m.get('role') in ('system', 'developer'))
    return {
        'seq': turn.get('seq'), 'path': urlsplit(turn.get('path') or '').path,
        'agent': _sha({'system': system_text, 'tools': [t.get('function', {}).get('name') for t in tools
                                                         if isinstance(t, dict)]}),
        'system_sha': _sha(system_text), 'system_chars': len(system_text),
        'system_roles': sorted({m.get('role') for m in systems}),
        'tools': len(tools), 'tools_chars': len(json.dumps(tools, ensure_ascii=False)) if tools else 0,
        'tools_sha': _sha(tools) if tools else None,
        'tool_names': [t.get('function', {}).get('name') for t in tools if isinstance(t, dict)],
        'messages': len(messages), 'message_shapes': sorted(shapes), 'later_system_messages': later_system,
        'stream': body.get('stream', False),
        'include_usage': bool(isinstance(body.get('stream_options'), dict)
                              and body['stream_options'].get('include_usage')),
        'sampling': {k: body[k] for k in SAMPLING_KEYS if k in body},
        'length': {k: body[k] for k in LENGTH_KEYS if k in body},
        'tool_choice': body.get('tool_choice', 'absent'),
        'other_keys': sorted(set(body) - KNOWN_KEYS),
        'keys_ignored_by_oxe': sorted(set(body) - OXE_KEYS),
        'model': body.get('model'),
    }


def history_check(prev_turn, turn):
    """Does `turn` re-send `prev_turn`'s messages unchanged and append the recorded reply?"""
    prev = (prev_turn.get('body') or {}).get('messages') or []
    cur = (turn.get('body') or {}).get('messages') or []
    if len(cur) <= len(prev):
        return {'verbatim_prefix': False, 'reason': 'history shrank or was rewritten (%d -> %d messages)'
                % (len(prev), len(cur))}
    for index, (a, b) in enumerate(zip(prev, cur)):
        if a != b:
            return {'verbatim_prefix': False, 'reason': 'message %d (%s) changed' % (index, a.get('role'))}
    echoed = cur[len(prev)]
    result = {'verbatim_prefix': True}
    if isinstance(echoed, dict) and echoed.get('role') == 'assistant':
        expected = assistant_from_response(prev_turn.get('response') or {})
        result['assistant_echo_matches'] = assistant_normal_form(echoed) == expected
        result['assistant_content_kind'] = ('null' if echoed.get('content') is None else
                                            'string' if isinstance(echoed.get('content'), str) else 'parts')
        result['assistant_extra_keys'] = sorted(set(echoed) - {'role', 'content', 'tool_calls'})
    else:
        result['assistant_echo_matches'] = False
        result['reason'] = 'next message is %r, not the assistant reply' % (
            echoed.get('role') if isinstance(echoed, dict) else None)
    return result


def common_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


class Tokenizer:
    """Exact prompt ids through a tokenizer-only oxe-server (`--engine stub`)."""

    def __init__(self, server, oxe_defaults=True, template_kwargs=None):
        self.parts = sr.parse_url(server)
        self.oxe_defaults = oxe_defaults
        self.template_kwargs = dict(template_kwargs or {})

    def prompt_ids(self, body):
        req = {k: body[k] for k in ('messages', 'tools', 'tool_choice', 'chat_template_kwargs') if k in body}
        if self.oxe_defaults:
            kwargs = req.setdefault('chat_template_kwargs', {})
            if isinstance(kwargs, dict):
                kwargs.setdefault('enable_thinking', False)
        if self.template_kwargs:
            req['chat_template_kwargs'] = {**(req.get('chat_template_kwargs') or {}), **self.template_kwargs}
        prompt = sr.post_plain(self.parts, '/apply-template', req)['prompt']
        return self.ids(prompt, True)

    def ids(self, text, add_special=False):
        if not text:
            return []
        return sr.post_plain(self.parts, '/tokenize', {'content': text, 'add_special': add_special,
                                                       'parse_special': True})['tokens']

    def anchor_tokens(self, body):
        """Tokens before the first user message: what the persistent prefix store can hold."""
        messages = body.get('messages') or []
        systems = leading_system(messages)
        variants = []
        for probe in ('A', 'B different'):
            variant = dict(body)
            variant['messages'] = systems + [{'role': 'user', 'content': probe}]
            variants.append(self.prompt_ids(variant))
        return common_prefix(*variants)


def analyze_recording(path, tokenizer=None):
    _header, turns, sha = sr.load_recording(path)
    # a failed or aborted exchange (a 503 while another request ran, a client disconnect)
    # has no usable reply and was retried by the client: leave it out
    chat = [t for t in turns if urlsplit(t.get('path') or '').path in CHAT_PATHS and isinstance(t.get('body'), dict)
            and not t.get('aborted') and t.get('status') in (None, 200)]
    rows = [describe_turn(t) for t in chat]
    last_by_agent = {}
    ids = {}
    for turn, row in zip(chat, rows):
        if tokenizer is not None:
            prompt = tokenizer.prompt_ids(turn['body'])
            completion = tokenizer.ids(sr.render_reply(turn.get('response') or {}))
            ids[row['seq']] = (prompt, completion)
            row['prompt_tokens'] = len(prompt)
            row['anchor_tokens'] = tokenizer.anchor_tokens(turn['body'])
        previous = last_by_agent.get(row['agent'])
        if previous is not None:
            prev_turn, prev_row = previous
            row['history'] = history_check(prev_turn, turn)
            if tokenizer is not None:
                prev_prompt, prev_completion = ids[prev_row['seq']]
                target = len(prev_prompt) + len(prev_completion)
                reused = common_prefix(prev_prompt + prev_completion, ids[row['seq']][0])
                row['reuse'] = {'previous_prompt_tokens': len(prev_prompt), 'previous_total_tokens': target,
                                'reusable_tokens': reused, 'new_tokens': len(ids[row['seq']][0]) - reused,
                                'extends_previous_prompt': reused >= len(prev_prompt),
                                'extends_previous_reply': reused >= target}
        last_by_agent[row['agent']] = (turn, row)
    return {'recording': str(path), 'recording_sha256': sha, 'requests': rows, 'first_prompt_ids':
            ids.get(rows[0]['seq'], (None,))[0] if rows else None}


def summarize(result):
    rows = result['requests']
    agents = {}
    for row in rows:
        agents.setdefault(row['agent'], []).append(row)
    summary = {'requests': len(rows), 'agents': []}
    for key, group in agents.items():
        first = group[0]
        followups = [r for r in group if 'history' in r]
        reuse = [r['reuse'] for r in group if 'reuse' in r]
        summary['agents'].append({
            'agent': key, 'requests': len(group), 'system_chars': first['system_chars'],
            'system_roles': first['system_roles'], 'tools': first['tools'], 'tools_chars': first['tools_chars'],
            'anchor_tokens': first.get('anchor_tokens'),
            'stream': sorted({r['stream'] for r in group}),
            'sampling': sorted({json.dumps(r['sampling'], sort_keys=True) for r in group}),
            'length': sorted({json.dumps(r['length'], sort_keys=True) for r in group}),
            'other_keys': sorted({k for r in group for k in r['other_keys']}),
            'keys_ignored_by_oxe': sorted({k for r in group for k in r['keys_ignored_by_oxe']}),
            'message_shapes': sorted({s for r in group for s in r['message_shapes']}),
            'history_verbatim': all(r['history']['verbatim_prefix'] for r in followups) if followups else None,
            'assistant_echo_matches': (all(r['history'].get('assistant_echo_matches') for r in followups)
                                       if followups else None),
            'prompt_extends_previous_prompt': all(r['extends_previous_prompt'] for r in reuse) if reuse else None,
            'prompt_extends_previous_reply': all(r['extends_previous_reply'] for r in reuse) if reuse else None,
            'min_reuse_share': (min(r['reusable_tokens'] / max(1, r['previous_total_tokens']) for r in reuse)
                                if reuse else None),
            # prompt tokens a perfect prefix cache still has to evaluate over the follow-up turns
            'followup_new_tokens': sum(r['new_tokens'] for r in reuse) if reuse else None,
            'followup_prompt_tokens': sum(r['prompt_tokens'] for r in group if 'reuse' in r) if reuse else None,
        })
    return summary


def cmd_analyze(args):
    extra = sr.parse_template_kwargs(args.template_kwarg)
    tokenizer = Tokenizer(args.server, not args.no_oxe_defaults, extra) if args.server else None
    results = [analyze_recording(path, tokenizer) for path in args.recordings]
    report = {'recordings': []}
    for result in results:
        first_ids = result.pop('first_prompt_ids')
        result['summary'] = summarize(result)
        result['_first_ids'] = first_ids
        report['recordings'].append(result)
    if tokenizer is not None and len(results) > 1:
        # the same harness started twice (another directory, another time): how much of the
        # first prompt is shared decides whether a stored prefix survives across sessions
        base = report['recordings'][0]
        report['cross_session'] = []
        for other in report['recordings'][1:]:
            if base['_first_ids'] and other['_first_ids']:
                shared = common_prefix(base['_first_ids'], other['_first_ids'])
                anchor = base['requests'][0].get('anchor_tokens')
                report['cross_session'].append({
                    'a': base['recording'], 'b': other['recording'], 'shared_prefix_tokens': shared,
                    'anchor_tokens': anchor, 'prefix_stable': anchor is not None and shared >= anchor})
    for result in report['recordings']:
        result.pop('_first_ids', None)
    text = json.dumps(report, indent=1, ensure_ascii=False)
    if args.out:
        Path(args.out).write_text(text + '\n')
    for result in report['recordings']:
        print(result['recording'])
        for agent in result['summary']['agents']:
            print('  agent %s: %d requests, system %d chars, %d tools (%d chars), anchor %s tokens' % (
                agent['agent'], agent['requests'], agent['system_chars'], agent['tools'], agent['tools_chars'],
                agent['anchor_tokens']))
            print('    sampling %s  length %s  stream %s' % (agent['sampling'], agent['length'], agent['stream']))
            print('    other keys %s (ignored by oxe: %s)' % (agent['other_keys'], agent['keys_ignored_by_oxe']))
            print('    history verbatim %s, assistant echo %s, extends prompt %s, extends reply %s, min reuse %s' % (
                agent['history_verbatim'], agent['assistant_echo_matches'], agent['prompt_extends_previous_prompt'],
                agent['prompt_extends_previous_reply'],
                None if agent['min_reuse_share'] is None else '%.3f' % agent['min_reuse_share']))
            if agent['followup_new_tokens'] is not None:
                print('    follow-up turns: %d new of %d prompt tokens' % (
                    agent['followup_new_tokens'], agent['followup_prompt_tokens']))
    for row in report.get('cross_session', []):
        print('cross-session shared prefix %s tokens, anchor %s: %s' % (
            row['shared_prefix_tokens'], row['anchor_tokens'], 'stable' if row['prefix_stable'] else 'NOT stable'))
    return 0


# ---------------------------------------------------------------- interleave

def prompt_blocks(turns):
    """Split a recording into blocks, one per user prompt of its main agent (the agent of
    the first turn). Side requests (titles, subagents) stay in the block they occurred in."""
    blocks, main = [], None
    for turn in turns:
        body = turn.get('body') if isinstance(turn.get('body'), dict) else {}
        messages = body.get('messages') if isinstance(body.get('messages'), list) else []
        agent = sr.agent_info(body)[0]
        if main is None:
            main = agent
        last = messages[-1] if messages and isinstance(messages[-1], dict) else {}
        if not blocks or (agent == main and last.get('role') == 'user'):
            blocks.append([])
        blocks[-1].append(turn)
    return blocks


def interleave(recordings):
    """Round-robin the prompt blocks of several recordings: A1 B1 A2 B2 ... (a user who
    switches harness after every prompt). Returns (header, turns) with new seq numbers."""
    loaded = [sr.load_recording(path) for path in recordings]
    queues = [prompt_blocks(turns) for _header, turns, _sha in loaded]
    header = dict(loaded[0][0])
    header['interleaved_from'] = [{'recording': str(path), 'sha256': sha, 'blocks': len(blocks)}
                                  for path, (_h, _t, sha), blocks in zip(recordings, loaded, queues)]
    out, seq = [], 0
    for round_index in range(max((len(q) for q in queues), default=0)):
        for source, blocks in enumerate(queues):
            if round_index >= len(blocks):
                continue
            for turn in blocks[round_index]:
                seq += 1
                out.append({**turn, 'seq': seq, 'source': source, 'source_seq': turn.get('seq'),
                            'block': round_index})
    return header, out


def cmd_interleave(args):
    header, turns = interleave(args.recordings)
    recording = sr.prepare_out_dir(args.out, args.allow_git)
    if recording.stat().st_size:
        raise sr.UsageError('refusing: %s already has content' % recording)
    with open(recording, 'w', encoding='utf-8') as handle:
        handle.write(json.dumps(header, ensure_ascii=False) + '\n')
        for turn in turns:
            handle.write(json.dumps(turn, ensure_ascii=False) + '\n')
    print('%d turns in %s from %s' % (len(turns), recording, ', '.join(
        '%d blocks' % item['blocks'] for item in header['interleaved_from'])))
    return 0


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    mock = sub.add_parser('mock', help='scripted upstream (tool calls, then text)')
    mock.add_argument('--port', type=int, default=18210)
    mock.add_argument('--rounds', type=int, default=2, help='tool rounds per user message')
    mock.add_argument('--model', default='probe-local')
    mock.add_argument('--headers-out', help='write the request headers seen (no credentials) on exit')
    ana = sub.add_parser('analyze', help='report request shapes and prefix reuse of recordings')
    ana.add_argument('recordings', nargs='+')
    ana.add_argument('--server', help='tokenizer-only oxe-server base URL for exact token counts')
    ana.add_argument('--no-oxe-defaults', action='store_true')
    ana.add_argument('--template-kwarg', action='append', metavar='KEY=JSON',
                     help='extra chat_template_kwargs for the what-if rendering, e.g. preserve_thinking=true')
    ana.add_argument('--out', help='write the full JSON report here')
    mix = sub.add_parser('interleave', help='merge recordings prompt by prompt (harness-switch scenario)')
    mix.add_argument('recordings', nargs='+')
    mix.add_argument('--out', required=True, help='directory for the merged recording.jsonl (outside git)')
    mix.add_argument('--allow-git', action='store_true')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return {'mock': cmd_mock, 'analyze': cmd_analyze, 'interleave': cmd_interleave}[args.command](args)
    except sr.UsageError as error:
        print('error: %s' % error, file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
