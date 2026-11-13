#!/usr/bin/env python3
"""Model-free probe of the TOOL-OUTPUT CAP with the REAL OpenCode client (no model, no controller).

A fake OpenAI-compatible server scripts three turns: (1) ask OpenCode's bash tool for `seq 1 N`;
(2) cap the returned tool result with tool_output_cap.cap_request (as the controller would) and ask
for the read-more tool on a hidden range; (3) check the read-more result is exactly those lines.
OpenCode runs with a private HOME/XDG under a temp dir and the read-more MCP server from this tree,
so nothing of the user's config, sessions or state is read or written.

Prints a JSON receipt: the read-more tool's exposed name, the tool message shapes, the session
header, and whether the range came back byte-exact. Exit 0 only if the whole loop worked.
Usage: tools/tool_output_cap_opencode_probe.py [--lines 3000] [--keep] [--no-auto]
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from snapdragon_ai_lab.runtime import tool_output_cap as cap  # noqa: E402

MODEL = 'probe-model'


def chunk(delta, finish=None):
    return {'id': 'p', 'object': 'chat.completion.chunk', 'created': 1, 'model': MODEL,
            'choices': [{'index': 0, 'delta': delta, 'finish_reason': finish}]}


def tool_call(name, arguments, call_id):
    return [chunk({'role': 'assistant', 'tool_calls': [{'index': 0, 'id': call_id, 'type': 'function',
                   'function': {'name': name, 'arguments': json.dumps(arguments)}}]}), chunk({}, 'tool_calls')]


def text(answer):
    return [chunk({'role': 'assistant', 'content': answer}), chunk({}, 'stop')]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lines', type=int, default=3000)
    parser.add_argument('--keep', action='store_true')
    parser.add_argument('--no-auto', action='store_true', help='no --auto: shows whether the read-more call needs a permission grant')
    args = parser.parse_args()
    tmp = Path(tempfile.mkdtemp(prefix='tool-cap-probe-'))
    home = tmp / 'home'; work = tmp / 'work'; store = tmp / 'store'
    for p in (home, work): p.mkdir()
    env_cap = {'LOCAL_AI_TOOL_OUTPUT_CAP': 'on', 'LOCAL_AI_TOOL_OUTPUT_CAP_STORE': str(store)}
    cfg = cap.settings(env_cap)
    receipt = {'requests': 0, 'tools_offered': None, 'reader': None, 'session_header': None,
               'bash_result_type': None, 'capped': None, 'marker_id': None, 'read_exact': None, 'errors': []}
    state = {'turn': 0, 'expected': None}

    class Fake(BaseHTTPRequestHandler):
        def log_message(self, *a): pass

        def do_GET(self):
            self.send_response(404); self.end_headers()

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))) or b'{}')
            if not self.path.endswith('/chat/completions'):
                self.send_response(404); self.end_headers(); return
            receipt['requests'] += 1
            names = [t.get('function', {}).get('name') for t in body.get('tools') or []]
            receipt['tools_offered'] = names
            reader = cap._read_tool_name(body)
            receipt['reader'] = reader
            receipt['session_header'] = next((h for h in cap.SESSION_HEADERS if self.headers.get(h)), None)
            msgs = body.get('messages') or []
            tool_msgs = [m for m in msgs if m.get('role') == 'tool']
            # title/summary side requests (no tools) just get text
            if not body.get('tools'):
                events = text('probe')
            elif not tool_msgs:
                events = tool_call('bash', {'command': 'seq 1 %d' % args.lines, 'description': 'Print numbers'}, 'call_1')
            elif len(tool_msgs) == 1:
                content = tool_msgs[0].get('content')
                receipt['bash_result_type'] = type(content).__name__
                capped, stats = cap.cap_request(body, self.headers, cfg)
                receipt['capped'] = stats
                rendered = [m for m in capped['messages'] if m.get('role') == 'tool'][0]['content']
                marker = rendered[rendered.find('id=') + 3:][:12] if 'id=' in rendered else None
                receipt['marker_id'] = marker
                lines = content.split('\n') if isinstance(content, str) else []
                state['expected'] = '\n'.join(lines[99:104])  # lines 100-104
                if reader and marker:
                    events = tool_call(reader, {'id': marker, 'offset': 100, 'limit': 5}, 'call_2')
                else:
                    receipt['errors'].append('no reader tool or no marker'); events = text('fail')
            else:
                got = tool_msgs[-1].get('content')
                if isinstance(got, list):
                    got = '\n'.join(p.get('text', '') for p in got if isinstance(p, dict))
                receipt['read_result_head'] = got[:200] if isinstance(got, str) else repr(got)[:200]
                receipt['read_exact'] = isinstance(got, str) and got.startswith(state['expected'] + '\n<<lines 100-104 of')
                events = text('done')
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream'); self.end_headers()
            for event in events:
                self.wfile.write(b'data: ' + json.dumps(event).encode() + b'\n\n')
            self.wfile.write(b'data: [DONE]\n\n'); self.wfile.flush()

    server = ThreadingHTTPServer(('127.0.0.1', 0), Fake)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    overlay = {
        'model': 'probe/' + MODEL,
        'providers': {'probe': {'name': 'Probe', 'package': '@opencode/ai/providers/openai-compatible',
                                'settings': {'baseURL': 'http://127.0.0.1:%d/v1' % server.server_port, 'apiKey': 'x'},
                                'models': {MODEL: {'name': 'Probe', 'type': 'chat', 'capabilities': {
                                    'tools': True, 'input': ['text'], 'output': ['text'], 'temperature': True,
                                    'reasoning': {'supported': False}},
                                    'limit': {'context': 131072, 'input': 65536, 'output': 4096}}}}},
        'mcp': {'servers': {cap.SERVER_NAME: cap.mcp_overlay_entry(
            ROOT / 'src/snapdragon_ai_lab/runtime/tool_output_cap.py', env_cap)}},
    }
    env = {'PATH': os.environ.get('PATH', '/usr/bin:/bin'), 'HOME': str(home),
           'XDG_CONFIG_HOME': str(home / '.config'), 'XDG_DATA_HOME': str(home / '.local/share'),
           'XDG_STATE_HOME': str(home / '.local/state'), 'XDG_CACHE_HOME': str(home / '.cache'),
           'OPENCODE_CONFIG_CONTENT': json.dumps(overlay), 'TERM': 'dumb'}
    try:
        command = ['opencode', 'run', '--standalone'] + ([] if args.no_auto else ['--auto']) + ['--archived', 'probe']
        run = subprocess.run(command,
                             env=env, cwd=work, capture_output=True, text=True, timeout=240)
        receipt['opencode_exit'] = run.returncode
        if run.returncode:
            receipt['stderr_tail'] = run.stderr[-1500:]
    except subprocess.TimeoutExpired:
        receipt['errors'].append('opencode timed out')
    finally:
        server.shutdown()
        if not args.keep:
            shutil.rmtree(tmp, ignore_errors=True)
        else:
            receipt['tmp'] = str(tmp)
    print(json.dumps(receipt, indent=1))
    return 0 if receipt.get('read_exact') else 1


if __name__ == '__main__':
    sys.exit(main())
