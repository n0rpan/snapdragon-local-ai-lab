"""A model-free stand-in for llama-server: serves /health and /v1/models on --port.

Used by controller tests that need a real child process and a real port, never a model.
`--fail` makes it exit before binding (a start failure); `--alias X` answers /v1/models with
model id X; `--no-bind` keeps running without ever listening (another process may answer the
port instead); `--bad-gzip` answers GET / with a corrupt `Content-Encoding: gzip` body. POST
/v1/chat/completions streams one SSE chunk and [DONE]; `--die-mid-stream` exits (code 9) right
after the first chunk instead (an engine crash while a client is reading).
"""
import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--port', type=int, required=True)
parser.add_argument('--alias', default='stub-local')
parser.add_argument('--fail', action='store_true')
parser.add_argument('--no-bind', action='store_true')
parser.add_argument('--bad-gzip', action='store_true')
parser.add_argument('--die-mid-stream', action='store_true')
args, _ = parser.parse_known_args()
if args.fail:
    sys.exit(3)
if args.no_bind:
    import time
    time.sleep(3600)
    sys.exit(0)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == '/health':
            body = b'{"status":"ok"}'
        elif self.path == '/' and args.bad_gzip:
            body = b'this is not gzip'
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Encoding', 'gzip')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        elif self.path == '/v1/models':
            body = json.dumps({'object': 'list', 'data': [{'id': args.alias, 'object': 'model'}]}).encode()
        else:
            self.send_response(404); self.end_headers(); return
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        self.rfile.read(int(self.headers.get('Content-Length') or 0))
        if self.path != '/v1/chat/completions':
            self.send_response(404); self.end_headers(); return
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Connection', 'close')
        self.end_headers()
        chunk = {'object': 'chat.completion.chunk', 'model': args.alias,
                 'choices': [{'index': 0, 'delta': {'content': 'partial'}, 'finish_reason': None}]}
        self.wfile.write(b'data: ' + json.dumps(chunk).encode() + b'\n\n')
        self.wfile.flush()
        if args.die_mid_stream:
            import os
            os._exit(9)
        self.wfile.write(b'data: [DONE]\n\n')
        self.wfile.flush()
        self.close_connection = True


HTTPServer(('127.0.0.1', args.port), Handler).serve_forever()
