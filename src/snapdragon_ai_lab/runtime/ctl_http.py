"""The HTTP front: replies, the Host/Origin/CORS/API-key gate every request passes first, and the
dispatch to the control routes (ctl_routes) or the engine proxy (ctl_proxy).
"""
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
import json
from urllib.parse import urlsplit
if __package__:
    from .ctl_proxy import ProxyRoutes
    from .ctl_routes import ControlRoutes, PASS
    from .ctl_seams import ctl
    from . import openai_api
else:
    from ctl_proxy import ProxyRoutes
    from ctl_routes import ControlRoutes, PASS
    from ctl_seams import ctl
    import openai_api


MODEL_HELPER_PATHS = {'/tokenize', '/detokenize', '/apply-template'}
INFERENCE_PATHS = {'/completion', '/completions', '/chat/completions', '/infill',
                   '/embedding', '/embeddings', '/rerank', '/reranking',
                   '/v1/completions', '/v1/chat/completions', '/v1/embeddings',
                   '/v1/rerank', '/v1/reranking', '/v1/responses'}


def canonical_target(raw):
    """The request target the controller routes and forwards: one leading '/', no trailing '/'
    (except the root), query kept, fragment dropped. None for anything else ('//x' would parse
    as a host name, absolute-form and '*' are not served)."""
    if not isinstance(raw, str) or not raw.startswith('/') or raw.startswith('//'):
        return None
    parts = urlsplit(raw)
    if parts.scheme or parts.netloc:
        return None
    path = parts.path.rstrip('/') or '/'
    if path.startswith('//'):
        return None
    return path + ('?' + parts.query if parts.query else '')


class Handler(ControlRoutes, ProxyRoutes, BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    def log_message(self, *args):
        pass

    @property
    def manager(self): return self.server.manager

    def reply(self, code, body, content_type='application/json'):
        self.send_response(code)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        if code in (429, 503): self.send_header('Retry-After', '5')
        for name, value in self.cors_reply_headers(): self.send_header(name, value)
        self.end_headers()
        if self.command != 'HEAD':
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                self.close_connection = True

    def json_reply(self, code, value):
        # Every error outside the control routes has the OpenAI shape, for any client library.
        if code >= 400 and not urlsplit(self.path).path.startswith('/_local/'):
            value = openai_api.normalize_error(code, value)
        self.reply(code, json.dumps(value).encode())

    def api_settings(self):
        """runtime.json as a dict: API key, CORS and alias settings. While the file is
        unreadable the last good copy is used, so a configured key never lapses."""
        try:
            value = json.loads(ctl.CONFIG.read_text())
            if not isinstance(value, dict): raise ValueError()
            self.server.api_settings_last = value
            return value
        except (OSError, ValueError):
            return getattr(self.server, 'api_settings_last', {})

    def cors_reply_headers(self):
        origin = getattr(self, 'cors_allowed_origin', None)
        return openai_api.cors_headers(origin) if origin else []

    def do_HEAD(self): self.forward()
    def do_GET(self): self.forward()
    def do_POST(self): self.forward()
    def do_OPTIONS(self): self.forward()
    def do_DELETE(self): self.forward()

    def browser_origin_allowed(self):
        origins = self.headers.get_all('Origin', [])
        sites = self.headers.get_all('Sec-Fetch-Site', [])
        if len(origins) > 1 or len(sites) > 1:
            return False
        # A deliberate top-level click may come from the separate Image Studio.
        navigation = {'Sec-Fetch-Mode':'navigate', 'Sec-Fetch-Dest':'document',
                      'Sec-Fetch-User':'?1'}
        if (not origins and self.command in ('GET', 'HEAD') and self.path == '/'
                and all(self.headers.get_all(key, []) == [value]
                        for key, value in navigation.items())):
            return True
        if sites and sites[0] not in ('same-origin', 'none'):
            return False
        allowed = {f'http://127.0.0.1:{self.server.server_port}',
                   f'http://localhost:{self.server.server_port}',
                   f'http://[::1]:{self.server.server_port}'}
        return not origins or origins[0] in allowed

    def host_allowed(self):
        """F80: only loopback names with the bound port. A DNS-rebinding page reaches 127.0.0.1
        under its own host name; without this check its same-origin GETs (no Origin header) could
        read keyless routes. One Host header, compared case-insensitively."""
        hosts = self.headers.get_all('Host', [])
        if len(hosts) != 1:
            return False
        if getattr(self, 'share_client', None) is not None:
            return True  # shared listener: the share key already authenticated this request
        port = self.server.server_port
        return hosts[0].strip().lower() in {f'127.0.0.1:{port}', f'localhost:{port}', f'[::1]:{port}'}

    def forward(self):
        # O-102 LAN share: a request on a shared listener passes lan_share's gate first (key,
        # route allowlist, rate, body and one-at-a-time limits); loopback is unchanged.
        self.share_client = None  # the handler persists across keep-alive requests
        listener = getattr(self.server, 'share_listener', None)
        if listener is not None:
            return listener.controller.handle(self, self._forward)
        return self._forward()

    def _forward(self):
        self.cors_allowed_origin = None  # the handler persists across keep-alive requests
        if not self.host_allowed():
            # before any work: no settings read, no admission, no backend
            self.close_connection = True
            return self.json_reply(403, {'error': 'Host not allowed: use 127.0.0.1, localhost or [::1] with this port.'})
        canonical = canonical_target(self.path)
        if canonical is None:
            self.close_connection = True
            return self.json_reply(404, {'error': {'message': 'Unknown route.', 'type': 'invalid_request_error'}})
        # S4: route and forward the same path. The Rust engine trims trailing slashes, so
        # '/v1/chat/completions/' must be an inference request here too (admission, ledger,
        # defaults, disconnect watcher), not a pass-through.
        self.path = canonical
        path = urlsplit(self.path).path
        origins = self.headers.get_all('Origin', [])
        api_value = self.api_settings()
        api_key = openai_api.api_key(api_value)
        if origins and not path.startswith('/_local/'):
            # Opt-in CORS for the API routes (openai_api.cors_origins); the control routes
            # stay same-origin only.
            self.cors_allowed_origin = openai_api.cors_origin(origins, api_value, api_key is not None)
        if self.cors_allowed_origin is None and not self.browser_origin_allowed():
            self.close_connection = True
            return self.json_reply(403, {'error': 'Cross-origin local AI request refused.'})
        if self.cors_allowed_origin is not None and self.command == 'OPTIONS':
            self.send_response(204)
            for name, value in openai_api.cors_headers(
                    self.cors_allowed_origin, preflight=True,
                    request_headers=self.headers.get('Access-Control-Request-Headers'),
                    private_network=self.headers.get('Access-Control-Request-Private-Network') == 'true'):
                self.send_header(name, value)
            return self.end_headers()
        remote = getattr(self, 'share_client', None) is not None
        if (api_key is not None and not remote and openai_api.key_required(self.command, path)
                and not openai_api.authorized(self.headers, api_key)):
            self.close_connection = True
            return self.json_reply(401, {'error': {
                'message': 'Missing or wrong API key. Send "Authorization: Bearer <key>".',
                'type': 'authentication_error', 'code': 'invalid_api_key'}})
        inference = self.command == 'POST' and path in INFERENCE_PATHS
        helper = self.command == 'POST' and path in MODEL_HELPER_PATHS
        if self.control_route(path, api_value) is not PASS:
            return
        return self.proxy_request(path, api_value, api_key, remote, inference, helper)


class Server(ThreadingHTTPServer):
    daemon_threads = True
