"""Proxy stage 1, request prep: the length and encoding checks, the body read, payload shaping
(shared/LAN payload, explicit route, model alias, tool-output cap, per-engine normalisation and
defaults, harness identify/observe), the Flash page and route rules and the cache hit before
admission; after admission, the backend connection and the header rules for the internal port.

Also the per-request state (`ProxyRequest`) the three stages share. Stage functions take the
Handler as `self` (they were its one method) and return ANSWERED once they replied.
"""
import gzip
import html
import http.client
import io
import json
import os
if __package__:
    from .ctl_env import Unavailable
    from .ctl_seams import ctl
    from .harness_registry import learning_settings
    from . import bigmoe_runtime, cache_turn, explicit_route, harness_compat, lan_share, openai_api, oxe_runtime, request_ledger, tool_output_cap
else:
    from ctl_env import Unavailable
    from ctl_seams import ctl
    from harness_registry import learning_settings
    import bigmoe_runtime
    import cache_turn
    import explicit_route
    import harness_compat
    import lan_share
    import openai_api
    import oxe_runtime
    import request_ledger
    import tool_output_cap


HOP_HEADERS = {'connection', 'keep-alive', 'proxy-authenticate', 'proxy-authorization',
               'te', 'trailer', 'transfer-encoding', 'upgrade', 'content-length'}
FLASH_CHAT_HTML = b'''<!doctype html><html><meta charset="utf-8"><title>Local Flash chat</title>
<style>body{font:16px system-ui;max-width:760px;margin:5vh auto;padding:16px;background:#171717;color:#eee}textarea{width:100%;min-height:90px}pre{white-space:pre-wrap}button{margin-top:8px}</style>
<body data-model="{MODEL}"><h1>Local Flash chat</h1><p>This page keeps messages in memory until it closes. The model loads when you send a message.</p>
<pre id="transcript"></pre><textarea id="prompt" aria-label="Message"></textarea><button id="send">Send</button><p id="status"></p>
<script>const messages=[];const transcript=document.getElementById('transcript');const status=document.getElementById('status');const button=document.getElementById('send');const input=document.getElementById('prompt');
button.onclick=async()=>{const text=input.value.trim();if(!text)return;button.disabled=true;status.textContent='Loading or answering...';messages.push({role:'user',content:text});transcript.textContent+='You: '+text+'\\n\\n';input.value='';try{const reply=await fetch('/v1/chat/completions',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({model:document.body.dataset.model,messages,stream:false,max_tokens:512,think:false})});const data=await reply.json();if(!reply.ok)throw new Error('The local backend could not answer.');const answer=data.choices?.[0]?.message?.content;if(typeof answer!=='string')throw new Error('Invalid local answer.');messages.push({role:'assistant',content:answer});transcript.textContent+='Assistant: '+answer+'\\n\\n';status.textContent='Ready';}catch(error){messages.pop();status.textContent=error.message;}finally{button.disabled=false;}};
</script></body></html>'''

# A stage returns this when it has already answered the client; the request ends there.
ANSWERED = object()


def answered(_reply):
    """ANSWERED, after the reply call in the argument has run."""
    return ANSWERED


class _BadFlashChat(Exception):
    """A Flash chat body the runtime could not convert (answered 400 after admission)."""


class ProxyRequest:
    """One proxied request: the locals the stages of the former single proxy_request shared."""
    __slots__ = ('path', 'api_value', 'api_key', 'remote', 'inference', 'helper',
                 'body', 'model_alias', 'payload', 'rewritten', 'target', 'harness_key', 'harness', 'turn',
                 'tool_cap', 'backend_path', 'completion_meta', 'request_process', 'request_engine_config',
                 'led', 'preempt', 'runtime', 'connection', 'backend_headers', 'started_response',
                 'client_write_failed', 'watcher', 'backend_status', 'request_error', 'content_type')

    def __init__(self, path, api_value, api_key, remote, inference, helper):
        self.path, self.api_value, self.api_key = path, api_value, api_key
        self.remote, self.inference, self.helper = remote, inference, helper
        self.body = self.model_alias = self.payload = self.target = None
        self.rewritten = False
        self.harness_key = self.harness = self.turn = self.tool_cap = None
        self.backend_path = self.completion_meta = None
        self.request_process = self.request_engine_config = self.led = self.preempt = None
        self.runtime = self.connection = self.backend_headers = None
        self.started_response = False
        self.client_write_failed = False
        self.watcher = None
        self.backend_status = None
        self.request_error = None
        self.content_type = None

    def client_gone(self):
        return self.client_write_failed or (self.watcher is not None and self.watcher.disconnected.is_set())

    def preempted(self):
        return self.preempt is not None and self.preempt.cancelled


def prepare(self, st):
    """Checks, body and payload shaping up to admission; ANSWERED when it replied (400, 404,
    411, 413, 503, the Flash page, 403 or a cache hit)."""
    path, remote, inference, helper = st.path, st.remote, st.inference, st.helper
    length = self.headers.get('Content-Length', '0')
    try: length = int(length)
    except ValueError:
        # the body's end is unknown: close, so its bytes are never read as the next request
        self.close_connection = True
        return answered(self.json_reply(400, {'error': 'Invalid content length.'}))
    if length < 0 or length > 16 * 1024 * 1024:
        self.close_connection = True
        return answered(self.json_reply(413, {'error': 'Request is too large.'}))
    if self.headers.get('Transfer-Encoding'):
        self.close_connection = True
        return answered(self.json_reply(411, {'error': 'Use a content length for request bodies.'}))
    st.body = self.rfile.read(length) if length else None
    st.model_alias = None
    st.payload = None
    st.rewritten = False
    if remote and inference and not st.body:
        return answered(self.json_reply(400, {'error': 'Invalid model request.'}))
    if (inference or helper) and st.body:
        try:
            raw = st.body
            if self.headers.get('Content-Encoding', '').lower() == 'gzip':
                with gzip.GzipFile(fileobj=io.BytesIO(st.body)) as stream: raw = stream.read(16 * 1024**2 + 1)
                if len(raw) > 16 * 1024**2: raise ValueError()
            st.payload = json.loads(raw)
            if remote:
                # O-102: a shared request carries the OpenAI fields only (no slot, cache or
                # native engine controls, no URLs to fetch) and a bounded output limit
                st.payload = lan_share.remote_payload(st.payload, self.server.share_listener.controller.opts['max_tokens'])
                st.body = json.dumps(st.payload).encode(); st.rewritten = True
            routed = explicit_route.rewrite(st.payload, enabled=os.environ.get(explicit_route.ENABLE_ENV) == '1')
            if routed is not st.payload:
                st.payload = routed
                st.body = json.dumps(st.payload).encode(); st.rewritten = True
            if isinstance(st.payload, dict) and 'model' in st.payload:
                if not isinstance(st.payload['model'], str): raise ValueError()
                st.model_alias = self.manager.config(st.payload['model'])['model_alias']
                st.payload['model'] = st.model_alias
                st.body = json.dumps(st.payload).encode(); st.rewritten = True
        except (ValueError, OSError, EOFError): return answered(self.json_reply(400, {'error': 'Invalid model request.'}))
        except Unavailable as error:
            return answered(self.json_reply(404, {'error': {'message': str(error), 'type': 'invalid_request_error',
                                                            'param': 'model', 'code': 'model_not_found'}}))
    try:
        st.target = self.manager.config(st.model_alias) if st.model_alias is not None else (self.manager.loaded_config or self.manager.config())
    except Unavailable as error:
        return answered(self.json_reply(503, {'error': {'message': lan_share.public_message(str(error)) if remote else str(error),
                                                        'type': 'local_backend_busy'}}))
    st.harness_key = None
    st.harness = None
    st.turn = None  # CACHE-PERFECT turn labels (cache_turn.py), oxe chat requests only
    st.tool_cap = None
    if inference and not remote and isinstance(st.payload, dict) and path in harness_compat.CHAT_PATHS:
        # v0.1.3 TOOL-OUTPUT CAP (LOCAL_AI_TOOL_OUTPUT_CAP=on; off = untouched): big new tool
        # results keep head/tail/error lines, the rest is stored for the read-more tool
        try:
            capped, st.tool_cap = tool_output_cap.cap_request(st.payload, self.headers)
        except Exception:  # never fail a request over the cap; it is an optimisation
            capped, st.tool_cap = st.payload, None
        if capped is not st.payload:
            st.payload = capped
            st.body = json.dumps(st.payload).encode(); st.rewritten = True
    st.backend_path = self.path
    st.completion_meta = None
    text_only = st.target.get('backend_kind') in (oxe_runtime.KIND, bigmoe_runtime.KIND)
    if text_only and self.command == 'POST' and path in openai_api.UNSUPPORTED:
        # Answered here: no model is loaded for an endpoint the engine does not have.
        # 404, not 501: client libraries retry every 5xx.
        return answered(self.json_reply(404, {'error': {
            'message': 'This local model server does not provide %s (%s).' % (openai_api.UNSUPPORTED[path], path),
            'type': 'invalid_request_error', 'code': 'unsupported_endpoint'}}))
    if inference and isinstance(st.payload, dict):
        if st.target.get('backend_kind') == oxe_runtime.KIND:
            st.payload, compat_notes = harness_compat.normalize(st.payload, path)
            compat_notes = list(compat_notes)
            st.payload = openai_api.single_choice(st.payload, compat_notes)
            st.payload = oxe_runtime.request_defaults(st.payload, path, st.target)
            if path == '/v1/completions':
                # The Rust engine has llama-server's native /completion only.
                try: st.payload, st.completion_meta = openai_api.completion_request(st.payload)
                except openai_api.Unsupported as error:
                    return answered(self.json_reply(400, {'error': {'message': str(error), 'type': 'invalid_request_error'}}))
                st.backend_path = '/completion'
            st.body = json.dumps(st.payload).encode(); st.rewritten = True
            st.harness = None
            if path in harness_compat.CHAT_PATHS:
                st.harness = harness_compat.identify(st.payload, self.headers.get('User-Agent'))
                try:
                    st.turn = cache_turn.label(st.payload)
                except Exception:  # labels are diagnostic; never fail the request
                    st.turn = None
            with self.manager.harness_mu:
                self.manager.compat_stats.update(compat_notes)
                # a shared request is never a candidate for the harness learning registry:
                # another device must not add, rename or re-time this machine's learned prefixes
                if st.harness is not None and not remote:
                    st.harness_key = self.manager.harness_registry.observe(
                        st.payload, st.harness, st.target, learning=learning_settings(st.api_value),
                        user_agent=self.headers.get('User-Agent'))
        elif st.target.get('backend_kind') == bigmoe_runtime.KIND and path in harness_compat.CHAT_PATHS:
            compat_notes = []
            st.payload = openai_api.flash_request(st.payload, compat_notes)
            if compat_notes:
                with self.manager.harness_mu: self.manager.compat_stats.update(compat_notes)
        else:
            defaulted = ctl.request_defaults(st.target, st.payload, path)
            if remote:
                # O-102 M2 (llama-server): no slot prefix reuse for a shared request (no
                # prefix-timing oracle); the oxe engine gets X-Oxe-State-Policy instead
                defaulted = {**defaulted, 'cache_prompt': False}
            if defaulted is not st.payload:
                st.payload = defaulted
                st.body = json.dumps(st.payload).encode(); st.rewritten = True
    streamed = st.target.get('backend_kind') == bigmoe_runtime.KIND
    if streamed:
        if path == '/' and self.command in ('GET', 'HEAD'):
            if not self.manager.eligible(st.target):
                return answered(self.json_reply(503, {'error': 'Flash profile is not qualified.'}))
            page = FLASH_CHAT_HTML.replace(b'{MODEL}', html.escape(st.target['model_alias'], quote=True).encode())
            return answered(self.reply(200, page, 'text/html; charset=utf-8'))
        # the route by its path; a query on a Flash route is refused (fail closed)
        if self.command != 'POST' or path != '/v1/chat/completions' or self.path != path:
            self.close_connection = True
            return answered(self.json_reply(403, {'error': 'This Flash route is not exposed by Local AI.'}))
        try:
            st.payload = bigmoe_runtime.normalize_chat(st.target, st.payload, ctl.FLASH_BACKEND_PORT)
            st.body = json.dumps(st.payload).encode(); st.rewritten = True
        except (ValueError, OSError, TypeError, KeyError):
            return answered(self.json_reply(400, {'error': 'Invalid Flash chat request or unqualified reasoning settings.'}))
    elif self.command in ('GET', 'HEAD') and path != '/' and self.path == path and path in self.manager.cache:
        # only a target without a query is a cached resource ('/props?x' goes to the engine)
        content_type, data = self.manager.cache[path]
        return answered(self.reply(200, data, content_type))
    return None


def backend_request(self, st):
    """After admission: the engine body for a runtime, the prefix store, the connection and the
    header rules for the internal backend port (st.connection, st.backend_headers). On any error
    the slot is released here and the client answered (ANSWERED)."""
    path, remote, inference, helper = st.path, st.remote, st.inference, st.helper
    # F80: from admission to the response's own try/finally, any exception must release the
    # slot, or every later request waits in the FIFO until its 503 (stuck queue)
    try:
        st.request_engine_config = (getattr(st.request_process, '_local_ai_engine_config', None)
                                    or self.manager.loaded_config)
        st.runtime = self.manager.backend_runtime
        if st.runtime is not None:
            try:
                if st.payload.get('model') != st.runtime.spec.backend_model_id:
                    st.payload = st.runtime.chat_body(st.payload)
                st.body = json.dumps(st.payload).encode(); st.rewritten = True
            except (ValueError, TypeError, KeyError):
                raise _BadFlashChat() from None
        elif inference and remote:
            # no prefix-store priming or restore for a shared request: it must not write to
            # (or evict from) this machine's prefix files; slot 0 no longer holds a known prefix
            self.manager.current_prefix_key = None
        elif inference:
            self.manager.prepare_prefix(path, st.payload)
        backend_port = st.runtime.spec.port if st.runtime is not None else ctl.BACKEND_PORT
        st.connection = http.client.HTTPConnection('127.0.0.1', backend_port, timeout=ctl.BACKEND_READ_TIMEOUT_SECONDS)  # v0.1.3 LONG-APPEND
        headers = {k: v for k, v in self.headers.items()
                   if k.lower() not in HOP_HEADERS | {'host', 'accept-encoding'}
                   # client credentials never reach an engine: the local API key is checked here,
                   # and each backend gets the controller's own auth (launch token, runtime headers)
                   and k.lower() not in {'authorization', 'x-api-key', 'api-key', 'cookie'}
                   and (st.runtime is None or k.lower() != 'origin')
                   # engine controls are the controller's to set (policy, session hint, turn kind),
                   # for every backend; a client's own copy never reaches one
                   and not k.lower().startswith('x-oxe-')}
        if st.rewritten:
            headers = {k: v for k, v in headers.items() if k.lower() != 'content-encoding'}
        headers['Host'] = '127.0.0.1:' + str(backend_port)
        if st.runtime is None:
            # B1: the backend port is internal. Browser context headers never reach it (the
            # engine refuses any Origin or cross-site fetch); the controller already applied
            # its own Host/Origin/CORS rules. An oxe engine gets its per-launch token instead of
            # whatever credentials the client sent.
            headers = {k: v for k, v in headers.items()
                       if k.lower() != 'origin' and not k.lower().startswith('sec-fetch-')}
            if remote:
                # O-102 M2: RAM-only, request-scoped engine state for another device's request:
                # no state-cache or echo read/write, no prefix shared with a local session (the
                # engine reads this header only from the controller's keyed loopback connection;
                # a client's own copy was removed above). No session hint or turn label.
                headers['X-Oxe-State-Policy'] = 'ephemeral'
            elif st.turn is not None:
                # CACHE-PERFECT: session hint (the ledger's keyed hash, never a cache key) and turn kind
                try:
                    hint = st.led.session if st.led is not None else request_ledger.session_key(self.headers, st.payload)
                except Exception:
                    hint = None
                headers.update(cache_turn.backend_headers(hint, st.turn))
                hints = getattr(self.manager, 'session_hints', None)  # for the next explicit load's preload
                if isinstance(hints, dict) and isinstance(hint, str) and isinstance(st.target.get('model_alias'), str):
                    hints[st.target['model_alias']] = hint
            backend_key = getattr(st.request_process, '_local_ai_api_key', None)
            if backend_key:
                headers = {k: v for k, v in headers.items()
                           if k.lower() not in {'authorization', 'x-api-key', 'api-key'}}
                headers.update(oxe_runtime.backend_headers(backend_key))
        if st.runtime is not None:
            if self.headers.get('Origin') is not None:
                headers['Origin'] = f'http://127.0.0.1:{backend_port}'
            headers.update(st.runtime.headers())
            headers['Content-Type'] = 'application/json'
        st.backend_headers = headers
    except _BadFlashChat:
        # sibling clauses: an exception raised while replying here is not caught by the next
        # clause, so the slot is released exactly once on every path
        self.manager.end(inference=inference, touch=helper)
        if st.led is not None: st.led.finish(400, 'bad_request')
        return answered(self.json_reply(400, {'error': 'Invalid Flash chat request or unqualified reasoning settings.'}))
    except Exception:
        self.manager.end(inference=inference, touch=helper)
        if st.led is not None: st.led.finish(500, 'local_controller_error')
        self.close_connection = True
        return answered(self.json_reply(500, {'error': {'message': 'Local controller error before the request reached the engine.',
                                                        'type': 'local_controller_error'}}))
    if self.command in ('GET', 'HEAD') and not path.startswith(('/v1/', '/props', '/slots', '/metrics', '/health', '/models')):
        # llama.cpp embeds its UI as gzip-only assets. Decode them below so
        # both browser clients and the in-memory cache get ordinary bytes.
        st.backend_headers['Accept-Encoding'] = 'gzip'
    return None
