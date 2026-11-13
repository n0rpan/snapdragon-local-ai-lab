"""Proxy stage 2, forwarding and streaming: connect (and attach a shared request's preempt), send,
the client-disconnect watcher, the response classification (gzip UI assets, cache, the
/v1/completions shim), the chunk loop, and the client, backend and preempt failure replies.
The bookkeeping that always follows is stage 3 (ctl_proxy_ledger.finish).
"""
import gzip
import http.client
import json
if __package__:
    from .ctl_proxy_prep import HOP_HEADERS
    from .ctl_seams import ctl
    from .harness_registry import timings_from_tail
    from . import lan_share, openai_api
else:
    from ctl_proxy_prep import HOP_HEADERS
    from ctl_seams import ctl
    from harness_registry import timings_from_tail
    import lan_share
    import openai_api


def backend_failed(self, st):
    native_died = st.request_process is not None and st.request_process.poll() is not None
    if native_died:
        self.manager.record_engine_death(st.request_process, st.request_engine_config)
    st.request_error = 'backend_died'
    if not st.started_response:
        self.json_reply(502, {'error': {'message': 'Local backend connection closed.', 'type': 'backend_died'}})
    else:
        self.close_connection = True


def write_client(self, st, data):
    try:
        self.wfile.write(data)
        self.wfile.flush()
    except (BrokenPipeError, ConnectionResetError):
        st.client_write_failed = True
        raise


def reply_preempted(self, st):
    """O-102 M4: the shared request was cancelled for a local one: an error for the remote
    client (503 before the response started, an SSE error event in a started stream, else a
    closed, incomplete response) and the connection is closed."""
    st.request_error = 'preempted'
    self.close_connection = True
    try:
        if not st.started_response:
            if st.watcher is not None:
                st.watcher.stop()
            self.json_reply(503, {'error': dict(lan_share.PREEMPTED)})
        elif 'text/event-stream' in (st.content_type or ''):
            data = b'data: ' + json.dumps({'error': dict(lan_share.PREEMPTED)}).encode() + b'\n\n'
            write_client(self, st, ('%x\r\n' % len(data)).encode() + data + b'\r\n0\r\n\r\n')
    except OSError:
        pass


def reply_client(self, st, status, data, content_type):
    if st.watcher is not None:
        st.watcher.stop()
    try:
        return self.reply(status, data, content_type)
    except (BrokenPipeError, ConnectionResetError):
        st.client_write_failed = True
        raise


def forward(self, st):
    """Send the prepared request to the engine and stream its answer to the client."""
    path, inference = st.path, st.inference
    connection = st.connection
    try:
        if st.preempt is not None:
            # attached before the request is sent: a local request arriving from now on cuts
            # the engine connection (a cancel that came earlier cuts it here at once)
            connection.connect()
            st.preempt.attach(connection.sock)
            if st.preempt.cancelled:
                raise ConnectionAbortedError('preempted')
        connection.request(self.command, st.backend_path, body=st.body, headers=st.backend_headers)
        # the watcher peeks at the client socket (MSG_PEEK), which a TLS socket refuses: on a
        # shared TLS listener it would cancel every request at once, so a vanished TLS client
        # is noticed by the failed (or timed-out) write instead
        if inference and connection.sock is not None and not lan_share.is_tls(self.connection):
            st.watcher = ctl.ClientDisconnect(self.connection, connection.sock).start()
        response = connection.getresponse()
        st.backend_status = response.status
        if inference and response.status >= 400: self.manager.current_prefix_key = None
        if response.status != 200: st.harness_key = None
        tail = b''
        st.content_type = response.getheader('Content-Type', 'application/octet-stream')
        # Static app resources and /props are safe to cache; never cache slots or user responses.
        # Keyed by the path, so only a target without a query is stored (never '/props?x' as '/props').
        cacheable = self.command == 'GET' and response.status == 200 and self.path == path and (
            path == '/props' or path.endswith(('.js', '.css', '.svg', '.ico', '.woff', '.woff2', '.html')))
        if response.getheader('Content-Encoding', '').lower() == 'gzip' and self.command != 'HEAD':
            try:
                data = gzip.decompress(response.read())
            except (OSError, EOFError, ValueError):  # N6: a corrupt gzip body is a 502, not a crash
                return reply_client(self, st, 502, json.dumps({'error': {'message': 'The local backend sent a corrupt '
                                    'compressed response.', 'type': 'local_backend_error'}}).encode(), 'application/json')
            if cacheable: self.manager.cache[path] = (st.content_type, data)
            return reply_client(self, st, response.status, data, st.content_type)
        if cacheable:
            data = response.read()
            self.manager.cache[path] = (st.content_type, data)
            return reply_client(self, st, response.status, data, st.content_type)
        shim = None
        if st.completion_meta is not None and response.status == 200:
            if not st.completion_meta['stream']:
                data = response.read()
                if st.led is not None: st.led.observe(data)
                try: data = json.dumps(openai_api.completion_response(json.loads(data), st.completion_meta)).encode()
                except ValueError: pass
                return reply_client(self, st, 200, data, 'application/json')
            shim = openai_api.CompletionStream(st.completion_meta)
        try:
            self.send_response(response.status)
            for k, v in response.getheaders():
                # CORS is decided here (openai_api.cors_origins), never by a backend.
                if k.lower().startswith('access-control-'): continue
                if k.lower() not in HOP_HEADERS and (st.runtime is None or k.lower() not in {'authorization', 'x-api-key', 'api-key', 'set-cookie'}): self.send_header(k, v)
            for k, v in self.cors_reply_headers(): self.send_header(k, v)
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Transfer-Encoding', 'chunked')
            self.end_headers()
        except (BrokenPipeError, ConnectionResetError):
            st.client_write_failed = True
            raise
        st.started_response = True
        if self.command != 'HEAD':
            while chunk := response.read1(65536):
                if st.harness_key is not None: tail = (tail + chunk)[-4096:]
                if st.led is not None: st.led.observe(chunk)
                if shim is not None:
                    chunk = shim.feed(chunk)
                    if not chunk: continue
                write_client(self, st, ('%x\r\n' % len(chunk)).encode() + chunk + b'\r\n')
            if shim is not None and (chunk := shim.finish()):
                write_client(self, st, ('%x\r\n' % len(chunk)).encode() + chunk + b'\r\n')
            if st.watcher is not None:
                st.watcher.stop()
            if st.preempted():
                # the engine connection was cut for a local request: not a complete answer
                reply_preempted(self, st)
            elif st.request_process is not None and st.request_process.poll() is not None:
                # Headers already escaped; a closed, incomplete response is
                # honest, whereas a final chunk would disguise engine death.
                st.request_error = 'backend_died'
                self.close_connection = True
            else:
                write_client(self, st, b'0\r\n\r\n')
                if st.harness_key is not None:
                    with self.manager.harness_mu:
                        self.manager.harness_registry.complete(st.harness_key, st.target, *timings_from_tail(tail))
    except (BrokenPipeError, ConnectionResetError):
        if st.client_gone():
            self.close_connection = True
        elif st.preempted():
            reply_preempted(self, st)
        else:
            backend_failed(self, st)
    except (OSError, http.client.HTTPException) as error:
        if st.client_gone():
            self.close_connection = True
        elif st.preempted():
            reply_preempted(self, st)
        else:
            if inference: self.manager.current_prefix_key = None
            if isinstance(error, http.client.RemoteDisconnected) or (
                    st.request_process is not None and st.request_process.poll() is not None):
                backend_failed(self, st)
            elif not st.started_response:
                self.json_reply(503, {'error': {'message': 'Local backend connection failed.', 'type': 'local_backend_error'}})
            else:
                self.close_connection = True
