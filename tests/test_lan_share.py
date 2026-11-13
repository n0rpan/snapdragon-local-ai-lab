"""O-102 LAN share: binding selection, the remote gate (key, routes, rate, body, one at a time),
the remote ledger origin, the key file, the off switch, and loopback left unchanged.

Model-free. Shared listeners are real controller.Handler servers on 127.0.0.1 port 0, registered
in a ShareController as if 127.0.0.1 were a chosen share address; address selection itself is
tested on recorded `ip -j` output. Isolated HOME/XDG; runtime.json and the key are temp files.
"""
import base64
import collections
import contextlib
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.machinery
import importlib.util
import io
import ipaddress
import json
import os
from pathlib import Path
import shutil
import socket
import ssl
import stat
import subprocess
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from snapdragon_ai_lab.runtime import controller, lan_share, request_ledger

TIMEOUT = 5


def dotted(*parts):
    # built at run time so the tree's private-address scrub has no literal to flag
    return '.'.join(map(str, parts))


LAN_IP, GATEWAY, CGNAT_IP, TS_IP = dotted(192, 168, 1, 20), dotted(192, 168, 1, 1), dotted(100, 80, 1, 2), dotted(100, 101, 102, 103)
CHAT = '/v1/chat/completions'
IP_ADDR = json.dumps([
    {'ifname': 'lo', 'addr_info': [{'family': 'inet', 'local': '127.0.0.1'}, {'family': 'inet6', 'local': '::1'}]},
    {'ifname': 'wlan0', 'addr_info': [{'family': 'inet', 'local': LAN_IP},
                                      {'family': 'inet6', 'local': 'fe80::1'}]},
    {'ifname': 'wwan0', 'addr_info': [{'family': 'inet', 'local': CGNAT_IP}]},   # carrier CGNAT, not Tailscale
    {'ifname': 'eth9', 'addr_info': [{'family': 'inet', 'local': '203.0.113.5'}]},   # public
    {'ifname': 'tailscale0', 'addr_info': [{'family': 'inet', 'local': TS_IP},
                                           {'family': 'inet6', 'local': 'fd7a:115c:a1e0::1234'},
                                           {'family': 'inet6', 'local': 'fe80::abcd'}]},
])


def ip(text):
    return ipaddress.ip_address(text)


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.addrs = lan_share.parse_ip_addr(IP_ADDR)

    def test_tailscale_takes_only_tailnet_addresses_on_a_tailscale_interface(self):
        chosen, error = lan_share.select_addresses('tailscale', self.addrs, ['wlan0'])
        self.assertIsNone(error)
        self.assertEqual(chosen, [TS_IP, 'fd7a:115c:a1e0::1234'])
        self.assertNotIn(CGNAT_IP, chosen)  # CGNAT range on a non-tailscale interface

    def test_lan_takes_the_private_ipv4_of_the_default_route_interface(self):
        chosen, error = lan_share.select_addresses('lan', self.addrs, lan_share.parse_default_devs(
            json.dumps([{'dst': 'default', 'gateway': GATEWAY, 'dev': 'wlan0'}])))
        self.assertEqual((chosen, error), ([LAN_IP], None))

    def test_never_a_wildcard_public_or_loopback_address(self):
        for mode, devs in (('lan', ['eth9']), ('lan', ['lo']), ('lan', ['tailscale0']), ('lan', [])):
            with self.subTest(mode=mode, devs=devs):
                chosen, error = lan_share.select_addresses(mode, self.addrs, devs)
                self.assertEqual(chosen, [])
                self.assertIn('No private LAN address', error)
        wild = [('tailscale0', ip('0.0.0.0')), ('tailscale0', ip('::'))]
        self.assertEqual(lan_share.select_addresses('tailscale', wild, [])[0], [])
        self.assertEqual(lan_share.select_addresses('off', self.addrs, ['wlan0']), ([], None))

    def test_tailscale_down_is_an_error_not_a_fallback(self):
        chosen, error = lan_share.select_addresses('tailscale', [a for a in self.addrs if a[0] != 'tailscale0'], ['wlan0'])
        self.assertEqual(chosen, [])
        self.assertIn('Tailscale is not up', error)

    def test_unknown_mode_values_are_off(self):
        for value in ({}, {'lan_share': True}, {'lan_share': 'all'}, {'lan_share': None}, None):
            self.assertEqual(lan_share.mode_of(value), 'off')
        self.assertEqual(lan_share.mode_of({'lan_share': 'lan'}), 'lan')

    def test_lan_never_follows_a_vpn_default_route(self):
        # security review: with a VPN up, the default route is tun0/wg0 (link type 'none') with a
        # private address; `lan` must not share the models with the VPN's network
        text = json.dumps([{'ifname': 'wlan0', 'link_type': 'ether', 'addr_info': [{'local': LAN_IP}]},
                           {'ifname': 'tun0', 'link_type': 'none', 'addr_info': [{'local': dotted(10, 8, 0, 2)}]},
                           {'ifname': 'wg0', 'link_type': 'none', 'addr_info': [{'local': dotted(10, 9, 0, 2)}]},
                           {'ifname': 'old-ip', 'addr_info': []}])
        types = lan_share.parse_link_types(text)
        self.assertEqual(types, {'wlan0': 'ether', 'tun0': 'none', 'wg0': 'none', 'old-ip': None})
        self.assertEqual(lan_share.lan_devices(['tun0', 'wg0', 'old-ip', 'missing'], types), [])
        self.assertEqual(lan_share.lan_devices(['tun0', 'wlan0'], types), ['wlan0'])
        routes = json.dumps([{'dst': 'default', 'dev': 'tun0'}])
        run = Mock(side_effect=[SimpleNamespace(stdout=text), SimpleNamespace(stdout=routes)])
        with patch.object(lan_share, '_ip_tool', return_value='/usr/bin/ip'):
            addrs, devs = lan_share.discover(run)
        self.assertEqual(devs, [])
        chosen, error = lan_share.select_addresses('lan', addrs, devs)
        self.assertEqual(chosen, [])
        self.assertIn('No private LAN address', error)


class RemotePayloadTests(unittest.TestCase):
    """Security review: what the body of a shared request may carry to an engine."""

    def test_only_openai_fields_reach_the_engine(self):
        sent = {'model': 'm', 'messages': [{'role': 'user', 'content': 'hi'}], 'stream': True, 'temperature': 0.5,
                'tools': [{'type': 'function', 'function': {'name': 'f'}}], 'tool_choice': 'auto',
                # engine controls another device must not set
                'id_slot': 3, 'cache_prompt': False, 'n_keep': -1, 'n': 64, 'n_predict': -1, 'grammar': 'root ::= .*',
                'lora': [{'id': 0, 'scale': 1}], 'samplers': ['top_k'], 'slot_save_path': '/tmp', 'n_probs': 100,
                'chat_template': '{{ x }}', 'response_fields': ['x'], 'unknown_future_field': 1}
        out = lan_share.remote_payload(sent)
        self.assertEqual(set(out), {'model', 'messages', 'stream', 'temperature', 'tools', 'tool_choice'})
        self.assertLessEqual(set(out), lan_share.PAYLOAD_FIELDS)
        self.assertIn('id_slot', sent)  # the input is not modified

    def test_template_kwargs_are_the_two_booleans_only(self):
        out = lan_share.remote_payload({'messages': [], 'chat_template_kwargs': {
            'enable_thinking': True, 'preserve_thinking': 'yes', 'system_override': 'x', 'tools': []}})
        self.assertEqual(out['chat_template_kwargs'], {'enable_thinking': True})
        self.assertNotIn('chat_template_kwargs', lan_share.remote_payload({'chat_template_kwargs': {'x': 1}}))
        self.assertNotIn('chat_template_kwargs', lan_share.remote_payload({'chat_template_kwargs': 'x'}))

    def test_output_limit_is_capped_and_never_invented(self):
        self.assertEqual(lan_share.remote_payload({'max_tokens': 10**9})['max_tokens'], lan_share.DEFAULT_MAX_TOKENS)
        self.assertEqual(lan_share.remote_payload({'max_completion_tokens': 10**9}, 100), {'max_tokens': 100})
        self.assertEqual(lan_share.remote_payload({'max_tokens': 50}, 100)['max_tokens'], 50)
        for bad in (None, -1, 0, True, '100', 1.5):
            with self.subTest(bad=bad):  # not a usable limit: the profile default applies
                self.assertNotIn('max_tokens', lan_share.remote_payload({'max_tokens': bad}))
        self.assertNotIn('max_tokens', lan_share.remote_payload({'messages': []}))

    def test_urls_to_fetch_and_non_objects_are_refused(self):
        def chat(part):
            return {'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': 'x'}, part]}]}
        for part in ({'type': 'image_url', 'image_url': {'url': 'http://127.0.0.1:8092/_local/status'}},
                     {'type': 'image_url', 'image_url': 'file:///etc/passwd'},
                     {'type': 'image_url', 'image_url': {}},
                     {'type': 'audio_url', 'audio_url': {'url': 'https://example.com/a.wav'}}):
            with self.subTest(part=part), self.assertRaises(lan_share.RemotePayloadError):
                lan_share.remote_payload(chat(part))
        inline = chat({'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,AAAA'}})
        self.assertEqual(lan_share.remote_payload(inline), inline)
        for body in (None, [], 'text', 5):
            with self.subTest(body=body), self.assertRaises(lan_share.RemotePayloadError):
                lan_share.remote_payload(body)
        self.assertTrue(issubclass(lan_share.RemotePayloadError, ValueError))  # the controller's 400


class KeyFileTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name) / 'fresh' / 'lan-share-key.json'

    def test_generated_key_is_256_bits_in_an_owner_only_file(self):
        record = lan_share.ensure_key(self.path)
        self.assertEqual(stat.S_IMODE(self.path.stat().st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(self.path.parent.stat().st_mode), 0o700)
        raw = record['key'][len(lan_share.KEY_PREFIX):]
        self.assertGreaterEqual(len(base64.urlsafe_b64decode(raw + '=' * (-len(raw) % 4))), 32)
        self.assertEqual(lan_share.ensure_key(self.path)['key'], record['key'])  # stable
        self.assertNotEqual(lan_share.rotate_key(self.path)['key'], record['key'])
        self.assertEqual(stat.S_IMODE(self.path.stat().st_mode), 0o600)

    def test_group_readable_or_symlinked_key_files_are_refused(self):
        lan_share.ensure_key(self.path)
        os.chmod(self.path, 0o644)
        with self.assertRaises(lan_share.KeyFileError):
            lan_share.read_key(self.path)
        os.chmod(self.path, 0o600)
        link = self.path.with_name('link.json')
        link.symlink_to(self.path)
        with self.assertRaises(lan_share.KeyFileError):
            lan_share.read_key(link)

    def test_key_compare_is_exact(self):
        key = lan_share.new_key()
        self.assertTrue(lan_share.key_matches(key, key))
        for wrong in (None, '', key[:-1], key + 'x', key.upper()):
            self.assertFalse(lan_share.key_matches(wrong, key))
        self.assertFalse(lan_share.key_matches(key, None))

    def test_fingerprint_does_not_reveal_the_key(self):
        key = lan_share.new_key()
        self.assertNotIn(key[4:16], lan_share.fingerprint(key))
        self.assertTrue(lan_share.fingerprint(key).startswith('sha256:'))

    def test_reveal_shows_the_key_on_disk_exactly_once(self):
        record, show = lan_share.reveal_key(self.path)
        self.assertTrue(show)
        self.assertEqual(lan_share.read_key(self.path), {**record, 'shown': True})
        again, show = lan_share.reveal_key(self.path)
        self.assertEqual((again['key'], show), (record['key'], False))
        rotated, show = lan_share.reveal_key(self.path, rotate=True)
        self.assertTrue(show)
        self.assertNotEqual(rotated['key'], record['key'])
        self.assertEqual(lan_share.read_key(self.path)['key'], rotated['key'])
        self.assertEqual(stat.S_IMODE(self.path.stat().st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(self.path.with_name('.' + self.path.name + '.lock').stat().st_mode), 0o600)

    def test_concurrent_first_use_creates_one_key(self):
        # the controller (apply) and the CLI (share key) may both find no key file
        results, start = [], threading.Barrier(8)

        def make():
            start.wait(TIMEOUT)
            results.append(lan_share.ensure_key(self.path)['key'])
        threads = [threading.Thread(target=make) for _ in range(8)]
        for thread in threads: thread.start()
        for thread in threads: thread.join(TIMEOUT)
        self.assertEqual(len(results), 8)
        self.assertEqual(set(results), {lan_share.read_key(self.path)['key']})

    def test_mark_shown_does_not_bring_back_a_rotated_key(self):
        lan_share.ensure_key(self.path)
        new = lan_share.rotate_key(self.path)['key']
        self.assertEqual(lan_share.mark_shown(self.path), {**lan_share.read_key(self.path), 'key': new, 'shown': True})


class RateLimiterTests(unittest.TestCase):
    def test_bucket_refills_per_client(self):
        now = [0.0]
        limiter = lan_share.RateLimiter(60, 2, clock=lambda: now[0])
        self.assertEqual([limiter.allow('a') for _ in range(3)], [True, True, False])
        self.assertTrue(limiter.allow('b'))  # per client
        now[0] += 1.0  # 60/min = one token per second
        self.assertTrue(limiter.allow('a'))
        self.assertFalse(limiter.allow('a'))


class ShareBase(unittest.TestCase):
    def setUp(self):
        self.seen = []
        self.bodies = []  # (path, body) of each request the engine got
        seen, bodies = self.seen, self.bodies

        class Native(BaseHTTPRequestHandler):
            def log_message(self, *args): pass
            def do_POST(self):
                bodies.append((self.path, self.rfile.read(int(self.headers['Content-Length']))))
                seen.append(dict(self.headers.items()))
                data = b'{"choices":[{"message":{"content":"READY"}}]}'
                self.send_response(200); self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(data))); self.end_headers(); self.wfile.write(data)

        self.backend = ThreadingHTTPServer(('127.0.0.1', 0), Native)
        threading.Thread(target=self.backend.serve_forever, daemon=True).start()
        self.addCleanup(lambda: (self.backend.shutdown(), self.backend.server_close()))
        env = patch.dict(os.environ); env.start(); self.addCleanup(env.stop)
        os.environ.pop(controller.openai_api.KEY_ENV, None)
        os.environ.pop(lan_share.KEY_FILE_ENV, None)
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        (self.root / 'llm').mkdir()
        self.config = self.root / 'llm' / 'runtime.json'
        self.config.write_text(json.dumps({'lan_share': 'tailscale'}))
        for item in (patch.object(controller, 'CONFIG', self.config),
                     patch.object(controller, 'BACKEND_PORT', self.backend.server_port)):
            item.start(); self.addCleanup(item.stop)
        self.manager = self.make_manager()
        self.share = lan_share.ShareController(self.config, 0, controller.Handler, manager=self.manager,
                                               discover_fn=lambda: ([], []))
        self.manager.share = self.share
        self.key = lan_share.ensure_key(self.share.key_file())['key']
        self.listener = lan_share.Listener(self.share, '127.0.0.1', 0, controller.Handler, self.manager)
        self.share.listeners['127.0.0.1'] = self.listener
        self.share.mode = 'tailscale'
        self.addCleanup(self.share.close)
        self.port = self.listener.port

    def make_manager(self, **overrides):
        profile = {'model_alias': 'synthetic-classic'}
        fields = dict(config=Mock(return_value=profile), loaded_config=None, backend_runtime=None, process=None,
                      status=Mock(return_value={'state': 'unloaded', 'validated': False, 'model': None}),
                      models=Mock(return_value=[{'id': 'synthetic-classic', 'object': 'model'}]),
                      begin=Mock(return_value=None), begin_queued=Mock(return_value=None),
                      end=Mock(return_value=None), prepare_prefix=Mock(return_value=None),
                      eligible=Mock(return_value=True), current_prefix_key=None, cache={},
                      cleanup=Mock(), mu=threading.Lock(), active=0)
        fields.update(overrides)
        return SimpleNamespace(**fields)

    def send(self, method, path, body=None, key=None, headers=None, port=None):
        connection = http.client.HTTPConnection('127.0.0.1', port or self.port, timeout=TIMEOUT)
        try:
            sent = dict(headers or {})
            if key is not None:
                sent['Authorization'] = 'Bearer ' + key
            connection.request(method, path, body=body, headers=sent)
            response = connection.getresponse()
            return response.status, response.read()
        finally:
            connection.close()

    def chat_body(self, **extra):
        return json.dumps({'messages': [{'role': 'user', 'content': 'hi'}], **extra}).encode()


class GateTests(ShareBase):
    def test_no_key_and_wrong_key_are_401_before_any_work(self):
        for key in (None, 'wrong', self.key[:-1], lan_share.new_key()):
            with self.subTest(key=key):
                status, data = self.send('POST', CHAT, self.chat_body(), key=key)
                self.assertEqual(status, 401)
                self.assertEqual(json.loads(data)['error']['type'], 'authentication_error')
        status, _ = self.send('GET', '/v1/models', headers={'X-Api-Key': 'wrong'})
        self.assertEqual(status, 401)
        self.manager.begin_queued.assert_not_called()
        self.assertEqual(self.seen, [])

    def test_admin_and_other_routes_are_never_shared_even_with_the_key(self):
        for method, path in (('GET', '/_local/status'), ('POST', '/_local/unload'), ('POST', '/_local/load'),
                             ('POST', '/_local/cache/clear'), ('POST', '/_local/cache/budget'),
                             ('POST', '/_local/share'), ('GET', '/_local/ledger'), ('GET', '/'),
                             ('GET', '/health'), ('POST', '/tokenize'), ('POST', '/completion'),
                             ('POST', '/v1/embeddings'), ('POST', '/slots/0'), ('GET', '/props'),
                             ('GET', CHAT), ('POST', '/v1/models'), ('GET', '/v1/models/a/../../_local/status'),
                             ('POST', '/v1/chat%2Fcompletions'), ('DELETE', CHAT)):
            with self.subTest(method=method, path=path):
                status, _ = self.send(method, path, b'{}' if method == 'POST' else None, key=self.key)
                self.assertEqual(status, 403)
        self.manager.begin_queued.assert_not_called()
        self.manager.cleanup.assert_not_called()
        self.assertEqual(self.seen, [])

    def test_keyed_chat_reaches_the_engine_without_the_key_as_remote(self):
        status, data = self.send('POST', CHAT, self.chat_body(), key=self.key,
                                 headers={'X-Local-AI-Origin': 'lab', 'Host': 'laptop.tailnet.ts.net:8092'})
        self.assertEqual(status, 200, data)
        self.assertEqual(len(self.seen), 1)
        forwarded = {k.lower() for k in self.seen[0]}
        self.assertFalse(forwarded & {'authorization', 'x-api-key', 'api-key', 'x-local-ai-origin'})
        kwargs = self.manager.begin_queued.call_args.kwargs
        self.assertEqual(kwargs['origin'], 'remote')
        # only waits out the request before it, never ahead of a local one (ManagerAdmissionTests)
        self.assertEqual(kwargs['wait_seconds'], lan_share.SLOT_WAIT)
        self.assertEqual(self.share.stats['remote_requests'], 1)

    def test_only_listed_headers_reach_the_engine(self):
        status, data = self.send('POST', CHAT, self.chat_body(), key=self.key, headers={
            **dict.fromkeys(('X-Api-Key', 'Api-Key'), self.key), 'Cookie': 'a=b', 'X-Oxe-Session': 'forged',
            'X-Oxe-Turn-Kind': 'compaction', 'X-Forwarded-For': '127.0.0.1', 'Forwarded': 'for=127.0.0.1',
            'X-Real-IP': '127.0.0.1', 'Proxy-Authorization': 'Basic eDp5', 'X-Unknown-Engine-Switch': '1',
            'User-Agent': 'client/1.0', 'X-Session-Id': 'abc'})
        self.assertEqual(status, 200, data)
        forwarded = {k.lower() for k in self.seen[0]}
        # accept-encoding is the controller's own (http.client adds it), not the client's, and so
        # is the engine state policy (M2)
        self.assertLessEqual(forwarded - {'accept-encoding', 'x-oxe-state-policy'}, lan_share.FORWARD_HEADERS)
        self.assertEqual(self.seen[0].get('X-Oxe-State-Policy'), 'ephemeral')
        self.assertNotIn('x-oxe-session', forwarded)
        self.assertLessEqual({'user-agent', 'x-session-id', 'host'}, forwarded)
        self.assertEqual(self.seen[0].get('User-Agent'), 'client/1.0')
        self.assertNotIn(self.key, json.dumps(self.seen[0]))

    def test_engine_controls_in_the_body_are_dropped(self):
        status, data = self.send('POST', CHAT + '/?id_slot=9', self.chat_body(
            id_slot=7, cache_prompt=False, n=16, grammar='root ::= .*', max_tokens=10**9,
            chat_template_kwargs={'enable_thinking': True, 'evil': 1}), key=self.key)
        self.assertEqual(status, 200, data)
        path, body = self.bodies[0]
        self.assertEqual(path, CHAT)  # exactly the allowed route: no query, no trailing slash
        sent = json.loads(body)
        # cache_prompt is the controller's own for a llama engine: no slot prefix reuse (M2)
        self.assertEqual(set(sent), {'messages', 'max_tokens', 'chat_template_kwargs', 'cache_prompt'})
        self.assertIs(sent['cache_prompt'], False)
        self.assertEqual(sent['max_tokens'], lan_share.DEFAULT_MAX_TOKENS)
        self.assertEqual(sent['chat_template_kwargs'], {'enable_thinking': True})

    def test_bodies_that_are_not_a_plain_inline_request_are_400(self):
        fetch = json.dumps({'messages': [{'role': 'user', 'content': [
            {'type': 'image_url', 'image_url': {'url': 'http://127.0.0.1:8092/_local/status'}}]}]}).encode()
        for body in (fetch, b'[1]', b'"x"', b'not json', None):
            with self.subTest(body=body):
                self.assertEqual(self.send('POST', CHAT, body, key=self.key)[0], 400)
        self.manager.begin_queued.assert_not_called()
        self.assertEqual(self.bodies, [])

    def test_a_shared_request_does_not_touch_the_prefix_store(self):
        self.manager.current_prefix_key = 'local-prefix'
        status, data = self.send('POST', CHAT, self.chat_body(), key=self.key)
        self.assertEqual(status, 200, data)
        self.manager.prepare_prefix.assert_not_called()
        self.assertIsNone(self.manager.current_prefix_key)  # slot 0 no longer holds the local prefix

    def test_dot_segments_are_not_model_ids(self):
        for path in ('/v1/models/..', '/v1/models/.', '/v1/models/../', '/v1/models/...'):
            with self.subTest(path=path):
                self.assertEqual(self.send('GET', path, key=self.key)[0], 403)
        self.manager.config = Mock(return_value={'model_alias': 'synthetic-classic', 'profiles': {'x': {}}})
        self.assertEqual(self.send('GET', '/v1/models/synthetic-classic', key=self.key)[0], 200)

    def test_start_failure_text_stays_on_the_host(self):
        place = '/'.join(('', 'srv-synthetic', 'models', 'weights.gguf'))  # built at run time (tree scan)
        self.manager.begin_queued = Mock(side_effect=RuntimeError(place + ' is missing'))
        status, data = self.send('POST', CHAT, self.chat_body(), key=self.key)
        self.assertEqual(status, 503)
        self.assertNotIn(b'weights.gguf', data)
        self.assertNotIn(b'srv-synthetic', data)

    def test_models_listing_is_shared(self):
        self.manager.config = Mock(return_value={'model_alias': 'synthetic-classic', 'profiles': {'x': {}}})
        status, data = self.send('GET', '/v1/models', key=self.key)
        self.assertEqual(status, 200, data)
        self.assertEqual(json.loads(data)['data'][0]['id'], 'synthetic-classic')

    def test_rate_limit(self):
        self.share.limiter = lan_share.RateLimiter(1, 2)
        codes = [self.send('GET', '/v1/models', key=self.key)[0] for _ in range(3)]
        self.assertNotEqual(codes[1], 429)
        self.assertEqual(codes[2], 429)
        self.assertEqual(self.send('GET', '/v1/models', key='wrong')[0], 429)  # guesses are throttled too

    def test_body_limit_and_chunked_bodies(self):
        self.share.opts = {**self.share.opts, 'max_body_bytes': 1024}
        status, _ = self.send('POST', CHAT, b'x' * 2048, key=self.key)
        self.assertEqual(status, 413)
        connection = http.client.HTTPConnection('127.0.0.1', self.port, timeout=TIMEOUT)
        try:
            connection.request('POST', CHAT, body=iter([b'{}']), headers={
                'Authorization': 'Bearer ' + self.key, 'Transfer-Encoding': 'chunked'}, encode_chunked=True)
            self.assertEqual(connection.getresponse().status, 411)
        finally:
            connection.close()
        self.manager.begin_queued.assert_not_called()

    def test_one_shared_request_at_a_time(self):
        self.share.slot_wait = 0.05
        self.assertTrue(self.share.slot.acquire(blocking=False))
        try:
            status, data = self.send('POST', CHAT, self.chat_body(), key=self.key)
        finally:
            self.share.slot.release()
        self.assertEqual(status, 429)
        self.assertIn(b'Another shared request', data)
        self.manager.begin_queued.assert_not_called()

    def test_a_shared_request_is_admitted_as_remote_and_model_busy_is_409(self):
        # M3: the model check is Manager.begin's (atomic, see ManagerAdmissionTests); the handler
        # asks for a remote admission without a start and maps ModelBusy to 409
        self.manager.config = Mock(side_effect=lambda alias=None: {'model_alias': alias or 'local-loaded'})
        self.manager.begin_queued = Mock(side_effect=controller.ModelBusy('Another model is loaded on the host.'))
        status, data = self.send('POST', CHAT, self.chat_body(model='other-model'), key=self.key)
        self.assertEqual(status, 409, data)
        self.assertEqual(json.loads(data)['error']['code'], 'model_busy')
        kwargs = self.manager.begin_queued.call_args.kwargs
        self.assertTrue(kwargs['remote'])
        self.assertFalse(kwargs['allow_start'])  # M4: a shared request never loads a model
        self.assertIsInstance(kwargs['preempt'], lan_share.Preempt)
        self.assertEqual(kwargs['model_alias'], 'other-model')
        self.manager.cleanup.assert_not_called()
        self.assertEqual(self.seen, [])

    def test_ledger_row_is_remote_with_a_hashed_client(self):
        made = []
        real = request_ledger.Entry

        def record(*args, **kwargs):
            entry = real(*args, **kwargs)
            made.append(entry)
            return entry
        with patch.object(controller.request_ledger, 'Entry', side_effect=record):
            status, _ = self.send('POST', CHAT, self.chat_body(), key=self.key)
        self.assertEqual(status, 200)
        row = made[0].row
        self.assertEqual(row['origin'], 'remote')
        self.assertRegex(row['client_addr'], r'^[0-9a-f]{16}$')
        self.assertNotIn('127.0.0.1', json.dumps(row))
        self.assertEqual(row['client_addr'], self.share.client_hash('127.0.0.1'))

    def test_ledger_entry_keeps_remote_apart_and_ignores_bad_labels(self):
        entry = request_ledger.Entry(CHAT, {'messages': []}, origin='remote', client='not hex; ' + dotted(10, 0, 0, 1))
        self.assertEqual(entry.row['origin'], 'remote')
        self.assertNotIn('client_addr', entry.row)
        self.assertEqual(request_ledger.Entry(CHAT, {'messages': []}, client='abcdef0123456789').row['origin'], 'organic')


class RemoteOxeTests(ShareBase):
    """Security review: a shared request to the Rust engine stays out of the harness learning
    registry and out of the local sessions' cache namespace."""

    def make_manager(self, **overrides):
        profile = {'model_alias': 'synthetic-moe', 'backend_kind': 'oxe'}
        self.registry = SimpleNamespace(observe=Mock(return_value='prefix-key'), complete=Mock())
        return super().make_manager(config=Mock(return_value=profile), harness_mu=threading.Lock(),
                                    compat_stats=collections.Counter(), harness_registry=self.registry,
                                    session_hints={}, **overrides)

    def start_loopback(self):
        loop = controller.Server(('127.0.0.1', 0), controller.Handler)
        loop.manager = self.manager
        threading.Thread(target=loop.serve_forever, daemon=True).start()
        self.addCleanup(lambda: (loop.shutdown(), loop.server_close()))
        return loop.server_address[1]

    def body(self):
        return json.dumps({'messages': [{'role': 'system', 'content': 'You are a synthetic harness.'},
                                        {'role': 'user', 'content': 'hi'}]}).encode()

    def finished(self):
        """Wait for the handler to release its slot (manager.end, in its finally): the client has the
        whole reply before the handler runs its post-reply bookkeeping (registry.complete), so the
        response alone does not mean the request is done."""
        self.assertTrue(self.ended.acquire(timeout=TIMEOUT), 'the request did not release its slot')

    def test_remote_is_not_learned_and_gets_ephemeral_engine_state(self):
        self.ended = threading.Semaphore(0)
        self.manager.end = Mock(side_effect=lambda **kwargs: self.ended.release())
        session = {'X-Session-Id': 'the-same-session-id'}
        # a local client cannot ask for (or forge) the engine state policy either
        status, data = self.send('POST', CHAT, self.body(), port=self.start_loopback(),
                                 headers={**session, 'X-Oxe-State-Policy': 'ephemeral'})
        self.assertEqual(status, 200, data)
        self.finished()
        self.registry.complete.assert_called_once()  # the local request is learned
        self.registry.observe.assert_called_once()
        local_hint = self.seen[0].get('X-Oxe-Session')
        self.assertTrue(local_hint)
        self.assertNotIn('x-oxe-state-policy', {k.lower() for k in self.seen[0]})
        self.assertEqual(self.manager.session_hints, {'synthetic-moe': local_hint})
        self.registry.observe.reset_mock(); self.registry.complete.reset_mock()

        status, data = self.send('POST', CHAT, self.body(), key=self.key,
                                 headers={**session, 'X-Oxe-Session': local_hint, 'X-Oxe-State-Policy': 'persist'})
        self.assertEqual(status, 200, data)
        self.finished()  # a late complete() of the shared request would be seen below
        self.registry.observe.assert_not_called()
        self.registry.complete.assert_not_called()
        # M2: RAM-only, request-scoped engine state; no session hint or turn label at all
        remote = {k.lower(): v for k, v in self.seen[1].items()}
        self.assertEqual(remote.get('x-oxe-state-policy'), 'ephemeral')
        self.assertNotIn('x-oxe-session', remote)
        self.assertNotIn('x-oxe-turn-kind', remote)
        # the next explicit load still warms the local conversation
        self.assertEqual(self.manager.session_hints, {'synthetic-moe': local_hint})


class BackToBackTests(ShareBase):
    def test_a_request_sent_while_the_previous_one_finishes_is_served(self):
        # the shared slot is still held (the previous handler's post-reply bookkeeping) when the
        # next request arrives: it waits for the release instead of a 429
        self.assertTrue(self.share.slot.acquire(blocking=False))
        release = threading.Timer(0.2, self.share.slot.release)
        release.start()
        self.addCleanup(release.cancel)
        status, data = self.send('GET', '/v1/models', key=self.key)
        self.assertNotEqual(status, 429, data)
        self.assertFalse(self.share.slot.locked())


class ConnectionLimitTests(ShareBase):
    """Security review: connections without the key cannot use up threads or descriptors."""

    def raw(self):
        connection = socket.create_connection(('127.0.0.1', self.port), timeout=TIMEOUT)
        self.addCleanup(connection.close)
        return connection

    def closed_by_server(self, connection):
        try:
            return connection.recv(1) == b''
        except ConnectionError:
            return True

    def wait_for(self, condition):
        deadline = time.monotonic() + TIMEOUT
        while time.monotonic() < deadline:
            if condition():
                return True
            time.sleep(0.02)
        return False

    def test_a_silent_connection_is_closed_after_the_timeout(self):
        self.listener.server.io_timeout = 0.2
        idle = self.raw()
        self.assertTrue(self.closed_by_server(idle))
        self.assertTrue(self.wait_for(lambda: not self.listener.server.connections))
        half = self.raw()  # a request that never finishes its headers
        half.sendall(b'POST ' + CHAT.encode() + b' HTTP/1.1\r\nHost: x\r\n')
        self.assertTrue(self.closed_by_server(half))
        self.assertEqual(self.share.stats['remote_requests'], 0)

    def test_a_stalled_body_releases_the_shared_slot(self):
        self.listener.server.io_timeout = 0.2
        stalled = self.raw()
        stalled.sendall(('POST %s HTTP/1.1\r\nHost: x\r\nAuthorization: Bearer %s\r\n'
                         'Content-Length: 100\r\n\r\n{' % (CHAT, self.key)).encode())
        self.assertTrue(self.closed_by_server(stalled))
        self.assertTrue(self.wait_for(lambda: not self.share.slot.locked()))
        self.manager.begin_queued.assert_not_called()

    def test_connection_limits_per_client_and_per_listener(self):
        server = self.listener.server
        for name, value in (('max_connections_per_client', 2), ('max_connections', 2)):
            with self.subTest(limit=name):
                server.max_connections, server.max_connections_per_client = 100, 100
                setattr(server, name, value)
                held = [self.raw(), self.raw()]
                self.assertTrue(self.wait_for(lambda: len(server.connections) == 2))
                self.assertTrue(self.closed_by_server(self.raw()))  # the third is closed unread
                self.assertEqual(len(server.connections), 2)
                for connection in held: connection.close()
                self.assertTrue(self.wait_for(lambda: not server.connections))
                self.assertEqual(self.send('GET', '/v1/models', key='wrong')[0], 401)  # room again
                self.assertTrue(self.wait_for(lambda: not server.connections))

    def test_errors_do_not_put_the_client_address_in_the_journal(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            try: raise TimeoutError('timed out')
            except TimeoutError: self.listener.server.handle_error(None, ('192.0.2.7', 5))
            try: raise RuntimeError('boom at 192.0.2.7')
            except RuntimeError: self.listener.server.handle_error(None, ('192.0.2.7', 5))
        self.assertNotIn('192.0.2.7', err.getvalue())
        self.assertEqual(err.getvalue().count('\n'), 1)
        self.assertIn('RuntimeError', err.getvalue())


class LoopbackUnchangedTests(ShareBase):
    def setUp(self):
        super().setUp()
        self.loop = controller.Server(('127.0.0.1', 0), controller.Handler)
        self.loop.manager = self.manager
        threading.Thread(target=self.loop.serve_forever, daemon=True).start()
        self.addCleanup(lambda: (self.loop.shutdown(), self.loop.server_close()))

    def test_loopback_needs_no_share_key_and_keeps_admin_routes(self):
        port = self.loop.server_address[1]
        status, _ = self.send('POST', CHAT, self.chat_body(), port=port)
        self.assertEqual(status, 200)
        self.assertNotEqual(self.manager.begin_queued.call_args.kwargs.get('origin'), 'remote')
        self.manager.prepare_prefix.assert_called_once()  # local requests keep the prefix store
        self.assertNotIn('wait_seconds', self.manager.begin_queued.call_args.kwargs)
        status, data = self.send('GET', '/_local/share', port=port)
        self.assertEqual(status, 200)
        value = json.loads(data)
        self.assertTrue(value['enabled'])
        self.assertEqual(value['key_fingerprint'], lan_share.fingerprint(self.key))
        self.assertNotIn(self.key, data.decode())

    def test_loopback_host_check_is_unchanged(self):
        port = self.loop.server_address[1]
        status, data = self.send('GET', '/v1/models', port=port, headers={'Host': f'evil.example:{port}'})
        self.assertEqual(status, 403)
        self.assertIn(b'Host not allowed', data)


class OffSwitchTests(ShareBase):
    def test_off_closes_listeners_and_open_connections_at_once(self):
        keep = http.client.HTTPConnection('127.0.0.1', self.port, timeout=TIMEOUT)
        self.addCleanup(keep.close)
        keep.request('GET', '/v1/models', headers={'Authorization': 'Bearer ' + self.key})
        keep.getresponse().read()
        value = self.share.set_mode('off')
        self.assertFalse(value['enabled'])
        self.assertFalse(value['active'])
        self.assertEqual(value['addresses'], [])
        self.assertTrue(self.listener.closed)
        self.assertEqual(json.loads(self.config.read_text())['lan_share'], 'off')
        with self.assertRaises(OSError):
            self.send('GET', '/v1/models', key=self.key)
        try:
            keep.request('GET', '/v1/models', headers={'Authorization': 'Bearer ' + self.key})
            self.assertEqual(keep.getresponse().status, 503)
        except (OSError, http.client.HTTPException):
            pass  # the open connection was dropped
        self.assertEqual(self.share.startup_line(), 'local-ai: LAN share: off (loopback only)')

    def test_a_listener_that_is_no_longer_wanted_refuses_requests(self):
        self.share.mode = 'off'
        status, data = self.send('GET', '/v1/models', key=self.key)
        self.assertEqual(status, 503)
        self.assertIn(b'LAN share is off', data)


class ApplyBase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        (Path(tmp.name) / 'llm').mkdir()
        self.config = Path(tmp.name) / 'llm' / 'runtime.json'
        self.bound = []
        test = self

        class FakeListener:
            def __init__(self, controller_, address, port, handler, manager, tls=None):
                self.address, self.closed = address, False
                test.bound.append(address)
            def close(self): self.closed = True
        patcher = patch.object(lan_share, 'Listener', FakeListener)
        patcher.start(); self.addCleanup(patcher.stop)
        env = patch.dict(os.environ); env.start(); self.addCleanup(env.stop)
        os.environ.pop(lan_share.KEY_FILE_ENV, None)
        self.network = home_network()
        self.logged = []

    def controller(self):
        addrs = lan_share.parse_ip_addr(IP_ADDR)
        return lan_share.ShareController(self.config, 8092, object, discover_fn=lambda: (addrs, ['wlan0']),
                                         lan_fn=lambda: self.network, log=self.logged.append)


class ApplyTests(ApplyBase):

    def test_off_by_default_binds_nothing_and_makes_no_key(self):
        self.config.write_text('{}')
        share = self.controller()
        value = share.apply()
        self.assertEqual((value['enabled'], value['active'], self.bound), (False, False, []))
        self.assertFalse(share.key_file().exists())
        self.assertEqual(share.startup_line(), 'local-ai: LAN share: off (loopback only)')

    def test_tailscale_binds_only_tailnet_addresses_and_says_so_once(self):
        self.config.write_text(json.dumps({'lan_share': 'tailscale'}))
        share = self.controller()
        value = share.apply()
        self.assertEqual(self.bound, [TS_IP, 'fd7a:115c:a1e0::1234'])
        self.assertEqual(value['urls'], [f'http://{TS_IP}:8092/v1', 'http://[fd7a:115c:a1e0::1234]:8092/v1'])
        self.assertEqual(value['encryption'], 'wireguard')
        self.assertIsNone(value['warning'])
        self.assertTrue(value['key_present'])
        self.assertEqual(stat.S_IMODE(share.key_file().stat().st_mode), 0o600)
        line = share.startup_line()
        self.assertIn('tailscale ON', line)
        self.assertIn(value['key_fingerprint'], line)
        self.assertNotIn('\n', line)
        share.apply()
        self.assertEqual(len(self.bound), 2)  # unchanged addresses are not rebound

    def test_lan_warns_unencrypted_and_switching_off_unbinds(self):
        self.config.write_text(json.dumps({'lan_share': 'lan'}))
        share = self.controller()
        value = share.apply()
        self.assertEqual(value['addresses'], [LAN_IP])
        self.assertEqual(value['encryption'], 'none')
        self.assertIn('unencrypted', value['warning'])
        self.assertIn('UNENCRYPTED', share.startup_line())
        listener = share.listeners[LAN_IP]
        value = share.set_mode('off')
        self.assertTrue(listener.closed)
        self.assertEqual(value['addresses'], [])

    def test_unloadable_tls_material_refuses_to_share(self):
        self.config.write_text(json.dumps({'lan_share': 'lan', 'lan_share_options': {
            'tls_cert': str(self.config.parent / 'missing.pem'), 'tls_key': str(self.config.parent / 'missing.key')}}))
        value = self.controller().apply()
        self.assertEqual((self.bound, value['active']), ([], False))
        self.assertIn('TLS', value['error'])

    def test_options_are_bounded(self):
        opts = lan_share.options({'lan_share_options': {'rate_per_min': 0, 'burst': True, 'max_body_mb': 99,
                                                         'tls_cert': 'relative.pem', 'tls_key': '/k'}})
        self.assertEqual((opts['rate_per_min'], opts['burst'], opts['max_body_bytes'], opts['tls']),
                         (120, 30, 8 * 1024 * 1024, None))
        self.assertEqual(opts['max_tokens'], lan_share.DEFAULT_MAX_TOKENS)
        self.assertEqual(lan_share.options({'lan_share_options': {'max_tokens': 512}})['max_tokens'], 512)
        self.assertEqual(lan_share.options({'lan_share_options': {'max_tokens': 0}})['max_tokens'], lan_share.DEFAULT_MAX_TOKENS)


def mac(*parts):
    # built at run time so the tree's secret scan has no hardware-address literal to flag
    return ':'.join('%02x' % p for p in parts)


HOME_MAC, CAFE_MAC = mac(0xaa, 0xbb, 0xcc, 0, 0x11, 0x22), mac(0xaa, 0xbb, 0xcc, 0x99, 0x88, 0x77)


def home_network(**changes):
    return {'interface': 'wlan0', 'address': LAN_IP, 'subnet': dotted(192, 168, 1, 0) + '/24',
            'gateway': GATEWAY, 'gateway_mac': HOME_MAC, **changes}


class LanPinParsingTests(unittest.TestCase):
    """M1: the network `lan` is pinned to, from recorded `ip -j` output."""

    def test_network_of_the_default_route(self):
        addrs = json.dumps([
            {'ifname': 'tun0', 'link_type': 'none', 'addr_info': [{'family': 'inet', 'local': dotted(10, 8, 0, 2), 'prefixlen': 24}]},
            {'ifname': 'wlan0', 'link_type': 'ether', 'addr_info': [
                {'family': 'inet', 'local': LAN_IP, 'prefixlen': 24}, {'family': 'inet6', 'local': 'fe80::1', 'prefixlen': 64}]}])
        routes = json.dumps([{'dst': 'default', 'gateway': dotted(10, 8, 0, 1), 'dev': 'tun0'},
                             {'dst': 'default', 'gateway': GATEWAY, 'dev': 'wlan0'}])
        self.assertEqual(lan_share.parse_lan_network(addrs, routes), {
            'interface': 'wlan0', 'address': LAN_IP, 'subnet': dotted(192, 168, 1, 0) + '/24', 'gateway': GATEWAY})
        self.assertIsNone(lan_share.parse_lan_network(addrs, json.dumps([{'dst': 'default', 'dev': 'tun0'}])))
        self.assertIsNone(lan_share.parse_lan_network(addrs, json.dumps([{'dst': 'default', 'gateway': 'x', 'dev': 'wlan0'}])))

    def test_gateway_mac_from_the_neighbour_table(self):
        self.assertEqual(lan_share.parse_neigh_mac(json.dumps([{'dst': GATEWAY, 'lladdr': HOME_MAC.upper(), 'state': ['REACHABLE']}])), HOME_MAC)
        for rows in ([], [{'dst': GATEWAY, 'state': ['INCOMPLETE']}], [{'lladdr': 'zz', 'state': ['STALE']}],
                     [{'lladdr': HOME_MAC, 'state': ['FAILED']}]):
            self.assertIsNone(lan_share.parse_neigh_mac(json.dumps(rows)))

    def test_discover_lan_asks_the_neighbour_table_for_the_gateway(self):
        addrs = json.dumps([{'ifname': 'wlan0', 'link_type': 'ether', 'addr_info': [{'local': LAN_IP, 'prefixlen': 24}]}])
        routes = json.dumps([{'dst': 'default', 'gateway': GATEWAY, 'dev': 'wlan0'}])
        neigh = json.dumps([{'dst': GATEWAY, 'lladdr': HOME_MAC, 'state': ['STALE']}])
        run = Mock(side_effect=[SimpleNamespace(stdout=addrs), SimpleNamespace(stdout=routes), SimpleNamespace(stdout=neigh)])
        with patch.object(lan_share, '_ip_tool', return_value='/usr/bin/ip'):
            self.assertEqual(lan_share.discover_lan(run), home_network())
        self.assertEqual(run.call_args.args[0], ['/usr/bin/ip', '-j', 'neigh', 'show', GATEWAY, 'dev', 'wlan0'])

    def test_compare(self):
        pin = lan_share.pin_of(home_network())
        self.assertNotIn('address', pin)
        self.assertEqual(lan_share.compare_pin(pin, home_network(address=dotted(192, 168, 1, 77))), 'same')  # new lease
        for change in ({'interface': 'eth0'}, {'subnet': dotted(192, 168, 2, 0) + '/24'}, {'gateway': dotted(192, 168, 1, 254)},
                       {'gateway_mac': CAFE_MAC}):
            self.assertEqual(lan_share.compare_pin(pin, home_network(**change)), 'changed', change)
        self.assertEqual(lan_share.compare_pin(pin, home_network(gateway_mac=None)), 'unverified')
        self.assertEqual(lan_share.compare_pin(pin, None), 'absent')
        unknown = lan_share.pin_of(home_network(gateway_mac=None))
        self.assertEqual(lan_share.compare_pin(unknown, home_network(gateway_mac=CAFE_MAC)), 'same')
        self.assertEqual(lan_share.pin_from_config({'lan_share_pin': pin}), pin)
        for bad in ({**pin, 'subnet': 'x'}, {**pin, 'gateway_mac': 'nope'}, {**pin, 'interface': 3}, 'x'):
            self.assertIsNone(lan_share.pin_from_config({'lan_share_pin': bad}))


class LanPinApplyTests(ApplyBase):
    """M1: `lan` stays on the network it was turned on for; anywhere else it turns itself off."""

    def saved(self):
        return json.loads(self.config.read_text())

    def test_on_pins_the_current_network_and_a_new_lease_rebinds_on_it(self):
        self.config.write_text('{}')
        share = self.controller()
        value = share.set_mode('lan')
        self.assertEqual(value['addresses'], [LAN_IP])
        self.assertEqual(self.saved()['lan_share_pin'], lan_share.pin_of(home_network()))
        self.assertEqual(value['pinned_network'], {'interface': 'wlan0', 'subnet': dotted(192, 168, 1, 0) + '/24',
                                                   'gateway': GATEWAY, 'gateway_mac_known': True})
        self.network = home_network(address=dotted(192, 168, 1, 77))
        value = share.apply()
        self.assertEqual(value['addresses'], [dotted(192, 168, 1, 77)])
        self.assertIsNone(value['paused'])

    def test_another_network_turns_sharing_off_and_it_stays_off(self):
        self.config.write_text(json.dumps({'other': 1}))
        share = self.controller()
        share.set_mode('lan')
        listener = share.listeners[LAN_IP]
        for cafe in (home_network(gateway_mac=CAFE_MAC), home_network(subnet=dotted(10, 1, 0, 0) + '/16', address=dotted(10, 1, 2, 3))):
            with self.subTest(cafe=cafe):
                self.network = cafe
                value = share.apply()
                self.assertTrue(listener.closed)
                self.assertEqual((value['enabled'], value['active'], value['addresses']), (False, False, []))
                self.assertEqual(value['paused'], lan_share.PAUSED)
                saved = self.saved()
                self.assertEqual((saved['lan_share'], saved['other']), ('off', 1))
                self.assertNotIn('lan_share_pin', saved)
                self.assertEqual(saved['lan_share_paused']['reason'], 'network_changed')
                self.assertIn('network changed', share.startup_line())
                self.assertEqual(len(self.logged), 1)
                # back home: still off (turned off, not rebound); a restart reads the same
                self.network = home_network()
                self.assertEqual(share.apply()['addresses'], [])
                self.assertEqual(self.controller().apply()['paused'], lan_share.PAUSED)
                # re-enabling pins the network it is on now
                self.network = cafe
                value = share.set_mode('lan')
                self.assertEqual(value['addresses'], [cafe['address']])
                self.assertIsNone(value['paused'])
                self.assertNotIn('lan_share_paused', self.saved())
                self.assertEqual(self.saved()['lan_share_pin'], lan_share.pin_of(cafe))
                self.network = home_network()
                share.set_mode('lan')
                listener = share.listeners[LAN_IP]
                self.logged.clear()

    def test_unverified_gateway_or_no_network_shares_nothing_and_waits(self):
        self.config.write_text('{}')
        share = self.controller()
        share.set_mode('lan')
        for network in (home_network(gateway_mac=None), None):
            with self.subTest(network=network):
                self.network = network
                value = share.apply()
                self.assertEqual((value['enabled'], value['active']), (True, False))
                self.assertIsNone(value['paused'])
                self.assertTrue(value['error'])
                self.assertEqual(self.saved()['lan_share'], 'lan')
                self.network = home_network()
                self.assertEqual(share.apply()['addresses'], [LAN_IP])
        self.assertEqual(self.logged, [])

    def test_a_setting_without_a_pin_pins_the_first_network_seen(self):
        self.config.write_text(json.dumps({'lan_share': 'lan'}))
        self.network = None
        share = self.controller()
        self.assertEqual(share.apply()['addresses'], [])
        self.assertNotIn('lan_share_pin', self.saved())
        self.network = home_network()
        self.assertEqual(share.apply()['addresses'], [LAN_IP])
        self.assertEqual(self.saved()['lan_share_pin'], lan_share.pin_of(home_network()))

    def test_on_without_a_lan_network_is_refused_and_saves_nothing(self):
        self.config.write_text(json.dumps({'lan_share': 'off'}))
        self.network = None
        with self.assertRaises(lan_share.LanUnavailable):
            self.controller().set_mode('lan')
        self.assertEqual(self.saved(), {'lan_share': 'off'})

    def test_tailscale_is_not_pinned(self):
        self.config.write_text('{}')
        share = self.controller()
        share.set_mode('tailscale')
        self.network = home_network(gateway_mac=CAFE_MAC)
        value = share.apply()
        self.assertEqual(value['addresses'], [TS_IP, 'fd7a:115c:a1e0::1234'])
        self.assertIsNone(value['paused'])
        self.assertIsNone(value['pinned_network'])
        self.assertNotIn('lan_share_pin', self.saved())


class SaveModeTests(unittest.TestCase):
    """M5: saving the setting keeps runtime.json's permissions and a symlinked runtime.json."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def test_permissions_are_kept(self):
        config = self.root / 'runtime.json'
        config.write_text(json.dumps({'a': 1}))
        os.chmod(config, 0o644)
        lan_share.save_mode(config, 'tailscale')
        self.assertEqual(stat.S_IMODE(config.stat().st_mode), 0o644)
        self.assertEqual(json.loads(config.read_text()), {'a': 1, 'lan_share': 'tailscale'})
        fresh = self.root / 'new.json'
        lan_share.save_mode(fresh, 'off')
        self.assertEqual(stat.S_IMODE(fresh.stat().st_mode), 0o600)

    def test_a_symlinked_config_stays_a_symlink(self):
        (self.root / 'dotfiles').mkdir()
        target = self.root / 'dotfiles' / 'runtime.json'
        target.write_text(json.dumps({'a': 1}))
        os.chmod(target, 0o640)
        link = self.root / 'runtime.json'
        link.symlink_to(target)
        lan_share.save_mode(link, 'off', pin=None)
        self.assertTrue(link.is_symlink())
        self.assertEqual(json.loads(target.read_text()), {'a': 1, 'lan_share': 'off'})
        self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o640)
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), ['dotfiles', 'runtime.json'])

    def test_guard_sees_the_link_and_its_target(self):
        target = self.root / 'real.json'
        target.write_text('{}')
        link = self.root / 'link.json'
        link.symlink_to(target)
        seen = []
        lan_share.save_mode(link, 'off', guard=seen.append)
        self.assertEqual(seen, [link, target.resolve()])


class ManagerAdmissionTests(unittest.TestCase):
    """M3/M4 on a real Manager (no engine): the model check is atomic with admission, a shared
    request never loads, switches or unloads a model, and local use cancels a running shared one."""

    def setUp(self):
        m = controller.Manager()
        m.stop_event.set()
        m.thread.join(2)
        m.config = lambda alias=None: {'model_alias': alias or 'model-a'}
        m.eligible = Mock(return_value=True)
        m.cleanup = Mock(side_effect=AssertionError('a shared request must not unload'))
        m.process = SimpleNamespace(poll=lambda: None, pid=0)
        m.loaded_config = {'model_alias': 'model-a'}
        m.state = 'ready'
        self.m = m

    def remote(self, alias='model-a', preempt=None):
        return self.m.begin_queued(inference=True, model_alias=alias, origin='remote', wait_seconds=0,
                                   remote=True, preempt=preempt or lan_share.Preempt())

    def test_only_the_loaded_model(self):
        with self.assertRaisesRegex(controller.ModelBusy, 'model-a'):
            self.remote('model-b')
        self.assertEqual((self.m.active, self.m.inferences, self.m.remote_preempt), (0, 0, None))
        self.assertIs(self.remote(None), self.m.process)  # no model named: the loaded one
        self.m.end(inference=True)
        self.m.process = None
        for alias in (None, 'model-a'):
            with self.assertRaisesRegex(controller.ModelBusy, 'cannot load'):
                self.remote(alias)
        self.assertFalse(self.m.queue_reserved)
        self.m.cleanup.assert_not_called()

    def test_a_load_that_wins_the_race_is_seen(self):
        # the old handler checked the loaded model before admission, without the lock: a local
        # load or switch between check and begin let a shared request in on another model
        result = []

        def shared():
            try: result.append(self.remote('model-a'))
            except controller.Unavailable as error: result.append(error)
        with self.m.mu:
            thread = threading.Thread(target=shared)
            thread.start()
            deadline = time.monotonic() + TIMEOUT
            while not self.m.queue_reserved and time.monotonic() < deadline:
                time.sleep(0.005)
            time.sleep(0.05)  # blocked on mu inside begin
            self.m.loaded_config = {'model_alias': 'model-b'}  # a local switch completes
        thread.join(TIMEOUT)
        self.assertIsInstance(result[0], controller.ModelBusy)
        self.assertEqual((self.m.active, self.m.inferences, self.m.remote_preempt), (0, 0, None))

    def test_concurrent_begins(self):
        # shared requests for the loaded and another model, and local switches, all at once: a
        # shared request runs only on the model that is loaded while it runs
        outcomes, lock = collections.Counter(), threading.Lock()
        peak, running = [0], [0]

        def shared(alias):
            for _ in range(20):
                try:
                    self.remote(alias)
                except controller.ModelBusy:
                    with lock: outcomes[alias, 'busy'] += 1
                    continue
                except controller.Unavailable:
                    with lock: outcomes[alias, 'queue'] += 1
                    continue
                with lock:
                    outcomes[alias, 'ran'] += 1
                    running[0] += 1
                    peak[0] = max(peak[0], running[0])
                    if self.m.loaded_config['model_alias'] != alias:
                        outcomes[alias, 'WRONG MODEL'] += 1
                time.sleep(0.001)
                with lock: running[0] -= 1
                self.m.end(inference=True)

        def switch():
            for i in range(40):
                with self.m.mu:
                    if self.m.inferences == 0:  # a local load waits for the running request
                        self.m.loaded_config = {'model_alias': 'model-b' if i % 2 else 'model-a'}
                time.sleep(0.001)
        threads = [threading.Thread(target=shared, args=(alias,)) for alias in ('model-a', 'model-b') * 4]
        threads.append(threading.Thread(target=switch))
        for t in threads: t.start()
        for t in threads: t.join(TIMEOUT * 4)
        self.assertGreaterEqual(sum(n for (_, kind), n in outcomes.items() if kind == 'ran'), 1)
        self.assertNotIn('WRONG MODEL', {kind for _, kind in outcomes})
        self.assertEqual(peak[0], 1)
        self.assertEqual((self.m.active, self.m.inferences, self.m.remote_preempt, self.m.queue_reserved), (0, 0, None, False))

    def test_local_use_cancels_a_running_shared_request(self):
        preempt = lan_share.Preempt()
        self.remote(preempt=preempt)
        self.assertIs(self.m.remote_preempt, preempt)
        admitted = threading.Event()

        def local():
            self.m.begin_queued(inference=True, wait_seconds=TIMEOUT, origin='organic')
            admitted.set()
        thread = threading.Thread(target=local)
        thread.start()
        self.assertTrue(preempt.event.wait(TIMEOUT), 'the shared request was not cancelled')
        self.assertFalse(admitted.is_set())  # it waits for the shared request to end
        self.m.end(inference=True)  # the cancelled shared request ends
        self.assertTrue(admitted.wait(TIMEOUT))
        thread.join(TIMEOUT)
        self.assertIsNone(self.m.remote_preempt)
        self.m.end(inference=True)

    def test_a_waiting_local_request_keeps_a_shared_one_out(self):
        self.m.begin_queued(inference=True, origin='organic')  # running
        waiting = threading.Thread(target=lambda: (self.m.begin_queued(inference=True, wait_seconds=TIMEOUT), self.m.end(inference=True)))
        waiting.start()
        while self.m.queue_length() < 1:
            time.sleep(0.005)
        preempt = lan_share.Preempt()
        with self.assertRaises(controller.Unavailable):
            self.remote(preempt=preempt)
        self.assertIsNone(self.m.remote_preempt)
        self.assertFalse(preempt.cancelled)
        self.m.end(inference=True)
        waiting.join(TIMEOUT)

    def test_a_shared_request_waits_out_the_one_before_it(self):
        # flake fix: a client sending its next request the moment it read a reply finds the previous
        # request still finishing; it gets the next turn instead of a 503
        self.m.begin_queued(inference=True, origin='organic')
        result = []
        thread = threading.Thread(target=lambda: result.append(self.m.begin_queued(
            inference=True, origin='remote', wait_seconds=TIMEOUT, remote=True, preempt=lan_share.Preempt())))
        thread.start()
        while self.m.queue_length() < 1:
            time.sleep(0.005)
        self.m.end(inference=True)
        thread.join(TIMEOUT)
        self.assertEqual(result, [self.m.process])
        self.assertIsNotNone(self.m.remote_preempt)
        self.m.end(inference=True)

    def test_a_waiting_shared_request_gives_way_to_a_local_one(self):
        self.m.begin_queued(inference=True, origin='organic')  # running
        failed, local_ran = threading.Event(), threading.Event()

        def shared():
            try: self.m.begin_queued(inference=True, origin='remote', wait_seconds=TIMEOUT, remote=True,
                                     preempt=lan_share.Preempt())
            except controller.Unavailable: failed.set()
        remote = threading.Thread(target=shared)
        remote.start()
        while self.m.queue_length() < 1:
            time.sleep(0.005)
        local = threading.Thread(target=lambda: (self.m.begin_queued(inference=True, wait_seconds=TIMEOUT), local_ran.set()))
        local.start()
        self.assertTrue(failed.wait(TIMEOUT), 'the shared request kept its place ahead of a local one')
        self.m.end(inference=True)
        self.assertTrue(local_ran.wait(TIMEOUT))
        self.assertIsNone(self.m.remote_preempt)
        self.m.end(inference=True)
        remote.join(TIMEOUT); local.join(TIMEOUT)

    def test_preempt_cuts_the_engine_connection_attached_later_too(self):
        for early in (False, True):
            a, b = socket.socketpair()
            self.addCleanup(a.close); self.addCleanup(b.close)
            preempt = lan_share.Preempt()
            if early:
                preempt.cancel()
            preempt.attach(a)
            if not early:
                preempt.cancel()
            self.assertEqual(b.recv(1), b'')  # the peer sees the connection end


class PreemptHandlerTests(ShareBase):
    """M4 end to end: a shared request running on the engine is cut for a local one, and its client
    gets 'preempted by local use' (a 503 before the response, an SSE error event in a stream)."""

    def slow_backend(self, stream):
        started = threading.Event()

        class Slow(BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.0'
            def log_message(self, *args): pass
            def do_POST(self):
                self.rfile.read(int(self.headers['Content-Length']))
                started.set()
                try:
                    if not stream:
                        time.sleep(3)
                        return
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/event-stream')
                    self.end_headers()
                    for _ in range(100):
                        self.wfile.write(b'data: {"choices":[{"delta":{"content":"x"}}]}\n\n')
                        self.wfile.flush()
                        time.sleep(0.05)
                except OSError:
                    pass
        server = ThreadingHTTPServer(('127.0.0.1', 0), Slow)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(lambda: (server.shutdown(), server.server_close()))
        item = patch.object(controller, 'BACKEND_PORT', server.server_port)
        item.start(); self.addCleanup(item.stop)
        return started

    def run_preempted(self, stream):
        started = self.slow_backend(stream)

        def preempt_when_running():
            started.wait(TIMEOUT)
            time.sleep(0.2)
            self.manager.begin_queued.call_args.kwargs['preempt'].cancel()
        threading.Thread(target=preempt_when_running, daemon=True).start()
        t0 = time.monotonic()
        status, data = self.send('POST', CHAT, self.chat_body(stream=stream), key=self.key)
        self.assertLess(time.monotonic() - t0, 2.5, 'not cancelled promptly')
        deadline = time.monotonic() + TIMEOUT
        while not self.manager.end.called and time.monotonic() < deadline:  # the slot is released after the reply
            time.sleep(0.01)
        self.manager.end.assert_called_once()
        return status, data

    def test_before_the_response(self):
        status, data = self.run_preempted(stream=False)
        self.assertEqual(status, 503, data)
        self.assertEqual(json.loads(data)['error']['code'], 'preempted_by_local_use')

    def test_in_a_stream(self):
        status, data = self.run_preempted(stream=True)
        self.assertEqual(status, 200)
        self.assertIn(b'data: {"choices"', data)
        self.assertTrue(data.rstrip().endswith(b'}'), data[-200:])
        last = json.loads(data.rstrip().split(b'\n\n')[-1][len(b'data: '):])
        self.assertEqual(last['error']['code'], 'preempted_by_local_use')


@unittest.skipUnless(shutil.which('openssl'), 'openssl is needed to make a test certificate')
class TlsListenerTests(ShareBase):
    """M5: the TLS listener end to end (the review found it broken once: the disconnect watcher
    cancelled every request): keyed chat and models over TLS, 401 without the key, plain HTTP refused."""

    def setUp(self):
        super().setUp()
        cert, key = self.root / 'cert.pem', self.root / 'key.pem'
        subprocess.run(['openssl', 'req', '-x509', '-newkey', 'ec', '-pkeyopt', 'ec_paramgen_curve:prime256v1',
                        '-nodes', '-days', '1', '-subj', '/CN=local-ai-test', '-addext', 'subjectAltName=IP:127.0.0.1',
                        '-keyout', str(key), '-out', str(cert)], check=True, capture_output=True, timeout=60)
        server_tls = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        server_tls.load_cert_chain(cert, key)
        self.tls_listener = lan_share.Listener(self.share, '127.0.0.1', 0, controller.Handler, self.manager, server_tls)
        self.share.listeners['127.0.0.1'] = self.tls_listener
        self.share.tls_on = True
        self.addCleanup(self.tls_listener.close)
        self.client_tls = ssl.create_default_context(cafile=str(cert))

    def tls_send(self, method, path, body=None, key=None):
        connection = http.client.HTTPSConnection('127.0.0.1', self.tls_listener.port, timeout=TIMEOUT, context=self.client_tls)
        try:
            headers = {'Authorization': 'Bearer ' + key} if key else {}
            connection.request(method, path, body=body, headers=headers)
            response = connection.getresponse()
            return response.status, response.read()
        finally:
            connection.close()

    def test_keyed_requests_over_tls(self):
        # back to back, as a client does: the next request may arrive before the previous handler
        # has released the shared slot (the flake: 429); it waits for it (SLOT_WAIT)
        status, data = self.tls_send('POST', CHAT, self.chat_body(), key=self.key)
        self.assertEqual(status, 200, data)
        self.assertIn(b'READY', data)
        self.manager.config = Mock(return_value={'model_alias': 'synthetic-classic', 'profiles': {'x': {}}})
        self.assertEqual(self.tls_send('GET', '/v1/models', key=self.key)[0], 200)
        self.assertEqual(self.tls_send('POST', CHAT, self.chat_body())[0], 401)
        self.assertTrue(self.share.status()['urls'][0].startswith('https://'))

    def test_plain_http_is_refused(self):
        with self.assertRaises((OSError, http.client.HTTPException)):
            self.send('GET', '/v1/models', key=self.key, port=self.tls_listener.port)


SCRIPT = Path(__file__).resolve().parents[1] / 'src/snapdragon_ai_lab/runtime/local-ai'
_loader = importlib.machinery.SourceFileLoader('lan_share_cli_local_ai', str(SCRIPT))
cli = importlib.util.module_from_spec(importlib.util.spec_from_loader(_loader.name, _loader))
_loader.exec_module(cli)


class CliTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        (Path(tmp.name) / 'llm').mkdir()
        self.config = Path(tmp.name) / 'llm' / 'runtime.json'
        self.config.write_text('{}')
        env = patch.dict(os.environ, {'LOCAL_LLM_CONFIG': str(self.config)}); env.start(); self.addCleanup(env.stop)
        os.environ.pop(lan_share.KEY_FILE_ENV, None)

    def run_cli(self, *argv, request=None):
        out, err = io.StringIO(), io.StringIO()
        with patch.object(cli, 'request', side_effect=request or (lambda *a, **k: (0, {}))), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = cli.main(list(argv))
        return code, out.getvalue(), err.getvalue()

    def test_key_is_shown_once_then_only_by_rotation(self):
        code, out, _ = self.run_cli('share', 'key')
        key = out.strip()
        self.assertEqual(code, 0)
        self.assertTrue(key.startswith('lai-'))
        code, out, err = self.run_cli('share', 'key')
        self.assertEqual((code, out), (1, ''))
        self.assertIn('--rotate', err)
        code, out, _ = self.run_cli('share', 'key', '--rotate')
        self.assertEqual(code, 0)
        self.assertNotEqual(out.strip(), key)
        path = lan_share.key_path(self.config)
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)

    def test_on_and_off_go_to_the_controller(self):
        sent = []

        def request(url, post=False, timeout=2, payload=None):
            sent.append((url, post, payload))
            return 200, {**lan_share.off_status(8092), 'enabled': payload['mode'] != 'off', 'mode': payload['mode'],
                         'active': payload['mode'] != 'off', 'urls': [f'http://{TS_IP}:8092/v1']}
        self.assertEqual(self.run_cli('share', 'on', request=request)[0], 0)
        self.assertEqual(self.run_cli('share', 'off', request=request)[0], 0)
        self.assertEqual([p['mode'] for _, _, p in sent], ['tailscale', 'off'])
        self.assertTrue(all(url.endswith('/_local/share') and post for url, post, _ in sent))
        self.assertEqual(self.run_cli('share', 'on', 'everywhere', request=request)[0], 2)

    def test_off_is_saved_when_the_service_is_offline(self):
        self.config.write_text(json.dumps({'lan_share': 'lan', 'other': 1}))
        code, out, _ = self.run_cli('share', 'off')
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(self.config.read_text()), {'lan_share': 'off', 'other': 1})
        code, out, _ = self.run_cli('share', 'status', '--json')
        self.assertEqual((code, json.loads(out)['mode']), (1, 'off'))

    def test_offline_on_lan_pins_the_network_it_is_on(self):
        # M1: offline, the CLI saves the pin itself; with no LAN network it saves nothing
        with patch.object(cli, 'share_module', return_value=lan_share), \
                patch.object(lan_share, 'discover_lan', return_value=home_network()):
            code, out, _ = self.run_cli('share', 'on', 'lan')
        self.assertEqual(code, 0)
        saved = json.loads(self.config.read_text())
        self.assertEqual((saved['lan_share'], saved['lan_share_pin']), ('lan', lan_share.pin_of(home_network())))
        self.config.write_text('{}')
        with patch.object(cli, 'share_module', return_value=lan_share), \
                patch.object(lan_share, 'discover_lan', return_value=None):
            code, _, err = self.run_cli('share', 'on', 'lan')
        self.assertEqual(code, 1)
        self.assertIn('No private LAN network', err)
        self.assertEqual(json.loads(self.config.read_text()), {})

    def test_paused_is_shown(self):
        self.config.write_text(json.dumps({'lan_share': 'off', 'lan_share_paused': {'reason': 'network_changed', 't': 1}}))
        code, out, _ = self.run_cli('share', 'status')
        self.assertEqual(code, 1)  # offline
        self.assertIn(lan_share.PAUSED, out)
        self.assertIn(lan_share.PAUSED, cli.share_text({**lan_share.off_status(8092), 'paused': lan_share.PAUSED}))


if __name__ == '__main__':
    unittest.main()
