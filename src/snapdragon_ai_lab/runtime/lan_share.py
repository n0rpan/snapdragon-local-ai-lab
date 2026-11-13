"""Opt-in LAN share (O-102): let your other devices use the local models over the network.

Off by default and never turned on by itself. runtime.json:

    "lan_share": "off" | "tailscale" | "lan"
    "lan_share_options": {"rate_per_min": 120, "burst": 30, "max_body_mb": 8, "max_tokens": 16384,
                          "tls_cert": "/abs/cert.pem", "tls_key": "/abs/key.pem"}   (all optional)

When on, the controller keeps its loopback listener unchanged and adds one listener per chosen
address on the same port: the Tailscale addresses (100.64/10 or fd7a:115c:a1e0::/48 on a
`tailscale*` interface) for `tailscale`, the private IPv4 address of the default-route interface
for `lan`. Never 0.0.0.0 or ::.

`lan` is pinned (security review M1): turning it on records the network it is on then
(`lan_share_pin`: interface, subnet, gateway and, when the neighbour table has it, the gateway's
MAC address). On any other network (another subnet, gateway or gateway MAC: a cafe, a hotel)
sharing turns itself off instead of rebinding (`lan_share` off, `lan_share_paused` set, one log
line; status `paused`); `local-ai share on lan` there pins the new network. `tailscale` is not
pinned.

Every request on a shared listener passes this gate before the controller sees it:
1. the listener is still wanted (an off switch closes listeners and their open connections);
2. a per-client token bucket (429);
3. `Authorization: Bearer <key>` (or x-api-key / api-key) against the share key, constant-time
   (401); the key is 256 bits from `secrets`, kept in a 0600 file next to the runtime config, and
   is never forwarded to an engine;
4. only the OpenAI inference routes: POST /v1/chat/completions, POST /v1/completions,
   GET/HEAD /v1/models[/id] (403 for everything else, /_local/* admin routes included); the
   controller then routes exactly the path this gate matched (query dropped), and sees only the
   request headers in FORWARD_HEADERS and the body fields in PAYLOAD_FIELDS (remote_payload);
5. a body limit (413) and no chunked bodies (411);
6. one shared request at a time (429), admitted only when the machine is idle (no queue jumping:
   a shared request never waits ahead of a local one), only to the model already loaded (409, in
   Manager.begin: a shared request never loads, switches or unloads a model), and a local request
   that arrives while it runs cancels it at once ('preempted by local use', see Preempt).
Before the gate: at most MAX_CONNECTIONS open connections per listener (MAX_CONNECTIONS_PER_CLIENT
per address) and an IO_TIMEOUT on every socket read and write, so idle or stalled connections
cannot use up the controller's threads and file descriptors.

Stdlib only: the `local-ai` CLI loads this file directly (it is installed apart from the package).
"""
import contextlib
import fcntl
import hashlib
import hmac
import ipaddress
import json
import os
from collections import OrderedDict
from http.server import ThreadingHTTPServer
from pathlib import Path
import re
import secrets
import shutil
import socket
import socketserver
import ssl
import stat
import subprocess
import sys
import threading
import time
from urllib.parse import urlsplit

MODES = ('off', 'tailscale', 'lan')
REMOTE_ORIGIN = 'remote'
KEY_FILE_ENV = 'LOCAL_AI_SHARE_KEY_FILE'
KEY_PREFIX = 'lai-'
KEY_BYTES = 32  # 256 bits
_KEY_RE = re.compile(r'lai-[A-Za-z0-9_-]{43,128}')
DEFAULT_RATE_PER_MIN = 120
DEFAULT_BURST = 30
DEFAULT_MAX_BODY_MB = 8
MAX_BODY_MB_CAP = 16  # the controller's own limit for any body
WATCH_SECONDS = 10.0
# (network as an integer, prefix): 100.64/10 (Tailscale CGNAT) and 10/8, 172.16/12, 192.168/16 (RFC 1918)
TS_NETS = (ipaddress.ip_network((0x64400000, 10)), ipaddress.ip_network('fd7a:115c:a1e0::/48'))
LAN_NETS = (ipaddress.ip_network((0x0A000000, 8)), ipaddress.ip_network((0xAC100000, 12)),
            ipaddress.ip_network((0xC0A80000, 16)))
POST_ROUTES = frozenset({'/v1/chat/completions', '/v1/completions'})
GET_ROUTES = frozenset({'/v1/models'})
MODEL_ID = re.compile(r'[A-Za-z0-9._:-]{1,128}')
ROUTES = ['POST /v1/chat/completions', 'POST /v1/completions', 'GET /v1/models', 'GET /v1/models/{id}']
DEFAULT_MAX_TOKENS = 16384  # the largest output limit a shared request may ask for
IO_TIMEOUT = 60.0  # seconds a shared connection may sit idle, or block in one read or write
MAX_CONNECTIONS = 32  # open connections per shared listener
MAX_CONNECTIONS_PER_CLIENT = 8
# How long a shared request waits for the previous one to finish (the shared slot, and its turn in
# the controller's FIFO while no local request waits): a client that sends its next request the
# moment it has read a reply finds the previous handler still in its post-reply bookkeeping
# (ledger, slot release), and must not get a spurious 429/503 for that.
SLOT_WAIT = 2.0
# Request headers of a shared request the controller may see and forward; every other header is
# removed in the gate: the share key (Authorization, X-Api-Key, Api-Key), cookies, the lab origin
# marker (X-Local-AI-Origin), the engine's own X-Oxe-* headers, X-Forwarded-* and anything unknown.
# Origin and Sec-Fetch-* stay so the controller's cross-origin refusal still applies.
FORWARD_HEADERS = frozenset({
    'host', 'content-type', 'content-length', 'content-encoding', 'accept', 'user-agent', 'connection',
    'origin', 'sec-fetch-site', 'sec-fetch-mode', 'sec-fetch-dest', 'sec-fetch-user',
    'x-session-id', 'x-claude-code-session-id', 'session-id'})
# Body fields of a shared request that reach an engine (the OpenAI API's own); anything else is
# dropped: slot and cache controls (id_slot, cache_prompt, n_keep), raw grammars, samplers
# lists, lora, native length fields and `n`.
PAYLOAD_FIELDS = frozenset({
    'model', 'messages', 'prompt', 'suffix', 'echo', 'stream', 'stream_options', 'temperature', 'top_p',
    'top_k', 'min_p', 'stop', 'seed', 'presence_penalty', 'frequency_penalty', 'repeat_penalty',
    'logit_bias', 'logprobs', 'top_logprobs', 'tools', 'tool_choice', 'parallel_tool_calls',
    'response_format', 'reasoning_effort', 'user'})
TEMPLATE_KWARGS = ('enable_thinking', 'preserve_thinking')  # booleans only
LENGTH_FIELDS = ('max_tokens', 'max_completion_tokens')
PAUSED = 'paused: network changed, re-enable to share on this network'
PIN_FIELDS = ('interface', 'subnet', 'gateway', 'gateway_mac')
_MAC_RE = re.compile(r'[0-9a-f]{2}(:[0-9a-f]{2}){5}')
LAN_WARNING = ('LAN mode is unencrypted HTTP: anyone on this network can read prompts and replies '
               'in transit. Prefer tailscale, or set lan_share_options.tls_cert/tls_key.')


# ------------------------------------------------------------------ settings
def mode_of(value):
    """The configured mode; anything unknown is 'off' (fail closed)."""
    raw = value.get('lan_share') if isinstance(value, dict) else None
    return raw if isinstance(raw, str) and raw in MODES else 'off'


def _int(raw, default, low, high):
    return raw if type(raw) is int and low <= raw <= high else default


def options(value):
    raw = value.get('lan_share_options') if isinstance(value, dict) else None
    raw = raw if isinstance(raw, dict) else {}
    cert, key = raw.get('tls_cert'), raw.get('tls_key')
    tls = None
    if isinstance(cert, str) and isinstance(key, str) and os.path.isabs(cert) and os.path.isabs(key):
        tls = (cert, key)
    return {'rate_per_min': _int(raw.get('rate_per_min'), DEFAULT_RATE_PER_MIN, 1, 6000),
            'burst': _int(raw.get('burst'), DEFAULT_BURST, 1, 1000),
            'max_body_bytes': _int(raw.get('max_body_mb'), DEFAULT_MAX_BODY_MB, 1, MAX_BODY_MB_CAP) * 1024 * 1024,
            'max_tokens': _int(raw.get('max_tokens'), DEFAULT_MAX_TOKENS, 1, 1_000_000),
            'tls': tls}


# ------------------------------------------------------------------ what a shared request may carry
class RemotePayloadError(ValueError):
    pass


def _remote_url(part):
    """True for a content part that names something to fetch (image_url, audio_url, ...) other
    than inline data: an engine with a media projector would fetch it from this machine (SSRF)."""
    for name, value in part.items():
        if isinstance(name, str) and name.endswith('_url'):
            url = value.get('url') if isinstance(value, dict) else value
            if not (isinstance(url, str) and url[:5].lower() == 'data:'):
                return True
    return False


def remote_payload(payload, max_tokens=DEFAULT_MAX_TOKENS):
    """The body of a shared inference request as the engine may see it: PAYLOAD_FIELDS only,
    chat_template_kwargs reduced to the boolean TEMPLATE_KWARGS, an explicit output limit capped at
    `max_tokens` (no limit sent: the profile's own default applies, as for a local request).
    Raises RemotePayloadError for a body that is not an object or that points at a URL to fetch."""
    if not isinstance(payload, dict):
        raise RemotePayloadError('Send a JSON object.')
    out = {k: v for k, v in payload.items() if k in PAYLOAD_FIELDS}
    kwargs = payload.get('chat_template_kwargs')
    if isinstance(kwargs, dict):
        kept = {k: kwargs[k] for k in TEMPLATE_KWARGS if type(kwargs.get(k)) is bool}
        if kept:
            out['chat_template_kwargs'] = kept
    asked = [payload[k] for k in LENGTH_FIELDS if type(payload.get(k)) is int and payload[k] > 0]
    if asked:
        out['max_tokens'] = min(asked[0], max_tokens)
    for message in out['messages'] if isinstance(out.get('messages'), list) else ():
        content = message.get('content') if isinstance(message, dict) else None
        for part in content if isinstance(content, list) else ():
            if isinstance(part, dict) and _remote_url(part):
                raise RemotePayloadError('Shared requests take inline (data:) attachments only.')
    return out


def is_tls(connection):
    return isinstance(connection, ssl.SSLSocket)


def read_config(path):
    try:
        value = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def save_mode(config_path, mode, guard=None, pin=None, paused=None):
    """Persist `lan_share` in runtime.json. `lan` takes the network `pin` (recorded now; without
    one the controller pins the network it sees first); every mode set here clears an earlier
    pause, `paused` (the M1 auto-off) records one and drops the pin. The file keeps its mode, and a
    symlinked runtime.json stays a symlink (its target is replaced)."""
    if mode not in MODES:
        raise ValueError('mode must be off, tailscale or lan')

    def change(value):
        value['lan_share'] = mode
        value.pop('lan_share_paused', None)
        if mode != 'lan' or paused is not None:
            value.pop('lan_share_pin', None)
        if mode == 'lan' and pin is not None:
            value['lan_share_pin'] = dict(pin)
        if paused is not None:
            value['lan_share_paused'] = paused
    update_config(config_path, change, guard)


def update_config(config_path, change, guard=None):
    """Read-modify-write runtime.json: `change(value)` edits the dict in place; atomic replace of
    the real file (a symlink is followed, never replaced) with its permission bits kept."""
    path = Path(os.path.realpath(config_path))
    if guard is not None:
        guard(Path(config_path))
        guard(path)
    try:
        mode = stat.S_IMODE(os.stat(path).st_mode)
        value = json.loads(path.read_text())
    except FileNotFoundError:
        mode, value = 0o600, {}
    if not isinstance(value, dict):
        raise ValueError('runtime config is not an object')
    change(value)
    _write_private(path, (json.dumps(value, indent=2) + '\n').encode(), mode)


def write_config(config_path, value):
    """Audit S6: replace runtime.json with `value` the same way as update_config: a private
    O_EXCL temporary (0600, independent of the umask), fsync, same-directory rename, directory
    fsync; the real file's permission bits are kept, a symlink is followed, never replaced."""
    path = Path(os.path.realpath(config_path))
    try:
        mode = stat.S_IMODE(os.stat(path).st_mode)
    except FileNotFoundError:
        mode = 0o600
    _write_private(path, (json.dumps(value, indent=2) + '\n').encode(), mode)


def _write_private(path, data, mode=0o600):
    path = Path(path)
    temporary = path.with_name('.' + path.name + '.' + secrets.token_hex(4) + '.tmp')
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
            if mode != 0o600:
                os.fchmod(stream.fileno(), mode)
        os.replace(temporary, path)
    except BaseException:
        try: os.unlink(temporary)
        except OSError: pass
        raise
    try:  # the rename itself is durable only once the directory is synced
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try: os.fsync(directory)
        finally: os.close(directory)
    except OSError:
        pass


_ABSOLUTE_PATH = re.compile(r'(?<![\w:/])(?:~|/)[^\s,;:()\'"]*[^\s,;:()\'".]')


def public_message(text):
    """Audit N9: a controller reason for another device, with filesystem paths (binary, model,
    lease, state directory) replaced; this host's layout is not the remote client's business."""
    return _ABSOLUTE_PATH.sub('<path>', text) if isinstance(text, str) else ''


# ------------------------------------------------------------------ key file
class KeyFileError(Exception):
    pass


def key_path(config_path):
    """$LOCAL_AI_SHARE_KEY_FILE, else lan-share-key.json beside the llm/ config directory
    (~/.config/local-ai/lan-share-key.json for the installed layout)."""
    override = os.environ.get(KEY_FILE_ENV, '').strip()
    if override:
        return Path(override)
    return Path(config_path).resolve().parent.parent / 'lan-share-key.json'


def new_key():
    return KEY_PREFIX + secrets.token_urlsafe(KEY_BYTES)


def fingerprint(key):
    """A short, non-secret name for the key (status, logs, the widget)."""
    return 'sha256:' + hashlib.sha256(key.encode()).hexdigest()[:12] if isinstance(key, str) else None


def read_key(path):
    """The key record {'key', 'created', 'shown'}, None when there is no file. A file that is not
    a regular owner-only file of this user raises KeyFileError (the share then refuses every key)."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except FileNotFoundError:
        return None
    except OSError as error:
        raise KeyFileError('key file is unreadable or a symlink') from error
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise KeyFileError('key file must be a regular file owned by you with mode 0600')
        raw = os.read(fd, 4097)
    finally:
        os.close(fd)
    try:
        record = json.loads(raw)
    except ValueError:
        raise KeyFileError('key file is not valid JSON') from None
    if not isinstance(record, dict) or not isinstance(record.get('key'), str) or not _KEY_RE.fullmatch(record['key']):
        raise KeyFileError('key file holds no valid key')
    return {'key': record['key'], 'created': record.get('created'), 'shown': record.get('shown') is True}


def _private_parent(path):
    created = not path.parent.exists()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if created:
        os.chmod(path.parent, 0o700)


def write_key(path, record, guard=None):
    path = Path(path)
    if guard is not None:
        guard(path)
    _private_parent(path)
    _write_private(path, (json.dumps(record) + '\n').encode())
    return record


@contextlib.contextmanager
def _key_lock(path, guard=None):
    """One writer of the key file at a time, across the controller and the CLI (flock on a
    sibling lock file): create-if-missing, rotate and mark-shown are read-modify-write steps, and
    without it two of them could leave a key on disk that is not the one that was printed."""
    path = Path(path)
    if guard is not None:
        guard(path)
    _private_parent(path)
    fd = os.open(path.with_name('.' + path.name + '.lock'), os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


def _fresh_record():
    return {'key': new_key(), 'created': int(time.time()), 'shown': False}


def ensure_key(path, guard=None):
    record = read_key(path)
    if record is None:
        with _key_lock(path, guard):
            record = read_key(path) or write_key(path, _fresh_record(), guard)
    return record


def rotate_key(path, guard=None):
    """A new key; the old one stops working at the controller's next request (stat change)."""
    with _key_lock(path, guard):
        return write_key(path, _fresh_record(), guard)


def mark_shown(path, guard=None):
    with _key_lock(path, guard):
        record = read_key(path)
        if record is not None and not record['shown']:
            record = write_key(path, {**record, 'shown': True}, guard)
        return record


def reveal_key(path, rotate=False, guard=None):
    """(record, show): the current key (a new one with `rotate`, or when there is none) and
    whether it may be printed now. In one locked step the key is marked shown, so the key that is
    printed is the key on disk and it is printed once."""
    with _key_lock(path, guard):
        record = None if rotate else read_key(path)
        if record is None:
            record = write_key(path, _fresh_record(), guard)
        if record['shown']:
            return record, False
        write_key(path, {**record, 'shown': True}, guard)
        return record, True


def presented_key(headers):
    value = headers.get('Authorization')
    if isinstance(value, str) and value[:7].lower() == 'bearer ':
        return value[7:].strip()
    for name in ('x-api-key', 'api-key'):
        value = headers.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def key_matches(given, key):
    """Constant-time comparison (hmac.compare_digest over equal-length digests)."""
    if not isinstance(given, str) or not isinstance(key, str):
        return False
    return hmac.compare_digest(hashlib.sha256(given.encode('utf-8', 'replace')).digest(),
                               hashlib.sha256(key.encode()).digest())


# ------------------------------------------------------------------ address selection
def _ip_tool():
    return shutil.which('ip') or next((p for p in ('/usr/bin/ip', '/sbin/ip', '/usr/sbin/ip') if os.path.exists(p)), None)


def parse_ip_addr(text):
    """[(ifname, ip_address)] from `ip -j addr show`."""
    rows = []
    for link in json.loads(text or '[]'):
        if not isinstance(link, dict) or not isinstance(link.get('ifname'), str):
            continue
        for info in link.get('addr_info') or ():
            try:
                rows.append((link['ifname'], ipaddress.ip_address(info['local'])))
            except (KeyError, TypeError, ValueError):
                continue
    return rows


def parse_default_devs(text):
    """Device names of the IPv4 default routes, in `ip -j -4 route show default` order."""
    return [row['dev'] for row in json.loads(text or '[]')
            if isinstance(row, dict) and isinstance(row.get('dev'), str)]


def parse_link_types(text):
    """{ifname: link_type} from `ip -j addr show` ('ether' for Ethernet and Wi-Fi, 'none' for
    tun and WireGuard devices)."""
    return {link['ifname']: link.get('link_type') for link in json.loads(text or '[]')
            if isinstance(link, dict) and isinstance(link.get('ifname'), str)}


def lan_devices(default_devs, link_types):
    """The default-route devices `lan` may share on: Ethernet/Wi-Fi only. With a VPN as the
    default route (tun0, wg0: link type 'none') `lan` would otherwise share the models with the
    VPN's network instead of the home network."""
    return [dev for dev in default_devs if link_types.get(dev) == 'ether']


def discover(run=subprocess.run):
    """(addresses, default route devices `lan` may use) from iproute2; raises OSError when
    unavailable."""
    tool = _ip_tool()
    if tool is None:
        raise OSError('iproute2 `ip` was not found')
    addrs = run([tool, '-j', 'addr', 'show'], capture_output=True, text=True, timeout=5, check=True).stdout
    routes = run([tool, '-j', '-4', 'route', 'show', 'default'], capture_output=True, text=True, timeout=5, check=True).stdout
    return parse_ip_addr(addrs), lan_devices(parse_default_devs(routes), parse_link_types(addrs))


def parse_lan_network(addrs_text, routes_text):
    """The network `lan` would share on, from `ip -j addr show` and `ip -j -4 route show default`:
    {'interface', 'address', 'subnet', 'gateway'} for the first default route on an Ethernet or
    Wi-Fi device with a private IPv4 address (as select_addresses picks it), else None."""
    links = [link for link in json.loads(addrs_text or '[]') if isinstance(link, dict) and isinstance(link.get('ifname'), str)]
    types = {link['ifname']: link.get('link_type') for link in links}
    for row in json.loads(routes_text or '[]'):
        if not isinstance(row, dict) or not isinstance(row.get('dev'), str):
            continue
        dev = row['dev']
        if dev.startswith('tailscale') or types.get(dev) != 'ether':
            continue
        try:
            gateway = str(ipaddress.IPv4Address(row['gateway'])) if row.get('gateway') is not None else None
        except (ValueError, TypeError):
            continue
        for link in links:
            if link['ifname'] != dev:
                continue
            for info in link.get('addr_info') or ():
                try:
                    iface = ipaddress.IPv4Interface('%s/%d' % (info['local'], int(info.get('prefixlen', 32))))
                except (KeyError, TypeError, ValueError):
                    continue
                a = iface.ip
                if any(a in net for net in LAN_NETS) and not (a.is_unspecified or a.is_loopback or a.is_link_local):
                    return {'interface': dev, 'address': str(a), 'subnet': str(iface.network), 'gateway': gateway}
    return None


def parse_neigh_mac(text):
    """The link-layer address in `ip -j neigh show <gateway> dev <dev>` output (lower case), or
    None (no entry, or an incomplete or failed one)."""
    for row in json.loads(text or '[]'):
        if isinstance(row, dict) and isinstance(row.get('lladdr'), str):
            mac = row['lladdr'].lower()
            if _MAC_RE.fullmatch(mac) and 'FAILED' not in (row.get('state') or ()):
                return mac
    return None


def discover_lan(run=subprocess.run):
    """parse_lan_network plus 'gateway_mac' (the neighbour table, no root needed; None when it
    has no entry yet); None when there is no LAN network. Raises OSError without iproute2."""
    tool = _ip_tool()
    if tool is None:
        raise OSError('iproute2 `ip` was not found')
    addrs = run([tool, '-j', 'addr', 'show'], capture_output=True, text=True, timeout=5, check=True).stdout
    routes = run([tool, '-j', '-4', 'route', 'show', 'default'], capture_output=True, text=True, timeout=5, check=True).stdout
    network = parse_lan_network(addrs, routes)
    if network is None:
        return None
    mac = None
    if network['gateway'] is not None:
        try:
            mac = parse_neigh_mac(run([tool, '-j', 'neigh', 'show', network['gateway'], 'dev', network['interface']],
                                      capture_output=True, text=True, timeout=5, check=True).stdout)
        except (OSError, ValueError, subprocess.SubprocessError):
            mac = None
    return {**network, 'gateway_mac': mac}


def pin_of(network):
    """The pin recorded for a network (no address: a new DHCP lease on the same network is fine)."""
    return {name: network.get(name) for name in PIN_FIELDS}


def pin_from_config(value):
    """runtime.json's `lan_share_pin`, validated; None when absent or malformed."""
    raw = value.get('lan_share_pin') if isinstance(value, dict) else None
    if not isinstance(raw, dict) or not isinstance(raw.get('interface'), str) or not isinstance(raw.get('subnet'), str):
        return None
    try:
        ipaddress.IPv4Network(raw['subnet'])
        if raw.get('gateway') is not None:
            ipaddress.IPv4Address(raw['gateway'])
    except (ValueError, TypeError):
        return None
    mac = raw.get('gateway_mac')
    if mac is not None and not (isinstance(mac, str) and _MAC_RE.fullmatch(mac)):
        return None
    return pin_of(raw)


def compare_pin(pin, network):
    """'same', 'changed' (another network: turn sharing off), 'absent' (no LAN network now: share
    nothing, wait) or 'unverified' (the pinned gateway's MAC is not in the neighbour table yet:
    share nothing, look again at the next tick)."""
    if network is None:
        return 'absent'
    if any(pin[name] != network.get(name) for name in ('interface', 'subnet', 'gateway')):
        return 'changed'
    if pin['gateway_mac'] is not None:
        if network.get('gateway_mac') is None:
            return 'unverified'
        if network['gateway_mac'] != pin['gateway_mac']:
            return 'changed'
    return 'same'


def _is_tailscale(ifname, address):
    return ifname.startswith('tailscale') and any(address in net for net in TS_NETS if net.version == address.version)


def select_addresses(mode, addrs, default_devs):
    """(['addr', ...], error or None). Never an unspecified, loopback or public address."""
    if mode == 'tailscale':
        chosen = [str(a) for ifname, a in addrs if _is_tailscale(ifname, a)
                  and not (a.is_unspecified or a.is_loopback or a.is_link_local)]
        return (list(dict.fromkeys(chosen)), None) if chosen else (
            [], 'Tailscale is not up: no 100.64/10 or fd7a:115c:a1e0::/48 address on a tailscale interface.')
    if mode == 'lan':
        for dev in default_devs:
            if dev.startswith('tailscale'):
                continue
            for ifname, a in addrs:
                if (ifname == dev and a.version == 4 and any(a in net for net in LAN_NETS)
                        and not (a.is_unspecified or a.is_loopback or a.is_link_local)):
                    return [str(a)], None
        return [], ('No private LAN address (10/8, 172.16/12, 192.168/16) on an Ethernet or Wi-Fi '
                    'default-route interface.')
    return [], None


# ------------------------------------------------------------------ local use first (M4)
class Preempt:
    """The running shared request, which a local request cancels on arrival (security review M4):
    `cancel` shuts the shared request's engine connection down (the engine stops generating as for
    a client that went away), and the controller answers the remote client 'preempted by local
    use'. A cancel before the engine connection exists is applied when it is attached."""

    def __init__(self):
        self.event = threading.Event()
        self._lock = threading.Lock()
        self._sock = None

    @property
    def cancelled(self):
        return self.event.is_set()

    def attach(self, sock):
        with self._lock:
            self._sock = sock
            cancelled = self.event.is_set()
        if cancelled:
            self._shutdown(sock)

    def cancel(self):
        with self._lock:
            self.event.set()
            sock = self._sock
        if sock is not None:
            self._shutdown(sock)

    @staticmethod
    def _shutdown(sock):
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass


PREEMPTED = {'message': 'Preempted by local use on the host; retry later.', 'type': 'preempted',
             'code': 'preempted_by_local_use'}


# ------------------------------------------------------------------ rate limit
class RateLimiter:
    """Token bucket per client address: `burst` requests at once, refilled at rate_per_min."""

    def __init__(self, rate_per_min, burst, clock=time.monotonic, max_clients=1024):
        self.rate = rate_per_min / 60.0
        self.burst = float(burst)
        self.clock = clock
        self.max_clients = max_clients
        self.buckets = OrderedDict()
        self.lock = threading.Lock()

    def allow(self, client):
        now = self.clock()
        with self.lock:
            tokens, then = self.buckets.pop(client, (self.burst, now))
            tokens = min(self.burst, tokens + (now - then) * self.rate)
            allowed = tokens >= 1.0
            self.buckets[client] = (tokens - 1.0 if allowed else tokens, now)
            while len(self.buckets) > self.max_clients:
                self.buckets.popitem(last=False)
            return allowed


# ------------------------------------------------------------------ listeners
class ShareServer(ThreadingHTTPServer):
    """One shared listener: the controller's Handler, plus connection tracking so an off switch
    can close open (keep-alive or streaming) connections at once, and so a client (with or
    without the key) cannot hold more than its share of threads and file descriptors."""
    daemon_threads = True
    allow_reuse_address = True
    io_timeout = IO_TIMEOUT
    max_connections = MAX_CONNECTIONS
    max_connections_per_client = MAX_CONNECTIONS_PER_CLIENT

    def __init__(self, address, handler, listener):
        self.share_listener = listener
        self.connections = {}  # open socket -> client address
        self.connections_lock = threading.Lock()
        super().__init__(address, handler)

    def server_bind(self):
        if self.address_family == socket.AF_INET6:
            self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
        # TCPServer's bind only: HTTPServer.server_bind would reverse-resolve the address
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[0], self.server_address[1]

    def get_request(self):
        request, client_address = super().get_request()
        # every read (the TLS handshake, the request line, headers, the body) and every write
        # gives up after io_timeout: a silent or stalled peer frees its thread, and a client that
        # stops reading a stream frees the inference slot
        request.settimeout(self.io_timeout)
        return request, client_address

    def verify_request(self, request, client_address):
        """In the accept loop, before a thread exists: refuse (close) when the listener is closed
        or the connection limits are reached."""
        if self.share_listener.closed:
            return False
        client = client_address[0]
        with self.connections_lock:
            if (len(self.connections) >= self.max_connections
                    or sum(1 for other in self.connections.values() if other == client) >= self.max_connections_per_client):
                return False
            self.connections[request] = client
        return True

    def shutdown_request(self, request):
        with self.connections_lock:
            self.connections.pop(request, None)
        super().shutdown_request(request)

    def handle_error(self, request, client_address):
        """No traceback with the client's address in the journal: a timeout or a reset is
        routine and silent, anything else is one line naming the exception type only."""
        error = sys.exc_info()[1]
        if not isinstance(error, OSError):
            print('local-ai: LAN share: a shared request failed (%s)' % type(error).__name__,
                  file=sys.stderr, flush=True)

    def drop_connections(self):
        with self.connections_lock:
            open_now = list(self.connections)
        for connection in open_now:
            try: connection.shutdown(socket.SHUT_RDWR)
            except OSError: pass


class ShareServer6(ShareServer):
    address_family = socket.AF_INET6


class Listener:
    def __init__(self, controller, address, port, handler, manager, tls=None):
        self.controller = controller
        self.address = address
        self.closed = False
        family6 = ipaddress.ip_address(address).version == 6
        self.server = (ShareServer6 if family6 else ShareServer)((address, port), handler, self)
        self.server.manager = manager
        if tls is not None:
            # handshake on the first read in the handler thread, never in the accept loop
            self.server.socket = tls.wrap_socket(self.server.socket, server_side=True,
                                                 do_handshake_on_connect=False)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={'poll_interval': 0.2},
                                       name='lan-share-' + address, daemon=True)
        self.thread.start()

    def close(self):
        self.closed = True
        try:
            self.server.shutdown()
        finally:
            self.server.server_close()
            self.server.drop_connections()
            self.thread.join(2)


def _host(address):
    return '[' + address + ']' if ':' in address else address


class LanUnavailable(ValueError):
    """`share on lan` with no LAN network to pin."""


class ShareController:
    """Owns the shared listeners, the key cache, the limiter and the one-at-a-time slot."""

    def __init__(self, config_path, port, handler, *, manager=None, guard=None, salt=None,
                 discover_fn=None, lan_fn=None, clock=time.monotonic, log=None):
        self.config_path = Path(config_path)
        self.port = port
        self.handler = handler
        self.manager = manager
        self.guard = guard
        self.salt = salt
        self.discover_fn = discover_fn or discover
        self.lan_fn = lan_fn or discover_lan
        self.log = log or (lambda line: print(line, file=sys.stderr, flush=True))
        self.pin = None  # the pinned `lan` network (runtime.json lan_share_pin)
        self.paused = None  # PAUSED after the M1 auto-off, until a mode is set again
        self.clock = clock
        self.lock = threading.RLock()
        self.listeners = {}
        self.mode = 'off'
        self.opts = options({})
        self.tls_on = False
        self.error = None
        self.limiter = RateLimiter(self.opts['rate_per_min'], self.opts['burst'], clock)
        self.slot = threading.Lock()
        self.slot_wait = SLOT_WAIT
        self.stats = {'remote_requests': 0, 'remote_refused': 0}
        self._key_cache = (None, None, None)  # (stat signature, key or None, error or None)
        self._process_salt = secrets.token_bytes(32)
        self._config_sig = None
        self.stop_event = threading.Event()
        self.watcher = None

    # ---- key
    def key_file(self):
        return key_path(self.config_path)

    def _key_state(self):
        path = self.key_file()
        try:
            info = os.stat(path, follow_symlinks=False)
            sig = (info.st_ino, info.st_mtime_ns, info.st_size, info.st_mode, info.st_uid)
        except FileNotFoundError:
            sig = 'missing'
        except OSError:
            sig = 'error'
        cached_sig, key, error = self._key_cache
        if sig != cached_sig:
            try:
                record = read_key(path)
                key, error = (record['key'] if record else None), None
            except KeyFileError as failure:
                key, error = None, str(failure)
            self._key_cache = (sig, key, error)
        return key, error

    def key(self):
        return self._key_state()[0]

    # ---- apply
    def _config_signature(self):
        try:
            info = self.config_path.stat()
            return (info.st_ino, info.st_mtime_ns, info.st_size)
        except OSError:
            return None

    def apply(self):
        """Bring the listeners in line with runtime.json and the current addresses."""
        with self.lock:
            self._config_sig = self._config_signature()
            value = read_config(self.config_path)
            mode, opts = mode_of(value), options(value)
            error, wanted, tls = None, [], None
            self.pin = pin_from_config(value) if mode == 'lan' else None
            self.paused = PAUSED if mode == 'off' and isinstance(value.get('lan_share_paused'), dict) else None
            if mode == 'lan':
                try:
                    wanted, error = self._lan_addresses()
                except (OSError, ValueError, subprocess.SubprocessError) as failure:
                    error = 'Could not list network addresses: ' + type(failure).__name__
                if self.paused is not None:  # the network changed just now: sharing is off
                    mode, wanted, error = 'off', [], None
            elif mode != 'off':
                try:
                    addrs, devs = self.discover_fn()
                    wanted, error = select_addresses(mode, addrs, devs)
                except (OSError, ValueError, subprocess.SubprocessError) as failure:
                    error = 'Could not list network addresses: ' + type(failure).__name__
            if mode != 'off':
                try:
                    ensure_key(self.key_file(), self.guard)
                except (OSError, KeyFileError) as failure:
                    error, wanted = 'Share key unavailable: ' + str(failure), []
                if opts['tls'] is not None and wanted:
                    try:
                        tls = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
                        tls.minimum_version = ssl.TLSVersion.TLSv1_2
                        tls.load_cert_chain(*opts['tls'])
                    except (OSError, ssl.SSLError, ValueError):
                        error, wanted, tls = 'TLS certificate or key could not be loaded; not sharing.', [], None
            restart = (opts['tls'] != self.opts['tls'])
            if (opts['rate_per_min'], opts['burst']) != (self.opts['rate_per_min'], self.opts['burst']):
                self.limiter = RateLimiter(opts['rate_per_min'], opts['burst'], self.clock)
            self.mode, self.opts, self.tls_on = mode, opts, tls is not None
            for address in list(self.listeners):
                if restart or address not in wanted:
                    self.listeners.pop(address).close()
            for address in wanted:
                if address in self.listeners:
                    continue
                try:
                    self.listeners[address] = Listener(self, address, self.port, self.handler, self.manager, tls)
                except OSError as failure:
                    error = f'Could not listen on {_host(address)}:{self.port}: {failure.strerror or failure}'
            self.error = error
            return self.status()

    def _lan_addresses(self):
        """(addresses, error) for `lan` against the pin (M1). No pin yet (runtime.json edited by
        hand, an older setting): the network seen now is pinned. Another network: sharing turns
        itself off (saved, logged, status `paused`), it never rebinds there."""
        network = self.lan_fn()
        if self.pin is None:
            if network is None:
                return [], ('No private LAN address (10/8, 172.16/12, 192.168/16) on an Ethernet or Wi-Fi '
                            'default-route interface.')
            pin = pin_of(network)
            update_config(self.config_path, lambda value: value.__setitem__('lan_share_pin', pin), self.guard)
            self._config_sig = self._config_signature()
            self.pin = pin
        verdict = compare_pin(self.pin, network)
        if verdict == 'same':
            return [network['address']], None
        if verdict == 'absent':
            return [], 'Not connected to the network LAN share was turned on for; waiting for it.'
        if verdict == 'unverified':
            return [], "Checking that this is the network LAN share was turned on for (the gateway's address is not known yet)."
        save_mode(self.config_path, 'off', self.guard, paused={'reason': 'network_changed', 't': int(time.time())})
        self._config_sig = self._config_signature()
        self.pin, self.paused = None, PAUSED
        self.log('local-ai: LAN share: the network changed (another subnet, gateway or gateway MAC); '
                 'sharing is OFF. Run `local-ai share on lan` to share on this network.')
        return [], None

    def set_mode(self, mode):
        """Save and apply a mode. `lan` pins the network the machine is on now (M1); with no LAN
        network to pin it raises LanUnavailable and nothing is saved."""
        if mode not in MODES:
            raise ValueError('Send {"mode": "off" | "tailscale" | "lan"}.')
        with self.lock:
            pin = None
            if mode == 'lan':
                try:
                    network = self.lan_fn()
                except (OSError, ValueError, subprocess.SubprocessError) as failure:
                    raise LanUnavailable('Could not list network addresses: ' + type(failure).__name__) from None
                if network is None:
                    raise LanUnavailable('No private LAN network to share on now (an Ethernet or Wi-Fi default '
                                         'route with a 10/8, 172.16/12 or 192.168/16 address).')
                pin = pin_of(network)
            save_mode(self.config_path, mode, self.guard, pin=pin)
            return self.apply()

    def close(self):
        self.stop_event.set()
        with self.lock:
            for address in list(self.listeners):
                self.listeners.pop(address).close()

    def start_watcher(self, interval=WATCH_SECONDS):
        """Re-apply when runtime.json changes, and re-discover addresses while sharing (Tailscale
        coming up late, a new DHCP lease)."""
        def loop():
            while not self.stop_event.wait(interval):
                try:
                    if self.mode != 'off' or self._config_signature() != self._config_sig:
                        self.apply()
                except Exception:
                    pass  # the next tick retries; status carries the last error
        self.watcher = threading.Thread(target=loop, name='lan-share-watch', daemon=True)
        self.watcher.start()

    def serving(self, listener):
        return (not listener.closed and self.mode != 'off'
                and self.listeners.get(listener.address) is listener)

    # ---- status
    def status(self):
        with self.lock:
            key, key_error = self._key_state()
            shown = False
            if key is not None:
                try: shown = bool((read_key(self.key_file()) or {}).get('shown'))
                except KeyFileError: shown = False
            addresses = list(self.listeners)
            scheme = 'https' if self.tls_on else 'http'
            encryption = ('tls' if self.tls_on else 'wireguard' if self.mode == 'tailscale'
                          else 'none' if self.mode == 'lan' else None)
            return {'enabled': self.mode != 'off', 'mode': self.mode, 'active': bool(addresses),
                    'addresses': addresses, 'port': self.port,
                    'urls': [f'{scheme}://{_host(a)}:{self.port}/v1' for a in addresses],
                    'key_present': key is not None, 'key_fingerprint': fingerprint(key), 'key_shown': shown,
                    'encryption': encryption,
                    'warning': LAN_WARNING if self.mode == 'lan' and not self.tls_on else None,
                    'error': self.error or (key_error if self.mode != 'off' else None),
                    'paused': self.paused,
                    'pinned_network': ({k: self.pin[k] for k in ('interface', 'subnet', 'gateway')}
                                       | {'gateway_mac_known': self.pin['gateway_mac'] is not None}
                                       if self.mode == 'lan' and self.pin else None),
                    'routes': list(ROUTES),
                    'limits': {'rate_per_min': self.opts['rate_per_min'], 'burst': self.opts['burst'],
                               'max_body_bytes': self.opts['max_body_bytes'], 'max_tokens': self.opts['max_tokens'],
                               'concurrency': 1},
                    'remote_active': 1 if self.slot.locked() else 0, **self.stats}

    def startup_line(self):
        value = self.status()
        if not value['enabled']:
            if value['paused']:
                return f"local-ai: LAN share: off (loopback only); {value['paused']}"
            return 'local-ai: LAN share: off (loopback only)'
        if not value['active']:
            return f"local-ai: LAN share: {value['mode']} requested but NOT active ({value['error']}); loopback only"
        note = {'wireguard': 'encrypted by WireGuard', 'tls': 'TLS', 'none': 'UNENCRYPTED'}[value['encryption']]
        return (f"local-ai: LAN share: {value['mode']} ON at {', '.join(value['urls'])} ({note}); "
                f"API key {value['key_fingerprint']} required; inference routes only")

    # ---- the gate
    def client_hash(self, address):
        """Keyed hash of the client address for the ledger: never the address itself."""
        salt = None
        if self.salt is not None:
            try: salt = self.salt()
            except Exception: salt = None
        digest = hmac.new(salt or self._process_salt, b'lan-share-client-v1\0' + str(address).encode(), hashlib.sha256)
        return digest.hexdigest()[:16]

    def refuse(self, handler, code, message, kind=None):
        with self.lock:
            self.stats['remote_refused'] += 1
        handler.close_connection = True
        error = {'message': message}
        if kind: error['type'] = kind
        return handler.json_reply(code, {'error': error})

    def handle(self, handler, inner):
        """Gate one request on a shared listener; `inner` is the controller's own routing."""
        listener = handler.server.share_listener
        if not self.serving(listener):
            return self.refuse(handler, 503, 'LAN share is off.')
        client = handler.client_address[0]
        if not self.limiter.allow(client):
            return self.refuse(handler, 429, 'Too many requests from this device; slow down.', 'rate_limit_error')
        key = self.key()
        if key is None or not key_matches(presented_key(handler.headers), key):
            return self.refuse(handler, 401, 'Missing or wrong share key. Send "Authorization: Bearer <key>" '
                               '(local-ai share key on the host).', 'authentication_error')
        raw = handler.path
        parts = urlsplit(raw) if isinstance(raw, str) and raw.startswith('/') and not raw.startswith('//') else None
        path = (parts.path.rstrip('/') or '/') if parts is not None and not parts.netloc else None
        method = handler.command
        allowed = path is not None and (
            (method == 'POST' and path in POST_ROUTES)
            or (method in ('GET', 'HEAD') and (path in GET_ROUTES or (
                path.startswith('/v1/models/') and MODEL_ID.fullmatch(path[len('/v1/models/'):]) is not None
                and path.strip('./') != 'v1/models'))))  # no '.' or '..' as the id
        if not allowed:
            return self.refuse(handler, 403, 'Only the OpenAI inference routes are shared '
                               '(POST /v1/chat/completions, POST /v1/completions, GET /v1/models).', 'permission_error')
        if handler.headers.get('Transfer-Encoding'):
            return self.refuse(handler, 411, 'Use a content length for request bodies.')
        try:
            length = int(handler.headers.get('Content-Length', '0'))
        except ValueError:
            return self.refuse(handler, 400, 'Invalid content length.')
        if length < 0 or length > self.opts['max_body_bytes']:
            return self.refuse(handler, 413, 'Request is too large for a shared request '
                               f"(limit {self.opts['max_body_bytes'] // (1024 * 1024)} MiB).")
        if not self.slot.acquire(timeout=self.slot_wait):
            return self.refuse(handler, 429, 'Another shared request is running; retry when it finishes.',
                               'rate_limit_error')
        try:
            with self.lock:
                self.stats['remote_requests'] += 1
            # the share key never reaches an engine or the ledger, nor does any other header
            # outside the list
            for name in {name for name in handler.headers.keys() if name.lower() not in FORWARD_HEADERS}:
                del handler.headers[name]
            # the controller routes the path matched above, not a second parse of the raw target
            handler.path = path
            handler.share_client = self.client_hash(client)
            return inner()
        finally:
            self.slot.release()


def off_status(port=None):
    """Status block when the controller runs without a share controller."""
    return {'enabled': False, 'mode': 'off', 'active': False, 'addresses': [], 'port': port, 'urls': [],
            'key_present': False, 'key_fingerprint': None, 'key_shown': False, 'encryption': None,
            'warning': None, 'error': None, 'paused': None, 'pinned_network': None, 'routes': list(ROUTES),
            'limits': {'rate_per_min': DEFAULT_RATE_PER_MIN, 'burst': DEFAULT_BURST,
                       'max_body_bytes': DEFAULT_MAX_BODY_MB * 1024 * 1024, 'max_tokens': DEFAULT_MAX_TOKENS,
                       'concurrency': 1},
            'remote_active': 0, 'remote_requests': 0, 'remote_refused': 0}
