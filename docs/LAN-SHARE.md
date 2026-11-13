# LAN share: use the local models from your other devices

Off by default. When it is off, the controller listens on `127.0.0.1:8092` only, as it always
has. Turning it on adds a second listener on one chosen network, protected by its own API key,
that serves the OpenAI inference routes and nothing else.

## Turn it on

```sh
local-ai share on            # Tailscale (recommended): your tailnet address, 100.x / fd7a:...
local-ai share on lan        # this network only: the private IPv4 of the default route, pinned
local-ai share key           # prints the API key ONCE; store it in your client now
local-ai share status        # what is shared, the URLs, the key fingerprint
local-ai share off           # back to loopback only, at once
local-ai share key --rotate  # new key; the old one stops working immediately
```

The setting is `lan_share` in `~/.config/local-ai/llm/runtime.json` (`off`, `tailscale` or `lan`).
It is never turned on by itself, and the service logs one line at start-up saying what is shared.
`local-ai share off` also works while the service is stopped (it saves the setting).

## Connect from another machine

Any OpenAI-compatible client: base URL `http://<address>:8092/v1` (the address is in
`local-ai share status`), API key = the printed key.

```sh
curl http://<address>:8092/v1/models -H "Authorization: Bearer $LOCAL_AI_KEY"
```

With Tailscale the MagicDNS name works too (`http://laptop:8092/v1`). A shared request uses the
model that is loaded on the host: it cannot load one (nothing loaded = 409) or switch to another
one (409 naming the loaded model). Local use comes first: when the host starts a request of its
own while a shared one runs, the shared one is cancelled at once and its client gets
`preempted_by_local_use` (a 503, or an error event in a stream that had started); retry later.

## Security model

- **Only when you ask.** `off` by default; an unknown or unreadable setting counts as `off`.
- **Only the chosen interface.** `tailscale` binds the addresses of the `tailscale*` interface
  in 100.64/10 and fd7a:115c:a1e0::/48 (a mobile carrier's 100.x address on another
  interface is not used); `lan` binds the private (10/8, 172.16/12, 192.168/16) IPv4 address
  of the default-route interface, and only when that is Ethernet or Wi-Fi (with a VPN as the
  default route, `lan` shares nothing rather than sharing with the VPN's network). Never
  `0.0.0.0` or `::`, never a public address. Loopback stays as it was, with no key needed there.
- **`lan` stays on its network.** `local-ai share on lan` pins the network the machine is on at
  that moment: the interface, the subnet, the gateway and, when the neighbour table has it, the
  gateway's MAC address (`lan_share_pin` in runtime.json; no root needed). A new address on the
  same network (DHCP) is fine. On any other network (another subnet, gateway or gateway MAC: a
  cafe, a hotel) sharing turns itself **off** within 10 seconds instead of following the laptop:
  the setting is saved as `off`, the journal has one line, and status says
  `paused: network changed, re-enable to share on this network`. It stays off when you come home;
  `local-ai share on lan` turns it on again and pins the network you are on then. While the
  gateway's MAC address is not known yet, or the machine is off the network, nothing is shared
  until it is back. `tailscale` is not pinned (the tailnet is the boundary).
- **A key on every shared request.** 256 random bits, kept in
  `~/.config/local-ai/lan-share-key.json` (mode 0600, inside your home; a file with wider
  permissions or a symlink is refused, which refuses every request). Compared in constant time,
  shown once, rotatable, and never passed on to an engine or written to a log. Status and the
  panel show only a fingerprint (`sha256:` + 12 hex digits).
- **Inference only.** `POST /v1/chat/completions`, `POST /v1/completions`,
  `GET /v1/models[/id]`. Everything else is 403, above all the `/_local/*` control routes
  (load, unload, cache, budget, clear, settings, share), the built-in chat page and the
  engine's native routes. Image Studio (port 8090) is never shared.
- **OpenAI fields only.** Of a shared request, the engine sees the standard body fields
  (`model`, `messages`, `prompt`, `stream`, sampling, `stop`, `tools`, `response_format`, ...)
  and `chat_template_kwargs.enable_thinking` / `preserve_thinking`. Everything else is dropped:
  slot and cache controls (`id_slot`, `cache_prompt`), raw grammars, `n`, native engine
  fields. Attachments must be inline (`data:`): a URL for the host to fetch is a 400. Of the
  headers, only content type/length/encoding, `Accept`, `User-Agent`, a session id and the
  browser's `Origin`/`Sec-Fetch-*` are kept; `X-Local-AI-Origin`, `X-Oxe-*`, `X-Forwarded-*`,
  cookies and the key are removed.
- **Kept apart from your own use.** A shared request is not learned by the harness registry and
  does not write or restore prefix-store files. On the Rust engine it runs with RAM-only,
  request-scoped state (`X-Oxe-State-Policy: ephemeral`, set by the controller only; a client's
  own copy is removed): it reuses no cached prefix (so its timing tells nothing about what was
  asked here before), reads and writes nothing in the disk state cache, the echo records or the
  prefix store (so it can never evict your saved conversations), and its state is dropped when it
  ends. Your live conversation is set aside in RAM for it and put back exactly as it was (without a
  RAM budget it is saved to the state cache first and resumes from there). On a llama.cpp engine a
  shared request is sent with `cache_prompt: false`.
- **Limits.** Per client: a token bucket (default 120 requests a minute, bursts of 30; wrong
  keys count too). Bodies up to 8 MiB, no chunked bodies. An output limit of at most 16384
  tokens per request (`max_tokens` option). One shared request at a time, and
  only when the machine is idle: a shared request never waits in the queue ahead of, preempts
  or cancels a local one (busy = 429/503), and a local request that arrives while it runs
  cancels it (`preempted_by_local_use`). It never loads, switches or unloads a model (409); the
  check is made in the same locked step that admits the request. At most 32 open connections per
  listener and 8 per device, with or without the key; a connection that is silent, or blocks
  in one read or write, for 60 seconds is closed (this also ends a stream nobody reads).
- **Off means off.** `local-ai share off` closes the shared listeners and drops their open
  connections, including a running stream.
- **Separate statistics.** Ledger rows of shared requests have `origin: "remote"` and
  `client_addr`, a keyed hash of the client address (never the address itself).
- **Browsers.** A request carrying a foreign `Origin` header is refused as before (CORS stays
  the opt-in `openai_api.cors_origins`); shared use is meant for apps and CLIs, not web pages.

### Encryption

- `tailscale`: traffic travels inside Tailscale's WireGuard tunnel, encrypted end to end
  between your devices; plain HTTP on top is fine. Only devices on your tailnet can connect,
  and Tailscale ACLs can narrow that further.
- `lan`: **unencrypted HTTP.** Anyone on the same network can read prompts and replies in
  transit (the key too, so they could then use the API). Use it on a network you trust, prefer
  `tailscale`, or give it a certificate:

```json
"lan_share_options": {"tls_cert": "/abs/path/cert.pem", "tls_key": "/abs/path/key.pem"}
```

With a certificate the URLs become `https://`; a certificate that cannot be loaded keeps the
share off (with the reason in status). Other options: `rate_per_min` (1 to 6000), `burst`
(1 to 1000), `max_body_mb` (1 to 16), `max_tokens` (1 to 1000000, default 16384).

### Known limits

- A shared request re-reads its whole prompt every time (no cache), so long conversations from
  another device are slower than local ones.
- The ephemeral engine state needs the Rust engine from v0.1.3 with `X-Oxe-State-Policy`; an
  older engine ignores the header and treats a shared request like a local one.
- With TLS, a client that disconnects mid-request is noticed at the next write (or after 60 s),
  not at once.

## Status fields

`GET /_local/status` has a `lan_share` block (also `GET /_local/share`, loopback only):

| field | meaning |
|---|---|
| `enabled` | the setting is not `off` |
| `mode` | `off`, `tailscale` or `lan` |
| `active` | at least one shared listener is up |
| `addresses`, `port`, `urls` | where it listens; `urls` are client base URLs |
| `key_present`, `key_fingerprint`, `key_shown` | the key exists, its fingerprint, whether `share key` showed it |
| `encryption` | `wireguard`, `tls`, `none` (LAN without a certificate) or null when off |
| `warning` | the unencrypted-LAN warning, else null |
| `error` | why sharing is requested but not active (Tailscale down, no LAN address, off the pinned network, bind failure, key or TLS file), else null |
| `paused` | `paused: network changed, re-enable to share on this network` after `lan` turned itself off, else null |
| `pinned_network` | for `lan`: `interface`, `subnet`, `gateway`, `gateway_mac_known`; else null |
| `routes`, `limits` | what is shared; `rate_per_min`, `burst`, `max_body_bytes`, `max_tokens`, `concurrency` (1) |
| `remote_active`, `remote_requests`, `remote_refused` | a shared request is running now; counters since start |

`POST /_local/share {"mode": "off" | "tailscale" | "lan"}` (loopback only) saves the setting,
applies it at once and answers with the same block (`lan` with no LAN network to pin: 409,
nothing saved). Changes to runtime.json and to the network
(Tailscale coming up later, a new DHCP address) are picked up within 10 seconds.
