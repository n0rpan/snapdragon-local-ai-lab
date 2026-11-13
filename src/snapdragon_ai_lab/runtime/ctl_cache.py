"""The cache API behind `/_local/cache*`, `/_local/flash/cache` and `/_local/storage/delete`: the
disk state cache (engine-owned while an oxe engine runs), its preload after an explicit load,
the legacy llama.cpp slot prefix store, the Flash cache preference, and the controller's own
JSON calls to the loaded engine.
"""
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import stat
import tempfile
import threading
import time
if __package__:
    from .ctl_env import Unavailable, _log
    from .ctl_seams import ctl
    from .prefix_plan import make_plan
    from . import lan_share, model_catalog, oxe_runtime, profile_policy, request_ledger, state_cache, storage_view
else:
    from ctl_env import Unavailable, _log
    from ctl_seams import ctl
    from prefix_plan import make_plan
    import lan_share
    import model_catalog
    import oxe_runtime
    import profile_policy
    import request_ledger
    import state_cache
    import storage_view


PRELOAD_TIMEOUT = 5.0  # seconds; the engine answers 202 at once and loads in the background
PRELOAD_POLL_SECONDS = 0.5  # V013: how often the controller follows a running preload
PRELOAD_FOLLOW_SECONDS = 300.0
PRELOAD_STATES = ('idle', 'running', 'done', 'cancelled', 'unavailable')


def preload_view(value):
    """(state, src, ms) of the contract from the engine's preload status: POST reply
    {'status': started|busy|live|none} or GET /state-cache `preload`
    {'status': idle|running|done|cancelled|failed|used, 'pos', 'ms'}. Numbers only."""
    status = value.get('status') if isinstance(value, dict) else None
    ms = value.get('ms') if isinstance(value, dict) else None
    ms = round(float(ms), 1) if type(ms) in (int, float) and ms == ms and 0 <= ms < 1e9 else None
    if status in ('started', 'running'):
        return 'running', None, None
    if status in ('done', 'used'):
        return 'done', 'disk', ms
    if status == 'live':
        return 'done', 'live', None   # the engine already held a conversation
    if status in ('busy', 'cancelled', 'failed'):
        return 'cancelled', None, ms if status != 'busy' else None
    if status in ('none', 'idle'):
        return 'idle', None, None     # nothing stored to preload
    return 'unavailable', None, None


class CacheMixin:
    """Manager: saved state, prefixes and cache settings."""

    def flash_cache_setting(self, value=None):
        """Opt-in Flash preference only; config reads never start or contact a model."""
        if value is None:
            value = json.loads(ctl.CONFIG.read_text())
        profile = value.get('profiles', {}).get('qwen3.8-flash-next-local', {})
        flash = profile_policy.rust_overrides(profile)[1].get('oxe_flash', {})
        if not isinstance(flash, dict) or flash.get('cache_pref') is None:
            return {'enabled': False}
        if not oxe_runtime.is_flash(self.config('qwen3.8-flash-next-local')):
            return {'enabled': False}
        pref = oxe_runtime.flash_cache_pref
        # S2: no 20/18 on a host without swap (the MemTotal rule would page the desktop to nowhere).
        return {'enabled': True, 'pref': pref.read_pref(os.environ, flash), 'choices': pref.offered_prefs(ctl.swap_bytes()[0]),
                'desktop_floor_gib': pref.desktop_floor(flash),
                'editable': not bool(os.environ.get(pref.ENV, '').strip())}

    def set_flash_cache_pref(self, preference):
        """Persist one enabled preference while idle; never enable a profile or change its guards."""
        if not isinstance(preference, str) or preference not in oxe_runtime.flash_cache_pref.PREFS:
            raise ValueError('Choose auto, 20, 18, 16 or 14.')
        if not self.mu.acquire(timeout=2):
            raise Unavailable('A model is loading; change the preference when it is idle.')
        temporary = None
        try:
            if self.active or self.queue_length():
                raise Unavailable('A request is running or queued.')
            value = json.loads(ctl.CONFIG.read_text())
            setting = self.flash_cache_setting(value)
            if not setting['enabled']:
                raise Unavailable('Flash cache preferences are not enabled for this profile.')
            if not setting['editable']:
                raise Unavailable('The environment overrides this preference; remove that override first.')
            if preference not in setting['choices']:
                raise ValueError('20 and 18 GiB need swap on this host; choose auto, 16 or 14.')
            profile = value['profiles']['qwen3.8-flash-next-local']
            flash = profile[profile_policy.rust_overrides(profile)[0]]['oxe_flash']
            if flash['cache_pref'] == preference:
                return setting
            ctl.guard_production_path(ctl.CONFIG)
            flash['cache_pref'] = preference
            with tempfile.NamedTemporaryFile(mode='w', prefix='.flash-cache-', suffix='.tmp',
                                             dir=ctl.CONFIG.parent, delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(json.dumps(value, indent=2) + '\n')
                stream.flush(); os.fsync(stream.fileno())
            if self.loaded_config and self.loaded_config.get('model_alias') == 'qwen3.8-flash-next-local':
                self.stop_for_control()
            ctl.guard_production_path(ctl.CONFIG)
            os.replace(temporary, ctl.CONFIG)
            temporary = None
            return {**setting, 'pref': preference}
        finally:
            try:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
            finally:
                self.mu.release()

    def delete_storage(self, entry_id):
        """Delete one unreferenced cache under the inference lease."""
        with self.mu:
            if self.active: raise Unavailable('A request is running.')
            ctl.guard_production_path(ctl.CONFIG)
            document = json.loads(ctl.CONFIG.read_text())
            # With a loaded model the lease is already ours; in-use checks still apply.
            with self.held_lease():
                return storage_view.delete(document, entry_id)

    def backend_key(self):
        """The loaded oxe engine's per-launch bearer token, or None."""
        return getattr(self.process, '_local_ai_api_key', None)

    def backend_json(self, path, payload):
        connection = http.client.HTTPConnection('127.0.0.1', ctl.BACKEND_PORT, timeout=900)
        try:
            connection.request('POST', path, body=json.dumps(payload).encode(),
                               headers={'Content-Type': 'application/json', **oxe_runtime.backend_headers(self.backend_key())})
            response = connection.getresponse()
            data = response.read(4 * 1024**2 + 1)
            if response.status != 200 or len(data) > 4 * 1024**2:
                raise ValueError('Prefix operation failed')
            value = json.loads(data)
            if not isinstance(value, dict): raise ValueError('Invalid prefix response')
            return value
        finally: connection.close()

    def prepare_prefix(self, path, payload):
        # begin(inference=True) already excludes competing inference. Do not
        # hold mu here: status must remain responsive during a cold primer.
        if self.backend_runtime is not None: return
        store = self.prefix_store
        if store is None: return
        plan = None
        try:
            profile = dict(self.loaded_config)
            if not profile.get('server_sha256'):
                digest = hashlib.sha256()
                with open(profile['server_binary'], 'rb') as stream:
                    for chunk in iter(lambda: stream.read(1024**2), b''): digest.update(chunk)
                profile['server_sha256'] = digest.hexdigest()
            plan = make_plan(payload, profile) if (path in ('/chat/completions', '/v1/chat/completions')
                and isinstance(payload, dict) and payload.get('cache_prompt') is not False
                and payload.get('id_slot', 0) == 0) else None
            if plan is None:
                self.current_prefix_key = None
                return
            if self.current_prefix_key == plan.key: return
            self.current_prefix_key = None
            filename = store.lookup(plan)
            if filename:
                restored = self.backend_json('/slots/0?action=restore', {'filename': filename})
                if restored.get('n_restored', 0) <= 0:
                    store.discard(plan.key)
                    raise ValueError('Prefix was not restored')
                self.prefix_stats['restores'] += 1
            else:
                self.backend_json('/v1/chat/completions', {**plan.primer, 'id_slot': 0})
                filename = store.pending(plan)
                self.backend_json('/slots/0?action=save', {'filename': filename})
                store.commit(plan)
                self.prefix_stats['primes'] += 1
            self.current_prefix_key = plan.key
        except (OSError, ValueError, KeyError, TypeError, http.client.HTTPException):
            # A cache failure must not log context or prevent ordinary inference.
            self.current_prefix_key = None
            self.prefix_stats['failures'] += 1
            if plan is not None:
                try: store.path(plan.key, 'pending').unlink(missing_ok=True)
                except OSError: pass

    # ---- CACHE-PERFECT P2: disk state cache (state_cache.py) ----
    def cache_settings(self):
        return state_cache.settings(self.runtime_value())

    def runtime_value(self):
        try:
            value = json.loads(ctl.CONFIG.read_text())
            return value if isinstance(value, dict) else None
        except (OSError, ValueError): return None

    def state_root(self, runtime=None):
        """The disk state store root: LOCAL_AI_STATE_CACHE_ROOT / runtime.json cache.root when set
        (ValueError when that override is invalid), else the default under the home."""
        return state_cache.root(runtime=self.runtime_value() if runtime is None else runtime)

    def state_engine_live(self):
        """True while a loaded oxe engine owns the store: budget and clear go to it, never to Python."""
        return bool(self.state_engine and self.process is not None and self.process.poll() is None)

    def state_engine_json(self, method, path, payload=None):
        connection = http.client.HTTPConnection('127.0.0.1', ctl.BACKEND_PORT, timeout=30)
        try:
            connection.request(method, path, body=None if payload is None else json.dumps(payload).encode(),
                               headers={'Content-Type': 'application/json', **oxe_runtime.backend_headers(self.backend_key())})
            response = connection.getresponse()
            data = response.read(1024**2 + 1)
            value = json.loads(data) if response.status == 200 and len(data) <= 1024**2 else None
            if not isinstance(value, dict): raise ValueError('state cache engine call failed')
            return value
        finally: connection.close()

    def preload_state(self, alias):
        """CACHE-PERFECT P2b: after an explicit load, ask the engine to read its newest stored
        conversation for `alias` back from disk (POST /state-cache/preload, with the last session
        hint when known). Background thread, short timeout; the engine cancels it itself when a
        request arrives. Only an oxe engine with the state-cache flags and the route. Returns
        True when a preload was sent off."""
        with self.mu:
            if not (self.state_preload and self.state_engine_live() and self.loaded_config
                    and self.loaded_config.get('model_alias') == alias):
                return False
            headers = {'Content-Type': 'application/json', **oxe_runtime.backend_headers(self.backend_key())}
            process = self.process
            self.note_preload('running', process)
        hint = self.session_hints.get(alias)
        if hint is None:
            try: hint = request_ledger.last_session(alias)
            except Exception: hint = None
        body = json.dumps({'session': hint} if isinstance(hint, str) and re.fullmatch(r'[0-9a-f]{16}', hint) else {}).encode()

        def send():
            connection = http.client.HTTPConnection('127.0.0.1', ctl.BACKEND_PORT, timeout=ctl.PRELOAD_TIMEOUT)
            try:
                connection.request('POST', oxe_runtime.STATE_PRELOAD_ROUTE, body=body, headers=headers)
                response = connection.getresponse()
                reply = response.read(4096)
                _log.info('state preload for %s: HTTP %s %s', alias, response.status, reply[:200])
                try: started = json.loads(reply) if response.status in (200, 202) else None
                except ValueError: started = None
            except (OSError, http.client.HTTPException) as error:
                _log.warning('state preload for %s failed: %s', alias, type(error).__name__)
                started = None
            finally:
                connection.close()
            self.note_preload_engine(started, process)
            # The engine loads in the background: follow its GET /state-cache `preload` until it ends.
            deadline = time.monotonic() + ctl.PRELOAD_FOLLOW_SECONDS
            while (self.preload_info or {}).get('state') == 'running' and time.monotonic() < deadline:
                if self.stop_event.wait(ctl.PRELOAD_POLL_SECONDS) or self.process is not process:
                    return
                try:
                    usage = self.state_engine_json('GET', '/state-cache')
                except (OSError, ValueError, http.client.HTTPException):
                    return self.note_preload('unavailable', process)
                self.note_preload_engine(usage.get('preload'), process)
        threading.Thread(target=send, name='state-preload', daemon=True).start()
        return True

    def note_preload(self, state, process, src=None, ms=None):
        """Remember the preload state of `process` (ignored once another engine runs)."""
        if process is not self.process:
            return
        self.preload_info = {'state': state, 'src': src, 'ms': ms, 'at': time.monotonic()}

    def note_preload_engine(self, value, process):
        """Map the engine's preload status (POST reply or GET /state-cache `preload`) onto the contract."""
        state, src, ms = preload_view(value)
        self.note_preload(state, process, src, ms)

    def cache_usage(self):
        """GET /_local/cache: a read-only walk of the store (no engine needed); the loaded
        engine's own figures are added under `engine` when it owns the store."""
        s = self.cache_settings()
        value = state_cache.usage(self.state_root(), s['disk_budget_bytes'], s['ttl_secs'])
        value['invalid_settings'] = s['invalid']
        if self.state_engine_live():
            try:
                engine = self.state_engine_json('GET', '/state-cache')
                value['engine'] = {k: engine[k] for k in ('budget_bytes', 'used_bytes') if type(engine.get(k)) is int}
            except (OSError, ValueError, http.client.HTTPException):
                value['engine'] = None
        return value

    def set_cache_budget(self, budget):
        """Persist runtime.json `cache.disk_budget_bytes`, then evict down to it at once: by the
        loaded engine when it owns the store, else in Python. Never touches RAM state."""
        if not state_cache.valid_budget(budget):
            raise ValueError('Send {"bytes": N}: 0 (disk tier off) or a whole number of at least 4000000000.')
        if not self.mu.acquire(timeout=2): raise Unavailable('A model is loading; change the budget when it is ready.')
        try:
            try:
                value = json.loads(ctl.CONFIG.read_text())
                if not isinstance(value, dict): raise ValueError()
                value['cache'] = {**(value.get('cache') if isinstance(value.get('cache'), dict) else {}), 'disk_budget_bytes': budget}
                ctl.guard_production_path(ctl.CONFIG)
                lan_share.write_config(ctl.CONFIG, value)
            except (OSError, ValueError):
                raise Unavailable('The cache budget could not be saved.')
            ttl = state_cache.settings(value)['ttl_secs']
            if self.state_engine_live():
                try: engine = self.state_engine_json('POST', '/state-cache/budget', {'bytes': budget})
                except (OSError, ValueError, http.client.HTTPException):
                    raise Unavailable('The budget was saved; the engine did not apply it yet (it will at the next load).')
                return {'budget_bytes': budget, 'used_bytes': engine.get('used_bytes'),
                        'freed_bytes': engine.get('freed_bytes'), 'applied_by': 'engine'}
            path = self.state_root(value)
            try:
                with self.held_lease():  # no engine may open the store between plan and delete
                    freed = state_cache.gc(path, budget, ttl)
            except Unavailable:
                raise Unavailable('The budget was saved; another job is running, so saved state is trimmed later '
                                  '(at the next load or budget change).') from None
            return {'budget_bytes': budget, 'used_bytes': state_cache.usage(path, budget, ttl)['used_bytes'],
                    'freed_bytes': freed, 'applied_by': 'controller'}
        finally: self.mu.release()

    def clear_state_cache(self, scope='all', model=None, session=None):
        """Called with mu held and no request running. Returns {'scope', 'freed_bytes'}."""
        if self.state_engine_live():
            # CP2-7: the engine takes {"scope": "model", "model": <16-hex key>}: its own key clears
            # its own model (RAM side too), another key removes that model's directory only.
            body = {'scope': scope, **({'session': session} if scope == 'session' else {}),
                    **({'model': model} if scope == 'model' else {})}
            engine = self.state_engine_json('POST', '/state-cache/clear', body)
            return {'scope': scope, 'freed_bytes': engine.get('freed_bytes'), 'applied_by': 'engine'}
        runtime = self.runtime_value()
        with self.held_lease():  # across planning and deletion; mu alone does not stop another process
            freed = state_cache.clear(self.state_root(runtime), scope, model, session)
            if scope == 'all':
                # as the engine's clear-all does while it runs: the legacy Rust prefix store too
                freed += self.clear_legacy_prefix(runtime)
        return {'scope': scope, 'freed_bytes': freed, 'applied_by': 'controller'}

    def clear_legacy_prefix(self, runtime):
        """Unlink the entries (<sha>.bin with an OXEPFX01 identity, regular, ours) of the Rust
        --prefix-cache stores of the non-Flash oxe profiles. Lease held by the caller."""
        dirs = {state_cache.store_dir('prefix', runtime=runtime)}
        if state_cache.override(runtime) is None:
            profiles = list((runtime or {}).get('profiles', {}).values()) + [runtime or {}]
            for profile in profiles:
                if (isinstance(profile, dict) and profile.get('backend_kind') == oxe_runtime.KIND
                        and not oxe_runtime.is_flash(profile)):
                    directory = model_catalog.rust_prefix_directory(profile)
                    if directory is not None:
                        dirs.add(Path(directory))
        freed = 0
        for directory in dirs:
            for paths in model_catalog.prefix_identities(directory, 'rust').values():
                for path in paths:
                    try:
                        info = os.lstat(path)
                        if stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid():
                            os.unlink(path)
                            freed += info.st_size
                    except OSError:
                        pass
        return freed
