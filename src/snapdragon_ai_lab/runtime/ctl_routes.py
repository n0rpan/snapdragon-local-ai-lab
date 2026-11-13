"""The control routes the controller answers itself: everything under `/_local/`, single model
rows, and the model-list and health polls that must never load a model.
"""
import http.client
import json
from urllib.parse import parse_qs
from urllib.parse import urlsplit
if __package__:
    from .ctl_env import JOURNAL_HINT, StopPending, Unavailable, request_origin, unexpected_cause
    from .ctl_seams import ctl
    from . import bigmoe_runtime, lan_share, openai_api, request_ledger, state_cache, storage_view
else:
    from ctl_env import JOURNAL_HINT, StopPending, Unavailable, request_origin, unexpected_cause
    from ctl_seams import ctl
    import bigmoe_runtime
    import lan_share
    import openai_api
    import request_ledger
    import state_cache
    import storage_view


# control_route() returns this when the request is not a control route.
PASS = object()


class ControlRoutes:
    """Handler: control routes."""

    def control_route(self, path, api_value):
        """Answer a control route (`/_local/*`), a model row or a model-list/health poll. Returns
        PASS when the request is none of these and goes on to the proxy."""
        if path == '/_local/status':
            return self.json_reply(200, self.manager.status())
        if path == '/_local/flash/cache' and self.command in ('GET', 'HEAD'):
            try: return self.json_reply(200, self.manager.flash_cache_setting())
            except (Unavailable, OSError, ValueError, KeyError, TypeError, AttributeError):
                return self.json_reply(503, {'error': 'Flash cache settings are unavailable.'})
        if path == '/_local/models' and self.command in ('GET', 'HEAD'):
            try: return self.json_reply(200, self.manager.catalog())
            except (Unavailable, OSError, ValueError, KeyError, TypeError):
                return self.json_reply(503, {'error': 'Model setup and validation are pending.'})
        if path == '/_local/storage' and self.command in ('GET', 'HEAD'):
            try:
                rows = storage_view.inventory(json.loads(ctl.CONFIG.read_text()))
            except (OSError, ValueError, KeyError, TypeError, AttributeError):
                return self.json_reply(503, {'error': 'Storage status is unavailable.'})
            return self.json_reply(200, {'object': 'list', 'data': storage_view.public(rows)})
        if self.command == 'POST' and path.startswith('/slots/'):
            self.close_connection = True
            return self.json_reply(403, {'error': 'Slot persistence is controller-owned.'})
        control = {}
        if path == '/_local/ledger' and self.command in ('GET', 'HEAD'):
            wanted = parse_qs(urlsplit(self.path).query).get('limit', ['50'])[-1]
            return self.json_reply(200, request_ledger.recent(int(wanted) if wanted.isdecimal() and len(wanted) < 5 else 50))
        if path == '/_local/harnesses' and self.command in ('GET', 'HEAD'):
            try: config = self.manager.loaded_config or self.manager.config()
            except Unavailable: config = {}
            return self.json_reply(200, self.manager.harness_status(config))
        if path == '/_local/cache' and self.command in ('GET', 'HEAD'):
            try: return self.json_reply(200, self.manager.cache_usage())
            except (OSError, ValueError, KeyError, TypeError):
                return self.json_reply(503, {'error': 'Cache usage is unavailable.'})
        if path in ('/_local/load', '/_local/unload', '/_local/select', '/_local/cache/clear', '/_local/cache/budget', '/_local/harness/prepare', '/_local/harness/forget',
                    '/_local/harness/learning', '/_local/harness/learn', '/_local/harness/label',
                    '/_local/storage/delete', '/_local/flash/cache', '/_local/mode', '/_local/share') and self.command == 'POST':
            # Consume control bodies so a persistent HTTP connection remains
            # usable. These actions accept no prompt or image data.
            try:
                control_length = int(self.headers.get('Content-Length', '0'))
            except ValueError:
                self.close_connection = True
                return self.json_reply(400, {'error': 'Invalid content length.'})
            if self.headers.get('Transfer-Encoding') or not 0 <= control_length <= 1024:
                self.close_connection = True
                return self.json_reply(413, {'error': 'Control request is too large.'})
            if control_length:
                try:
                    control = json.loads(self.rfile.read(control_length))
                    if not isinstance(control, dict): raise ValueError()
                except ValueError: return self.json_reply(400, {'error': 'Invalid control request.'})
        if path == '/_local/share':
            share = getattr(self.manager, 'share', None)
            if self.command in ('GET', 'HEAD'):
                return self.json_reply(200, share.status() if share is not None else lan_share.off_status(ctl.PROXY_PORT))
            if self.command == 'POST':
                if share is None:
                    return self.json_reply(503, {'error': 'LAN share is not available in this controller.'})
                if set(control) != {'mode'} or control.get('mode') not in lan_share.MODES:
                    return self.json_reply(400, {'error': 'Send {"mode": "off" | "tailscale" | "lan"}.'})
                # runtime.json has other read-modify-write writers under manager.mu (the Flash
                # cache preference); taking it here keeps one of them from writing an older
                # `lan_share` back. A model load can hold mu for long, and the off switch must not
                # wait for that: after 5 s the change is made anyway.
                held = self.manager.mu.acquire(timeout=5)
                try: return self.json_reply(200, share.set_mode(control['mode']))
                except lan_share.LanUnavailable as error: return self.json_reply(409, {'error': str(error)})
                except (OSError, ValueError): return self.json_reply(503, {'error': 'The LAN share setting could not be saved.'})
                finally:
                    if held: self.manager.mu.release()
        if path == '/_local/flash/cache' and self.command == 'POST':
            if set(control) != {'pref'}:
                return self.json_reply(400, {'error': 'Send only a cache preference.'})
            try: result = self.manager.set_flash_cache_pref(control.get('pref'))
            except StopPending as error: return self.json_reply(503, {'error': str(error)})
            except Unavailable as error: return self.json_reply(409, {'error': str(error)})
            except ValueError as error: return self.json_reply(400, {'error': str(error)})
            except (OSError, KeyError, TypeError, AttributeError):
                return self.json_reply(503, {'error': 'Flash cache preference could not be saved.'})
            return self.json_reply(200, result)
        if path == '/_local/cache/budget' and self.command == 'POST':
            if set(control) != {'bytes'}:
                return self.json_reply(400, {'error': 'Send only {"bytes": N}.'})
            try: return self.json_reply(200, self.manager.set_cache_budget(control.get('bytes')))
            except ValueError as error: return self.json_reply(400, {'error': str(error)})
            except Unavailable as error: return self.json_reply(409, {'error': str(error)})
            except (OSError, KeyError, TypeError): return self.json_reply(503, {'error': 'The cache budget could not be applied.'})
        if path == '/_local/cache/clear' and self.command == 'POST':
            # No body: the prefix store as before plus all saved state. With a body, only the
            # disk state cache in that scope (all also clears the prefix store).
            scope = control.get('scope', 'all')
            if (not set(control) <= {'scope', 'model', 'session'} or scope not in state_cache.SCOPES
                    or (control.get('model') is not None and not (isinstance(control['model'], str) and state_cache.MODEL.fullmatch(control['model'])))
                    or (scope == 'model' and control.get('model') is None)
                    or (scope == 'session' and not (isinstance(control.get('session'), str) and state_cache.SID.fullmatch(control['session'])))):
                return self.json_reply(400, {'error': 'Send {"scope": "all"}, {"scope": "model", "model": key} or {"scope": "session", "session": id}.'})
            with self.manager.mu:
                if self.manager.active:
                    return self.json_reply(409, {'error': 'A request is running.'})
                if scope == 'all':
                    if self.manager.config().get('backend_kind') == bigmoe_runtime.KIND:
                        return self.json_reply(409, {'error': 'Unload Flash to clear its resident cache.'})
                    try:
                        cache_config = self.manager.config().get('prefix_cache', {})
                        if cache_config.get('enabled'):
                            ctl.PrefixStore(cache_config['directory'], cache_config.get('max_bytes', 3 * 1024**3)).clear()
                        self.manager.current_prefix_key = None
                        with self.manager.harness_mu: self.manager.harness_registry.cleared()
                    except (OSError, ValueError, KeyError):
                        return self.json_reply(503, {'error': 'Prefix cache could not be cleared.'})
                try: state = self.manager.clear_state_cache(scope, control.get('model'), control.get('session'))
                except Unavailable as error: return self.json_reply(409, {'error': str(error)})
                except (OSError, ValueError, http.client.HTTPException):
                    return self.json_reply(503, {'error': 'Saved state could not be cleared.'})
            return self.json_reply(200, {'cleared': True, 'state_cache': state})
        if path == '/_local/storage/delete' and self.command == 'POST':
            try: result = self.manager.delete_storage(control.get('id'))
            except Unavailable as error: return self.json_reply(409, {'error': str(error)})
            except LookupError: return self.json_reply(404, {'error': 'Unknown storage entry.'})
            except PermissionError: return self.json_reply(409, {'error': 'This cache is referenced or in use and was kept.'})
            except (OSError, ValueError, KeyError, TypeError): return self.json_reply(400, {'error': 'Storage entry could not be deleted.'})
            return self.json_reply(200, result)
        if path == '/_local/harness/forget' and self.command == 'POST':
            harness = control.get('harness')
            if harness is not None and not isinstance(harness, str):
                return self.json_reply(400, {'error': 'Invalid harness name.'})
            with self.manager.harness_mu: self.manager.harness_registry.forget(harness, ignore=control.get('ignore') is True)
            return self.json_reply(200, {'forgotten': harness or 'all', 'ignored': control.get('ignore') is True})
        if path == '/_local/harness/learning' and self.command == 'POST':
            try: return self.json_reply(200, self.manager.set_harness_learning(control.get('enabled'), control.get('min_requests')))
            except Unavailable as error: return self.json_reply(400, {'error': str(error)})
        if path == '/_local/harness/learn' and self.command == 'POST':
            with self.manager.harness_mu: approved = self.manager.harness_registry.approve(control.get('harness'))
            if not approved: return self.json_reply(404, {'error': 'No such unknown client; it must have sent a request since the controller started.'})
            return self.json_reply(200, {'approved': control.get('harness')})
        if path == '/_local/harness/label' and self.command == 'POST':
            with self.manager.harness_mu: labelled = self.manager.harness_registry.set_label(control.get('harness'), control.get('label'))
            if not labelled: return self.json_reply(400, {'error': 'Only a learned harness can be labelled, with a short plain-text label.'})
            return self.json_reply(200, {'harness': control.get('harness'), 'labelled': True})
        if path == '/_local/harness/prepare' and self.command == 'POST':
            try: result = self.manager.prepare_harness(control.get('harness'), control.get('model'))
            except Unavailable as error: return self.json_reply(409, {'error': str(error)})
            except Exception: return self.json_reply(503, {'error': 'Harness prefix could not be prepared.'})
            return self.json_reply(200, result)
        if path == '/_local/mode' and self.command == 'POST':
            # V013: {model, mode} -> 200 with the model's new `context`, 409 {'error'} when unavailable
            if set(control) != {'model', 'mode'}:
                return self.json_reply(400, {'error': 'Send {"model": alias, "mode": "fast"|"deep"|"max"}.'})
            try: context = self.manager.set_mode(control.get('model'), control.get('mode'))
            except StopPending as error: return self.json_reply(503, {'error': str(error)})
            except Unavailable as error: return self.json_reply(409, {'error': str(error)})
            return self.json_reply(200, context)
        if path == '/_local/select' and self.command == 'POST':
            try: alias = self.manager.select(control.get('model'))
            except StopPending as error: return self.json_reply(503, {'error': str(error)})
            except Unavailable as error: return self.json_reply(409, {'error': str(error)})
            return self.json_reply(200, {'selected_model': alias, 'state': self.manager.state})
        if path == '/_local/load' and self.command == 'POST':
            launches = self.manager.launches
            try:
                self.manager.begin(allow_start=True, model_alias=control.get('model'),
                                   origin=request_origin(self.headers), explicit_load=True)
            except Unavailable as error:
                return self.json_reply(503, {'error': str(error)})
            except Exception as error:
                return self.json_reply(503, {'error': 'Local model could not be started '
                                             f'({unexpected_cause(error)}). {JOURNAL_HINT}'})
            launched = self.manager.launches != launches
            alias = (self.manager.loaded_config or {}).get('model_alias')
            self.manager.end(touch=True)
            self.json_reply(200, {'state': 'ready'})
            if launched:  # an explicit load that started an engine: warm it from disk (P2b)
                try: self.manager.preload_state(alias)
                except Exception: pass  # best effort; the load already answered
            return
        if path == '/_local/unload' and self.command == 'POST':
            with self.manager.mu:
                if self.manager.active:
                    return self.json_reply(409, {'error': 'A request is running.'})
                # Optional {"model": alias}: unload only if that model is the loaded one.
                wanted, loaded = control.get('model'), self.manager.loaded_config
                if wanted is not None and (loaded is None or self.manager.process is None
                                           or loaded.get('model_alias') != wanted):
                    return self.json_reply(409, {'error': 'That model is not loaded.'})
                try: self.manager.stop_for_control()
                except StopPending as error: return self.json_reply(503, {'error': str(error)})
            return self.json_reply(200, {'state': 'unloaded'})
        if self.command in ('GET', 'HEAD') and path.startswith(('/v1/models/', '/models/')):
            wanted = path.split('/models/', 1)[1]
            try:
                rows = self.manager.models()
                target = openai_api.resolve_model(wanted, api_value)
            except (Unavailable, OSError, ValueError, KeyError, TypeError):
                return self.json_reply(503, {'error': 'Model setup and validation are pending.'})
            for row in rows:
                if row['id'] == wanted or (row['id'] == target and wanted.startswith(openai_api.PROVIDER_PREFIXES)):
                    return self.json_reply(200, row)
            return self.json_reply(404, {'error': {'message': 'Unknown local model. Choose one listed in /v1/models.',
                                                   'type': 'invalid_request_error', 'param': 'model', 'code': 'model_not_found'}})
        if path.startswith('/_local/'):
            # Every control route is answered above; nothing under /_local reaches an engine.
            return self.json_reply(404, {'error': 'Unknown local control route.'})
        # Polling these routes must never repeatedly reload an idle model.
        if self.command in ('GET', 'HEAD') and path in ('/health', '/v1/models', '/models'):
            if path != '/health':
                try:
                    if self.manager.config().get('profiles'):
                        # Qualification is independent of the backend lifecycle lock.
                        models = self.manager.models()
                        if models:
                            return self.json_reply(200, {'object': 'list', 'data': models})
                        return self.json_reply(503, {'error': 'Model setup and validation are pending.'})
                except (Unavailable, OSError, ValueError, KeyError, TypeError):
                    return self.json_reply(503, {'error': 'Model setup and validation are pending.'})
            status = self.manager.status()
            if not status.get('validated'):
                return self.json_reply(503, {'error': 'Model setup and validation are pending.'})
            if path == '/health': return self.json_reply(200, {'status': 'ok', 'sleeping': status['state'] == 'unloaded'})
            if '/v1/models' in self.manager.cache:
                content_type, data = self.manager.cache['/v1/models']
                return self.reply(200, data, content_type)
            return self.json_reply(200, {'object': 'list', 'data': [
                {'id': status['model'], 'object': 'model', 'created': 0, 'owned_by': 'local'}]})
        return PASS
