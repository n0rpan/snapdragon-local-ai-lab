"""runtime.json as the controller sees it: the resolved profile of a model, the model list
and chooser rows, and the settings a user changes (selected model, context mode, harness
learning, harness prefix preparation).
"""
import json
import os
import time
if __package__:
    from .ctl_env import Unavailable
    from .ctl_seams import ctl
    from .harness_registry import learning_settings
    from . import bigmoe_runtime, classic_beta, explicit_route, harness_registry, lan_share, model_catalog, openai_api, oxe_runtime, profile_policy
else:
    from ctl_env import Unavailable
    from ctl_seams import ctl
    from harness_registry import learning_settings
    import bigmoe_runtime
    import classic_beta
    import explicit_route
    import harness_registry
    import lan_share
    import model_catalog
    import openai_api
    import oxe_runtime
    import profile_policy


class ModelsMixin:
    """Manager: profiles, catalog and persisted settings."""

    def config(self, alias=None):
        try:
            value = json.loads(ctl.CONFIG.read_text())
            if not isinstance(value, dict): raise ValueError()
            base_alias = value.get('model_alias', 'qwen-local')
            profiles = value.get('profiles', {})
            selected = value.get('selected_model', base_alias)
            # local-default / local-heavy / openai/<name>: static names only (openai_api.py)
            alias = openai_api.resolve_model(alias, value)
            requested = selected if alias in (None, '', 'qwen-local') else alias
            if requested != base_alias and requested not in profiles:
                if requested in classic_beta.RETIRED_ALIASES:
                    raise Unavailable('This local model has been retired.')
                if profiles: raise Unavailable('Unknown local model. Choose one listed in /v1/models.')
                requested = base_alias
            selected_profile = profiles.get(requested, {})
            merged = {**value, **selected_profile, 'model_alias': requested}
            # Unlike legacy shared defaults, execution overrides are per model.
            if requested != base_alias and 'memlat_overrides' not in selected_profile:
                merged.pop('memlat_overrides', None)
            # V013: the model's stored context mode (fast/deep/max; max raises the engine cap
            # to 262144 only with an RSS proof, else deep).
            return profile_policy.apply_mode(profile_policy.resolve(merged),
                                             profile_policy.stored_mode(value, requested))
        except (OSError, ValueError):
            raise Unavailable('Local model setup is still being prepared.')

    def eligible(self, config):
        if classic_beta.is_retired(config): return False
        if config.get('backend_kind') == oxe_runtime.KIND:
            return oxe_runtime.eligible(config)
        if classic_beta.is_beta(config):
            return classic_beta.eligible(config)
        if config.get('backend_kind') not in (None, 'llama', bigmoe_runtime.KIND): return False
        if config.get('backend_kind') == bigmoe_runtime.KIND:
            return bigmoe_runtime.eligible(config)
        return bool(config.get('ready'))

    def models(self):
        value = json.loads(ctl.CONFIG.read_text())
        aliases = [value.get('model_alias', 'qwen-local'), *value.get('profiles', {})]
        if value.get('profiles'): aliases.insert(0, 'qwen-local')
        rows = [{'id': alias, 'object': 'model', 'created': 0, 'owned_by': 'local',
                 'name': model_catalog.SELECTED_ALIAS_NAME if alias == 'qwen-local' else model_catalog.display_name(alias, self.config(alias))} for alias in dict.fromkeys(aliases)
                if self.eligible(self.config(alias))]
        models = openai_api.model_rows(value, rows, value.get('selected_model', value.get('model_alias', 'qwen-local')))
        if (os.environ.get(explicit_route.ENABLE_ENV) == '1'
                and explicit_route.TARGETS['code'] in {entry['id'] for entry in rows}):
            models.append({'id': explicit_route.ALIAS, 'object': 'model', 'created': 0,
                           'owned_by': 'local', 'name': 'Local explicit route (code / think-hard)'})
        return models

    def catalog(self):
        """Chooser rows for eligible profiles; metadata only, never loads a model."""
        value = json.loads(ctl.CONFIG.read_text())
        if not isinstance(value, dict): raise ValueError('runtime_object_required')
        base = value.get('model_alias', 'qwen-local')
        profiles = value.get('profiles') if isinstance(value.get('profiles'), dict) else {}
        selected = self.config().get('model_alias')
        # Unlocked snapshot: status polling must not wait for a model load.
        loaded, state = self.loaded_config, self.state
        rows = []
        for alias in dict.fromkeys([base, *profiles]):
            try: config = self.config(alias)
            except Unavailable: continue
            if not self.eligible(config): continue
            raw = profiles.get(alias, value if alias == base else {})
            rows.append(model_catalog.row(alias, raw, config, selected=selected,
                                          loaded_alias=loaded.get('model_alias') if loaded else None, state=state,
                                          flash_cache=(getattr(self, 'flash_cache_status', None) if loaded and
                                              loaded.get('model_alias') == alias else None)))
        rows.sort(key=model_catalog.sort_key)
        return {'object': 'list', 'default_model': model_catalog.DEFAULT_ALIAS,
                'selected_model': selected, 'data': rows}

    def flash_qualification(self):
        """M3: Flash's derived status (model_catalog.qualification of the resolved installed
        profile), or None when there is no Flash profile or it is not on the Rust engine."""
        try:
            config = self.config(model_catalog.FLASH_ALIAS)
        except Unavailable:
            return None
        if config.get('model_alias') != model_catalog.FLASH_ALIAS or not oxe_runtime.is_flash(config):
            return None
        return model_catalog.qualification(config)

    def select(self, alias):
        if not isinstance(alias, str): raise Unavailable('Choose a local model.')
        if not self.mu.acquire(blocking=False): raise Unavailable('A model is loading or a request is running.')
        try:
            if self.active: raise Unavailable('A request is running; model selection was not changed.')
            chosen_config = self.config(alias)
            if classic_beta.is_retired(chosen_config):
                raise Unavailable('This local model has been retired.')
            if classic_beta.is_beta(chosen_config) and not self.eligible(chosen_config):
                raise Unavailable('This profile needs its reviewed quality and current model identity proof.')
            if chosen_config.get('backend_kind') == oxe_runtime.KIND and not self.eligible(chosen_config):
                raise Unavailable(oxe_runtime.ineligible_reason(chosen_config) or 'The Rust engine is not enabled and ready.')
            if chosen_config.get('backend_kind') == bigmoe_runtime.KIND and not self.eligible(chosen_config):
                raise Unavailable('Flash profile has not passed native qualification and RAM admission.')
            chosen = chosen_config['model_alias']
            value = json.loads(ctl.CONFIG.read_text())
            if chosen != value.get('model_alias') and chosen not in value.get('profiles', {}):
                raise Unavailable('Selected model is not a registered local profile.')
            ctl.guard_production_path(ctl.CONFIG)
            if self.loaded_config and self.loaded_config['model_alias'] != chosen: self.stop_for_control()
            value['selected_model'] = chosen
            ctl.guard_production_path(ctl.CONFIG)
            lan_share.write_config(ctl.CONFIG, value)
            self.cache.pop('/props', None)
            return chosen
        finally: self.mu.release()

    def context_block(self, config, runtime=None):
        """`status.context` for one resolved profile (V013-PLAN sec. 3)."""
        runtime = self.runtime_value() if runtime is None else runtime
        alias = config.get('model_alias') if isinstance(config, dict) else None
        return profile_policy.context_status(config or {}, profile_policy.stored_mode(runtime, alias))

    def set_mode(self, alias, mode):
        """POST /_local/mode: persist `mode` for `alias` in runtime.json `context_modes.models`.
        Returns the model's new `context`. Unavailable (409) for a mode this model cannot run.
        fast/deep change only the OpenCode limits (applied by the next OpenCode session); a cap
        change (to or from max) unloads that model when it is loaded and idle, so its next load
        runs the new cap."""
        if not isinstance(alias, str) or not alias:
            raise Unavailable('Name the model.')
        if mode not in harness_registry.MODES:
            raise Unavailable('Mode must be fast, deep or max.')
        if not self.mu.acquire(timeout=2): raise Unavailable('A model is loading; change the mode when it is ready.')
        try:
            config = self.config(alias)
            rows = {row['id']: row for row in profile_policy.mode_rows(config)}
            if not rows:
                raise Unavailable('This model has no context modes.')
            if not rows[mode]['available']:
                raise Unavailable(rows[mode]['reason'] or 'This mode is unavailable.')
            loaded = self.loaded_config if self.process is not None else None
            cap_change = (loaded is not None and loaded.get('model_alias') == config['model_alias']
                          and loaded.get('context_size') != profile_policy.apply_mode(config, mode).get('context_size'))
            if cap_change and self.active:
                raise Unavailable('A request is running; the mode was not changed.')
            # v0.1.3 integration: the same writer as `share` (LAN M5): the file keeps its permission
            # bits and a symlinked runtime.json stays a symlink; P56 guard on both paths first
            saved = {}
            def change(value):
                saved['value'] = profile_policy.with_stored_mode(value, config['model_alias'], mode)
                value.clear(); value.update(saved['value'])
            try:
                lan_share.update_config(ctl.CONFIG, change, ctl.guard_production_path)
                value = saved['value']
            except (OSError, ValueError):
                raise Unavailable('The mode could not be saved.')
            if cap_change:
                self.stop_for_control()  # the next request loads the model at the new cap
            self.cache.pop('/props', None)
            return self.context_block(self.config(config['model_alias']), value)
        finally: self.mu.release()

    def harness_learning(self):
        """The opt-in as it is in runtime.json now (a loaded model keeps an older snapshot)."""
        try: return learning_settings(json.loads(ctl.CONFIG.read_text()))
        except (OSError, ValueError): return learning_settings(None)

    def harness_status(self, config):
        learning = self.harness_learning()
        with self.harness_mu:
            return {'harnesses': self.harness_registry.status(config), 'request_compat': dict(self.compat_stats),
                    'harness_learning': self.harness_registry.learning_status(learning)}

    def set_harness_learning(self, enabled, min_requests=None):
        """Persist the opt-in (runtime.json `harness_learning`); nothing else is changed."""
        if type(enabled) is not bool or (min_requests is not None and (type(min_requests) is not int or not 1 <= min_requests <= 100)):
            raise Unavailable('Send {"enabled": true or false} and optionally "min_requests" (1-100).')
        # runtime.json is rewritten under mu (like select); mu is held for a whole model load.
        if not self.mu.acquire(timeout=2): raise Unavailable('A model is loading; change the setting when it is ready.')
        try:
            try:
                value = json.loads(ctl.CONFIG.read_text())
                if not isinstance(value, dict): raise ValueError()
                current = value.get('harness_learning') if isinstance(value.get('harness_learning'), dict) else {}
                value['harness_learning'] = {**current, 'enabled': enabled,
                                             **({'min_requests': min_requests} if min_requests is not None else {})}
                ctl.guard_production_path(ctl.CONFIG)
                lan_share.write_config(ctl.CONFIG, value)
            except (OSError, ValueError):
                raise Unavailable('The harness learning setting could not be saved.')
            return learning_settings(value)
        finally: self.mu.release()

    def prepare_harness(self, harness, model_alias=None):
        """Build or load the stored prefixes of one harness (explicit user action: this
        loads the model). One 1-token primer per known prefix, through the normal queue."""
        if not isinstance(harness, str) or not harness:
            raise Unavailable('Name the harness to prepare.')
        target = self.config(model_alias)
        if target.get('backend_kind') != oxe_runtime.KIND:
            raise Unavailable('Harness prefixes are prepared on the Rust engine models only.')
        if harness == '*':
            # "Prepare all" (after an engine update): every harness whose newest prefix is
            # not expected warm for this model, one after the other through the queue.
            with self.harness_mu:
                names = self.harness_registry.harness_names(target, cold_only=True)
            done = []
            for name in names:
                try: done.append(self.prepare_harness(name, target['model_alias']))
                except Unavailable: pass  # no usable primer for this one
            return {'harness': '*', 'model': target['model_alias'], 'harnesses': done,
                    'prepared': [row for item in done for row in item['prepared']]}
        with self.harness_mu:
            primers = self.harness_registry.primers(harness, target['model_alias'])
        if not primers:
            raise Unavailable('No recorded prefix for this harness yet; run it once against the local model.')
        prepared = []
        for key, body in primers:
            self.begin_queued(inference=True, allow_start=True, model_alias=target['model_alias'])
            try:
                started = time.monotonic()
                reply = self.backend_json('/v1/chat/completions', oxe_runtime.request_defaults(body, '/v1/chat/completions', target))
                timings = reply.get('timings') if isinstance(reply.get('timings'), dict) else {}
                with self.harness_mu:
                    self.harness_registry.complete(key, self.loaded_config or target, timings.get('cache_n'), timings.get('prompt_n'))
                prepared.append({'key': key[:12], 'cache_n': timings.get('cache_n'), 'prompt_n': timings.get('prompt_n'),
                                 'seconds': round(time.monotonic() - started, 2)})
            finally:
                self.end(inference=True)
        return {'harness': harness, 'model': target['model_alias'], 'prepared': prepared}
