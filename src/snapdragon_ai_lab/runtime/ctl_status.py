"""Status derivation, in one place: what `/_local/status` says about the controller and the
loaded engine. Reads state; never loads, stops or contacts a model.
"""
import json
from pathlib import Path
import time
if __package__:
    from .ctl_env import CAPABILITIES, Unavailable
    from .ctl_seams import ctl
    from . import lan_share, oxe_runtime, profile_policy
else:
    from ctl_env import CAPABILITIES, Unavailable
    from ctl_seams import ctl
    import lan_share
    import oxe_runtime
    import profile_policy


def engine_summary(config):
    """M4: the engine a resolved profile will use, and why it is not the Rust one when Rust is
    configured. Metadata only (one stat of the Rust binary)."""
    kind = config.get('backend_kind') or 'llama'
    engine = {'oxe': 'rust', 'llama': 'llama.cpp', 'bigmoe-cpu-stream': 'bigmoe'}.get(kind, 'unknown')
    note = None
    if engine == 'rust':
        note = oxe_runtime.ineligible_reason(config)
    else:
        source, rust = profile_policy.rust_overrides(config)
        if rust:
            flash = kind == profile_policy.FLASH_KIND
            switch = profile_policy.OXE_FLASH_ENABLE_ENV if flash else profile_policy.OXE_ENABLE_ENV
            if not profile_policy.switch_on(switch):
                note = f'classic fallback: {switch}=0'
            elif source == 'memlat_overrides':
                note = 'classic fallback: Rust is configured under memlat_overrides and MEMLAT is not active on this boot'
            else:
                note = 'classic fallback: Rust configuration did not apply'
    return {'engine': engine, 'engine_note': note}


def idle_seconds(config):
    config = config or {}
    if config.get('backend_kind') == oxe_runtime.KIND and oxe_runtime.is_flash(config):
        return config['oxe_flash'].get('idle_unload_seconds', ctl.IDLE_SECONDS)
    return ctl.IDLE_SECONDS


class StatusMixin:
    """Manager: the status reply and its parts."""

    def active_block(self, loading=False):
        """What is happening now, for a progress line. No lock: plain attribute reads.

        phase: idle, queued (requests wait, none admitted), loading, answering. Token
        progress needs the engine to report it and is not available yet (None)."""
        since = getattr(self, 'answering_since', None)
        queued = self.queue_length()
        if loading or self.state == 'loading': phase = 'loading'
        elif self.inferences and since is not None: phase = 'answering'
        elif queued: phase = 'queued'
        else: phase = 'idle'
        loaded = self.loaded_config
        return {'phase': phase, 'model': loaded.get('model_alias') if loaded else None,
                'seconds': round(time.monotonic() - since, 1) if phase == 'answering' else None,
                'queued_requests': queued, 'prompt_tokens_done': None, 'prompt_tokens_total': None,
                'tokens_out': None, 'tokens_per_second': None}

    def preload_status(self):
        """`status.preload` (V013-PLAN sec. 3): {state, src, ms, age_s}."""
        if not (self.process is not None and self.state_engine):
            return {'state': 'unavailable' if self.process is not None else 'idle', 'src': None, 'ms': None, 'age_s': None}
        if not self.state_preload:
            return {'state': 'unavailable', 'src': None, 'ms': None, 'age_s': None}
        info = self.preload_info
        if info is None:
            return {'state': 'idle', 'src': None, 'ms': None, 'age_s': None}
        return {'state': info['state'], 'src': info['src'], 'ms': info['ms'],
                'age_s': int(max(0.0, time.monotonic() - info['at']))}

    def unload_info(self):
        """What an 'Unload now' control needs; call with self.mu held. No model access."""
        loaded = self.process is not None and self.loaded_config is not None
        remaining = None
        if loaded and self.state == 'ready' and self.active == 0:
            remaining = max(0.0, idle_seconds(self.loaded_config) - (time.monotonic() - self.last_used))
        return {'loaded_model': self.loaded_config.get('model_alias') if loaded else None,
                'resident_bytes': ctl.resident_bytes(self.process.pid) if loaded else None,
                'idle_unload_remaining_seconds': None if remaining is None else round(remaining, 1),
                'unload': {'method': 'POST', 'path': '/_local/unload', 'available': loaded and self.active == 0}}

    def status(self):
        # Avoid blocking status requests while a large model is being loaded.
        if not self.mu.acquire(timeout=.05):
            if self.stop_pending is not None:  # a stop retry holds the lock; say so, promptly
                return {'state': 'stopping', 'stop_pending': self.stop_pending, 'active_requests': 0,
                        'capabilities': list(CAPABILITIES), 'active': self.active_block(loading=True)}
            return {'state': 'loading', 'active_requests': 0, 'capabilities': list(CAPABILITIES),
                    'active': self.active_block(loading=True)}
        try:
            selection_error = None
            try:
                selected_config = self.config()
            except Unavailable:
                selected_config = {}
                selection_error = 'invalid_or_unavailable_selection'
            config = self.loaded_config or selected_config
            setup = {}
            try:
                setup = json.loads((ctl.CONFIG.parent / 'download-status.json').read_text())
            except (OSError, ValueError):
                pass
            expected = config.get('expected_bytes', setup.get('expected_bytes'))
            downloaded = 0
            try:
                model = Path(config['model_path'])
                partial = model.with_suffix(model.suffix + '.part')
                downloaded = model.stat().st_size if model.is_file() else partial.stat().st_size if partial.is_file() else 0
                if expected is None:
                    expected = json.loads((ctl.CONFIG.parent / 'model.json').read_text()).get('expected_bytes')
            except (OSError, ValueError, KeyError):
                pass
            return {**({'flash_cache': dict(self.flash_cache_status)} if self.flash_cache_status else {}),
                    **({'stop_pending': self.stop_pending} if self.stop_pending is not None else {}),
                    'state': self.state, 'validated': self.eligible(config),
                    'download_state': setup.get('state', 'preparing'),
                    'downloaded_bytes': downloaded, 'expected_bytes': expected,
                    'model': config.get('model_alias', 'qwen-local'),
                    'selected_model': selected_config.get('model_alias'),
                    'selection_error': selection_error,
                    'backend': config.get('backend', 'cpu'),
                    'context_size': config.get('context_size', 8192),
                    'threads': config.get('threads'),
                    'execution_policy_mode': config.get('execution_policy_mode', 'stock'),
                    'memory_policy': self.classic_memory_guard.metadata() if self.classic_memory_guard is not None else self.last_memory_guard,
                    'validation': config.get('validation', {}),
                    'flash_memory': config.get('flash_memory'),
                    'gpu_layers': config.get('gpu_layers', 0),
                    'active_requests': self.inferences, 'queued_requests': self.queue_length(), 'idle_unload_seconds': idle_seconds(config),
                    'image_active': self.image_active, 'image_broker_error': self.image_broker_error,
                    **self.unload_info(),
                    'prefix_cache': {'enabled': self.backend_runtime is None and config.get('prefix_cache', {}).get('enabled') is True, **self.prefix_stats},
                    'backend_kind': config.get('backend_kind', 'llama'),
                    **engine_summary(config),
                    'flash_qualification': self.flash_qualification(),
                    'engine_ledger': self.engine_ledger.metadata() if self.engine_ledger is not None else {'enabled': False},
                    **self.harness_status(config),
                    'flash_idle_qos': {'enabled': isinstance(config.get('oxe_flash'), dict) and 'idle_qos' in config['oxe_flash'],
                                       'unit': self.flash_idle_qos_unit, 'started': self.flash_idle_qos_started,
                                       'error': self.flash_idle_qos_error},
                    'capabilities': list(CAPABILITIES),
                    'active': self.active_block(),
                    'last_error': self.last_failure,
                    'last_failure': dict(self.last_failure_info) if self.last_failure_info else None,
                    # V013: when and for whom the last failure happened (origin: organic or lab), so
                    # the panel shows only its own actions' failures without guessing from the ledger
                    'error_at': (self.last_failure_info or {}).get('t'),
                    'error_origin': (self.last_failure_info or {}).get('origin'),
                    'context': self.context_block(config),
                    'preload': self.preload_status(),
                    'stop_pending': self.stop_pending,
                    'image_slot_stale': len(getattr(self.image_broker, 'stale', None) or ()),
                    'lan_share': (self.share.status() if getattr(self, 'share', None) is not None
                                  else lan_share.off_status(ctl.PROXY_PORT)),
                    'chat_url': 'http://127.0.0.1:' + str(ctl.PROXY_PORT),
                    'api_base': 'http://127.0.0.1:' + str(ctl.PROXY_PORT) + '/v1'}
        finally:
            self.mu.release()
