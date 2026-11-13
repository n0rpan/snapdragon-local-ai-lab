"""Per-engine launch adapters for Manager._begin (ctl_lifecycle).

_begin is the orchestrator: it runs the engine-independent admission checks and calls the steps
below in a fixed order. Each adapter answers only where the engines differ. One adapter is chosen
per launch from the profile (engine_for):
  LlamaLaunch     classic llama-server (backend_kind None or 'llama'), also the classic beta
  OxeMoeLaunch    the Rust engine (oxe) for a non-Flash profile
  OxeFlashLaunch  the Rust engine for a Flash profile (oxe_flash): Flash admission, scope, power guard
  BigMoeLaunch    the streamed Flash backend (bigmoe): its own port, runtime object and readiness

LaunchSpec is the resolved launch: what Popen gets (argv, environment, options) and the per-launch
access token the process carries.
"""
import http.client
import json
import os
if __package__:
    from .ctl_env import Unavailable
    from .ctl_repack import repack_cache_enabled, repack_cache_identity
    from .ctl_seams import ctl
    from . import bigmoe_runtime, classic_beta, memory_policy, oxe_runtime, state_cache
else:
    from ctl_env import Unavailable
    from ctl_repack import repack_cache_enabled, repack_cache_identity
    from ctl_seams import ctl
    import bigmoe_runtime
    import classic_beta
    import memory_policy
    import oxe_runtime
    import state_cache


class LaunchSpec:
    """The resolved launch of one engine start."""

    def __init__(self, args, environment):
        self.args = args
        self.environment = environment
        self.options = {}  # extra Popen keywords (pass_fds for the native ledger)
        self.token = None
        self.token_file = None
        self.repack_cache = False


def engine_for(config):
    """The adapter for this profile; called once per launch after the backend kind was checked."""
    if config.get('backend_kind') == bigmoe_runtime.KIND:
        return BigMoeLaunch()
    if config.get('backend_kind') == oxe_runtime.KIND:
        return OxeFlashLaunch() if oxe_runtime.is_flash(config) else OxeMoeLaunch()
    return LlamaLaunch()


def open_append_lease():
    ctl.guard_production_path(ctl.LEASE)
    ctl.LEASE.parent.mkdir(parents=True, exist_ok=True)
    ctl.guard_production_path(ctl.LEASE)
    # N6: never follow a symlink planted at the lease path (as at the other sites).
    return ctl.open_lease_append()


def health_ready(manager, config, backend_key):
    """One /health probe of a llama or oxe engine; True when it is ready."""
    connection = http.client.HTTPConnection('127.0.0.1', ctl.BACKEND_PORT, timeout=2)
    try:
        connection.request('GET', '/health', headers=oxe_runtime.backend_headers(backend_key))
        response = connection.getresponse()
        health_body = response.read()
        classic = config.get('backend_kind') != oxe_runtime.KIND
        if (response.status == 200 and classic
                and ctl.port_listener_in_group(ctl.BACKEND_PORT, manager.process.pid) is False):
            # S6: a 200 from a process outside our child's group is not our engine
            # (oxe is pinned by its per-launch bearer token instead).
            raise Unavailable(f'Local engine port {ctl.BACKEND_PORT} is answered by '
                              f'{ctl.listening_process(ctl.BACKEND_PORT)}, not the engine this '
                              'controller started. Close that app or set '
                              'LOCAL_LLM_BACKEND_PORT to an unused port.')
        if response.status == 200 and (classic or oxe_runtime.health_ready(health_body)):
            manager.classic_memory_guard.mark_ready()
            if manager.classic_memory_guard.failure():
                raise Unavailable('Local model stopped to preserve system memory.')
            return True
    except (OSError, http.client.HTTPException):
        pass
    finally:
        connection.close()
    return False


def capture_models(manager, backend_key):
    metadata_connection = http.client.HTTPConnection('127.0.0.1', ctl.BACKEND_PORT, timeout=3)
    try:
        metadata_connection.request('GET', '/v1/models', headers=oxe_runtime.backend_headers(backend_key))
        metadata_response = metadata_connection.getresponse()
        metadata = metadata_response.read()
        parsed = json.loads(metadata)
        if metadata_response.status == 200 and isinstance(parsed.get('data'), list):
            manager.cache['/v1/models'] = ('application/json', metadata)
    except (OSError, ValueError, http.client.HTTPException):
        pass
    finally:
        metadata_connection.close()


class EngineLaunch:
    """Defaults shared by the engines served on BACKEND_PORT and watched by the classic memory guard."""
    flash = False  # may carry a flash_memory policy (its own scope and host floor)
    runtime = None  # bigmoe only: the streamed backend's runtime object

    def backend_port(self):
        return ctl.BACKEND_PORT, 'LOCAL_LLM_BACKEND_PORT'

    def admit(self, config, model, available):
        """RAM admission, then the lease file (not yet locked); returns (config, lease)."""
        required = model.stat().st_size + 3 * 1024**3
        if available is not None and available < required:
            raise Unavailable(f'This model needs about {required / 1024**3:.1f} GiB of available RAM; '
                              f'{available / 1024**3:.1f} GiB is available. Close other large apps and retry.')
        return config, open_append_lease()

    def verify(self, config):
        if classic_beta.is_beta(config):
            classic_beta.verify_engine(config)

    def command(self, manager, config, model):
        return []

    def shape(self, manager, config, args, flash_memory, cache_config):
        if cache_config.get('enabled'):
            manager.prefix_store = ctl.PrefixStore(cache_config['directory'],
                cache_config.get('max_bytes', 3 * 1024**3))
            args += ['--slots', '--slot-save-path', str(manager.prefix_store.root)]
        return args

    def launch_spec(self, config, args):
        environment = os.environ.copy()
        spec = LaunchSpec(args, self.environment(config, environment))
        self.issue_token(spec)
        spec.repack_cache = self.use_repack_cache(config, spec.environment)
        return spec

    def environment(self, config, environment):
        environment.update(config.get('environment', {}))
        return environment

    def issue_token(self, spec):
        pass

    def use_repack_cache(self, config, environment):
        return False

    def guard_options(self, flash_memory):
        return {}

    def stderr_options(self, manager):
        return {}

    def start_guards(self, manager, flash_memory):
        manager.classic_memory_guard.start(manager.process)

    def start_seconds(self):
        return ctl.START_SECONDS

    def ready(self, manager, config, backend_key):
        return health_ready(manager, config, backend_key)

    def after_ready(self, manager, backend_key):
        capture_models(manager, backend_key)


class LlamaLaunch(EngineLaunch):
    def command(self, manager, config, model):
        return [str(config['server_binary']), '--model', str(model),
            '--alias', config.get('model_alias', 'qwen-local'),
            '--host', '127.0.0.1', '--port', str(ctl.BACKEND_PORT),
            '--ctx-size', str(int(config.get('context_size', 8192))),
            '--threads', str(int(config.get('threads', 10))),
            '--n-gpu-layers', str(int(config.get('gpu_layers', 0))),
            *config.get('extra_args', [])]

    def issue_token(self, spec):
        # HF-3: the classic llama-server fallback gets a per-launch key too
        # (--api-key-file); the proxy and the controller's own calls send it.
        try:
            spec.token, spec.token_file = oxe_runtime.new_api_key(ctl.BACKEND_PORT)
        except OSError:
            raise Unavailable('Local model cannot start: no private runtime directory for its access token.') from None
        spec.args += ['--api-key-file', str(spec.token_file)]

    def use_repack_cache(self, config, environment):
        repack_cache = repack_cache_enabled(environment)
        if repack_cache:
            identity = repack_cache_identity(config)
            if identity is None:
                # Never let the native cache fall back to its weaker
                # default identity: load without it, while still capturing engine stderr.
                ctl.disable_repack_cache(environment, config.get('model_alias', ''))
                repack_cache = False
            else:
                environment.update(identity)
        return repack_cache


class OxeMoeLaunch(EngineLaunch):
    def verify(self, config):
        # An alias switched to Rust is pinned by its Rust engine
        # hash, never by the classic Ornith engine allowlist.
        try:
            oxe_runtime.verify_engine(config)
        except FileNotFoundError:
            # M4: a missing Rust binary fails loudly and names the fallback switch.
            raise Unavailable(oxe_runtime.ineligible_reason(config)
                              or 'The Rust engine binary is missing.') from None

    def command(self, manager, config, model):
        try:
            args = oxe_runtime.command(config, ctl.BACKEND_PORT)
        except ValueError as error:
            if str(error) != 'state_cache_root_override_invalid': raise
            raise Unavailable(f'{state_cache.ROOT_ENV} (or runtime.json cache.root) must be an absolute '
                              'path inside your home; the model was not started.') from None
        manager.state_engine = '--state-cache' in args
        manager.state_preload = (manager.state_engine and
                                 oxe_runtime.engine_has_state_preload(config.get('server_binary')))
        return args

    def environment(self, config, environment):
        environment = oxe_runtime.launch_environment(environment)
        environment.update(config.get('environment', {}))
        return environment

    def issue_token(self, spec):
        # B1: per-launch bearer token; set after the profile environment so no
        # profile can replace or drop it. The proxy and status calls send it.
        try:
            spec.token, spec.token_file = oxe_runtime.new_api_key(ctl.BACKEND_PORT)
        except OSError:
            raise Unavailable('Local model cannot start: no private runtime directory for its access token.') from None
        spec.environment[oxe_runtime.API_KEY_ENV] = str(spec.token_file)


class OxeFlashLaunch(OxeMoeLaunch):
    flash = True

    def admit(self, config, model, available):
        flash_swap_guard = None
        if not oxe_runtime.flash_files_ready(config):
            raise Unavailable('Flash model shards or the expert sidecar are missing or changed.')
        if oxe_runtime.flash_power_failure(bigmoe_runtime.power_snapshot(), starting=True):
            raise Unavailable('Flash needs at least 30% battery to start and stops at 20%.')
        try:
            # A smaller-cache fallback, when configured, replaces the launch snapshot.
            # With a flash_memory policy the engine's limit is its own scope's.
            scoped = isinstance(config.get('flash_memory'), dict)
            pref_enabled = oxe_runtime.flash_cache_pref.enabled(os.environ, config['oxe_flash'])
            mem_total = ctl.total_memory_bytes() if pref_enabled else None
            # S2: the MemTotal rule for 20/18 also needs swap for what must leave RAM.
            swap_free = ctl.swap_bytes()[1] if pref_enabled else None
            config = oxe_runtime.flash_select(
                config, available, oxe_runtime.flash_limits(config)[1] if scoped else ctl.cgroup_memory_max(),
                env=os.environ, mem_total=mem_total, swap_free=swap_free)
            # F127: opt-in swap-tolerant guard, only for a preferred 20/18 on the MemTotal rule.
            flash_swap_guard = oxe_runtime.flash_swap_tolerant_guard(
                config, available, mem_total, env=os.environ, swap_free=swap_free)
        except ValueError:
            if oxe_runtime.flash_cache_pref.enabled(os.environ, config['oxe_flash']):
                raise Unavailable('Flash cache preference cannot fit the current native memory budget; '
                                  'the host floor and runtime pressure guards remain active.') from None
            raise Unavailable(f'Flash (Rust) needs about {oxe_runtime.flash_smallest_required_available(config) / 1024**3:.1f} '
                              'GiB of available RAM; close other large apps and retry.') from None
        return config, open_append_lease()

    def shape(self, manager, config, args, flash_memory, cache_config):
        if flash_memory is not None:
            args = memory_policy.scope_command(args, flash_memory['scope_max_bytes'])
            return args
        return super().shape(manager, config, args, flash_memory, cache_config)

    def guard_options(self, flash_memory):
        return ({'minimum_available_bytes': flash_memory['host_floor_bytes'],
                 'follow_process_cgroup': True}
                if flash_memory is not None else {})

    def stderr_options(self, manager):
        # G-F127: only a Flash oxe launch gets the start-up line hook
        return {'on_line': manager._swap_guard_hook(manager.classic_memory_guard)}

    def start_guards(self, manager, flash_memory):
        manager.classic_memory_guard.start(manager.process)
        manager.flash_power_guard = oxe_runtime.FlashPowerGuard(manager.process)
        manager.flash_power_guard.start()

    def start_seconds(self):
        # The Rust Flash server prefaults a 20 GiB cache before it reports ready.
        return max(ctl.START_SECONDS, 300.0)

    def after_ready(self, manager, backend_key):
        # Capture actual native selection once after readiness, under the existing lease.
        # Status/catalog polling only reads this snapshot and never calls the backend.
        props_connection = http.client.HTTPConnection('127.0.0.1', ctl.BACKEND_PORT, timeout=3)
        try:
            props_connection.request('GET', '/props', headers=oxe_runtime.backend_headers(backend_key))
            props_response = props_connection.getresponse()
            props_body = props_response.read(65537)
            if props_response.status == 200 and len(props_body) <= 65536:
                manager.flash_cache_status = oxe_runtime.flash_cache_pref.status_from_props(json.loads(props_body))
        except (OSError, ValueError, http.client.HTTPException):
            pass
        finally:
            props_connection.close()
        # F127: keep swap-tolerant mode only when the engine itself reports it; else the 1 GiB floor.
        if (manager.classic_memory_guard is not None
                and manager.classic_memory_guard.mode() == memory_policy.SWAP_TOLERANT_MODE
                and (manager.flash_cache_status or {}).get('reason') != 'preferred-swap-tolerant'):
            manager.classic_memory_guard.leave_swap_tolerant()
        super().after_ready(manager, backend_key)


class BigMoeLaunch(EngineLaunch):
    """The streamed Flash backend: no classic guard, no token; _begin returns right after readiness."""
    flash = True

    def backend_port(self):
        return ctl.FLASH_BACKEND_PORT, 'LOCAL_FLASH_BACKEND_PORT'

    def admit(self, config, model, available):
        try:
            self.runtime = bigmoe_runtime.Runtime(config, ctl.FLASH_BACKEND_PORT)
        except (ValueError, OSError, KeyError, TypeError):
            raise Unavailable('Flash profile is unqualified or has insufficient admitted RAM.') from None
        # Do not manufacture a second lock for a streamed backend.
        ctl.guard_production_path(ctl.LEASE)
        lease = ctl.open_lease_rw()
        return config, lease

    def shape(self, manager, config, args, flash_memory, cache_config):
        runtime = self.runtime
        runtime.require_memory(ctl.available_memory_bytes(), lease_owned=manager.lease is not None)
        args = runtime.prepare(lease_owned=manager.lease is not None)
        if flash_memory is not None:
            args = memory_policy.scope_command(args, flash_memory['scope_max_bytes'])
        return args

    def environment(self, config, environment):
        return self.runtime.environment(environment)

    def start_guards(self, manager, flash_memory):
        runtime = self.runtime
        if flash_memory is not None:
            runtime.start_guard(manager.process, ctl.available_memory_bytes, lease_owned=manager.lease is not None,
                                reserve_bytes=flash_memory['host_floor_bytes'])
        else:
            runtime.start_guard(manager.process, ctl.available_memory_bytes, lease_owned=manager.lease is not None)

    def ready(self, manager, config, backend_key):
        if self.runtime.readiness(lease_owned=manager.lease is not None).ready:
            self.runtime.verify_child(manager.process)
            return True
        return False

    def after_ready(self, manager, backend_key):
        pass  # never reached: _begin returns for a streamed backend right after readiness
