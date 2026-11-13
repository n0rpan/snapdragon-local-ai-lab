"""The engine process: start (admission checks, memory and power guards, launch, readiness; the
steps where the engines differ are in ctl_engines), stop (every step runs, the lease is released
only after the exit is proven), the reaper (idle unload, guard stops, stop retries) and the
record of why the last start failed.
"""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
if __package__:
    from .ctl_engines import engine_for
    from .ctl_env import LAB_ORIGIN, ModelBusy, StopPending, Unavailable, unexpected_cause
    from .ctl_seams import ctl
    from .ctl_status import idle_seconds
    from .engine_diagnostics import swap_guard_mode
    from . import bigmoe_runtime, classic_beta, engine_ledger, explicit_route, memory_policy, oxe_runtime
else:
    from ctl_engines import engine_for
    from ctl_env import LAB_ORIGIN, ModelBusy, StopPending, Unavailable, unexpected_cause
    from ctl_seams import ctl
    from ctl_status import idle_seconds
    from engine_diagnostics import swap_guard_mode
    import bigmoe_runtime
    import classic_beta
    import engine_ledger
    import explicit_route
    import memory_policy
    import oxe_runtime


class LifecycleMixin:
    """Manager: begin, cleanup and the reaper."""

    def _engine_writer(self):
        """Create the fixed-path sink only when an engine is actually starting."""
        if self.engine_log_writer is None:
            self.engine_log_writer = ctl.EngineLogWriter(ctl.ENGINE_DIAGNOSTIC_LOG)
        return self.engine_log_writer

    def record_engine_death(self, process, config=None, guard_failure=None):
        """Record a native child exit once, without including request data."""
        if process is None:
            return False
        with self.engine_death_lock:
            exit_code = process.poll()
            if exit_code is None or vars(process).get('_local_ai_engine_death_logged', False):
                return False
            try:
                process._local_ai_engine_death_logged = True
            except (AttributeError, TypeError):
                pass
            if guard_failure is None and process is self.process:
                if self.backend_runtime is not None:
                    guard_failure = self.backend_runtime.guard_failure()
                elif self.classic_memory_guard is not None:
                    guard_failure = self.classic_memory_guard.failure()
            config = config or (self.loaded_config if process is self.process else None) or {}
            event = {
                'event': 'engine_died',
                'pid': getattr(process, 'pid', None),
                'exit_code': exit_code,
                'model_alias': config.get('model_alias'),
                'engine_sha256': config.get('server_sha256'),
            }
            if guard_failure:
                event['guard_failure'] = str(guard_failure)
            writer = self.engine_log_writer
            if writer is not None:
                writer.write(json.dumps(event, sort_keys=True) + '\n')
            return True

    def _flash_idle_qos(self, verb, config, pid):
        """The only place that runs systemctl for the Flash idle-QoS hook; returns (unit, started, error).

        Best effort and never raises: a denied, missing or slow unit is reported in /_local/status and
        never fails or blocks a model load or unload. unit is None when the profile has no idle_qos.
        """
        unit = None
        try:
            unit = oxe_runtime.flash_idle_qos_unit(config, pid)
            command = oxe_runtime.flash_idle_qos_command(config, pid, verb)
            if command is None:
                return None, False, None
            done = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                  stderr=subprocess.DEVNULL, timeout=5, check=False)
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            return unit, False, type(error).__name__
        if done.returncode != 0:
            return unit, False, 'CalledProcessError'
        return unit, True, None

    def stop_for_control(self):
        """cleanup() for a control action (unload, select, mode, Flash cache), caller holds `mu`.

        Audit S5: cleanup() re-raises its first teardown error after releasing what it can, and
        keeps the lease while the engine's exit is unproven. A control route must answer that
        with a 503 naming the condition, not drop the connection."""
        try:
            self.cleanup()
        except Exception as error:
            traceback.print_exc(file=sys.stderr)
            if self.stop_pending is None:
                raise StopPending('The local engine did not stop cleanly '
                                  f'({unexpected_cause(error)}). Retry shortly.') from error
        if self.stop_pending is not None:
            raise StopPending('The previous local engine has not exited yet; its memory and the '
                              'inference lease stay reserved. Retry shortly.')

    def cleanup(self):
        """Stop the loaded engine and release everything it held.

        S1 (Codex review): every teardown step runs even when an earlier one raises, and the
        outermost finally always makes sure the child's process group is dead and then releases
        the lease and the loaded state. The first error is re-raised after that.
        """
        errors = []
        state = {'guard_failure': None, 'process': self.process}

        def step(function):
            try:
                function()
            except Exception as error:  # keep going; re-raised after the lease is released
                errors.append(error)

        try:
            runtime = self.backend_runtime

            def read_guard_failure():
                guard_failure = runtime.guard_failure() if runtime is not None else None
                if self.classic_memory_guard is not None:
                    guard_failure = self.classic_memory_guard.failure() or guard_failure
                state['guard_failure'] = guard_failure
            step(read_guard_failure)
            guard_failure = state['guard_failure']
            process, self.process = self.process, None

            def stop_process():
                died_before_cleanup = process.poll() is not None
                if not died_before_cleanup:
                    retry = self.stop_pending is not None
                    if not retry:  # first attempt: the engine gets its SIGTERM grace
                        try:
                            os.killpg(process.pid, signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                    try:
                        if retry:
                            raise subprocess.TimeoutExpired('engine', 0)
                        process.wait(timeout=ctl.TERM_WAIT_SECONDS)
                    except subprocess.TimeoutExpired:
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        # Bounded (Codex review 3): an engine stuck in D-state may not die at once;
                        # a timeout leaves it 'stopping' and the reaper retries.
                        process.wait(timeout=ctl.KILL_WAIT_SECONDS)
                else:
                    process.wait()
                state['process'] = None  # reaped; the finally below has nothing left to kill
                if died_before_cleanup or guard_failure:
                    self.record_engine_death(process, self.loaded_config, guard_failure)
            if process is not None:
                step(stop_process)
                step(lambda: oxe_runtime.remove_api_key(getattr(process, '_local_ai_api_key_file', None)))

            def stop_idle_qos():
                if self.flash_idle_qos_pid is not None:
                    pid, self.flash_idle_qos_pid = self.flash_idle_qos_pid, None
                    _, _, self.flash_idle_qos_error = self._flash_idle_qos('stop', self.loaded_config or {}, pid)
                    self.flash_idle_qos_unit, self.flash_idle_qos_started = None, False
            step(stop_idle_qos)
            reader, self.repack_reader = self.repack_reader, None
            if reader is not None:
                # The child (and its process group) is gone, so stderr reaches EOF.
                step(lambda: reader.join(timeout=5))
            engine_reader, self.engine_reader = self.engine_reader, None
            if engine_reader is not None:
                step(lambda: engine_reader.stop(timeout=5))
            ledger, self.engine_ledger = self.engine_ledger, None
            if ledger is not None:
                step(lambda: ledger.stop(timeout=5))
            alias = (self.loaded_config or {}).get('model_alias')
            reasons = []
            if runtime is not None:
                def stop_runtime_guard():
                    try:
                        reasons.append(runtime.guard_failure() or guard_failure)
                    finally:
                        runtime.stop_guard()
                step(stop_runtime_guard)
            guard, self.classic_memory_guard = self.classic_memory_guard, None
            if guard is not None:
                def stop_classic_guard():
                    guard.stop()
                    reasons.append(guard.failure() or guard_failure)
                    self.last_memory_guard = guard.metadata()
                step(stop_classic_guard)
            flash_guard, self.flash_power_guard = self.flash_power_guard, None
            if flash_guard is not None:
                def stop_flash_guard():
                    flash_guard.stop()
                    reasons.append(flash_guard.failure)
                step(stop_flash_guard)
            stop_reason = next((reason for reason in reversed(reasons) if reason), None)
            if stop_reason:
                step(lambda: self.record_failure(stop_reason, kind='guard', alias=alias))
        finally:
            # The engine must be gone before the lease is released, or another job could start
            # beside it. Codex review (round 2): a failed signal or a wait timeout is not proof of
            # exit; then the controller keeps the process, the lease and the loaded config, latches
            # state 'stopping' and retries from the reaper and from every new begin().
            leftover = state['process']
            try:
                if leftover is not None and leftover.poll() is None:
                    try:
                        os.killpg(leftover.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    leftover.wait(timeout=ctl.KILL_WAIT_SECONDS)
            except Exception as error:
                errors.append(error)
            unconfirmed = False
            try:
                unconfirmed = leftover is not None and leftover.poll() is None
            except Exception as error:  # cannot even ask: not proof of exit either
                errors.append(error)
                unconfirmed = True
            if unconfirmed:
                self.process = leftover
                self.state = 'stopping'
                self.stop_pending = (f'engine pid {getattr(leftover, "pid", "?")} has not exited: '
                                     + (unexpected_cause(errors[-1]) if errors else 'still running'))
                print(f'local-ai: {self.stop_pending}; lease kept, retrying', file=sys.stderr, flush=True)
            else:
                self.stop_pending = None
                self.process = None
                if self.lease is not None:
                    try:
                        self.lease.close()
                    except Exception as error:
                        errors.append(error)
                    finally:
                        self.lease = None
                self.state = 'unloaded'
                self.flash_cache_status = None
                self.flash_swap_guard_reported = None
                self.loaded_config = None
                self.state_engine = False
                self.state_preload = False
                self.preload_info = None
                self.current_prefix_key = None
                self.prefix_store = None
                self.backend_runtime = None
        if errors:
            raise errors[0]

    def _swap_guard_hook(self, guard):
        """G-F127 (v0.1.3): follow the Flash engine's own swap-guard decision from its first start-up
        line, before /props exists: anything but swap-tolerant leaves swap-tolerant mode at once
        (one-way, as the /props check after readiness, which stays for engines without the token).
        Bound to this launch's guard object, so a later launch's guard is never touched."""
        def hook(line):
            mode = swap_guard_mode(line)
            if mode is None:
                return
            self.flash_swap_guard_reported = mode
            if (guard is not None and mode != memory_policy.SWAP_TOLERANT_MODE
                    and guard.mode() == memory_policy.SWAP_TOLERANT_MODE):
                guard.leave_swap_tolerant()
        return hook

    def record_failure(self, reason, *, kind, alias=None, origin=None, detail=None):
        """M1: remember why the last start failed or a guard stopped a model; call with self.mu held."""
        if not reason:
            return
        info = {'reason': str(reason), 'kind': kind, 'alias': alias,
                'origin': origin or self.activity_origin, 't': int(time.time())}
        if detail and str(detail) != info['reason']:
            info['detail'] = str(detail)
        self.last_failure_info = info

    def clear_failure(self):
        self.last_failure_info = None

    @property
    def last_failure(self):
        """The user-facing error string: None for lab-origin failures (REAL-1 header)."""
        info = self.last_failure_info
        if not info or info.get('origin') == LAB_ORIGIN:
            return None
        return info['reason']

    def reaper(self):
        while not self.stop_event.wait(1):
            try:
                self.reap_once()
            except Exception as error:  # S1: one bad pass must not end idle unload for good
                try:
                    print(f'local-ai: reaper pass failed: {type(error).__name__}', file=sys.stderr, flush=True)
                except Exception:
                    pass

    def reap_once(self):
        with self.mu:
            if self.stop_pending is not None:
                self.cleanup()  # S1: retry an unproven stop; the lease stays held until it succeeds
            elif self.process is not None:
                if ((self.classic_memory_guard is not None and self.classic_memory_guard.failure())
                        or (self.flash_power_guard is not None and self.flash_power_guard.failure)
                        or self.process.poll() is not None):
                    self.cleanup()
                elif self.active == 0 and time.monotonic() - self.last_used >= idle_seconds(self.loaded_config):
                    self.cleanup()

    def begin(self, *, inference=False, allow_start=False, model_alias=None, origin=None, explicit_load=False, remote=False):
        """Admit a request, loading the model when needed (see _begin).

        M1: a new explicit load clears the previous failure; a successful start clears it
        too. Only a failure after a start was really attempted is recorded, with its origin.
        """
        with self.mu:
            origin = origin or 'organic'
            if explicit_load:
                self.clear_failure()
            previous = self.last_failure_info
            self._start_attempt = False
            try:
                process = self._begin(inference=inference, allow_start=allow_start, model_alias=model_alias,
                                      origin=origin, remote=remote)
            except Exception as error:
                if not isinstance(error, Unavailable):
                    # S14: an unexpected error keeps its cause in the journal and in last_failure.
                    traceback.print_exc(file=sys.stderr)
                if self._start_attempt:
                    guard = self.last_failure_info if self.last_failure_info is not previous else None
                    reason = str(error) if isinstance(error, Unavailable) else 'Local model could not be started.'
                    detail = (guard or {}).get('reason')
                    if not isinstance(error, Unavailable):
                        detail = unexpected_cause(error)
                    self.record_failure(reason, kind='start', alias=model_alias, origin=origin, detail=detail)
                raise
            finally:
                self._start_attempt = False
            self.activity_origin = origin
            return process

    def _begin(self, *, inference=False, allow_start=False, model_alias=None, origin='organic', remote=False):
        with self.mu:
            if self.image_active:
                raise Unavailable('An image is running; the resident chat model is retained.')
            if self.stop_pending is not None:
                try:
                    self.cleanup()
                except Exception:
                    pass
                if self.stop_pending is not None:
                    raise Unavailable('The previous local engine has not exited yet; its memory and the '
                                      'inference lease stay reserved. Retry shortly.')
            if inference and self.inferences:
                raise Unavailable('A language-model request is already running; retry when it finishes.')
            if (self.classic_memory_guard is not None and self.classic_memory_guard.failure()
                    or self.process is not None and self.process.poll() is not None):
                self.cleanup()
            # O-102 M3/M4, atomic with admission (under mu): a shared request uses the model that
            # is loaded now and never loads, switches or unloads one
            if remote and inference and (self.process is None or not self.loaded_config):
                raise ModelBusy('No model is loaded on the host; a shared request cannot load one. '
                                'Load it on the host first.')
            config = self.config(model_alias) if model_alias is not None or self.process is None else self.loaded_config
            if remote and inference:
                if config.get('model_alias') != self.loaded_config.get('model_alias'):
                    raise ModelBusy('Another model is loaded on the host; a shared request cannot switch '
                                    'models. Use "%s".' % self.loaded_config.get('model_alias'))
                config = self.loaded_config
            transition = None
            if (os.environ.get(explicit_route.ENABLE_ENV) == '1'
                    and (self.process is None or config['model_alias'] != self.loaded_config['model_alias'])):
                transition = (self.loaded_config['model_alias'] if self.loaded_config else None,
                              config['model_alias'], time.monotonic())
            if classic_beta.is_retired(config):
                raise Unavailable('This local model has been retired.')
            # Retain the child's execution snapshot, but an explicit request for
            # a proof-gated profile must still respect fresh proof revocation.
            if model_alias is not None and classic_beta.is_beta(config) and not self.eligible(config):
                raise Unavailable('Model proof or identity changed; requalification is required.')
            if self.process is not None and config['model_alias'] == self.loaded_config['model_alias']:
                config = self.loaded_config
            if classic_beta.is_beta(config) and not self.eligible(config):
                raise Unavailable('Model proof or identity changed; requalification is required.')
            if self.process is not None and config['model_alias'] != self.loaded_config['model_alias']:
                if self.active: raise Unavailable('Another model request is running; retry when it finishes.')
                self.cleanup()
            if self.process is None:
                if config.get('backend_kind') not in (None, 'llama', bigmoe_runtime.KIND, oxe_runtime.KIND):
                    raise Unavailable('Unknown local backend kind.')
                if config.get('backend_kind') == oxe_runtime.KIND and not self.eligible(config):
                    raise Unavailable(oxe_runtime.ineligible_reason(config) or 'The Rust engine is not enabled and ready.')
                if not allow_start:
                    raise Unavailable('Model is sleeping; submitting a message will load it.')
                self._start_attempt = True
                self.activity_origin = origin  # a guard stop during loading belongs to this request
                if not config.get('ready'):
                    raise Unavailable('Qwen setup and validation are still running. Check /_local/status.')
                # Check before model files, RAM admission, scopes or token files.
                # A loaded engine is reused above and never probes its own port.
                engine = engine_for(config)
                ctl.require_backend_port(*engine.backend_port())
                model = Path(config['model_path'])
                if not model.is_file():
                    raise Unavailable('The verified local model is not available yet.')
                if config.get('expected_bytes') is not None and model.stat().st_size != config['expected_bytes']:
                    raise Unavailable('The local model size does not match its verified profile.')
                available = ctl.available_memory_bytes()
                config, lease = engine.admit(config, model, available)
                runtime = engine.runtime
                try:
                    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    lease.close()
                    raise Unavailable('Another image, chat or setup job is running. Please try again when it finishes.')
                self.lease = lease
                self.state = 'loading'
                self.loaded_config = config
                self.backend_runtime = runtime
                self.cache.pop('/props', None)
                try:
                    engine.verify(config)
                    args = engine.command(self, config, model)
                    cache_config = config.get('prefix_cache', {})
                    # Flash only: its own memory scope and host floor. Every other model stays
                    # inside this unit's limit with the default floor.
                    try:
                        flash_memory = memory_policy.flash_policy(config)
                    except ValueError:
                        raise Unavailable('The Flash memory policy in this profile is not a qualified one.') from None
                    if flash_memory is not None and not engine.flash:
                        raise Unavailable('A Flash memory policy is set on a model that is not Flash.')
                    args = engine.shape(self, config, args, flash_memory, cache_config)
                    spec = engine.launch_spec(config, args)
                    backend_key, backend_key_file = spec.token, spec.token_file
                    if runtime is None:
                        try:
                            scoped_flash = engine.guard_options(flash_memory)
                            # O-39 replaces the old host floor/F127 stop mode;
                            # admission and the engine's prefill settings are unchanged.
                            self.classic_memory_guard = memory_policy.MemoryGuard(
                                loading_timeout_seconds=min(300.0, ctl.START_SECONDS),
                                host_guard_version=memory_policy.GUARD_2_VERSION, **scoped_flash)
                        except memory_policy.MemoryPolicyError as error:
                            oxe_runtime.remove_api_key(backend_key_file)  # never leave a token behind
                            self.record_failure(error, kind='start', alias=config.get('model_alias'), origin=origin)
                            self.last_memory_guard = {'reason': str(error)}
                            raise Unavailable('Local model cannot load under the current memory conditions.') from None
                    engine_writer = self._engine_writer()
                    if engine_ledger.enabled(config, spec.args):
                        try:
                            self.engine_ledger = engine_ledger.NativeLedger(ctl.ENGINE_LEDGER_LOG, config)
                        except (OSError, ValueError):
                            engine_writer.write('{"event":"ledger_disabled","reason":"log_unavailable"}\n')
                        else:
                            spec.args += self.engine_ledger.arguments
                            spec.options['pass_fds'] = self.engine_ledger.pass_fds
                    try:
                        self.process = subprocess.Popen(spec.args, stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE, start_new_session=True, env=spec.environment, umask=0o077,
                            **spec.options)
                    except BaseException:
                        oxe_runtime.remove_api_key(backend_key_file)
                        raise
                    self.process._local_ai_engine_config = config
                    self.process._local_ai_api_key = backend_key
                    self.process._local_ai_api_key_file = backend_key_file
                    if self.engine_ledger is not None:
                        self.engine_ledger.start()
                    if spec.repack_cache:
                        self.repack_reader = ctl.RepackErrorReader(self.process.stderr,
                            label=config.get('model_alias', ''), tee_writer=engine_writer).start()
                    else:
                        hook = engine.stderr_options(self)
                        self.engine_reader = ctl.EngineStderrReader(self.process.stderr, engine_writer, **hook)
                        self.engine_reader.start()
                    engine.start_guards(self, flash_memory)
                    deadline = time.monotonic() + engine.start_seconds()
                    while time.monotonic() < deadline:
                        if self.classic_memory_guard is not None and self.classic_memory_guard.failure():
                            raise Unavailable('Local model stopped to preserve system memory.')
                        if self.flash_power_guard is not None and self.flash_power_guard.failure:
                            raise Unavailable('Flash stopped to preserve power.')
                        if runtime is not None and runtime.guard_failure():
                            raise Unavailable('Flash stopped to preserve power or available memory.')
                        if self.process.poll() is not None:
                            raise Unavailable('Local model backend exited during loading. Setup diagnostics are in the LLM project.')
                        if engine.ready(self, config, backend_key):
                            break
                        time.sleep(.25)
                    else:
                        raise Unavailable('Local model loading timed out; no generation was started.')
                    self.last_used = time.monotonic()
                    self.state = 'ready'
                    self.launches += 1
                    self.clear_failure()  # M1: a successful start supersedes any older failure
                    if transition is not None:
                        engine_ledger.record_transition(ctl.ENGINE_LEDGER_LOG, transition[0], transition[1],
                                                        (time.monotonic() - transition[2]) * 1000, success=True)
                    # Idle-QoS hook: only now, with the engine reporting ready. The stop is in cleanup().
                    self.flash_idle_qos_pid = self.process.pid
                    self.flash_idle_qos_unit, self.flash_idle_qos_started, self.flash_idle_qos_error = \
                        self._flash_idle_qos('start', config, self.process.pid)
                    if runtime is not None:
                        self.active += 1
                        if inference: self.inferences += 1
                        return self.process
                    # Preserve llama.cpp's complete model metadata for its UI,
                    # including architecture/context fields; polling stays cheap
                    # after idle unload and never reopens the model.
                    engine.after_ready(self, backend_key)
                except Exception:
                    self.cleanup()
                    if transition is not None:
                        engine_ledger.record_transition(ctl.ENGINE_LEDGER_LOG, transition[0], transition[1],
                                                        (time.monotonic() - transition[2]) * 1000, success=False)
                    raise
            self.active += 1
            if inference: self.inferences += 1
            return self.process

    def close(self):
        self.stop_event.set()
        if self.image_broker is not None:
            self.image_broker.close()
        with self.mu: self.cleanup()
        self.thread.join(timeout=2)
