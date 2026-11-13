"""The Manager: one engine process, one lease, one inference at a time.

`__init__` below is the inventory of its state. The behaviour is in the mixins: ctl_lease
(lease and queue), ctl_lifecycle (begin, cleanup, reaper), ctl_cache, ctl_models, ctl_status.
"""
import collections
import os
from pathlib import Path
import threading
if __package__:
    from .ctl_cache import CacheMixin
    from .ctl_lease import LeaseMixin
    from .ctl_lifecycle import LifecycleMixin
    from .ctl_models import ModelsMixin
    from .ctl_seams import ctl
    from .ctl_status import StatusMixin
    from . import image_slot
else:
    from ctl_cache import CacheMixin
    from ctl_lease import LeaseMixin
    from ctl_lifecycle import LifecycleMixin
    from ctl_models import ModelsMixin
    from ctl_seams import ctl
    from ctl_status import StatusMixin
    import image_slot


class Manager(StatusMixin, ModelsMixin, CacheMixin, LeaseMixin, LifecycleMixin):
    def __init__(self):
        self.mu = threading.RLock()
        self.process = None
        self.lease = None
        self.active = 0
        self.inferences = 0
        self.last_used = 0
        self.state = 'unloaded'
        self.cache = {}
        self.flash_cache_status = None
        self.flash_swap_guard_reported = None  # G-F127: the engine's start-up [swap-guard mode=...]
        self.loaded_config = None
        self.state_engine = False  # the loaded oxe engine owns the disk state store (--state-cache)
        self.state_preload = False  # ... and serves POST /state-cache/preload (P2b)
        self.launches = 0  # engines that reached ready; an explicit load preloads only its own launch
        self.session_hints = {}  # model alias -> last X-Oxe-Session hint forwarded to it
        # V013: the last preload of the loaded engine: {state, src, ms, at (monotonic)}; None = idle
        self.preload_info = None
        self.remote_preempt = None  # O-102 M4: the running shared request's lan_share.Preempt
        self.prefix_store = None
        # FIFO admission of inference requests (see begin_queued)
        self.queue_cv = threading.Condition(threading.Lock())
        self.queue_waiters = collections.deque()
        self.queue_reserved = False
        self.backend_runtime = None
        self.classic_memory_guard = None
        self.flash_power_guard = None
        self.repack_reader = None
        self.engine_reader = None
        self.engine_log_writer = None
        self.engine_ledger = None
        # Optional Flash idle-QoS hook (root unit, see oxe_runtime.flash_idle_qos_unit); best effort.
        self.flash_idle_qos_pid = None
        self.flash_idle_qos_unit = None
        self.flash_idle_qos_started = False
        self.flash_idle_qos_error = None
        self.engine_death_lock = threading.Lock()
        self.last_memory_guard = None
        # M1: {reason, kind, alias, origin, t, detail?}; cleared on a successful start and on
        # every new explicit load. Lab-origin entries never reach the user-facing last_error.
        self.last_failure_info = None
        self.activity_origin = 'organic'
        self.stop_pending = None  # S1: set while an engine's exit is unproven (lease kept)
        self._start_attempt = False
        self.current_prefix_key = None
        self.prefix_stats = {'restores': 0, 'primes': 0, 'failures': 0}
        self.harness_registry = ctl.HarnessRegistry(ctl.HARNESS_STATE or None)
        self.compat_stats = collections.Counter()
        self.answering_since = None
        self.image_active = False
        self.image_broker = None
        self.image_broker_error = None
        self.harness_mu = threading.Lock()  # registry only; never held across a backend call
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.reaper, daemon=True)
        self.thread.start()
        if os.environ.get('LOCAL_AI_IMAGE_NPU') == '1':
            path = Path(os.environ.get('LOCAL_AI_IMAGE_SOCKET', str(Path.home() / '.local/state/local-ai/image-slot/permit.sock')))
            try:
                # Explicit opt-in: only the exact value "1" kills the stale client; anything else is off.
                kill_client = os.environ.get('LOCAL_AI_IMAGE_WATCHDOG_KILL_CLIENT') == '1'
                self.image_broker = image_slot.Broker(path, self.reserve_image, kill_client=kill_client).start()
            except (OSError, image_slot.SlotUnavailable):
                self.image_broker_error = 'Image slot could not be started; NPU images remain unavailable.'
