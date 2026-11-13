#!/usr/bin/env python3
"""Local llama.cpp proxy with an exclusive inference lease and idle unloading.

HTTP request/response bodies are not logged. Native stderr goes to a bounded,
private diagnostic log. Settings live in runtime.json; inference shares a file
lock with image jobs.

One exception to "nothing of a request is kept": for requests to the Rust engine the
harness registry saves the leading system messages and tool definitions (never user
turns, tool results or replies) as primers under LOCAL_AI_HARNESS_STATE, mode 0600, so
a harness prefix can be rebuilt on request. Set LOCAL_AI_HARNESS_STATE='' to keep the
registry in memory only.

Since v0.1.4 this file is the entry point and the one namespace; the code is in its siblings:
  ctl_env        paths, ports, time limits, the test guard, Unavailable/ModelBusy
  ctl_host       host probes (memory, swap, thermal, listeners)
  ctl_repack     llama.cpp repack-cache identity and error log
  ctl_lease      the inference lease, FIFO admission, the NPU image slot
  ctl_lifecycle  engine start, stop and the reaper
  ctl_engines    per-engine launch adapters for _begin
  ctl_cache      state cache, preload, prefix store, Flash cache preference
  ctl_models     runtime.json profiles, catalog, selection, context modes, harness settings
  ctl_status     the /_local/status reply
  ctl_manager    Manager (state fields + the mixins above)
  ctl_http       Handler: replies and the Host/Origin/CORS/key gate; Server
  ctl_routes     /_local/* control routes
  ctl_proxy      the proxy to the engine; its stages are ctl_proxy_prep, ctl_proxy_ledger, ctl_proxy_stream
  ctl_seams      `ctl`, through which those modules read names that tests patch here
Every name the single file had is still importable from here (tests/test_controller_contract.py).
"""
# All imports below are part of this namespace on purpose (controller.subprocess,
# controller.memory_policy, ... are used by tests and tools); do not prune the unused ones.
import collections
from contextlib import contextmanager
import errno
import fcntl
import gzip
import hashlib
import html
import http.client
import io
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import os
from pathlib import Path
import re
import resource
import shutil
import signal
import socket
import stat
import pwd
import subprocess
import sys
import threading
import traceback
import tempfile
import time
from urllib.parse import parse_qs, urlsplit
if __package__:
    from .prefix_plan import make_plan
    from .prefix_store import PrefixStore
    from . import bigmoe_runtime, classic_beta, memory_policy, model_catalog, profile_policy, oxe_runtime, engine_ledger, explicit_route, storage_view
    from .engine_diagnostics import EngineLogWriter, EngineStderrReader, swap_guard_mode
    from .client_disconnect import ClientDisconnect
    from . import harness_compat, openai_api, request_ledger, image_slot, image_npu, cache_turn, state_cache, lan_share
    from . import harness_registry
    from .harness_registry import HarnessRegistry, learning_settings, timings_from_tail
    from .ctl_seams import ctl
    from .ctl_env import (
        BACKEND_PORT, CAPABILITIES, CONFIG, ENGINE_DIAGNOSTIC_LOG, ENGINE_LEDGER_LOG,
        FLASH_BACKEND_PORT, HARNESS_STATE, IDLE_SECONDS, JOURNAL_HINT, KILL_WAIT_SECONDS,
        LAB_ORIGIN, LEASE, ModelBusy, ORIGIN_HEADER, PROXY_PORT, QUEUE_WAIT_SECONDS, StopPending,
        REAL_PRODUCTION_PATHS, START_SECONDS, TERM_WAIT_SECONDS, BACKEND_READ_TIMEOUT_SECONDS, Unavailable, _REAL_HOME, _log,
        guard_production_path, request_origin, under_test_runner, unexpected_cause)
    from .ctl_host import (
        CPUFREQ_ROOT, MAX_CPU_POLICIES, MAX_THERMAL_ZONES, THERMAL_ROOT, _sysfs_int,
        available_memory_bytes, cgroup_memory_max, listening_inodes, listening_process,
        port_listener_in_group, require_backend_port, resident_bytes, swap_bytes, thermal_sample,
        total_memory_bytes)
    from .ctl_repack import (
        REPACK_CACHE_ENV, REPACK_ERROR_LINES_PER_LOAD, REPACK_ERROR_LINE_CHARS, REPACK_ERROR_LOG,
        REPACK_ERROR_LOG_BYTES, REPACK_ERROR_LOG_KEEP, REPACK_ERROR_PREFIX, RepackErrorReader,
        SHA256_RE, disable_repack_cache, notify_repack_errors, repack_cache_enabled,
        repack_cache_identity, write_repack_error)
    from .ctl_status import StatusMixin, engine_summary, idle_seconds
    from .ctl_models import ModelsMixin
    from .ctl_cache import (
        CacheMixin, PRELOAD_FOLLOW_SECONDS, PRELOAD_POLL_SECONDS, PRELOAD_STATES, PRELOAD_TIMEOUT,
        preload_view)
    from .ctl_lease import LeaseMixin, open_lease_append, open_lease_rw
    from .ctl_lifecycle import LifecycleMixin
    from .ctl_manager import Manager
    from .ctl_routes import ControlRoutes, PASS
    from .ctl_proxy import FLASH_CHAT_HTML, HOP_HEADERS, ProxyRoutes, _BadFlashChat, request_defaults
    from .ctl_http import Handler, INFERENCE_PATHS, MODEL_HELPER_PATHS, Server, canonical_target
else:
    from prefix_plan import make_plan
    from prefix_store import PrefixStore
    import bigmoe_runtime
    import classic_beta
    import memory_policy
    import model_catalog
    import profile_policy
    import oxe_runtime
    import engine_ledger
    import explicit_route
    from engine_diagnostics import EngineLogWriter, EngineStderrReader, swap_guard_mode
    from client_disconnect import ClientDisconnect
    import harness_compat
    import openai_api
    import request_ledger
    import cache_turn
    import state_cache
    import image_slot
    import image_npu
    import harness_registry
    import lan_share
    from harness_registry import HarnessRegistry, learning_settings, timings_from_tail
    import storage_view
    from ctl_seams import ctl
    from ctl_env import (
        BACKEND_PORT, CAPABILITIES, CONFIG, ENGINE_DIAGNOSTIC_LOG, ENGINE_LEDGER_LOG,
        FLASH_BACKEND_PORT, HARNESS_STATE, IDLE_SECONDS, JOURNAL_HINT, KILL_WAIT_SECONDS,
        LAB_ORIGIN, LEASE, ModelBusy, ORIGIN_HEADER, PROXY_PORT, QUEUE_WAIT_SECONDS, StopPending,
        REAL_PRODUCTION_PATHS, START_SECONDS, TERM_WAIT_SECONDS, BACKEND_READ_TIMEOUT_SECONDS, Unavailable, _REAL_HOME, _log,
        guard_production_path, request_origin, under_test_runner, unexpected_cause)
    from ctl_host import (
        CPUFREQ_ROOT, MAX_CPU_POLICIES, MAX_THERMAL_ZONES, THERMAL_ROOT, _sysfs_int,
        available_memory_bytes, cgroup_memory_max, listening_inodes, listening_process,
        port_listener_in_group, require_backend_port, resident_bytes, swap_bytes, thermal_sample,
        total_memory_bytes)
    from ctl_repack import (
        REPACK_CACHE_ENV, REPACK_ERROR_LINES_PER_LOAD, REPACK_ERROR_LINE_CHARS, REPACK_ERROR_LOG,
        REPACK_ERROR_LOG_BYTES, REPACK_ERROR_LOG_KEEP, REPACK_ERROR_PREFIX, RepackErrorReader,
        SHA256_RE, disable_repack_cache, notify_repack_errors, repack_cache_enabled,
        repack_cache_identity, write_repack_error)
    from ctl_status import StatusMixin, engine_summary, idle_seconds
    from ctl_models import ModelsMixin
    from ctl_cache import (
        CacheMixin, PRELOAD_FOLLOW_SECONDS, PRELOAD_POLL_SECONDS, PRELOAD_STATES, PRELOAD_TIMEOUT,
        preload_view)
    from ctl_lease import LeaseMixin, open_lease_append, open_lease_rw
    from ctl_lifecycle import LifecycleMixin
    from ctl_manager import Manager
    from ctl_routes import ControlRoutes, PASS
    from ctl_proxy import FLASH_CHAT_HTML, HOP_HEADERS, ProxyRoutes, _BadFlashChat, request_defaults
    from ctl_http import Handler, INFERENCE_PATHS, MODEL_HELPER_PATHS, Server, canonical_target

ROOT = Path(__file__).resolve().parent
# The split modules (ctl_*.py) read patchable names through this namespace: see ctl_seams.py.
ctl.bind(sys.modules[__name__])


def log_effective_engines(manager, stream=None):
    """M4: one startup line per profile naming the engine it will really use, so a missing
    switch or binary is visible in the journal instead of silently running the classic stack."""
    stream = stream or sys.stderr
    try:
        value = json.loads(CONFIG.read_text())
        aliases = [value.get('model_alias', 'qwen-local'), *value.get('profiles', {})]
    except (OSError, ValueError, AttributeError, TypeError):
        print(f'local-ai: no readable runtime config at {CONFIG}', file=stream, flush=True)
        return
    for alias in dict.fromkeys(aliases):
        try:
            summary = engine_summary(manager.config(alias))
            note = f" ({summary['engine_note']})" if summary.get('engine_note') else ''
            print(f"local-ai: {alias}: engine {summary['engine']}{note}", file=stream, flush=True)
        except Exception as error:
            print(f'local-ai: {alias}: profile not usable: {error}', file=stream, flush=True)


def main():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    manager = Manager()
    log_effective_engines(manager)
    server = Server(('127.0.0.1', PROXY_PORT), Handler)
    server.manager = manager
    share = start_lan_share(manager)
    # Raising out of the serve loop guarantees child cleanup on service stop.
    def stop(*_): raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM, stop)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally:
        share.close()
        server.server_close()
        manager.close()


def start_lan_share(manager, stream=None):
    """O-102: shared listeners per runtime.json `lan_share` (off by default), one clear start-up
    line saying what is shared, and a watcher that applies changes and the off switch."""
    share = lan_share.ShareController(CONFIG, PROXY_PORT, Handler, manager=manager,
                                      guard=guard_production_path, salt=request_ledger._install_salt)
    manager.share = share
    try:
        share.apply()
    except Exception as error:
        share.error = 'LAN share could not start: ' + unexpected_cause(error)
    print(share.startup_line(), file=stream or sys.stderr, flush=True)
    share.start_watcher()
    return share


if __name__ == '__main__': main()
