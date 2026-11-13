"""Image Studio's job worker: the generation lease, one engine run (`run_engine`, the single engine
boundary, with the prompt memfd, the NPU worker hand-off, the memory guard and the stall watchdog),
the step pipeline of a job and the thread that runs it.

Every function takes the loaded Studio's context first (studio_context.py) and reads that Studio's
paths, settings, job state and collaborators as `ctx.NAME`; nothing here is per-process state.
"""
import fcntl
import json
import math
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

# What `studio` binds to its context and exposes under the same names (StudioContext.bind).
FUNCTIONS = (
    'unload_idle_llm', 'claim_generation_slot', 'register_image', 'update_job_progress',
    'vas_counter_thread', 'oxi_stall_seconds', 'run_engine', 'run_pipeline', 'run_job', 'start_job',)


def unload_idle_llm(ctx):
    """Ask the local controller to release only a nonbusy resident chat model."""
    request = ctx.Request('http://127.0.0.1:8092/_local/unload', data=b'{}',
                      headers={'Content-Type': 'application/json'})
    try:
        with ctx.urlopen(request, timeout=15) as response:
            value = json.loads(response.read(4096))
            return response.status == 200 and isinstance(value, dict) and value.get('state') == 'unloaded'
    except (OSError, ValueError):
        return False


def claim_generation_slot(ctx, generation_lock):
    try:
        fcntl.flock(generation_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except BlockingIOError:
        if not ctx.unload_idle_llm():
            return False
    # The controller refuses an unload while chat inference is active. Other
    # benchmark/image leases remain untouched, even when it reports unloaded.
    try:
        fcntl.flock(generation_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except BlockingIOError:
        return False


def register_image(ctx, record):
    """Remember a finished image server-side; the prompt never leaves this process."""
    image_id = uuid.uuid4().hex
    with ctx.LOCK:
        ctx.IMAGES[image_id] = record
        while len(ctx.IMAGES) > ctx.IMAGE_RECORD_LIMIT:
            ctx.IMAGES.pop(next(iter(ctx.IMAGES)))
    return image_id


def update_job_progress(ctx, line, job):
    """Keep only fixed lifecycle labels and numerical progress, never output text."""
    lower = line.lower()
    if 'sampling completed' in lower:
        job['phase'] = 'decoding'
        job['state'] = 'preparing decoder'
    elif 'decoding' in lower or 'decode_first_stage' in lower:
        job['phase'] = 'decoding'
        job['state'] = 'decoding'
    elif 'generating image:' in lower:
        job['phase'] = 'sampling'
        job['state'] = 'starting sampling'
    elif lower.startswith('forward: total'):
        # oxi (IMG-SHIP): one line per DiT forward = one sampling step; the VAE decode follows the last
        job['oxi_forwards'] = job.get('oxi_forwards', 0) + 1
        done = job['oxi_forwards']
        job['phase'] = 'decoding' if done >= ctx.EDIT_STEPS else 'sampling'
        job['state'] = 'decoding' if done >= ctx.EDIT_STEPS else f'generating {done + 1}/{ctx.EDIT_STEPS}'
    elif 'get_learned_condition completed' in lower:
        job['state'] = 'prompt encoded'
    elif 'loading tensors completed' in lower:
        job['state'] = ('decoding' if job.get('phase') == 'decoding' else
                        'starting sampling' if job.get('phase') == 'sampling' else
                        'preparing generation')
    elif 'loading tensors' in lower:
        job['state'] = 'decoding' if job.get('phase') == 'decoding' else 'loading model'
    else:
        m = re.search(r"(\d+)/(\d+) - [\d.]+s/it", line)
        if m:
            stage = 'decoding' if job.get('phase') == 'decoding' else 'generating'
            job['state'] = f'{stage} {m.group(1)}/{m.group(2)}'


def vas_counter_thread(ctx, proc, job, stop=None, ready=None, beat=None):
    """Recognize progress even when sd-cli prefixes CR and omits a newline.

    `stop(line)` returning True terminates the process (the oxi hand-over stops sd-cli once the
    first model call's inputs are on disk); the output is still read to the end. `beat()` is called
    on every output line (CR or newline) for the stall watchdog."""
    line = ''
    while char := proc.stdout.read(1):
        if char in '\r\n':
            if beat is not None:
                beat()
            ctx.update_job_progress(line, job)
            if ready is not None and (line.startswith('forward:') or 'loading tensors completed' in line.lower()):
                ready()
            if stop is not None and line and stop(line) and proc.poll() is None:
                proc.terminate()
            line = ''
        else:
            line = (line + char)[-2048:]
            # The current progress record may have no trailing separator until
            # the next (potentially minutes-long) sampling step completes.
            if char == 't' and line.endswith('s/it'):
                ctx.update_job_progress(line, job)
    if line:
        ctx.update_job_progress(line, job)


def oxi_stall_seconds(ctx):
    try:
        value = json.loads((ctx.PROJECT / 'runtime-settings.json').read_text()).get('oxi', {}).get('stall_seconds')
    except (OSError, ValueError, AttributeError):
        value = None
    if type(value) in (int, float) and math.isfinite(value) and 0 <= value <= ctx.OXI_MAX_STALL_SECONDS:
        return float(value)
    return ctx.OXI_DEFAULT_STALL_SECONDS


class StallWatchdog:
    """Stops `proc` when no output line arrived for `seconds` (see OXI_DEFAULT_STALL_SECONDS)."""

    ctx = None  # the loaded Studio: set on the subclass made by StudioContext.bind_class

    def __init__(self, proc, seconds, grace=None):
        self.proc = proc
        self.seconds = float(seconds)
        self.grace = self.ctx.OXI_STALL_GRACE_SECONDS if grace is None else float(grace)
        self.last = time.monotonic()
        self.stalled = False
        self.done = threading.Event()
        self.thread = threading.Thread(target=self._watch, daemon=True, name='engine-stall-watchdog')

    def beat(self):
        self.last = time.monotonic()

    def start(self):
        self.thread.start()
        return self

    def stop(self):
        self.done.set()
        if self.thread.is_alive() and self.thread is not threading.current_thread():
            self.thread.join(timeout=self.grace + 5)

    def _watch(self):
        while not self.done.wait(min(1.0, max(0.05, self.seconds / 4))):
            if time.monotonic() - self.last < self.seconds or self.proc.poll() is not None:
                continue
            self.stalled = True
            try:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=self.grace)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    self.proc.wait(timeout=self.grace + 5)
            except (OSError, subprocess.TimeoutExpired):
                pass
            return


def run_engine(ctx, args, job, environment=None, prompt=None, prompt_flag='--prompt-file', stop=None):
    """Run one sd-cli (or oxi) invocation, streaming progress into `job`; returns the exit code.

    This is the single engine boundary: tests replace it with a stub. The prompt goes in through an
    anonymous memory file named by `prompt_flag` (oxi generate reads it as --cond-prompt-file).
    oxi, probe sd-cli and NPU worker runs have a stall watchdog (oxi_stall_seconds()).
    """
    prompt_fd = None
    proc = None
    npu_worker = False
    guard = None
    watchdog = None
    try:
        command = list(args)
        watched = bool(command) and str(command[0]) in (str(ctx.OXI_BIN), str(ctx.SD_PROBE_CLI))
        engine_name = 'sd-cli (probe)' if watched and str(command[0]) == str(ctx.SD_PROBE_CLI) else 'oxi'
        if prompt is not None:
            # Anonymous memory only: no prompt in argv, named files, or logs.
            prompt_fd = os.memfd_create('prompt-studio', os.MFD_CLOEXEC)
            content = prompt.encode('utf-8')
            offset = 0
            while offset < len(content):
                written = os.write(prompt_fd, content[offset:])
                if written <= 0: raise OSError('Prompt memory write failed')
                offset += written
            os.lseek(prompt_fd, 0, os.SEEK_SET)
            command += [prompt_flag, f'/proc/self/fd/{prompt_fd}']
        permit = ctx.IMAGE_LEASES.get(id(job))
        inherited = (() if permit is None else permit.pass_fds()) + (() if prompt_fd is None else (prompt_fd,))
        npu_te = (command[:2] == [str(ctx.OXI_BIN), 'cond-native']
                  and ctx.OXI_TE_NPU_ARITH in command and '--te-backend' not in command)
        if job.get('npu_requested') and (command[:2] == [str(ctx.OXI_BIN), 'generate'] or npu_te):
            # Arithmetic is pinned by the checked worker, not by inherited CLI knobs.
            native = []
            skip = {'--backend', '--chunk1', '--chunk2', '--attn-prec', '--threads', '--vae-conv'}
            i = 1
            while i < len(command):
                if command[i] in skip:
                    i += 2
                else:
                    native.append(command[i]); i += 1
            if permit is None:
                raise ctx.OxiNpuRecovery('NPU generation has no controller permit')
            worker = Path(ctx.__file__).with_name('image_npu_worker.py')
            npu_worker = True
            watched, engine_name = True, 'NPU worker'
            command = [sys.executable, '-B', str(worker), '--profile', job['npu_profile'],
                       '--lease-fd', str(permit.fileno()), '--slot-fd', str(permit.connection.fileno()),
                       *([] if prompt_fd is None else ['--prompt-fd', str(prompt_fd)]), '--', *native]
            environment = ctx.image_npu.clean_environment(environment or os.environ)
            environment.pop('PYTHONPATH', None)
            environment.pop('PYTHONHOME', None)
        elif permit is not None:
            # A retained chat also needs protection while CPU conditioning or
            # fallback runs. These native commands exec in place (no timeout
            # process group), so the existing PID guard owns the whole worker.
            guard = ctx.memory_policy.MemoryGuard(host_guard_version=ctx.memory_policy.GUARD_2_VERSION,
                loading_timeout_seconds=120, follow_process_cgroup=True)
            command = ctx.memory_policy.scope_command(command, 28 * 1024**3)
        # Text mode recognizes both CR progress updates and newline log records.
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace", bufsize=1,
                                env=environment, pass_fds=inherited)
        if guard is not None:
            guard.start(proc)
        stall_seconds = ctx.oxi_stall_seconds() if watched else 0
        if stall_seconds > 0:
            watchdog = ctx.StallWatchdog(proc, stall_seconds).start()
        if watchdog is None:
            ctx.vas_counter_thread(proc, job, stop, None if guard is None else guard.mark_ready)
        else:
            ctx.vas_counter_thread(proc, job, stop, None if guard is None else guard.mark_ready, beat=watchdog.beat)
            watchdog.stop()
            if watchdog.stalled:
                message = (f'{engine_name} stalled: no progress output for {stall_seconds:g} s, '
                           'so it was stopped')
                if npu_worker:
                    raise ctx.OxiNpuRecovery(message + '; refusing a CPU retry until native cleanup is confirmed')
                raise ctx.OxiFailure(message)
        if guard is not None and guard.failure():
            raise ctx.OxiNpuRecovery('Image memory guard stopped compute: ' + guard.failure())
        return proc.wait()
    except BaseException as error:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        if npu_worker and not isinstance(error, ctx.OxiNpuRecovery):
            raise ctx.OxiNpuRecovery('NPU worker interrupted; refusing a CPU retry until native cleanup is confirmed')
        raise
    finally:
        if watchdog is not None:
            watchdog.stop()
        if guard is not None:
            guard.stop()
        if proc is not None and proc.stdout is not None:
            proc.stdout.close()
        if prompt_fd is not None:
            os.close(prompt_fd)


def run_pipeline(ctx, job_id, steps, generation_lock, environment=None, prompt=None, results=None):
    """Run engine steps in order under one generation lease.

    Each step: {'args', 'outputs': [Path], 'label', 'prompt': bool,
    'environment' (optional), 'expect_size': (w, h) or None}. sd-cli exits 0
    even when ESRGAN fails (it then saves the unscaled input), so the PNG size
    is checked. Optional 'pre'/'post' callables run before/after the engine
    (edit mode: prepare the working image, composite the mask). `results` lists finished-image records to register on success.
    """
    job = ctx.JOBS[job_id]
    try:
        for index, step in enumerate(steps, 1):
            job['step'] = f"{step.get('label', 'working')} ({index}/{len(steps)})" if len(steps) > 1 else ''
            job['state'] = 'starting'
            if step.get('pre'):
                step['pre']()
            if step.get('run'):
                # IMG-SHIP: a step that drives its own engine calls (oxi, with the sd-cli fallback)
                rc = step['run'](job, prompt if step.get('prompt', True) else None)
            else:
                rc = ctx.run_engine(step['args'], job, step.get('environment', environment),
                                prompt if step.get('prompt', True) else None)
            missing = [p for p in step['outputs'] if not Path(p).exists()]
            if rc != 0 or missing:
                job['state'] = 'error'
                job['error'] = f"{job.get('engine_binary', 'sd-cli')} exited {rc}" + (' without writing its image' if rc == 0 else '')
                return
            expected = step.get('expect_size')
            if expected is not None:
                for p in step['outputs']:
                    actual = ctx.png_size(p)
                    if actual != tuple(expected):
                        job['state'] = 'error'
                        job['error'] = (f"{step.get('label', 'step')} produced "
                                        f"{'an unreadable image' if actual is None else '%d×%d' % actual}, "
                                        f"expected {expected[0]}×{expected[1]}")
                        return
            if step.get('post'):
                job['state'] = 'compositing'
                step['post']()
        images = []
        for record in results or []:
            image_id = ctx.register_image(record)
            entry = {'id': image_id, 'path': record['path'], 'seed': record['seed'],
                     'width': record['width'], 'height': record['height'],
                     'kind': record['kind'], 'actions': ctx.action_estimates(record)}
            if 'crop' in record:
                entry['crop'] = record['crop']
            if 'backend' in record:
                entry['backend'] = record['backend']
            images.append(entry)
        job['images'] = images
        if images:
            job['path'] = images[0]['path']
        elif steps:
            job['path'] = str(steps[-1]['outputs'][-1])
        job.pop('step', None)
        job['state'] = 'done'
    except Exception as e:
        job["state"] = "error"
        job["error"] = str(e)
    finally:
        for scratch in ctx.SCRATCH.pop(job_id, []):
            try:
                Path(scratch).unlink()
            except OSError:
                pass
        ctx.IMAGE_LEASES.pop(id(job), None)
        generation_lock.close()


def run_job(ctx, job_id, args, out_path, generation_lock, environment=None, prompt=None):
    """Single-command job (kept for callers of the original interface)."""
    ctx.run_pipeline(job_id, [{'args': args, 'outputs': [out_path], 'label': 'generating'}],
                 generation_lock, environment, prompt)


def start_job(ctx, plan):
    """Claim the generation lease and run the plan in the background; (job_id, error)."""
    job_id = uuid.uuid4().hex
    with ctx.LOCK:
        if any(j["state"] not in ("done", "error") for j in ctx.JOBS.values()):
            return None, "A generation is already running"
        npu_enabled, npu_profile = ctx.oxi_npu_settings()
        # The NPU permit/profile is qualified for 512 px only: gated sizes run oxi on the CPU.
        use_broker = (npu_enabled and plan.get('engine', {}).get('engine') == 'oxi'
                      and all((r.get('width'), r.get('height')) in ctx.OXI_SIZES for r in plan.get('results') or [{}]))
        if use_broker:
            try:
                generation_lock = ctx.image_slot.connect(
                    Path(os.environ.get('LOCAL_AI_IMAGE_SOCKET', str(ctx.HOME / '.local/state/local-ai/image-slot/permit.sock'))),
                    'klein-q8-512-npu')
            except (OSError, ValueError, ctx.image_slot.SlotUnavailable):
                return None, 'Image permit unavailable; resident chat was kept loaded. Try again when chat is idle.'
        else:
            generation_lock = ctx.GENERATION_LOCK.open('a')
            if not ctx.claim_generation_slot(generation_lock):
                generation_lock.close()
                return None, "Another image or chat job is running; please try again shortly"
        ctx.JOBS[job_id] = {"state": "queued", "model": plan['model'], "device": plan['device']}
        if use_broker:
            job = ctx.JOBS[job_id]
            ctx.IMAGE_LEASES[id(job)] = generation_lock
            job['npu_requested'] = generation_lock.metadata.get('backend') == 'npu' and npu_profile is not None
            job['npu_profile'] = npu_profile
            job['engine_note'] = generation_lock.metadata.get('reason', '')
        if plan.get('engine'):
            ctx.JOBS[job_id]['engine'] = 'oxi' if plan['engine']['engine'] == 'oxi' else 'sd.cpp'
            ctx.JOBS[job_id]['engine_reason'] = plan['engine']['reason']
        ctx.SCRATCH[job_id] = list(plan.get('scratch', []))
    threading.Thread(target=ctx.run_pipeline,
                     args=(job_id, plan['steps'], generation_lock, plan['environment'], plan['prompt'], plan['results']),
                     daemon=True).start()
    return job_id, None
