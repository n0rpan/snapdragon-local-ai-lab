"""v0.1.3 contract for the Studio module split (phase 2): what a loaded `studio` exposes must not change.

Studio is a script that tests and tools load fresh from its path (SourceFileLoader); every load has
its own paths (from HOME and the environment), its own job state and its own replaceable names
(`patch.object(studio, 'run_engine', ...)`). These lists were taken from the single-file worker and
planners at v0.1.3 (878a492). A split may move code; it may not drop or rename a name, change a
signature, share state between two loads or stop honouring a name replaced on the loaded module.
"""
import importlib.machinery
import importlib.util
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

SOURCE = Path(__file__).resolve().parents[1] / 'src/snapdragon_ai_lab/runtime/studio'
ENVIRONMENT = {'IMG_OUTDIR': 'output', 'LOCAL_IMAGE_CONFIG_DIR': 'config', 'LOCAL_AI_LEASE': 'inference.lock',
               'SD_HOME': 'stack'}
UNSET = ('SD_CLI', 'OXI_BIN', 'SD_CLI_PROBE', 'OXI_COND_CACHE', 'OXI_WORK', 'OXI_SIZE_GATE_DIR')

NAMES = """
    BaseHTTPRequestHandler DEFAULT_EDIT_PRESET DEFAULT_EDIT_SIZE EDIT_ESTIMATE_DEFAULTS EDIT_PRESETS
    EDIT_SIZES EDIT_STEPS ENGINE_CHOICES ENGINE_LABELS ESTIMATE_DEFAULTS GENERATION_LOCK GPU_BINARY_CHECKS H
    HANDOVER_TAGS HOME IMAGES IMAGE_LEASES IMAGE_RECORD_LIMIT JOBS KLEIN_DIR KLEIN_FILES KLEIN_LLM
    KLEIN_MODELS KLEIN_T2I_SIZE_MAX KLEIN_T2I_SIZE_MIN KLEIN_VAE LLM LOCK MAX_INSTRUCTION MAX_MASK_PIXELS
    MAX_VARIATIONS MIN_CROP_SIDE MODELS MODEL_FILES OUT OXI_BIN OXI_COND_CACHE OXI_COND_KEEP
    OXI_DEFAULT_STALL_SECONDS OXI_DEFAULT_THREADS OXI_DEFAULT_VAE_CONV OXI_ESTIMATE OXI_FAST_ARGS
    OXI_GATE_DIR OXI_GATE_SCHEMA OXI_MAX_STALL_SECONDS OXI_MODELS OXI_NEEDS_REFERENCE OXI_NO_CONDITIONING
    OXI_SHA_CACHE OXI_SIZES OXI_STALL_GRACE_SECONDS OXI_TEXT_ENCODERS OXI_TE_CPU_ARITH OXI_TE_NPU_ARITH
    OXI_VAE_CONVS OXI_WORK OxiFailure OxiNpuFallback OxiNpuRecovery PAGE PROJECT Path Request SCRATCH SDCLI
    SDCPP_DEFAULT_SEED SD_PROBE_CLI STACK S_ISREG StallWatchdog StudioError ThreadingHTTPServer UPLOADS
    UPSCALE_DEFAULTS UPSCALE_SUFFIXES VAE _ceil16 _fit_short _floor16 _grow_axis _taps _widen_axis
    action_estimates box_blur build_edit_args build_generation_args build_klein_t2i_args build_upscale_args
    choose_t2i_engine chosen_image claim_generation_slot composite_crop composite_files composite_masked
    crop_box_for_mask crop_pixels edit_estimate_seconds edit_estimate_settings edit_info edit_missing_files
    edit_strength edit_work_size engine_info engine_setting estimate_seconds fcntl handover_stop hashlib
    image_npu image_presets image_size image_slot jpeg_size json klein_info make_conditioning
    mask_bounding_box math memory_policy models_relative_file optimized_cpu_profile optimized_gpu_profile os
    oxi_binary_sha256 oxi_cache_environment oxi_cond_native_args oxi_cond_put_args oxi_estimate_seconds
    oxi_fast_setting oxi_gated_sizes oxi_generate_all oxi_generate_args oxi_missing oxi_npu_settings
    oxi_prompt_cache_enabled oxi_size_gate_passed oxi_stall_seconds oxi_te_arith oxi_te_key_args
    oxi_text_encoder oxi_threads oxi_vae_conv parse_qs plan_edit plan_generation plan_klein_text
    plan_quick_upscale plan_refine png_size prepare_edit_inputs prune_cond_cache re read_mask_at read_png
    refine_size refine_source register_image resize_pixels run_engine run_job run_oxi_t2i run_pipeline
    runtime_tuning sampled_steps save_engine_setting shutil start_job struct subprocess sys tempfile
    threading time unload_idle_llm update_job_progress upscale_settings urlopen urlparse uuid valid_edge
    vas_counter_thread write_png zlib
""".split()

SIGNATURES = {
    'unload_idle_llm': '()',
    'claim_generation_slot': '(generation_lock)',
    'optimized_gpu_profile': '(model, width, height)',
    'optimized_cpu_profile': '(model, width, height, steps, has_init=False)',
    'image_presets': '()',
    'models_relative_file': '(value)',
    'valid_edge': '(value)',
    'upscale_settings': '()',
    'edit_strength': '(body)',
    'sampled_steps': '(steps, strength=None)',
    'estimate_seconds': "(settings, kind, model='q8', width=256, height=256, steps=8, strength=None, count=1, optimized=True, input_size=None, scale=4)",
    'build_generation_args': '(binary, diffusion, width, height, steps, out, threads, fa, cpu=True, seed=None, gpu_profile=None, cpu_profile=None, init=None, strength=None, mask=None, batch=1, init_filter=None)',
    'build_upscale_args': '(binary, model_path, source, out, threads, tile_size)',
    'runtime_tuning': '(model)',
    'register_image': '(record)',
    'action_estimates': '(record, settings=None)',
    'update_job_progress': '(line, job)',
    'vas_counter_thread': '(proc, job, stop=None, ready=None, beat=None)',
    'oxi_stall_seconds': '()',
    'run_engine': "(args, job, environment=None, prompt=None, prompt_flag='--prompt-file', stop=None)",
    'run_pipeline': '(job_id, steps, generation_lock, environment=None, prompt=None, results=None)',
    'run_job': '(job_id, args, out_path, generation_lock, environment=None, prompt=None)',
    'plan_generation': '(body)',
    'chosen_image': '(body)',
    'plan_quick_upscale': '(body)',
    'refine_source': '(record)',
    'plan_refine': '(body)',
    'edit_estimate_settings': '()',
    'edit_estimate_seconds': '(width, height, steps=4, masked=False, t2i=False, count=1)',
    'edit_missing_files': "(model='q8')",
    'klein_info': '()',
    'edit_info': '()',
    'prepare_edit_inputs': '(source, mask, work_size, ref_out)',
    'build_edit_args': "(binary, diffusion, ref, out, width, height, steps, threads, fa='off', seed=None)",
    'build_klein_t2i_args': "(binary, diffusion, width, height, out, threads, fa='off', seed=None, batch=1)",
    'plan_klein_text': '(body)',
    'engine_setting': '()',
    'save_engine_setting': '(value)',
    'oxi_threads': '()',
    'oxi_prompt_cache_enabled': '()',
    'oxi_npu_settings': '()',
    'oxi_text_encoder': '()',
    'oxi_te_arith': '(job)',
    'oxi_te_key_args': '(spec)',
    'oxi_cache_environment': '(cache_dir)',
    'oxi_vae_conv': '()',
    'oxi_missing': '()',
    'oxi_binary_sha256': '(path=None)',
    'oxi_fast_setting': '(flag)',
    'oxi_size_gate_passed': '(width, height)',
    'oxi_gated_sizes': '()',
    'choose_t2i_engine': '(setting, model, width, height, missing)',
    'engine_info': '()',
    'oxi_estimate_seconds': '(count, width=512, height=512)',
    'oxi_generate_args': '(spec, seed, out, cache_dir=None)',
    'oxi_cond_native_args': '(spec, cache_dir=None)',
    'oxi_cond_put_args': '(spec, handover, cache_dir=None)',
    'handover_stop': '()',
    'prune_cond_cache': '(keep=16, cache_dir=None)',
    'make_conditioning': '(job, prompt, spec, cache_dir=None)',
    'oxi_generate_all': '(job, prompt, spec, cache_dir=None)',
    'run_oxi_t2i': '(job, prompt, spec)',
    'plan_edit': '(body)',
    'start_job': '(plan)',
}

CLASSES = {
    'StudioError': (['Exception'], ['__init__']),
    'StallWatchdog': ([], ['__init__', 'beat', 'start', 'stop', '_watch']),
    'H': (['BaseHTTPRequestHandler'], ['log_message', '_host_ok', '_post_ok', '_refuse', '_send', 'do_GET', 'do_POST']),
    'OxiFailure': (['Exception'], []),
    'OxiNpuFallback': (['OxiFailure'], []),
    'OxiNpuRecovery': (['OxiFailure'], []),
}

PER_LOAD_PATHS = ('STACK', 'SDCLI', 'MODELS', 'OUT', 'UPLOADS', 'LLM', 'VAE', 'KLEIN_DIR', 'KLEIN_LLM', 'KLEIN_VAE',
                  'PROJECT', 'GENERATION_LOCK', 'OXI_BIN', 'SD_PROBE_CLI', 'OXI_COND_CACHE', 'OXI_WORK', 'OXI_GATE_DIR')
PER_LOAD_STATE = ('JOBS', 'IMAGE_LEASES', 'LOCK', 'IMAGES', 'SCRATCH', 'GPU_BINARY_CHECKS', 'OXI_SHA_CACHE',
                  'MODEL_FILES', 'KLEIN_FILES', 'ESTIMATE_DEFAULTS', 'UPSCALE_DEFAULTS', 'EDIT_ESTIMATE_DEFAULTS',
                  'EDIT_PRESETS', 'ENGINE_LABELS', 'OXI_ESTIMATE')
PER_LOAD_CLASSES = ('StudioError', 'StallWatchdog', 'H', 'OxiFailure', 'OxiNpuFallback', 'OxiNpuRecovery')


def load(root, name):
    with patch.dict(os.environ, {key: str(Path(root) / value) for key, value in ENVIRONMENT.items()}):
        for key in UNSET:
            os.environ.pop(key, None)
        loader = importlib.machinery.SourceFileLoader(name, str(SOURCE))
        spec = importlib.util.spec_from_loader(loader.name, loader)
        module = importlib.util.module_from_spec(spec)
        loader.exec_module(module)
    return module


class StudioContract(unittest.TestCase):
    maxDiff = None

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.a = load(self.root / 'a', 'contract_studio_a')
        self.b = load(self.root / 'b', 'contract_studio_b')

    def test_every_old_name_is_still_there(self):
        self.assertEqual(sorted(set(NAMES) - set(vars(self.a))), [])

    def test_signatures(self):
        self.assertEqual({name: str(inspect.signature(getattr(self.a, name))) for name in SIGNATURES}, SIGNATURES)

    def test_classes_keep_their_bases_and_methods(self):
        s = self.a
        for name, (bases, methods) in CLASSES.items():
            cls = getattr(s, name)
            self.assertIsInstance(cls, type, name)
            self.assertEqual([base for base in bases if base not in [c.__name__ for c in cls.__mro__[1:]]], [], name)
            self.assertEqual([method for method in methods if not callable(getattr(cls, method, None))], [], name)
        self.assertEqual(str(inspect.signature(s.StudioError)), '(message, code=400)')
        self.assertEqual(str(inspect.signature(s.StallWatchdog)), '(proc, seconds, grace=None)')
        self.assertEqual(s.StudioError('x', 409).code, 409)
        self.assertTrue(issubclass(s.OxiNpuFallback, s.OxiFailure) and issubclass(s.OxiNpuRecovery, s.OxiFailure))

    def test_each_load_has_its_own_paths_state_and_classes(self):
        a, b = self.a, self.b
        for name in PER_LOAD_PATHS:
            self.assertTrue(getattr(a, name).is_relative_to(self.root / 'a'), name)
            self.assertTrue(getattr(b, name).is_relative_to(self.root / 'b'), name)
        self.assertEqual(a.HOME, Path.home())
        for name in PER_LOAD_STATE + PER_LOAD_CLASSES:
            self.assertIsNot(getattr(a, name), getattr(b, name), name)
        for path in list(a.MODEL_FILES.values()) + list(a.KLEIN_FILES.values()):
            self.assertTrue(path.is_relative_to(self.root / 'a'))
        self.assertTrue(a.OUT.is_dir() and a.UPLOADS.is_dir() and a.GENERATION_LOCK.parent.is_dir())

    def test_values_that_are_arguments_defaults(self):
        self.assertEqual((self.a.EDIT_STEPS, self.a.OXI_COND_KEEP), (4, 16))

    def test_a_replaced_function_is_what_the_worker_calls_on_that_load_only(self):
        a, b = self.a, self.b
        output = self.root / 'image.png'
        output.write_bytes(b'x')
        for module in (a, b):
            module.JOBS['job'] = {'state': 'queued'}
            module.SCRATCH['job'] = []
        lock = Mock()
        with patch.object(a, 'run_engine', return_value=0) as engine, \
                patch.object(b, 'run_engine', side_effect=AssertionError('the other load ran')):
            a.run_pipeline('job', [{'args': ['engine', 'argument'], 'outputs': [output]}], lock)
        engine.assert_called_once_with(['engine', 'argument'], a.JOBS['job'], None, None)
        lock.close.assert_called_once_with()
        self.assertEqual(a.JOBS['job'], {'state': 'done', 'images': [], 'path': str(output)})
        self.assertEqual(b.JOBS['job'], {'state': 'queued'})
        with patch.object(a, 'run_pipeline') as pipeline:
            a.run_job('job', ['x'], output, lock, {'K': 'V'}, 'prompt')
        pipeline.assert_called_once_with('job', [{'args': ['x'], 'outputs': [output], 'label': 'generating'}],
                                         lock, {'K': 'V'}, 'prompt')

    def test_a_replaced_function_is_what_the_planners_call_on_that_load_only(self):
        a, b = self.a, self.b
        with patch.object(a, 'edit_missing_files', return_value=['sentinel']) as missing:
            self.assertEqual(a.oxi_missing(), ['oxi', 'sd-cli (probe)', 'sentinel'])
            self.assertNotIn('sentinel', b.oxi_missing())
        missing.assert_called_once_with('q8')
        with patch.object(a, 'oxi_size_gate_passed', return_value=True) as gate:
            self.assertEqual(a.choose_t2i_engine('auto', 'klein-q8', 768, 768, [])[0], 'oxi')
            self.assertEqual(b.choose_t2i_engine('auto', 'klein-q8', 768, 768, [])[0], 'sdcpp')
        gate.assert_called_once_with(768, 768)

    def test_a_replaced_constant_is_read_on_that_load_only(self):
        a, b = self.a, self.b
        with patch.object(a, 'OXI_SIZES', ((640, 640),)), patch.object(a, 'OXI_STALL_GRACE_SECONDS', 0.25), \
                patch.object(a, 'OXI_ESTIMATE', {'handover_s': 1.0, 'image_s': 2.0}), \
                patch.object(a, 'EDIT_STEPS', 2):
            self.assertEqual(a.choose_t2i_engine('auto', 'klein-q8', 640, 640, [])[0], 'oxi')
            self.assertEqual(b.choose_t2i_engine('auto', 'klein-q8', 640, 640, [])[0], 'sdcpp')
            self.assertEqual((a.StallWatchdog(Mock(), 5).grace, b.StallWatchdog(Mock(), 5).grace), (0.25, 10.0))
            self.assertEqual((a.oxi_estimate_seconds(2), b.oxi_estimate_seconds(2)), (5.0, 94.0))
            job = {}
            a.update_job_progress('forward: total 1.0 s', job)
            a.update_job_progress('forward: total 1.0 s', job)
            self.assertEqual(job, {'oxi_forwards': 2, 'phase': 'decoding', 'state': 'decoding'})
            # a default argument keeps the value it had when the module was loaded
            self.assertEqual(str(inspect.signature(a.edit_estimate_seconds)), SIGNATURES['edit_estimate_seconds'])
        self.assertEqual(a.StallWatchdog(Mock(), 5, grace=3).grace, 3.0)

    def test_settings_are_read_from_the_config_directory_of_that_load(self):
        a, b = self.a, self.b
        a.PROJECT.mkdir(parents=True, exist_ok=True)
        (a.PROJECT / 'runtime-settings.json').write_text(json.dumps({'oxi': {'stall_seconds': 7, 'threads': 3}}))
        (a.PROJECT / 'engine.json').write_text(json.dumps({'engine': 'sdcpp'}))
        self.assertEqual((a.oxi_stall_seconds(), b.oxi_stall_seconds()), (7.0, 180.0))
        self.assertEqual((a.engine_setting(), b.engine_setting()), ('sdcpp', 'auto'))

    def test_the_script_loads_without_the_package_as_the_installed_unit_runs_it(self):
        """`python3 -I`: no PYTHONPATH, so the sibling modules come from the script's own directory."""
        program = (
            'import importlib.machinery, importlib.util, json, sys\n'
            'before = list(sys.path)\n'
            'loader = importlib.machinery.SourceFileLoader("flat_studio", sys.argv[1])\n'
            'module = importlib.util.module_from_spec(importlib.util.spec_from_loader(loader.name, loader))\n'
            'loader.exec_module(module)\n'
            'names = json.loads(sys.argv[2])\n'
            'print(json.dumps({"missing": sorted(set(names) - set(vars(module))), "path_restored": sys.path == before,\n'
            '                  "package": "snapdragon_ai_lab" in sys.modules,\n'
            '                  "choice": module.choose_t2i_engine("auto", "klein-q8", 512, 512, [])[0],\n'
            '                  "missing_files": module.oxi_missing()}))\n')
        environment = {key: value for key, value in os.environ.items() if key not in UNSET and key != 'PYTHONPATH'}
        environment.update({key: str(self.root / 'flat' / value) for key, value in ENVIRONMENT.items()})
        done = subprocess.run([sys.executable, '-I', '-c', program, str(SOURCE), json.dumps(NAMES)], env=environment,
                              capture_output=True, text=True, timeout=60)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(json.loads(done.stdout), {
            'missing': [], 'path_restored': True, 'package': False, 'choice': 'oxi',
            'missing_files': ['oxi', 'sd-cli (probe)', 'flux-2-klein-4b-Q8_0.gguf', 'Qwen3-4B-Q8_0.gguf',
                              'flux2-vae.safetensors']})


if __name__ == '__main__':
    unittest.main()
