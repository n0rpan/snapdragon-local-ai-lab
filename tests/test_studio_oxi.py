"""IMG-SHIP: FLUX.2 klein 512 px text-to-image on oxi, with sd-cli as the fallback (docs/STUDIO-OXI.md).

Model-free: oxi, the probe sd-cli and sd-cli are stubs at the run_engine boundary, except one test
that runs run_engine on a tiny shell script to check the hand-over stop and the prompt memfd.
No image model, engine binary, service or live endpoint is used.
"""
import importlib.machinery
import importlib.util
from http.server import ThreadingHTTPServer
import json
import os
from pathlib import Path
import struct
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from urllib.request import Request, urlopen

SOURCE = Path(__file__).resolve().parents[1] / 'src/snapdragon_ai_lab/runtime/studio'
PROMPT = 'a secret red bicycle'


def option(args, name):
    return args[args.index(name) + 1]


def fake_png(path, width, height):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ihdr = struct.pack('>II', width, height) + b'\x08\x02\x00\x00\x00'
    path.write_bytes(b'\x89PNG\r\n\x1a\n' + struct.pack('>I', 13) + b'IHDR' + ihdr + b'\x00' * 4)


def executable(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('#!/bin/sh\nexit 0\n')
    path.chmod(0o700)


class OxiCase(unittest.TestCase):
    install_oxi = True

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        environment = {'IMG_OUTDIR': str(self.root / 'output'), 'LOCAL_IMAGE_CONFIG_DIR': str(self.root / 'config'),
                       'LOCAL_AI_LEASE': str(self.root / 'inference.lock'), 'SD_HOME': str(self.root / 'stack')}
        with patch.dict(os.environ, environment):
            for key in ('OXI_BIN', 'SD_CLI_PROBE', 'OXI_COND_CACHE', 'OXI_WORK'):
                os.environ.pop(key, None)
            loader = importlib.machinery.SourceFileLoader('oxi_studio', str(SOURCE))
            spec = importlib.util.spec_from_loader(loader.name, loader)
            self.studio = importlib.util.module_from_spec(spec)
            loader.exec_module(self.studio)
        s = self.studio
        for path in (s.KLEIN_FILES['q8'], s.KLEIN_LLM, s.KLEIN_VAE, s.MODEL_FILES['q8'], s.LLM, s.VAE):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'stub')
        s.PROJECT.mkdir(parents=True, exist_ok=True)
        if self.install_oxi:
            executable(s.OXI_BIN)
            executable(s.SD_PROBE_CLI)

    def tearDown(self):
        self.temporary.cleanup()

    def plan(self, **body):
        return self.studio.plan_generation({'prompt': PROMPT, 'model': 'klein-q8', 'width': 512, 'height': 512, **body})


class Routing(OxiCase):
    def test_routing_table(self):
        route = self.studio.choose_t2i_engine
        self.assertEqual(route('auto', 'klein-q8', 512, 512, [])[0], 'oxi')
        self.assertEqual(route('oxi', 'klein-q8', 512, 512, [])[0], 'oxi')
        engine, reason = route('auto', 'klein-q8', 1024, 1024, [])
        self.assertEqual(engine, 'sdcpp')
        self.assertIn('slower', reason)
        self.assertIn('1024 × 1024', reason)
        for width, height in ((768, 768), (512, 768), (256, 256), (768, 512)):
            self.assertEqual(route('auto', 'klein-q8', width, height, [])[0], 'sdcpp', (width, height))
            self.assertEqual(route('oxi', 'klein-q8', width, height, [])[0], 'sdcpp', (width, height))
        self.assertEqual(route('auto', 'klein-q4', 512, 512, []), ('sdcpp', 'oxi runs FLUX.2 klein Q8 only; this model runs on sd.cpp'))
        self.assertEqual(route('sdcpp', 'klein-q8', 512, 512, [])[0], 'sdcpp')
        engine, reason = route('auto', 'klein-q8', 512, 512, ['oxi'])
        self.assertEqual(engine, 'sdcpp')
        self.assertIn('not installed', reason)

    def test_512_klein_q8_plans_one_oxi_step_with_the_sd_fallback_ready(self):
        plan = self.plan(seed=7)
        self.assertEqual(plan['engine']['engine'], 'oxi')
        self.assertEqual(len(plan['steps']), 1)
        step = plan['steps'][0]
        self.assertTrue(callable(step['run']))
        self.assertNotIn('args', step)
        self.assertEqual(step['expect_size'], (512, 512))
        self.assertEqual([r['backend'] for r in plan['results']], ['oxi'])
        self.assertEqual(plan['results'][0]['seed'], 7)

    def test_other_sizes_q4_and_the_sdcpp_setting_keep_the_sd_cli_plan(self):
        s = self.studio
        s.KLEIN_FILES['q4'].write_bytes(b'stub')
        for body in ({'width': 1024, 'height': 1024}, {'width': 768, 'height': 512}, {'model': 'klein-q4'}):
            plan = self.plan(**body)
            self.assertEqual(plan['steps'][0]['args'][0], str(s.SDCLI), body)
            self.assertEqual(plan['results'][0]['backend'], 'sd.cpp', body)
        s.save_engine_setting('sdcpp')
        plan = self.plan()
        self.assertEqual(plan['steps'][0]['args'][0], str(s.SDCLI))
        self.assertEqual(plan['engine'], {'setting': 'sdcpp', 'engine': 'sdcpp', 'reason': 'engine setting: sd.cpp'})

    def test_edit_and_z_image_flows_never_route_to_oxi(self):
        s = self.studio
        plan = s.plan_generation({'prompt': 'x', 'model': 'q8', 'width': 512, 'height': 512})
        self.assertNotIn('run', plan['steps'][0])
        self.assertNotIn(str(s.OXI_BIN), json.dumps([st.get('args') for st in plan['steps']]))

    def test_engine_setting_defaults_to_auto_and_persists(self):
        s = self.studio
        self.assertEqual(s.engine_setting(), 'auto')
        for value in ('oxi', 'sdcpp', 'auto'):
            s.save_engine_setting(value)
            self.assertEqual(s.engine_setting(), value)
        with self.assertRaises(s.StudioError):
            s.save_engine_setting('gpu')
        (s.PROJECT / 'engine.json').write_text('{not json')
        self.assertEqual(s.engine_setting(), 'auto')
        (s.PROJECT / 'engine.json').write_text(json.dumps({'engine': 'turbo'}))
        self.assertEqual(s.engine_setting(), 'auto')

    def test_oxi_argv_is_the_gated_fast_path_and_carries_no_prompt(self):
        s = self.studio
        spec = {'width': 512, 'height': 512}
        args = s.oxi_generate_args(spec, 43, s.OUT / 'x.png')
        self.assertEqual(args[:2], [str(s.OXI_BIN), 'generate'])
        for flag, value in (('--backend', 'fast'), ('--chunk1', '32'), ('--chunk2', '32'), ('--attn-prec', 'f32'),
                            ('--noise', 'sdcpp'), ('--vae-conv', 'f32'), ('--steps', '4'), ('--seed', '43'),
                            ('--gguf', str(s.KLEIN_FILES['q8'])), ('--vae', str(s.KLEIN_VAE)),
                            ('--width', '512'), ('--height', '512'), ('--threads', '12'),
                            ('--cache', str(s.OXI_COND_CACHE))):
            self.assertEqual(option(args, flag), value, flag)
        self.assertNotIn('--cond-prompt-file', args)          # added by run_engine from the memfd
        put = s.oxi_cond_put_args(spec, Path('/h'))
        self.assertEqual((option(put, '--sigmas'), option(put, '--steps'), option(put, '--handoff')), ('computed', '4', '/h'))
        self.assertNotIn(PROMPT, ' '.join(args + put))

    def test_oxi_threads_come_from_runtime_settings(self):
        s = self.studio
        (s.PROJECT / 'runtime-settings.json').write_text(json.dumps({'oxi': {'threads': 10}}))
        self.assertEqual(option(s.oxi_generate_args({'width': 512, 'height': 512}, 1, 'o.png'), '--threads'), '10')
        (s.PROJECT / 'runtime-settings.json').write_text(json.dumps({'oxi': {'threads': 0}}))
        self.assertEqual(s.oxi_threads(), 12)

    def test_oxi_vae_conv_comes_from_runtime_settings(self):
        s = self.studio
        args = lambda: s.oxi_generate_args({'width': 512, 'height': 512}, 1, 'o.png')
        self.assertEqual(option(args(), '--vae-conv'), 'f32')         # no settings file: gated default
        for value in ('winograd', 'int8', 'f32'):
            (s.PROJECT / 'runtime-settings.json').write_text(json.dumps({'oxi': {'threads': 12, 'vae_conv': value}}))
            self.assertEqual(option(args(), '--vae-conv'), value)
            self.assertEqual(args().count('--vae-conv'), 1)
        for bad in ('fp16', 7, None, ['int8']):
            (s.PROJECT / 'runtime-settings.json').write_text(json.dumps({'oxi': {'vae_conv': bad}}))
            self.assertEqual(s.oxi_vae_conv(), 'f32')
        (s.PROJECT / 'runtime-settings.json').write_text(json.dumps({'oxi': {'threads': 12}}))   # key removed = rollback
        self.assertEqual(option(args(), '--vae-conv'), 'f32')

    def test_vae_selection_preserves_prompt_cache_opt_in(self):
        s = self.studio
        settings = s.PROJECT / 'runtime-settings.json'
        settings.write_text(json.dumps({'oxi': {'vae_conv': 'int8'}}))
        self.assertFalse(s.oxi_prompt_cache_enabled())
        private_cache = Path('/private/job-cache')
        args = s.oxi_generate_args({'width': 512, 'height': 512}, 1, 'o.png', cache_dir=private_cache)
        self.assertEqual(option(args, '--cache'), str(private_cache))
        self.assertEqual(option(args, '--vae-conv'), 'int8')
        self.assertEqual(args.count('--vae-conv'), 1)
        settings.write_text(json.dumps({'oxi': {'vae_conv': 'int8', 'prompt_cache': True}}))
        self.assertTrue(s.oxi_prompt_cache_enabled())
        self.assertEqual(s.oxi_vae_conv(), 'int8')

    def test_handover_stop_waits_for_the_line_after_the_rope_dump(self):
        stop = self.studio.handover_stop()
        lines = ['[I] get_learned_condition completed, taking 4.27s',
                 'SDPROBE dump.flux.txt call=0 t_ns=1 type=f32 ne=7680x512x1x1',
                 'SDPROBE dump.pe call=0 t_ns=2 type=f32 ne=2x2x64x1536',
                 '[I] some log line']
        self.assertEqual([stop(line) for line in lines], [False, False, False, False])
        self.assertTrue(stop('SDPROBE time.double.0.in call=0 t_ns=3 type=f32 ne=3072x1024x1x1'))
        # a later call's pe line does not arm it
        stop = self.studio.handover_stop()
        self.assertFalse(stop('SDPROBE dump.pe call=1 t_ns=2 type=f32 ne=2x2x64x1536'))
        self.assertFalse(stop('SDPROBE dump.single.0.in call=1 t_ns=3 type=f32 ne=3072x1536x1x1'))


class NotInstalled(OxiCase):
    install_oxi = False

    def test_missing_oxi_falls_back_to_the_sd_cli_plan_at_512(self):
        s = self.studio
        self.assertEqual(s.oxi_missing(), ['oxi', 'sd-cli (probe)'])
        plan = self.plan()
        self.assertEqual(plan['steps'][0]['args'][0], str(s.SDCLI))
        self.assertIn('not installed', plan['engine']['reason'])
        s.save_engine_setting('oxi')
        self.assertEqual(self.plan()['engine']['engine'], 'sdcpp')

    def test_a_non_executable_oxi_is_not_ready(self):
        s = self.studio
        s.OXI_BIN.parent.mkdir(parents=True, exist_ok=True)
        s.OXI_BIN.write_text('x')
        s.OXI_BIN.chmod(0o600)
        self.assertIn('oxi', s.oxi_missing())


class FakeEngines:
    """Stands in for run_engine: oxi generate / cond-put, the probe sd-cli and sd-cli."""

    def __init__(self, studio, cached=False, oxi_rc=0, oxi_fail_at=None, oxi_size=(512, 512),
                 handover_tags=None, put_rc=0):
        self.studio = studio
        self.calls = []
        self.cached = cached
        self.cached_paths = {str(studio.OXI_COND_CACHE)} if cached else set()
        self.oxi_rc = oxi_rc
        self.oxi_fail_at = oxi_fail_at
        self.oxi_size = oxi_size
        self.handover_tags = handover_tags
        self.put_rc = put_rc
        self.generated = 0
        self.handover_dirs = []

    def __call__(self, args, job, environment=None, prompt=None, prompt_flag='--prompt-file', stop=None):
        s = self.studio
        args = list(args)
        call = {'args': args, 'prompt': prompt, 'flag': prompt_flag, 'env': environment, 'stop': stop}
        self.calls.append(call)
        if args[0] == str(s.OXI_BIN) and args[1] == 'generate':
            call['kind'] = 'oxi-generate'
            if option(args, '--cache') not in self.cached_paths:
                return s.OXI_NO_CONDITIONING
            if self.oxi_fail_at is not None and self.generated == self.oxi_fail_at:
                return self.oxi_rc or 1
            self.generated += 1
            fake_png(option(args, '--out'), *self.oxi_size)
            job['state'] = 'decoding'
            return 0
        if args[0] == str(s.OXI_BIN) and args[1] == 'cond-put':
            call['kind'] = 'oxi-cond-put'
            handover = Path(option(args, '--handoff'))
            call['handover_files'] = sorted(p.name for p in handover.iterdir())
            if self.put_rc == 0:
                self.cached = True
                cache = Path(option(args, '--cache'))
                cache.mkdir(parents=True, exist_ok=True, mode=0o700)
                (cache / 'stub-conditioning').mkdir(exist_ok=True, mode=0o700)
                self.cached_paths.add(str(cache))
            return self.put_rc
        if args[0] == str(s.SD_PROBE_CLI):
            call['kind'] = 'probe'
            dump = Path(environment['SD_PROBE_DUMP_DIR'])
            self.handover_dirs.append(dump.parent)
            tags = self.handover_tags if self.handover_tags is not None else s.HANDOVER_TAGS + ('dump.single.0.in',)
            for tag in tags:
                (dump / f'{tag}.call0.f32.4x4x1x1.bin').write_bytes(b'\0' * 64)
            return -15                                  # terminated by the stop predicate
        call['kind'] = 'sd-cli'
        out = option(args, '-o')
        count = int(option(args, '-b')) if '-b' in args else 1
        for i in range(count):
            fake_png(out.replace('%d', str(i)), int(option(args, '-W')), int(option(args, '-H')))
        return 0

    def kinds(self):
        return [c['kind'] for c in self.calls]


class Pipeline(OxiCase):
    def setUp(self):
        super().setUp()
        # These reproduce the qualified persistent-cache pipeline; public OFF is below.
        (self.studio.PROJECT / 'runtime-settings.json').write_text(json.dumps({'oxi': {'prompt_cache': True}}))

    def run_plan(self, fake, **body):
        s = self.studio
        plan = self.plan(**body)
        job_id = 'job'
        s.JOBS[job_id] = {'state': 'queued'}
        with patch.object(s, 'run_engine', side_effect=fake):
            s.run_pipeline(job_id, plan['steps'], open(self.root / 'lease', 'a'), plan['environment'],
                           plan['prompt'], plan['results'])
        return s.JOBS[job_id], plan

    def test_cached_prompt_runs_only_oxi(self):
        fake = FakeEngines(self.studio, cached=True)
        job, _ = self.run_plan(fake, seed=5)
        self.assertEqual(job['state'], 'done', job)
        self.assertEqual(fake.kinds(), ['oxi-generate'])
        call = fake.calls[0]
        self.assertEqual((call['prompt'], call['flag']), (PROMPT, '--cond-prompt-file'))
        self.assertEqual(option(call['args'], '--seed'), '5')
        self.assertEqual(job['engine'], 'oxi')
        self.assertEqual(job['images'][0]['backend'], 'oxi')
        self.assertNotIn(PROMPT, json.dumps(job))

    def test_new_prompt_runs_the_handover_then_cond_put_then_oxi(self):
        s = self.studio
        fake = FakeEngines(s)
        job, _ = self.run_plan(fake)
        self.assertEqual(job['state'], 'done', job)
        self.assertEqual(fake.kinds(), ['oxi-generate', 'probe', 'oxi-cond-put', 'oxi-generate'])
        probe = fake.calls[1]
        self.assertEqual(probe['env']['SD_PROBE'], '1')
        self.assertEqual(probe['env']['SD_PROBE_DUMP_CALL'], '0')
        self.assertIsNotNone(probe['stop'])
        self.assertEqual(probe['prompt'], PROMPT)
        self.assertEqual(option(probe['args'], '--diffusion-model'), str(s.KLEIN_FILES['q8']))
        self.assertEqual((option(probe['args'], '-W'), option(probe['args'], '-H')), ('512', '512'))
        put = fake.calls[2]
        self.assertEqual((put['prompt'], put['flag']), (PROMPT, '--prompt-file'))
        # only the four tags oxi needs reach cond-put, whatever else the probe dumped
        self.assertEqual(len(put['handover_files']), 4)
        self.assertFalse(any('single' in name for name in put['handover_files']))
        # the work directory with the dumps is gone
        self.assertTrue(fake.handover_dirs)
        self.assertFalse(any(d.exists() for d in fake.handover_dirs))
        self.assertEqual(job['images'][0]['backend'], 'oxi')

    def test_variations_encode_once_and_use_consecutive_sdcpp_seeds(self):
        fake = FakeEngines(self.studio)
        job, plan = self.run_plan(fake, variations=3, seed=10)
        self.assertEqual(job['state'], 'done', job)
        self.assertEqual(fake.kinds(), ['oxi-generate', 'probe', 'oxi-cond-put', 'oxi-generate',
                                        'oxi-generate', 'oxi-generate'])
        seeds = [option(c['args'], '--seed') for c in fake.calls if c['kind'] == 'oxi-generate']
        self.assertEqual(seeds, ['10', '10', '11', '12'])
        self.assertEqual([im['seed'] for im in job['images']], [10, 11, 12])
        self.assertEqual(len({im['path'] for im in job['images']}), 3)

    def test_auto_falls_back_to_sd_cli_on_an_oxi_failure_and_drops_partial_images(self):
        s = self.studio
        fake = FakeEngines(s, cached=True, oxi_fail_at=1)
        job, plan = self.run_plan(fake, variations=2, seed=3)
        self.assertEqual(job['state'], 'done', job)
        self.assertEqual(fake.kinds(), ['oxi-generate', 'oxi-generate', 'sd-cli'])
        sd = fake.calls[-1]
        self.assertEqual(sd['args'][0], str(s.SDCLI))
        self.assertEqual((option(sd['args'], '--seed'), option(sd['args'], '-b')), ('3', '2'))
        self.assertEqual(sd['prompt'], PROMPT)
        self.assertEqual(job['engine'], 'sd.cpp')
        self.assertIn('oxi failed (oxi exited 1)', job['engine_note'])
        self.assertEqual([im['backend'] for im in job['images']], ['sd.cpp', 'sd.cpp'])

    def test_auto_falls_back_on_an_incomplete_handover(self):
        fake = FakeEngines(self.studio, handover_tags=('dump.flux.txt',))
        job, _ = self.run_plan(fake)
        self.assertEqual(job['state'], 'done', job)
        self.assertEqual(fake.kinds(), ['oxi-generate', 'probe', 'sd-cli'])
        self.assertIn('incomplete', job['engine_note'])

    def test_auto_falls_back_when_cond_put_fails(self):
        fake = FakeEngines(self.studio, put_rc=1)
        job, _ = self.run_plan(fake)
        self.assertEqual(job['state'], 'done', job)
        self.assertEqual(fake.kinds(), ['oxi-generate', 'probe', 'oxi-cond-put', 'sd-cli'])
        self.assertIn('cond-put exited 1', job['engine_note'])

    def test_auto_falls_back_on_a_wrong_size_image(self):
        fake = FakeEngines(self.studio, cached=True, oxi_size=(256, 256))
        job, _ = self.run_plan(fake)
        self.assertEqual(job['state'], 'done', job)
        self.assertEqual(fake.kinds(), ['oxi-generate', 'sd-cli'])
        self.assertEqual(self.studio.png_size(job['images'][0]['path']), (512, 512))

    def test_auto_falls_back_when_oxi_raises(self):
        s = self.studio
        fake = FakeEngines(s, cached=True)

        def engine(args, job, *rest, **kw):
            if args[0] == str(s.OXI_BIN):
                raise OSError('exec format error')
            return fake(args, job, *rest, **kw)
        job, _ = self.run_plan(engine)
        self.assertEqual(job['state'], 'done', job)
        self.assertIn('exec format error', job['engine_note'])

    def test_oxi_setting_reports_the_failure_without_sd_cli(self):
        s = self.studio
        s.save_engine_setting('oxi')
        fake = FakeEngines(s, cached=True, oxi_fail_at=1)
        job, _ = self.run_plan(fake, variations=2)
        self.assertEqual(job['state'], 'error', job)
        self.assertEqual(fake.kinds(), ['oxi-generate', 'oxi-generate'])
        self.assertIn('oxi failed', job['error'])
        self.assertNotIn('sd-cli', fake.kinds())
        self.assertFalse(any(s.OUT.glob('*.png')))

    def test_sd_cli_failure_after_fallback_is_the_job_error(self):
        s = self.studio
        fake = FakeEngines(s, cached=True, oxi_fail_at=0)

        def engine(args, job, *rest, **kw):
            if args[0] == str(s.SDCLI):
                fake.calls.append({'kind': 'sd-cli'})
                return 2
            return fake(args, job, *rest, **kw)
        job, _ = self.run_plan(engine)
        self.assertEqual(job['state'], 'error')
        self.assertIn('sd-cli exited 2', job['error'])

    def test_cond_cache_keeps_the_newest_entries(self):
        s = self.studio
        for i in range(5):
            d = s.OXI_COND_CACHE / f'k{i}'
            d.mkdir(parents=True)
            os.utime(d, (1000 + i, 1000 + i))
        s.prune_cond_cache(keep=2)
        self.assertEqual(sorted(p.name for p in s.OXI_COND_CACHE.iterdir()), ['k3', 'k4'])


class PromptCachePolicy(OxiCase):
    run_plan = Pipeline.run_plan

    def test_default_off_uses_private_per_job_cache_and_never_reuses_persistent_entries(self):
        s = self.studio
        s.OXI_COND_CACHE.mkdir(parents=True, mode=0o700)
        sentinel = s.OXI_COND_CACHE / 'persistent-sentinel'; sentinel.write_bytes(b'keep')
        fake = FakeEngines(s, cached=True)
        seen = []

        def engine(args, job, environment=None, *rest, **kw):
            if args[0] == str(s.OXI_BIN):
                cache = Path(option(args, '--cache'))
                self.assertNotEqual(cache, s.OXI_COND_CACHE)
                self.assertEqual(cache.stat().st_mode & 0o777, 0o700)
                self.assertEqual(environment['OXI_COND_CACHE'], str(cache))
                seen.append(cache)
            elif args[0] == str(s.SD_PROBE_CLI):
                self.assertEqual(environment['OXI_COND_CACHE'], str(seen[-1]))
            return fake(args, job, environment, *rest, **kw)

        with patch.dict(os.environ, {'OXI_COND_CACHE': str(s.OXI_COND_CACHE)}):
            for _ in range(2):
                job, _ = self.run_plan(engine, variations=2)
                self.assertEqual(job['state'], 'done', job)
        self.assertEqual(fake.kinds().count('probe'), 2)
        self.assertEqual(len(set(seen)), 2)  # same directory within a job, fresh across jobs
        self.assertTrue(all(not cache.exists() for cache in seen))
        self.assertEqual(list(s.OXI_COND_CACHE.iterdir()), [sentinel])
        self.assertEqual(sentinel.read_bytes(), b'keep')

    def test_off_also_cleans_cache_when_cond_put_fails(self):
        s = self.studio
        fake = FakeEngines(s, put_rc=1)
        s.save_engine_setting('oxi')
        job, _ = self.run_plan(fake)
        self.assertEqual(job['state'], 'error')
        caches = [Path(option(c['args'], '--cache')) for c in fake.calls if c['kind'].startswith('oxi-')]
        self.assertTrue(caches)
        self.assertTrue(all(not cache.exists() for cache in caches))
        self.assertFalse(s.OXI_COND_CACHE.exists())

    def test_only_explicit_boolean_true_enables_persistence(self):
        s = self.studio
        for value in (False, None, 1, 'true', [], {}):
            with self.subTest(value=value):
                (s.PROJECT / 'runtime-settings.json').write_text(json.dumps({'oxi': {'prompt_cache': value}}))
                self.assertFalse(s.oxi_prompt_cache_enabled())
        (s.PROJECT / 'runtime-settings.json').write_text('{bad json')
        self.assertFalse(s.oxi_prompt_cache_enabled())
        (s.PROJECT / 'runtime-settings.json').write_text(json.dumps({'oxi': {'prompt_cache': True}}))
        self.assertTrue(s.oxi_prompt_cache_enabled())

    def test_opt_in_preserves_persistent_warm_hit_and_last_sixteen(self):
        s = self.studio
        (s.PROJECT / 'runtime-settings.json').write_text(json.dumps({'oxi': {'prompt_cache': True}}))
        s.OXI_COND_CACHE.mkdir(parents=True, mode=0o700)
        for i in range(18):
            entry = s.OXI_COND_CACHE / ('entry-%02d' % i); entry.mkdir(mode=0o700)
            os.utime(entry, (1000 + i, 1000 + i))
        fake = FakeEngines(s)
        with patch.object(s.tempfile, 'TemporaryDirectory', side_effect=AssertionError('ON keeps native persistent path')):
            self.assertEqual(self.run_plan(fake)[0]['state'], 'done')
            self.assertEqual(self.run_plan(fake)[0]['state'], 'done')
        self.assertEqual(fake.kinds().count('probe'), 1)
        self.assertEqual(len(list(s.OXI_COND_CACHE.iterdir())), 16)
        self.assertEqual(s.OXI_COND_CACHE.stat().st_mode & 0o777, 0o700)
        for call in fake.calls:
            if call['kind'].startswith('oxi-'):
                self.assertEqual(option(call['args'], '--cache'), str(s.OXI_COND_CACHE))
                self.assertIsNone(call['env'])  # qualified ON path stays unchanged


class RealSubprocess(OxiCase):
    """run_engine itself on a shell script: the stop predicate ends a hanging probe, and the prompt
    reaches the child only through the memfd named by prompt_flag."""

    def test_stop_terminates_and_prompt_flag_is_used(self):
        s = self.studio
        script = self.root / 'fake-probe.sh'
        seen = self.root / 'seen.txt'
        script.write_text('#!/bin/sh\n'
                          'while [ $# -gt 0 ]; do if [ "$1" = "--cond-prompt-file" ]; then cat "$2" > "%s"; fi; shift; done\n'
                          'echo "SDPROBE dump.flux.txt call=0 t_ns=1 type=f32 ne=1x1x1x1"\n'
                          'echo "SDPROBE dump.pe call=0 t_ns=2 type=f32 ne=1x1x1x1"\n'
                          'echo "SDPROBE time.double.0.in call=0 t_ns=3 type=f32 ne=1x1x1x1"\n'
                          'exec sleep 30\n' % seen)
        script.chmod(0o700)
        job = {}
        started = time.monotonic()
        rc = s.run_engine([str(script)], job, None, PROMPT, prompt_flag='--cond-prompt-file', stop=s.handover_stop())
        self.assertLess(time.monotonic() - started, 10)
        self.assertNotEqual(rc, 0)
        self.assertEqual(seen.read_text(), PROMPT)


class Endpoints(OxiCase):
    def setUp(self):
        super().setUp()
        self.fake = FakeEngines(self.studio, cached=True)
        self.engine = patch.object(self.studio, 'run_engine', side_effect=self.fake)
        self.engine.start()
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), self.studio.H)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(2)
        self.engine.stop()
        super().tearDown()

    def request(self, path, body=None):
        url = f'http://127.0.0.1:{self.server.server_port}{path}'
        data = None if body is None else json.dumps(body).encode()
        try:
            with urlopen(Request(url, data=data), timeout=10) as response:
                return json.loads(response.read())
        except Exception as error:
            if hasattr(error, 'read'):
                with error:
                    return json.loads(error.read())
            raise

    def finish(self, body):
        started = self.request('/generate', body)
        self.assertIn('id', started, started)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            job = self.request('/job?id=' + started['id'])
            if job['state'] in ('done', 'error'):
                return job
            time.sleep(0.02)
        self.fail('job did not finish')

    def test_engine_setting_endpoints(self):
        info = self.request('/engine')
        self.assertEqual((info['engine'], info['oxi_ready'], info['choices']), ('auto', True, ['auto', 'oxi', 'sdcpp']))
        self.assertEqual(self.request('/engine', {'engine': 'sdcpp'})['engine'], 'sdcpp')
        self.assertEqual(self.request('/engine')['engine'], 'sdcpp')
        self.assertIn('error', self.request('/engine', {'engine': 'cuda'}))
        self.assertEqual(self.request('/models')['engine']['engine'], 'sdcpp')

    def test_estimate_names_the_engine(self):
        at512 = self.request('/estimate?model=klein-q8&width=512&height=512&variations=1')
        self.assertEqual(at512['engine'], 'oxi')
        self.assertEqual(at512['seconds'], self.studio.oxi_estimate_seconds(1))
        self.assertFalse(at512['calibrated'])
        at1024 = self.request('/estimate?model=klein-q8&width=1024&height=1024&variations=1')
        self.assertEqual(at1024['engine'], 'sd.cpp')
        self.assertIn('slower', at1024['engine_reason'])

    def test_generate_512_on_oxi_and_1024_on_sd_cli(self):
        job = self.finish({'prompt': PROMPT, 'model': 'klein-q8', 'width': 512, 'height': 512})
        self.assertEqual((job['state'], job['engine'], job['images'][0]['backend']), ('done', 'oxi', 'oxi'), job)
        self.assertNotIn(PROMPT, json.dumps(job))
        job = self.finish({'prompt': PROMPT, 'model': 'klein-q8', 'width': 1024, 'height': 1024})
        self.assertEqual((job['state'], job['engine']), ('done', 'sd.cpp'), job)
        self.assertIn('slower', job['engine_reason'])
        # Public default OFF must encode even when persistent conditioning was primed.
        self.assertEqual(self.fake.kinds(), ['oxi-generate', 'probe', 'oxi-cond-put', 'oxi-generate', 'sd-cli'])

    def test_engine_post_is_refused_cross_site(self):
        url = f'http://127.0.0.1:{self.server.server_port}/engine'
        request = Request(url, data=json.dumps({'engine': 'sdcpp'}).encode(), headers={'Origin': 'http://evil.example'})
        with self.assertRaises(Exception) as caught:
            urlopen(request, timeout=10)
        self.assertEqual(caught.exception.code, 403)
        caught.exception.close()
        self.assertEqual(self.request('/engine')['engine'], 'auto')


if __name__ == '__main__':
    unittest.main()
