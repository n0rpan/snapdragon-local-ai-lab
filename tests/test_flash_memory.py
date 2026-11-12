"""Flash-only memory policy: own 29 GiB scope and 1 GiB floor; synthetic, starts nothing."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from snapdragon_ai_lab.runtime import bigmoe_runtime, controller, exec_in_scope, memory_policy

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('stage_flash_memory', ROOT / 'tools/stage_flash_memory.py')
stage_tool = importlib.util.module_from_spec(spec); spec.loader.exec_module(stage_tool)

GIB = 1024**3
POLICY = {'scope_max_bytes': 29 * GIB, 'host_floor_bytes': GIB}
UNIT = '''[Service]
WorkingDirectory=/synthetic/releases/old
Environment="LOCAL_LLM_CONFIG=/synthetic/runtime.json"
Environment="LOCAL_AI_ENABLE_OXE=1"
ExecStart=/usr/bin/python3 "/synthetic/releases/old/controller.py"
MemoryMax=28G
MemorySwapMax=0
'''


def sample(**changes):
    value = {'available_bytes': 4 * GIB, 'psi_full_avg10': 0., 'psi_full_total_us': 0,
             'swap_current_bytes': 0, 'swap_max': '0', 'memory_high': 'max', 'memory_max': str(28 * GIB),
             'events': {'oom': 0, 'oom_kill': 0, 'oom_group_kill': 0}, 'pids_max_events': 0}
    value.update(changes)
    return value


class Policy(unittest.TestCase):
    def test_absent_means_unchanged_and_only_the_measured_pair_is_accepted(self):
        self.assertIsNone(memory_policy.flash_policy({}))
        self.assertEqual(memory_policy.flash_policy({'flash_memory': dict(POLICY)}), POLICY)
        for bad in ({'scope_max_bytes': 30 * GIB, 'host_floor_bytes': GIB}, {'scope_max_bytes': 29 * GIB},
                    {**POLICY, 'host_floor_bytes': GIB // 2}, {**POLICY, 'extra': 1}, 'yes',
                    {'scope_max_bytes': 29.0 * GIB, 'host_floor_bytes': GIB}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                memory_policy.flash_policy({'flash_memory': bad})

    def test_floor_is_a_parameter_of_the_same_rule(self):
        low = sample(available_bytes=GIB + 4096)
        baseline = memory_policy.event_baseline(low)
        self.assertEqual(memory_policy.evaluate_sample(low, 0., None, baseline)[0], 'host_available_below_floor')
        self.assertIsNone(memory_policy.evaluate_sample(low, 0., None, baseline, GIB)[0])
        self.assertEqual(memory_policy.evaluate_sample(sample(available_bytes=GIB - 1), 0., None, baseline, GIB)[0],
                         'host_available_below_floor')

    def test_guard_floor_bounds_and_metadata(self):
        guard = memory_policy.MemoryGuard(reader=lambda: sample(available_bytes=GIB + 1), minimum_available_bytes=GIB)
        self.assertIsNone(guard.failure())
        self.assertEqual(guard.metadata()['policy']['minimum_available_bytes'], GIB)
        default = memory_policy.MemoryGuard(reader=sample)
        self.assertEqual(default.metadata()['policy']['minimum_available_bytes'], memory_policy.MIN_AVAILABLE_BYTES)
        for bad in (GIB - 1, 2 * GIB, 1.0 * GIB, True):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                memory_policy.MemoryGuard(reader=sample, minimum_available_bytes=bad)

    def test_scope_command_wraps_without_touching_the_arguments(self):
        args = ['/synthetic/server', '--port', '8094']
        command = memory_policy.scope_command(args, 29 * GIB, runtime_dir='/run/user/1000')
        self.assertEqual(command[:3], [sys.executable, '-I', '-B'])
        self.assertTrue(command[3].endswith('/exec_in_scope.py'))
        self.assertEqual(command[4:], ['--memory-max', str(29 * GIB), '--runtime-dir', '/run/user/1000', '--', *args])
        with self.assertRaises(ValueError):
            memory_policy.scope_command([], 29 * GIB)


class FollowedCgroup(unittest.TestCase):
    """The Flash child's own scope is sampled once it has moved; its counters start at zero."""
    def test_baseline_resets_when_the_child_enters_its_scope(self):
        home, scope = Path('/sys/fs/cgroup/unit'), Path('/sys/fs/cgroup/app.slice/local-ai-flash-7.scope')
        seen = []
        samples = {home: sample(events={'oom': 3, 'oom_kill': 3, 'oom_group_kill': 0}),
                   scope: sample(memory_max=str(29 * GIB))}

        def read(cgroup):
            seen.append(cgroup)
            return copy.deepcopy(samples[cgroup])
        where = {'cgroup': home}
        with patch.object(memory_policy, 'current_cgroup', return_value=home), \
             patch.object(memory_policy, 'read_sample', side_effect=read), \
             patch.object(memory_policy, 'process_cgroup', side_effect=lambda pid: where['cgroup']):
            guard = memory_policy.MemoryGuard(follow_process_cgroup=True, minimum_available_bytes=GIB)
            guard._process = Mock(pid=7)
            self.assertIsNone(guard._observe(guard._read_followed()))
            self.assertIsNone(guard.metadata()['cgroup'])
            where['cgroup'] = scope
            self.assertIsNone(guard._observe(guard._read_followed()))
            self.assertEqual(guard.metadata()['cgroup'], scope.name)
            self.assertEqual(guard.metadata()['memory_max'], str(29 * GIB))
            # one OOM kill inside the scope is seen although the unit's old count was 3
            samples[scope]['events']['oom_kill'] = 1
            self.assertEqual(guard._observe(guard._read_followed()), 'cgroup_oom')
        self.assertEqual(seen, [home, home, scope, scope])

    def test_unreadable_child_cgroup_falls_back_instead_of_failing(self):
        home = Path('/sys/fs/cgroup/unit')
        with patch.object(memory_policy, 'current_cgroup', return_value=home), \
             patch.object(memory_policy, 'read_sample', return_value=sample()), \
             patch.object(memory_policy, 'process_cgroup', side_effect=OSError()):
            guard = memory_policy.MemoryGuard(follow_process_cgroup=True)
            guard._process = Mock(pid=7)
            self.assertIsNone(guard._observe(guard._read_followed()))


class BigMoeGuard(unittest.TestCase):
    def power(self):
        return {'external_online': True, 'battery_status': 'Charging', 'battery_percent': 80.0}

    def test_reserve_is_four_gib_unless_the_flash_floor_is_given(self):
        process = Mock(pid=1)
        stock = bigmoe_runtime.BackendGuard(process, lambda: 3 * GIB, power=self.power)
        self.assertEqual(stock.check(), 'flash_available_memory_below_reserve')
        flash = bigmoe_runtime.BackendGuard(process, lambda: GIB, power=self.power, reserve_bytes=GIB)
        self.assertIsNone(flash.check())
        flash.available = lambda: GIB - 1
        self.assertEqual(flash.check(), 'flash_available_memory_below_reserve')
        for bad in (GIB - 1, 5 * GIB, 1.5 * GIB):
            with self.subTest(bad=bad), self.assertRaises(Exception):
                bigmoe_runtime.BackendGuard(process, lambda: GIB, reserve_bytes=bad)


class ScopeWrapper(unittest.TestCase):
    def test_bus_call_adopts_one_pid_with_both_limits(self):
        call = exec_in_scope.scope_call('local-ai-flash-9.scope', 9, 29 * GIB)
        self.assertEqual(call[:4], ['/usr/bin/busctl', '--user', '--quiet', 'call'])
        self.assertIn('StartTransientUnit', call)
        tail = call[call.index('ssa(sv)a(sa(sv))') + 1:]
        self.assertEqual(tail[:3], ['local-ai-flash-9.scope', 'fail', '4'])
        self.assertEqual(tail[tail.index('PIDs'):tail.index('PIDs') + 4], ['PIDs', 'au', '1', '9'])
        self.assertEqual(tail[tail.index('MemoryMax'):tail.index('MemoryMax') + 3], ['MemoryMax', 't', str(29 * GIB)])
        self.assertEqual(tail[-4:], ['MemorySwapMax', 't', '0', '0'])

    def scope(self, root, name, memory_max, swap='0'):
        cgroup = Path(root) / name
        cgroup.mkdir()
        (cgroup / 'memory.max').write_text(memory_max + '\n'); (cgroup / 'memory.swap.max').write_text(swap + '\n')
        return cgroup

    def test_enters_only_when_its_own_cgroup_shows_both_limits(self):
        name = 'local-ai-flash-%d.scope' % os.getpid()
        with tempfile.TemporaryDirectory() as root:
            good = self.scope(root, name, str(29 * GIB))
            run = Mock(return_value=Mock(returncode=0))
            self.assertTrue(exec_in_scope.enter_scope(29 * GIB, '/run/user/1000', run=run, cgroup=lambda: good))
            environment = run.call_args.kwargs['env']
            self.assertEqual(environment, {'PATH': '/usr/bin:/bin', 'XDG_RUNTIME_DIR': '/run/user/1000'})
            ticks = iter(range(100))
            fast = dict(clock=lambda: next(ticks), sleep=lambda _: None, timeout=3)
            wrong = self.scope(root, 'other.scope', str(29 * GIB))
            self.assertFalse(exec_in_scope.enter_scope(29 * GIB, '/run/user/1000', run=run, cgroup=lambda: wrong, **fast))
        with tempfile.TemporaryDirectory() as root:
            loose = self.scope(root, name, 'max')
            self.assertFalse(exec_in_scope.enter_scope(29 * GIB, '/run/user/1000', run=run, cgroup=lambda: loose, **fast))
        with tempfile.TemporaryDirectory() as root:
            swapping = self.scope(root, name, str(29 * GIB), swap='max')
            self.assertFalse(exec_in_scope.enter_scope(29 * GIB, '/run/user/1000', run=run, cgroup=lambda: swapping, **fast))
        refused = Mock(return_value=Mock(returncode=1))
        self.assertFalse(exec_in_scope.enter_scope(29 * GIB, '/run/user/1000', run=refused, cgroup=lambda: good))
        hung = Mock(side_effect=subprocess.TimeoutExpired('busctl', 5))
        self.assertFalse(exec_in_scope.enter_scope(29 * GIB, '/run/user/1000', run=hung, cgroup=lambda: good))

    def test_main_fails_closed_before_any_exec(self):
        runtime = '/run/user/%d' % os.getuid()
        with patch.object(exec_in_scope.os, 'execv') as execv, \
             patch.object(exec_in_scope, 'enter_scope', return_value=False) as enter:
            for argv in (['--memory-max', str(29 * GIB), '--runtime-dir', runtime, '--', 'relative/server'],
                         ['--memory-max', str(30 * GIB), '--runtime-dir', runtime, '--', '/synthetic/server'],
                         ['--memory-max', str(29 * GIB), '--runtime-dir', '/tmp/elsewhere', '--', '/synthetic/server'],
                         ['--memory-max', str(29 * GIB), '--runtime-dir', runtime]):
                with self.subTest(argv=argv):
                    self.assertEqual(exec_in_scope.main(argv), 78)
            enter.assert_not_called()
            self.assertEqual(exec_in_scope.main(['--memory-max', str(29 * GIB), '--runtime-dir', runtime, '--',
                                                 '/synthetic/server', '--port', '1']), 78)
            enter.assert_called_once_with(29 * GIB, runtime)
            execv.assert_not_called()
        with patch.object(exec_in_scope.os, 'execv') as execv, patch.object(exec_in_scope, 'enter_scope', return_value=True):
            exec_in_scope.main(['--memory-max', str(29 * GIB), '--runtime-dir', runtime, '--', '/synthetic/server', '-x'])
            execv.assert_called_once_with('/synthetic/server', ['/synthetic/server', '-x'])


def runtime_json():
    flash = {'ready': True, 'backend_kind': bigmoe_runtime.KIND, 'model_path': '/synthetic/flash-1.gguf',
             'bigmoe': {'cache_mib': 12288}}
    return {'model_alias': 'qwen-local', 'selected_model': 'ornith-35b-a3b-local',
            'profiles': {'ornith-35b-a3b-local': {'ready': True, 'model_path': '/synthetic/ornith.gguf'},
                         stage_tool.ALIAS: flash}}


class StageTool(unittest.TestCase):
    def test_only_the_flash_profile_gains_the_policy(self):
        original = runtime_json(); before = copy.deepcopy(original)
        updated = stage_tool.add_flash_memory(original)
        self.assertEqual(original, before)
        self.assertEqual(updated['profiles'][stage_tool.ALIAS].pop('flash_memory'), POLICY)
        self.assertEqual(updated, before)
        with self.assertRaises(ValueError):
            stage_tool.add_flash_memory(stage_tool.add_flash_memory(original))
        classic = runtime_json(); classic['profiles'][stage_tool.ALIAS]['backend_kind'] = 'llama'
        with self.assertRaises(ValueError):
            stage_tool.add_flash_memory(classic)
        other = runtime_json(); other['profiles']['ornith-35b-a3b-local']['flash_memory'] = dict(POLICY)
        with self.assertRaises(ValueError):
            stage_tool.add_flash_memory(other)

    def test_unit_keeps_every_line_but_the_release_path(self):
        unit = stage_tool.rewrite_unit(UNIT, '/synthetic/releases/new')
        changed = [(a, b) for a, b in zip(UNIT.splitlines(), unit.splitlines()) if a != b]
        self.assertEqual([a.split('=')[0] for a, _ in changed], ['WorkingDirectory', 'ExecStart'])
        self.assertIn('MemoryMax=28G', unit.splitlines())
        self.assertIn('Environment="LOCAL_AI_ENABLE_OXE=1"', unit.splitlines())
        for bad in (UNIT.replace('MemoryMax=28G\n', ''), UNIT + 'MemoryMax=29G\n', UNIT.replace('ExecStart=', 'Exec=')):
            with self.subTest(bad=bad[-40:]), self.assertRaises(ValueError):
                stage_tool.rewrite_unit(bad, '/synthetic/releases/new')

    def test_stage_is_fresh_bound_and_rolls_back_exactly(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            runtime_raw = (json.dumps(runtime_json(), indent=2) + '\n').encode()
            (root / 'runtime.json').write_bytes(runtime_raw); (root / 'unit').write_text(UNIT)
            arguments = dict(runtime_path=root / 'runtime.json', runtime_sha=stage_tool.sha(runtime_raw),
                             unit_path=root / 'unit', unit_sha=stage_tool.sha(UNIT.encode()),
                             releases_root=root / 'releases')
            result = stage_tool.stage(root / 'stage', **arguments)
            stage = root / 'stage'
            self.assertEqual((result['installed'], result['models_started'], result['unit_limit_unchanged']),
                             (False, False, 'MemoryMax=28G'))
            staged = json.loads((stage / 'runtime.json').read_text())
            self.assertEqual(staged['profiles'][stage_tool.ALIAS]['flash_memory'], POLICY)
            self.assertNotIn('flash_memory', staged['profiles']['ornith-35b-a3b-local'])
            self.assertEqual((stage / 'rollback/runtime.original').read_bytes(), runtime_raw)
            self.assertEqual((stage / 'rollback/unit.original').read_text(), UNIT)
            self.assertTrue((stage / 'runtime-source/exec_in_scope.py').is_file())
            self.assertIn(result['release_dir'], (stage / 'local-ai-chat.service').read_text())
            self.assertEqual(result['runtime_sha256'], stage_tool.sha((stage / 'runtime.json').read_bytes()))
            self.assertFalse((root / 'releases').exists())
            with self.assertRaises(ValueError):
                stage_tool.stage(root / 'stage', **arguments)
            with self.assertRaises(ValueError):
                stage_tool.stage(root / 'stage2', **{**arguments, 'runtime_sha': '0' * 64})


sys.path.insert(0, str(Path(__file__).resolve().parent))
from _stub_manager import StubManagerCase  # noqa: E402


class ControllerUse(StubManagerCase):
    """N8: exercised, not grepped: a classic profile carrying a Flash memory policy is refused
    before any child starts."""
    def test_other_models_refuse_the_flash_policy(self):
        self.assertEqual(memory_policy.flash_policy({'flash_memory': dict(POLICY)})['host_floor_bytes'], GIB)
        self.manager.config = Mock(return_value={**self.config(), 'flash_memory': dict(POLICY)})
        with self.run_stub() as popen, \
                self.assertRaisesRegex(controller.Unavailable, 'Flash memory policy is set on a model that is not Flash'):
            self.manager.begin(allow_start=True, model_alias='stub-local')
        popen.assert_not_called()
        self.assertIsNone(self.manager.process)


class LintControllerFlashWiring(unittest.TestCase):
    """Static lint (N8: named as such): the Flash guard floor and scope wiring need the Rust
    Flash runtime to exercise; until a stub covers that path, check the call sites exist."""
    def test_flash_policy_reaches_the_guard_and_the_scope(self):
        source = (ROOT / 'src/snapdragon_ai_lab/runtime/controller.py').read_text()
        self.assertIn("reserve_bytes=flash_memory['host_floor_bytes']", source)
        self.assertIn("memory_policy.scope_command(args, flash_memory['scope_max_bytes'])", source)


if __name__ == '__main__':
    unittest.main()
