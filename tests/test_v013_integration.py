"""v0.1.3 integration (cross-lane) glue: what lane D's panel reads from lane C's controller.

- `local-ai ledger` rows carry the context `mode` (lane C) and the REAL-1 `origin`, allowlisted;
- `local-ai status --json` passes the controller's `last_failure` {reason, model, t, origin} through,
  allowlisted (the panel shows the real reason of its own failed action);
- `python -m snapdragon_ai_lab install-plugin --restart-shell` reaches install(restart_shell=True);
  nothing restarts without the flag.

No network, no controller, no real config: the CLI is loaded from source and only its pure helpers run.
"""
import importlib.machinery
import importlib.util
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / 'src/snapdragon_ai_lab/runtime/local-ai'
loader = importlib.machinery.SourceFileLoader('v013_int_local_ai', str(SCRIPT))
spec = importlib.util.spec_from_loader(loader.name, loader)
cli = importlib.util.module_from_spec(spec)
loader.exec_module(cli)

BASE = {'t': 7, 'path': '/v1/chat/completions', 'model': 'qwen3.6-35b-a3b-local', 'status': 200}
ALIASES = {'qwen3.6-35b-a3b-local': 'moe', 'ornith-35b-a3b-local': 'ornith'}


class LedgerModeAndOrigin(unittest.TestCase):
    def test_mode_and_origin_pass_when_allowlisted(self):
        row = cli.ledger_row({**BASE, 'mode': 'deep', 'origin': 'organic'})
        self.assertEqual((row['mode'], row['origin']), ('deep', 'organic'))
        for mode in ('fast', 'deep', 'max'):
            self.assertEqual(cli.ledger_row({**BASE, 'mode': mode})['mode'], mode)
        self.assertEqual(cli.ledger_row({**BASE, 'origin': 'remote'})['origin'], 'remote')

    def test_unknown_or_missing_values_are_dropped(self):
        self.assertFalse({'mode', 'origin'} & set(cli.ledger_row(BASE)))
        bad = cli.ledger_row({**BASE, 'mode': 'turbo', 'origin': 'not-an-origin'})
        self.assertFalse({'mode', 'origin'} & set(bad))
        self.assertNotIn('not-an-origin', json.dumps(bad))


class StatusLastFailure(unittest.TestCase):
    def status(self, failure):
        return cli.optional_status({'last_failure': failure}, ALIASES)

    def test_reason_model_time_origin_pass_through(self):
        got = self.status({'reason': 'Ornith did not start: not enough free RAM', 'kind': 'start',
                           'alias': 'ornith-35b-a3b-local', 'origin': 'organic', 't': 1_800_000_000,
                           'detail': 'PRIVATE-DETAIL'})
        self.assertEqual(got['last_failure'], {'reason': 'Ornith did not start: not enough free RAM', 'model': 'ornith',
                                               't': 1_800_000_000, 'origin': 'organic'})
        self.assertNotIn('PRIVATE', json.dumps(got))

    def test_reason_is_plain_and_bounded_and_bad_shapes_are_dropped(self):
        got = self.status({'reason': 'a\x1b[31m\nb' + 'x' * 400, 'alias': 'unknown-model', 'origin': 'evil', 't': -1})
        self.assertTrue(got['last_failure']['reason'].startswith('a [31m b'))
        self.assertLessEqual(len(got['last_failure']['reason']), 200)
        self.assertEqual((got['last_failure']['model'], got['last_failure']['origin'], got['last_failure']['t']), (None, None, None))
        for bad in (None, 'text', [], {'reason': '', 't': None}, {}):
            self.assertNotIn('last_failure', self.status(bad), bad)


class InstallPluginRestartShell(unittest.TestCase):
    def run_main(self, *argv):
        from snapdragon_ai_lab import __main__ as entry
        with patch.object(entry, 'install', return_value={'ok': True}) as install, \
                patch.object(sys, 'argv', ['snapdragon_ai_lab', 'install-plugin', *argv]), \
                patch('builtins.print'):
            entry.main()
        return install.call_args.kwargs

    def test_restart_only_with_the_flag(self):
        self.assertFalse(self.run_main('--dry-run')['restart_shell'])
        self.assertTrue(self.run_main('--restart-shell')['restart_shell'])


class ImageSizeGateShips(unittest.TestCase):
    def test_bootstrap_installs_every_runtime_module(self):
        from snapdragon_ai_lab import bootstrap
        runtime = Path(__file__).resolve().parents[1] / 'src/snapdragon_ai_lab/runtime'
        # launch_local_plan.py is the OpenCode launcher, installed with the harness, not the runtime
        modules = {p.name for p in runtime.glob('*.py')} - {'__init__.py', 'launch_local_plan.py'}
        self.assertIn('image_size_gate.py', bootstrap.RUNTIME)
        self.assertEqual(modules - set(bootstrap.RUNTIME), set())


class SwapGuardFromTheStartLine(unittest.TestCase):
    """G-F127: the controller follows the Flash engine's `[swap-guard mode=...]` start-up token."""

    def test_token_parse(self):
        from snapdragon_ai_lab.runtime.engine_diagnostics import swap_guard_mode
        line = b'flash cache: 18 GiB; swap-tolerant (explicit pref ...) -> cache 18 GiB [swap-guard mode=swap-tolerant]'
        self.assertEqual(swap_guard_mode(line), 'swap-tolerant')
        self.assertEqual(swap_guard_mode(b'x [swap-guard mode=memavailable]'), 'memavailable')
        self.assertEqual(swap_guard_mode(b'x [swap-guard mode=refuse]'), 'refuse')
        for bad in (b'x [swap-guard mode=other]', b'no token', 'str [swap-guard mode=refuse]', None):
            self.assertIsNone(swap_guard_mode(bad))

    def test_reader_hands_each_line_to_the_hook_and_survives_its_errors(self):
        import os
        import tempfile
        import time
        from snapdragon_ai_lab.runtime.engine_diagnostics import EngineLogWriter, EngineStderrReader
        seen = []
        def hook(line):
            seen.append(line)
            raise RuntimeError('a hook error never stops the drain')
        with tempfile.TemporaryDirectory() as directory:
            r, w = os.pipe()
            reader = EngineStderrReader(os.fdopen(r, 'rb', buffering=0), EngineLogWriter(Path(directory) / 'engine.log'), on_line=hook)
            reader.start()
            os.write(w, b'one [swap-guard mode=refuse]\ntwo\n')
            os.close(w)
            deadline = time.monotonic() + 10
            while len(seen) < 2 and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertTrue(reader.stop(5))
            self.assertEqual(seen, [b'one [swap-guard mode=refuse]', b'two'])
            self.assertIn(b'two', (Path(directory) / 'engine.log').read_bytes())

    def test_hook_leaves_swap_tolerant_only_when_the_engine_says_so(self):
        from types import SimpleNamespace
        from snapdragon_ai_lab.runtime import controller, memory_policy

        class Guard:
            def __init__(self):
                self.m, self.left = memory_policy.SWAP_TOLERANT_MODE, 0
            def mode(self):
                return self.m
            def leave_swap_tolerant(self):
                self.left += 1
                self.m = 'standard'

        for mode, leaves in (('swap-tolerant', 0), ('memavailable', 1), ('refuse', 1)):
            guard, owner = Guard(), SimpleNamespace(flash_swap_guard_reported=None)
            hook = controller.Manager._swap_guard_hook(owner, guard)
            hook(b'unrelated line')
            self.assertEqual((guard.left, owner.flash_swap_guard_reported), (0, None))
            hook(('decision [swap-guard mode=%s]' % mode).encode())
            hook(('decision [swap-guard mode=%s]' % mode).encode())
            self.assertEqual((guard.left, owner.flash_swap_guard_reported), (leaves, mode), mode)
        controller.Manager._swap_guard_hook(SimpleNamespace(flash_swap_guard_reported=None), None)(b'[swap-guard mode=refuse]')


class ModeWriteKeepsTheConfigFile(unittest.TestCase):
    """C x LAN M5: POST /_local/mode writes runtime.json through the same writer as `share`:
    a symlinked config stays a symlink and the file keeps its permission bits."""

    def test_symlink_and_mode_survive(self):
        import os
        import stat
        from openai_api_stack import FakeEngine, Stack, ORNITH
        spec = importlib.util.spec_from_file_location('openai_api_check_int', Path(__file__).resolve().parents[1] / 'tools/openai_api_check.py')
        check = importlib.util.module_from_spec(spec); spec.loader.exec_module(check)
        engine = FakeEngine()
        self.addCleanup(engine.close)
        stack = Stack(engine.port)
        self.addCleanup(stack.close)
        real = stack.config_path.with_name('real-runtime.json')
        os.replace(stack.config_path, real)
        os.chmod(real, 0o640)
        stack.config_path.symlink_to(real)
        code, _, got = check.call(stack.base, 'POST', '/_local/mode', {'model': ORNITH, 'mode': 'deep'})
        self.assertEqual((code, got['mode']), (200, 'deep'))
        self.assertTrue(stack.config_path.is_symlink(), 'the symlink was replaced by a file')
        self.assertEqual(stat.S_IMODE(os.stat(real).st_mode), 0o640)
        self.assertEqual(json.loads(real.read_text())['context_modes']['models'][ORNITH], 'deep')


if __name__ == '__main__':
    unittest.main()
