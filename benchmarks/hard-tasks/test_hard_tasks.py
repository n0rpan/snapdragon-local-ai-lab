"""Model-free tests of the hard-task runner pieces: no controller, model or OpenCode run."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_hard_task  # noqa: E402
import run_set  # noqa: E402
import summarize  # noqa: E402
import validate  # noqa: E402


class Runner(unittest.TestCase):
    def test_every_task_has_prompt_project_hidden_and_reference(self):
        tasks = validate.all_tasks()
        self.assertGreaterEqual(len(tasks), 10)
        for task in tasks:
            base = HERE / 'tasks' / task
            with self.subTest(task=task):
                self.assertTrue((base / 'TASK.md').is_file())
                self.assertTrue((base / 'hidden/check.sh').is_file())
                self.assertTrue((base / 'project').is_dir())
                self.assertTrue((HERE / 'reference' / task).is_dir())
                self.assertLessEqual(validate.fixture_bytes(base / 'project'), 13 * 1024)

    def test_prepare_gives_fresh_fixture_without_hidden_or_reference(self):
        task = validate.all_tasks()[0]
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'run'
            output.mkdir()
            fixture = run_hard_task.prepare(output, task, 'ornith-35b-a3b-local', 900, dict(run_hard_task.DEFAULT_LIMIT))
            self.assertEqual(run_hard_task.tree_sha(output / 'project'), run_hard_task.tree_sha(HERE / 'tasks' / task / 'project'))
            names = {p.name for p in output.rglob('*')}
            self.assertNotIn('check.sh', names)
            self.assertFalse(any('hidden' in p.parts or 'reference' in p.parts for p in output.rglob('*')))
            config = json.loads((output / 'opencode.json').read_text())
            self.assertEqual(config['model'], 'local-qwen/ornith-35b-a3b-local')
            provider = config['providers']['local-qwen']
            self.assertEqual(list(provider['models']), ['ornith-35b-a3b-local'])
            self.assertEqual(provider['body']['temperature'], 0)
            self.assertEqual(provider['settings']['baseURL'], 'http://127.0.0.1:18096/v1')
            self.assertEqual(fixture['cap_seconds'], 900)
            command = run_hard_task.sandbox_command(output, fixture, Path('/usr/bin/true'), Path(tmp) / 'c.sock')
            self.assertEqual(command[-1], '/work/hard_inside.py')
            self.assertIn('--unshare-all', command)
            self.assertNotIn('/engine', command)
            self.assertNotIn('/model.gguf', command)

    def test_alias_limit_reads_config_and_falls_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'opencode.json'
            path.write_text(json.dumps({'providers': {'p': {'models': {'a': {'limit': {
                'context': 8192, 'input': 6000, 'output': 1024}}}}}}))
            self.assertEqual(run_hard_task.alias_limit('a', path)[0], dict(context=8192, input=6000, output=1024))
            self.assertEqual(run_hard_task.alias_limit('missing', path), (run_hard_task.DEFAULT_LIMIT, 'default'))


class Verdict(unittest.TestCase):
    def table(self, flash, ornith):
        table = {}
        for i, (f, o) in enumerate(zip(flash, ornith)):
            table[(f't{i}', 'F')] = dict(solved_majority=bool(f))
            table[(f't{i}', 'O')] = dict(solved_majority=bool(o))
        return table, [f't{i}' for i in range(len(flash))]

    def test_o1_rule(self):
        cases = [((1, 0, 0), (0, 1, 1), True),    # one Flash-only solve is enough
                 ((1, 1, 0), (1, 1, 0), False),   # equal
                 ((0, 0, 0), (0, 0, 0), False),
                 ((1, 1, 1), (1, 1, 1), False)]
        for flash, ornith, keep in cases:
            table, tasks = self.table(flash, ornith)
            text, _, _ = summarize.verdict(table, tasks, 'F', 'O')
            self.assertEqual(text.startswith('FULL PRIORITY'), keep, (flash, ornith, text))


class Resume(unittest.TestCase):
    """Synthetic results.jsonl: infra failures are re-run, retries are not double-counted."""
    ROWS = [dict(task='t1', model_alias='A', rep=0, harness_ok=False, solved=False, error='Bridge:down', output='/x/1'),
            dict(task='t1', model_alias='A', rep=0, harness_ok=True, solved=True, task_seconds=10, output='/x/2'),
            dict(task='t2', model_alias='A', rep=0, harness_ok=True, solved=False, task_seconds=20, output='/x/3'),
            dict(task='t3', model_alias='A', rep=0, harness_ok=False, solved=False, error='Ns:fail', output='/x/4'),
            dict(task='t2', model_alias='A', rep=0, harness_ok=True, solved=True, task_seconds=30, output='/x/5')]

    def test_done_rows_and_last_valid_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'results.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in self.ROWS) + 'not json\n')
            # t1 retried OK, t2 is a model failure with harness_ok (done), t3 only an infra failure (not done)
            self.assertEqual(run_set.done_rows(root / 'results.jsonl'), {('t1', 'A', 0), ('t2', 'A', 0)})
            good = summarize.last_valid(self.ROWS)
            self.assertEqual([r['output'] for r in good], ['/x/2', '/x/5'])
            (root / 'results.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in self.ROWS))
            out = root / 'HARD-TASKS.md'
            argv = sys.argv
            sys.argv = ['summarize.py', '--root', str(root), '--out', str(out), '--flash', 'F', '--ornith', 'A']
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    summarize.main()
            finally:
                sys.argv = argv
            text = out.read_text()
            self.assertIn('| t1 | PASS (-) 10s', text)
            self.assertIn('| t2 | PASS (-) 30s', text)   # last valid attempt, one run (no 1/2 count)
            self.assertIn('| t3 | - |', text)
            self.assertIn('2/2 solved, 40s', text)
            self.assertIn('## Infrastructure failures', text)
            self.assertIn('t1@A r0 (superseded by a valid retry): Bridge:down [/x/1]', text)
            self.assertIn('t3@A r0 (NOT YET RE-RUN): Ns:fail [/x/4]', text)


if __name__ == '__main__':
    unittest.main()
