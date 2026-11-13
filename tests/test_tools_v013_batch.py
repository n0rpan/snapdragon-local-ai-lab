"""Model-free tests for tools/v013_batch.py and benchmarks/v013-batch.json (no hardware, no real state)."""
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location('v013_batch', ROOT / 'tools/v013_batch.py')
vb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vb)
MANIFEST = ROOT / 'benchmarks/v013-batch.json'


def run(argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = vb.main(argv)
    return code, out.getvalue(), err.getvalue()


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.m = vb.load_manifest(MANIFEST)

    def test_shipped_manifest_is_valid(self):
        self.assertEqual(vb.validate(self.m), [])

    def test_two_batches_with_the_plan_arms(self):
        ids = [b['id'] for b in self.m['batches']]
        self.assertEqual(ids, ['chat', 'image-flash'])
        chat = self.m['batches'][0]
        self.assertGreaterEqual(len(chat['arms']), 14)
        self.assertEqual(self.m['batches'][1]['hours_est'], 3)

    def test_kill_numbers_from_plan_sec5_are_present(self):
        arms = {a['id']: a for b in self.m['batches'] for a in b['arms']}
        self.assertIn('15%', arms['pp512-fa2-vs-today']['kill'])
        self.assertIn('27 GiB', arms['rss-ladder-256k']['kill'])
        self.assertIn('10%', arms['encoder-cpu-arm']['kill'])
        self.assertIn('3%', arms['a2-abba']['kill'])

    def test_validate_catches_bad_manifest(self):
        bad = copy.deepcopy(self.m)
        del bad['batches'][0]['arms'][0]['kill']
        self.assertTrue(any('kill' in e for e in vb.validate(bad)))
        bad = copy.deepcopy(self.m)
        bad['state_roots']['template'] = '/tmp/x'
        self.assertTrue(vb.validate(bad))

    def test_state_roots_are_private_per_arm(self):
        roots = set()
        for b in self.m['batches']:
            for a in b['arms']:
                roots.add(vb.state_root(self.m, b['id'], a['id'], '/state'))
        total = sum(len(b['arms']) for b in self.m['batches'])
        self.assertEqual(len(roots), total)
        self.assertTrue(all('/v013-batch/' in r for r in roots))


class CliTests(unittest.TestCase):
    def test_dry_run_is_the_default_and_executes_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            code, out, _ = run(['--state-home', d, '--batch', 'chat'])
        self.assertEqual(code, 0)
        self.assertIn('nothing executed', out)
        self.assertIn(d, out)  # state roots resolve under the given home

    def test_execute_without_lease_is_refused(self):
        code, _, err = run(['--execute', '--batch', 'chat'])
        self.assertEqual(code, 2)
        self.assertIn('--lease', err)

    def test_execute_without_batch_is_refused(self):
        code, _, _ = run(['--execute', '--lease', '/nonexistent'])
        self.assertEqual(code, 2)

    def test_lease_checks(self):
        with tempfile.TemporaryDirectory() as d:
            lease = Path(d) / 'lease.json'
            cases = [
                ({'batch': 'chat', 'expires_unix': 2000}, 1000, None),
                ({'batch': 'chat', 'expires_unix': 900}, 1000, 'expired'),
                ({'batch': 'image-flash', 'expires_unix': 2000}, 1000, 'not'),
                ({'batch': 'chat'}, 1000, 'expired'),
            ]
            for body, now, expect in cases:
                lease.write_text(json.dumps(body), encoding='utf-8')
                code, _, err = run(['--execute', '--batch', 'chat', '--lease', str(lease), '--now', str(now)])
                if expect is None:
                    self.assertEqual(code, 4, err)  # valid lease: lane E still runs no arms
                else:
                    self.assertEqual(code, 2)
                    self.assertIn(expect, err)
            lease.write_text('not json', encoding='utf-8')
            code, _, _ = run(['--execute', '--batch', 'chat', '--lease', str(lease)])
            self.assertEqual(code, 2)

    def test_invalid_manifest_exits_one(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'm.json'
            path.write_text(json.dumps({'schema': 'nope'}), encoding='utf-8')
            code, _, _ = run(['--manifest', str(path)])
        self.assertEqual(code, 1)


if __name__ == '__main__':
    unittest.main()
