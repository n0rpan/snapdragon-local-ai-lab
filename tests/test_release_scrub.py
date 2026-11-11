"""The tracked tree must stay free of private machine details. Model-free, stdlib only."""
import fnmatch
import os
import importlib.util
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


er = load("export_receipt")
sr = load("scrub_receipts")
HOME = "/home/" + "quentin"


def tracked_files():
    try:
        run = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-z"], capture_output=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if run.returncode != 0:
        return None
    return [ROOT / name for name in run.stdout.decode("utf-8", "replace").split("\0") if name]


class ScrubText(unittest.TestCase):
    def test_roots_users_units_and_ids_become_placeholders(self):
        text = ('{"a": "HOME/Work/local-ai/run/x.json", "b": "HOME/.local/share/ai-stack/models/m.gguf", '
                '"c": "HOME/.local/state/local-ai/q", "d": "HOME/.config/x", "e": "HOME", '
                '"f": "/sys/fs/cgroup/user.slice/user-1000.slice/user@1000.service/app.slice", '
                '"g": "/run/user/1000/bus", "boot_id": "49f90165-6491-4d83-8184-47ee8551af30", '
                '"h": "projects/-home-quentin/memory", "who": "quentin", "n": 123.5}').replace("HOME", HOME)
        new, counts = sr.scrub_text(text, HOME, "quentin")
        self.assertEqual(new, '{"a": "$LAB_ROOT/run/x.json", "b": "$AI_STACK/models/m.gguf", '
                              '"c": "$LOCAL_AI_STATE/q", "d": "$HOME/.config/x", "e": "$HOME", '
                              '"f": "/sys/fs/cgroup/user.slice/user-UID.slice/user@UID.service/app.slice", '
                              '"g": "$XDG_RUNTIME_DIR/bus", "boot_id": "<redacted-uuid>", '
                              '"h": "projects/-HOME/memory", "who": "user", "n": 123.5}')
        self.assertEqual(counts, {"home-path": 6, "user-slice": 3, "uuid": 1, "user-name": 1})
        self.assertEqual([r for r, _ in er.scan_text(new, {"quentin": "user-name"}) if r not in er.TREE_PROFILE_SKIPS], [])

    def test_other_names_numbers_and_prose_are_untouched(self):
        text = HOME + "sson/x Quentin approved 12.5 tok/s; /home/other/y; 0f8fad5b-d9cb-469f"
        self.assertEqual(sr.scrub_text(text, HOME, "quentin")[0], text)

    def test_scrub_is_idempotent(self):
        once = sr.scrub_text(HOME + "/Work/local-ai/a user-1000.slice", HOME, "quentin")[0]
        self.assertEqual(sr.scrub_text(once, HOME, "quentin"), (once, {}))


class TrackedTree(unittest.TestCase):
    def setUp(self):
        self.files = tracked_files()
        if not self.files:
            self.skipTest("not a git checkout")

    def test_receipts_need_no_further_scrub(self):
        """Run with the fixed stand-in and with the real home and login of whoever runs the tests."""
        import os
        real_home, real_user = os.path.expanduser("~"), os.environ.get("USER", "")
        left = []
        for path in self.files:
            rel = path.relative_to(ROOT).as_posix()
            if not rel.startswith("benchmarks/") or not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            counts = sr.scrub_text(text, real_home, "")[1]
            counts.pop("user-name", None)
            if counts:
                left.append((rel, sorted(counts)))
        self.assertEqual(left, [])
        self.assertTrue(real_user is not None)

    def test_tree_scan_has_no_unexplained_hit(self):
        entries, exact, allowed = er.load_allow_file(ROOT / "tools" / "tree-scan.allow")
        deny = {w: r for w, r in er.system_deny_words().items() if w.lower() not in allowed}
        if os.environ.get("CI"):
            # A CI runner's own user/host names ("runner", ...) are ordinary words, not this lab's machine.
            deny = {}
        hits = [h for h in er.scan_paths([str(p) for p in self.files if p.is_file()], deny, exact)
                if h[1] not in er.TREE_PROFILE_SKIPS]
        left, unused = er.apply_allow(hits, entries, str(ROOT))
        summary = sorted({(Path(p).relative_to(ROOT).as_posix(), rule) for p, rule, _ in left})
        self.assertEqual(summary, [])
        # An exemption is stale only if its glob still names a file here: the same allow file serves the
        # engine repository, and the public snapshot leaves out excluded files (tools/public-exclude*.txt).
        present = [p.relative_to(ROOT).as_posix() for p in self.files if p.is_file()]
        stale = [e[:2] for e in unused if any(fnmatch.fnmatchcase(rel, e[0]) for rel in present)]
        if not os.environ.get("CI"):  # without this machine's words, name exemptions are unused by design
            self.assertEqual(stale, [], "remove exemptions that no longer match anything")


class PreCommitHook(unittest.TestCase):
    """tools/hooks/pre-commit refuses staged receipts that still carry private machine details."""

    def test_hook_blocks_dirty_receipts_and_passes_scrubbed_ones(self):
        import os
        import shutil
        import tempfile
        if shutil.which("git") is None or shutil.which("bash") is None:
            self.skipTest("git and bash are needed")
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            (repo / "tools" / "hooks").mkdir(parents=True)
            (repo / "benchmarks" / "results").mkdir(parents=True)
            shutil.copy(ROOT / "tools" / "scrub_receipts.py", repo / "tools")
            shutil.copy(ROOT / "tools" / "hooks" / "pre-commit", repo / "tools" / "hooks")
            env = dict(os.environ, HOME=HOME, USER="quentin", GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_SYSTEM=os.devnull)

            def run(*argv):
                return subprocess.run(list(argv), cwd=repo, env=env, capture_output=True, text=True, timeout=60)

            self.assertEqual(run("git", "init", "-q").returncode, 0)
            hook = str(repo / "tools" / "hooks" / "pre-commit")
            receipt = repo / "benchmarks" / "results" / "r.json"
            receipt.write_text('{"path": "%s/Work/local-ai/x", "ms": 12.5}\n' % HOME)
            (repo / "notes.md").write_text("see %s/notes\n" % HOME)
            run("git", "add", "-A")
            blocked = run("bash", hook)
            self.assertEqual(blocked.returncode, 1, blocked.stderr)
            self.assertIn("scrub_receipts.py", blocked.stderr)
            self.assertNotIn(HOME, blocked.stderr)
            self.assertEqual(run("python3", "-I", "tools/scrub_receipts.py", "--home", HOME, "--user", "quentin",
                                 "benchmarks").returncode, 0)
            run("git", "add", "-A")
            passed = run("bash", hook)
            self.assertEqual(passed.returncode, 0, passed.stderr)
            self.assertEqual(receipt.read_text(), '{"path": "$LAB_ROOT/x", "ms": 12.5}\n')
            (repo / "tools" / "scrub_receipts.py").unlink()
            self.assertEqual(run("bash", hook).returncode, 0, "the hook fails open when the checker is absent")


if __name__ == "__main__":
    unittest.main()
