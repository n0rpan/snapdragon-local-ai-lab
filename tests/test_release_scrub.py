"""The tracked tree must stay free of private machine details. Model-free, stdlib only."""
import codecs
import fnmatch
import os
import importlib.util
import re
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
        """Run with the real home of whoever runs the tests. The login name is not checked here
        (user-name counts are dropped: a CI login such as "runner" is an ordinary word); the tree scan
        with this machine's deny words covers it (test_tree_scan_has_no_unexplained_hit)."""
        import os
        real_home = os.path.expanduser("~")
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
        self.assertTrue(real_home.startswith('/') and real_home != '/', 'the home scan needs a real home')

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

    def test_owner_identity_only_in_copyright_and_author_lines(self):
        """A FIXED denylist, independent of who runs the tests: the deny words above come from the
        current login, so CI running as another user never saw the owner's first name in a source
        comment (v0.1.3 audit). The owner's first name and handle may appear only on copyright/author
        lines; the handle also as the public repository owner (`handle/repo`, a noreply address).
        Skipped: files exempted for every rule ('*') in tools/tree-scan.allow (private documents and
        scanner tests) and files the public snapshot leaves out (tools/public-exclude.txt)."""
        entries = er.load_allow_file(ROOT / "tools" / "tree-scan.allow")[0]
        exempt = [glob for glob, rule, _ in entries if rule == "*"]
        source_excludes = ROOT / "tools" / "public-exclude.txt"
        if source_excludes.is_file():
            for raw in source_excludes.read_text(encoding="utf-8").splitlines():
                raw = raw.strip()
                if raw and not raw.startswith("#"):
                    exempt.append(raw + "*" if raw.endswith("/") else raw)
        else:
            # Exported trees intentionally omit private release tooling. Scan all
            # their tracked files; never skip this owner-identity assertion.
            self.assertFalse((ROOT / "tools" / "build_public_snapshot.sh").exists(),
                             "a development tree must retain its private exclusion list")
        # rot13, so this file does not itself carry the words it scans for.
        first, handle = (codecs.decode(w, "rot13") for w in OWNER_WORDS_ROT13)
        words = re.compile(r"(?i)(?<![A-Za-z0-9])(?:%s|%s)(?![A-Za-z0-9])" % (re.escape(first), re.escape(handle)))
        repo_owner = re.compile(r"(?i)(?<![A-Za-z0-9])%s(?=/|@users\.noreply)" % re.escape(handle))
        left = []
        for path in self.files:
            rel = path.relative_to(ROOT).as_posix()
            if not path.is_file() or any(fnmatch.fnmatchcase(rel, g) for g in exempt):
                continue
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except (UnicodeDecodeError, OSError):
                continue
            for no, line in enumerate(lines, 1):
                if not words.search(repo_owner.sub("", line)):
                    continue
                if ATTRIBUTION.search(line) or (no > 1 and ATTRIBUTION_CONT.search(lines[no - 2])):
                    continue
                left.append("%s:%d" % (rel, no))
        self.assertEqual(left, [], "the owner's name/handle belongs only on copyright or author lines")

    def test_owner_check_catches_a_comment_and_spares_attribution(self):
        first = codecs.decode(OWNER_WORDS_ROT13[0], "rot13").capitalize()
        self.assertTrue(ATTRIBUTION.search("Copyright (c) 2026 %s X (h)" % first))
        self.assertTrue(ATTRIBUTION.search("From: %s X <1+h@users.noreply.github.com>" % first))
        self.assertTrue(ATTRIBUTION_CONT.search("   Changes:  the patches modify upstream files; modified by"))
        self.assertIsNone(ATTRIBUTION.search("MODE = 'fast'  # O-105 (%s): fast is the default" % first))


# The owner's first name and GitHub handle, rot13-encoded (see test_owner_identity_only_...).
OWNER_WORDS_ROT13 = ("bfpne", "a0ecna")
# Copyright and author/attribution lines (owner decision O-23: the name appears only there).
ATTRIBUTION = re.compile(r"(?i)\bcopyright\b|^\s*from: |\bauthor(?:ed)?\b|\bmodified by\b|\bpatch(?:es)? (?:is|are) by\b")
# A wrapped attribution: the name starts the next line.
ATTRIBUTION_CONT = re.compile(r"(?i)\bmodified by\s*$")


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
