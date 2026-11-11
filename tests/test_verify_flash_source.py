import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


TOOL = Path(__file__).resolve().parents[1] / "tools/verify-flash-source.py"
SPEC = importlib.util.spec_from_file_location("verify_flash_source", TOOL)
VERIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFY)


class FlashSourceVerifierTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.parent = self.base / "parent"
        self.llama = self.parent / "third_party/llama.cpp"
        self.env = dict(os.environ, GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1",
                        GIT_AUTHOR_NAME="Source fixture", GIT_AUTHOR_EMAIL="fixture@invalid",
                        GIT_COMMITTER_NAME="Source fixture", GIT_COMMITTER_EMAIL="fixture@invalid")
        for repo in (self.parent, self.llama):
            repo.mkdir(parents=True, exist_ok=True)
            self.git(repo, "init", "--quiet")
        (self.parent / "CMakeLists.txt").write_text("project(source_fixture)\n")
        (self.llama / "src").mkdir()
        (self.llama / "src/llama-model.cpp").write_text("mapping(false);\n")
        (self.llama / "link").symlink_to("missing-source-target")
        (self.llama / "script.sh").write_text("#!/bin/sh\nexit 0\n")
        (self.llama / "script.sh").chmod(0o755)
        self.git(self.llama, "add", "src", "link", "script.sh")
        self.git(self.llama, "-c", "commit.gpgsign=false", "commit", "--quiet", "-m", "fixture")
        self.git(self.parent, "add", "CMakeLists.txt", "third_party/llama.cpp")
        self.git(self.parent, "-c", "commit.gpgsign=false", "commit", "--quiet", "-m", "fixture")
        rows = []
        for repo, prefix in [(self.parent, ""), (self.llama, "third_party/llama.cpp/")]:
            for raw in self.git(repo, "ls-tree", "-r", "-z", "HEAD").split(b"\0"):
                if raw:
                    header, name = raw.split(b"\t", 1)
                    mode, kind, oid = header.decode().split()
                    if kind == "blob":
                        rows.append({"path": prefix + name.decode(), "mode": mode,
                                     "type": kind, "git_object": oid})
        rows.sort(key=lambda row: row["path"])
        lines = "".join(f"{r['mode']} {r['git_object']} {r['path']}\n" for r in rows).encode()
        self.inventory = self.base / "inventory.json"
        self.inventory.write_text(json.dumps({"files": rows,
            "flattened_content_sha256": hashlib.sha256(lines).hexdigest()}))

    def git(self, repo, *args):
        return subprocess.check_output(["git", "-C", str(repo), *args], env=self.env,
                                       stderr=subprocess.DEVNULL)

    def test_real_two_repository_source_including_dangling_symlink_and_mode(self):
        result = VERIFY.verify_source(self.parent, self.inventory)
        self.assertTrue(result["working_tree_blob_bytes_and_modes_verified"])
        self.assertEqual(result["tracked_blobs"], 4)
        self.assertFalse(result["native_execution"])

    def test_working_source_mutation_is_rejected(self):
        path = self.llama / "src/llama-model.cpp"
        info = path.stat()
        path.write_text("mapping(true );\n")
        os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns))
        with self.assertRaises(VERIFY.VerificationError):
            VERIFY.verify_source(self.parent, self.inventory)

    def test_staged_index_mutation_is_rejected(self):
        (self.llama / "src/llama-model.cpp").write_text("mapping(true);\n")
        self.git(self.llama, "add", "src/llama-model.cpp")
        with self.assertRaisesRegex(VERIFY.VerificationError, "index inventory"):
            VERIFY.verify_source(self.parent, self.inventory)

    def test_mode_change_is_rejected_even_when_git_ignores_filemode(self):
        self.git(self.llama, "config", "core.filemode", "false")
        (self.llama / "script.sh").chmod(0o644)
        with self.assertRaisesRegex(VERIFY.VerificationError, "mode differs"):
            VERIFY.verify_source(self.parent, self.inventory)

    def test_symlink_target_change_is_rejected_without_following_it(self):
        link = self.llama / "link"
        link.unlink()
        link.symlink_to("other-missing-source")
        with self.assertRaises(VERIFY.VerificationError):
            VERIFY.verify_source(self.parent, self.inventory)

    def test_oversized_working_file_is_rejected_before_blob_read(self):
        row = json.loads(self.inventory.read_text())["files"][0]
        path = self.parent / row["path"]
        expected = path.stat().st_size
        with path.open("r+b") as stream:
            stream.truncate(8 * 1024 * 1024)
        with self.assertRaisesRegex(VERIFY.VerificationError, "size or mode"):
            VERIFY.source_blob(self.parent, row["path"], row["mode"], expected)

    def test_inventory_paths_reject_parent_escape(self):
        row = json.loads(self.inventory.read_text())["files"][0]
        row["path"] = "../outside"
        with self.assertRaisesRegex(VERIFY.VerificationError, "Unsafe"):
            VERIFY.validate_rows([row])

    def test_declared_windows_script_crlf_is_the_only_normalization(self):
        name = "windows.ps1"
        canonical = b"line1\nline2\n"
        (self.parent / name).write_bytes(canonical.replace(b"\n", b"\r\n"))
        expected = hashlib.sha1(b"blob 12\0" + canonical).hexdigest()
        self.assertEqual(VERIFY.source_blob(self.parent, name, "100644", 12, allow_crlf=True), expected)
        with self.assertRaises(VERIFY.VerificationError):
            VERIFY.source_blob(self.parent, name, "100644", 12)

    def test_installed_prefix_or_explicit_inventory_location(self):
        prefix = self.base / "prefix"
        installed = prefix / "share/snapdragon-local-ai-lab/dependencies/flash-source-inventory.json"
        installed.parent.mkdir(parents=True)
        installed.write_bytes(self.inventory.read_bytes())
        script = prefix / "bin/verify-flash-source.py"
        self.assertEqual(VERIFY.resolve_inventory(script=script), installed)
        self.assertEqual(VERIFY.resolve_inventory(self.inventory), self.inventory)


if __name__ == "__main__":
    unittest.main()
