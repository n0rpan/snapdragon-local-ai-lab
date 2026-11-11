import contextlib
import getpass
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("mac_reproduce", _ROOT / "tools" / "mac_reproduce.py")
mr = importlib.util.module_from_spec(_spec)
sys.modules["mac_reproduce"] = mr
_spec.loader.exec_module(mr)

PROMPT_SECRET_TEXT = "PROMPT-TEXT-THAT-MUST-NOT-APPEAR-IN-ARGV"
FAKE_MAC = ":".join(["00", "1A", "2B", "3C", "4D", "5E"])
FAKE_KEY = "sk-" + "A1b2C3d4E5f6G7h8I9j0K1l2"  # assembled, not a real key


def _bench_stdout():
    commit = mr.PINNED_COMMIT
    return json.dumps([
        {"build_commit": commit, "model_filename": "model.gguf", "n_prompt": 512, "n_gen": 0,
         "n_threads": 8, "avg_ts": 101.0, "stddev_ts": 1.0, "samples_ts": [100.0, 101.0, 102.0],
         "samples_ns": [1, 2, 3]},
        {"build_commit": commit, "model_filename": "model.gguf", "n_prompt": 0, "n_gen": 128,
         "n_threads": 8, "avg_ts": 31.0, "stddev_ts": 1.0, "samples_ts": [30.0, 31.0, 32.0],
         "samples_ns": [1, 2, 3]},
    ])


PERF_SAMPLE = (
    "common_perf_print:        load time =    1234.56 ms\n"
    "common_perf_print: prompt eval time =     812.34 ms /   512 tokens (    1.59 ms per token,   630.35 tokens per second)\n"
    "common_perf_print:        eval time =    3456.78 ms /   127 runs   (   27.22 ms per token,    36.74 tokens per second)\n"
    "common_perf_print:       total time =    4269.12 ms /   639 tokens\n"
)


class BenchParseTests(unittest.TestCase):
    def test_select_rows_and_values(self):
        rows = mr.parse_bench_json(_bench_stdout())
        prefill, decode = mr.select_bench_rows(rows)
        self.assertEqual(prefill["n_prompt"], 512)
        self.assertEqual(prefill["samples_ts"], [100.0, 101.0, 102.0])
        self.assertEqual(decode["n_gen"], 128)
        self.assertEqual(decode["samples_ts"], [30.0, 31.0, 32.0])

    def test_bad_json_and_missing_rows(self):
        with self.assertRaises(ValueError):
            mr.parse_bench_json("not json")
        with self.assertRaises(ValueError):
            mr.parse_bench_json("[]")
        only_prefill = mr.parse_bench_json(json.dumps([{"n_prompt": 512, "n_gen": 0, "samples_ts": [1.0]}]))
        with self.assertRaises(ValueError):
            mr.select_bench_rows(only_prefill)

    def test_commit_check(self):
        self.assertTrue(mr.commit_matches(mr.PINNED_COMMIT[:7]))
        self.assertTrue(mr.commit_matches(mr.PINNED_COMMIT))
        self.assertFalse(mr.commit_matches("deadbeef"))
        self.assertFalse(mr.commit_matches(""))
        self.assertFalse(mr.commit_matches(None))


class CompletionParseTests(unittest.TestCase):
    def test_perf_lines(self):
        perf = mr.parse_completion_perf(PERF_SAMPLE)
        self.assertEqual(perf["prompt_tokens"], 512)
        self.assertAlmostEqual(perf["prefill_tok_s"], 630.35)
        self.assertEqual(perf["generated_tokens"], 127)
        self.assertAlmostEqual(perf["decode_tok_s"], 36.74)

    def test_prompt_line_is_not_taken_as_eval_line(self):
        only_prompt = PERF_SAMPLE.split("\n")[1] + "\n"
        with self.assertRaises(ValueError):
            mr.parse_completion_perf(only_prompt)

    def test_missing_perf_raises(self):
        with self.assertRaises(ValueError):
            mr.parse_completion_perf("no timings here")


class SummaryTests(unittest.TestCase):
    def _runs(self, values, warm=2000.0, rc=0):
        runs = [mr.make_run("prompt:p:w0", "prompt", "p", None, 0, True, 4096, 8, 99, [], rc,
                            decode_tok_s=warm, prefill_tok_s=warm)]
        for i, v in enumerate(values, 1):
            runs.append(mr.make_run("prompt:p:r%d" % i, "prompt", "p", None, i, False, 4096, 8, 99, [], rc,
                                    decode_tok_s=v, prefill_tok_s=v * 10))
        return runs

    def test_warmup_dropped_median_min_max(self):
        runs = self._runs([10.0, 12.0, 11.0])
        summary = {(s["metric"]): s for s in mr.summarize(runs)}
        self.assertEqual(summary["decode_tok_s"]["n"], 3)
        self.assertEqual(summary["decode_tok_s"]["median"], 11.0)
        self.assertEqual(summary["decode_tok_s"]["min"], 10.0)
        self.assertEqual(summary["decode_tok_s"]["max"], 12.0)
        self.assertEqual(summary["prefill_tok_s"]["median"], 110.0)

    def test_evidence_m_only_with_three_clean_reps(self):
        self.assertEqual(mr.evidence_for(self._runs([10.0, 12.0, 11.0]), "ok"), {"class": "M", "n": 3})
        self.assertEqual(mr.evidence_for(self._runs([10.0, 12.0]), "ok"), {"class": "S", "n": 2})
        self.assertEqual(mr.evidence_for(self._runs([10.0, 12.0, 11.0]), "failed")["class"], "S")
        self.assertEqual(mr.evidence_for(self._runs([10.0, 12.0, 11.0], rc=1), "ok")["class"], "S")
        self.assertEqual(mr.evidence_for([], "skipped"), {"class": "S", "n": 0})


class ArgvTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="mrtest-")
        self.bin_dir = os.path.join(self.tmp, "llama", "build", "bin")
        self.models = os.path.join(self.tmp, "models")
        os.makedirs(self.bin_dir)
        os.makedirs(self.models)
        self.model = os.path.join(self.models, "model.gguf")
        self.prompt = os.path.join(self.tmp, "prompt.txt")
        with open(self.prompt, "w", encoding="utf-8") as fh:
            fh.write(PROMPT_SECRET_TEXT)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_sanitised_argv_has_no_absolute_paths(self):
        path_map = {os.path.join(self.bin_dir, "llama-bench"): "<llama-dir>/build/bin",
                    os.path.join(self.bin_dir, "llama-completion"): "<llama-dir>/build/bin",
                    self.model: "<models-dir>", self.prompt: "<tmp>"}
        argvs = [
            mr.sanitize_argv(mr.bench_argv(os.path.join(self.bin_dir, "llama-bench"), self.model,
                                           "llama-cpu", 8, 3), path_map),
            mr.sanitize_argv(mr.completion_argv(os.path.join(self.bin_dir, "llama-completion"), self.model,
                                                self.prompt, "llama-metal", 8), path_map),
        ]
        for argv in argvs:
            for element in argv:
                self.assertNotIn(self.tmp, element)
                self.assertNotIn("/home/", element)
                self.assertNotIn("/Users/", element)
                self.assertFalse(element.startswith("/"), element)
            self.assertNotIn(PROMPT_SECRET_TEXT, " ".join(argv))
        self.assertIn("<models-dir>/model.gguf", argvs[0])
        self.assertIn("<tmp>/prompt.txt", argvs[1])
        self.assertEqual(argvs[0][0], "<llama-dir>/build/bin/llama-bench")

    def test_unmapped_absolute_path_is_refused(self):
        with self.assertRaises(ValueError):
            mr.sanitize_argv(["/usr/bin/whatever"], {})

    def test_mlx_argv_uses_exact_driver_placeholder(self):
        argv = mr.mlx_argv(sys.executable, self.tmp, self.prompt, 3)
        stored = mr.sanitize_argv(argv, {sys.executable: "<python-dir>", self.tmp: "<mlx-model-dir>",
                                         self.prompt: "<tmp>"},
                                  {mr.MLX_DRIVER: "<mlx-driver>"})
        self.assertEqual(stored[0], "<python-dir>/" + os.path.basename(sys.executable))
        self.assertEqual(stored[2], "<mlx-driver>")
        self.assertNotIn(PROMPT_SECRET_TEXT, " ".join(stored))


class ScanPublicTests(unittest.TestCase):
    def test_home_path_detected(self):
        self.assertIn("home-path-prefix", mr.scan_public("cwd /home/alice/work"))
        self.assertIn("home-path-prefix", mr.scan_public("cwd /Users/bob/work"))
        self.assertIn("home-path-prefix", mr.scan_public("/private/var/folders/x"))

    def test_login_name_detected(self):
        user = getpass.getuser()
        if len(user) < 3:
            self.skipTest("login name shorter than 3 characters")
        self.assertIn("login-name", mr.scan_public("ran as %s today" % user))

    def test_mac_address_detected(self):
        self.assertIn("mac-address", mr.scan_public("adapter %s up" % FAKE_MAC))
        self.assertIn("mac-address", mr.scan_public("adapter %s up" % FAKE_MAC.replace(":", "-")))

    def test_serial_detected(self):
        self.assertIn("serial-number", mr.scan_public("serial" + "_number: C02ABC123XYZ"))
        self.assertIn("serial-number", mr.scan_public('{"ser' + 'ial": "C02ABC123XYZ"}'))

    def test_email_and_secrets_detected(self):
        self.assertIn("email-address", mr.scan_public("contact a.b@example.com"))
        self.assertIn("api-key-sk", mr.scan_public(FAKE_KEY))
        self.assertIn("github-token", mr.scan_public("gh" + "p_" + "A" * 36))
        self.assertIn("private-key-block", mr.scan_public("-----BEGIN RSA " + "PRIVATE KEY-----"))

    def test_hits_are_rule_names_only(self):
        hits = mr.scan_public("/home/alice/x " + FAKE_MAC + " " + FAKE_KEY)
        joined = " ".join(hits)
        self.assertNotIn("alice", joined)
        self.assertNotIn(FAKE_KEY, joined)
        self.assertNotIn("00:1A", joined)

    def test_clean_receipt_passes(self):
        hw = mr.collect_hardware(system="Linux")
        cell = mr.new_cell("ornith", "llama-metal", hw)
        cell["runs"].append(mr.make_run("prompt:p:r1", "prompt", "p", "ab" * 32, 1, False, 4096, 8, 99,
                                        ["<llama-dir>/build/bin/llama-completion", "-m",
                                         "<models-dir>/model.gguf"], 0, decode_tok_s=30.0))
        mr.finalize_cell(cell, "ok")
        self.assertEqual(mr.scan_public(json.dumps(cell, indent=2)), [])


class WriteRefusalTests(unittest.TestCase):
    def test_planted_home_path_is_refused_and_not_written(self):
        out = tempfile.mkdtemp(prefix="mrout-")
        try:
            cell = mr.new_cell("ornith", "llama-metal", mr.collect_hardware(system="Linux"))
            cell["notes"].append("copied from /Users/alice/private/run")
            mr.finalize_cell(cell, "ok")
            with self.assertRaises(mr.PublicSafetyRefusal) as ctx:
                mr.write_receipt(cell, out)
            self.assertTrue(ctx.exception.rules)
            self.assertEqual(os.listdir(out), [])
        finally:
            shutil.rmtree(out, ignore_errors=True)


class HardwareTests(unittest.TestCase):
    FORBIDDEN = ("system_profiler", "hostname", "ioreg", "scutil")

    def _fake_run(self, calls, table):
        def fake(argv, *args, **kwargs):
            calls.append(list(argv))
            key = " ".join(argv)
            stdout = table.get(key, "")
            return subprocess.CompletedProcess(argv, 0 if key in table else 1, stdout=stdout, stderr="")
        return fake

    def test_darwin_block_and_no_forbidden_commands(self):
        table = {
            "sysctl -n machdep.cpu.brand_string": "Apple M3 Pro\n",
            "sysctl -n hw.memsize": "17179869184\n",
            "sysctl -n hw.perflevel0.physicalcpu": "6\n",
            "sysctl -n hw.perflevel1.physicalcpu": "6\n",
            "sysctl -n hw.model": "Mac15,6\n",
            "sw_vers -productVersion": "15.1\n",
            "pmset -g batt": "Now drawing from 'AC Power'\n",
            "pmset -g therm": "CPU_Scheduler_Limit = 100\nCPU_Speed_Limit = 100\n",
        }
        calls = []
        with mock.patch.object(mr.subprocess, "run", side_effect=self._fake_run(calls, table)):
            with mock.patch.object(mr.platform, "machine", return_value="arm64"):
                hw = mr.collect_hardware(system="Darwin")
        self.assertEqual(hw["chip"], "Apple M3 Pro")
        self.assertEqual(hw["ram_gb"], 16)
        self.assertEqual(hw["performance_cores"], 6)
        self.assertEqual(hw["model_identifier"], "Mac15,6")
        self.assertEqual(hw["power_source"], "ac")
        self.assertEqual(hw["cpu_speed_limit"], 100)
        self.assertTrue(hw["is_apple_silicon"])
        self.assertTrue(calls)
        for argv in calls:
            joined = " ".join(argv)
            for word in self.FORBIDDEN:
                self.assertNotIn(word, joined)

    def test_thermal_limit_absent_is_none(self):
        table = {"pmset -g batt": "Now drawing from 'Battery Power'\n", "pmset -g therm": "nothing here\n"}
        calls = []
        with mock.patch.object(mr.subprocess, "run", side_effect=self._fake_run(calls, table)):
            self.assertEqual(mr.sample_power_thermal(), ("battery", None))

    def test_linux_block_runs_no_commands(self):
        calls = []
        with mock.patch.object(mr.subprocess, "run", side_effect=self._fake_run(calls, {})):
            hw = mr.collect_hardware(system="Linux")
        self.assertEqual(calls, [])
        self.assertEqual(hw["os"], "Linux")
        self.assertFalse(hw["is_apple_silicon"])
        self.assertEqual(hw["power_source"], "unknown")


class PlanVerifySelfcheckTests(unittest.TestCase):
    def test_plan_runs_on_linux_and_prints_no_home_path(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = mr.main(["plan"])
        out = buf.getvalue()
        self.assertEqual(rc, 0)
        self.assertNotIn("/home/", out)
        self.assertNotIn("/Users/", out)
        self.assertIn("llama-metal", out)
        self.assertIn("mlx", out)
        self.assertIn("<models-dir>", out)

    def test_verify_models_reports_mismatch_and_missing(self):
        tmp = tempfile.mkdtemp(prefix="mrverify-")
        try:
            with open(os.path.join(tmp, "Ornith-1.5-35B-A3B-APEX-MTP-Compact.gguf"), "wb") as fh:
                fh.write(b"x")
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = mr.main(["verify-models", "--models-dir", tmp])
            out = buf.getvalue()
            self.assertEqual(rc, 1)
            self.assertIn("mismatch", out)
            self.assertIn("missing", out)
            self.assertIn("NOT USABLE", out)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_selfcheck_exit_codes(self):
        tmp = tempfile.mkdtemp(prefix="mrself-")
        try:
            bad = os.path.join(tmp, "bad.json")
            good = os.path.join(tmp, "good.json")
            with open(bad, "w", encoding="utf-8") as fh:
                fh.write('{"k": "%s"}' % FAKE_KEY)
            with open(good, "w", encoding="utf-8") as fh:
                fh.write('{"k": "clean"}')
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(mr.main(["selfcheck", bad]), 1)
                self.assertEqual(mr.main(["selfcheck", good]), 0)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_run_refuses_without_performance_cores(self):
        tmp = tempfile.mkdtemp(prefix="mrrun-")
        out_dir = os.path.join(tmp, "receipts")
        try:
            with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
                rc = mr.main(["run", "--models-dir", tmp, "--out", out_dir, "--stacks", "mlx"])
            self.assertEqual(rc, 2)
            self.assertFalse(os.path.exists(out_dir))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class ManifestAndFixtureTests(unittest.TestCase):
    def test_manifest_matches_dependency_inventory(self):
        path = _ROOT / "dependencies" / "model-artifacts.json"
        if not path.exists():
            self.skipTest("dependencies/model-artifacts.json not present")
        data = json.loads(path.read_text(encoding="utf-8"))
        by_id = {m["id"]: m for m in data["models"]}
        pairs = {
            "ornith": "ornith-1.5-35b-a3b-apex-mtp-compact",
            "qwen": "qwen36-moe-mtp-ud-q4-k-m",
            "flash": "qwen38-flash-next-ud-iq3-xxs",
        }
        for mid, json_id in pairs.items():
            entry = by_id[json_id]
            spec = mr.MANIFEST[mid]
            self.assertEqual(entry["repository"], spec["repo"])
            self.assertEqual(entry["revision"], spec["revision"])
            if "files" in entry:
                json_files = [(os.path.basename(f["filename"]), f["bytes"], f["sha256"]) for f in entry["files"]]
            else:
                json_files = [(entry["file"], entry["bytes"], entry["sha256"])]
            mine = [(f["name"], f["bytes"], f["sha256"]) for f in spec["files"]]
            self.assertEqual(mine, json_files)

    def test_repo_fixture_loads(self):
        path = _ROOT / "benchmarks" / "fixtures" / "code-screen-v1.json"
        if not path.exists():
            self.skipTest("fixture not present")
        prompts = mr.load_fixtures(str(path))
        ids = [pid for pid, _ in prompts]
        self.assertTrue(ids)
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(all(text for _, text in prompts))

    def test_driver_compiles_and_parser_takes_last_json_line(self):
        compile(mr.MLX_DRIVER, "<mlx-driver>", "exec")
        text = "loading weights...\nwarning: noise\n{\"status\": \"skipped\", \"reason\": \"x\"}\n"
        self.assertEqual(mr.parse_mlx_output(text)["status"], "skipped")
        with self.assertRaises(ValueError):
            mr.parse_mlx_output("no json here")


class LlamaCheckoutTests(unittest.TestCase):
    def test_non_checkout_is_not_ready(self):
        tmp = tempfile.mkdtemp(prefix="mrcheck-")
        try:
            ok, problems, info = mr.check_llama_checkout(tmp)
            self.assertFalse(ok)
            self.assertTrue(any("not a git checkout" in p for p in problems))
            self.assertTrue(any("llama-bench" in p for p in problems))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class CellSkipTests(unittest.TestCase):
    def test_unusable_model_is_skipped(self):
        cell = mr.new_cell("qwen", "llama-metal", mr.collect_hardware(system="Linux"))
        ctx = {"verification": {"qwen": {"ok": False, "files": [
            {"name": "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf", "bytes": None, "sha256": None, "status": "missing"}]}},
            "mlx_map": {}}
        self.assertEqual(mr.run_one_cell(cell, "qwen", "llama-metal", ctx), "skipped")
        self.assertTrue(any("cells skipped" in n for n in cell["notes"]))
        self.assertEqual(cell["model"]["files"][0]["sha256_matches_manifest"], False)

    def test_mlx_without_model_dir_is_skipped(self):
        cell = mr.new_cell("ornith", "mlx", mr.collect_hardware(system="Linux"))
        ctx = {"verification": {"ornith": {"ok": True, "files": []}}, "mlx_map": {}}
        self.assertEqual(mr.run_one_cell(cell, "ornith", "mlx", ctx), "skipped")
        self.assertTrue(any("no --mlx-model" in n for n in cell["notes"]))


class FakeBinaryEndToEndTests(unittest.TestCase):
    """Runs run_llama_cell against shell stubs (not llama.cpp) that print the same formats."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="mrfake-")
        self.bin_dir = os.path.join(self.tmp, "build", "bin")
        os.makedirs(self.bin_dir)
        self.models = os.path.join(self.tmp, "models")
        os.makedirs(self.models)
        self.work = os.path.join(self.tmp, "work")
        os.makedirs(self.work)
        self.bench_stub = os.path.join(self.bin_dir, "llama-bench")
        self.comp_stub = os.path.join(self.bin_dir, "llama-completion")
        self._write_stub(self.bench_stub, "cat <<'EOF'\n%s\nEOF\n" % _bench_stdout())
        self._write_stub(self.comp_stub,
                         "printf 'generated text\\n'\ncat >&2 <<'EOF'\n%sEOF\n" % PERF_SAMPLE)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write_stub(self, path, body):
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("#!/bin/sh\n" + body)
        os.chmod(path, 0o755)

    def _run(self):
        cell = mr.new_cell("ornith", "llama-metal", mr.collect_hardware(system="Linux"))
        info = {"bin_dir": self.bin_dir, "binaries": [], "tree_clean": True}
        status = mr.run_llama_cell(cell, "ornith", "llama-metal", info, self.models,
                                   [("p1", PROMPT_SECRET_TEXT)], 3, 8, self.work)
        return cell, status

    def test_full_cell_with_stub_binaries(self):
        cell, status = self._run()
        self.assertEqual(status, "ok")
        mr.finalize_cell(cell, status)
        self.assertEqual(cell["engine"]["version"], mr.PINNED_COMMIT)
        bench = [r for r in cell["runs"] if r["test"] == "bench"]
        prompt = [r for r in cell["runs"] if r["test"] == "prompt"]
        self.assertEqual(len(bench), 1 + 3 + 3)
        self.assertEqual(len(prompt), 4)
        self.assertTrue(prompt[0]["dropped_warmup"])
        self.assertEqual(sum(1 for r in prompt if not r["dropped_warmup"]), 3)
        summary = {(s["test"], s["prompt_id"], s["metric"]): s for s in cell["summary"]}
        self.assertEqual(summary[("bench", "pp512", "prefill_tok_s")]["median"], 101.0)
        self.assertEqual(summary[("bench", "tg128", "decode_tok_s")]["median"], 31.0)
        self.assertEqual(summary[("prompt", "p1", "decode_tok_s")]["median"], 36.74)
        self.assertEqual(cell["evidence"], {"class": "M", "n": 3})
        self.assertEqual(len(prompt[1]["output_sha256"]), 64)
        self.assertEqual(prompt[1]["generated_tokens"], 127)
        blob = json.dumps(cell)
        self.assertNotIn(PROMPT_SECRET_TEXT, blob)
        self.assertNotIn(self.tmp, blob)
        self.assertEqual(mr.scan_public(blob), [])

    def test_failing_completion_marks_cell_failed(self):
        self._write_stub(self.comp_stub, "exit 3\n")
        cell, status = self._run()
        self.assertEqual(status, "failed")
        failed = [r for r in cell["runs"] if r["returncode"] != 0]
        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0]["returncode"], 3)
        mr.finalize_cell(cell, status)
        self.assertEqual(cell["evidence"]["class"], "S")


if __name__ == "__main__":
    unittest.main()
