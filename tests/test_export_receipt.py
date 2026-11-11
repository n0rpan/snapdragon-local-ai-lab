"""Secret/path scan tests for tools/export_receipt.py. Model-free, stdlib only."""
import importlib.util
import json
import random
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "export_receipt.py"
spec = importlib.util.spec_from_file_location("export_receipt", TOOL)
er = importlib.util.module_from_spec(spec)
spec.loader.exec_module(er)

# Fake secrets are assembled at run time so that the repository's own pre-push secret scan,
# which reads this file as text, does not see key-, MAC- or serial-shaped literals.
FAKE_MAC = ":".join(["a4", "5e", "60", "c1", "22", "9f"])
FAKE_KEY = "sk" + "-abcdefghijklmnopqrstuvwx"
SERIAL_KEY = "serial" + "_number"
DISK_KEY = "id_" + "serial"
# Fixed stand-ins for the machine's identifiers, so the tests do not depend on who runs them.
DENY = {"quentin": "user-name", "Testsson": "real-name", "yogabox": "host-name",
        "PF4ZZ9Q7": "device-serial-value", "S6XNNA0W123456": "disk-serial-value"}
PLANTS = ["/home/quentin", "quentin", "Testsson", "yogabox", "PF4ZZ9Q7", "S6XNNA0W123456", FAKE_MAC,
          FAKE_KEY, "PRIVATE-PROMPT-TEXT", "PRIVATE-ANSWER-TEXT", "quentin@example.org",
          "user-1000.slice", "0f8fad5b-d9cb-469f-a165-70867728950e", "hunter2hunter2", "/run/user/1000",
          "192.168.1.44", "WINDOW-NOTE-TEXT"]
SHA = "a1a8cf0f87904875d81a107f3d85d873a0961f057a10b617d5b505a684b12091"


def private_receipt():
    return {
        "label": "01-A", "fuse": 0, "cache_gib": 20, "binary_sha256": SHA,
        "source_commit": "5ac1243436693e14bed75e0a4052cfca288ef89c",
        "window": "FLASH-RUN TIMING WINDOW-NOTE-TEXT\nno builds",
        "raw_receipts_path": "/home/quentin/Work/local-ai/run/receipts.json",
        "model_path": "/home/quentin/models/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00004.gguf",
        "binary_path": "/home/quentin/Work/root-flash-m3/target/release/oxe-flash",
        "profile": "fast",
        "prompt": "PRIVATE-PROMPT-TEXT write a function for Testsson",
        "prompt_tokens": 156, "prompt_ids": [1, 5, 99, 1024],
        "answer": "PRIVATE-ANSWER-TEXT", "messages": [{"role": "user", "content": "PRIVATE-PROMPT-TEXT"}],
        "lines": ["loaded /home/quentin/x in 3 s", "PRIVATE-ANSWER-TEXT"],
        "argv": ["/home/quentin/bin/oxe-flash", "-m", "/home/quentin/models/flash.gguf", "--cache-gib", "20", "-t", "8",
                 "-C", "0x0ff", "-p", "PRIVATE-PROMPT-TEXT hello", "--api-key", FAKE_KEY,
                 "--out=/home/quentin/out/result.json", "-n", "128", "--temp", "0", "free text PRIVATE-ANSWER-TEXT"],
        "bench_argv": "llama-bench -m /home/quentin/m.gguf -p 512 -n 128 -r 3",
        "environment": {"GOMP_CPU_AFFINITY": "0-7", "OMP_WAIT_POLICY": "ACTIVE", "HOME": "/home/quentin",
                        "USER": "quentin", "OPENROUTER_API_KEY": FAKE_KEY,
                        "OXE_PF_TILES": 13, "LLAMA_CACHE": "/home/quentin/.cache/llama"},
        "hostname": "yogabox", "user": "quentin", SERIAL_KEY: "PF4ZZ9Q7", "disk": {DISK_KEY: "S6XNNA0W123456"},
        "net": {"mac": FAKE_MAC, "address": "192.168.1.44"}, "contact": "quentin@example.org",
        "machine_id": "0f8fad5b-d9cb-469f-a165-70867728950e", "password": "hunter2hunter2",
        "before": {"unix_s": 1794332465.1, "/sys/class/thermal/thermal_zone36/temp": "39600",
                   "/sys/class/thermal/thermal_zone26/temp": "42300", "thermal_zone2/temp": 37300},
        "guard_summary": {"ok": True, "rc": 0, "wall_s": 63.35, "stop_reason": None,
                          "scope_cgroup": "/sys/fs/cgroup/user.slice/user-1000.slice/user@1000.service/app.slice/run-p1.scope",
                          "xdg": "/run/user/1000/bus", "memory_max": "31138512896", "phase": "ready"},
        "result": {"ms_per_token": {"mean": 123.554, "median": 119.4, "max": 215.7}, "n_ctx": 4096, "threads": 8,
                   "threads_batch": 8, "kld": None, "path": "fast", "engine_commit": "5ac124343669", "cache_options": {"policy": "Clock"},
                   "decode_stage_ms_per_token": {"compute": 123.4, "inputs": 0.002}, "tokens_identical": False,
                   "prefill_from_file": False, "nan": float("nan"),
                   "ggml_prod_cpu_sha256": "adf384be7a9de1c0a4b3cdaf39efa84975df945ac9af8909a811579f8b8e1d01",
                   "utc": "2026-11-10T08:57:08.807429+00:00", "verdict": "PASS correctness; no >=3% speed gain"},
    }


class Sanitize(unittest.TestCase):
    def setUp(self):
        self.report = er.Report()
        self.clean = er.sanitize(private_receipt(), self.report)
        self.text = json.dumps(self.clean)

    def test_no_planted_private_value_survives(self):
        for plant in PLANTS:
            self.assertNotIn(plant, self.text, plant)
        self.assertEqual(er.scan_text(json.dumps(self.clean, indent=1), DENY), [])

    def test_numbers_hashes_and_identifiers_survive(self):
        c = self.clean
        self.assertEqual(c["binary_sha256"], SHA)
        self.assertEqual(c["source_commit"], "5ac1243436693e14bed75e0a4052cfca288ef89c")
        self.assertEqual(c["result"]["ms_per_token"]["mean"], 123.554)
        self.assertEqual(c["result"]["decode_stage_ms_per_token"]["inputs"], 0.002)
        self.assertEqual(c["result"]["threads"], 8)
        self.assertEqual(c["result"]["path"], "fast")
        self.assertEqual(c["profile"], "fast")
        self.assertEqual(c["result"]["cache_options"]["policy"], "Clock")
        self.assertEqual(c["result"]["verdict"], "PASS correctness; no >=3% speed gain")
        self.assertEqual(c["result"]["utc"], "2026-11-10T08:57:08.807429+00:00")
        self.assertIs(c["result"]["prefill_from_file"], False)
        self.assertIsNone(c["result"]["kld"])
        self.assertNotIn("nan", c["result"])
        self.assertEqual(c["prompt_tokens"], 156)
        self.assertEqual(c["guard_summary"]["memory_max"], "31138512896")
        self.assertEqual(c["label"], "01-A")

    def test_paths_reduce_to_model_and_program_names(self):
        self.assertEqual(self.clean["model_path"], "Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00004.gguf")
        self.assertEqual(self.clean["binary_path"], "oxe-flash")
        self.assertNotIn("raw_receipts_path", self.clean)

    def test_argv_keeps_flags_and_numbers_only(self):
        argv = self.clean["argv"]
        self.assertEqual(argv[:9], ["oxe-flash", "-m", "<path>/flash.gguf", "--cache-gib", "20", "-t", "8", "-C", "0x0ff"])
        self.assertEqual(argv[9], "-p")
        self.assertTrue(argv[10].startswith("<prompt bytes=25 sha256="))
        self.assertEqual(argv[11:13], ["--api-key", "<redacted>"])
        self.assertEqual(argv[13], "--out=<path>/result.json")
        self.assertEqual(argv[14:], ["-n", "128", "--temp", "0", "<text>"])
        self.assertEqual(self.clean["bench_argv"],
                         ["llama-bench", "-m", "<path>/m.gguf", "-p", "512", "-n", "128", "-r", "3"])

    def test_environment_is_allow_listed(self):
        self.assertEqual(self.clean["environment"], {"GOMP_CPU_AFFINITY": "0-7", "OMP_WAIT_POLICY": "ACTIVE",
                                                     "OXE_PF_TILES": "13", "LLAMA_CACHE": "<path>/llama"})

    def test_prompt_and_token_ids_become_digests(self):
        stub = self.clean["prompt_redacted"]
        self.assertEqual(stub["redacted"], "prompt")
        self.assertRegex(stub["sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(self.clean["prompt_ids_redacted"]["count"], 4)
        clean = er.sanitize(private_receipt(), er.Report(), {"prompt_hash": False})
        self.assertNotIn("sha256", clean["prompt_redacted"])
        self.assertNotIn("sha256", clean["prompt_ids_redacted"])
        self.assertNotIn("sha256=", " ".join(map(str, clean["argv"])))

    def test_thermal_map_collapses(self):
        self.assertEqual(self.clean["before"]["thermal_summary"], {"sensors": 3, "min_c": 37.3, "max_c": 42.3})
        self.assertNotIn("/sys/", self.text)

    def test_report_names_reasons_not_values(self):
        self.assertGreater(self.report.dropped["denied-key"], 3)
        self.assertIn("prompt", self.report.dropped)
        detail = json.dumps(self.report.detail)
        for plant in PLANTS:
            self.assertNotIn(plant, detail)

    def test_units_are_not_paths_but_paths_in_text_are(self):
        opts = {"allow_text": {"lines"}}
        clean = er.sanitize({"lines": ["verify-bench: 2 lanes: 35.696 ms/run, 1.587x one plain step",
                                       "loaded models/private.gguf", "see ~/x", "13/13 checks, 44.1 tok/s (108 GB/s)"],
                             "note": "8.9 GMAC/s vs models/x"}, er.Report(), opts)
        self.assertEqual(clean, {"lines": ["verify-bench: 2 lanes: 35.696 ms/run, 1.587x one plain step",
                                           "13/13 checks, 44.1 tok/s (108 GB/s)"]})

    def test_text_keys_refuse_paths_and_long_text(self):
        clean = er.sanitize({"verdict": "see /home/quentin/log", "note": "x" * 300, "kind": "a\nb", "title": "mail quentin@example.org"},
                            er.Report())
        self.assertEqual(clean, {})

    def test_random_trees_never_leak(self):
        rng = random.Random(20261110)
        keys = ["result", "meta", "note", "verdict", "label", "name", "data", "x", "run", "env", "argv", "model", "kind",
                "model_path", "title", "output", "status", "items", "config", "args", "command", "environment"]

        # Values a sanitiser can recognise: by shape, or because they are on the deny list.
        plants = [p for p in PLANTS if p not in ("hunter2hunter2", "PRIVATE-PROMPT-TEXT", "PRIVATE-ANSWER-TEXT",
                                                 "WINDOW-NOTE-TEXT")]

        def tree(depth):
            roll = rng.random()
            if depth > 3 or roll < 0.25:
                return rng.choice([rng.random(), rng.randint(0, 9), True, None, "ok", SHA, rng.choice(plants),
                                   "pre " + rng.choice(plants) + " post", rng.choice(plants) + "/x.gguf"])
            if roll < 0.5:
                return [tree(depth + 1) for _ in range(rng.randint(0, 4))]
            return {rng.choice(keys + ["/home/quentin/k", "PRIVATE-PROMPT-TEXT k"]): tree(depth + 1) for _ in range(rng.randint(0, 5))}

        for _ in range(400):
            clean = er.sanitize(tree(0), er.Report(), {"deny_words": DENY})
            text = json.dumps(clean, indent=1)
            for plant in plants:
                self.assertNotIn(plant, text)
            self.assertEqual(er.scan_text(text, DENY), [])


class Scan(unittest.TestCase):
    def rules(self, text, deny=None):
        return {rule for rule, _ in er.scan_text(text, deny)}

    def test_each_rule_fires(self):
        cases = {
            "home-path": ['"p": "/home/somebody/x"', "see /Users/me/models", "/root/.cache", "C:\\Users\\me\\x"],
            "tilde-path": ["~/.config/x", "$HOME/models", "${HOME}/x"],
            "abs-path": ["/sys/fs/cgroup/x", " /dev/nvme0n1", "/tmp/run.json", "/var/folders/ab"],
            "mount-path": ["/mnt/backup-disk/x", "/run/media/someone/USB", "/Volumes/Data"],
            "user-slice": ["user-1000.slice", "user@1000.service", "/run/user/1000/bus"],
            "mac-address": [FAKE_MAC, FAKE_MAC.upper().replace(":", "-")],
            "device-serial": ['"%s": "PF4ZZ9Q7"' % SERIAL_KEY, "ID_" + "SERIAL_SHORT=S6XNNA0W123456", "Service " + "Tag: ABC1234"],
            "uuid": ["0f8fad5b-d9cb-469f-a165-70867728950e"],
            "email": ["someone@example.org"],
            "private-ip": ["192.168.1.44", "10.0.0.7", "172.20.3.4"],
            "private-key": ["-----BEGIN OPENSSH " + "PRIVATE KEY-----"],
            "api-key": [FAKE_KEY, "ghp_" + "a" * 36, "hf_" + "B" * 34],
            "secret-assignment": ['"api_key": "abcd1234efgh"', "password=hunter2hunter2", "Authorization: Bearer12345678"],
            "price": ["costs $1299", "1 299 EUR".replace(" ", "", 1), "12000 kr", "€ 999".replace(" ", "")],
        }
        self.assertEqual(set(cases), set(er.RULES))
        for rule, samples in cases.items():
            for sample in samples:
                self.assertIn(rule, self.rules(sample), sample)

    def test_mount_points_and_unit_names_are_not_identities(self):
        for sample in ['"mount": "/home"', "on /home every tensor", "interconnect@26400000.interconnect",
                       "user@UID.service", "see /home/<owner>/x"]:
            self.assertEqual(self.rules(sample) - {"abs-path"}, set(), sample)

    def test_exact_spelling_exemption_keeps_the_login_form_flagged(self):
        deny = {"quentin": "user-name"}
        self.assertEqual(er.scan_text("approved by Quentin", deny, {"Quentin"}), [])
        self.assertEqual(er.scan_text("user quentin, approved by Quentin", deny, {"Quentin"}), [("user-name", 1)])

    def test_allow_file_needs_reasons_and_reports_unused_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            allow = Path(tmp) / "a.allow"
            allow.write_text("# c\n!allow-exact\tQuentin\towner\n!allow-word\tYogabox\tproduct\n"
                             "tests/*.py\temail\tfixtures\ndocs/x.md\t*\tinternal\n")
            entries, exact, words = er.load_allow_file(allow)
            self.assertEqual((exact, words), ({"Quentin"}, {"yogabox"}))
            hits = [(tmp + "/tests/t.py", "email", 3), (tmp + "/tests/t.py", "uuid", 4), (tmp + "/src/a.py", "email", 1)]
            left, unused = er.apply_allow(hits, entries, tmp)
            self.assertEqual(left, hits[1:])
            self.assertEqual([e[0] for e in unused], ["docs/x.md"])
            allow.write_text("tests/*.py\temail\n")
            with self.assertRaises(ValueError):
                er.load_allow_file(allow)

    def test_deny_words_are_whole_words_case_insensitive(self):
        self.assertIn("user-name", self.rules("run by Quentin today", DENY))
        self.assertIn("host-name", self.rules('"x": "yogabox"', DENY))
        self.assertNotIn("user-name", self.rules("quentinsson", DENY))

    def test_clean_public_text_passes(self):
        text = json.dumps({"argv": ["llama-bench", "-m", "<path>/m.gguf", "-p", "512"], "ms": 123.5, "sha": SHA,
                           "utc": "2026-11-10T08:57:08+00:00", "note": "8 threads, 20 GiB cache, 44.1 tok/s, GB/s",
                           "mac": "00:00:00:00:00:00", "version": "23.1.4", "url": "https://example.com/home/page"}, indent=1)
        self.assertEqual(er.scan_text(text, DENY), [])

    def test_hit_report_never_contains_the_value(self):
        hits = er.scan_text('"%s": "PF4ZZ9Q7" /home/quentin' % SERIAL_KEY, DENY)
        self.assertNotIn("PF4ZZ9Q7", repr(hits))
        self.assertNotIn("quentin", repr(hits))

    def test_system_deny_words_do_not_crash_and_skip_short_words(self):
        words = er.system_deny_words()
        self.assertTrue(all(len(w) >= 3 for w in words))


class Export(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        raw = self.dir / "runq999"
        (raw / "01-A" / "cache20").mkdir(parents=True)
        (raw / "receipts.json").write_text(json.dumps([private_receipt()]))
        (raw / "01-A" / "cache20" / "result.json").write_text(json.dumps(private_receipt()["result"]))
        (raw / "01-A" / "cache20" / "answer.txt").write_text("PRIVATE-ANSWER-TEXT")
        (raw / "01-A" / "cache20" / "samples.jsonl").write_text('{"text": "PRIVATE-ANSWER-TEXT"}\n')
        (raw / "broken.json").write_text("{not json PRIVATE-ANSWER-TEXT")
        self.raw = raw
        self.meta = {"ticket": "999", "title": "Fusion screen, 20 GiB", "evidence_class": "M", "n": 2,
                     "quality_class": "lossless", "model_name": "Qwen3.8-Flash-Next UD-IQ3_XXS",
                     "model_files": [("/home/quentin/models/a.gguf", SHA)], "notes": ["2 runs per arm, ABBA"]}

    def tearDown(self):
        self.tmp.cleanup()

    def test_export_writes_scan_clean_receipt(self):
        out = self.dir / "public" / "runq999.json"
        code, hits, report = er.export([str(self.raw)], out, self.meta, deny_words=DENY)
        self.assertEqual((code, hits), (0, []))
        text = out.read_text()
        for plant in PLANTS + [self.tmp.name]:
            self.assertNotIn(plant, text)
        doc = json.loads(text)
        self.assertEqual(doc["schema"], er.SCHEMA)
        self.assertEqual(doc["public_safe"], {"scanner": er.SCANNER, "hits": 0})
        self.assertEqual(doc["engine"]["binary_sha256"], SHA)
        self.assertEqual(doc["engine"]["commit"], "5ac1243436693e14bed75e0a4052cfca288ef89c")
        self.assertEqual(er.build_receipt([str(self.raw)], dict(self.meta), {})[0]["engine"]["commit"],
                         "5ac1243436693e14bed75e0a4052cfca288ef89c")
        self.assertEqual(doc["config"], {"n_ctx": 4096, "threads": 8, "threads_batch": 8})
        self.assertEqual(doc["model"]["files"], [{"name": "a.gguf", "sha256": SHA}])
        self.assertEqual([s["name"] for s in doc["sources"]], ["01-A/cache20/result.json", "receipts.json"])
        self.assertEqual(doc["export"]["files_not_read"], {".jsonl": 1, ".txt": 1})
        self.assertEqual(doc["export"]["dropped"]["not-json"], 1)
        self.assertEqual(er.scan_paths([str(out)], DENY), [])

    def test_deny_listed_text_is_dropped_and_counted(self):
        out = self.dir / "dropped.json"
        code, _, report = er.export([str(self.raw)], out, dict(self.meta, title="measured by quentin"), deny_words=DENY)
        self.assertEqual(code, 0)
        self.assertIsNone(json.loads(out.read_text())["title"])
        self.assertGreaterEqual(report.dropped["scan-rule"], 1)

    def test_export_is_fail_closed(self):
        """If the sanitiser ever lets something through, the final scan refuses to write."""
        out = self.dir / "refused.json"
        with mock.patch.object(er, "sanitize", lambda obj, report, *a, **k: obj):
            code, hits, _ = er.export([str(self.raw)], out, self.meta, deny_words=DENY)
        self.assertEqual(code, 3)
        self.assertTrue({"home-path", "user-name", "api-key", "mac-address"} <= {rule for rule, _ in hits})
        self.assertFalse(out.exists())

    def test_private_word_used_as_a_key_is_dropped(self):
        (self.raw / "keys.json").write_text(json.dumps({"quentin": 1, "kept": 2}))
        out = self.dir / "keys-out.json"
        code, _, _ = er.export([str(self.raw)], out, self.meta, deny_words=DENY)
        self.assertEqual(code, 0)
        self.assertIn({"name": "keys.json", "data": {"kept": 2}},
                      [{"name": s["name"], "data": s["data"]} for s in json.loads(out.read_text())["sources"]])

    def test_bad_meta_is_refused(self):
        with self.assertRaises(ValueError):
            er.export([str(self.raw)], self.dir / "x.json", dict(self.meta, binary_sha256="/home/quentin/bin"), deny_words=DENY)
        with self.assertRaises(ValueError):
            er.export([str(self.raw)], self.dir / "x.json", dict(self.meta, ticket="../../etc"), deny_words=DENY)
        self.assertFalse((self.dir / "x.json").exists())

    def test_hardware_file_is_sanitised_too(self):
        hw = self.dir / "hw.json"
        hw.write_text(json.dumps({"chip": "Snapdragon X Elite X1E80100", "ram_gb": 32, SERIAL_KEY: "PF4ZZ9Q7",
                                  "hostname": "yogabox"}))
        out = self.dir / "hw-out.json"
        code, _, _ = er.export([str(self.raw)], out, self.meta, hardware_path=str(hw), deny_words=DENY)
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out.read_text())["hardware"], {"chip": "Snapdragon X Elite X1E80100", "ram_gb": 32})

    def test_cli_export_and_scan(self):
        out = self.dir / "cli.json"
        run = subprocess.run([sys.executable, "-I", str(TOOL), "export", str(self.raw), "--out", str(out), "--ticket", "999",
                              "--evidence-class", "M", "--n", "2", "--deny-word", "quentin", "--deny-word", "yogabox",
                              "--no-system-denylist",
                              "--hardware", str(ROOT / "benchmarks/hardware/test-laptop.public.json")],
                             capture_output=True, text=True, timeout=60)
        self.assertEqual(run.returncode, 0, run.stderr)
        for plant in PLANTS:
            self.assertNotIn(plant, run.stdout + run.stderr)
        self.assertEqual(json.loads(out.read_text())["hardware"]["cpu_cores"], 12)
        ok = subprocess.run([sys.executable, "-I", str(TOOL), "scan", str(out), "--deny-word", "quentin", "--no-system-denylist"],
                            capture_output=True, text=True, timeout=60)
        self.assertEqual(ok.returncode, 0, ok.stdout)
        bad = subprocess.run([sys.executable, "-I", str(TOOL), "scan", str(self.raw), "--deny-word", "quentin"],
                             capture_output=True, text=True, timeout=60)
        self.assertEqual(bad.returncode, 1)
        self.assertIn("home-path", bad.stdout)
        for plant in PLANTS[1:]:
            self.assertNotIn(plant, bad.stdout.replace(str(self.raw), ""))


class TrackedExtracts(unittest.TestCase):
    """The hand-made numeric extracts already in the repo carry private paths; the exporter must clean them."""

    def test_existing_runq_extracts_export_clean(self):
        files = sorted((ROOT / "benchmarks" / "results").glob("runq*.json"))
        if not files:
            self.skipTest("no runq extracts in this tree")
        words = dict(DENY, **er.system_deny_words())
        with tempfile.TemporaryDirectory() as tmp:
            for path in files:
                out = Path(tmp) / path.name
                code, hits, _ = er.export([str(path)], out, {"ticket": path.name[4:7]}, deny_words=words)
                self.assertEqual(code, 0, "%s: %s" % (path.name, sorted({r for r, _ in hits})))
                self.assertEqual(er.scan_paths([str(out)], words), [])


if __name__ == "__main__":
    unittest.main()


class MacScriptAgreement(unittest.TestCase):
    """tools/mac_reproduce.py writes the same receipt schema, and its receipts pass this scanner."""

    def test_schema_and_empty_cell_scan_clean(self):
        path = ROOT / "tools" / "mac_reproduce.py"
        if not path.is_file():
            self.skipTest("mac_reproduce.py not in this tree")
        mac_spec = importlib.util.spec_from_file_location("mac_reproduce", path)
        mac = importlib.util.module_from_spec(mac_spec)
        mac_spec.loader.exec_module(mac)
        self.assertEqual(mac.RECEIPT_SCHEMA, er.SCHEMA)
        hardware = {"chip": "Apple M4 Max", "model_identifier": "Mac16,5", "performance_cores": 12,
                    "efficiency_cores": 4, "ram_gb": 64, "os": "macOS", "os_version": "15.1",
                    "is_apple_silicon": True, "power_source": "ac", "cpu_speed_limit": 100}
        for model_id in mac.MANIFEST:
            for stack in mac.STACKS:
                cell = mac.new_cell(model_id, stack, hardware)
                cell["runs"].append(mac.make_run(
                    "bench:pp512:r1", "bench", "pp512", None, 1, False, None, 12, 99,
                    mac.sanitize_argv(mac.bench_argv("/x/llama-bench", "/m/a.gguf", stack, 12, 3),
                                      {"/x/llama-bench": "<llama-dir>/build/bin", "/m/a.gguf": "<models-dir>"}),
                    0, prefill_tok_s=100.0))
                mac.finalize_cell(cell, "ok")
                self.assertEqual(er.scan_text(json.dumps(cell, indent=2), DENY), [], (model_id, stack))
