#!/usr/bin/env python3
"""Turn private run receipts into PUBLIC-SAFE receipts, and scan files for private data.

Usage:
  export_receipt.py export INPUT... --out FILE [--ticket NNN] [--title TEXT] [meta options]
  export_receipt.py scan PATH...          exit 1 on any hit; prints file:line and rule NAMES only
  export_receipt.py rules                 list the scan rules

`export` reads JSON receipts (files, or directories searched for *.json) and keeps only:
numbers, booleans, digests (hex), timestamps, short identifiers, a few allow-listed text
fields, sanitised argv and an allow-listed part of the environment. It drops prompts,
answers, logs, window notes, paths, cgroup names, serials and every other free text.
Model and binary paths are reduced to their file name. A prompt is replaced by its length
and SHA-256 (use --no-prompt-hash to drop even that).

The export is FAIL-CLOSED: the finished receipt is scanned, and if any rule hits, nothing
is written and the exit code is 3. Matched values are never printed.

Nothing here runs a model, reads model weights or touches the network. Stdlib only.
"""
import argparse
import datetime
import hashlib
import json
import math
import os
import re
import shlex
import sys
from pathlib import Path

SCHEMA = "snapdragon-local-ai-lab/receipt/v1"
SCANNER = "export_receipt/1"

# ---------------------------------------------------------------- scan rules
_HOME_DIRS = r"home|Users|root|var/home"
RULES = {
    # A user directory, in any spelling a receipt could carry.
    "home-path": re.compile(r"(?:^|(?<![A-Za-z0-9_.<>-]))/(?:" + _HOME_DIRS + r")/[A-Za-z0-9_.-]|[A-Za-z]:\\Users\\"),
    "tilde-path": re.compile(r"(?:^|(?<![A-Za-z0-9_.<>/-]))~/|\$\{?HOME\b|%USERPROFILE%"),
    # Any other absolute system path. Placeholders such as <path>/model.gguf do not match.
    "abs-path": re.compile(r"(?:^|(?<![A-Za-z0-9_.<>/-]))/(?:run|tmp|var|etc|opt|srv|dev|sys|proc|usr|boot|private|Library)/[A-Za-z0-9_.@-]"),
    "user-slice": re.compile(r"\buser-[0-9]+\.slice|\buser@[0-9]+\.service|/run/user/[0-9]+"),
    "mac-address": re.compile(r"(?<![0-9A-Fa-f:-])[0-9A-Fa-f]{2}([:-])(?:[0-9A-Fa-f]{2}\1){4}[0-9A-Fa-f]{2}(?![0-9A-Fa-f:-])"),
    "device-serial": re.compile(
        r"(?i)(?<![A-Za-z0-9])(?:id_serial(?:_short)?|(?:product|board|system|device|chassis|disk|hw|hardware)?[_ -]?serial(?:[_ -]?(?:number|no|num))?|"
        r"service[_ -]?tag|imei|hwid)[\"']?\s*[:=]\s*[\"']?[A-Za-z0-9][A-Za-z0-9._-]{5,}"),
    "uuid": re.compile(r"(?<![0-9A-Fa-f-])[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}(?![0-9A-Fa-f-])"),
    "email": re.compile(r"[A-Za-z0-9._%+-]+@(?![0-9a-fA-F]+\.)[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.(?!(?:service|slice|scope|mount|socket|target|timer)\b)[A-Za-z]{2,}"),
    # Removable or extra volumes are named after disks and labels.
    "mount-path": re.compile(r"(?:^|(?<![A-Za-z0-9_.<>/-]))/(?:mnt|media|run/media|Volumes)/[A-Za-z0-9_.@-]"),
    "private-ip": re.compile(r"(?<![0-9.])(?:10\.\d{1,3}|192\.168|172\.(?:1[6-9]|2\d|3[01])|169\.254|100\.(?:6[4-9]|[7-9]\d|1[01]\d|12[0-7]))\.\d{1,3}\.\d{1,3}(?![0-9.])"),
    "private-key": re.compile(r"-----BEGIN (?:[A-Z]+ )*PRIVATE KEY(?: BLOCK)?-----"),
    "api-key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}|\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{30,}|\bgithub_pat_[A-Za-z0-9_]{30,}|\bxox[bpas]-[A-Za-z0-9-]{10,}|\b(?:AKIA|ASIA)[0-9A-Z]{16}\b|\bhf_[A-Za-z0-9]{30,}"),
    "secret-assignment": re.compile(
        r"(?i)\b[a-z0-9_-]*(?:api[_-]?key|secret|passw(?:or)?d|bearer|authorization|access[_-]?token|refresh[_-]?token|auth[_-]?token)[\"']?\s*[:=]\s*[\"']?(?!<redacted>|null|none|false|true)[A-Za-z0-9_/+=.-]{8,}"),
    # Project rule: hardware prices never appear in public material.
    "price": re.compile(r"[$€£]\s?\d|\b\d[\d.,]*\s?(?:USD|EUR|SEK|NOK|DKK|GBP)\b|\b\d[\d.,]*\s?kr\b"),
}
_MAC_TRIVIAL = re.compile(r"^(?:00|ff|FF)(?:[:-](?:00|ff|FF)){5}$")


def system_deny_words():
    """Identifiers of THIS machine and user that must never appear in a public receipt.

    Returns {word: rule_name}. Every probe is best effort; unreadable sources are skipped."""
    words = {}

    def add(word, rule, minimum=3):
        word = (word or "").strip()
        if len(word) >= minimum and not word.isdigit():
            words.setdefault(word, rule)

    for env in ("USER", "LOGNAME", "SUDO_USER"):
        add(os.environ.get(env), "user-name")
    try:
        import pwd
        entry = pwd.getpwuid(os.getuid())
        add(entry.pw_name, "user-name")
        for part in re.split(r"[\s,]+", entry.pw_gecos or ""):
            add(part, "real-name", 4)
    except (ImportError, KeyError, OSError):
        pass
    try:
        import socket
        host = socket.gethostname()
        add(host, "host-name")
        add(host.split(".")[0], "host-name")
    except OSError:
        pass
    probes = [("/etc/machine-id", "machine-id"), ("/etc/hostname", "host-name")]
    for pattern, rule in (("/sys/class/dmi/id/*serial", "device-serial-value"),
                          ("/sys/class/dmi/id/product_uuid", "device-serial-value"),
                          ("/sys/class/nvme/*/serial", "disk-serial-value"),
                          ("/sys/block/*/device/serial", "disk-serial-value"),
                          ("/sys/class/net/*/address", "mac-address-value"),
                          ("/sys/firmware/devicetree/base/serial-number", "device-serial-value")):
        try:
            probes += [(str(p), rule) for p in Path("/").glob(pattern.lstrip("/"))]
        except OSError:
            pass
    for path, rule in probes:
        try:
            value = Path(path).read_text(errors="replace").strip().strip("\x00")
        except OSError:
            continue
        if rule == "mac-address-value" and (_MAC_TRIVIAL.match(value) or not value):
            continue
        add(value, rule, 6 if rule != "host-name" else 3)
    for word in os.environ.get("RECEIPT_DENY_WORDS", "").split(","):
        add(word, "deny-word")
    return words


def _word_rx(word):
    return re.compile(r"(?i)(?<![A-Za-z0-9])" + re.escape(word) + r"(?![A-Za-z0-9])")


# Rules that are too broad for source code and documentation, where system paths, "~/" and
# shell "$1" are legitimate. Receipts are always scanned with every rule.
TREE_PROFILE_SKIPS = ("abs-path", "tilde-path", "price", "binary-file")


def scan_text(text, deny_words=None, allow_exact=()):
    """Return a list of (rule_name, line_no). The matched value is never returned.

    allow_exact: spellings that are exempt from the per-machine word rules when they match
    with exactly this capitalisation (a person's name in prose, a product name)."""
    deny = [(_word_rx(w), rule) for w, rule in (deny_words or {}).items()]
    hits = []
    for no, line in enumerate(text.splitlines(), 1):
        line = line[:200_000]
        for name, rx in RULES.items():
            for match in rx.finditer(line):
                if name == "mac-address" and _MAC_TRIVIAL.match(match.group(0)):
                    continue
                hits.append((name, no))
                break
        for rx, rule in deny:
            if any(match.group(0) not in allow_exact for match in rx.finditer(line)):
                hits.append((rule, no))
    return hits


def load_allow_file(path):
    """Read exemptions: one per line, 'GLOB<TAB>RULE<TAB>REASON'. A reason is mandatory.

    Two directives use the same three columns: '!allow-exact<TAB>Spelling<TAB>REASON' and
    '!allow-word<TAB>word<TAB>REASON'. Returns (entries, exact_spellings, allowed_words)."""
    entries, exact, words = [], set(), set()
    for no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) != 3 or not all(part.strip() for part in parts):
            raise ValueError("%s:%d: expected GLOB<TAB>RULE<TAB>REASON" % (Path(path).name, no))
        if parts[0].strip() == "!allow-exact":
            exact.add(parts[1].strip())
        elif parts[0].strip() == "!allow-word":
            words.add(parts[1].strip().lower())
        else:
            entries.append((parts[0].strip(), parts[1].strip(), parts[2].strip()))
    return entries, exact, words


def apply_allow(hits, entries, base=None):
    """Drop exempted hits. Returns (remaining_hits, entries_that_matched_nothing)."""
    import fnmatch
    used, remaining = set(), []
    for path, rule, line in hits:
        rel = os.path.relpath(path, base) if base else path
        rel = rel.replace(os.sep, "/")
        match = next((i for i, (glob, allowed, _) in enumerate(entries)
                      if allowed in (rule, "*") and fnmatch.fnmatchcase(rel, glob)), None)
        if match is None:
            remaining.append((path, rule, line))
        else:
            used.add(match)
    return remaining, [entries[i] for i in range(len(entries)) if i not in used]


def scan_paths(paths, deny_words=None, allow_exact=()):
    """Scan files and directories. Returns list of (path, rule, line_no)."""
    out = []
    for raw in paths:
        root = Path(raw)
        files = sorted(p for p in root.rglob("*") if p.is_file() and ".git" not in p.parts) if root.is_dir() else [root]
        for path in files:
            try:
                data = path.read_bytes()
            except OSError:
                out.append((str(path), "unreadable", 0))
                continue
            if b"\x00" in data[:8192]:
                out.append((str(path), "binary-file", 0))
                continue
            out += [(str(path), rule, no)
                    for rule, no in scan_text(data.decode("utf-8", "replace"), deny_words, allow_exact)]
    return out


# ---------------------------------------------------------------- sanitiser
def _norm(key):
    return re.sub(r"[^a-z0-9]", "", str(key).lower())


# Identity and secret keys: dropped whatever the value is.
DENY_KEYS = {
    "home", "cwd", "pwd", "user", "login", "owner", "author", "email", "host", "node", "nodename", "machine",
    "bootid", "mac", "address", "ip", "url", "uri", "endpoint", "pid", "ppid", "tid", "uid", "gid", "evidence",
    "authorization", "bearer", "cookie", "sessionid", "sn", "wwid", "eui", "nguid", "imei", "hwid", "servicetag",
    "assettag", "devicename", "computername", "key",
}
DENY_KEY_PARTS = ("serial", "uuid", "hostname", "username", "passw", "secret", "apikey", "machineid", "macaddr",
                  "wwn", "credential", "privatekey")
_TOKEN_KEY = re.compile(r"^token$|(?:access|refresh|auth|hf|api|bearer|id|session|secret|bot|oauth)token$")
# Text keys: strings, lists and objects are dropped; a plain number or boolean may stay.
TEXT_DENY_KEYS = {
    "prompt", "prompts", "systemprompt", "system", "userprompt", "input", "inputs", "text", "texts", "answer",
    "answers", "content", "contents", "message", "messages", "response", "responses", "completion", "completions",
    "output", "outputs", "stdout", "stderr", "log", "logs", "lines", "tail", "transcript", "window", "comment",
    "comments", "description", "reasoning", "thinking", "toolcalls", "tools", "choices", "body", "request",
    "scope", "scopecgroup", "cgroup", "session",
}
PROMPT_KEYS = {"prompt", "systemprompt", "userprompt", "input"}
# A list under one of these keys is a token-id sequence, which is the text itself.
_ID_LIST_KEY = re.compile(r"(?:ids|tokenids|tokens|promptids|replyids)$")
# Strings under keys whose LAST WORD is one of these are paths: only a model/binary file NAME may survive.
_PATH_KEY = re.compile(r"(?:^|[_. -])(?:path|paths|dir|dirs|directory|file|files|root|folder|location)$|[a-z](?:Path|Paths|Dir|File|Files|Root)$")
# Free text is kept only under these keys, and only if it passes _safe_text.
TEXT_KEYS = {
    "verdict", "kind", "classification", "title", "model", "modelname", "quant", "quantization", "name", "label",
    "workload", "status", "phase", "policy", "class", "qualityclass", "evidenceclass", "note", "notes", "caveat",
    "comparisoncaveat", "scopenote", "chip", "os", "osversion", "storage", "memory", "engine", "stack", "test",
    "metric", "unit", "arm", "variant", "step", "laptop", "powersource",
}
ARGV_KEYS = {"argv", "args", "command", "cmd", "cmdline", "commandline", "serverargs", "engineargs", "nativeargv"}
ENV_KEYS = {"environment", "env", "environ"}
ENV_ALLOW = re.compile(r"^(?:OMP_|GOMP_|KMP_|OXE_|GGML_|LLAMA_|LOCAL_AI_|RAYON_|MALLOC_|OPENBLAS_|VECLIB_|MLX_)[A-Z0-9_]*$")
MODEL_EXT = (".gguf", ".safetensors", ".npz", ".dtbo")
_PROGRAM = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,47}$")

_HEX = re.compile(r"^(?:sha256:)?[0-9a-fA-F]{7,64}(?:…|\.\.\.)?$")
_NUM = re.compile(r"^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$")
_IDENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.+:=,x%-]{0,63}$")
_TIME = re.compile(r"^\d{4}-\d\d-\d\d(?:[T ]\d\d:\d\d(?::\d\d(?:\.\d+)?)?(?:Z|[+-]\d\d:?\d\d)?)?$")
_KEY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:+%-]{0,79}$")
_THERMAL = re.compile(r"(?:^|/)thermal_zone\d+/temp$")
_FILE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.+=,-]{0,159}$")
_PROMPT_FLAGS = {"-p", "--prompt", "-sys", "--system-prompt", "--system", "--grammar", "--json-schema", "-j",
                 "--chat-template", "--reverse-prompt", "-r", "--in-prefix", "--in-suffix", "--negative-prompt"}
_SECRET_FLAGS = {"--api-key", "--hf-token", "-hft", "--token", "--password", "--ssl-key-file", "--api-key-file"}


class Report:
    def __init__(self):
        self.kept = 0
        self.dropped = {}
        self.detail = []          # (reason, json-pointer); key names only, never values

    def drop(self, reason, pointer):
        self.dropped[reason] = self.dropped.get(reason, 0) + 1
        self.detail.append((reason, pointer))


_UNIT_SLASH = re.compile(r"^[(\[]?(?:[A-Za-zµ%]{1,8}/(?:s|sec|ms|us|min|h|token|tokens|tok|run|step|core|thread|pass|layer|row|block|request|req|image|px|W|GB|GiB|MiB|K|1K)|\d{1,6}/\d{1,6})[)\].,;:]*$")


def _is_pathish(value):
    """True for anything that could be a path. Units and counts (ms/token, GB/s, 13/13) are not paths."""
    if "\\" in value or value.startswith("~"):
        return True
    return any("/" in word and not _UNIT_SLASH.match(word) for word in value.split())


def _denied_key(k):
    return k in DENY_KEYS or any(part in k for part in DENY_KEY_PARTS) or _TOKEN_KEY.search(k) is not None


def _argv_key(k):
    return k in ARGV_KEYS or k.endswith("argv")


def _digest_stub(kind, value, opts):
    data = value if isinstance(value, str) else json.dumps(value, sort_keys=True, ensure_ascii=False)
    stub = {"redacted": kind, "bytes": len(data.encode("utf-8"))}
    if isinstance(value, list):
        stub["count"] = len(value)
    if opts.get("prompt_hash", True):
        stub["sha256"] = hashlib.sha256(data.encode("utf-8")).hexdigest()
    return stub


def _safe_text(value, limit=240):
    """Short human text with no path, address or markup that could hide one."""
    if len(value) > limit or "\n" in value or "\r" in value:
        return False
    if _is_pathish(value) or "@" in value or "~" in value:
        return False
    return re.fullmatch(r"[A-Za-z0-9 /_.,;:+=()%<>\[\]'*×≤≥→±°µ#!?&|^-]*", value) is not None


def _file_name(value):
    """Basename of a model/binary path if it is a plain file name, else None."""
    name = re.split(r"[/\\]", value.rstrip("/\\"))[-1]
    return name if _FILE_NAME.match(name) and ".." not in name else None


def _prompt_stub(value, opts):
    return _digest_stub("prompt", value, opts)


def _argv_value(value, opts):
    if _NUM.match(value) or _HEX.match(value):
        return value
    if _is_pathish(value):
        name = _file_name(value)
        return "<path>/" + name if name and not scan_text(name, opts.get("deny_words")) else "<path>"
    if _IDENT.match(value) and not scan_text(value, opts.get("deny_words")):
        return value
    return "<text>"


def sanitize_argv(argv, opts=None):
    """Keep flags, numbers and identifiers; reduce paths to <path>/name; drop prompts and secrets."""
    opts = opts or {}
    if isinstance(argv, str):
        try:
            argv = shlex.split(argv)
        except ValueError:
            return ["<unparsed>"]
    out, prev = [], None
    for index, item in enumerate(argv):
        if not isinstance(item, str):
            out.append(item if isinstance(item, (int, float)) and not isinstance(item, bool) else "<text>")
            prev = None
            continue
        flag, eq, inline = item.partition("=")
        if item.startswith("-") and not _NUM.match(item):
            if not re.fullmatch(r"--?[A-Za-z0-9][A-Za-z0-9_.-]*", flag):
                out.append("<text>")
                prev = None
                continue
            if eq:
                if flag in _SECRET_FLAGS:
                    out.append(flag + "=<redacted>")
                elif flag in _PROMPT_FLAGS and not _NUM.match(inline):
                    out.append(flag + "=<prompt>")
                else:
                    out.append(flag + "=" + _argv_value(inline, opts))
                prev = None
            else:
                out.append(flag)
                prev = flag
            continue
        if prev in _SECRET_FLAGS:
            out.append("<redacted>")
        elif prev in _PROMPT_FLAGS and not _NUM.match(item):
            stub = _prompt_stub(item, opts)
            out.append("<prompt bytes=%d%s>" % (stub["bytes"], " sha256=" + stub["sha256"] if "sha256" in stub else ""))
        elif eq and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", flag):
            # NAME=value: an environment assignment in front of the program, or a key=value argument
            out.append(flag + "=" + (_argv_value(inline, opts) if ENV_ALLOW.match(flag) else "<redacted>"))
        elif index == 0 or (out and all("=" in str(o) for o in out)):
            name = _file_name(item)
            safe = name and _PROGRAM.match(name) and not scan_text(name, opts.get("deny_words"))
            out.append(name if safe else "<program>")
        else:
            out.append(_argv_value(item, opts))
        prev = None
    return out


def _thermal(entries):
    values = []
    for value in entries:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            values.append(number / 1000.0 if abs(number) > 1000 else number)
    if not values:
        return None
    return {"sensors": len(values), "min_c": round(min(values), 1), "max_c": round(max(values), 1)}


def sanitize(obj, report, opts=None, pointer="", key=None):
    """Return a public-safe copy of obj. Dropped members are counted in report."""
    opts = opts or {}
    nkey = _norm(key) if key is not None else ""
    if obj is None or isinstance(obj, bool):
        report.kept += 1
        return obj
    if isinstance(obj, (int, float)):
        if isinstance(obj, float) and not math.isfinite(obj):
            report.drop("non-finite-number", pointer)
            return None
        report.kept += 1
        return obj
    if isinstance(obj, str):
        if _NUM.match(obj) or _TIME.match(obj):
            report.kept += 1
            return obj
        if not _is_pathish(obj) and scan_text(obj, opts.get("deny_words")):
            report.drop("scan-rule", pointer)
            return None
        if _HEX.match(obj):
            report.kept += 1
            return obj
        if nkey == "path" and re.fullmatch(r"[a-z][a-z0-9_-]{0,23}", obj):
            report.kept += 1          # a code-path label such as "fast" or "exact", not a file path
            return obj
        if (key is not None and _PATH_KEY.search(str(key))) or _is_pathish(obj):
            name = _file_name(obj)
            if name and nkey.startswith(("model", "mtp", "head", "sidecar", "shard", "weights")) and name.lower().endswith(MODEL_EXT):
                report.kept += 1
                return name
            if name and nkey.startswith(("binary", "server", "cli", "engine", "program")) and _PROGRAM.match(name):
                report.kept += 1
                return name
            report.drop("path", pointer)
            return None
        if nkey in TEXT_KEYS or nkey in opts.get("allow_text", ()):
            if _safe_text(obj):
                report.kept += 1
                return obj
            report.drop("unsafe-text", pointer)
            return None
        if _IDENT.match(obj):
            report.kept += 1
            return obj
        report.drop("free-text", pointer)
        return None
    if isinstance(obj, list):
        if _argv_key(nkey):
            report.kept += 1
            return sanitize_argv(obj, opts)
        out = []
        for index, item in enumerate(obj):
            value = sanitize(item, report, opts, "%s/%d" % (pointer, index), key)
            if value is not None or item is None:
                out.append(value)
        return out
    if isinstance(obj, dict):
        out, thermal = {}, []
        for raw_key, value in obj.items():
            raw_key = str(raw_key)
            here = pointer + "/" + (raw_key if _KEY.match(raw_key) else "<key>")
            k = _norm(raw_key)
            if _THERMAL.search(raw_key):
                thermal.append(value)
                continue
            if not _KEY.match(raw_key) or _is_pathish(raw_key) or scan_text(raw_key, opts.get("deny_words")):
                report.drop("unsafe-key", here)
                continue
            if _denied_key(k):
                report.drop("denied-key", here)
                continue
            if k in PROMPT_KEYS and isinstance(value, (str, list)) and value:
                out[_public_key(raw_key) + "_redacted"] = _prompt_stub(value, opts)
                report.drop("prompt", here)
                continue
            if isinstance(value, list) and _ID_LIST_KEY.search(k) and value and all(
                    isinstance(v, int) and not isinstance(v, bool) for v in value):
                out[_public_key(raw_key) + "_redacted"] = _digest_stub("token-ids", value, opts)
                report.drop("token-ids", here)
                continue
            if k in TEXT_DENY_KEYS and k not in opts.get("allow_text", ()) and isinstance(value, (str, list, dict)):
                report.drop("denied-text-key", here)
                continue
            if _argv_key(k) and isinstance(value, (list, str)):
                out[raw_key] = sanitize_argv(value, opts)
                report.kept += 1
                continue
            if k in ENV_KEYS and isinstance(value, dict):
                env = {}
                for name, val in value.items():
                    if isinstance(name, str) and ENV_ALLOW.match(name) and isinstance(val, (str, int, float)):
                        env[name] = _argv_value(str(val), opts)
                        report.kept += 1
                    else:
                        report.drop("environment", here + "/<name>")
                out[raw_key] = env
                continue
            if _PATH_KEY.search(raw_key) and isinstance(value, (list, dict)):
                report.drop("path", here)
                continue
            clean = sanitize(value, report, opts, here, raw_key)
            if clean is not None or value is None:
                out[raw_key] = clean
        if thermal:
            summary = _thermal(thermal)
            if summary:
                out["thermal_summary"] = summary
                report.kept += 1
        return out
    report.drop("unsupported-type", pointer)
    return None


def _public_key(key):
    return re.sub(r"[^A-Za-z0-9_]", "_", key)[:60]


# ---------------------------------------------------------------- export
_HARVEST = {
    "binary_sha256": ("binary_sha256", "server_sha256", "cli_sha256"),
    "engine_commit": ("engine_commit", "source_commit", "commit"),
    "model": ("model", "model_name"),
    "quant": ("quant", "quantization"),
    "threads": ("threads",),
    "threads_batch": ("threads_batch", "batch_threads"),
    "n_ctx": ("n_ctx", "ctx", "context"),
}


def _harvest(tree, found):
    if isinstance(tree, dict):
        for field, names in _HARVEST.items():
            for name in names:
                value = tree.get(name)
                if value is not None and not isinstance(value, (dict, list)):
                    found.setdefault(field, set()).add(value)
        for value in tree.values():
            _harvest(value, found)
    elif isinstance(tree, list):
        for value in tree:
            _harvest(value, found)


def collect_inputs(inputs):
    """Yield (public_name, path) for every JSON receipt; other files are counted, not read."""
    files, skipped = [], {}
    for raw in inputs:
        root = Path(raw)
        if root.is_dir():
            for path in sorted(root.rglob("*")):
                if not path.is_file():
                    continue
                if path.suffix.lower() == ".json":
                    files.append(("/".join(path.relative_to(root).parts), path))
                else:
                    skipped[path.suffix.lower() or "(none)"] = skipped.get(path.suffix.lower() or "(none)", 0) + 1
        elif root.is_file():
            files.append((root.name, root))
        else:
            raise FileNotFoundError("no such receipt: %s" % root.name)
    return files, skipped


def _public_name(name):
    parts = [p if _FILE_NAME.match(p) else "file" for p in name.split("/")]
    return "/".join(parts)


def _hardware(path, report, opts):
    if not path:
        return None
    return sanitize(json.loads(Path(path).read_text(encoding="utf-8")), report, opts, "/hardware")


def build_receipt(inputs, meta, opts=None, hardware=None, now=None):
    """Return (receipt_dict, Report). Does not scan and does not write."""
    opts = opts or {}
    report = Report()
    files, skipped = collect_inputs(inputs)
    sources, found = [], {}
    for name, path in files:
        raw = path.read_bytes()
        try:
            tree = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            report.drop("not-json", "/" + _public_name(name))
            continue
        clean = sanitize(tree, report, opts, "/" + _public_name(name))
        _harvest(clean, found)
        entry = {"name": _public_name(name), "data": clean}
        if opts.get("source_hash", True):
            entry["private_file_sha256"] = hashlib.sha256(raw).hexdigest()
        sources.append(entry)

    def single(field):
        values = found.get(field) or set()
        if len(values) > 1 and all(isinstance(v, str) for v in values):
            longest = max(values, key=len)          # short and full forms of one commit or digest
            if all(longest.startswith(v) for v in values):
                return longest
        return next(iter(values)) if len(values) == 1 else None

    def text(value, pointer):
        if value is None:
            return None
        return sanitize(value, report, opts, pointer, "note")

    def digest(value, pointer):
        if value is None:
            return None
        if not isinstance(value, str) or not _HEX.match(value):
            raise ValueError("%s must be a hex digest" % pointer)
        return value

    stamp = (now or datetime.datetime.now(datetime.timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ")
    ticket = meta.get("ticket")
    if ticket is not None and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,31}", str(ticket)):
        raise ValueError("ticket must be a short identifier")
    evidence_class = meta.get("evidence_class")
    if evidence_class not in (None, "M", "S", "E"):
        raise ValueError("evidence class must be M, S or E")
    receipt = {
        "schema": SCHEMA,
        "kind": "runq-export",
        "ticket": ticket,
        "title": text(meta.get("title"), "/title"),
        "created_utc": stamp,
        "hardware": hardware,
        "engine": {
            "name": text(meta.get("engine_name"), "/engine/name"),
            "commit": digest(meta.get("engine_commit") or single("engine_commit"), "engine commit"),
            "binary_sha256": digest(meta.get("binary_sha256") or single("binary_sha256"), "binary sha256"),
        },
        "model": {
            "name": text(meta.get("model_name") or single("model"), "/model/name"),
            "quant": text(meta.get("quant") or single("quant"), "/model/quant"),
            "files": [{"name": _file_name(n) or "file", "sha256": digest(s, "model sha256")}
                      for n, s in meta.get("model_files", [])],
        },
        "config": {"n_ctx": single("n_ctx"), "threads": single("threads"), "threads_batch": single("threads_batch")},
        "evidence": {"class": evidence_class, "n": meta.get("n"),
                     "quality_class": text(meta.get("quality_class"), "/evidence/quality_class")},
        "notes": [n for n in (text(note, "/notes") for note in meta.get("notes", [])) if n],
        "sources": sources,
        "export": {"scanner": SCANNER, "values_kept": report.kept, "dropped": dict(sorted(report.dropped.items())),
                   "files_not_read": dict(sorted(skipped.items())), "prompt_hashes": bool(opts.get("prompt_hash", True))},
    }
    return receipt, report


def export(inputs, out_path, meta, opts=None, hardware_path=None, deny_words=None, now=None):
    """Build, scan and write. Returns (exit_code, hits, Report). Writes nothing on a hit."""
    opts = dict(opts or {}, deny_words=deny_words)
    report = Report()
    hardware = _hardware(hardware_path, report, opts)
    receipt, body = build_receipt(inputs, meta, opts, hardware, now)
    body.kept += report.kept
    for reason, count in report.dropped.items():
        body.dropped[reason] = body.dropped.get(reason, 0) + count
    body.detail += report.detail
    text = json.dumps(receipt, indent=1, ensure_ascii=False, sort_keys=False) + "\n"
    hits = scan_text(text, deny_words)
    if hits:
        return 3, hits, body
    receipt["public_safe"] = {"scanner": SCANNER, "hits": 0}
    text = json.dumps(receipt, indent=1, ensure_ascii=False) + "\n"
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    return 0, [], body


# ---------------------------------------------------------------- CLI
def _deny(args):
    words = {} if args.no_system_denylist else system_deny_words()
    for word in getattr(args, "allow_word", None) or []:
        for known in [w for w in words if w.lower() == word.lower()]:
            del words[known]
    for word in args.deny_word or []:
        if len(word) >= 3:
            words[word] = "deny-word"
    for path in args.deny_file or []:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if len(line.strip()) >= 3 and not line.startswith("#"):
                words[line.strip()] = "deny-word"
    return words


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    def deny_options(p):
        p.add_argument("--deny-word", action="append", help="extra word that must not appear (repeatable)")
        p.add_argument("--deny-file", action="append", help="file with one forbidden word per line")
        p.add_argument("--allow-word", action="append",
                       help="exempt one machine identifier that is also an ordinary public word, "
                            "for example a host named after the laptop model (repeatable; use sparingly)")
        p.add_argument("--no-system-denylist", action="store_true",
                       help="do not add this machine's user, host and device identifiers (tests only)")

    ex = sub.add_parser("export", help="private receipts -> one public-safe receipt")
    ex.add_argument("inputs", nargs="+")
    ex.add_argument("--out", required=True)
    ex.add_argument("--ticket")
    ex.add_argument("--title")
    ex.add_argument("--engine-name")
    ex.add_argument("--engine-commit")
    ex.add_argument("--binary-sha256")
    ex.add_argument("--model-name")
    ex.add_argument("--quant")
    ex.add_argument("--model-file", action="append", default=[], metavar="NAME=SHA256")
    ex.add_argument("--evidence-class", choices=["M", "S", "E"])
    ex.add_argument("--n", type=int, help="runs per arm")
    ex.add_argument("--quality-class", choices=["lossless", "reorder", "lossy"])
    ex.add_argument("--note", action="append", default=[])
    ex.add_argument("--hardware", help="JSON file with the public hardware description")
    ex.add_argument("--allow-text-key", action="append", default=[],
                    help="also keep short safe text under this key (still scanned)")
    ex.add_argument("--no-prompt-hash", action="store_true")
    ex.add_argument("--no-source-hash", action="store_true")
    ex.add_argument("--private-report", help="write the list of dropped members (key names only) to this PRIVATE file")
    deny_options(ex)

    sc = sub.add_parser("scan", help="scan files or directories; exit 1 on hits")
    sc.add_argument("paths", nargs="+")
    sc.add_argument("--skip-rule", action="append", default=[], help="rule name to ignore (repeatable)")
    sc.add_argument("--summary", action="store_true", help="print counts per rule and per file only")
    sc.add_argument("--profile", choices=["receipt", "tree"], default="receipt",
                    help="tree: for source and documentation; skips " + ", ".join(TREE_PROFILE_SKIPS))
    sc.add_argument("--allow-exact", action="append", default=[],
                    help="exact spelling exempt from the per-machine word rules, e.g. a name in prose (repeatable)")
    sc.add_argument("--allow-file", help="exemptions, one per line: GLOB<TAB>RULE<TAB>REASON")
    sc.add_argument("--base", help="directory the globs in --allow-file are relative to (default: current directory)")
    deny_options(sc)

    sub.add_parser("rules", help="list scan rules")
    args = parser.parse_args(argv)

    if args.cmd == "rules":
        for name in RULES:
            print(name)
        print("plus per-machine words: user-name, real-name, host-name, machine-id, device-serial-value, "
              "disk-serial-value, mac-address-value, deny-word")
        return 0

    if args.cmd == "scan":
        skips = set(args.skip_rule) | (set(TREE_PROFILE_SKIPS) if args.profile == "tree" else set())
        entries, exact, allowed_words = load_allow_file(args.allow_file) if args.allow_file else ([], set(), set())
        deny = {w: r for w, r in _deny(args).items() if w.lower() not in allowed_words}
        hits = [h for h in scan_paths(args.paths, deny, set(args.allow_exact) | exact) if h[1] not in skips]
        if args.allow_file:
            hits, unused = apply_allow(hits, entries, args.base or os.getcwd())
            for glob, rule, _ in unused:
                print("unused exemption: %s %s" % (glob, rule), file=sys.stderr)
        if args.summary:
            by_rule, by_file = {}, {}
            for path, rule, _ in hits:
                by_rule[rule] = by_rule.get(rule, 0) + 1
                by_file[path] = by_file.get(path, 0) + 1
            for rule, count in sorted(by_rule.items(), key=lambda kv: -kv[1]):
                print("%6d  %s" % (count, rule))
            print("%d hits in %d files" % (len(hits), len(by_file)))
        else:
            for path, rule, line in hits:
                print("%s:%d: %s" % (path, line, rule))
        return 1 if hits else 0

    model_files = []
    for item in args.model_file:
        name, sep, digest = item.partition("=")
        if not sep:
            parser.error("--model-file needs NAME=SHA256")
        model_files.append((name, digest))
    meta = {"ticket": args.ticket, "title": args.title, "engine_name": args.engine_name,
            "engine_commit": args.engine_commit, "binary_sha256": args.binary_sha256, "model_name": args.model_name,
            "quant": args.quant, "model_files": model_files, "evidence_class": args.evidence_class, "n": args.n,
            "quality_class": args.quality_class, "notes": args.note}
    opts = {"prompt_hash": not args.no_prompt_hash, "source_hash": not args.no_source_hash,
            "allow_text": {_norm(k) for k in args.allow_text_key}}
    try:
        code, hits, report = export(args.inputs, args.out, meta, opts, args.hardware, _deny(args))
    except (ValueError, FileNotFoundError) as error:
        print("export refused: %s" % error, file=sys.stderr)
        return 2
    if args.private_report:
        Path(args.private_report).write_text(
            "".join("%s\t%s\n" % item for item in report.detail), encoding="utf-8")
    dropped = ", ".join("%s=%d" % kv for kv in sorted(report.dropped.items())) or "nothing"
    print("kept %d values; dropped: %s" % (report.kept, dropped), file=sys.stderr)
    if code:
        rules = sorted({rule for rule, _ in hits})
        print("export REFUSED, nothing written. Scan rules hit: %s (output lines %s)" % (
            ", ".join(rules), ", ".join(str(no) for _, no in hits[:20])), file=sys.stderr)
        return code
    print("wrote %s" % Path(args.out).name, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
