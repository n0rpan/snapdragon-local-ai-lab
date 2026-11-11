#!/usr/bin/env python3
"""Rewrite private machine details in tracked receipt files into portable forms, in place.

Usage:
  scrub_receipts.py [--check] [--home DIR] [--user NAME] PATH...

What it rewrites, as plain text so that file layout and every number stay untouched:
  HOME/Work/local-ai            -> $LAB_ROOT
  HOME/.local/share/ai-stack    -> $AI_STACK
  HOME/.local/state/local-ai    -> $LOCAL_AI_STATE
  HOME                          -> $HOME
  /run/user/<uid>               -> $XDG_RUNTIME_DIR
  user-<uid>.slice, user@<uid>.service -> user-UID.slice, user@UID.service
  any UUID (boot IDs, filesystem UUIDs)  -> <redacted-uuid>
  the login name as a whole lower-case word -> user

HOME defaults to this user's home directory and NAME to the login name. The placeholders are
names only; nothing reads them. --check changes nothing and exits 1 if a file would change.

Text files only (.json .jsonl .txt .csv .md .log .patch and files without a NUL byte).
It never reads model weights and runs no model. Stdlib only.
"""
import argparse
import os
import re
import sys
from pathlib import Path

UUID = re.compile(r"(?<![0-9A-Fa-f-])[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}(?![0-9A-Fa-f-])")
RUN_USER = re.compile(r"/run/user/[0-9]+")
USER_SLICE = re.compile(r"\buser-[0-9]+\.slice")
USER_SERVICE = re.compile(r"\buser@[0-9]+\.service")
ROOTS = (("/Work/local-ai", "$LAB_ROOT"), ("/.local/share/ai-stack", "$AI_STACK"),
         ("/.local/state/local-ai", "$LOCAL_AI_STATE"), ("", "$HOME"))


def scrub_text(text, home, user):
    """Return (new_text, {rule: count})."""
    counts = {}

    def sub(rule, pattern, replacement, value):
        new, n = pattern.subn(replacement, value)
        if n:
            counts[rule] = counts.get(rule, 0) + n
        return new

    home = home.rstrip("/")
    if home and home != "/":
        for suffix, name in ROOTS:
            # the directory itself or a path below it, never a longer sibling name
            pattern = re.compile(re.escape(home + suffix) + r"(?![A-Za-z0-9_.-])")
            text = sub("home-path", pattern, lambda _m, name=name: name, text)
        # the same directory spelled as a flattened name, e.g. "-home-name-" in tool state folders
        flat = home.replace("/", "-")
        text = sub("home-path", re.compile(re.escape(flat) + r"(?![A-Za-z0-9_.])"), "-HOME", text)
    text = sub("user-slice", RUN_USER, "$XDG_RUNTIME_DIR", text)
    text = sub("user-slice", USER_SLICE, "user-UID.slice", text)
    text = sub("user-slice", USER_SERVICE, "user@UID.service", text)
    text = sub("uuid", UUID, "<redacted-uuid>", text)
    if user and len(user) >= 3:
        pattern = re.compile(r"(?<![A-Za-z0-9])" + re.escape(user) + r"(?![A-Za-z0-9])")
        text = sub("user-name", pattern, "user", text)
    return text, counts


def iter_files(paths):
    for raw in paths:
        root = Path(raw)
        if root.is_dir():
            yield from sorted(p for p in root.rglob("*") if p.is_file() and ".git" not in p.parts)
        elif root.is_file():
            yield root


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--check", action="store_true", help="change nothing; exit 1 if anything would change")
    parser.add_argument("--home", default=os.path.expanduser("~"))
    parser.add_argument("--user", default=os.environ.get("USER") or os.environ.get("LOGNAME") or "")
    args = parser.parse_args(argv)
    total, changed = {}, 0
    for path in iter_files(args.paths):
        data = path.read_bytes()
        if b"\x00" in data[:8192]:
            continue
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            continue
        new, counts = scrub_text(text, args.home, args.user)
        if new == text:
            continue
        changed += 1
        for rule, n in counts.items():
            total[rule] = total.get(rule, 0) + n
        if not args.check:
            path.write_bytes(new.encode("utf-8"))
    summary = ", ".join("%s=%d" % kv for kv in sorted(total.items())) or "nothing"
    print("%s %d files; replacements: %s" % ("would change" if args.check else "changed", changed, summary))
    return 1 if args.check and changed else 0


if __name__ == "__main__":
    sys.exit(main())
