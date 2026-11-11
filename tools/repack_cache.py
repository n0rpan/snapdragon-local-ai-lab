#!/usr/bin/env python3
"""Inspect and prune the ggml-cpu persistent repack cache (GGML_CPU_REPACK_CACHE).

  repack_cache.py list  [--dir D] [--json]
  repack_cache.py prune [--dir D] [--older-than DAYS] [--build B] [--patch P] [--engine E] [--keep-latest] [--dry-run]
  repack_cache.py clear [--dir D] [--yes]

Each cache file (<dir>/<model>-<id>/<tensor>.<keyhash>.rpk) starts with a header page holding the full
cache key as text (model path/identity, build, patch, cpu features, repack layout, tensor shape).
Lab copy of the native cache maintenance tool; adds runtime-union filters.
"last use" is the file mtime, which ggml touches on every warm hit. See docs/repack-cache.md.
"""

import argparse
import json
import os
import stat
import struct
import sys
import time

MAGIC = b"GGMLRPC1"
HDR_FIXED = {1: 56, 2: 128}  # bytes before the key text, per format version
TMP_MAX_AGE = 3600  # seconds before a leftover temp file counts as abandoned

M64 = (1 << 64) - 1
P1, P2, P3, P4, P5 = (0x9E3779B185EBCA87, 0xC2B2AE3D27D4EB4F, 0x165667B19E3779F9,
                      0x85EBCA77C2B2AE63, 0x27D4EB2F165667C5)


def _rotl(x, r):
    return ((x << r) | (x >> (64 - r))) & M64


def _round(acc, v):
    acc = (acc + v * P2) & M64
    return (_rotl(acc, 31) * P1) & M64


def _merge(acc, v):
    acc ^= _round(0, v)
    return (acc * P1 + P4) & M64


def xxh64(data, seed=0):
    n = len(data)
    p = 0
    if n >= 32:
        v1, v2, v3, v4 = (seed + P1 + P2) & M64, (seed + P2) & M64, seed, (seed - P1) & M64
        while p + 32 <= n:
            a, b, c, d = struct.unpack_from("<4Q", data, p)
            v1, v2, v3, v4 = _round(v1, a), _round(v2, b), _round(v3, c), _round(v4, d)
            p += 32
        h = (_rotl(v1, 1) + _rotl(v2, 7) + _rotl(v3, 12) + _rotl(v4, 18)) & M64
        for v in (v1, v2, v3, v4):
            h = _merge(h, v)
    else:
        h = (seed + P5) & M64
    h = (h + n) & M64
    while p + 8 <= n:
        h ^= _round(0, struct.unpack_from("<Q", data, p)[0])
        h = (_rotl(h, 27) * P1 + P4) & M64
        p += 8
    if p + 4 <= n:
        h ^= (struct.unpack_from("<I", data, p)[0] * P1) & M64
        h = (_rotl(h, 23) * P2 + P3) & M64
        p += 4
    while p < n:
        h ^= (data[p] * P5) & M64
        h = (_rotl(h, 11) * P1) & M64
        p += 1
    h ^= h >> 33
    h = (h * P2) & M64
    h ^= h >> 29
    h = (h * P3) & M64
    h ^= h >> 32
    return h


def read_entry(path):
    e = {"path": path, "status": "ok", "key": {}}
    try:
        st = os.stat(path)
        e["file_size"] = st.st_size
        e["last_use"] = st.st_mtime
        with open(path, "rb") as f:
            head = f.read(56)
            if len(head) < 56 or head[:8] != MAGIC:
                e["status"] = "corrupt"
                return e
            version, hsize, psize, _phash, _khash, created, klen, flags = struct.unpack_from("<IIQQQQII", head, 8)
            if not 56 <= hsize <= 65536:
                e["status"] = "corrupt"
                return e
            f.seek(0)
            hdr = f.read(hsize)
        fixed = HDR_FIXED.get(version)
        if fixed is None or len(hdr) != hsize or fixed + klen > hsize - 8:
            e["status"] = "corrupt"
            return e
        if struct.unpack_from("<Q", hdr, hsize - 8)[0] != xxh64(hdr[:hsize - 8]):
            e["status"] = "corrupt"
            return e
        if st.st_size != hsize + psize:
            e["status"] = "truncated"
        e["payload_size"] = psize
        e["created"] = created / 1e9
        e["verified"] = version >= 2 and bool(flags & 1)
        for line in hdr[fixed:fixed + klen].decode("utf-8", "replace").splitlines():
            k, _, v = line.partition("=")
            e["key"][k] = v
    except OSError as ex:
        e["status"] = f"io-error: {ex}"
        return e
    if e["status"] == "ok":
        e["status"] = model_status(e["key"])
    return e


def model_status(key):
    if key.get("src") != "file":
        return "ok"
    model = key.get("model", "")
    try:
        st = os.stat(model)
    except OSError:
        return "stale-model-missing"
    mtime_ns = st.st_mtime_ns
    if (str(st.st_size) != key.get("model_size") or str(mtime_ns) != key.get("model_mtime_ns")
            or str(st.st_ino) != key.get("model_ino")
            or ("model_ctime_ns" in key and str(st.st_ctime_ns) != key["model_ctime_ns"])):
        return "stale-model-changed"
    return "ok"


def scan(root):
    entries, tmps = [], []
    info = os.lstat(root)
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid():
        raise ValueError('cache_root_must_be_owned_directory_not_symlink')
    for dirpath, dirs, files in os.walk(root):
        dirs[:] = [n for n in dirs if not os.path.islink(os.path.join(dirpath, n))]
        for n in files:
            p = os.path.join(dirpath, n)
            info = os.lstat(p)
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
                continue
            if n.endswith(".rpk"):
                entries.append(read_entry(p))
            elif ".rpk.tmp." in n:
                tmps.append(p)
    return entries, tmps


def human(n):
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024 or unit == "TiB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024


def fmt_time(t):
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(t)) if t else "-"


def cmd_list(args):
    entries, tmps = scan(args.dir)
    if args.json:
        json.dump({"entries": entries, "temp_files": tmps}, sys.stdout, indent=1, default=str)
        print()
        return 0
    groups = {}
    for e in entries:
        k = e["key"]
        g = (k.get("model", "(content-keyed)"), k.get("build", "?"), k.get("patch", "?"), k.get("engine", "-"))
        groups.setdefault(g, []).append(e)
    total = 0
    for (model, build, patch, engine), es in sorted(groups.items()):
        size = sum(e.get("file_size", 0) for e in es)
        total += size
        status = {}
        for e in es:
            status[e["status"]] = status.get(e["status"], 0) + 1
        print(f"model: {model}")
        print(f"  build: {build}  patch: {patch}  engine: {engine}")
        print(f"  tensors: {len(es)}  size: {human(size)}  last use: {fmt_time(max(e.get('last_use', 0) for e in es))}"
              f"  status: {', '.join(f'{k}={v}' for k, v in sorted(status.items()))}")
        if args.verbose:
            for e in sorted(es, key=lambda e: e["key"].get("tensor", "")):
                k = e["key"]
                print(f"    {k.get('tensor', '?'):48s} {k.get('type', '?'):7s} {k.get('repack', '?'):22s} "
                      f"{human(e.get('file_size', 0)):>10s}  {fmt_time(e.get('last_use'))}  {e['status']}"
                      f"{'' if e.get('verified') else ' (unverified)'}")
    print(f"total: {len(entries)} entries, {human(total)}, {len(tmps)} temp files, dir {args.dir}")
    return 0


def remove(path, dry, why):
    print(f"{'would remove' if dry else 'remove'} {path} ({why})")
    if not dry:
        try:
            os.unlink(path)
        except OSError as ex:
            print(f"  failed: {ex}", file=sys.stderr)


def remove_empty_dirs(root, dry):
    if dry:
        return
    for dirpath, _dirs, _files in os.walk(root, topdown=False):
        if dirpath != root:
            try:
                os.rmdir(dirpath)
            except OSError:
                pass


def cmd_prune(args):
    entries, tmps = scan(args.dir)
    now = time.time()
    doomed = {}
    engines = set(args.engine if isinstance(args.engine, list) else [args.engine]) if args.engine else set()
    model_shas = set(getattr(args, 'model_sha', None) or [])
    for e in entries:
        k = e["key"]
        if e["status"].startswith('io-error'):
            continue
        if e["status"] != "ok":
            doomed[e["path"]] = e["status"]
        elif args.build and k.get("build") != args.build:
            doomed[e["path"]] = "other-build"
        elif args.patch and k.get("patch") != args.patch:
            doomed[e["path"]] = "other-patch"
        elif engines and k.get("engine") not in engines:
            doomed[e["path"]] = "other-engine"
        elif model_shas and k.get("model_sha256") not in model_shas:
            doomed[e["path"]] = "other-model"
        elif args.older_than is not None and now - e.get("last_use", now) > args.older_than * 86400:
            doomed[e["path"]] = "unused"
    if args.keep_latest:
        latest = {}
        for e in entries:
            if e["path"] in doomed:
                continue
            g = (os.path.dirname(e["path"]), e["key"].get("tensor"), e["key"].get("engine"), e["key"].get("model_sha256"))
            if g not in latest or e.get("last_use", 0) > latest[g].get("last_use", 0):
                latest[g] = e
        keep = {e["path"] for e in latest.values()}
        for e in entries:
            if (not e["status"].startswith('io-error') and
                    e["path"] not in doomed and e["path"] not in keep):
                doomed[e["path"]] = "superseded"
    freed = 0
    for e in entries:
        if e["path"] in doomed:
            freed += e.get("file_size", 0)
            remove(e["path"], args.dry_run, doomed[e["path"]])
    for t in tmps:
        try:
            if now - os.stat(t).st_mtime > TMP_MAX_AGE:
                freed += os.stat(t).st_size
                remove(t, args.dry_run, "abandoned-temp")
        except OSError:
            pass
    remove_empty_dirs(args.dir, args.dry_run)
    print(f"{'would free' if args.dry_run else 'freed'} {human(freed)}")
    return 0


def cmd_clear(args):
    if not args.yes:
        print(f"refusing to clear {args.dir} without --yes", file=sys.stderr)
        return 1
    entries, tmps = scan(args.dir)
    for p in [e["path"] for e in entries] + tmps:
        remove(p, False, "clear")
    remove_empty_dirs(args.dir, False)
    return 0


def main():
    default_dir = os.environ.get("GGML_CPU_REPACK_CACHE") or os.path.expanduser("~/.cache/local-ai/repack")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("list")
    p.add_argument("--json", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true", help="one line per tensor")
    p = sub.add_parser("prune")
    p.add_argument("--older-than", type=float, metavar="DAYS", help="remove entries not used for DAYS")
    p.add_argument("--build", help="remove entries whose build differs (e.g. '0.29.0/954de24')")
    p.add_argument("--patch", help="remove entries whose patch identity differs")
    p.add_argument("--engine", action="append", help="remove entries whose GGML_CPU_REPACK_CACHE_ENGINE_ID differs")
    p.add_argument("--model-sha", action="append", help="keep this runtime model identity (repeatable)")
    p.add_argument("--keep-latest", action="store_true", help="per model and tensor keep only the most recently used entry")
    p.add_argument("--dry-run", action="store_true")
    p = sub.add_parser("clear")
    p.add_argument("--yes", action="store_true")
    for sp in sub.choices.values():
        sp.add_argument("--dir", default=default_dir)
    args = ap.parse_args()
    if not os.path.isdir(args.dir):
        print(f"no cache directory {args.dir}", file=sys.stderr)
        return 0 if args.cmd != "clear" else 1
    return {"list": cmd_list, "prune": cmd_prune, "clear": cmd_clear}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
