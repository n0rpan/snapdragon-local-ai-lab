#!/usr/bin/env python3
"""Verify qualified Flash source content without building, networking, or running models.

Only tracked paths in the reviewed source inventory are opened. Git blob sizes
bound file reads; symlink targets are hashed as text and never followed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess


class VerificationError(ValueError):
    pass


def resolve_inventory(explicit=None, script=None):
    if explicit is not None:
        path = Path(explicit)
        if not path.is_file():
            raise VerificationError("Inventory file is missing")
        return path
    anchor = Path(script or __file__).resolve().parent.parent
    candidates = [anchor / "dependencies/flash-source-inventory.json"]
    for name in ("snapdragon-local-ai-lab", "snapdragon-ai-lab"):
        candidates.append(anchor / "share" / name / "dependencies/flash-source-inventory.json")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise VerificationError("Inventory not found; pass --inventory explicitly")


def git_env():
    env = os.environ.copy()
    for key in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY",
                "GIT_ALTERNATE_OBJECT_DIRECTORIES"):
        env.pop(key, None)
    env.update(GIT_NO_LAZY_FETCH="1", GIT_TERMINAL_PROMPT="0", GIT_OPTIONAL_LOCKS="0",
               GIT_NO_REPLACE_OBJECTS="1", GIT_CONFIG_COUNT="1",
               GIT_CONFIG_KEY_0="protocol.allow", GIT_CONFIG_VALUE_0="never")
    return env


def git(repo, *args, data=None):
    result = subprocess.run(["git", "-C", str(repo), *args], input=data,
                            capture_output=True, env=git_env())
    if result.returncode:
        raise VerificationError("Git source/index check failed: " + args[0])
    return result.stdout


def canonical_digest(rows):
    lines = "".join(f"{r['mode']} {r['git_object']} {r['path']}\n" for r in rows)
    return hashlib.sha256(lines.encode()).hexdigest()


def validate_rows(rows):
    previous = None
    for row in rows:
        name = row["path"]
        path = PurePosixPath(name)
        if (path.is_absolute() or not path.parts or ".." in path.parts or ".git" in path.parts
                or path.as_posix() != name or "\x00" in name or "\n" in name):
            raise VerificationError("Unsafe inventory path")
        if previous is not None and name <= previous:
            raise VerificationError("Inventory paths are duplicated or not sorted")
        if row["type"] != "blob" or row["mode"] not in ("100644", "100755", "120000"):
            raise VerificationError("Unsupported source object type/mode")
        oid = row["git_object"]
        if len(oid) != 40 or any(char not in "0123456789abcdef" for char in oid):
            raise VerificationError("Invalid source blob identity")
        previous = name


def indexed_rows(repo, prefix):
    # Byte checks below also catch differences hidden by index stat caching or
    # checkout filters; keep the ordinary Git check as an independent signal.
    git(repo, "diff", "--quiet", "--no-ext-diff", "--no-textconv", "--ignore-submodules=all")
    rows = []
    for record in git(repo, "ls-files", "--stage", "-z").split(b"\0"):
        if not record:
            continue
        header, path = record.split(b"\t", 1)
        mode, oid, stage = header.decode().split()
        if stage != "0":
            raise VerificationError("Unmerged source index")
        if mode != "160000":
            rows.append({"path": prefix + path.decode(), "mode": mode,
                         "type": "blob", "git_object": oid})
    if git(repo, "ls-files", "--others", "--exclude-standard", "-z"):
        raise VerificationError("Untracked source files; use a fresh out-of-tree build")
    return rows


def object_sizes(repo, rows):
    oids = sorted({row["git_object"] for row in rows})
    raw = git(repo, "cat-file", "--batch-check=%(objectname) %(objecttype) %(objectsize)",
              data=("\n".join(oids) + "\n").encode())
    sizes = {}
    for line in raw.decode().splitlines():
        fields = line.split()
        if len(fields) != 3 or fields[1] != "blob":
            raise VerificationError("Required source blob is unavailable locally")
        sizes[fields[0]] = int(fields[2])
    if set(sizes) != set(oids):
        raise VerificationError("Incomplete source object size inventory")
    return sizes


def source_blob(root, relative, mode, expected_size, allow_crlf=False):
    parts = PurePosixPath(relative).parts
    parent = root
    for part in parts[:-1]:
        parent = parent / part
        if not stat.S_ISDIR(parent.lstat().st_mode):
            raise VerificationError("Source directory is missing or symlinked: " + relative)
    path = parent / parts[-1]
    info = path.lstat()
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {expected_size}\0".encode())
    if mode == "120000":
        if not stat.S_ISLNK(info.st_mode):
            raise VerificationError("Expected a source symlink: " + relative)
        data = os.fsencode(os.readlink(path))
        if len(data) != expected_size:
            raise VerificationError("Source symlink size differs: " + relative)
        digest.update(data)
    else:
        size_ok = (expected_size <= info.st_size <= expected_size * 2) if allow_crlf else info.st_size == expected_size
        if (not stat.S_ISREG(info.st_mode) or not size_ok
                or bool(info.st_mode & 0o111) != (mode == "100755")):
            raise VerificationError("Source file type, size or mode differs: " + relative)
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd, "rb") as stream:
            opened = os.fstat(stream.fileno())
            if ((opened.st_dev, opened.st_ino, opened.st_size) !=
                    (info.st_dev, info.st_ino, info.st_size)):
                raise VerificationError("Source file changed while opening: " + relative)
            remaining = info.st_size
            script_parts = []
            while remaining:
                block = stream.read(min(1024 * 1024, remaining))
                if not block:
                    raise VerificationError("Source file was truncated: " + relative)
                if allow_crlf:
                    script_parts.append(block)
                else:
                    digest.update(block)
                remaining -= len(block)
            after = os.fstat(stream.fileno())
            if (opened.st_size, opened.st_mtime_ns, opened.st_ctime_ns) != (
                    after.st_size, after.st_mtime_ns, after.st_ctime_ns):
                raise VerificationError("Source file changed during verification: " + relative)
            if allow_crlf:
                canonical = b"".join(script_parts).replace(b"\r\n", b"\n")
                if len(canonical) != expected_size:
                    raise VerificationError("Declared script line endings do not match: " + relative)
                digest.update(canonical)
    return digest.hexdigest()


def verify_source(parent, inventory_path):
    root = Path(parent).resolve(strict=True)
    inventory = json.loads(Path(inventory_path).read_text())
    expected = inventory["files"]
    validate_rows(expected)
    if canonical_digest(expected) != inventory["flattened_content_sha256"]:
        raise VerificationError("Inventory content digest differs")
    crlf_paths = set(inventory.get("checkout_crlf_paths", []))
    names = {row["path"] for row in expected}
    if not crlf_paths <= names or any(
            Path(name).suffix not in (".ps1", ".bat") or name.startswith("third_party/llama.cpp/")
            for name in crlf_paths):
        raise VerificationError("Invalid declared Windows script normalization")
    components = [(root, ""), (root / "third_party/llama.cpp", "third_party/llama.cpp/")]
    actual, sizes = [], {}
    for repo, prefix in components:
        rows = indexed_rows(repo, prefix)
        actual.extend(rows)
        sizes.update(object_sizes(repo, rows))
    actual.sort(key=lambda row: row["path"])
    if actual != expected:
        raise VerificationError("Tracked source/index inventory differs from qualified engine")
    for row in expected:
        if source_blob(root, row["path"], row["mode"], sizes[row["git_object"]],
                       allow_crlf=row["path"] in crlf_paths) != row["git_object"]:
            raise VerificationError("Tracked source bytes differ: " + row["path"])
    return {"qualified_content_match": True, "tracked_blobs": len(actual),
            "flattened_content_sha256": canonical_digest(actual),
            "working_tree_blob_bytes_and_modes_verified": True,
            "declared_windows_script_crlf_policy_paths": sorted(crlf_paths),
            "gitlink_commit_identity_excluded": True, "native_execution": False,
            "network_access": False, "model_payload_read": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent", type=Path)
    parser.add_argument("--inventory", type=Path)
    args = parser.parse_args()
    try:
        result = verify_source(args.parent, resolve_inventory(args.inventory))
    except (VerificationError, OSError, KeyError, ValueError) as error:
        parser.exit(1, "Flash source verification failed: " + str(error) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
