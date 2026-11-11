"""Private, bounded storage for prefix-only llama.cpp state. No request text."""
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import time

NAME = re.compile(r'^[0-9a-f]{64}\.(?:bin|json|pending)$')

class PrefixStore:
    def __init__(self, root, max_bytes=3 * 1024**3):
        self.root = Path(root)
        self.max_bytes = int(max_bytes)
        if self.max_bytes < 1: raise ValueError('Invalid cache budget')
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        info = self.root.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid():
            raise ValueError('Unsafe cache directory')
        os.chmod(self.root, 0o700)
        # A single controller owns this directory. Pending writes are never
        # restorable and are discarded after an interrupted backend lifetime.
        for path in self.root.iterdir():
            if NAME.fullmatch(path.name) and path.suffix == '.pending':
                path.unlink(missing_ok=True)
        self.prune()

    def path(self, key, suffix):
        name = key + '.' + suffix
        if not NAME.fullmatch(name): raise ValueError('Invalid cache name')
        return self.root / name

    def _info(self, path):
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise ValueError('Unsafe cache entry')
        return info

    def _digest(self, path):
        self._info(path)
        digest = hashlib.sha256()
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd, 'rb') as stream:
            for chunk in iter(lambda: stream.read(1024**2), b''): digest.update(chunk)
        return digest.hexdigest()

    def _sync(self):
        fd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
        try: os.fsync(fd)
        finally: os.close(fd)

    def discard(self, key):
        for suffix in ('bin', 'json', 'pending'):
            self.path(key, suffix).unlink(missing_ok=True)

    def pending(self, plan):
        path = self.path(plan.key, 'pending')
        path.unlink(missing_ok=True)
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
        os.close(fd)
        return path.name

    def commit(self, plan):
        source = self.path(plan.key, 'pending')
        size = self._info(source).st_size
        if not 0 < size <= self.max_bytes:
            source.unlink(missing_ok=True)
            raise ValueError('Cache entry exceeds budget')
        digest = self._digest(source)
        fd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
        try: os.fsync(fd)
        finally: os.close(fd)
        source.replace(self.path(plan.key, 'bin'))
        metadata = {**plan.metadata(), 'state_sha256': digest, 'state_bytes': size,
                    'last_used': time.time()}
        temporary = self.path(plan.key, 'pending')
        fd = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, 'w') as stream:
            json.dump(metadata, stream); stream.flush(); os.fsync(stream.fileno())
        temporary.replace(self.path(plan.key, 'json'))
        self._sync()
        self.prune(keep=plan.key)
        return metadata

    def lookup(self, plan):
        try:
            metadata_path = self.path(plan.key, 'json')
            if self._info(metadata_path).st_size > 8192: raise ValueError('Invalid metadata')
            fd = os.open(metadata_path, os.O_RDONLY | os.O_NOFOLLOW)
            with os.fdopen(fd) as stream: metadata = json.load(stream)
            if any(metadata.get(k) != v for k, v in plan.metadata().items()): raise ValueError('Identity mismatch')
            state = self.path(plan.key, 'bin')
            size = self._info(state).st_size
            if not 0 < size <= self.max_bytes or size != metadata['state_bytes']: raise ValueError('Size mismatch')
            if self._digest(state) != metadata['state_sha256']: raise ValueError('Checksum mismatch')
            os.utime(metadata_path, None, follow_symlinks=False)
            return state.name
        except (OSError, ValueError, KeyError, TypeError):
            self.discard(plan.key)
            return None

    def prune(self, keep=None):
        entries = []
        for path in self.root.iterdir():
            if not NAME.fullmatch(path.name) or path.suffix != '.bin': continue
            try:
                size = self._info(path).st_size
                meta = self.path(path.stem, 'json')
                age = meta.lstat().st_mtime if meta.exists() else 0
                entries.append((age, path.stem, size))
            except (OSError, ValueError): continue
        total = sum(row[2] for row in entries)
        for _, key, size in sorted(entries):
            if total <= self.max_bytes: break
            if key == keep: continue
            self.discard(key); total -= size

    def clear(self):
        for path in self.root.iterdir():
            if NAME.fullmatch(path.name): path.unlink(missing_ok=True)
        self._sync()

    def stats(self):
        sizes = []
        for path in self.root.iterdir():
            if NAME.fullmatch(path.name) and path.suffix == '.bin':
                try: sizes.append(self._info(path).st_size)
                except (OSError, ValueError): pass
        return {'entries': len(sizes), 'bytes': sum(sizes), 'max_bytes': self.max_bytes}
