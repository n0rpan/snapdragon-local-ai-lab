"""S11 (v0.1.1): relative Markdown links must resolve in the public snapshot. A link from a shipped
Markdown file to a missing file, or to a path tools/public-exclude.txt leaves out, is dead for every
reader of the public repo. Model-free, no network (external links are not checked)."""
import fnmatch
from pathlib import Path
import re
import subprocess
import unittest
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r'(?<!!)\[[^\]]*\]\(([^)\s]+)(?:\s+"[^"]*")?\)')


def tracked():
    try:
        out = subprocess.run(['git', '-C', str(ROOT), 'ls-files', '-z'], capture_output=True, check=True).stdout
        return {p for p in out.decode().split('\0') if p}
    except (OSError, subprocess.CalledProcessError):
        return {str(p.relative_to(ROOT)) for p in ROOT.rglob('*') if p.is_file() and '.git' not in p.parts}


def excluded_patterns():
    path = ROOT / 'tools/public-exclude.txt'
    if not path.exists():  # the public snapshot ships without its own exclude list
        return []
    return [line.strip() for line in path.read_text().splitlines() if line.strip() and not line.startswith('#')]


def is_excluded(rel, patterns):
    return any(fnmatch.fnmatch(rel, p) or (p.endswith('/') and rel.startswith(p)) for p in patterns)


def strip_code(text):
    text = re.sub(r'```.*?```', '', text, flags=re.S)
    return re.sub(r'`[^`\n]*`', '', text)


class DocLinks(unittest.TestCase):
    def test_relative_links_resolve_and_are_public(self):
        files = tracked()
        patterns = excluded_patterns()
        dirs = {str(Path(f).parent) for f in files} | {str(p) for f in files for p in Path(f).parents}
        dead = []
        for rel in sorted(f for f in files if f.endswith('.md') and not is_excluded(f, patterns)):
            for target in LINK.findall(strip_code((ROOT / rel).read_text(errors='replace'))):
                if re.match(r'^[a-z][a-z0-9+.-]*:', target, re.I) or target.startswith('#'):
                    continue
                path = unquote(target.split('#', 1)[0])
                if not path:
                    continue
                resolved = (Path(rel).parent / path)
                norm = str(Path(*[p for p in resolved.parts if p != '.']))
                parts = []
                for p in Path(norm).parts:
                    if p == '..':
                        if parts: parts.pop()
                    else:
                        parts.append(p)
                norm = '/'.join(parts).rstrip('/') or '.'
                if (norm not in files and norm not in dirs) or is_excluded(norm, patterns):
                    dead.append(f'{rel} -> {target}')
        self.assertEqual(dead, [], 'dead or non-public links:\n' + '\n'.join(dead))


if __name__ == '__main__':
    unittest.main()
