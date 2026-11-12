"""S11 (v0.1.1): relative Markdown links must resolve in the public snapshot. A link from a shipped
Markdown file to a missing file, or to a path tools/public-exclude.txt leaves out, is dead for every
reader of the public repo. Model-free, no network (external links are not checked)."""
import fnmatch
import os
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


def github_slugs(text):
    """Heading anchors as GitHub renders them: lowercase, punctuation dropped (except - and _),
    spaces to '-', duplicates suffixed -1, -2, ...; explicit <a name/id> anchors too."""
    slugs, seen = set(), {}
    for line in strip_code(text).splitlines():
        match = re.match(r'^ {0,3}#{1,6}\s+(.*?)\s*#*\s*$', line)
        if not match:
            continue
        title = re.sub(r'\[([^\]]*)\]\([^)]*\)', r'\1', match.group(1))  # link text only
        title = re.sub(r'<[^>]+>', '', title)
        slug = re.sub(r'[^\w\- ]', '', title.lower().replace('`', '')).replace(' ', '-')
        count = seen.get(slug, 0)
        seen[slug] = count + 1
        slugs.add(slug if count == 0 else f'{slug}-{count}')
    slugs.update(re.findall(r'<a\s+(?:name|id)="([^"]+)"', text))
    return slugs


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

    def test_markdown_anchors_resolve(self):
        """S15 (audit): `#fragment` links (in-page or to another .md) must name a real heading."""
        files = tracked()
        patterns = excluded_patterns()
        slugs = {}
        dead = []
        for rel in sorted(f for f in files if f.endswith('.md') and not is_excluded(f, patterns)):
            for target in LINK.findall(strip_code((ROOT / rel).read_text(errors='replace'))):
                if re.match(r'^[a-z][a-z0-9+.-]*:', target, re.I) or '#' not in target:
                    continue
                path, fragment = target.split('#', 1)
                fragment = unquote(fragment)
                if not fragment or fragment.startswith(('L', 'l')) and re.fullmatch(r'[Ll]\d+(-[Ll]?\d+)?', fragment):
                    continue  # line anchors on code files
                target_rel = rel if not path else str(Path(os.path.normpath(Path(rel).parent / unquote(path))))
                if not target_rel.endswith('.md') or target_rel not in files:
                    continue  # missing files are the other test's job; only Markdown has heading anchors
                if target_rel not in slugs:
                    slugs[target_rel] = github_slugs((ROOT / target_rel).read_text(errors='replace'))
                if fragment.lower() not in slugs[target_rel]:
                    dead.append(f'{rel} -> {target}')
        self.assertEqual(dead, [], 'links to headings that do not exist:\n' + '\n'.join(dead))


if __name__ == '__main__':
    unittest.main()
