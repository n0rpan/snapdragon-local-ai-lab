"""S5 (v0.1.1, owner decision O-30): the QoS helper and polkit rule carry no owner login or uid;
the installer renders the installing user, and the docs say so. Static, model-free."""
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
QOS = ROOT / 'packaging/flash-qos'


class NoOwnerIdentityInPackaging(unittest.TestCase):
    def test_sources_grant_nobody(self):
        helper = (QOS / 'local-ai-flash-qos').read_text()
        rule = (QOS / '49-local-ai-flash-qos.rules').read_text()
        self.assertEqual(re.findall(r'^ALLOWED_UID = .*$', helper, re.M), ['ALLOWED_UID = None'])
        self.assertEqual(re.findall(r'^var ALLOWED_USER = .*$', rule, re.M), ['var ALLOWED_USER = null;'])
        for path in QOS.iterdir():
            text = path.read_text()
            # no account name or fixed uid anywhere (the rendered values come from install-root.sh)
            self.assertIsNone(re.search(r'ALLOWED_USER\s*=\s*["\']', text), path.name)
            self.assertIsNone(re.search(r'\b(uid|UID)\s*=\s*1000\b', text), path.name)

    def test_docs_describe_the_installer_rendering(self):
        for doc in (QOS / 'README.md', ROOT / 'docs/FLASH-RUST-PRODUCTION.md'):
            text = doc.read_text()
            with self.subTest(doc=doc.name):
                self.assertIn('SUDO_USER', text)
                self.assertIn('--user', text)
                self.assertNotIn("rewrites\n  `ALLOWED_UID` to the owner's uid", text)
                self.assertNotIn("owner's uid", text)


if __name__ == '__main__':
    unittest.main()
