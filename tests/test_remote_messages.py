"""Audit N9: a 503 reason sent to a LAN-share client carries no path of this host."""
import unittest

from snapdragon_ai_lab.runtime import lan_share, oxe_runtime


class RemoteMessages(unittest.TestCase):
    def test_paths_are_replaced_and_plain_reasons_kept(self):
        cases = {
            'The Rust engine binary is missing: /synthetic/home/bin/oxe-server. Build or install it '
            '(docs/ENGINE-SELECTION.md) or set X=0.':
                'The Rust engine binary is missing: <path>. Build or install it (docs/ENGINE-SELECTION.md) or set X=0.',
            'The inference lease ~/.local/state/local-ai/inference.lock is a symbolic link':
                'The inference lease <path> is a symbolic link',
            'A request is running; retry when it finishes.': 'A request is running; retry when it finishes.',
            'Flash needs at least 30% battery to start and stops at 20%.':
                'Flash needs at least 30% battery to start and stops at 20%.',
        }
        for given, expected in cases.items():
            with self.subTest(given=given):
                self.assertEqual(lan_share.public_message(given), expected)
        self.assertEqual(lan_share.public_message(None), '')

    def test_the_missing_binary_reason_loses_its_path(self):
        config = {'backend_kind': oxe_runtime.KIND, 'ready': True, 'server_binary': '/synthetic/private/oxe-server'}
        reason = oxe_runtime.ineligible_reason(config)
        if reason is None or '/synthetic/private' not in reason:
            self.skipTest('this profile stops earlier; the string cases above cover the scrub')
        self.assertNotIn('/synthetic', lan_share.public_message(reason))


if __name__ == '__main__':
    unittest.main()
