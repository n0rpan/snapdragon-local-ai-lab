"""N6: the lease is never opened through a symlink; a corrupt gzip body from the backend is a 502."""
import http.client
import json
import os
import sys
import threading
import unittest.mock
import unittest
from pathlib import Path

from snapdragon_ai_lab.runtime import controller

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _stub_manager import StubManagerCase  # noqa: E402


class LeaseSymlinkTests(StubManagerCase):
    def plant_symlink(self):
        target = self.root / 'victim'
        target.write_text('untouched')
        os.symlink(target, controller.LEASE)
        return target

    def test_classic_start_refuses_a_symlinked_lease(self):
        target = self.plant_symlink()
        with self.run_stub(), self.assertRaisesRegex(controller.Unavailable, "symbolic link"):
            self.manager.begin(allow_start=True, model_alias='stub-local')
        self.assertIsNone(self.manager.process)
        self.assertEqual(target.read_text(), 'untouched')

    def test_storage_delete_refuses_a_symlinked_lease(self):
        target = self.plant_symlink()
        config = self.root / 'runtime.json'
        config.write_text('{}')
        with unittest.mock.patch.object(controller, 'CONFIG', config), \
                unittest.mock.patch.object(controller.storage_view, 'delete', return_value={'deleted': True}) as delete:
            with self.assertRaisesRegex(controller.Unavailable, "symbolic link"):
                self.manager.delete_storage('entry')
        delete.assert_not_called()
        self.assertEqual(target.read_text(), 'untouched')


class CorruptGzipTests(StubManagerCase):
    def test_corrupt_gzip_backend_body_is_a_502(self):
        self.use('--bad-gzip')
        server = controller.Server(('127.0.0.1', 0), controller.Handler)
        server.manager = self.manager
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        with self.run_stub():
            client = http.client.HTTPConnection('127.0.0.1', server.server_port, timeout=20)
            self.addCleanup(client.close)
            client.request('GET', '/', headers={'Host': f'127.0.0.1:{server.server_port}'})
            response = client.getresponse()
            body = response.read()
        self.assertEqual(response.status, 502, body[:200])
        self.assertIn('corrupt', json.loads(body)['error']['message'])
        for _ in range(100):  # the handler releases the slot just after the reply is written
            if self.manager.active == 0:
                break
            __import__('time').sleep(.02)
        self.assertEqual(self.manager.active, 0, 'the slot must be released after the 502')


if __name__ == '__main__':
    unittest.main()
