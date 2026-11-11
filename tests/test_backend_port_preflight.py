"""No models: fail occupied ports before RAM allocation or child launch."""
import errno
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import Mock, patch

from snapdragon_ai_lab.runtime import controller


class BackendPortPreflight(unittest.TestCase):
    def test_free_port_is_not_reserved_after_check(self):
        with socket.socket() as listener:
            listener.bind(('127.0.0.1', 0))
            port = listener.getsockname()[1]
        controller.require_backend_port(port, 'LOCAL_LLM_BACKEND_PORT')
        with socket.socket() as listener:
            listener.bind(('127.0.0.1', port))
            listener.listen()

    def test_listener_reports_port_process_and_configurable_setting(self):
        with socket.socket() as listener:
            listener.bind(('127.0.0.1', 0)); listener.listen()
            port = listener.getsockname()[1]
            with patch.object(controller, 'listening_process', return_value='example-app'):
                with self.assertRaisesRegex(controller.Unavailable, f'port {port} is in use by example-app.*LOCAL_LLM_BACKEND_PORT'):
                    controller.require_backend_port(port, 'LOCAL_LLM_BACKEND_PORT')
            # The preflight never connected to or closed the existing listener.
            self.assertEqual(listener.getsockname()[1], port)

    def test_owner_lookup_uses_listening_inode_and_sanitized_comm_only(self):
        with tempfile.TemporaryDirectory() as directory:
            proc = Path(directory); (proc / 'net').mkdir()
            (proc / 'net/tcp').write_text('header\n 0: 0100007F:1F9D 00000000:0000 0A 0:0 0:0 0 1000 0 12345\n')
            child = proc / '123'; (child / 'fd').mkdir(parents=True)
            (child / 'fd/9').symlink_to('socket:[12345]')
            (child / 'comm').write_text('example\x1bapp\n')
            self.assertEqual(controller.listening_process(8093, proc), 'example_app')
            self.assertEqual(controller.listening_process(8094, proc), 'another process')

    def test_missing_owner_metadata_remains_a_clear_port_error(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(controller.listening_process(8093, Path(directory)), 'another process')

    def test_owner_diagnostic_does_not_blame_a_listener_on_another_interface(self):
        with tempfile.TemporaryDirectory() as directory:
            proc = Path(directory); (proc / 'net').mkdir()
            (proc / 'net/tcp').write_text('header\n 0: 0101A8C0:1F9D 00000000:0000 0A 0:0 0:0 0 1000 0 12345\n')
            child = proc / '123'; (child / 'fd').mkdir(parents=True)
            (child / 'fd/9').symlink_to('socket:[12345]')
            (child / 'comm').write_text('unrelated-listener\n')
            self.assertEqual(controller.listening_process(8093, proc), 'another process')

    def test_other_bind_error_is_not_misreported_as_port_in_use(self):
        probe = Mock()
        probe.__enter__ = Mock(return_value=probe); probe.__exit__ = Mock(return_value=False)
        probe.bind.side_effect = OSError(errno.EACCES, 'permission denied')
        with patch.object(controller.socket, 'socket', return_value=probe), patch.object(controller, 'listening_process') as owner:
            with self.assertRaisesRegex(controller.Unavailable, 'unavailable.*error 13'):
                controller.require_backend_port(8093, 'LOCAL_LLM_BACKEND_PORT')
            owner.assert_not_called()

    def manager(self, config):
        manager = controller.Manager(); manager.stop_event.set()
        manager.config = Mock(return_value=config); manager.eligible = Mock(return_value=True)
        return manager

    def test_busy_port_stops_all_backends_before_model_ram_lease_or_launch(self):
        for kind in (None, controller.oxe_runtime.KIND, controller.bigmoe_runtime.KIND):
            config = {'model_alias': 'synthetic-local', 'backend_kind': kind, 'ready': True,
                      'model_path': '/not-a-model.gguf'}
            manager = self.manager(config)
            with self.subTest(kind=kind), patch.object(controller, 'require_backend_port', side_effect=controller.Unavailable('occupied')) as probe, \
                    patch.object(controller, 'available_memory_bytes') as ram, \
                    patch.object(controller.subprocess, 'Popen') as launch, \
                    patch.object(controller.bigmoe_runtime, 'Runtime') as runtime, \
                    patch.object(controller.oxe_runtime, 'new_api_key') as key:
                with self.assertRaisesRegex(controller.Unavailable, 'occupied'):
                    manager.begin(allow_start=True, model_alias='synthetic-local')
                streamed = kind == controller.bigmoe_runtime.KIND
                probe.assert_called_once_with(controller.FLASH_BACKEND_PORT if streamed else controller.BACKEND_PORT,
                                              'LOCAL_FLASH_BACKEND_PORT' if streamed else 'LOCAL_LLM_BACKEND_PORT')
                for action in (ram, launch, runtime, key): action.assert_not_called()
                self.assertIsNone(manager.lease); self.assertIsNone(manager.process)
                self.assertEqual(manager.state, 'unloaded')

    def test_loaded_engine_is_reused_without_probing_its_own_port(self):
        config = {'model_alias': 'synthetic-local'}
        manager = self.manager(config); manager.loaded_config = config
        process = Mock(); process.poll.return_value = None; manager.process = process
        with patch.object(controller, 'require_backend_port') as probe:
            self.assertIs(manager.begin(allow_start=True, model_alias='synthetic-local'), process)
            probe.assert_not_called()
        manager.process = None

    def test_sleeping_status_request_does_not_probe_or_start(self):
        manager = self.manager({'model_alias': 'synthetic-local', 'ready': True})
        with patch.object(controller, 'require_backend_port') as probe:
            with self.assertRaisesRegex(controller.Unavailable, 'sleeping'):
                manager.begin(allow_start=False, model_alias='synthetic-local')
            probe.assert_not_called()


if __name__ == '__main__':
    unittest.main()
