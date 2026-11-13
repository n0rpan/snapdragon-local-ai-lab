"""`ctl`: how the controller's split modules (ctl_*.py) read patchable names.

The controller was one file until v0.1.4, and its tests replace single names on that module:
`patch.object(controller, 'CONFIG', ...)`, `controller.available_memory_bytes`, and so on. Code
that moved out of controller.py keeps seeing those replacements by reading such names as
`ctl.NAME`, which looks NAME up in the controller module at call time, as a global lookup in
the single file did.

Read through `ctl`: configuration (paths, ports, time limits), anything that asks the host
(memory, sockets, processes, files, notifications) and the collaborator classes the Manager
and Handler construct. Import directly: exceptions, pure helpers and fixed tables.

Rules: never read `ctl.NAME` at import time (only inside functions), and never import
`controller` from a ctl module; controller.py binds itself here when it is imported.
"""
import importlib


class _Controller:
    def __init__(self):
        self.__dict__['_module'] = None

    def bind(self, module):
        self.__dict__['_module'] = module

    def __getattr__(self, name):
        module = self.__dict__['_module']
        if module is None:
            # A ctl module used on its own (no controller imported yet): import it now. Its
            # import calls bind(); as the unit's script it is already bound as __main__.
            imported = importlib.import_module('.controller', __package__) if __package__ \
                else importlib.import_module('controller')
            module = self.__dict__['_module'] or imported
        return getattr(module, name)


ctl = _Controller()
