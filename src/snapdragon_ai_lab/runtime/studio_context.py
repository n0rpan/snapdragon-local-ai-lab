"""StudioContext: how Image Studio's split modules read the Studio they work for.

`studio` is a script that is loaded fresh from its path, by the service and by every test, and
each load has its own paths (from HOME and the environment), settings tables, job state and
names that tests replace on it (`patch.object(studio, 'run_engine', ...)`). The worker and the
planners are ordinary shared modules, imported once, so they keep none of that themselves:
every function there takes a context as its first argument and reads `ctx.NAME`.

A context is one loaded Studio's namespace. `ctx.NAME` is looked up there at call time, as a
global lookup in the single file was, so a replaced function or constant is seen by its callers
and two loaded Studios never see each other's paths or jobs. `studio` passes its context in
with `bind`, which gives each function its old name and signature there.

Rules for the split modules: import only the standard library; read everything else (paths,
tables, state, exceptions, sibling functions, image_slot / image_npu / memory_policy, the pixel
helpers, urlopen) through `ctx`; keep no module-level state. A constant used as a default
argument is defined beside its function and re-exported by `studio`.
"""
import functools
import inspect


class StudioContext:
    __slots__ = ('_names',)

    def __init__(self, names):
        self._names = names

    def __getattr__(self, name):
        try:
            return self._names[name]
        except KeyError:
            raise AttributeError(f'Studio has no {name!r}') from None

    def bind(self, function):
        """`function(ctx, ...)` as a function of this Studio: same name, doc and signature without ctx."""
        @functools.wraps(function)
        def bound(*args, **kwargs):
            return function(self, *args, **kwargs)
        signature = inspect.signature(function)
        bound.__signature__ = signature.replace(parameters=list(signature.parameters.values())[1:])
        return bound

    def bind_class(self, cls):
        """A subclass of `cls` for this Studio (its methods read `self.ctx`), under the same name."""
        return type(cls.__name__, (cls,), {'ctx': self, '__doc__': cls.__doc__, '__module__': self._names['__name__']})
