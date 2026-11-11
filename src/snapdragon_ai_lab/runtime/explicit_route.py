"""Opt-in, explicit per-session routing; no heuristics, I/O or model loading."""

ENABLE_ENV = 'LOCAL_AI_ENABLE_ROUTING'
ALIAS = 'local-auto'
TARGETS = {'code': 'ornith-35b-a3b-local', 'think-hard': 'qwen3.8-flash-next-local'}


def rewrite(payload, *, enabled=False):
    """Return a payload with a concrete model, consuming our local extension.

    Real model/panel choices always win. Only local-auto invokes routing, and
    requires an explicit local_route on every request (session clients keep
    that choice). No prompt text is examined. The caller keeps normal model
    eligibility, queue, admission, lease, idle-unload and switch behaviour.
    """
    if not isinstance(payload, dict):
        return payload
    automatic = payload.get('model') == ALIAS
    if not automatic and 'local_route' not in payload:
        return payload
    updated = dict(payload)
    route = updated.pop('local_route', None)
    if not automatic:
        return updated
    if not enabled:
        raise ValueError('local-auto is disabled; choose a model explicitly')
    if not isinstance(route, str) or route not in TARGETS:
        raise ValueError('local-auto requires local_route=code or think-hard')
    updated['model'] = TARGETS[route]
    return updated
