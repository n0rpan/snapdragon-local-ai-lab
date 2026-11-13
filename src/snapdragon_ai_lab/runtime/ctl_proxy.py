"""The proxy to the loaded engine: request defaults per engine and `ProxyRoutes.proxy_request`,
which runs the three proxy stages in order:
  ctl_proxy_prep    request prep: checks, body, payload shaping, Flash/cache answers, backend headers
  ctl_proxy_ledger  ledger entry, admission through the Manager, the bookkeeping after the reply
  ctl_proxy_stream  forwarding: send, stream the answer, client disconnects, preemption of a
                    shared (LAN) request
"""
if __package__:
    from .ctl_proxy_prep import ANSWERED, FLASH_CHAT_HTML, HOP_HEADERS, ProxyRequest, _BadFlashChat
    from . import classic_beta, ctl_proxy_ledger, ctl_proxy_prep, ctl_proxy_stream, oxe_runtime
else:
    from ctl_proxy_prep import ANSWERED, FLASH_CHAT_HTML, HOP_HEADERS, ProxyRequest, _BadFlashChat
    import classic_beta
    import ctl_proxy_ledger
    import ctl_proxy_prep
    import ctl_proxy_stream
    import oxe_runtime


def request_defaults(target, payload, path):
    """Backend request defaults for the resolved target (qwen-local is already resolved).

    The classic beta's 2048 output default belongs to its llama.cpp engine only; an
    oxe-backed Ornith gets the oxe defaults (per-profile default_max_tokens) instead.
    Returns payload itself when no backend defaults apply.
    """
    if target.get('backend_kind') == oxe_runtime.KIND:
        return oxe_runtime.request_defaults(payload, path, target)
    if classic_beta.is_beta(target):
        return classic_beta.request_defaults(payload, path, alias=target['model_alias'])
    return payload


class ProxyRoutes:
    """Handler: forwarding to the engine."""

    def proxy_request(self, path, api_value, api_key, remote, inference, helper):
        """Shape the request for the target engine, admit it, forward it and stream the answer."""
        request = ProxyRequest(path, api_value, api_key, remote, inference, helper)
        if ctl_proxy_prep.prepare(self, request) is ANSWERED:
            return
        ctl_proxy_ledger.open_entry(self, request)
        if ctl_proxy_ledger.admit(self, request) is ANSWERED:
            return
        if ctl_proxy_prep.backend_request(self, request) is ANSWERED:
            return
        try:
            ctl_proxy_stream.forward(self, request)
        finally:
            ctl_proxy_ledger.finish(self, request)
