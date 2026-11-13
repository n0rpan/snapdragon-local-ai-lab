"""Proxy stage 3, ledger and finish: the ledger entry of an inference request and its labels,
admission through the Manager (with its error replies), and the bookkeeping after every
forwarded request: watcher stop, engine death record, thermal sample, the ledger finish, the
`request_end` engine-log line and, last, the release of the slot (`manager.end`).
"""
import html
import json
if __package__:
    from .ctl_env import JOURNAL_HINT, ModelBusy, Unavailable, request_origin, unexpected_cause
    from .ctl_proxy_prep import answered
    from .ctl_seams import ctl
    from . import lan_share, profile_policy, request_ledger
else:
    from ctl_env import JOURNAL_HINT, ModelBusy, Unavailable, request_origin, unexpected_cause
    from ctl_proxy_prep import answered
    from ctl_seams import ctl
    import lan_share
    import profile_policy
    import request_ledger


def open_entry(self, st):
    """The ledger row of an inference request (st.led), labelled before admission."""
    st.request_process = None
    st.request_engine_config = None
    # One ledger line per inference request: numbers and labels only (request_ledger.py).
    st.led = None
    if st.inference:
        st.led = request_ledger.Entry(st.path, st.payload, self.headers.get('User-Agent'), st.target.get('model_alias'),
                                      headers=self.headers, origin=lan_share.REMOTE_ORIGIN if st.remote else None,
                                      client=self.share_client)
        st.led.label(st.harness, st.harness_key)
        st.led.label_turn(st.turn)
        if st.tool_cap is not None and any(st.tool_cap.values()):
            st.led.row['tool_cap'] = dict(st.tool_cap)  # counts only
        try:  # v0.1.3 (lane C x D): the context mode selected for this model, on its ledger row
            if st.target.get('model_alias') in profile_policy.MODE_MODELS:
                st.led.row['mode'] = profile_policy.stored_mode(self.manager.runtime_value(), st.target.get('model_alias'))
        except Exception:  # advisory telemetry must never fail inference
            pass


def admit(self, st):
    """Admission (st.request_process); ANSWERED with the 409/503 reply when it failed."""
    path, remote, inference, helper, led = st.path, st.remote, st.inference, st.helper, st.led
    # O-102 M4: a local request that arrives while this shared one runs cancels it
    st.preempt = lan_share.Preempt() if remote and inference else None
    try:
        # a shared request is admitted only when the slot is free now: it never waits in the
        # FIFO ahead of a local request, and only to the loaded model (ModelBusy, in begin)
        st.request_process = self.manager.begin_queued(inference=inference,
            allow_start=(inference or helper or (self.command in ('GET', 'HEAD') and path == '/')) and not remote,
            model_alias=st.model_alias,
            origin=lan_share.REMOTE_ORIGIN if remote else request_origin(self.headers),
            **({'wait_seconds': lan_share.SLOT_WAIT, 'remote': True, 'preempt': st.preempt} if remote else {}))
    except ModelBusy as error:
        if led is not None: led.finish(409, 'model_busy')
        return answered(self.json_reply(409, {'error': {'message': str(error), 'type': 'conflict_error', 'code': 'model_busy'}}))
    except Unavailable as error:
        if led is not None: led.finish(503, 'busy' if 'running' in str(error) else 'unavailable')
        if path == '/':
            # Deliberately simple setup status; the chat UI comes from llama.cpp.
            page = b'''<!doctype html><meta charset="utf-8"><title>Local Qwen</title>
<style>body{font:17px system-ui;max-width:650px;margin:12vh auto;padding:20px;background:#171717;color:#eee}a{color:#9dcaff}</style>
<h1>Local Qwen</h1><p>{MESSAGE}</p>
<p><a href="/">Try opening chat again</a> &middot; <a href="/_local/status">Setup status</a> &middot; <a href="http://127.0.0.1:8090">Image studio</a></p>'''
            page = page.replace(b'{MESSAGE}', html.escape(str(error)).encode())
            return answered(self.reply(503, page, 'text/html; charset=utf-8'))
        return answered(self.json_reply(503, {'error': {'message': lan_share.public_message(str(error)) if remote else str(error),
                                                        'type': 'local_backend_busy'}}))
    except Exception as error:
        if led is not None: led.finish(503, 'start_failed')
        if remote:  # no exception text (paths) for another device
            return answered(self.json_reply(503, {'error': {'message': 'The host could not start its local model.',
                                                            'type': 'local_backend_error'}}))
        return answered(self.json_reply(503, {'error': {'message': 'Local backend could not be started '
                                                        f'({unexpected_cause(error)}). {JOURNAL_HINT}',
                                                        'type': 'local_backend_error'}}))
    # Admitted: from here the slot is released by finish() (or backend_request's error path), so
    # the ledger's admission timestamps are advisory and must never raise past this point.
    if led is not None:
        try:
            led.admitted_now(*(getattr(self.manager, 'admission_times', None) or (None, None)))
        except Exception:
            pass
    return None


def finish(self, st):
    """After the forwarding stage, on every path (its `finally`): bookkeeping, then end()."""
    try:
        # Stop and join before end() admits another FIFO request. The watcher
        # owns only this request's socket; no later request can be cancelled.
        if st.watcher is not None:
            st.watcher.stop()
        if st.client_gone():
            st.request_error = 'client_disconnected'
            self.close_connection = True
        if st.request_process is not None and st.request_process.poll() is not None:
            self.manager.record_engine_death(st.request_process, st.request_engine_config)
        st.connection.close()
        if st.led is not None:
            try: st.led.row.update(ctl.thermal_sample())  # ST-4: numbers only
            except Exception: pass
            st.led.finish(st.backend_status if st.backend_status is not None else 503, st.request_error)
        if st.request_error and self.manager.engine_log_writer is not None:
            self.manager.engine_log_writer.write(json.dumps({
                'event': 'request_end', 'error': st.request_error,
                'status': st.backend_status,
                'model_alias': (st.request_engine_config or {}).get('model_alias'),
            }, sort_keys=True) + '\n')
    finally:
        # F80: release the slot even if the bookkeeping above raises
        self.manager.end(inference=st.inference, touch=st.helper)
