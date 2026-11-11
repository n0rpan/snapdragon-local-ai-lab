"""Cancel an upstream request when its fully-read HTTP client goes away."""
import select
import socket
import threading


class ClientDisconnect:
    """Watch only; never consume pipelined bytes or change the client timeout.

    Start after the request body and upstream request have been sent. Shutdown
    wakes a proxy blocked in getresponse/read, including before the first token.
    The owner joins this watcher before releasing its inference/FIFO slot.
    """

    def __init__(self, client, upstream, interval=0.05):
        self.client = client
        self.upstream = upstream
        self.interval = interval
        self.disconnected = threading.Event()
        self.finished = threading.Event()
        self.thread = threading.Thread(target=self._watch, name='local-ai-client-eof', daemon=True)

    def start(self):
        self.thread.start()
        return self

    def cancel(self):
        self.disconnected.set()
        try:
            self.upstream.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass

    def _watch(self):
        while not self.finished.is_set():
            try:
                readable, _, _ = select.select([self.client], [], [], self.interval)
                if not readable:
                    continue
                data = self.client.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT)
                if not data:
                    if not self.finished.is_set():
                        self.cancel()
                    return
            except BlockingIOError:
                continue
            except (OSError, ValueError):
                if not self.finished.is_set():
                    self.cancel()
                return
            # An HTTP keep-alive/pipelined byte belongs to the request parser.
            self.finished.wait(self.interval)

    def stop(self):
        self.finished.set()
        self.thread.join()
