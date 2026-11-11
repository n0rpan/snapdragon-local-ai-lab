"""SPDX-License-Identifier: MIT. Bounded persistent compiler output, no model use."""
MARKER = b'\n[build output truncated; first and last bytes retained]\n'
MAX_BYTES = 2*1024*1024

class BuildLog:
    def __init__(self, stream, capacity=MAX_BYTES):
        if capacity <= len(MARKER)+2: raise ValueError('log_capacity_too_small')
        self.stream = stream
        self.capacity = capacity
        self.head_cap = capacity//2
        self.tail_cap = capacity-self.head_cap-len(MARKER)
        self.head = bytearray()
        self.tail = bytearray()
        self.total = 0

    @property
    def truncated(self): return self.total > self.head_cap+self.tail_cap

    def append(self, value):
        self.total += len(value)
        take = min(len(value),self.head_cap-len(self.head))
        self.head.extend(value[:take])
        self.tail.extend(value[take:])
        if len(self.tail) > self.tail_cap: del self.tail[:-self.tail_cap]
        # Persist after every received chunk; the file stays bounded even if
        # the whole compile unit is killed before final result collection.
        self.stream.seek(0)
        self.stream.write(self.head)
        if self.truncated: self.stream.write(MARKER)
        self.stream.write(self.tail)
        self.stream.truncate()
        self.stream.flush()

    def drain(self, descriptor):
        import os
        for _ in range(16): # Bound drain work so resource/deadline checks run.
            try: value = os.read(descriptor,65536)
            except BlockingIOError: return
            if not value: return
            self.append(value)
