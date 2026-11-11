"""MIT. Header-only GGUF reader derived from the offline NPU-budget tool.

Keeps scalar and short-array metadata and tensor descriptors; never reads payloads.
"""
import os
import struct

MIB = 1024 * 1024
MAX_SESSIONS = 3
GGUF_MAGIC = b'GGUF'

# ggml_type -> (name, block_elems, block_bytes); from ggml-common.h.
GGML_TYPES = {
    0: ('F32', 1, 4), 1: ('F16', 1, 2), 2: ('Q4_0', 32, 18), 3: ('Q4_1', 32, 20),
    6: ('Q5_0', 32, 22), 7: ('Q5_1', 32, 24), 8: ('Q8_0', 32, 34), 9: ('Q8_1', 32, 36),
    10: ('Q2_K', 256, 84), 11: ('Q3_K', 256, 110), 12: ('Q4_K', 256, 144),
    13: ('Q5_K', 256, 176), 14: ('Q6_K', 256, 210), 15: ('Q8_K', 256, 292),
    16: ('IQ2_XXS', 256, 66), 17: ('IQ2_XS', 256, 74), 18: ('IQ3_XXS', 256, 98),
    19: ('IQ1_S', 256, 50), 20: ('IQ4_NL', 32, 18), 21: ('IQ3_S', 256, 110),
    22: ('IQ2_S', 256, 82), 23: ('IQ4_XS', 256, 136), 24: ('I8', 1, 1), 25: ('I16', 1, 2),
    26: ('I32', 1, 4), 27: ('I64', 1, 8), 28: ('F64', 1, 8), 29: ('IQ1_M', 256, 56),
    30: ('BF16', 1, 2), 34: ('TQ1_0', 256, 54), 35: ('TQ2_0', 256, 66),
    39: ('MXFP4', 32, 17), 40: ('NVFP4', 64, 36), 41: ('Q1_0', 128, 18),
}

# value type ids for KV parsing
_SCALAR = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i', 6: 'f', 7: '?', 10: 'Q',
           11: 'q', 12: 'd'}


class GGUFError(Exception):
    pass


class Tensor:
    __slots__ = ('name', 'shape', 'type_id', 'type_name', 'nbytes', 'nelem')

    def __init__(self, name, shape, type_id):
        self.name = name
        self.shape = tuple(shape)
        self.type_id = type_id
        info = GGML_TYPES.get(type_id)
        if info is None:
            raise GGUFError('unknown ggml type %d for %s' % (type_id, name))
        self.type_name, blk, bb = info
        n = 1
        for d in shape:
            n *= d
        if n % blk:
            raise GGUFError('%s: %d elements not a multiple of block %d' % (name, n, blk))
        self.nelem = n
        self.nbytes = n // blk * bb


class _Reader:
    def __init__(self, stream, size):
        self.stream, self.size = stream, size

    def read(self, size):
        if size < 0 or size > self.size - self.stream.tell():
            raise GGUFError('truncated header')
        result = self.stream.read(size)
        if len(result) != size:
            raise GGUFError('truncated header')
        return result

    def take(self, fmt):
        values = struct.unpack('<' + fmt, self.read(struct.calcsize(fmt)))
        return values[0] if len(values) == 1 else values

    def string(self):
        return self.read(self.take('Q')).decode('utf-8', 'replace')

    def value(self, vtype):
        if vtype in _SCALAR:
            return self.take(_SCALAR[vtype])
        if vtype == 8:
            return self.string()
        if vtype == 9:
            et, n = self.take('I'), self.take('Q')
            if et in _SCALAR:
                count = struct.calcsize(_SCALAR[et]) * n
                if count > self.size - self.stream.tell():
                    raise GGUFError('truncated array')
                if n <= 64:
                    return list(struct.unpack('<%d%s' % (n, _SCALAR[et]), self.read(count)))
                self.stream.seek(count, 1)
                return None
            for _ in range(n):
                self.value(et)
            return None
        raise GGUFError('bad kv value type %d' % vtype)


def read_gguf(path):
    # Unbuffered reads avoid Python read-ahead into the first tensor payload.
    with open(path, 'rb', buffering=0) as stream:
        size = os.fstat(stream.fileno()).st_size
        r = _Reader(stream, size)
        if r.read(4) != GGUF_MAGIC or r.take('I') not in (2, 3):
            raise GGUFError('unsupported GGUF')
        n_tensors, n_kv = r.take('Q'), r.take('Q')
        kv = {}
        for _ in range(n_kv):
            key = r.string()
            kv[key] = r.value(r.take('I'))
        tensors = []
        for _ in range(n_tensors):
            name = r.string()
            nd = r.take('I')
            if not 1 <= nd <= 4:
                raise GGUFError('invalid dimensions')
            shape = [r.take('Q') for _ in range(nd)]
            tid = r.take('I')
            r.take('Q')
            tensors.append(Tensor(name, shape, tid))
        return kv, tensors, size
