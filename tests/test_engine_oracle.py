import importlib.util
from pathlib import Path
import struct
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location('engine_oracle', Path(__file__).resolve().parents[1]/'tools/engine_oracle.py')
oracle = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(oracle)


def record(name, shape, payload=b''):
    name = name.encode()
    return (struct.pack('<I', len(name))+name+struct.pack('<II', 0, len(shape))+
            struct.pack('<'+'q'*len(shape), *shape)+payload)


class OracleRecordTests(unittest.TestCase):
    def parse(self, data):
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp)/'pos0.tensors'; file.write_bytes(data)
            return oracle.tensor_index(file)

    def test_contiguous_record_offsets_and_zero_extent_views(self):
        rows = self.parse(record('l_out-0', [2, 1, 1, 1], struct.pack('<ff', 1., 2.))+
                          record('empty_view', [2048, 0, 1, 1]))
        self.assertEqual([r['bytes'] for r in rows], [8, 0])
        self.assertEqual(rows[0]['shape'], [2, 1, 1, 1])

    def test_truncated_payload_cannot_be_an_oracle(self):
        with self.assertRaisesRegex(ValueError, 'truncated_tensor_payload'):
            self.parse(record('l_out-0', [2], struct.pack('<f', 1.)))

    def test_negative_shape_is_corrupt(self):
        with self.assertRaisesRegex(ValueError, 'invalid_tensor_shape'):
            self.parse(record('bad', [-1]))

    def test_truncated_header_is_corrupt(self):
        with self.assertRaisesRegex(ValueError, 'truncated_tensor_header'):
            self.parse(struct.pack('<I', 1)+b'x')


if __name__ == '__main__': unittest.main()
