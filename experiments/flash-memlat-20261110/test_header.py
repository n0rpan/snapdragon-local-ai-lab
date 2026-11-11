"""Synthetic read-boundary check: the audit must not read tensor payload."""
import io
import struct
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import gguf_header as header

class HeaderReadTest(unittest.TestCase):
    def test_no_payload_read_ahead(self):
        name=b'output.weight'
        raw=(b'GGUF'+struct.pack('<IQQ',3,1,0)+struct.pack('<Q',len(name))+name+
             struct.pack('<IQQIQ',2,32,4,2,0))
        class Stream(io.BytesIO):
            def read(self,size=-1):
                if size < 0 or self.tell()+size>len(raw):
                    raise AssertionError('tensor payload read')
                return super().read(size)
            def fileno(self):return 123
        with patch('builtins.open',return_value=Stream(raw)) as opened,patch.object(header.os,'fstat',return_value=SimpleNamespace(st_size=len(raw)+4096)):
            kv,tensors,size=header.read_gguf('synthetic.gguf')
        opened.assert_called_once_with('synthetic.gguf','rb',buffering=0)
        self.assertEqual(tensors[0].nbytes,72)
        self.assertEqual(size,len(raw)+4096)

if __name__=='__main__':unittest.main()
