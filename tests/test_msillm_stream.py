import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('msillm_stream', Path(__file__).resolve().parents[1] / 'experiments/new_codecs/msillm_stream.py')
codec = importlib.util.module_from_spec(spec)
spec.loader.exec_module(codec)


class FramingTest(unittest.TestCase):
    def test_roundtrip(self):
        args = (4, (1360, 2048), (1408, 2048), b'latent', b'hyper')
        stream = codec.pack(*args)
        self.assertEqual(codec.unpack(stream), args)
        self.assertEqual(len(stream), 28 + 6 + 5)

    def test_reject_malformed(self):
        stream = codec.pack(4, (64, 64), (64, 64), b'l', b'h')
        for data in [b'', stream[:20], stream[:-1], stream+b'x', b'XX'+stream[2:], stream[:2]+b'\x02'+stream[3:]]:
            with self.subTest(data=data), self.assertRaises(ValueError):
                codec.unpack(data)
        with self.assertRaises(ValueError):
            codec.pack(4, (65, 64), (64, 64), b'l', b'h')


if __name__ == '__main__':
    unittest.main()
